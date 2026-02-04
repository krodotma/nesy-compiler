/**
 * Training Loop - The Actual Generative Training Component
 *
 * This is the missing piece from @nesy/learning.
 *
 * The Training Loop combines:
 * 1. Experience replay (from ExperienceBuffer)
 * 2. RLCF (Reinforcement Learning from Compiler Feedback)
 * 3. Denoising (epistemic uncertainty reduction)
 * 4. Generation (producing new code)
 *
 * The Denoise Loop:
 * observation → decompose(aleatoric, epistemic) →
 *     denoise_epistemic(learn more) →
 *     accept_aleatoric(maintain diversity) →
 *     evolve
 */

import type { Experience, ExperienceBuffer } from './experience.js';
import type { FeedbackCollector, Feedback } from './feedback.js';
import type { RLCFLoop, RLCFState, CompilerDiagnostic } from './rlcf.js';
import type { Generator, GenerationResult, GenerationRequest } from './generator.js';

export interface TrainingConfig {
  /** Batch size for experience replay */
  batchSize: number;
  /** Learning rate */
  learningRate: number;
  /** Discount factor */
  gamma: number;
  /** Number of training epochs */
  epochs: number;
  /** How often to evaluate (epochs) */
  evalInterval: number;
  /** Minimum positive samples before training */
  minPositiveSamples: number;
  /** Aleatoric uncertainty threshold (accept as diversity) */
  aleatoricThreshold: number;
  /** Epistemic uncertainty threshold (denoise via learning) */
  epistemicThreshold: number;
}

const DEFAULT_CONFIG: TrainingConfig = {
  batchSize: 32,
  learningRate: 0.001,
  gamma: 0.99,
  epochs: 100,
  evalInterval: 10,
  minPositiveSamples: 50,
  aleatoricThreshold: 0.3,
  epistemicThreshold: 0.7,
};

export interface TrainingState {
  epoch: number;
  totalSamples: number;
  positiveSamples: number;
  negativeSamples: number;
  avgReward: number;
  successRate: number;
  losses: number[];
  isTraining: boolean;
}

export interface UncertaintyDecomposition {
  /** Total uncertainty */
  total: number;
  /** Aleatoric (irreducible, maintain diversity) */
  aleatoric: number;
  /** Epistemic (reducible, learn more) */
  epistemic: number;
  /** Classification */
  classification: 'certain' | 'aleatoric' | 'epistemic' | 'mixed';
}

export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  successRate: number;
  avgReward: number;
  uncertainty: UncertaintyDecomposition;
  timestamp: number;
}

export interface EditTriplet {
  /** Code before the edit */
  preState: string;
  /** The diff/patch */
  diff: string;
  /** Code after the edit */
  postState: string;
  /** Whether the edit was good (compiled, passed tests) */
  isGood: boolean;
  /** Source (commit hash, file path) */
  source: string;
}

/**
 * Training Loop: The actual generative training component.
 */
export class TrainingLoop {
  private config: TrainingConfig;
  private state: TrainingState;
  private experienceBuffer: ExperienceBuffer;
  private feedbackCollector: FeedbackCollector;
  private rlcf: RLCFLoop | null;
  private generator: Generator | null;
  private editTriplets: EditTriplet[];
  private metricsHistory: TrainingMetrics[];

  constructor(
    experienceBuffer: ExperienceBuffer,
    feedbackCollector: FeedbackCollector,
    config?: Partial<TrainingConfig>
  ) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.experienceBuffer = experienceBuffer;
    this.feedbackCollector = feedbackCollector;
    this.rlcf = null;
    this.generator = null;
    this.editTriplets = [];
    this.metricsHistory = [];

    this.state = {
      epoch: 0,
      totalSamples: 0,
      positiveSamples: 0,
      negativeSamples: 0,
      avgReward: 0,
      successRate: 0,
      losses: [],
      isTraining: false,
    };
  }

  /**
   * Set RLCF component.
   */
  setRLCF(rlcf: RLCFLoop): void {
    this.rlcf = rlcf;
  }

  /**
   * Set Generator component.
   */
  setGenerator(generator: Generator): void {
    this.generator = generator;
  }

  /**
   * Add edit triplet for training.
   */
  addEditTriplet(triplet: EditTriplet): void {
    this.editTriplets.push(triplet);
  }

  /**
   * Run the training loop.
   */
  async train(
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>,
    onProgress?: (metrics: TrainingMetrics) => void
  ): Promise<TrainingMetrics[]> {
    if (!this.canTrain()) {
      throw new Error(
        `Not enough data. Need ${this.config.minPositiveSamples} positive samples, ` +
        `have ${this.feedbackCollector.getPositive().length}`
      );
    }

    this.state.isTraining = true;
    const metrics: TrainingMetrics[] = [];

    try {
      for (let epoch = 0; epoch < this.config.epochs; epoch++) {
        this.state.epoch = epoch;

        // Sample batch
        const batch = this.sampleBatch();

        // Train on batch
        const epochMetrics = await this.trainEpoch(batch, compile);

        metrics.push(epochMetrics);
        this.metricsHistory.push(epochMetrics);

        // Progress callback
        if (onProgress) {
          onProgress(epochMetrics);
        }

        // Evaluate periodically
        if (epoch % this.config.evalInterval === 0) {
          await this.evaluate(compile);
        }
      }
    } finally {
      this.state.isTraining = false;
    }

    return metrics;
  }

  /**
   * Check if we have enough data to train.
   */
  canTrain(): boolean {
    return this.feedbackCollector.getPositive().length >= this.config.minPositiveSamples;
  }

  /**
   * Sample a batch from experience buffer.
   */
  private sampleBatch(): Experience[] {
    return this.experienceBuffer.sample(this.config.batchSize);
  }

  /**
   * Train on one epoch.
   */
  private async trainEpoch(
    batch: Experience[],
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<TrainingMetrics> {
    let totalLoss = 0;
    let totalReward = 0;
    let successes = 0;

    for (const experience of batch) {
      // Decompose uncertainty
      const uncertainty = this.decomposeUncertainty(experience);

      // Handle based on uncertainty type
      if (uncertainty.classification === 'epistemic') {
        // Denoise via learning
        const loss = await this.denoiseEpistemic(experience, compile);
        totalLoss += loss;
      } else if (uncertainty.classification === 'aleatoric') {
        // Accept as diversity (no training, just record)
        this.acceptAleatoric(experience);
      } else {
        // Mixed or certain - normal training
        const { loss, reward, success } = await this.trainOnExperience(experience, compile);
        totalLoss += loss;
        totalReward += reward;
        if (success) successes++;
      }

      this.state.totalSamples++;
    }

    const avgLoss = totalLoss / batch.length;
    const avgReward = totalReward / batch.length;
    const successRate = successes / batch.length;

    this.state.losses.push(avgLoss);
    this.state.avgReward = avgReward;
    this.state.successRate = successRate;

    return {
      epoch: this.state.epoch,
      loss: avgLoss,
      accuracy: successRate,
      successRate,
      avgReward,
      uncertainty: this.aggregateUncertainty(batch),
      timestamp: Date.now(),
    };
  }

  /**
   * Decompose uncertainty into aleatoric and epistemic.
   */
  private decomposeUncertainty(experience: Experience): UncertaintyDecomposition {
    const feedback = experience.feedback;
    if (!feedback) {
      // No feedback = high epistemic uncertainty
      return {
        total: 1.0,
        aleatoric: 0.1,
        epistemic: 0.9,
        classification: 'epistemic',
      };
    }

    // Quality as uncertainty measure
    const totalUncertainty = 1 - feedback.quality;

    // Heuristics for decomposition:
    // - Inconsistent results across runs = aleatoric
    // - Consistent failures = epistemic
    // - Near threshold = mixed

    // For now, use a simple heuristic based on quality
    const aleatoricRatio = feedback.correctness ? 0.3 : 0.1;
    const aleatoric = totalUncertainty * aleatoricRatio;
    const epistemic = totalUncertainty * (1 - aleatoricRatio);

    let classification: UncertaintyDecomposition['classification'];
    if (totalUncertainty < 0.2) {
      classification = 'certain';
    } else if (aleatoric > this.config.aleatoricThreshold) {
      classification = 'aleatoric';
    } else if (epistemic > this.config.epistemicThreshold) {
      classification = 'epistemic';
    } else {
      classification = 'mixed';
    }

    return { total: totalUncertainty, aleatoric, epistemic, classification };
  }

  /**
   * Denoise epistemic uncertainty (learn more).
   */
  private async denoiseEpistemic(
    experience: Experience,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<number> {
    // Use RLCF to refine and learn
    // Extract code from execution plan if available
    const emitOutput = experience.result.stages.emit;
    const executionPlan = emitOutput?.artifact?.plan;

    // Try to extract code from the execution plan or artifact
    const codeToRefine = this.extractCodeFromExperience(experience);

    if (this.rlcf && codeToRefine) {
      const result = await this.rlcf.refine(
        codeToRefine,
        compile
      );

      // Loss based on iterations needed
      return result.iteration / 3; // Normalize by max iterations
    }

    // No RLCF or no code available, return high loss
    return 1.0;
  }

  /**
   * Extract code from experience for refinement.
   * Code can be in different places depending on the compilation pipeline.
   */
  private extractCodeFromExperience(experience: Experience): string | null {
    // Check if there's a custom code property (extended result)
    const result = experience.result as unknown as Record<string, unknown>;
    if (typeof result.code === 'string') {
      return result.code;
    }

    // Try to get from artifact's execution plan
    const plan = experience.result.stages.emit?.artifact?.plan;
    if (plan && plan.length > 0) {
      // Look for code in execution steps
      for (const step of plan) {
        if ('code' in step && typeof (step as Record<string, unknown>).code === 'string') {
          return (step as Record<string, unknown>).code as string;
        }
      }
    }

    // No code found
    return null;
  }

  /**
   * Accept aleatoric uncertainty (maintain diversity).
   */
  private acceptAleatoric(experience: Experience): void {
    // Record as "diverse" sample - don't try to fix it
    // This preserves exploration and prevents overfitting

    // Update state
    if (experience.feedback?.correctness) {
      this.state.positiveSamples++;
    } else {
      this.state.negativeSamples++;
    }
  }

  /**
   * Train on a single experience.
   */
  private async trainOnExperience(
    experience: Experience,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<{ loss: number; reward: number; success: boolean }> {
    const feedback = experience.feedback;
    const quality = feedback?.quality || 0;
    const correct = feedback?.correctness || false;

    // Reward calculation
    const reward = correct ? quality : quality * 0.1;

    // Loss = 1 - quality (simplified)
    const loss = 1 - quality;

    // Update counters
    if (correct) {
      this.state.positiveSamples++;
    } else {
      this.state.negativeSamples++;
    }

    return { loss, reward, success: correct };
  }

  /**
   * Aggregate uncertainty over batch.
   */
  private aggregateUncertainty(batch: Experience[]): UncertaintyDecomposition {
    if (batch.length === 0) {
      return { total: 0, aleatoric: 0, epistemic: 0, classification: 'certain' };
    }

    let totalAleatoric = 0;
    let totalEpistemic = 0;
    let totalUncertainty = 0;

    for (const exp of batch) {
      const decomp = this.decomposeUncertainty(exp);
      totalAleatoric += decomp.aleatoric;
      totalEpistemic += decomp.epistemic;
      totalUncertainty += decomp.total;
    }

    const avgAleatoric = totalAleatoric / batch.length;
    const avgEpistemic = totalEpistemic / batch.length;
    const avgTotal = totalUncertainty / batch.length;

    let classification: UncertaintyDecomposition['classification'];
    if (avgTotal < 0.2) {
      classification = 'certain';
    } else if (avgAleatoric > avgEpistemic) {
      classification = 'aleatoric';
    } else if (avgEpistemic > avgAleatoric) {
      classification = 'epistemic';
    } else {
      classification = 'mixed';
    }

    return {
      total: avgTotal,
      aleatoric: avgAleatoric,
      epistemic: avgEpistemic,
      classification,
    };
  }

  /**
   * Evaluate current model.
   */
  private async evaluate(
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<void> {
    if (!this.generator) return;

    // Generate a few test samples
    const testRequests: GenerationRequest[] = [
      {
        type: 'function',
        description: 'Add two numbers',
        language: 'typescript',
        signature: '(a: number, b: number) => number',
      },
      {
        type: 'function',
        description: 'Check if a string is a palindrome',
        language: 'typescript',
        signature: '(s: string) => boolean',
      },
    ];

    let successes = 0;
    for (const request of testRequests) {
      const result = await this.generator.generate(request, compile);
      if (result.success) {
        successes++;
      }
    }

    const evalSuccessRate = successes / testRequests.length;
    console.log(`Evaluation success rate: ${(evalSuccessRate * 100).toFixed(1)}%`);
  }

  /**
   * Get current training state.
   */
  getState(): TrainingState {
    return { ...this.state };
  }

  /**
   * Get metrics history.
   */
  getMetricsHistory(): TrainingMetrics[] {
    return [...this.metricsHistory];
  }

  /**
   * Export trained model (policy weights, etc.)
   */
  exportModel(): {
    rlcfPolicy?: ReturnType<RLCFLoop['exportPolicy']>;
    metricsHistory: TrainingMetrics[];
    state: TrainingState;
  } {
    return {
      rlcfPolicy: this.rlcf?.exportPolicy(),
      metricsHistory: this.metricsHistory,
      state: this.state,
    };
  }

  /**
   * Import trained model.
   */
  importModel(data: {
    rlcfPolicy?: ReturnType<RLCFLoop['exportPolicy']>;
    metricsHistory?: TrainingMetrics[];
    state?: Partial<TrainingState>;
  }): void {
    if (data.rlcfPolicy && this.rlcf) {
      this.rlcf.importPolicy(data.rlcfPolicy);
    }
    if (data.metricsHistory) {
      this.metricsHistory = data.metricsHistory;
    }
    if (data.state) {
      this.state = { ...this.state, ...data.state };
    }
  }
}

/**
 * Create a training loop with all components.
 */
export function createTrainingLoop(
  experienceBuffer: ExperienceBuffer,
  feedbackCollector: FeedbackCollector,
  config?: Partial<TrainingConfig>
): TrainingLoop {
  return new TrainingLoop(experienceBuffer, feedbackCollector, config);
}
