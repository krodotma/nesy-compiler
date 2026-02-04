/**
 * Training Loop - The Actual Generative Training Component
 *
 * The Denoise Loop:
 * observation → decompose(aleatoric, epistemic) →
 *     denoise_epistemic(learn more) →
 *     accept_aleatoric(maintain diversity) →
 *     evolve
 */

import type { Experience, ExperienceBuffer } from './experience.js';
import type { FeedbackCollector, Feedback } from './feedback.js';
import type { RLCFLoop, CompilerDiagnostic } from './rlcf.js';
import type { Generator, GenerationRequest } from './generator.js';

export interface TrainingConfig {
  batchSize: number;
  learningRate: number;
  gamma: number;
  epochs: number;
  evalInterval: number;
  minPositiveSamples: number;
  aleatoricThreshold: number;
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
  total: number;
  aleatoric: number;
  epistemic: number;
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
  preState: string;
  diff: string;
  postState: string;
  isGood: boolean;
  source: string;
}

export class TrainingLoop {
  private config: TrainingConfig;
  private state: TrainingState;
  private experienceBuffer: ExperienceBuffer;
  private feedbackCollector: FeedbackCollector;
  private rlcf: RLCFLoop | null = null;
  private generator: Generator | null = null;
  private editTriplets: EditTriplet[] = [];
  private metricsHistory: TrainingMetrics[] = [];

  constructor(experienceBuffer: ExperienceBuffer, feedbackCollector: FeedbackCollector, config?: Partial<TrainingConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.experienceBuffer = experienceBuffer;
    this.feedbackCollector = feedbackCollector;
    this.state = { epoch: 0, totalSamples: 0, positiveSamples: 0, negativeSamples: 0, avgReward: 0, successRate: 0, losses: [], isTraining: false };
  }

  setRLCF(rlcf: RLCFLoop): void { this.rlcf = rlcf; }
  setGenerator(generator: Generator): void { this.generator = generator; }
  addEditTriplet(triplet: EditTriplet): void { this.editTriplets.push(triplet); }

  async train(
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>,
    onProgress?: (metrics: TrainingMetrics) => void
  ): Promise<TrainingMetrics[]> {
    if (!this.canTrain()) {
      throw new Error(`Not enough data. Need ${this.config.minPositiveSamples} positive samples, have ${this.feedbackCollector.getPositive().length}`);
    }

    this.state.isTraining = true;
    const metrics: TrainingMetrics[] = [];

    try {
      for (let epoch = 0; epoch < this.config.epochs; epoch++) {
        this.state.epoch = epoch;
        const batch = this.sampleBatch();
        const epochMetrics = await this.trainEpoch(batch, compile);
        metrics.push(epochMetrics);
        this.metricsHistory.push(epochMetrics);
        if (onProgress) onProgress(epochMetrics);
        if (epoch % this.config.evalInterval === 0) await this.evaluate(compile);
      }
    } finally {
      this.state.isTraining = false;
    }

    return metrics;
  }

  canTrain(): boolean {
    return this.feedbackCollector.getPositive().length >= this.config.minPositiveSamples;
  }

  private sampleBatch(): Experience[] {
    return this.experienceBuffer.sample(this.config.batchSize);
  }

  private async trainEpoch(
    batch: Experience[],
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<TrainingMetrics> {
    let totalLoss = 0, totalReward = 0, successes = 0;

    for (const experience of batch) {
      const uncertainty = this.decomposeUncertainty(experience);

      if (uncertainty.classification === 'epistemic') {
        totalLoss += await this.denoiseEpistemic(experience, compile);
      } else if (uncertainty.classification === 'aleatoric') {
        this.acceptAleatoric(experience);
      } else {
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

    return { epoch: this.state.epoch, loss: avgLoss, accuracy: successRate, successRate, avgReward, uncertainty: this.aggregateUncertainty(batch), timestamp: Date.now() };
  }

  private decomposeUncertainty(experience: Experience): UncertaintyDecomposition {
    const feedback = experience.feedback;
    if (!feedback) return { total: 1.0, aleatoric: 0.1, epistemic: 0.9, classification: 'epistemic' };

    const totalUncertainty = 1 - feedback.quality;
    const aleatoricRatio = feedback.correctness ? 0.3 : 0.1;
    const aleatoric = totalUncertainty * aleatoricRatio;
    const epistemic = totalUncertainty * (1 - aleatoricRatio);

    let classification: UncertaintyDecomposition['classification'];
    if (totalUncertainty < 0.2) classification = 'certain';
    else if (aleatoric > this.config.aleatoricThreshold) classification = 'aleatoric';
    else if (epistemic > this.config.epistemicThreshold) classification = 'epistemic';
    else classification = 'mixed';

    return { total: totalUncertainty, aleatoric, epistemic, classification };
  }

  private async denoiseEpistemic(
    experience: Experience,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<number> {
    const codeToRefine = this.extractCodeFromExperience(experience);
    if (this.rlcf && codeToRefine) {
      const result = await this.rlcf.refine(codeToRefine, compile);
      return result.iteration / 3;
    }
    return 1.0;
  }

  private extractCodeFromExperience(experience: Experience): string | null {
    const result = experience.result as unknown as Record<string, unknown>;
    if (typeof result.code === 'string') return result.code;

    const plan = experience.result.stages.emit?.artifact?.plan;
    if (plan && plan.length > 0) {
      for (const step of plan) {
        if ('code' in step && typeof (step as Record<string, unknown>).code === 'string') {
          return (step as Record<string, unknown>).code as string;
        }
      }
    }
    return null;
  }

  private acceptAleatoric(experience: Experience): void {
    if (experience.feedback?.correctness) this.state.positiveSamples++;
    else this.state.negativeSamples++;
  }

  private async trainOnExperience(
    experience: Experience,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<{ loss: number; reward: number; success: boolean }> {
    const feedback = experience.feedback;
    const quality = feedback?.quality || 0;
    const correct = feedback?.correctness || false;
    const reward = correct ? quality : quality * 0.1;
    const loss = 1 - quality;

    if (correct) this.state.positiveSamples++;
    else this.state.negativeSamples++;

    return { loss, reward, success: correct };
  }

  private aggregateUncertainty(batch: Experience[]): UncertaintyDecomposition {
    if (batch.length === 0) return { total: 0, aleatoric: 0, epistemic: 0, classification: 'certain' };

    let totalAleatoric = 0, totalEpistemic = 0, totalUncertainty = 0;
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
    if (avgTotal < 0.2) classification = 'certain';
    else if (avgAleatoric > avgEpistemic) classification = 'aleatoric';
    else if (avgEpistemic > avgAleatoric) classification = 'epistemic';
    else classification = 'mixed';

    return { total: avgTotal, aleatoric: avgAleatoric, epistemic: avgEpistemic, classification };
  }

  private async evaluate(
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<void> {
    if (!this.generator) return;

    const testRequests: GenerationRequest[] = [
      { type: 'function', description: 'Add two numbers', language: 'typescript', signature: '(a: number, b: number) => number' },
      { type: 'function', description: 'Check if palindrome', language: 'typescript', signature: '(s: string) => boolean' },
    ];

    let successes = 0;
    for (const request of testRequests) {
      const result = await this.generator.generate(request, compile);
      if (result.success) successes++;
    }
    console.log(`Eval success rate: ${(successes / testRequests.length * 100).toFixed(1)}%`);
  }

  getState(): TrainingState { return { ...this.state }; }
  getMetricsHistory(): TrainingMetrics[] { return [...this.metricsHistory]; }

  exportModel(): { rlcfPolicy?: ReturnType<RLCFLoop['exportPolicy']>; metricsHistory: TrainingMetrics[]; state: TrainingState } {
    return { rlcfPolicy: this.rlcf?.exportPolicy(), metricsHistory: this.metricsHistory, state: this.state };
  }

  importModel(data: { rlcfPolicy?: ReturnType<RLCFLoop['exportPolicy']>; metricsHistory?: TrainingMetrics[]; state?: Partial<TrainingState> }): void {
    if (data.rlcfPolicy && this.rlcf) this.rlcf.importPolicy(data.rlcfPolicy);
    if (data.metricsHistory) this.metricsHistory = data.metricsHistory;
    if (data.state) this.state = { ...this.state, ...data.state };
  }
}

export function createTrainingLoop(experienceBuffer: ExperienceBuffer, feedbackCollector: FeedbackCollector, config?: Partial<TrainingConfig>): TrainingLoop {
  return new TrainingLoop(experienceBuffer, feedbackCollector, config);
}
