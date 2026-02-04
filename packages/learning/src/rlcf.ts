/**
 * RLCF - Reinforcement Learning from Compiler Feedback
 *
 * Step 85 of NeSy Evolution Phase 8 (Neuro-Symbolic Generation)
 *
 * Core Insight: Compilers are perfect reward functions.
 * - They don't hallucinate (deterministic)
 * - They catch all syntax errors (complete for syntax)
 * - They catch many semantic errors (types, imports)
 *
 * The RLCF Loop:
 * 1. Generate candidate code (draft)
 * 2. Run compiler (verify)
 * 3. Extract feedback (error messages)
 * 4. Update policy (what to generate next)
 * 5. Repeat until pass or max iterations
 */

import type { CompilationResult } from '@nesy/pipeline';
import type { Experience } from './experience.js';
import type { Feedback } from './feedback.js';

export interface RLCFConfig {
  /** Maximum refinement iterations */
  maxIterations: number;
  /** Reward for compilation success */
  successReward: number;
  /** Penalty per error */
  errorPenalty: number;
  /** Penalty per warning */
  warningPenalty: number;
  /** Learning rate for policy updates */
  learningRate: number;
  /** Discount factor for future rewards */
  gamma: number;
  /** Exploration rate (epsilon-greedy) */
  epsilon: number;
}

const DEFAULT_CONFIG: RLCFConfig = {
  maxIterations: 3,
  successReward: 1.0,
  errorPenalty: -0.3,
  warningPenalty: -0.1,
  learningRate: 0.1,
  gamma: 0.99,
  epsilon: 0.1,
};

export interface CompilerDiagnostic {
  type: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  location?: {
    line: number;
    column: number;
    file?: string;
  };
  suggestion?: string;
}

export interface RLCFState {
  iteration: number;
  code: string;
  diagnostics: CompilerDiagnostic[];
  reward: number;
  done: boolean;
  success: boolean;
}

export interface RLCFPolicy {
  /** Action probabilities for different fix strategies */
  actionProbs: Map<string, number>;
  /** Value estimate for current state */
  stateValue: number;
  /** History of state-action-reward tuples */
  trajectory: RLCFTransition[];
}

export interface RLCFTransition {
  state: RLCFState;
  action: string;
  reward: number;
  nextState: RLCFState;
}

export interface FixStrategy {
  name: string;
  /** Diagnostic codes this strategy can fix */
  handles: string[];
  /** Apply fix and return modified code */
  apply: (code: string, diagnostic: CompilerDiagnostic) => string;
}

/**
 * RLCF Loop: Learn to generate correct code from compiler feedback.
 */
export class RLCFLoop {
  private config: RLCFConfig;
  private policy: RLCFPolicy;
  private fixStrategies: Map<string, FixStrategy>;
  private experienceHistory: RLCFTransition[];

  constructor(config?: Partial<RLCFConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.policy = {
      actionProbs: new Map(),
      stateValue: 0,
      trajectory: [],
    };
    this.fixStrategies = new Map();
    this.experienceHistory = [];

    // Register default fix strategies
    this.registerDefaultStrategies();
  }

  /**
   * Register a fix strategy.
   */
  registerStrategy(strategy: FixStrategy): void {
    this.fixStrategies.set(strategy.name, strategy);
    // Initialize action probability
    if (!this.policy.actionProbs.has(strategy.name)) {
      this.policy.actionProbs.set(strategy.name, 1.0 / (this.fixStrategies.size + 1));
    }
  }

  /**
   * Run the RLCF refinement loop.
   */
  async refine(
    initialCode: string,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RLCFState> {
    let state = await this.createInitialState(initialCode, compile);
    this.policy.trajectory = [];

    for (let i = 0; i < this.config.maxIterations && !state.done; i++) {
      // Select action (fix strategy) using epsilon-greedy
      const action = this.selectAction(state);

      // Apply action to get new code
      const newCode = this.applyAction(state.code, state.diagnostics, action);

      // Get next state
      const nextState = await this.createState(newCode, i + 1, compile);

      // Calculate reward
      const reward = this.calculateReward(state, nextState);

      // Record transition
      const transition: RLCFTransition = {
        state,
        action,
        reward,
        nextState,
      };
      this.policy.trajectory.push(transition);
      this.experienceHistory.push(transition);

      // Update policy (online learning)
      this.updatePolicy(transition);

      state = nextState;
    }

    return state;
  }

  /**
   * Create initial state from code.
   */
  private async createInitialState(
    code: string,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RLCFState> {
    return this.createState(code, 0, compile);
  }

  /**
   * Create state from code and compilation result.
   */
  private async createState(
    code: string,
    iteration: number,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RLCFState> {
    const result = await compile(code);
    const errors = result.diagnostics.filter(d => d.type === 'error');

    return {
      iteration,
      code,
      diagnostics: result.diagnostics,
      reward: 0, // Will be calculated
      done: result.passed || iteration >= this.config.maxIterations,
      success: result.passed,
    };
  }

  /**
   * Select action using epsilon-greedy policy.
   */
  private selectAction(state: RLCFState): string {
    // Exploration: random action
    if (Math.random() < this.config.epsilon) {
      const strategies = Array.from(this.fixStrategies.keys());
      return strategies[Math.floor(Math.random() * strategies.length)] || 'no-op';
    }

    // Exploitation: best action for current diagnostics
    const applicableStrategies = this.getApplicableStrategies(state.diagnostics);

    if (applicableStrategies.length === 0) {
      return 'no-op';
    }

    // Select highest probability strategy
    let bestStrategy = applicableStrategies[0];
    let bestProb = this.policy.actionProbs.get(bestStrategy) || 0;

    for (const strategy of applicableStrategies) {
      const prob = this.policy.actionProbs.get(strategy) || 0;
      if (prob > bestProb) {
        bestProb = prob;
        bestStrategy = strategy;
      }
    }

    return bestStrategy;
  }

  /**
   * Get strategies applicable to current diagnostics.
   */
  private getApplicableStrategies(diagnostics: CompilerDiagnostic[]): string[] {
    const diagnosticCodes = new Set(diagnostics.map(d => d.code));
    const applicable: string[] = [];

    for (const [name, strategy] of this.fixStrategies) {
      if (strategy.handles.some(code => diagnosticCodes.has(code))) {
        applicable.push(name);
      }
    }

    return applicable;
  }

  /**
   * Apply action to code.
   */
  private applyAction(
    code: string,
    diagnostics: CompilerDiagnostic[],
    action: string
  ): string {
    if (action === 'no-op') {
      return code;
    }

    const strategy = this.fixStrategies.get(action);
    if (!strategy) {
      return code;
    }

    // Find first applicable diagnostic
    const applicable = diagnostics.find(d =>
      strategy.handles.includes(d.code)
    );

    if (!applicable) {
      return code;
    }

    return strategy.apply(code, applicable);
  }

  /**
   * Calculate reward for transition.
   */
  private calculateReward(state: RLCFState, nextState: RLCFState): number {
    if (nextState.success) {
      return this.config.successReward;
    }

    const prevErrors = state.diagnostics.filter(d => d.type === 'error').length;
    const nextErrors = nextState.diagnostics.filter(d => d.type === 'error').length;

    const prevWarnings = state.diagnostics.filter(d => d.type === 'warning').length;
    const nextWarnings = nextState.diagnostics.filter(d => d.type === 'warning').length;

    // Reward for reducing errors
    const errorDelta = prevErrors - nextErrors;
    const warningDelta = prevWarnings - nextWarnings;

    return (
      errorDelta * Math.abs(this.config.errorPenalty) +
      warningDelta * Math.abs(this.config.warningPenalty)
    );
  }

  /**
   * Update policy based on transition.
   */
  private updatePolicy(transition: RLCFTransition): void {
    const { action, reward } = transition;

    // Update action probability (simple bandit-style update)
    const currentProb = this.policy.actionProbs.get(action) || 0;
    const newProb = currentProb + this.config.learningRate * (reward - currentProb);
    this.policy.actionProbs.set(action, Math.max(0.01, Math.min(0.99, newProb)));

    // Normalize probabilities
    this.normalizeActionProbs();
  }

  /**
   * Normalize action probabilities to sum to 1.
   */
  private normalizeActionProbs(): void {
    let sum = 0;
    for (const prob of this.policy.actionProbs.values()) {
      sum += prob;
    }
    if (sum > 0) {
      for (const [action, prob] of this.policy.actionProbs) {
        this.policy.actionProbs.set(action, prob / sum);
      }
    }
  }

  /**
   * Register default fix strategies for common errors.
   */
  private registerDefaultStrategies(): void {
    // TypeScript: Missing import
    this.registerStrategy({
      name: 'add-import',
      handles: ['TS2304', 'TS2305', 'TS2307'],
      apply: (code, diagnostic) => {
        // Extract what needs to be imported from error message
        const match = diagnostic.message.match(/Cannot find name '(\w+)'/);
        if (match) {
          const name = match[1];
          // Simple heuristic: add import at top
          return `import { ${name} } from './${name.toLowerCase()}';\n${code}`;
        }
        return code;
      },
    });

    // TypeScript: Missing semicolon
    this.registerStrategy({
      name: 'add-semicolon',
      handles: ['TS1005'],
      apply: (code, diagnostic) => {
        if (diagnostic.location) {
          const lines = code.split('\n');
          const line = lines[diagnostic.location.line - 1];
          if (line && !line.trimEnd().endsWith(';')) {
            lines[diagnostic.location.line - 1] = line.trimEnd() + ';';
            return lines.join('\n');
          }
        }
        return code;
      },
    });

    // TypeScript: Missing type annotation
    this.registerStrategy({
      name: 'add-any-type',
      handles: ['TS7006', 'TS7031'],
      apply: (code, diagnostic) => {
        if (diagnostic.location) {
          const lines = code.split('\n');
          const line = lines[diagnostic.location.line - 1];
          if (line) {
            // Add : any after parameter name
            const col = diagnostic.location.column - 1;
            const before = line.slice(0, col);
            const after = line.slice(col);
            // Find end of identifier
            const match = after.match(/^(\w+)/);
            if (match) {
              const identifier = match[1];
              lines[diagnostic.location.line - 1] =
                before + identifier + ': any' + after.slice(identifier.length);
              return lines.join('\n');
            }
          }
        }
        return code;
      },
    });

    // TypeScript: Unused variable (remove)
    this.registerStrategy({
      name: 'remove-unused',
      handles: ['TS6133', 'TS6196'],
      apply: (code, diagnostic) => {
        if (diagnostic.location) {
          const lines = code.split('\n');
          // Comment out the line (safer than removing)
          lines[diagnostic.location.line - 1] =
            '// REMOVED: ' + lines[diagnostic.location.line - 1];
          return lines.join('\n');
        }
        return code;
      },
    });

    // ESLint: Missing return type
    this.registerStrategy({
      name: 'add-void-return',
      handles: ['@typescript-eslint/explicit-function-return-type'],
      apply: (code, diagnostic) => {
        if (diagnostic.location) {
          const lines = code.split('\n');
          const line = lines[diagnostic.location.line - 1];
          if (line) {
            // Add : void before {
            const modified = line.replace(/\)\s*{/, '): void {');
            lines[diagnostic.location.line - 1] = modified;
            return lines.join('\n');
          }
        }
        return code;
      },
    });
  }

  /**
   * Get learning statistics.
   */
  getStats(): {
    totalIterations: number;
    successRate: number;
    avgIterationsToSuccess: number;
    topStrategies: Array<{ name: string; prob: number }>;
  } {
    const trajectories = this.groupTrajectories();
    const successful = trajectories.filter(t =>
      t[t.length - 1]?.nextState.success
    );

    const successRate = trajectories.length > 0
      ? successful.length / trajectories.length
      : 0;

    const avgIterations = successful.length > 0
      ? successful.reduce((sum, t) => sum + t.length, 0) / successful.length
      : 0;

    const topStrategies = Array.from(this.policy.actionProbs.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([name, prob]) => ({ name, prob }));

    return {
      totalIterations: this.experienceHistory.length,
      successRate,
      avgIterationsToSuccess: avgIterations,
      topStrategies,
    };
  }

  /**
   * Group experience history into trajectories.
   */
  private groupTrajectories(): RLCFTransition[][] {
    const trajectories: RLCFTransition[][] = [];
    let current: RLCFTransition[] = [];

    for (const transition of this.experienceHistory) {
      current.push(transition);
      if (transition.nextState.done) {
        trajectories.push(current);
        current = [];
      }
    }

    if (current.length > 0) {
      trajectories.push(current);
    }

    return trajectories;
  }

  /**
   * Export policy for persistence.
   */
  exportPolicy(): {
    actionProbs: Record<string, number>;
    stateValue: number;
  } {
    return {
      actionProbs: Object.fromEntries(this.policy.actionProbs),
      stateValue: this.policy.stateValue,
    };
  }

  /**
   * Import policy from persistence.
   */
  importPolicy(data: {
    actionProbs: Record<string, number>;
    stateValue: number;
  }): void {
    this.policy.actionProbs = new Map(Object.entries(data.actionProbs));
    this.policy.stateValue = data.stateValue;
  }
}

/**
 * Quick function to run RLCF refinement.
 */
export async function refineWithRLCF(
  code: string,
  compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>,
  config?: Partial<RLCFConfig>
): Promise<RLCFState> {
  const rlcf = new RLCFLoop(config);
  return rlcf.refine(code, compile);
}
