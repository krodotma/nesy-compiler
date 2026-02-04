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
      const action = this.selectAction(state);
      const newCode = this.applyAction(state.code, state.diagnostics, action);
      const nextState = await this.createState(newCode, i + 1, compile);
      const reward = this.calculateReward(state, nextState);

      const transition: RLCFTransition = { state, action, reward, nextState };
      this.policy.trajectory.push(transition);
      this.experienceHistory.push(transition);
      this.updatePolicy(transition);

      state = nextState;
    }

    return state;
  }

  private async createInitialState(
    code: string,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RLCFState> {
    return this.createState(code, 0, compile);
  }

  private async createState(
    code: string,
    iteration: number,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RLCFState> {
    const result = await compile(code);
    return {
      iteration,
      code,
      diagnostics: result.diagnostics,
      reward: 0,
      done: result.passed || iteration >= this.config.maxIterations,
      success: result.passed,
    };
  }

  private selectAction(state: RLCFState): string {
    if (Math.random() < this.config.epsilon) {
      const strategies = Array.from(this.fixStrategies.keys());
      return strategies[Math.floor(Math.random() * strategies.length)] || 'no-op';
    }

    const applicableStrategies = this.getApplicableStrategies(state.diagnostics);
    if (applicableStrategies.length === 0) return 'no-op';

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

  private applyAction(code: string, diagnostics: CompilerDiagnostic[], action: string): string {
    if (action === 'no-op') return code;

    const strategy = this.fixStrategies.get(action);
    if (!strategy) return code;

    const applicable = diagnostics.find(d => strategy.handles.includes(d.code));
    if (!applicable) return code;

    return strategy.apply(code, applicable);
  }

  private calculateReward(state: RLCFState, nextState: RLCFState): number {
    if (nextState.success) return this.config.successReward;

    const prevErrors = state.diagnostics.filter(d => d.type === 'error').length;
    const nextErrors = nextState.diagnostics.filter(d => d.type === 'error').length;
    const prevWarnings = state.diagnostics.filter(d => d.type === 'warning').length;
    const nextWarnings = nextState.diagnostics.filter(d => d.type === 'warning').length;

    const errorDelta = prevErrors - nextErrors;
    const warningDelta = prevWarnings - nextWarnings;

    return (
      errorDelta * Math.abs(this.config.errorPenalty) +
      warningDelta * Math.abs(this.config.warningPenalty)
    );
  }

  private updatePolicy(transition: RLCFTransition): void {
    const { action, reward } = transition;
    const currentProb = this.policy.actionProbs.get(action) || 0;
    const newProb = currentProb + this.config.learningRate * (reward - currentProb);
    this.policy.actionProbs.set(action, Math.max(0.01, Math.min(0.99, newProb)));
    this.normalizeActionProbs();
  }

  private normalizeActionProbs(): void {
    let sum = 0;
    for (const prob of this.policy.actionProbs.values()) sum += prob;
    if (sum > 0) {
      for (const [action, prob] of this.policy.actionProbs) {
        this.policy.actionProbs.set(action, prob / sum);
      }
    }
  }

  private registerDefaultStrategies(): void {
    this.registerStrategy({
      name: 'add-import',
      handles: ['TS2304', 'TS2305', 'TS2307'],
      apply: (code, diagnostic) => {
        const match = diagnostic.message.match(/Cannot find name '(\w+)'/);
        if (match) {
          const name = match[1];
          return `import { ${name} } from './${name.toLowerCase()}';\n${code}`;
        }
        return code;
      },
    });

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

    this.registerStrategy({
      name: 'add-any-type',
      handles: ['TS7006', 'TS7031'],
      apply: (code, diagnostic) => {
        if (diagnostic.location) {
          const lines = code.split('\n');
          const line = lines[diagnostic.location.line - 1];
          if (line) {
            const col = diagnostic.location.column - 1;
            const before = line.slice(0, col);
            const after = line.slice(col);
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

    this.registerStrategy({
      name: 'remove-unused',
      handles: ['TS6133', 'TS6196'],
      apply: (code, diagnostic) => {
        if (diagnostic.location) {
          const lines = code.split('\n');
          lines[diagnostic.location.line - 1] = '// REMOVED: ' + lines[diagnostic.location.line - 1];
          return lines.join('\n');
        }
        return code;
      },
    });
  }

  getStats(): {
    totalIterations: number;
    successRate: number;
    avgIterationsToSuccess: number;
    topStrategies: Array<{ name: string; prob: number }>;
  } {
    const trajectories = this.groupTrajectories();
    const successful = trajectories.filter(t => t[t.length - 1]?.nextState.success);

    const successRate = trajectories.length > 0 ? successful.length / trajectories.length : 0;
    const avgIterations = successful.length > 0
      ? successful.reduce((sum, t) => sum + t.length, 0) / successful.length
      : 0;

    const topStrategies = Array.from(this.policy.actionProbs.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([name, prob]) => ({ name, prob }));

    return { totalIterations: this.experienceHistory.length, successRate, avgIterationsToSuccess: avgIterations, topStrategies };
  }

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

    if (current.length > 0) trajectories.push(current);
    return trajectories;
  }

  exportPolicy(): { actionProbs: Record<string, number>; stateValue: number } {
    return {
      actionProbs: Object.fromEntries(this.policy.actionProbs),
      stateValue: this.policy.stateValue,
    };
  }

  importPolicy(data: { actionProbs: Record<string, number>; stateValue: number }): void {
    this.policy.actionProbs = new Map(Object.entries(data.actionProbs));
    this.policy.stateValue = data.stateValue;
  }
}

export async function refineWithRLCF(
  code: string,
  compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>,
  config?: Partial<RLCFConfig>
): Promise<RLCFState> {
  const rlcf = new RLCFLoop(config);
  return rlcf.refine(code, compile);
}
