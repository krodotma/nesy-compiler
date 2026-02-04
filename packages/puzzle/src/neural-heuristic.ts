/**
 * Neural Heuristic - Learned Guidance for Proof Search
 *
 * Phase 3 Step 30: Neural Value Network
 *
 * Provides neural network-based heuristics for guiding proof search:
 * - State value estimation (how promising is this state?)
 * - Action policy (which actions should we try?)
 * - Auxiliary construction proposal
 *
 * Inspired by AlphaGeometry's approach of using language models
 * to propose geometric constructions.
 */

import type { Term } from '@nesy/core';
import { termToString, isVariable, isCompound } from '@nesy/core';
import type { ProofState, NeuralGuide, ProposedAction, InferenceRule } from './proof-search.js';

/** Neural model interface */
export interface NeuralModel {
  /** Generate text completion */
  complete(prompt: string, options?: CompletionOptions): Promise<string>;
  /** Get embedding for text */
  embed(text: string): Promise<number[]>;
  /** Score multiple candidates */
  score(prompt: string, candidates: string[]): Promise<number[]>;
}

/** Completion options */
export interface CompletionOptions {
  temperature?: number;
  maxTokens?: number;
  stopSequences?: string[];
}

/** Heuristic configuration */
export interface HeuristicConfig {
  /** Model for value estimation */
  valueModel: NeuralModel | null;
  /** Model for action proposal */
  policyModel: NeuralModel | null;
  /** Available inference rules */
  rules: InferenceRule[];
  /** Temperature for action sampling */
  temperature: number;
  /** Number of actions to propose */
  topK: number;
  /** Cache embeddings for efficiency */
  useCache: boolean;
}

const DEFAULT_CONFIG: HeuristicConfig = {
  valueModel: null,
  policyModel: null,
  rules: [],
  temperature: 0.7,
  topK: 5,
  useCache: true,
};

/** Action type weights for balancing exploration */
const ACTION_TYPE_WEIGHTS: Record<string, number> = {
  'apply-rule': 0.6,
  'introduce-auxiliary': 0.2,
  'split-goal': 0.15,
  'lemma-lookup': 0.05,
};

/**
 * NeuralHeuristic: Neural guidance for proof search.
 */
export class NeuralHeuristic implements NeuralGuide {
  private config: HeuristicConfig;
  private embeddingCache: Map<string, number[]> = new Map();
  private stateCache: Map<string, number> = new Map();

  constructor(config?: Partial<HeuristicConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Score a proof state (higher = more promising).
   */
  async scoreState(state: ProofState): Promise<number> {
    // Check cache
    const key = this.stateKey(state);
    if (this.config.useCache && this.stateCache.has(key)) {
      return this.stateCache.get(key)!;
    }

    // Use value model if available
    if (this.config.valueModel) {
      const prompt = this.stateToPrompt(state);
      const response = await this.config.valueModel.complete(prompt, {
        temperature: 0.0,
        maxTokens: 10,
      });
      const score = this.parseValueResponse(response);
      this.stateCache.set(key, score);
      return score;
    }

    // Fallback: heuristic scoring
    const score = this.heuristicScore(state);
    this.stateCache.set(key, score);
    return score;
  }

  /**
   * Propose next actions for a state.
   */
  async proposeActions(state: ProofState): Promise<ProposedAction[]> {
    const actions: ProposedAction[] = [];

    // 1. Propose rule applications
    const ruleActions = await this.proposeRuleApplications(state);
    actions.push(...ruleActions);

    // 2. Propose auxiliary constructions (if neural model available)
    if (this.config.policyModel) {
      const auxActions = await this.proposeAuxiliaries(state);
      actions.push(...auxActions);
    }

    // 3. Propose goal splits
    const splitActions = this.proposeGoalSplits(state);
    actions.push(...splitActions);

    // Sort by confidence and return top-k
    actions.sort((a, b) => b.confidence - a.confidence);
    return actions.slice(0, this.config.topK);
  }

  /**
   * Score a specific action.
   */
  async scoreAction(state: ProofState, action: ProposedAction): Promise<number> {
    if (this.config.policyModel) {
      const statePrompt = this.stateToPrompt(state);
      const actionStr = this.actionToString(action);
      const scores = await this.config.policyModel.score(statePrompt, [actionStr]);
      return scores[0] || 0.5;
    }

    // Fallback: use base confidence
    return action.confidence;
  }

  /**
   * Propose rule applications.
   */
  private async proposeRuleApplications(state: ProofState): Promise<ProposedAction[]> {
    const actions: ProposedAction[] = [];

    for (const rule of this.config.rules) {
      // Check if rule is applicable
      const applicable = this.isRuleApplicable(rule, state);
      if (!applicable) continue;

      // Score the rule application
      let confidence = rule.priority / 10; // Normalize priority to [0, 1]

      // Boost if conclusion matches a goal
      for (const goal of state.goals) {
        if (this.termsMatch(rule.conclusion, goal)) {
          confidence += 0.3;
          break;
        }
      }

      actions.push({
        type: 'apply-rule',
        rule,
        confidence: Math.min(1, confidence),
      });
    }

    // If we have a policy model, re-score
    if (this.config.policyModel && actions.length > 0) {
      const actionStrs = actions.map(a => this.actionToString(a));
      const scores = await this.config.policyModel.score(
        this.stateToPrompt(state),
        actionStrs
      );
      for (let i = 0; i < actions.length; i++) {
        actions[i].confidence = (actions[i].confidence + (scores[i] || 0.5)) / 2;
      }
    }

    return actions;
  }

  /**
   * Propose auxiliary constructions using neural model.
   */
  private async proposeAuxiliaries(state: ProofState): Promise<ProposedAction[]> {
    if (!this.config.policyModel) {
      return [];
    }

    const prompt = this.auxiliaryPrompt(state);
    const response = await this.config.policyModel.complete(prompt, {
      temperature: this.config.temperature,
      maxTokens: 50,
      stopSequences: ['\n', '.'],
    });

    const auxiliary = this.parseAuxiliaryResponse(response, state);
    if (!auxiliary) {
      return [];
    }

    return [{
      type: 'introduce-auxiliary',
      auxiliary,
      confidence: 0.6,
    }];
  }

  /**
   * Propose goal splits.
   */
  private proposeGoalSplits(state: ProofState): ProposedAction[] {
    const actions: ProposedAction[] = [];

    for (const goal of state.goals) {
      // Check if goal is compound and can be split
      if (isCompound(goal) && goal.functor === 'and' && goal.args.length >= 2) {
        actions.push({
          type: 'split-goal',
          subgoals: goal.args,
          confidence: 0.5,
        });
      }
    }

    return actions;
  }

  /**
   * Check if rule is applicable to state.
   */
  private isRuleApplicable(rule: InferenceRule, state: ProofState): boolean {
    // Check if all premises can be matched
    const matchedPremises = new Set<string>();

    for (const premise of rule.premises) {
      let matched = false;
      for (const fact of state.facts) {
        if (this.termsMatch(premise, fact)) {
          matched = true;
          break;
        }
      }
      if (!matched) return false;
    }

    return true;
  }

  /**
   * Check if two terms can match (ignoring variables).
   */
  private termsMatch(pattern: Term, target: Term): boolean {
    // Variables match anything
    if (isVariable(pattern)) return true;
    if (isVariable(target)) return true;

    // Constants must match exactly
    if ('value' in pattern && 'value' in target) {
      return pattern.value === target.value;
    }

    // Compound terms
    if (isCompound(pattern) && isCompound(target)) {
      if (pattern.functor !== target.functor) return false;
      if (pattern.args.length !== target.args.length) return false;
      return pattern.args.every((arg, i) => this.termsMatch(arg, target.args[i]));
    }

    return false;
  }

  /**
   * Convert state to prompt for neural model.
   */
  private stateToPrompt(state: ProofState): string {
    const parts: string[] = [];

    parts.push('Proof State:');
    parts.push(`Facts: ${state.facts.map(f => termToString(f)).join(', ')}`);
    parts.push(`Goals: ${state.goals.map(g => termToString(g)).join(', ')}`);
    parts.push(`Depth: ${state.depth}`);
    parts.push(`Steps so far: ${state.steps.length}`);

    return parts.join('\n');
  }

  /**
   * Generate prompt for auxiliary construction.
   */
  private auxiliaryPrompt(state: ProofState): string {
    const parts: string[] = [];

    parts.push('Given the following proof state, suggest a useful auxiliary construction:');
    parts.push('');
    parts.push(this.stateToPrompt(state));
    parts.push('');
    parts.push('Auxiliary construction (one term):');

    return parts.join('\n');
  }

  /**
   * Convert action to string for scoring.
   */
  private actionToString(action: ProposedAction): string {
    switch (action.type) {
      case 'apply-rule':
        return `apply rule "${action.rule?.name}"`;
      case 'introduce-auxiliary':
        return `introduce ${action.auxiliary ? termToString(action.auxiliary) : 'auxiliary'}`;
      case 'split-goal':
        return `split goal into ${action.subgoals?.length || 2} subgoals`;
      case 'lemma-lookup':
        return `lookup lemma "${action.lemma}"`;
    }
  }

  /**
   * Parse value response from model.
   */
  private parseValueResponse(response: string): number {
    // Try to extract a number
    const match = response.match(/(\d+\.?\d*)/);
    if (match) {
      const value = parseFloat(match[1]);
      if (value >= 0 && value <= 1) {
        return value;
      }
      if (value >= 0 && value <= 10) {
        return value / 10;
      }
    }

    // Default based on keywords
    if (response.toLowerCase().includes('promising') || response.toLowerCase().includes('good')) {
      return 0.7;
    }
    if (response.toLowerCase().includes('unlikely') || response.toLowerCase().includes('bad')) {
      return 0.3;
    }

    return 0.5;
  }

  /**
   * Parse auxiliary response from model.
   */
  private parseAuxiliaryResponse(response: string, state: ProofState): Term | null {
    const text = response.trim();
    if (!text) return null;

    // Try to parse as term
    // Simple parsing: assume format "name(args)" or just "name"
    const match = text.match(/^(\w+)(?:\(([^)]*)\))?$/);
    if (!match) return null;

    const [, functor, argsStr] = match;

    if (!argsStr) {
      // Constant or variable
      if (functor[0] === functor[0].toUpperCase()) {
        return { type: 'variable', name: functor };
      }
      return { type: 'constant', name: functor, value: functor };
    }

    // Compound term
    const argNames = argsStr.split(',').map(s => s.trim());
    const args: Term[] = argNames.map(name => {
      if (name[0] === name[0].toUpperCase()) {
        return { type: 'variable', name };
      }
      return { type: 'constant', name, value: name };
    });

    return { type: 'compound', functor, args };
  }

  /**
   * Heuristic score without neural model.
   */
  private heuristicScore(state: ProofState): number {
    let score = 0.5;

    // Reward fewer remaining goals
    if (state.goals.length === 0) {
      score += 0.3;
    } else {
      score -= 0.05 * state.goals.length;
    }

    // Reward more facts
    score += Math.min(0.2, state.facts.length * 0.01);

    // Penalize depth
    score -= Math.min(0.2, state.depth * 0.01);

    // Reward progress (steps taken)
    score += Math.min(0.1, state.steps.length * 0.02);

    return Math.max(0, Math.min(1, score));
  }

  /**
   * Generate state key for caching.
   */
  private stateKey(state: ProofState): string {
    return JSON.stringify({
      facts: state.facts.map(f => termToString(f)).sort(),
      goals: state.goals.map(g => termToString(g)).sort(),
    });
  }

  /**
   * Clear caches.
   */
  clearCache(): void {
    this.embeddingCache.clear();
    this.stateCache.clear();
  }
}

/**
 * Create a simple mock neural model for testing.
 */
export function createMockNeuralModel(): NeuralModel {
  return {
    async complete(prompt: string, options?: CompletionOptions): Promise<string> {
      // Return random-ish response based on prompt hash
      const hash = prompt.split('').reduce((a, c) => a + c.charCodeAt(0), 0);
      return `auxiliary_${hash % 100}`;
    },

    async embed(text: string): Promise<number[]> {
      // Return simple hash-based embedding
      const embedding = new Array(128).fill(0);
      for (let i = 0; i < text.length; i++) {
        embedding[i % 128] += text.charCodeAt(i) / 1000;
      }
      return embedding;
    },

    async score(prompt: string, candidates: string[]): Promise<number[]> {
      // Return uniform scores
      return candidates.map(() => 0.5 + Math.random() * 0.3);
    },
  };
}

/**
 * Factory to create neural heuristic with rules.
 */
export function createNeuralGuide(
  rules: InferenceRule[],
  model?: NeuralModel
): NeuralHeuristic {
  return new NeuralHeuristic({
    rules,
    valueModel: model || null,
    policyModel: model || null,
  });
}
