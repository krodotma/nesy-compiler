/**
 * Proof Search - Neural-Guided Proof Discovery
 *
 * Phase 3 Step 28: Neural Search Integration
 *
 * Implements AlphaGeometry-style proof search where:
 * - Neural network proposes construction steps
 * - Symbolic engine verifies each step
 * - Search is guided by learned heuristics
 *
 * Key algorithms:
 * - Best-first search with neural value estimates
 * - MCTS for exploration/exploitation balance
 * - Beam search for parallel hypothesis tracking
 */

import type { Term, Constraint, SymbolicStructure, Substitution } from '@nesy/core';
import { unify, applySubstitution } from '@nesy/core';
import type { Puzzle, PuzzleStep } from './types.js';

/** Proof state during search */
export interface ProofState {
  id: string;
  facts: Term[];
  goals: Term[];
  constraints: Constraint[];
  steps: ProofStep[];
  depth: number;
  score: number;
  parent?: string;
}

/** A single proof step */
export interface ProofStep {
  rule: string;
  premises: Term[];
  conclusion: Term;
  justification: Justification;
  confidence: number;
}

/** Justification for a proof step */
export type Justification =
  | { type: 'axiom'; name: string }
  | { type: 'assumption' }
  | { type: 'inference'; rule: string; premises: number[] }
  | { type: 'neural'; modelId: string; logprob: number };

/** Inference rule */
export interface InferenceRule {
  name: string;
  premises: Term[];
  conclusion: Term;
  constraints?: Constraint[];
  priority: number;
}

/** Neural guidance interface */
export interface NeuralGuide {
  /** Score a state (higher = more promising) */
  scoreState(state: ProofState): Promise<number>;
  /** Propose next actions */
  proposeActions(state: ProofState): Promise<ProposedAction[]>;
  /** Score an action */
  scoreAction(state: ProofState, action: ProposedAction): Promise<number>;
}

/** Proposed action from neural guide */
export interface ProposedAction {
  type: 'apply-rule' | 'introduce-auxiliary' | 'split-goal' | 'lemma-lookup';
  rule?: InferenceRule;
  auxiliary?: Term;
  subgoals?: Term[];
  lemma?: string;
  confidence: number;
}

/** Search configuration */
export interface SearchConfig {
  maxDepth: number;
  maxStates: number;
  beamWidth: number;
  explorationWeight: number;
  timeoutMs: number;
  strategy: 'best-first' | 'beam' | 'mcts';
}

const DEFAULT_CONFIG: SearchConfig = {
  maxDepth: 50,
  maxStates: 10000,
  beamWidth: 10,
  explorationWeight: 1.4,
  timeoutMs: 30000,
  strategy: 'best-first',
};

/** Search statistics */
export interface SearchStats {
  statesExplored: number;
  maxDepthReached: number;
  rulesApplied: number;
  backtrackCount: number;
  neuralCalls: number;
  timeElapsedMs: number;
}

/** Search result */
export interface SearchResult {
  success: boolean;
  proof?: ProofState;
  stats: SearchStats;
  explored: ProofState[];
}

/**
 * ProofSearcher: Neural-guided proof search.
 */
export class ProofSearcher {
  private config: SearchConfig;
  private rules: InferenceRule[] = [];
  private guide: NeuralGuide | null = null;
  private stateCounter = 0;

  constructor(config?: Partial<SearchConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Set neural guide for action proposal and scoring.
   */
  setGuide(guide: NeuralGuide): void {
    this.guide = guide;
  }

  /**
   * Register an inference rule.
   */
  registerRule(rule: InferenceRule): void {
    this.rules.push(rule);
    // Sort by priority
    this.rules.sort((a, b) => b.priority - a.priority);
  }

  /**
   * Search for a proof.
   */
  async search(puzzle: Puzzle): Promise<SearchResult> {
    const startTime = Date.now();
    const stats: SearchStats = {
      statesExplored: 0,
      maxDepthReached: 0,
      rulesApplied: 0,
      backtrackCount: 0,
      neuralCalls: 0,
      timeElapsedMs: 0,
    };

    // Initialize
    const initial = this.createInitialState(puzzle);
    const explored: ProofState[] = [];

    // Choose search strategy
    let result: SearchResult;
    switch (this.config.strategy) {
      case 'beam':
        result = await this.beamSearch(initial, puzzle, stats);
        break;
      case 'mcts':
        result = await this.mctsSearch(initial, puzzle, stats);
        break;
      default:
        result = await this.bestFirstSearch(initial, puzzle, stats);
    }

    stats.timeElapsedMs = Date.now() - startTime;
    return result;
  }

  /**
   * Best-first search with neural guidance.
   */
  private async bestFirstSearch(
    initial: ProofState,
    puzzle: Puzzle,
    stats: SearchStats
  ): Promise<SearchResult> {
    // Priority queue (max-heap by score)
    const frontier: ProofState[] = [initial];
    const explored = new Map<string, ProofState>();

    while (frontier.length > 0 && stats.statesExplored < this.config.maxStates) {
      // Check timeout
      if (this.isTimedOut(stats)) {
        break;
      }

      // Get best state
      frontier.sort((a, b) => b.score - a.score);
      const current = frontier.shift()!;

      // Check if goal reached
      if (this.isGoalState(current, puzzle)) {
        return {
          success: true,
          proof: current,
          stats,
          explored: Array.from(explored.values()),
        };
      }

      // Skip if already explored
      const stateKey = this.stateKey(current);
      if (explored.has(stateKey)) {
        continue;
      }
      explored.set(stateKey, current);
      stats.statesExplored++;
      stats.maxDepthReached = Math.max(stats.maxDepthReached, current.depth);

      // Generate successors
      const successors = await this.generateSuccessors(current, puzzle, stats);

      for (const successor of successors) {
        if (!explored.has(this.stateKey(successor))) {
          frontier.push(successor);
        }
      }
    }

    return {
      success: false,
      stats,
      explored: Array.from(explored.values()),
    };
  }

  /**
   * Beam search - parallel hypothesis tracking.
   */
  private async beamSearch(
    initial: ProofState,
    puzzle: Puzzle,
    stats: SearchStats
  ): Promise<SearchResult> {
    let beam: ProofState[] = [initial];
    const explored: ProofState[] = [];

    while (beam.length > 0 && stats.statesExplored < this.config.maxStates) {
      if (this.isTimedOut(stats)) {
        break;
      }

      // Check for goal in current beam
      for (const state of beam) {
        if (this.isGoalState(state, puzzle)) {
          return {
            success: true,
            proof: state,
            stats,
            explored,
          };
        }
      }

      // Generate all successors
      const allSuccessors: ProofState[] = [];
      for (const state of beam) {
        explored.push(state);
        stats.statesExplored++;
        stats.maxDepthReached = Math.max(stats.maxDepthReached, state.depth);

        const successors = await this.generateSuccessors(state, puzzle, stats);
        allSuccessors.push(...successors);
      }

      // Keep top beamWidth successors
      allSuccessors.sort((a, b) => b.score - a.score);
      beam = allSuccessors.slice(0, this.config.beamWidth);
    }

    return {
      success: false,
      stats,
      explored,
    };
  }

  /**
   * Monte Carlo Tree Search.
   */
  private async mctsSearch(
    initial: ProofState,
    puzzle: Puzzle,
    stats: SearchStats
  ): Promise<SearchResult> {
    // MCTS node with UCB1 statistics
    interface MCTSNode {
      state: ProofState;
      visits: number;
      totalScore: number;
      children: MCTSNode[];
      parent: MCTSNode | null;
    }

    const root: MCTSNode = {
      state: initial,
      visits: 0,
      totalScore: 0,
      children: [],
      parent: null,
    };

    const explored: ProofState[] = [];
    let bestProof: ProofState | null = null;

    while (stats.statesExplored < this.config.maxStates && !this.isTimedOut(stats)) {
      // Selection: UCB1 tree policy
      let node = root;
      while (node.children.length > 0) {
        node = this.selectChild(node);
      }

      // Expansion
      if (node.visits > 0 && node.state.depth < this.config.maxDepth) {
        const successors = await this.generateSuccessors(node.state, puzzle, stats);
        for (const succ of successors) {
          node.children.push({
            state: succ,
            visits: 0,
            totalScore: 0,
            children: [],
            parent: node,
          });
        }
        if (node.children.length > 0) {
          node = node.children[0];
        }
      }

      // Simulation: evaluate state
      explored.push(node.state);
      stats.statesExplored++;

      let score = node.state.score;
      if (this.isGoalState(node.state, puzzle)) {
        score = 1.0;
        if (!bestProof || node.state.steps.length < bestProof.steps.length) {
          bestProof = node.state;
        }
      }

      // Backpropagation
      let backNode: MCTSNode | null = node;
      while (backNode !== null) {
        backNode.visits++;
        backNode.totalScore += score;
        backNode = backNode.parent;
      }
    }

    return {
      success: bestProof !== null,
      proof: bestProof ?? undefined,
      stats,
      explored,
    };
  }

  /**
   * Select child node using UCB1.
   */
  private selectChild(node: { children: { visits: number; totalScore: number }[]; visits: number }): any {
    const c = this.config.explorationWeight;

    let best = node.children[0];
    let bestUCB = -Infinity;

    for (const child of node.children) {
      let ucb: number;
      if (child.visits === 0) {
        ucb = Infinity; // Explore unvisited
      } else {
        const exploitation = child.totalScore / child.visits;
        const exploration = c * Math.sqrt(Math.log(node.visits) / child.visits);
        ucb = exploitation + exploration;
      }

      if (ucb > bestUCB) {
        bestUCB = ucb;
        best = child;
      }
    }

    return best;
  }

  /**
   * Generate successor states.
   */
  private async generateSuccessors(
    state: ProofState,
    puzzle: Puzzle,
    stats: SearchStats
  ): Promise<ProofState[]> {
    const successors: ProofState[] = [];

    // Get neural proposals if available
    if (this.guide) {
      stats.neuralCalls++;
      const proposals = await this.guide.proposeActions(state);

      for (const proposal of proposals) {
        const successor = await this.applyAction(state, proposal, stats);
        if (successor) {
          successors.push(successor);
        }
      }
    }

    // Also try standard inference rules
    for (const rule of this.rules) {
      const ruleSuccessors = this.applyRule(state, rule, stats);
      successors.push(...ruleSuccessors);
    }

    // Score successors
    for (const succ of successors) {
      if (this.guide) {
        stats.neuralCalls++;
        succ.score = await this.guide.scoreState(succ);
      } else {
        succ.score = this.heuristicScore(succ, puzzle);
      }
    }

    return successors;
  }

  /**
   * Apply a neural-proposed action.
   */
  private async applyAction(
    state: ProofState,
    action: ProposedAction,
    stats: SearchStats
  ): Promise<ProofState | null> {
    switch (action.type) {
      case 'apply-rule':
        if (action.rule) {
          const results = this.applyRule(state, action.rule, stats);
          return results[0] || null;
        }
        return null;

      case 'introduce-auxiliary':
        if (action.auxiliary) {
          return this.introduceAuxiliary(state, action.auxiliary);
        }
        return null;

      case 'split-goal':
        if (action.subgoals && state.goals.length > 0) {
          return this.splitGoal(state, action.subgoals);
        }
        return null;

      default:
        return null;
    }
  }

  /**
   * Apply an inference rule to state.
   */
  private applyRule(
    state: ProofState,
    rule: InferenceRule,
    stats: SearchStats
  ): ProofState[] {
    const successors: ProofState[] = [];

    // Try to match rule premises with facts
    const matches = this.findMatches(rule.premises, state.facts);

    for (const match of matches) {
      // Apply substitution to conclusion
      const conclusion = applySubstitution(rule.conclusion, match);

      // Check if this proves a goal
      const provedGoals: number[] = [];
      for (let i = 0; i < state.goals.length; i++) {
        if (unify(conclusion, state.goals[i])) {
          provedGoals.push(i);
        }
      }

      // Create new state
      const newStep: ProofStep = {
        rule: rule.name,
        premises: rule.premises.map(p => applySubstitution(p, match)),
        conclusion,
        justification: { type: 'inference', rule: rule.name, premises: [] },
        confidence: 0.9,
      };

      const newState: ProofState = {
        id: this.nextStateId(),
        facts: [...state.facts, conclusion],
        goals: state.goals.filter((_, i) => !provedGoals.includes(i)),
        constraints: state.constraints,
        steps: [...state.steps, newStep],
        depth: state.depth + 1,
        score: 0,
        parent: state.id,
      };

      stats.rulesApplied++;
      successors.push(newState);
    }

    return successors;
  }

  /**
   * Find all ways to match premises against facts.
   */
  private findMatches(premises: Term[], facts: Term[]): Substitution[] {
    if (premises.length === 0) {
      return [{}];
    }

    const [first, ...rest] = premises;
    const matches: Substitution[] = [];

    for (const fact of facts) {
      const subst = unify(first, fact);
      if (subst) {
        // Apply to remaining premises
        const remainingPremises = rest.map(p => applySubstitution(p, subst));
        const remainingMatches = this.findMatches(remainingPremises, facts);

        for (const restMatch of remainingMatches) {
          // Combine substitutions
          const combined: Substitution = { ...subst, ...restMatch };
          matches.push(combined);
        }
      }
    }

    return matches;
  }

  /**
   * Introduce auxiliary construction.
   */
  private introduceAuxiliary(state: ProofState, auxiliary: Term): ProofState {
    const step: ProofStep = {
      rule: 'auxiliary',
      premises: [],
      conclusion: auxiliary,
      justification: { type: 'neural', modelId: 'guide', logprob: 0 },
      confidence: 0.8,
    };

    return {
      id: this.nextStateId(),
      facts: [...state.facts, auxiliary],
      goals: state.goals,
      constraints: state.constraints,
      steps: [...state.steps, step],
      depth: state.depth + 1,
      score: 0,
      parent: state.id,
    };
  }

  /**
   * Split goal into subgoals.
   */
  private splitGoal(state: ProofState, subgoals: Term[]): ProofState {
    return {
      id: this.nextStateId(),
      facts: state.facts,
      goals: [...state.goals.slice(1), ...subgoals],
      constraints: state.constraints,
      steps: state.steps,
      depth: state.depth + 1,
      score: 0,
      parent: state.id,
    };
  }

  /**
   * Check if state achieves goal.
   */
  private isGoalState(state: ProofState, puzzle: Puzzle): boolean {
    // All goals must be satisfied
    return state.goals.length === 0 ||
      state.goals.every(goal =>
        state.facts.some(fact => unify(fact, goal) !== null)
      );
  }

  /**
   * Heuristic score for state (when no neural guide).
   */
  private heuristicScore(state: ProofState, puzzle: Puzzle): number {
    // Reward fewer remaining goals
    const goalScore = 1 - (state.goals.length / Math.max(1, puzzle.constraints.length));

    // Reward more facts
    const factScore = Math.min(1, state.facts.length / 20);

    // Penalize depth
    const depthPenalty = Math.min(0.5, state.depth / this.config.maxDepth);

    return (goalScore * 0.5 + factScore * 0.3) * (1 - depthPenalty);
  }

  /**
   * Create initial search state.
   */
  private createInitialState(puzzle: Puzzle): ProofState {
    return {
      id: this.nextStateId(),
      facts: puzzle.givens,
      goals: [puzzle.goal],
      constraints: puzzle.constraints,
      steps: [],
      depth: 0,
      score: 0,
    };
  }

  /**
   * Generate unique state ID.
   */
  private nextStateId(): string {
    return `s${this.stateCounter++}`;
  }

  /**
   * Generate state key for duplicate detection.
   */
  private stateKey(state: ProofState): string {
    return JSON.stringify({
      facts: state.facts.map(f => JSON.stringify(f)).sort(),
      goals: state.goals.map(g => JSON.stringify(g)).sort(),
    });
  }

  /**
   * Check if search has timed out.
   */
  private isTimedOut(stats: SearchStats): boolean {
    return stats.timeElapsedMs > this.config.timeoutMs;
  }
}

/**
 * Quick search helper.
 */
export function searchProof(
  puzzle: Puzzle,
  rules: InferenceRule[],
  guide?: NeuralGuide,
  config?: Partial<SearchConfig>
): Promise<SearchResult> {
  const searcher = new ProofSearcher(config);

  for (const rule of rules) {
    searcher.registerRule(rule);
  }

  if (guide) {
    searcher.setGuide(guide);
  }

  return searcher.search(puzzle);
}
