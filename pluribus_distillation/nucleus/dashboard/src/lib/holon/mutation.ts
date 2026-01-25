/**
 * Mutation Engine Schema
 * [Ultrathink Agent 1: Architect]
 * 
 * Defines the strategies, risks, and payload structures for the 
 * Evolutionary Code Operations (ECO).
 */

export type MutationStrategyType = 'optimization' | 'refactor' | 'hgt_injection' | 'bug_fix' | 'crossover';

export interface MutationStrategy {
  id: MutationStrategyType;
  label: string;
  description: string;
  risk: 'low' | 'medium' | 'high' | 'extreme';
  energyCost: number; // Entropy cost
  cmpPotential: number; // Potential CMP gain
}

export const MUTATION_STRATEGIES: MutationStrategy[] = [
  {
    id: 'optimization',
    label: 'Loop Optimization',
    description: 'Unroll loops and memoize invariants.',
    risk: 'low',
    energyCost: 5,
    cmpPotential: 2
  },
  {
    id: 'refactor',
    label: 'Structural Refactor',
    description: 'Decompose monolithic functions into atomic units.',
    risk: 'medium',
    energyCost: 15,
    cmpPotential: 8
  },
  {
    id: 'hgt_injection',
    label: 'HGT Injection',
    description: 'Horizontal Gene Transfer from high-CMP lineages.',
    risk: 'high',
    energyCost: 40,
    cmpPotential: 25
  },
  {
    id: 'crossover',
    label: 'Sexual Crossover',
    description: 'Recombine traits with a compatible mate from the registry.',
    risk: 'extreme',
    energyCost: 80,
    cmpPotential: 50
  }
];

export interface MutationProposal {
  id: string;
  strategy: MutationStrategyType;
  targetNodeId: string;
  diffSummary: string; // Placeholder for diff
  predictedCmpDelta: number;
}
