import type { SymbolicStructure, Term, Constraint, TrustLevel } from '@nesy/core';

export interface Puzzle {
  id: string;
  description: string;
  givens: Term[];
  goal: Term;
  constraints: Constraint[];
  domain?: string;
}

export interface PuzzleStep {
  id: string;
  action: string;
  justification: string;
  terms: Term[];
  confidence: number;
}

export interface PuzzleSolution {
  puzzle: Puzzle;
  steps: PuzzleStep[];
  proof: SymbolicStructure;
  verified: boolean;
  trust: TrustLevel;
  solvedAt: number;
}
