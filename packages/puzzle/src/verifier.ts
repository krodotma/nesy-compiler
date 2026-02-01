import type { Term } from '@nesy/core';
import { unify } from '@nesy/core';
import type { Puzzle, PuzzleStep } from './types.js';

/**
 * Symbolic verifier for puzzle solutions
 */
export class PuzzleVerifier {
  verifyStep(step: PuzzleStep, puzzle: Puzzle): boolean {
    // Check that step terms unify with known terms
    for (const stepTerm of step.terms) {
      const unified = puzzle.givens.some(given => unify(stepTerm, given) !== null);
      if (!unified && step.confidence < 0.5) {
        return false;
      }
    }
    return step.confidence > 0;
  }

  verifySolution(steps: PuzzleStep[], puzzle: Puzzle): boolean {
    if (steps.length === 0) return false;

    // All steps must have positive confidence
    const allValid = steps.every(s => s.confidence > 0);
    if (!allValid) return false;

    // Last step should relate to the goal
    const lastStep = steps[steps.length - 1];
    const goalReached = lastStep.terms.some(t => unify(t, puzzle.goal) !== null);

    return goalReached || lastStep.confidence >= 0.8;
  }
}
