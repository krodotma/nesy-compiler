import type { Term } from '@nesy/core';
import type { Puzzle } from './types.js';

export interface SubPuzzle {
  description: string;
  givens: Term[];
  goal: Term;
  order: number;
}

/**
 * Decomposes complex puzzles into sub-problems
 */
export class PuzzleDecomposer {
  decompose(puzzle: Puzzle): SubPuzzle[] {
    // Simple decomposition: one sub-puzzle per constraint + final goal
    const subs: SubPuzzle[] = [];

    for (let i = 0; i < puzzle.constraints.length; i++) {
      const constraint = puzzle.constraints[i];
      subs.push({
        description: `Satisfy constraint ${i + 1}: ${constraint.type}`,
        givens: puzzle.givens,
        goal: puzzle.goal,
        order: i,
      });
    }

    // Final goal step
    subs.push({
      description: `Reach goal: ${puzzle.description}`,
      givens: puzzle.givens,
      goal: puzzle.goal,
      order: subs.length,
    });

    return subs;
  }
}
