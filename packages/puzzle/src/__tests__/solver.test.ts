import { describe, it, expect } from 'vitest';
import { PuzzleSolver } from '../solver.js';
import type { Puzzle } from '../types.js';

describe('PuzzleSolver', () => {
  it('solves puzzle', async () => {
    const puzzle: Puzzle = {
      id: 'g1', description: 'test',
      givens: [{ type: 'constant', name: 'A' }],
      goal: { type: 'constant', name: 'A' },
      constraints: [{ type: 'equality', left: { type: 'constant', name: 'A' }, right: { type: 'constant', name: 'A' } }],
    };
    const s = await new PuzzleSolver({ maxSteps: 5 }).solve(puzzle);
    expect(s.steps.length).toBeGreaterThan(0);
    expect(typeof s.verified).toBe('boolean');
  });

  it('handles no constraints', async () => {
    const s = await new PuzzleSolver().solve({
      id: 's1', description: 'simple',
      givens: [{ type: 'constant', name: 'x' }],
      goal: { type: 'constant', name: 'x' },
      constraints: [],
    });
    expect(s.steps).toHaveLength(1);
  });
});
