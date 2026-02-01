import { describe, it, expect } from 'vitest';
import { PuzzleVerifier } from '../verifier.js';
import { PuzzleDecomposer } from '../decomposition.js';
import type { Puzzle, PuzzleStep } from '../types.js';

const puzzle: Puzzle = {
  id: 'test-1',
  description: 'Test puzzle',
  givens: [{ type: 'constant', name: 'a' }, { type: 'constant', name: 'b' }],
  goal: { type: 'constant', name: 'b' },
  constraints: [{ type: 'equality', left: { type: 'variable', name: 'X' }, right: { type: 'constant', name: 'a' } }],
};

describe('PuzzleVerifier', () => {
  it('verifies valid step', () => {
    const verifier = new PuzzleVerifier();
    const step: PuzzleStep = {
      id: 's1', action: 'apply', justification: 'given',
      terms: [{ type: 'constant', name: 'a' }], confidence: 0.9,
    };
    expect(verifier.verifyStep(step, puzzle)).toBe(true);
  });

  it('verifies complete solution', () => {
    const verifier = new PuzzleVerifier();
    const steps: PuzzleStep[] = [
      { id: 's1', action: 'step1', justification: 'ok', terms: [{ type: 'constant', name: 'a' }], confidence: 0.9 },
      { id: 's2', action: 'step2', justification: 'ok', terms: [{ type: 'constant', name: 'b' }], confidence: 0.9 },
    ];
    expect(verifier.verifySolution(steps, puzzle)).toBe(true);
  });

  it('rejects empty solution', () => {
    const verifier = new PuzzleVerifier();
    expect(verifier.verifySolution([], puzzle)).toBe(false);
  });
});

describe('PuzzleDecomposer', () => {
  it('decomposes puzzle into sub-puzzles', () => {
    const decomposer = new PuzzleDecomposer();
    const subs = decomposer.decompose(puzzle);
    // 1 constraint + 1 final goal
    expect(subs).toHaveLength(2);
    expect(subs[subs.length - 1].description).toContain('goal');
  });
});
