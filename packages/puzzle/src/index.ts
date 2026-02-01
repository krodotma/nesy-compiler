/**
 * @nesy/puzzle - AlphaGeometry-inspired puzzle solving
 *
 * Neural proposes, symbolic verifies
 */
export { PuzzleSolver, type PuzzleConfig } from './solver.js';
export { PuzzleVerifier } from './verifier.js';
export { PuzzleDecomposer, type SubPuzzle } from './decomposition.js';
export type { Puzzle, PuzzleSolution, PuzzleStep } from './types.js';
