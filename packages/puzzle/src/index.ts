/**
 * @nesy/puzzle - AlphaGeometry-inspired Constraint Solving
 *
 * Phase 3 of NeSy Evolution: Constraint Solving & Verification
 *
 * Components:
 * - PuzzleSolver: High-level puzzle solving interface
 * - SATEncoder: Convert constraints to CNF
 * - SMTSolver: SMT-LIB2 compatible constraint solving
 * - ProofSearcher: Neural-guided proof search
 * - TheoremProver: Formal proof generation
 * - NeuralHeuristic: Learned search guidance
 *
 * Core pattern: Neural network proposes, symbolic engine verifies.
 */

// Core puzzle types and solver
export { PuzzleSolver, type PuzzleConfig } from './solver.js';
export { PuzzleVerifier } from './verifier.js';
export { PuzzleDecomposer, type SubPuzzle } from './decomposition.js';
export type { Puzzle, PuzzleSolution, PuzzleStep } from './types.js';

// SAT Encoding (Step 26)
export {
  SATEncoder,
  constraintsToDIMACS,
  type Literal,
  type Clause,
  type CNFFormula,
  type SATResult,
  type EncoderConfig,
} from './sat-encoder.js';

// SMT Interface (Step 27)
export {
  SMTSolver,
  checkConstraints,
  type Sort,
  type SMTExpr,
  type SMTDeclaration,
  type SMTAssertion,
  type CheckResult,
  type SMTModel,
  type UnsatCore,
  type SMTConfig,
  type SMTLogic,
} from './smt-interface.js';

// Proof Search (Step 28)
export {
  ProofSearcher,
  searchProof,
  type ProofState,
  type ProofStep,
  type Justification,
  type InferenceRule,
  type NeuralGuide,
  type ProposedAction,
  type SearchConfig,
  type SearchStats,
  type SearchResult,
} from './proof-search.js';

// Theorem Prover (Step 29)
export {
  TheoremProver,
  exportProof,
  type ProofNode,
  type ProofDocument,
  type ProofLine,
  type ProofFormat,
  type ProverConfig,
} from './theorem-prover.js';

// Neural Heuristic (Step 30)
export {
  NeuralHeuristic,
  createNeuralGuide,
  createMockNeuralModel,
  type NeuralModel,
  type CompletionOptions,
  type HeuristicConfig,
} from './neural-heuristic.js';
