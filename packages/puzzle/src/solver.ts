import type { SymbolicStructure, TrustLevel } from '@nesy/core';
import { NeSyCompiler, type CompilationResult } from '@nesy/pipeline';
import type { Puzzle, PuzzleSolution, PuzzleStep } from './types.js';
import { PuzzleVerifier } from './verifier.js';
import { PuzzleDecomposer } from './decomposition.js';

export interface PuzzleConfig {
  maxSteps?: number;
  verifyEachStep?: boolean;
  trustThreshold?: TrustLevel;
}

/**
 * AlphaGeometry-inspired puzzle solver
 *
 * Neural network proposes construction steps,
 * symbolic engine verifies each step
 */
export class PuzzleSolver {
  private config: Required<PuzzleConfig>;
  private compiler: NeSyCompiler;
  private verifier: PuzzleVerifier;
  private decomposer: PuzzleDecomposer;

  constructor(config: PuzzleConfig = {}) {
    this.config = {
      maxSteps: 50,
      verifyEachStep: true,
      trustThreshold: 2,
      ...config,
    };
    this.compiler = new NeSyCompiler({ maxIterations: this.config.maxSteps });
    this.verifier = new PuzzleVerifier();
    this.decomposer = new PuzzleDecomposer();
  }

  async solve(puzzle: Puzzle): Promise<PuzzleSolution> {
    const subPuzzles = this.decomposer.decompose(puzzle);
    const allSteps: PuzzleStep[] = [];

    for (const sub of subPuzzles) {
      const result = await this.compiler.compile({
        mode: 'atom',
        intent: `Solve sub-puzzle: ${sub.description}`,
      });

      const step: PuzzleStep = {
        id: `step_${allSteps.length}`,
        action: sub.description,
        justification: result.stages.verify.passed ? 'verified' : 'unverified',
        terms: sub.givens,
        confidence: result.stages.verify.passed ? 0.9 : 0.3,
      };

      if (this.config.verifyEachStep) {
        const valid = this.verifier.verifyStep(step, puzzle);
        if (!valid) {
          step.justification = 'failed_verification';
          step.confidence = 0;
        }
      }

      allSteps.push(step);
    }

    const proof: SymbolicStructure = {
      terms: puzzle.givens,
      constraints: puzzle.constraints,
      metadata: { steps: allSteps.length },
    };

    const verified = this.verifier.verifySolution(allSteps, puzzle);

    return {
      puzzle,
      steps: allSteps,
      proof,
      verified,
      trust: verified ? 1 : 3,
      solvedAt: Date.now(),
    };
  }
}
