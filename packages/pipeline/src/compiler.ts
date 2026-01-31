/**
 * NeSyCompiler - Main Neurosymbolic Compiler
 *
 * Orchestrates the 5-stage pipeline:
 * PERCEIVE → GROUND → CONSTRAIN → VERIFY → EMIT
 *
 * Handles all 4 compilation modes:
 * - Genesis: Create new holon from specification
 * - Seed: Evolve existing holon
 * - Atom: Compile atomic operation
 * - Constraint: Apply constraint refinement
 */

import type {
  CompilationContext,
  CompilationRequest,
  CompiledHolon,
  NeuralFeatures,
  SymbolicStructure,
} from '@nesy/core';
import { createContext, addTrace } from '@nesy/core';

import { perceive, type PerceiveInput, type PerceiveOutput } from './stages/perceive';
import { ground, type GroundInput, type GroundOutput } from './stages/ground';
import { constrain, type ConstrainInput, type ConstrainOutput } from './stages/constrain';
import { verify, type VerifyInput, type VerifyOutput, type ProvenanceTrace } from './stages/verify';
import { emit, type EmitInput, type EmitOutput, type StageHistoryEntry } from './stages/emit';
import type {
  PerceiveIR,
  GroundIR,
  ConstrainIR,
  VerifyIR,
  CompiledIR,
  VerificationProof,
} from './ir';

// =============================================================================
// Compiler Configuration
// =============================================================================

export interface CompilerConfig {
  /** Enable caching of intermediate results */
  enableCache?: boolean;
  /** Maximum compilation time in ms */
  timeout?: number;
  /** Maximum constraint search iterations */
  maxIterations?: number;
  /** Enable verbose tracing */
  trace?: boolean;
  /** Model to use for neural operations */
  model?: string;
}

const DEFAULT_CONFIG: Required<CompilerConfig> = {
  enableCache: true,
  timeout: 60000,
  maxIterations: 100,
  trace: false,
  model: 'claude-opus-4-5',
};

// =============================================================================
// Compilation Result
// =============================================================================

export interface CompilationResult {
  /** Final compiled IR */
  ir: CompiledIR;
  /** Compiled holon (if applicable) */
  holon?: CompiledHolon;
  /** Stage outputs for debugging */
  stages: {
    perceive: PerceiveOutput;
    ground: GroundOutput;
    constrain: ConstrainOutput;
    verify: VerifyOutput;
    emit: EmitOutput;
  };
  /** Compilation metrics */
  metrics: CompilationMetrics;
}

export interface CompilationMetrics {
  totalDurationMs: number;
  stageDurations: Record<string, number>;
  cacheHits: number;
  constraintIterations: number;
  verificationGatesPassed: number;
  verificationGatesFailed: number;
}

// =============================================================================
// NeSyCompiler Class
// =============================================================================

export class NeSyCompiler {
  private config: Required<CompilerConfig>;
  private context: CompilationContext;

  constructor(config: CompilerConfig = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.context = createContext({
      model: this.config.model,
      maxIterations: this.config.maxIterations,
      timeoutMs: this.config.timeout,
    });
  }

  /**
   * Main compilation entry point
   */
  async compile(request: CompilationRequest): Promise<CompilationResult> {
    const startTime = Date.now();
    const stageDurations: Record<string, number> = {};
    const stageHistory: StageHistoryEntry[] = [];

    // Extract input based on mode
    const input = this.extractInput(request);

    // Stage 1: PERCEIVE
    const perceiveStart = Date.now();
    const perceiveOutput = await this.runPerceive(input);
    stageDurations.perceive = Date.now() - perceiveStart;
    stageHistory.push({
      stage: 'perceive',
      inputHash: perceiveOutput.ir.inputHash,
      outputHash: perceiveOutput.ir.outputHash,
      timestamp: Date.now(),
    });

    // Stage 2: GROUND
    const groundStart = Date.now();
    const groundOutput = await this.runGround(perceiveOutput);
    stageDurations.ground = Date.now() - groundStart;
    stageHistory.push({
      stage: 'ground',
      inputHash: groundOutput.ir.inputHash,
      outputHash: groundOutput.ir.outputHash,
      timestamp: Date.now(),
    });

    // Stage 3: CONSTRAIN
    const constrainStart = Date.now();
    const constrainOutput = await this.runConstrain(groundOutput, request);
    stageDurations.constrain = Date.now() - constrainStart;
    stageHistory.push({
      stage: 'constrain',
      inputHash: constrainOutput.ir.inputHash,
      outputHash: constrainOutput.ir.outputHash,
      timestamp: Date.now(),
    });

    // Stage 4: VERIFY
    const verifyStart = Date.now();
    const verifyOutput = await this.runVerify(constrainOutput, stageHistory);
    stageDurations.verify = Date.now() - verifyStart;
    stageHistory.push({
      stage: 'verify',
      inputHash: verifyOutput.ir.inputHash,
      outputHash: verifyOutput.ir.outputHash,
      timestamp: Date.now(),
    });

    // Stage 5: EMIT
    const emitStart = Date.now();
    const emitOutput = await this.runEmit(verifyOutput, stageHistory, request);
    stageDurations.emit = Date.now() - emitStart;

    // Build metrics
    const metrics: CompilationMetrics = {
      totalDurationMs: Date.now() - startTime,
      stageDurations,
      cacheHits: this.countCacheHits(),
      constraintIterations: constrainOutput.ir.searchSteps,
      verificationGatesPassed: verifyOutput.ir.gatesPasssed.length,
      verificationGatesFailed: verifyOutput.ir.gatesFailed.length,
    };

    // Build holon if verification passed
    const holon = verifyOutput.passed
      ? this.buildHolon(emitOutput, verifyOutput)
      : undefined;

    return {
      ir: emitOutput.ir,
      holon,
      stages: {
        perceive: perceiveOutput,
        ground: groundOutput,
        constrain: constrainOutput,
        verify: verifyOutput,
        emit: emitOutput,
      },
      metrics,
    };
  }

  // ===========================================================================
  // Stage Runners
  // ===========================================================================

  private extractInput(request: CompilationRequest): PerceiveInput {
    switch (request.mode) {
      case 'genesis':
        return { text: JSON.stringify(request.specification) };

      case 'seed':
        return { text: JSON.stringify(request.mutations) };

      case 'atom':
        return { text: request.intent };

      case 'constraint':
        return { text: JSON.stringify(request.constraints) };

      default:
        throw new Error(`Unknown compilation mode: ${(request as { mode: string }).mode}`);
    }
  }

  private async runPerceive(input: PerceiveInput): Promise<PerceiveOutput> {
    const output = await perceive(input, this.context);

    if (this.config.trace) {
      this.context = addTrace(
        this.context,
        'perceive',
        output.ir.inputHash,
        output.ir.outputHash,
        0
      );
    }

    return output;
  }

  private async runGround(perceiveOutput: PerceiveOutput): Promise<GroundOutput> {
    const input: GroundInput = {
      perceiveIR: perceiveOutput.ir,
      features: perceiveOutput.features,
    };

    const output = await ground(input, this.context);

    if (this.config.trace) {
      this.context = addTrace(
        this.context,
        'ground',
        output.ir.inputHash,
        output.ir.outputHash,
        0
      );
    }

    return output;
  }

  private async runConstrain(
    groundOutput: GroundOutput,
    request: CompilationRequest
  ): Promise<ConstrainOutput> {
    // Extract external constraints from request
    const externalConstraints = request.mode === 'constraint'
      ? request.constraints
      : undefined;

    const input: ConstrainInput = {
      groundIR: groundOutput.ir,
      symbols: groundOutput.symbols,
      externalConstraints,
    };

    const output = await constrain(input, this.context);

    if (this.config.trace) {
      this.context = addTrace(
        this.context,
        'constrain',
        output.ir.inputHash,
        output.ir.outputHash,
        0
      );
    }

    return output;
  }

  private async runVerify(
    constrainOutput: ConstrainOutput,
    stageHistory: StageHistoryEntry[]
  ): Promise<VerifyOutput> {
    const provenance: ProvenanceTrace = {
      stages: stageHistory.map(s => s.stage),
      inputHashes: stageHistory.map(s => s.inputHash),
      outputHashes: stageHistory.map(s => s.outputHash),
      taintVector: this.context.taintVector,
    };

    const input: VerifyInput = {
      constrainIR: constrainOutput.ir,
      satisfied: constrainOutput.satisfied,
      provenance,
    };

    const output = await verify(input, this.context);

    if (this.config.trace) {
      this.context = addTrace(
        this.context,
        'verify',
        output.ir.inputHash,
        output.ir.outputHash,
        0
      );
    }

    return output;
  }

  private async runEmit(
    verifyOutput: VerifyOutput,
    stageHistory: StageHistoryEntry[],
    request: CompilationRequest
  ): Promise<EmitOutput> {
    const input: EmitInput = {
      verifyIR: verifyOutput.ir,
      stageHistory,
      intentMetadata: this.extractIntentMetadata(request),
    };

    const output = await emit(input, this.context);

    if (this.config.trace) {
      this.context = addTrace(
        this.context,
        'emit',
        output.ir.inputHash,
        output.ir.outputHash,
        0
      );
    }

    return output;
  }

  // ===========================================================================
  // Helpers
  // ===========================================================================

  private extractIntentMetadata(request: CompilationRequest): EmitInput['intentMetadata'] {
    switch (request.mode) {
      case 'genesis':
        return {
          what: 'Create new holon',
          why: 'Genesis compilation',
        };

      case 'seed':
        return {
          what: 'Evolve holon',
          why: 'Seed mutation',
        };

      case 'atom':
        return {
          what: request.intent,
          why: 'Atomic operation',
        };

      case 'constraint':
        return {
          what: 'Apply constraints',
          why: 'Constraint refinement',
        };

      default:
        return {};
    }
  }

  private countCacheHits(): number {
    // Count from embedding cache
    return this.context.embeddingCache.size;
  }

  private buildHolon(
    emitOutput: EmitOutput,
    verifyOutput: VerifyOutput
  ): CompiledHolon {
    const artifact = emitOutput.artifact;

    return {
      holon: {
        id: emitOutput.ir.id,
        pentad: artifact.pentad as CompiledHolon['holon']['pentad'],
        trust: artifact.trust,
        timestamp: artifact.compiledAt,
      },
      ir: {
        steps: artifact.plan,
        optimized: true,
      },
      proof: {
        id: verifyOutput.proof.id,
        gates: {
          provenance: verifyOutput.proof.provenance.status,
          effects: verifyOutput.proof.effects.status,
          liveness: verifyOutput.proof.liveness.status,
          recovery: verifyOutput.proof.recovery.status,
          quality: verifyOutput.proof.quality.status,
          omega: verifyOutput.proof.omega.status,
        },
        derivation: verifyOutput.proof.derivation,
      },
      provenance: emitOutput.ir.provenance,
    };
  }

  // ===========================================================================
  // Context Management
  // ===========================================================================

  /**
   * Get current compilation context (for debugging)
   */
  getContext(): CompilationContext {
    return this.context;
  }

  /**
   * Reset compiler state
   */
  reset(): void {
    this.context = createContext({
      model: this.config.model,
      maxIterations: this.config.maxIterations,
      timeoutMs: this.config.timeout,
    });
  }

  /**
   * Update configuration
   */
  configure(config: Partial<CompilerConfig>): void {
    this.config = { ...this.config, ...config };
    this.reset();
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a new NeSyCompiler instance
 */
export function createCompiler(config?: CompilerConfig): NeSyCompiler {
  return new NeSyCompiler(config);
}

// =============================================================================
// Quick Compile Functions
// =============================================================================

/**
 * Quick compile from text input
 */
export async function compileText(
  text: string,
  config?: CompilerConfig
): Promise<CompilationResult> {
  const compiler = createCompiler(config);
  return compiler.compile({
    mode: 'atom',
    intent: text,
  });
}

/**
 * Quick compile with verification check
 */
export async function compileAndVerify(
  text: string,
  config?: CompilerConfig
): Promise<{ result: CompilationResult; verified: boolean }> {
  const result = await compileText(text, config);
  return {
    result,
    verified: result.stages.verify.passed,
  };
}
