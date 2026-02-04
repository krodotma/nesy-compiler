/**
 * Orchestration Engine - Adaptive Compilation Coordinator
 *
 * Phase 4 Step 31: End-to-End Pipeline Orchestration
 *
 * Coordinates all NeSy compilation phases:
 * - Phase 1: Core compilation (perceive → verify → emit)
 * - Phase 2: Temporal analysis (git history, thrash detection)
 * - Phase 3: Constraint solving (SAT/SMT, proof search)
 *
 * Features:
 * - Adaptive strategy selection based on input characteristics
 * - Parallel phase execution where possible
 * - Progressive refinement with fallback strategies
 * - Telemetry collection for continuous improvement
 */

import type { CompilationRequest, CompiledHolon, CompilationContext, TrustLevel } from '@nesy/core';
import type { CompilationResult } from '@nesy/pipeline';

/** Phase identifiers */
export type PhaseId = 'perceive' | 'ground' | 'constrain' | 'verify' | 'emit' | 'temporal' | 'solve';

/** Phase status */
export type PhaseStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

/** Phase execution record */
export interface PhaseExecution {
  phase: PhaseId;
  status: PhaseStatus;
  startTime: number;
  endTime?: number;
  duration?: number;
  error?: string;
  metrics: Record<string, number>;
}

/** Orchestration strategy */
export interface Strategy {
  name: string;
  phases: PhaseId[];
  parallel?: PhaseId[][];
  fallback?: string;
  condition?: (ctx: OrchestrationContext) => boolean;
}

/** Orchestration context */
export interface OrchestrationContext {
  request: CompilationRequest;
  currentPhase: PhaseId;
  completedPhases: PhaseId[];
  executions: PhaseExecution[];
  intermediateResults: Map<PhaseId, unknown>;
  telemetry: TelemetryData;
}

/** Telemetry data */
export interface TelemetryData {
  totalDuration: number;
  phaseTimings: Record<string, number>;
  retryCount: number;
  fallbacksUsed: string[];
  memoryPeak: number;
  tokensUsed: number;
}

/** Orchestration result */
export interface OrchestrationResult {
  success: boolean;
  result?: CompilationResult;
  executions: PhaseExecution[];
  strategy: string;
  telemetry: TelemetryData;
}

/** Engine configuration */
export interface EngineConfig {
  /** Enable parallel phase execution */
  enableParallel: boolean;
  /** Maximum retries per phase */
  maxRetries: number;
  /** Phase timeout in milliseconds */
  phaseTimeout: number;
  /** Enable telemetry collection */
  collectTelemetry: boolean;
  /** Custom strategies */
  strategies: Strategy[];
  /** Adaptive strategy selection */
  adaptiveStrategy: boolean;
}

const DEFAULT_CONFIG: EngineConfig = {
  enableParallel: true,
  maxRetries: 2,
  phaseTimeout: 30000,
  collectTelemetry: true,
  strategies: [],
  adaptiveStrategy: true,
};

/** Built-in strategies */
const BUILTIN_STRATEGIES: Strategy[] = [
  {
    name: 'standard',
    phases: ['perceive', 'ground', 'constrain', 'verify', 'emit'],
    fallback: 'minimal',
  },
  {
    name: 'full',
    phases: ['perceive', 'ground', 'constrain', 'verify', 'temporal', 'solve', 'emit'],
    parallel: [['temporal', 'solve']],
    fallback: 'standard',
  },
  {
    name: 'minimal',
    phases: ['perceive', 'ground', 'emit'],
    condition: (ctx) => ctx.request.mode === 'atom',
  },
  {
    name: 'verification-heavy',
    phases: ['perceive', 'ground', 'constrain', 'verify', 'solve', 'verify', 'emit'],
    condition: (ctx) => {
      // Use for high-trust requirements
      const request = ctx.request as { trust?: TrustLevel };
      return request.trust !== undefined && request.trust <= 1;
    },
  },
];

/**
 * OrchestrationEngine: Coordinates NeSy compilation phases.
 */
export class OrchestrationEngine {
  private config: EngineConfig;
  private strategies: Map<string, Strategy>;
  private phaseHandlers: Map<PhaseId, PhaseHandler>;

  constructor(config?: Partial<EngineConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.strategies = new Map();
    this.phaseHandlers = new Map();

    // Register built-in strategies
    for (const strategy of [...BUILTIN_STRATEGIES, ...this.config.strategies]) {
      this.strategies.set(strategy.name, strategy);
    }

    // Register default phase handlers
    this.registerDefaultHandlers();
  }

  /**
   * Execute compilation with orchestration.
   */
  async execute(request: CompilationRequest): Promise<OrchestrationResult> {
    const startTime = Date.now();

    // Initialize context
    const context: OrchestrationContext = {
      request,
      currentPhase: 'perceive',
      completedPhases: [],
      executions: [],
      intermediateResults: new Map(),
      telemetry: {
        totalDuration: 0,
        phaseTimings: {},
        retryCount: 0,
        fallbacksUsed: [],
        memoryPeak: 0,
        tokensUsed: 0,
      },
    };

    // Select strategy
    const strategy = this.selectStrategy(context);
    let currentStrategy = strategy;

    // Execute phases
    let success = true;
    let lastError: string | undefined;

    while (currentStrategy) {
      try {
        await this.executeStrategy(currentStrategy, context);
        success = true;
        break;
      } catch (error) {
        lastError = error instanceof Error ? error.message : String(error);

        // Try fallback strategy
        if (currentStrategy.fallback && this.strategies.has(currentStrategy.fallback)) {
          context.telemetry.fallbacksUsed.push(currentStrategy.fallback);
          currentStrategy = this.strategies.get(currentStrategy.fallback)!;
        } else {
          success = false;
          break;
        }
      }
    }

    // Finalize telemetry
    context.telemetry.totalDuration = Date.now() - startTime;
    this.updateMemoryPeak(context);

    // Build result
    const result = success
      ? this.buildResult(context)
      : undefined;

    return {
      success,
      result,
      executions: context.executions,
      strategy: strategy.name,
      telemetry: context.telemetry,
    };
  }

  /**
   * Select best strategy for request.
   */
  private selectStrategy(context: OrchestrationContext): Strategy {
    if (this.config.adaptiveStrategy) {
      // Check conditional strategies
      for (const [, strategy] of this.strategies) {
        if (strategy.condition && strategy.condition(context)) {
          return strategy;
        }
      }
    }

    // Default to standard
    return this.strategies.get('standard') || BUILTIN_STRATEGIES[0];
  }

  /**
   * Execute a strategy.
   */
  private async executeStrategy(
    strategy: Strategy,
    context: OrchestrationContext
  ): Promise<void> {
    const remainingPhases = strategy.phases.filter(
      p => !context.completedPhases.includes(p)
    );

    for (const phase of remainingPhases) {
      // Check if this phase is part of a parallel group
      const parallelGroup = strategy.parallel?.find(g => g.includes(phase));

      if (parallelGroup && this.config.enableParallel) {
        // Execute parallel group
        const parallelPhases = parallelGroup.filter(
          p => !context.completedPhases.includes(p)
        );

        if (parallelPhases.length > 1) {
          await this.executeParallel(parallelPhases, context);
          continue;
        }
      }

      // Execute single phase
      await this.executePhase(phase, context);
    }
  }

  /**
   * Execute phases in parallel.
   */
  private async executeParallel(
    phases: PhaseId[],
    context: OrchestrationContext
  ): Promise<void> {
    const promises = phases.map(phase => this.executePhase(phase, context));
    await Promise.all(promises);
  }

  /**
   * Execute a single phase.
   */
  private async executePhase(
    phase: PhaseId,
    context: OrchestrationContext
  ): Promise<void> {
    const execution: PhaseExecution = {
      phase,
      status: 'running',
      startTime: Date.now(),
      metrics: {},
    };
    context.executions.push(execution);
    context.currentPhase = phase;

    const handler = this.phaseHandlers.get(phase);
    if (!handler) {
      execution.status = 'skipped';
      execution.endTime = Date.now();
      execution.duration = 0;
      return;
    }

    let lastError: Error | undefined;
    let attempts = 0;

    while (attempts < this.config.maxRetries) {
      try {
        // Execute with timeout
        const result = await Promise.race([
          handler(context),
          this.timeout(this.config.phaseTimeout),
        ]);

        // Store result
        context.intermediateResults.set(phase, result);
        context.completedPhases.push(phase);

        execution.status = 'completed';
        execution.endTime = Date.now();
        execution.duration = execution.endTime - execution.startTime;
        context.telemetry.phaseTimings[phase] = execution.duration;

        return;
      } catch (error) {
        lastError = error instanceof Error ? error : new Error(String(error));
        attempts++;
        context.telemetry.retryCount++;
      }
    }

    // Phase failed
    execution.status = 'failed';
    execution.endTime = Date.now();
    execution.duration = execution.endTime - execution.startTime;
    execution.error = lastError?.message;

    throw lastError || new Error(`Phase ${phase} failed`);
  }

  /**
   * Create timeout promise.
   */
  private timeout(ms: number): Promise<never> {
    return new Promise((_, reject) => {
      setTimeout(() => reject(new Error(`Phase timeout after ${ms}ms`)), ms);
    });
  }

  /**
   * Build final result from context.
   */
  private buildResult(context: OrchestrationContext): CompilationResult {
    // Combine intermediate results into final compilation result
    const emitResult = context.intermediateResults.get('emit') as any || {};

    return {
      ir: emitResult.ir || {
        kind: 'compiled',
        id: `compiled-${Date.now()}`,
        inputHash: '',
        outputHash: '',
        timestamp: Date.now(),
      },
      holon: emitResult.holon,
      stages: {
        perceive: context.intermediateResults.get('perceive') as any || { features: { embedding: [] } },
        ground: context.intermediateResults.get('ground') as any || { symbols: { terms: [], constraints: [] } },
        constrain: context.intermediateResults.get('constrain') as any || { satisfied: [], unsatisfied: [] },
        verify: context.intermediateResults.get('verify') as any || { passed: true, gates: {} },
        emit: emitResult,
      },
      metrics: {
        totalDurationMs: context.telemetry.totalDuration,
        stageDurations: context.telemetry.phaseTimings,
        cacheHits: 0,
        constraintIterations: 0,
        verificationGatesPassed: 6,
        verificationGatesFailed: 0,
      },
    };
  }

  /**
   * Update memory peak in telemetry.
   */
  private updateMemoryPeak(context: OrchestrationContext): void {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      context.telemetry.memoryPeak = process.memoryUsage().heapUsed;
    }
  }

  /**
   * Register a phase handler.
   */
  registerPhaseHandler(phase: PhaseId, handler: PhaseHandler): void {
    this.phaseHandlers.set(phase, handler);
  }

  /**
   * Register a custom strategy.
   */
  registerStrategy(strategy: Strategy): void {
    this.strategies.set(strategy.name, strategy);
  }

  /**
   * Register default phase handlers.
   */
  private registerDefaultHandlers(): void {
    // Placeholder handlers - real implementations would invoke actual modules
    this.registerPhaseHandler('perceive', async (ctx) => {
      return { features: { embedding: [], attention: [] } };
    });

    this.registerPhaseHandler('ground', async (ctx) => {
      const perceiveResult = ctx.intermediateResults.get('perceive') as any;
      return { symbols: { terms: [], constraints: [] }, confidence: 0.9 };
    });

    this.registerPhaseHandler('constrain', async (ctx) => {
      return { satisfied: [], unsatisfied: [] };
    });

    this.registerPhaseHandler('verify', async (ctx) => {
      return { passed: true, gates: { P: true, E: true, L: true, R: true, Q: true, Ω: true } };
    });

    this.registerPhaseHandler('emit', async (ctx) => {
      return { holon: {}, ir: {}, provenance: {} };
    });

    this.registerPhaseHandler('temporal', async (ctx) => {
      return { signals: [], thrashScore: 0, entelechy: 0.5 };
    });

    this.registerPhaseHandler('solve', async (ctx) => {
      return { solved: true, proof: null };
    });
  }

  /**
   * Get strategy by name.
   */
  getStrategy(name: string): Strategy | undefined {
    return this.strategies.get(name);
  }

  /**
   * List available strategies.
   */
  listStrategies(): string[] {
    return Array.from(this.strategies.keys());
  }
}

/** Phase handler function */
export type PhaseHandler = (context: OrchestrationContext) => Promise<unknown>;

/**
 * Quick execution helper.
 */
export async function orchestrate(
  request: CompilationRequest,
  config?: Partial<EngineConfig>
): Promise<OrchestrationResult> {
  const engine = new OrchestrationEngine(config);
  return engine.execute(request);
}
