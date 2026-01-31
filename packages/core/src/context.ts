/**
 * Compilation context
 *
 * Carries configuration, caches, and state through the pipeline
 */

import type { TrustLevel, DiscretizationConfig } from './types';

export interface CompilationContext {
  // Model configuration
  model: string;
  temperature: number;

  // Trust tracking
  trustLevel: TrustLevel;
  taintVector: string[];

  // Discretization
  discretization: DiscretizationConfig;

  // Caching
  embeddingCache: Map<string, Float32Array>;

  // Logging
  trace: TraceEntry[];

  // Limits
  maxIterations: number;
  timeoutMs: number;
}

export interface TraceEntry {
  stage: string;
  input: string;   // Hash
  output: string;  // Hash
  durationMs: number;
  timestamp: number;
}

export function createContext(
  overrides: Partial<CompilationContext> = {}
): CompilationContext {
  return {
    model: 'claude-opus-4-5',
    temperature: 0.7,
    trustLevel: 3, // UNTRUSTED by default
    taintVector: [],
    discretization: {
      temperature: 1.0,
      annealing: 'cosine',
      threshold: 0.5,
    },
    embeddingCache: new Map(),
    trace: [],
    maxIterations: 100,
    timeoutMs: 60000,
    ...overrides,
  };
}

export function addTrace(
  context: CompilationContext,
  stage: string,
  input: string,
  output: string,
  durationMs: number
): CompilationContext {
  return {
    ...context,
    trace: [
      ...context.trace,
      { stage, input, output, durationMs, timestamp: Date.now() },
    ],
  };
}

export function elevateContext(
  context: CompilationContext,
  newLevel: TrustLevel
): CompilationContext {
  // Can only lower trust level number (increase trust)
  if (newLevel >= context.trustLevel) {
    return context;
  }

  return {
    ...context,
    trustLevel: newLevel,
    taintVector: [...context.taintVector, `elevated:${newLevel}`],
  };
}

export function taintContext(
  context: CompilationContext,
  reason: string
): CompilationContext {
  return {
    ...context,
    taintVector: [...context.taintVector, reason],
  };
}
