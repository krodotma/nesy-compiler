import type { CompilationRequest, CompiledHolon, TrustLevel } from '@nesy/core';
import type { CompilationResult, CompilerConfig } from '@nesy/pipeline';

export interface HolonCompilerConfig extends CompilerConfig {
  autoRetry?: boolean;
  maxRetries?: number;
}

export interface BatchOptions {
  concurrency?: number;
  stopOnError?: boolean;
  progressCallback?: (done: number, total: number) => void;
}

export interface AnalysisReport {
  totalCompilations: number;
  passed: number;
  failed: number;
  averageDurationMs: number;
  gatePassRates: Record<string, number>;
  trustDistribution: Record<string, number>;
}
