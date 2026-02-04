/**
 * @nesy/integration - High-level Integration Layer
 *
 * Phase 4 of NeSy Evolution: Integration & Orchestration
 *
 * Components:
 * - HolonCompiler: High-level compilation interface
 * - OrchestrationEngine: Adaptive phase coordination
 * - BusIntegration: Pluribus event bus connection
 * - CompilationAnalytics: Performance tracking
 */

// Core integration
export { HolonCompiler } from './holon-compiler.js';
export { PipelineRunner } from './pipeline-runner.js';
export { ResultAnalyzer } from './result-analyzer.js';
export { BatchCompiler } from './batch.js';
export type { HolonCompilerConfig, BatchOptions, AnalysisReport } from './types.js';

// Phase 1.5: Sensor Fusion
export * from './linter-bridge.js';
export * from './antipattern-mapper.js';
export * from './type-graph.js';
export * from './signal-booster.js';

// Phase 4: Integration & Orchestration (Steps 31-33)
export {
  OrchestrationEngine,
  orchestrate,
  type PhaseId,
  type PhaseStatus,
  type PhaseExecution,
  type Strategy,
  type OrchestrationContext,
  type TelemetryData,
  type OrchestrationResult,
  type EngineConfig,
  type PhaseHandler,
} from './orchestration-engine.js';

export {
  BusIntegration,
  createBusIntegration,
  type NeSyEventType,
  type BusEvent,
  type EventMetadata,
  type CompileRequestedEvent,
  type CompileStartedEvent,
  type PhaseCompletedEvent,
  type CompileCompletedEvent,
  type CompileFailedEvent,
  type HolonActualizedEvent,
  type BusConnection,
  type EventHandler,
  type Subscription,
  type BusConfig,
} from './bus-integration.js';

export {
  CompilationAnalytics,
  createAnalytics,
  type DataPoint,
  type Statistics,
  type PhaseAnalytics,
  type AnalyticsSummary,
  type Bottleneck,
  type Recommendation,
  type AnalyticsConfig,
} from './compilation-analytics.js';
