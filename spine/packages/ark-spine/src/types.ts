/**
 * @ark/spine/types - Core Type Definitions for Spine Coordination
 *
 * Types for module management, lifecycle control, and dependency resolution.
 *
 * @module
 */

import type { Ring } from '@ark/core';

/**
 * Module state in the lifecycle
 */
export type ModuleState =
  | 'unloaded'
  | 'loading'
  | 'loaded'
  | 'initializing'
  | 'ready'
  | 'starting'
  | 'running'
  | 'stopping'
  | 'stopped'
  | 'error'
  | 'disposed';

/**
 * Module health status
 */
export type ModuleHealth = 'unknown' | 'healthy' | 'degraded' | 'unhealthy';

/**
 * Dependency resolution status
 */
export type DependencyStatus = 'unresolved' | 'resolving' | 'resolved' | 'failed' | 'circular';

/**
 * Module capability type
 */
export type ModuleCapability =
  | 'event-emitter'
  | 'event-consumer'
  | 'storage'
  | 'computation'
  | 'coordination'
  | 'gateway'
  | 'monitoring';

/**
 * Module metadata
 */
export interface ModuleMetadata {
  /** Module name (e.g., '@ark/bus') */
  name: string;
  /** Semantic version */
  version: string;
  /** Human-readable description */
  description?: string;
  /** Author/maintainer */
  author?: string;
  /** Ring level required */
  ring: Ring;
  /** Module capabilities */
  capabilities: ModuleCapability[];
  /** Tags for categorization */
  tags: string[];
  /** Whether this is a core module */
  core?: boolean;
  /** Custom metadata */
  extra?: Record<string, unknown>;
}

/**
 * Module dependency specification
 */
export interface ModuleDependency {
  /** Dependency module name */
  name: string;
  /** Version range (semver) */
  versionRange?: string;
  /** Is this dependency optional? */
  optional?: boolean;
  /** Is this a peer dependency? */
  peer?: boolean;
}

/**
 * Module definition - static configuration
 */
export interface ModuleDefinition {
  /** Unique module ID */
  id: string;
  /** Module metadata */
  metadata: ModuleMetadata;
  /** Module dependencies */
  dependencies: ModuleDependency[];
  /** Module exports (capability names) */
  exports: string[];
  /** Entry point path (optional) */
  entryPoint?: string;
  /** Configuration schema (JSON Schema) */
  configSchema?: Record<string, unknown>;
}

/**
 * Module instance - runtime state
 */
export interface ModuleInstance {
  /** Module definition ID */
  definitionId: string;
  /** Unique instance ID */
  instanceId: string;
  /** Current state */
  state: ModuleState;
  /** Health status */
  health: ModuleHealth;
  /** Configuration used */
  config: Record<string, unknown>;
  /** Start timestamp */
  startedAt?: number;
  /** Stop timestamp */
  stoppedAt?: number;
  /** Last health check timestamp */
  lastHealthCheck?: number;
  /** Error message if in error state */
  error?: string;
  /** Runtime exports (actual module exports) */
  exports?: Record<string, unknown>;
}

/**
 * Dependency graph node
 */
export interface DependencyNode {
  /** Module ID */
  moduleId: string;
  /** Direct dependencies (module IDs) */
  dependencies: string[];
  /** Direct dependents (module IDs) */
  dependents: string[];
  /** Resolution status */
  status: DependencyStatus;
  /** Resolved order (lower = earlier) */
  order?: number;
}

/**
 * Lifecycle hook types
 */
export type LifecycleHook =
  | 'beforeLoad'
  | 'afterLoad'
  | 'beforeInit'
  | 'afterInit'
  | 'beforeStart'
  | 'afterStart'
  | 'beforeStop'
  | 'afterStop'
  | 'beforeDispose'
  | 'afterDispose'
  | 'onError'
  | 'onHealthCheck';

/**
 * Lifecycle hook handler
 */
export type LifecycleHandler = (
  instance: ModuleInstance,
  context: LifecycleContext
) => void | Promise<void>;

/**
 * Lifecycle context passed to hooks
 */
export interface LifecycleContext {
  /** Coordinator reference */
  coordinator: unknown; // SpineCoordinator - avoid circular import
  /** Module registry reference */
  registry: unknown; // ModuleRegistry
  /** Current timestamp */
  timestamp: number;
  /** Extra context data */
  extra?: Record<string, unknown>;
}

/**
 * Coordinator configuration
 */
export interface CoordinatorConfig {
  /** Coordinator ID */
  id?: string;
  /** Default ring level for modules */
  defaultRing?: Ring;
  /** Enable strict dependency checking */
  strictDependencies?: boolean;
  /** Health check interval (ms) */
  healthCheckInterval?: number;
  /** Maximum concurrent module operations */
  maxConcurrency?: number;
  /** Enable debug logging */
  debug?: boolean;
}

/**
 * Module registry configuration
 */
export interface RegistryConfig {
  /** Enable caching */
  enableCache?: boolean;
  /** Maximum cache size */
  maxCacheSize?: number;
  /** Cache TTL (ms) */
  cacheTtl?: number;
}

/**
 * Dependency resolver configuration
 */
export interface ResolverConfig {
  /** Maximum resolution depth */
  maxDepth?: number;
  /** Allow circular dependencies (with warning) */
  allowCircular?: boolean;
  /** Timeout for resolution (ms) */
  timeout?: number;
}

/**
 * Lifecycle manager configuration
 */
export interface LifecycleConfig {
  /** Default timeout for lifecycle operations (ms) */
  defaultTimeout?: number;
  /** Retry count for failed operations */
  retryCount?: number;
  /** Retry delay (ms) */
  retryDelay?: number;
  /** Enable parallel operations where safe */
  parallelOps?: boolean;
}

/**
 * Coordinator event types
 */
export type CoordinatorEventType =
  | 'module.registered'
  | 'module.unregistered'
  | 'module.loaded'
  | 'module.initialized'
  | 'module.started'
  | 'module.stopped'
  | 'module.disposed'
  | 'module.error'
  | 'module.health'
  | 'dependency.resolved'
  | 'dependency.failed'
  | 'dependency.circular'
  | 'lifecycle.hook'
  | 'coordinator.started'
  | 'coordinator.stopped';

/**
 * Coordinator event data
 */
export interface CoordinatorEvent<T = unknown> {
  /** Event type */
  type: CoordinatorEventType;
  /** Event timestamp */
  timestamp: number;
  /** Module ID (if applicable) */
  moduleId?: string;
  /** Instance ID (if applicable) */
  instanceId?: string;
  /** Event data */
  data: T;
}

/**
 * Coordinator statistics
 */
export interface CoordinatorStats {
  /** Total registered modules */
  totalModules: number;
  /** Running instances */
  runningInstances: number;
  /** Modules by state */
  byState: Record<ModuleState, number>;
  /** Modules by health */
  byHealth: Record<ModuleHealth, number>;
  /** Resolution statistics */
  resolution: {
    total: number;
    resolved: number;
    failed: number;
    circular: number;
  };
  /** Uptime (ms) */
  uptime: number;
}

/**
 * Module load options
 */
export interface LoadOptions {
  /** Configuration to apply */
  config?: Record<string, unknown>;
  /** Skip dependency resolution */
  skipDependencies?: boolean;
  /** Force reload even if already loaded */
  force?: boolean;
  /** Timeout (ms) */
  timeout?: number;
}

/**
 * Module start options
 */
export interface StartOptions {
  /** Configuration to apply */
  config?: Record<string, unknown>;
  /** Start dependencies first */
  startDependencies?: boolean;
  /** Timeout (ms) */
  timeout?: number;
}

/**
 * Module stop options
 */
export interface StopOptions {
  /** Stop dependents first */
  stopDependents?: boolean;
  /** Force stop (no graceful shutdown) */
  force?: boolean;
  /** Timeout (ms) */
  timeout?: number;
}

/**
 * Health check result
 */
export interface HealthCheckResult {
  /** Module instance ID */
  instanceId: string;
  /** Health status */
  health: ModuleHealth;
  /** Check timestamp */
  timestamp: number;
  /** Check duration (ms) */
  duration: number;
  /** Health details */
  details?: Record<string, unknown>;
  /** Error message if unhealthy */
  error?: string;
}

/**
 * Resolution result
 */
export interface ResolutionResult {
  /** Successfully resolved? */
  success: boolean;
  /** Resolved order (module IDs in load order) */
  order: string[];
  /** Failed dependencies */
  failed: string[];
  /** Circular dependencies detected */
  circular: string[][];
  /** Resolution time (ms) */
  duration: number;
}

/**
 * Module operation result
 */
export interface OperationResult<T = void> {
  /** Operation succeeded? */
  success: boolean;
  /** Result data */
  data?: T;
  /** Error message if failed */
  error?: string;
  /** Duration (ms) */
  duration: number;
}

/**
 * Type guard for ModuleDefinition
 */
export function isModuleDefinition(value: unknown): value is ModuleDefinition {
  return (
    typeof value === 'object' &&
    value !== null &&
    'id' in value &&
    'metadata' in value &&
    'dependencies' in value
  );
}

/**
 * Type guard for ModuleInstance
 */
export function isModuleInstance(value: unknown): value is ModuleInstance {
  return (
    typeof value === 'object' &&
    value !== null &&
    'definitionId' in value &&
    'instanceId' in value &&
    'state' in value
  );
}

/**
 * Create a default module metadata
 */
export function createDefaultMetadata(name: string): ModuleMetadata {
  return {
    name,
    version: '0.0.0',
    ring: 2 as Ring,
    capabilities: [],
    tags: [],
  };
}

/**
 * Create a default module instance
 */
export function createDefaultInstance(
  definitionId: string,
  instanceId?: string
): ModuleInstance {
  return {
    definitionId,
    instanceId: instanceId ?? crypto.randomUUID(),
    state: 'unloaded',
    health: 'unknown',
    config: {},
  };
}
