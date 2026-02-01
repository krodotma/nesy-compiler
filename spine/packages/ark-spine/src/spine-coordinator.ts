/**
 * @ark/spine/spine-coordinator - Central Coordinator for @ark Packages
 *
 * Orchestrates module loading, dependency resolution, and lifecycle management
 * for the Neo Pluribus @ark package ecosystem.
 *
 * @module
 */

import type {
  ModuleDefinition,
  ModuleInstance,
  ModuleState,
  ModuleHealth,
  CoordinatorConfig,
  CoordinatorEvent,
  CoordinatorEventType,
  CoordinatorStats,
  LoadOptions,
  StartOptions,
  StopOptions,
  OperationResult,
  ResolutionResult,
  HealthCheckResult,
  LifecycleContext,
  LifecycleHandler,
  LifecycleHook,
} from './types.js';
import { ModuleRegistry, createModuleRegistry } from './module-registry.js';
import {
  DependencyResolver,
  createDependencyResolver,
} from './dependency-resolver.js';
import {
  LifecycleManager,
  createLifecycleManager,
} from './lifecycle-manager.js';
import { Ring } from '@ark/core';

/**
 * Event handler type
 */
export type EventHandler<T = unknown> = (event: CoordinatorEvent<T>) => void;

/**
 * Spine Coordinator - Central orchestrator for @ark packages
 */
export class SpineCoordinator {
  private id: string;
  private registry: ModuleRegistry;
  private resolver: DependencyResolver;
  private lifecycle: LifecycleManager;
  private config: Required<CoordinatorConfig>;
  private startTime: number;
  private eventHandlers: Map<string, Set<EventHandler>> = new Map();
  private healthCheckInterval?: ReturnType<typeof setInterval>;

  constructor(config: CoordinatorConfig = {}) {
    this.id = config.id ?? crypto.randomUUID();
    this.startTime = Date.now();

    this.config = {
      id: this.id,
      defaultRing: config.defaultRing ?? (2 as Ring),
      strictDependencies: config.strictDependencies ?? true,
      healthCheckInterval: config.healthCheckInterval ?? 30000, // 30 seconds
      maxConcurrency: config.maxConcurrency ?? 10,
      debug: config.debug ?? false,
    };

    // Initialize components
    this.registry = createModuleRegistry();
    this.resolver = createDependencyResolver({
      allowCircular: !this.config.strictDependencies,
    });
    this.lifecycle = createLifecycleManager();

    // Set up lifecycle context provider
    this.lifecycle.setContextProvider(() => this.createContext());
  }

  // ==================== Module Registration ====================

  /**
   * Register a module definition
   */
  registerModule(definition: ModuleDefinition): boolean {
    const success = this.registry.registerDefinition(definition);

    if (success) {
      // Add to dependency graph
      this.resolver.addModule(definition);

      // Emit event
      this.emitEvent('module.registered', {
        moduleId: definition.id,
        metadata: definition.metadata,
      });
    }

    return success;
  }

  /**
   * Unregister a module definition
   */
  unregisterModule(moduleId: string): boolean {
    // Check for running instances
    const instances = this.registry.getInstancesByDefinition(moduleId);
    const running = instances.filter((i) => i.state === 'running');

    if (running.length > 0) {
      throw new Error(
        `Cannot unregister module ${moduleId}: ${running.length} running instances`
      );
    }

    // Remove from dependency graph
    this.resolver.removeModule(moduleId);

    // Remove from registry
    const success = this.registry.unregisterDefinition(moduleId);

    if (success) {
      this.emitEvent('module.unregistered', { moduleId });
    }

    return success;
  }

  /**
   * Get a module definition
   */
  getModule(moduleId: string): ModuleDefinition | undefined {
    return this.registry.getDefinition(moduleId);
  }

  /**
   * Check if a module is registered
   */
  hasModule(moduleId: string): boolean {
    return this.registry.hasDefinition(moduleId);
  }

  /**
   * Get all registered modules
   */
  getAllModules(): ModuleDefinition[] {
    return this.registry.getAllDefinitions();
  }

  // ==================== Instance Management ====================

  /**
   * Create a module instance
   */
  createInstance(
    moduleId: string,
    config?: Record<string, unknown>
  ): ModuleInstance {
    return this.registry.createInstance(moduleId, config);
  }

  /**
   * Get a module instance
   */
  getInstance(instanceId: string): ModuleInstance | undefined {
    return this.registry.getInstance(instanceId);
  }

  /**
   * Get all instances
   */
  getAllInstances(): ModuleInstance[] {
    return this.registry.getAllInstances();
  }

  /**
   * Get running instances
   */
  getRunningInstances(): ModuleInstance[] {
    return this.registry.getRunningInstances();
  }

  // ==================== Dependency Resolution ====================

  /**
   * Resolve dependencies for all registered modules
   */
  resolveDependencies(): ResolutionResult {
    const definitions = this.registry.getAllDefinitions();
    this.resolver.buildGraph(definitions);
    const result = this.resolver.resolve();

    if (!result.success) {
      for (const failed of result.failed) {
        this.emitEvent('dependency.failed', { moduleId: failed });
      }

      for (const cycle of result.circular) {
        this.emitEvent('dependency.circular', { cycle });
      }
    } else {
      this.emitEvent('dependency.resolved', {
        order: result.order,
        count: result.order.length,
      });
    }

    return result;
  }

  /**
   * Resolve dependencies for a specific module
   */
  resolveDependenciesFor(moduleId: string): ResolutionResult {
    return this.resolver.resolveFor(moduleId);
  }

  /**
   * Get module dependencies
   */
  getDependencies(moduleId: string): string[] {
    return this.resolver.getDependencies(moduleId);
  }

  /**
   * Get module dependents
   */
  getDependents(moduleId: string): string[] {
    return this.resolver.getDependents(moduleId);
  }

  /**
   * Get transitive dependencies
   */
  getTransitiveDependencies(moduleId: string): string[] {
    return this.resolver.getTransitiveDependencies(moduleId);
  }

  // ==================== Lifecycle Operations ====================

  /**
   * Load a module (creates instance and loads it)
   */
  async loadModule(
    moduleId: string,
    options?: LoadOptions
  ): Promise<OperationResult<ModuleInstance>> {
    const startTime = Date.now();

    // Check if module exists
    const definition = this.registry.getDefinition(moduleId);
    if (!definition) {
      return {
        success: false,
        error: `Module not found: ${moduleId}`,
        duration: Date.now() - startTime,
      };
    }

    // Resolve dependencies if needed
    if (!options?.skipDependencies) {
      const resolution = this.resolver.resolveFor(moduleId);
      if (!resolution.success) {
        return {
          success: false,
          error: `Dependency resolution failed: ${resolution.failed.join(', ')}`,
          duration: Date.now() - startTime,
        };
      }

      // Load dependencies first
      for (const depId of resolution.order) {
        if (depId === moduleId) continue;

        const depInstances = this.registry.getInstancesByDefinition(depId);
        const loadedDep = depInstances.find(
          (i) => i.state === 'loaded' || i.state === 'ready' || i.state === 'running'
        );

        if (!loadedDep) {
          const depResult = await this.loadModule(depId, {
            ...options,
            skipDependencies: true,
          });
          if (!depResult.success) {
            return {
              success: false,
              error: `Failed to load dependency ${depId}: ${depResult.error}`,
              duration: Date.now() - startTime,
            };
          }
        }
      }
    }

    // Create or get instance
    let instance: ModuleInstance;
    const existingInstances = this.registry.getInstancesByDefinition(moduleId);

    if (existingInstances.length > 0 && !options?.force) {
      instance = existingInstances[0];
      if (
        instance.state === 'loaded' ||
        instance.state === 'ready' ||
        instance.state === 'running'
      ) {
        return {
          success: true,
          data: instance,
          duration: Date.now() - startTime,
        };
      }
    } else {
      instance = this.registry.createInstance(moduleId, options?.config);
    }

    // Perform load
    const loadResult = await this.lifecycle.load(
      instance,
      async () => {
        // Simulated module loading
        if (this.config.debug) {
          console.log(`[Spine] Loading module: ${moduleId}`);
        }
      },
      (state) => this.registry.updateInstanceState(instance.instanceId, state),
      { timeout: options?.timeout }
    );

    if (loadResult.success) {
      this.emitEvent('module.loaded', {
        moduleId,
        instanceId: instance.instanceId,
      });
    } else {
      this.emitEvent('module.error', {
        moduleId,
        instanceId: instance.instanceId,
        error: loadResult.error,
      });
    }

    return {
      success: loadResult.success,
      data: this.registry.getInstance(instance.instanceId),
      error: loadResult.error,
      duration: Date.now() - startTime,
    };
  }

  /**
   * Initialize a module
   */
  async initializeModule(instanceId: string): Promise<OperationResult> {
    const instance = this.registry.getInstance(instanceId);
    if (!instance) {
      return {
        success: false,
        error: `Instance not found: ${instanceId}`,
        duration: 0,
      };
    }

    const result = await this.lifecycle.initialize(
      instance,
      async () => {
        if (this.config.debug) {
          console.log(`[Spine] Initializing module: ${instance.definitionId}`);
        }
      },
      (state) => this.registry.updateInstanceState(instanceId, state)
    );

    if (result.success) {
      this.emitEvent('module.initialized', {
        moduleId: instance.definitionId,
        instanceId,
      });
    }

    return result;
  }

  /**
   * Start a module
   */
  async startModule(
    instanceId: string,
    options?: StartOptions
  ): Promise<OperationResult> {
    const instance = this.registry.getInstance(instanceId);
    if (!instance) {
      return {
        success: false,
        error: `Instance not found: ${instanceId}`,
        duration: 0,
      };
    }

    // Start dependencies first if requested
    if (options?.startDependencies) {
      const deps = this.resolver.getDependencies(instance.definitionId);

      for (const depId of deps) {
        const depInstances = this.registry.getInstancesByDefinition(depId);
        const runningDep = depInstances.find((i) => i.state === 'running');

        if (!runningDep) {
          const readyDep = depInstances.find(
            (i) => i.state === 'ready' || i.state === 'stopped'
          );
          if (readyDep) {
            await this.startModule(readyDep.instanceId, { startDependencies: true });
          }
        }
      }
    }

    const result = await this.lifecycle.start(
      instance,
      async () => {
        if (this.config.debug) {
          console.log(`[Spine] Starting module: ${instance.definitionId}`);
        }
      },
      (state) => this.registry.updateInstanceState(instanceId, state),
      { timeout: options?.timeout }
    );

    if (result.success) {
      this.registry.updateInstanceHealth(instanceId, 'healthy');
      this.emitEvent('module.started', {
        moduleId: instance.definitionId,
        instanceId,
      });
    }

    return result;
  }

  /**
   * Stop a module
   */
  async stopModule(
    instanceId: string,
    options?: StopOptions
  ): Promise<OperationResult> {
    const instance = this.registry.getInstance(instanceId);
    if (!instance) {
      return {
        success: false,
        error: `Instance not found: ${instanceId}`,
        duration: 0,
      };
    }

    // Stop dependents first if requested
    if (options?.stopDependents) {
      const dependents = this.resolver.getDependents(instance.definitionId);

      for (const depId of dependents) {
        const depInstances = this.registry.getInstancesByDefinition(depId);
        const runningDep = depInstances.find((i) => i.state === 'running');

        if (runningDep) {
          await this.stopModule(runningDep.instanceId, { stopDependents: true });
        }
      }
    }

    const result = await this.lifecycle.stop(
      instance,
      async () => {
        if (this.config.debug) {
          console.log(`[Spine] Stopping module: ${instance.definitionId}`);
        }
      },
      (state) => this.registry.updateInstanceState(instanceId, state),
      { timeout: options?.timeout }
    );

    if (result.success) {
      this.emitEvent('module.stopped', {
        moduleId: instance.definitionId,
        instanceId,
      });
    }

    return result;
  }

  /**
   * Dispose a module
   */
  async disposeModule(instanceId: string): Promise<OperationResult> {
    const instance = this.registry.getInstance(instanceId);
    if (!instance) {
      return {
        success: false,
        error: `Instance not found: ${instanceId}`,
        duration: 0,
      };
    }

    // Stop first if running
    if (instance.state === 'running') {
      await this.stopModule(instanceId);
    }

    const result = await this.lifecycle.dispose(
      instance,
      async () => {
        if (this.config.debug) {
          console.log(`[Spine] Disposing module: ${instance.definitionId}`);
        }
      },
      (state) => this.registry.updateInstanceState(instanceId, state)
    );

    if (result.success) {
      this.registry.deleteInstance(instanceId);
      this.emitEvent('module.disposed', {
        moduleId: instance.definitionId,
        instanceId,
      });
    }

    return result;
  }

  /**
   * Load and start a module in one operation
   */
  async loadAndStart(
    moduleId: string,
    options?: LoadOptions & StartOptions
  ): Promise<OperationResult<ModuleInstance>> {
    const startTime = Date.now();

    // Load
    const loadResult = await this.loadModule(moduleId, options);
    if (!loadResult.success || !loadResult.data) {
      return {
        success: false,
        error: loadResult.error ?? 'Load failed',
        duration: Date.now() - startTime,
      };
    }

    // Initialize
    const initResult = await this.initializeModule(loadResult.data.instanceId);
    if (!initResult.success) {
      return {
        success: false,
        error: initResult.error ?? 'Initialize failed',
        duration: Date.now() - startTime,
      };
    }

    // Start
    const startResult = await this.startModule(loadResult.data.instanceId, options);
    if (!startResult.success) {
      return {
        success: false,
        error: startResult.error ?? 'Start failed',
        duration: Date.now() - startTime,
      };
    }

    return {
      success: true,
      data: this.registry.getInstance(loadResult.data.instanceId),
      duration: Date.now() - startTime,
    };
  }

  // ==================== Health Checks ====================

  /**
   * Check health of a module instance
   */
  async checkHealth(instanceId: string): Promise<HealthCheckResult> {
    const startTime = Date.now();
    const instance = this.registry.getInstance(instanceId);

    if (!instance) {
      return {
        instanceId,
        health: 'unknown',
        timestamp: Date.now(),
        duration: 0,
        error: 'Instance not found',
      };
    }

    const result = await this.lifecycle.healthCheck(
      instance,
      async () => {
        // Default health check: module is healthy if running
        return instance.state === 'running';
      }
    );

    const health: ModuleHealth = result.data ? 'healthy' : 'unhealthy';

    this.registry.updateInstanceHealth(instanceId, health, result.error);

    const checkResult: HealthCheckResult = {
      instanceId,
      health,
      timestamp: Date.now(),
      duration: Date.now() - startTime,
      error: result.error,
    };

    this.emitEvent('module.health', checkResult);

    return checkResult;
  }

  /**
   * Check health of all running instances
   */
  async checkAllHealth(): Promise<HealthCheckResult[]> {
    const results: HealthCheckResult[] = [];
    const running = this.registry.getRunningInstances();

    for (const instance of running) {
      const result = await this.checkHealth(instance.instanceId);
      results.push(result);
    }

    return results;
  }

  /**
   * Start periodic health checks
   */
  startHealthChecks(): void {
    if (this.healthCheckInterval) {
      return;
    }

    this.healthCheckInterval = setInterval(async () => {
      await this.checkAllHealth();
    }, this.config.healthCheckInterval);
  }

  /**
   * Stop periodic health checks
   */
  stopHealthChecks(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = undefined;
    }
  }

  // ==================== Lifecycle Hooks ====================

  /**
   * Register a lifecycle hook
   */
  onLifecycle(hook: LifecycleHook, handler: LifecycleHandler): () => void {
    return this.lifecycle.onHook(hook, handler);
  }

  // ==================== Events ====================

  /**
   * Subscribe to coordinator events
   */
  on<T = unknown>(
    eventType: CoordinatorEventType | '*',
    handler: EventHandler<T>
  ): () => void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }

    this.eventHandlers.get(eventType)!.add(handler as EventHandler);

    return () => {
      const handlers = this.eventHandlers.get(eventType);
      if (handlers) {
        handlers.delete(handler as EventHandler);
      }
    };
  }

  /**
   * Emit a coordinator event
   */
  private emitEvent<T>(type: CoordinatorEventType, data: T): void {
    const event: CoordinatorEvent<T> = {
      type,
      timestamp: Date.now(),
      data,
    };

    // Add moduleId/instanceId to event if present in data
    if (data && typeof data === 'object') {
      if ('moduleId' in data) {
        event.moduleId = data.moduleId as string;
      }
      if ('instanceId' in data) {
        event.instanceId = data.instanceId as string;
      }
    }

    // Notify specific handlers
    const handlers = this.eventHandlers.get(type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          handler(event);
        } catch (error) {
          if (this.config.debug) {
            console.error(`[Spine] Event handler error:`, error);
          }
        }
      }
    }

    // Notify wildcard handlers
    const wildcardHandlers = this.eventHandlers.get('*');
    if (wildcardHandlers) {
      for (const handler of wildcardHandlers) {
        try {
          handler(event);
        } catch (error) {
          if (this.config.debug) {
            console.error(`[Spine] Wildcard handler error:`, error);
          }
        }
      }
    }
  }

  // ==================== Statistics ====================

  /**
   * Get coordinator statistics
   */
  getStats(): CoordinatorStats {
    const registryStats = this.registry.getStats();
    const resolverStats = this.resolver.getStats();

    return {
      totalModules: registryStats.definitions,
      runningInstances: registryStats.byState.running,
      byState: registryStats.byState,
      byHealth: registryStats.byHealth,
      resolution: {
        total: resolverStats.nodeCount,
        resolved: resolverStats.resolvedCount,
        failed: resolverStats.failedCount,
        circular: resolverStats.circularCount,
      },
      uptime: Date.now() - this.startTime,
    };
  }

  // ==================== Coordinator Lifecycle ====================

  /**
   * Start the coordinator
   */
  async start(): Promise<void> {
    this.startHealthChecks();
    this.emitEvent('coordinator.started', { id: this.id });
  }

  /**
   * Stop the coordinator and all modules
   */
  async stop(): Promise<void> {
    this.stopHealthChecks();

    // Stop all running modules in reverse dependency order
    const running = this.registry.getRunningInstances();

    for (const instance of running) {
      await this.stopModule(instance.instanceId, { stopDependents: false });
    }

    this.emitEvent('coordinator.stopped', { id: this.id });
  }

  /**
   * Shutdown coordinator and dispose all modules
   */
  async shutdown(): Promise<void> {
    await this.stop();

    // Dispose all instances
    const instances = this.registry.getAllInstances();

    for (const instance of instances) {
      await this.disposeModule(instance.instanceId);
    }

    // Clear event handlers
    this.eventHandlers.clear();
  }

  // ==================== Internal Helpers ====================

  /**
   * Create lifecycle context
   */
  private createContext(): LifecycleContext {
    return {
      coordinator: this,
      registry: this.registry,
      timestamp: Date.now(),
    };
  }

  /**
   * Get coordinator ID
   */
  getId(): string {
    return this.id;
  }

  /**
   * Get configuration
   */
  getConfig(): Required<CoordinatorConfig> {
    return { ...this.config };
  }

  /**
   * Get the module registry
   */
  getRegistry(): ModuleRegistry {
    return this.registry;
  }

  /**
   * Get the dependency resolver
   */
  getResolver(): DependencyResolver {
    return this.resolver;
  }

  /**
   * Get the lifecycle manager
   */
  getLifecycleManager(): LifecycleManager {
    return this.lifecycle;
  }
}

/**
 * Create a new spine coordinator
 */
export function createSpineCoordinator(
  config?: CoordinatorConfig
): SpineCoordinator {
  return new SpineCoordinator(config);
}
