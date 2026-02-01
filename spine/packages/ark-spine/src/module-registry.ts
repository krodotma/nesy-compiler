/**
 * @ark/spine/module-registry - Module Registry for Managing Loaded Modules
 *
 * Provides registration, lookup, and caching of module definitions and instances.
 *
 * @module
 */

import type {
  ModuleDefinition,
  ModuleInstance,
  ModuleMetadata,
  ModuleDependency,
  ModuleState,
  ModuleHealth,
  ModuleCapability,
  RegistryConfig,
  HealthCheckResult,
} from './types.js';
import { createDefaultMetadata, createDefaultInstance } from './types.js';

/**
 * Cache entry with TTL
 */
interface CacheEntry<T> {
  value: T;
  expiresAt: number;
}

/**
 * Module Registry - Manages module definitions and instances
 */
export class ModuleRegistry {
  private definitions: Map<string, ModuleDefinition> = new Map();
  private instances: Map<string, ModuleInstance> = new Map();
  private cache: Map<string, CacheEntry<unknown>> = new Map();
  private config: Required<RegistryConfig>;

  constructor(config: RegistryConfig = {}) {
    this.config = {
      enableCache: config.enableCache ?? true,
      maxCacheSize: config.maxCacheSize ?? 1000,
      cacheTtl: config.cacheTtl ?? 60000, // 1 minute default
    };
  }

  // ==================== Definition Management ====================

  /**
   * Register a module definition
   */
  registerDefinition(definition: ModuleDefinition): boolean {
    if (this.definitions.has(definition.id)) {
      return false;
    }

    this.definitions.set(definition.id, { ...definition });
    this.invalidateCache(`def:${definition.id}`);
    return true;
  }

  /**
   * Unregister a module definition
   */
  unregisterDefinition(id: string): boolean {
    if (!this.definitions.has(id)) {
      return false;
    }

    // Check for running instances
    const runningInstances = this.getInstancesByDefinition(id).filter(
      (i) => i.state === 'running'
    );

    if (runningInstances.length > 0) {
      throw new Error(
        `Cannot unregister module ${id}: ${runningInstances.length} running instances`
      );
    }

    this.definitions.delete(id);
    this.invalidateCache(`def:${id}`);
    return true;
  }

  /**
   * Get a module definition by ID
   */
  getDefinition(id: string): ModuleDefinition | undefined {
    return this.definitions.get(id);
  }

  /**
   * Check if a definition exists
   */
  hasDefinition(id: string): boolean {
    return this.definitions.has(id);
  }

  /**
   * Get all definitions
   */
  getAllDefinitions(): ModuleDefinition[] {
    return Array.from(this.definitions.values());
  }

  /**
   * Get definitions by capability
   */
  getDefinitionsByCapability(capability: ModuleCapability): ModuleDefinition[] {
    return Array.from(this.definitions.values()).filter((def) =>
      def.metadata.capabilities.includes(capability)
    );
  }

  /**
   * Get definitions by tag
   */
  getDefinitionsByTag(tag: string): ModuleDefinition[] {
    return Array.from(this.definitions.values()).filter((def) =>
      def.metadata.tags.includes(tag)
    );
  }

  /**
   * Find definitions matching a predicate
   */
  findDefinitions(
    predicate: (def: ModuleDefinition) => boolean
  ): ModuleDefinition[] {
    return Array.from(this.definitions.values()).filter(predicate);
  }

  /**
   * Create a module definition from minimal input
   */
  createDefinition(
    id: string,
    metadata: Partial<ModuleMetadata> & { name: string },
    options?: {
      dependencies?: ModuleDependency[];
      exports?: string[];
      entryPoint?: string;
      configSchema?: Record<string, unknown>;
    }
  ): ModuleDefinition {
    return {
      id,
      metadata: {
        ...createDefaultMetadata(metadata.name),
        ...metadata,
      },
      dependencies: options?.dependencies ?? [],
      exports: options?.exports ?? [],
      entryPoint: options?.entryPoint,
      configSchema: options?.configSchema,
    };
  }

  // ==================== Instance Management ====================

  /**
   * Create a module instance
   */
  createInstance(
    definitionId: string,
    config?: Record<string, unknown>
  ): ModuleInstance {
    const definition = this.definitions.get(definitionId);
    if (!definition) {
      throw new Error(`Module definition not found: ${definitionId}`);
    }

    const instance = createDefaultInstance(definitionId);
    instance.config = config ?? {};

    this.instances.set(instance.instanceId, instance);
    return instance;
  }

  /**
   * Get a module instance by ID
   */
  getInstance(instanceId: string): ModuleInstance | undefined {
    return this.instances.get(instanceId);
  }

  /**
   * Check if an instance exists
   */
  hasInstance(instanceId: string): boolean {
    return this.instances.has(instanceId);
  }

  /**
   * Update instance state
   */
  updateInstanceState(instanceId: string, state: ModuleState): boolean {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return false;
    }

    instance.state = state;

    if (state === 'running' && !instance.startedAt) {
      instance.startedAt = Date.now();
    } else if (state === 'stopped' || state === 'disposed') {
      instance.stoppedAt = Date.now();
    }

    return true;
  }

  /**
   * Update instance health
   */
  updateInstanceHealth(
    instanceId: string,
    health: ModuleHealth,
    error?: string
  ): boolean {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return false;
    }

    instance.health = health;
    instance.lastHealthCheck = Date.now();
    if (error !== undefined) {
      instance.error = error;
    }

    return true;
  }

  /**
   * Update instance config
   */
  updateInstanceConfig(
    instanceId: string,
    config: Record<string, unknown>
  ): boolean {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return false;
    }

    instance.config = { ...instance.config, ...config };
    return true;
  }

  /**
   * Set instance error
   */
  setInstanceError(instanceId: string, error: string): boolean {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return false;
    }

    instance.state = 'error';
    instance.health = 'unhealthy';
    instance.error = error;
    return true;
  }

  /**
   * Clear instance error
   */
  clearInstanceError(instanceId: string): boolean {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return false;
    }

    instance.error = undefined;
    return true;
  }

  /**
   * Set instance exports
   */
  setInstanceExports(
    instanceId: string,
    exports: Record<string, unknown>
  ): boolean {
    const instance = this.instances.get(instanceId);
    if (!instance) {
      return false;
    }

    instance.exports = exports;
    return true;
  }

  /**
   * Delete a module instance
   */
  deleteInstance(instanceId: string): boolean {
    return this.instances.delete(instanceId);
  }

  /**
   * Get all instances
   */
  getAllInstances(): ModuleInstance[] {
    return Array.from(this.instances.values());
  }

  /**
   * Get instances by definition ID
   */
  getInstancesByDefinition(definitionId: string): ModuleInstance[] {
    return Array.from(this.instances.values()).filter(
      (i) => i.definitionId === definitionId
    );
  }

  /**
   * Get instances by state
   */
  getInstancesByState(state: ModuleState): ModuleInstance[] {
    return Array.from(this.instances.values()).filter((i) => i.state === state);
  }

  /**
   * Get instances by health
   */
  getInstancesByHealth(health: ModuleHealth): ModuleInstance[] {
    return Array.from(this.instances.values()).filter(
      (i) => i.health === health
    );
  }

  /**
   * Get running instances
   */
  getRunningInstances(): ModuleInstance[] {
    return this.getInstancesByState('running');
  }

  /**
   * Get unhealthy instances
   */
  getUnhealthyInstances(): ModuleInstance[] {
    return Array.from(this.instances.values()).filter(
      (i) => i.health === 'unhealthy' || i.health === 'degraded'
    );
  }

  /**
   * Find instances matching a predicate
   */
  findInstances(
    predicate: (instance: ModuleInstance) => boolean
  ): ModuleInstance[] {
    return Array.from(this.instances.values()).filter(predicate);
  }

  /**
   * Record health check result
   */
  recordHealthCheck(result: HealthCheckResult): boolean {
    const instance = this.instances.get(result.instanceId);
    if (!instance) {
      return false;
    }

    instance.health = result.health;
    instance.lastHealthCheck = result.timestamp;
    if (result.error) {
      instance.error = result.error;
    }

    return true;
  }

  // ==================== Cache Management ====================

  /**
   * Get cached value
   */
  getCached<T>(key: string): T | undefined {
    if (!this.config.enableCache) {
      return undefined;
    }

    const entry = this.cache.get(key);
    if (!entry) {
      return undefined;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return undefined;
    }

    return entry.value as T;
  }

  /**
   * Set cached value
   */
  setCached<T>(key: string, value: T, ttl?: number): void {
    if (!this.config.enableCache) {
      return;
    }

    // Enforce max cache size
    if (this.cache.size >= this.config.maxCacheSize) {
      this.evictOldestCache();
    }

    this.cache.set(key, {
      value,
      expiresAt: Date.now() + (ttl ?? this.config.cacheTtl),
    });
  }

  /**
   * Invalidate cache entry
   */
  invalidateCache(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clear all cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Evict oldest cache entries
   */
  private evictOldestCache(): void {
    const entries = Array.from(this.cache.entries());
    entries.sort((a, b) => a[1].expiresAt - b[1].expiresAt);

    // Remove oldest 10%
    const toRemove = Math.ceil(entries.length * 0.1);
    for (let i = 0; i < toRemove; i++) {
      this.cache.delete(entries[i][0]);
    }
  }

  // ==================== Statistics ====================

  /**
   * Get registry statistics
   */
  getStats(): {
    definitions: number;
    instances: number;
    byState: Record<ModuleState, number>;
    byHealth: Record<ModuleHealth, number>;
    cacheSize: number;
    cacheHitRate?: number;
  } {
    const byState: Record<ModuleState, number> = {
      unloaded: 0,
      loading: 0,
      loaded: 0,
      initializing: 0,
      ready: 0,
      starting: 0,
      running: 0,
      stopping: 0,
      stopped: 0,
      error: 0,
      disposed: 0,
    };

    const byHealth: Record<ModuleHealth, number> = {
      unknown: 0,
      healthy: 0,
      degraded: 0,
      unhealthy: 0,
    };

    for (const instance of this.instances.values()) {
      byState[instance.state]++;
      byHealth[instance.health]++;
    }

    return {
      definitions: this.definitions.size,
      instances: this.instances.size,
      byState,
      byHealth,
      cacheSize: this.cache.size,
    };
  }

  /**
   * Clear all data (for testing)
   */
  clear(): void {
    this.definitions.clear();
    this.instances.clear();
    this.cache.clear();
  }
}

/**
 * Create a new module registry
 */
export function createModuleRegistry(config?: RegistryConfig): ModuleRegistry {
  return new ModuleRegistry(config);
}
