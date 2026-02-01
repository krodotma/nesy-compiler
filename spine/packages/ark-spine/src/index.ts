/**
 * @ark/spine - Central Backbone Coordination Layer for Neo Pluribus
 *
 * The spine package provides two complementary systems:
 *
 * 1. CentralRegistry (LMDB-based):
 *    - URN-based symbol grounding (ring{0-3}/{domain}/{type}/{id})
 *    - Ring hierarchy enforcement
 *    - Temporal queries (resolve_at)
 *    - TTL expiration
 *    - Soft deletes (tombstones)
 *    - Bus event emission
 *
 * 2. SpineCoordinator (Module orchestration):
 *    - Module registration and lifecycle management
 *    - Dependency resolution with topological sort
 *    - Health monitoring
 *    - Integration with @ark/bus for coordination events
 *
 * @module
 * @example
 * ```typescript
 * // CentralRegistry usage
 * import { CentralRegistry, initRegistry } from '@ark/spine';
 *
 * const registry = initRegistry({ path: '/path/to/db' });
 * await registry.register('ring2/agents/claude/opus', { model: 'opus-4.5' });
 *
 * // SpineCoordinator usage
 * import { createSpineCoordinator, createModuleRegistry } from '@ark/spine';
 *
 * const coordinator = createSpineCoordinator();
 * coordinator.registerModule({
 *   id: '@ark/bus',
 *   metadata: { name: '@ark/bus', version: '0.1.0', ring: 1, capabilities: ['event-emitter'], tags: [] },
 *   dependencies: [],
 *   exports: ['Bus', 'createBus'],
 * });
 *
 * const result = await coordinator.loadAndStart('@ark/bus');
 * ```
 */

import { open, Database, RootDatabase } from 'lmdb';
import { encode, decode } from '@msgpack/msgpack';
import { Ring, parseURN, formatURN, type URN } from '@ark/core';
import type { Bus, BusEvent } from '@ark/bus';

// ==================== Central Registry (LMDB) ====================

/**
 * Registry entry metadata
 */
export interface RegistryEntry<T = unknown> {
  /** URN address */
  urn: string;
  /** Ring level */
  ring: Ring;
  /** Domain */
  domain: string;
  /** Type within domain */
  type: string;
  /** Unique ID */
  id: string;
  /** The actual data */
  data: T;
  /** Creation timestamp */
  createdAt: number;
  /** Last update timestamp */
  updatedAt: number;
  /** Expiration timestamp (0 = never) */
  expiresAt: number;
  /** Tombstone flag for soft delete */
  tombstone: boolean;
  /** Version number for optimistic locking */
  version: number;
  /** Metadata key-value pairs */
  metadata: Record<string, unknown>;
}

/**
 * Query options for registry lookups
 */
export interface QueryOptions {
  /** Filter by ring level */
  ring?: Ring;
  /** Filter by domain */
  domain?: string;
  /** Filter by type */
  type?: string;
  /** Include tombstoned entries */
  includeTombstones?: boolean;
  /** Resolve at specific timestamp (temporal query) */
  resolveAt?: number;
  /** Maximum results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
}

/**
 * Registry configuration (for CentralRegistry)
 */
export interface CentralRegistryConfig {
  /** Path to LMDB database */
  path: string;
  /** Maximum database size in bytes */
  maxSize?: number;
  /** Bus instance for event emission */
  bus?: Bus;
  /** Default TTL in seconds (0 = never expire) */
  defaultTtl?: number;
  /** Enable write-ahead logging */
  wal?: boolean;
}

/**
 * @deprecated Use CentralRegistryConfig instead
 */
export type RegistryConfig = CentralRegistryConfig;

/**
 * Central Registry - Unified symbol grounding for Pluribus
 */
export class CentralRegistry {
  private db: RootDatabase;
  private entries: Database<Uint8Array, string>;
  private index: Database<string[], string>;
  private config: Omit<Required<CentralRegistryConfig>, 'bus'> & { bus?: Bus };
  private bus?: Bus;

  constructor(config: CentralRegistryConfig) {
    this.config = {
      path: config.path,
      maxSize: config.maxSize ?? 10 * 1024 * 1024 * 1024, // 10GB default
      bus: config.bus,
      defaultTtl: config.defaultTtl ?? 0,
      wal: config.wal ?? true,
    };

    this.bus = config.bus;

    // Open LMDB database
    this.db = open({
      path: this.config.path,
      maxDbs: 10,
      mapSize: this.config.maxSize,
    });

    // Main entries database (URN -> msgpack entry)
    this.entries = this.db.openDB('entries', {
      encoding: 'binary',
    });

    // Index database for queries (index_key -> URN[])
    this.index = this.db.openDB('index', {
      encoding: 'json',
    });
  }

  /**
   * Register a new entry
   */
  async register<T>(
    urn: string,
    data: T,
    options?: {
      ttl?: number;
      metadata?: Record<string, unknown>;
    }
  ): Promise<RegistryEntry<T>> {
    const parsed = parseURN(urn);
    if (!parsed) {
      throw new Error(`Invalid URN format: ${urn}`);
    }

    const now = Date.now();
    const ttl = options?.ttl ?? this.config.defaultTtl;

    const entry: RegistryEntry<T> = {
      urn,
      ring: parsed.ring,
      domain: parsed.domain,
      type: parsed.type,
      id: parsed.id,
      data,
      createdAt: now,
      updatedAt: now,
      expiresAt: ttl > 0 ? now + ttl * 1000 : 0,
      tombstone: false,
      version: 1,
      metadata: options?.metadata ?? {},
    };

    // Store entry
    const packed = encode(entry);
    await this.entries.put(urn, packed);

    // Update indexes
    await this.updateIndexes(entry);

    // Emit bus event
    await this.emitEvent('registry.entry.created', entry);

    return entry;
  }

  /**
   * Resolve a URN to its entry
   */
  async resolve<T>(urn: string, resolveAt?: number): Promise<RegistryEntry<T> | null> {
    const packed = await this.entries.get(urn);
    if (!packed) return null;

    const entry = decode(packed) as RegistryEntry<T>;

    // Check tombstone
    if (entry.tombstone) return null;

    // Check expiration
    const now = resolveAt ?? Date.now();
    if (entry.expiresAt > 0 && now > entry.expiresAt) {
      return null;
    }

    return entry;
  }

  /**
   * Update an existing entry
   */
  async update<T>(
    urn: string,
    data: Partial<T>,
    options?: {
      metadata?: Record<string, unknown>;
      expectedVersion?: number;
    }
  ): Promise<RegistryEntry<T> | null> {
    const existing = await this.resolve<T>(urn);
    if (!existing) return null;

    // Optimistic locking
    if (options?.expectedVersion !== undefined && existing.version !== options.expectedVersion) {
      throw new Error(`Version conflict: expected ${options.expectedVersion}, got ${existing.version}`);
    }

    const updated: RegistryEntry<T> = {
      ...existing,
      data: { ...existing.data, ...data } as T,
      updatedAt: Date.now(),
      version: existing.version + 1,
      metadata: {
        ...existing.metadata,
        ...options?.metadata,
      },
    };

    const packed = encode(updated);
    await this.entries.put(urn, packed);

    await this.emitEvent('registry.entry.updated', updated);

    return updated;
  }

  /**
   * Soft delete (tombstone) an entry
   */
  async delete(urn: string): Promise<boolean> {
    const existing = await this.resolve(urn);
    if (!existing) return false;

    const tombstoned: RegistryEntry = {
      ...existing,
      tombstone: true,
      updatedAt: Date.now(),
      version: existing.version + 1,
    };

    const packed = encode(tombstoned);
    await this.entries.put(urn, packed);

    await this.emitEvent('registry.entry.deleted', { urn });

    return true;
  }

  /**
   * Query entries with filters
   */
  async query<T>(options: QueryOptions = {}): Promise<RegistryEntry<T>[]> {
    const results: RegistryEntry<T>[] = [];
    const now = options.resolveAt ?? Date.now();

    // Build index key for efficient lookup
    const indexKey = this.buildIndexKey(options);

    // If we have an index key, use it
    let urns: string[];
    if (indexKey) {
      urns = (await this.index.get(indexKey)) ?? [];
    } else {
      // Full scan (expensive, but necessary for complex queries)
      urns = await this.getAllUrns();
    }

    // Filter and collect results
    let count = 0;
    const offset = options.offset ?? 0;
    const limit = options.limit ?? 1000;

    for (const urn of urns) {
      const entry = await this.resolve<T>(urn, now);
      if (!entry) continue;

      // Apply filters
      if (options.ring !== undefined && entry.ring !== options.ring) continue;
      if (options.domain !== undefined && entry.domain !== options.domain) continue;
      if (options.type !== undefined && entry.type !== options.type) continue;
      if (!options.includeTombstones && entry.tombstone) continue;

      // Pagination
      if (count < offset) {
        count++;
        continue;
      }
      if (results.length >= limit) break;

      results.push(entry);
      count++;
    }

    return results;
  }

  /**
   * List all entries in a domain
   */
  async listDomain(domain: string, ring?: Ring): Promise<string[]> {
    const entries = await this.query({
      domain,
      ring,
      limit: 10000,
    });
    return entries.map((e) => e.urn);
  }

  /**
   * Check if a URN exists and is not expired/tombstoned
   */
  async exists(urn: string): Promise<boolean> {
    const entry = await this.resolve(urn);
    return entry !== null;
  }

  /**
   * Get statistics about the registry
   */
  async stats(): Promise<{
    totalEntries: number;
    byRing: Record<Ring, number>;
    byDomain: Record<string, number>;
    tombstoned: number;
    expired: number;
  }> {
    const stats = {
      totalEntries: 0,
      byRing: { 0: 0, 1: 0, 2: 0, 3: 0 } as Record<Ring, number>,
      byDomain: {} as Record<string, number>,
      tombstoned: 0,
      expired: 0,
    };

    const now = Date.now();
    const urns = await this.getAllUrns();

    for (const urn of urns) {
      const packed = await this.entries.get(urn);
      if (!packed) continue;

      const entry = decode(packed) as RegistryEntry;
      stats.totalEntries++;

      stats.byRing[entry.ring]++;
      stats.byDomain[entry.domain] = (stats.byDomain[entry.domain] ?? 0) + 1;

      if (entry.tombstone) stats.tombstoned++;
      if (entry.expiresAt > 0 && now > entry.expiresAt) stats.expired++;
    }

    return stats;
  }

  /**
   * Close the database
   */
  async close(): Promise<void> {
    await this.db.close();
  }

  // Private helpers

  private async getAllUrns(): Promise<string[]> {
    const urns: string[] = [];
    for await (const { key } of this.entries.getRange({})) {
      urns.push(key as string);
    }
    return urns;
  }

  private buildIndexKey(options: QueryOptions): string | null {
    if (options.ring !== undefined && options.domain) {
      return `ring${options.ring}:${options.domain}`;
    }
    if (options.domain) {
      return `domain:${options.domain}`;
    }
    if (options.ring !== undefined) {
      return `ring:${options.ring}`;
    }
    return null;
  }

  private async updateIndexes(entry: RegistryEntry): Promise<void> {
    // Index by ring
    const ringKey = `ring:${entry.ring}`;
    const ringUrns = (await this.index.get(ringKey)) ?? [];
    if (!ringUrns.includes(entry.urn)) {
      ringUrns.push(entry.urn);
      await this.index.put(ringKey, ringUrns);
    }

    // Index by domain
    const domainKey = `domain:${entry.domain}`;
    const domainUrns = (await this.index.get(domainKey)) ?? [];
    if (!domainUrns.includes(entry.urn)) {
      domainUrns.push(entry.urn);
      await this.index.put(domainKey, domainUrns);
    }

    // Composite index
    const compositeKey = `ring${entry.ring}:${entry.domain}`;
    const compositeUrns = (await this.index.get(compositeKey)) ?? [];
    if (!compositeUrns.includes(entry.urn)) {
      compositeUrns.push(entry.urn);
      await this.index.put(compositeKey, compositeUrns);
    }
  }

  private async emitEvent(topic: string, data: unknown): Promise<void> {
    if (this.bus) {
      await this.bus.emit(topic, data, { kind: 'event' });
    }
  }
}

// Singleton instance for CentralRegistry
let centralRegistryInstance: CentralRegistry | null = null;

/**
 * Get or create the singleton CentralRegistry instance
 */
export function getRegistry(config?: CentralRegistryConfig): CentralRegistry {
  if (!centralRegistryInstance && config) {
    centralRegistryInstance = new CentralRegistry(config);
  }
  if (!centralRegistryInstance) {
    throw new Error('Registry not initialized. Call with config first.');
  }
  return centralRegistryInstance;
}

/**
 * Initialize the CentralRegistry with config
 */
export function initRegistry(config: CentralRegistryConfig): CentralRegistry {
  if (centralRegistryInstance) {
    throw new Error('Registry already initialized');
  }
  centralRegistryInstance = new CentralRegistry(config);
  return centralRegistryInstance;
}

/**
 * Reset the CentralRegistry singleton (for testing)
 */
export function resetRegistry(): void {
  centralRegistryInstance = null;
}

// ==================== Re-exports from submodules ====================

// Types
export * from './types.js';

// Module Registry
export { ModuleRegistry, createModuleRegistry } from './module-registry.js';

// Dependency Resolver
export { DependencyResolver, createDependencyResolver } from './dependency-resolver.js';

// Lifecycle Manager
export { LifecycleManager, createLifecycleManager } from './lifecycle-manager.js';

// Spine Coordinator
export {
  SpineCoordinator,
  createSpineCoordinator,
  type EventHandler,
} from './spine-coordinator.js';

// Version
export const VERSION = '0.1.0';
