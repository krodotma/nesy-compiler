/**
 * @ark/spine - Central Registry for Neo Pluribus
 *
 * LMDB-based unified registry with:
 * - URN-based symbol grounding (ring{0-3}/{domain}/{type}/{id})
 * - Ring hierarchy enforcement
 * - Temporal queries (resolve_at)
 * - TTL expiration
 * - Soft deletes (tombstones)
 * - Bus event emission
 *
 * This is the TypeScript port of central_registry.py
 *
 * @module
 */
import { Ring } from '@ark/core';
import type { Bus } from '@ark/bus';
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
 * Registry configuration
 */
export interface RegistryConfig {
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
 * Central Registry - Unified symbol grounding for Pluribus
 */
export declare class CentralRegistry {
    private db;
    private entries;
    private index;
    private config;
    private bus?;
    constructor(config: RegistryConfig);
    /**
     * Register a new entry
     */
    register<T>(urn: string, data: T, options?: {
        ttl?: number;
        metadata?: Record<string, unknown>;
    }): Promise<RegistryEntry<T>>;
    /**
     * Resolve a URN to its entry
     */
    resolve<T>(urn: string, resolveAt?: number): Promise<RegistryEntry<T> | null>;
    /**
     * Update an existing entry
     */
    update<T>(urn: string, data: Partial<T>, options?: {
        metadata?: Record<string, unknown>;
        expectedVersion?: number;
    }): Promise<RegistryEntry<T> | null>;
    /**
     * Soft delete (tombstone) an entry
     */
    delete(urn: string): Promise<boolean>;
    /**
     * Query entries with filters
     */
    query<T>(options?: QueryOptions): Promise<RegistryEntry<T>[]>;
    /**
     * List all entries in a domain
     */
    listDomain(domain: string, ring?: Ring): Promise<string[]>;
    /**
     * Check if a URN exists and is not expired/tombstoned
     */
    exists(urn: string): Promise<boolean>;
    /**
     * Get statistics about the registry
     */
    stats(): Promise<{
        totalEntries: number;
        byRing: Record<Ring, number>;
        byDomain: Record<string, number>;
        tombstoned: number;
        expired: number;
    }>;
    /**
     * Close the database
     */
    close(): Promise<void>;
    private getAllUrns;
    private buildIndexKey;
    private updateIndexes;
    private emitEvent;
}
/**
 * Get or create the singleton registry instance
 */
export declare function getRegistry(config?: RegistryConfig): CentralRegistry;
/**
 * Initialize the registry with config
 */
export declare function initRegistry(config: RegistryConfig): CentralRegistry;
//# sourceMappingURL=index.d.ts.map