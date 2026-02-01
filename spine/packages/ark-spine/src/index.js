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
import { open } from 'lmdb';
import { encode, decode } from '@msgpack/msgpack';
import { parseURN } from '@ark/core';
/**
 * Central Registry - Unified symbol grounding for Pluribus
 */
export class CentralRegistry {
    db;
    entries;
    index;
    config;
    bus;
    constructor(config) {
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
    async register(urn, data, options) {
        const parsed = parseURN(urn);
        if (!parsed) {
            throw new Error(`Invalid URN format: ${urn}`);
        }
        const now = Date.now();
        const ttl = options?.ttl ?? this.config.defaultTtl;
        const entry = {
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
    async resolve(urn, resolveAt) {
        const packed = await this.entries.get(urn);
        if (!packed)
            return null;
        const entry = decode(packed);
        // Check tombstone
        if (entry.tombstone)
            return null;
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
    async update(urn, data, options) {
        const existing = await this.resolve(urn);
        if (!existing)
            return null;
        // Optimistic locking
        if (options?.expectedVersion !== undefined && existing.version !== options.expectedVersion) {
            throw new Error(`Version conflict: expected ${options.expectedVersion}, got ${existing.version}`);
        }
        const updated = {
            ...existing,
            data: { ...existing.data, ...data },
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
    async delete(urn) {
        const existing = await this.resolve(urn);
        if (!existing)
            return false;
        const tombstoned = {
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
    async query(options = {}) {
        const results = [];
        const now = options.resolveAt ?? Date.now();
        // Build index key for efficient lookup
        const indexKey = this.buildIndexKey(options);
        // If we have an index key, use it
        let urns;
        if (indexKey) {
            urns = (await this.index.get(indexKey)) ?? [];
        }
        else {
            // Full scan (expensive, but necessary for complex queries)
            urns = await this.getAllUrns();
        }
        // Filter and collect results
        let count = 0;
        const offset = options.offset ?? 0;
        const limit = options.limit ?? 1000;
        for (const urn of urns) {
            const entry = await this.resolve(urn, now);
            if (!entry)
                continue;
            // Apply filters
            if (options.ring !== undefined && entry.ring !== options.ring)
                continue;
            if (options.domain !== undefined && entry.domain !== options.domain)
                continue;
            if (options.type !== undefined && entry.type !== options.type)
                continue;
            if (!options.includeTombstones && entry.tombstone)
                continue;
            // Pagination
            if (count < offset) {
                count++;
                continue;
            }
            if (results.length >= limit)
                break;
            results.push(entry);
            count++;
        }
        return results;
    }
    /**
     * List all entries in a domain
     */
    async listDomain(domain, ring) {
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
    async exists(urn) {
        const entry = await this.resolve(urn);
        return entry !== null;
    }
    /**
     * Get statistics about the registry
     */
    async stats() {
        const stats = {
            totalEntries: 0,
            byRing: { 0: 0, 1: 0, 2: 0, 3: 0 },
            byDomain: {},
            tombstoned: 0,
            expired: 0,
        };
        const now = Date.now();
        const urns = await this.getAllUrns();
        for (const urn of urns) {
            const packed = await this.entries.get(urn);
            if (!packed)
                continue;
            const entry = decode(packed);
            stats.totalEntries++;
            stats.byRing[entry.ring]++;
            stats.byDomain[entry.domain] = (stats.byDomain[entry.domain] ?? 0) + 1;
            if (entry.tombstone)
                stats.tombstoned++;
            if (entry.expiresAt > 0 && now > entry.expiresAt)
                stats.expired++;
        }
        return stats;
    }
    /**
     * Close the database
     */
    async close() {
        await this.db.close();
    }
    // Private helpers
    async getAllUrns() {
        const urns = [];
        for await (const { key } of this.entries.getRange({})) {
            urns.push(key);
        }
        return urns;
    }
    buildIndexKey(options) {
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
    async updateIndexes(entry) {
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
    async emitEvent(topic, data) {
        if (this.bus) {
            await this.bus.emit(topic, data, { kind: 'event' });
        }
    }
}
// Singleton instance
let instance = null;
/**
 * Get or create the singleton registry instance
 */
export function getRegistry(config) {
    if (!instance && config) {
        instance = new CentralRegistry(config);
    }
    if (!instance) {
        throw new Error('Registry not initialized. Call with config first.');
    }
    return instance;
}
/**
 * Initialize the registry with config
 */
export function initRegistry(config) {
    if (instance) {
        throw new Error('Registry already initialized');
    }
    instance = new CentralRegistry(config);
    return instance;
}
//# sourceMappingURL=index.js.map