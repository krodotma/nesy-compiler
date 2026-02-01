/**
 * @ark/bus/ledger - DKIN Immutable Audit Ledger
 *
 * Append-only NDJSON ledger for accountability and audit trails.
 * Implements hash-chain integrity for tamper detection.
 *
 * Features:
 * - Append-only NDJSON logging
 * - SHA-256 hash chain for integrity verification
 * - Event recording with timestamps
 * - Query and replay capabilities
 * - Bus integration for event sourcing
 *
 * Ported from nucleus/tools/dkin_ledger.py
 *
 * @module
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import * as crypto from 'node:crypto';
import type { Bus, BusEvent } from './index.js';

/**
 * Genesis block hash (64 zeros)
 */
export const GENESIS_HASH = '0'.repeat(64);

/**
 * Produce a canonical JSON string with sorted keys (recursively)
 * This ensures the same object always produces the same string
 * Matches JSON.stringify behavior: undefined values are skipped
 */
function canonicalJson(obj: unknown): string {
  if (obj === null) {
    return 'null';
  }

  if (obj === undefined) {
    // undefined is not valid JSON, but we handle it for completeness
    return 'null';
  }

  if (Array.isArray(obj)) {
    // Arrays: undefined elements become null (JSON.stringify behavior)
    const items = obj.map((item) => (item === undefined ? 'null' : canonicalJson(item)));
    return '[' + items.join(',') + ']';
  }

  if (typeof obj === 'object') {
    const sortedKeys = Object.keys(obj).sort();
    const pairs: string[] = [];
    for (const key of sortedKeys) {
      const value = (obj as Record<string, unknown>)[key];
      // Skip undefined values (JSON.stringify behavior)
      if (value !== undefined) {
        pairs.push(JSON.stringify(key) + ':' + canonicalJson(value));
      }
    }
    return '{' + pairs.join(',') + '}';
  }

  return JSON.stringify(obj);
}

/**
 * Ledger entry representing an immutable record
 */
export interface LedgerEntry<T = unknown> {
  /** Entry type (e.g., 'action', 'decision', 'audit') */
  type: string;
  /** Unix timestamp in seconds */
  timestamp: number;
  /** Entry payload */
  data: T;
  /** Hash of previous entry (chain link) */
  prev_hash: string;
  /** SHA-256 hash of this entry */
  hash: string;
  /** Sequence number (0-indexed) */
  seq?: number;
  /** Actor who created this entry */
  actor?: string;
  /** Correlation ID for tracing */
  correlationId?: string;
}

/**
 * Ledger query options
 */
export interface LedgerQueryOptions {
  /** Filter by entry type */
  type?: string;
  /** Filter by types (OR) */
  types?: string[];
  /** Start timestamp (inclusive) */
  since?: number;
  /** End timestamp (inclusive) */
  until?: number;
  /** Filter by actor */
  actor?: string;
  /** Filter by correlation ID */
  correlationId?: string;
  /** Maximum entries to return */
  limit?: number;
  /** Skip first N entries */
  offset?: number;
  /** Reverse order (newest first) */
  reverse?: boolean;
  /** Custom predicate */
  filter?: (entry: LedgerEntry) => boolean;
}

/**
 * Ledger verification result
 */
export interface VerificationResult {
  /** Is the ledger valid? */
  valid: boolean;
  /** Total entries verified */
  entriesVerified: number;
  /** Index of first invalid entry (-1 if valid) */
  invalidAt: number;
  /** Error message if invalid */
  error?: string;
  /** First entry hash */
  genesisHash: string;
  /** Last entry hash */
  headHash: string;
}

/**
 * Ledger statistics
 */
export interface LedgerStats {
  /** Total entry count */
  totalEntries: number;
  /** Entries by type */
  byType: Record<string, number>;
  /** Entries by actor */
  byActor: Record<string, number>;
  /** First entry timestamp */
  firstTimestamp: number | null;
  /** Last entry timestamp */
  lastTimestamp: number | null;
  /** File size in bytes */
  fileSizeBytes: number;
  /** Current head hash */
  headHash: string;
}

/**
 * Ledger configuration
 */
export interface LedgerConfig {
  /** Directory for ledger storage */
  ledgerDir: string;
  /** Ledger file name (default: ledger.jsonl) */
  ledgerFile?: string;
  /** Actor name for entries */
  actor?: string;
  /** Bus instance for event emission */
  bus?: Bus;
  /** Enable fsync after each write (default: true) */
  fsync?: boolean;
}

/**
 * DKIN Immutable Audit Ledger
 *
 * Provides append-only, hash-chained logging for accountability.
 * Each entry links to the previous via SHA-256, enabling
 * tamper detection across the entire history.
 */
export class DKINLedger {
  private config: Required<Omit<LedgerConfig, 'bus'>> & { bus?: Bus };
  private ledgerPath: string;
  private headHash: string = GENESIS_HASH;
  private entryCount: number = 0;
  private bus?: Bus;

  constructor(config: LedgerConfig) {
    this.config = {
      ledgerDir: config.ledgerDir,
      ledgerFile: config.ledgerFile ?? 'ledger.jsonl',
      actor: config.actor ?? process.env.PLURIBUS_ACTOR ?? 'unknown',
      bus: config.bus,
      fsync: config.fsync ?? true,
    };

    this.bus = config.bus;
    this.ledgerPath = path.join(this.config.ledgerDir, this.config.ledgerFile);

    this.initialize();
  }

  /**
   * Initialize the ledger
   */
  private initialize(): void {
    // Ensure directory exists
    if (!fs.existsSync(this.config.ledgerDir)) {
      fs.mkdirSync(this.config.ledgerDir, { recursive: true });
    }

    // Load head hash from existing file
    if (fs.existsSync(this.ledgerPath)) {
      const { hash, count } = this.loadHead();
      this.headHash = hash;
      this.entryCount = count;
    }
  }

  /**
   * Load the last hash and entry count from the ledger file
   */
  private loadHead(): { hash: string; count: number } {
    if (!fs.existsSync(this.ledgerPath)) {
      return { hash: GENESIS_HASH, count: 0 };
    }

    try {
      const stats = fs.statSync(this.ledgerPath);
      if (stats.size === 0) {
        return { hash: GENESIS_HASH, count: 0 };
      }

      // Read file to find last entry and count
      const content = fs.readFileSync(this.ledgerPath, 'utf-8');
      const lines = content.trim().split('\n').filter(Boolean);

      if (lines.length === 0) {
        return { hash: GENESIS_HASH, count: 0 };
      }

      const lastLine = lines[lines.length - 1];
      try {
        const entry = JSON.parse(lastLine) as LedgerEntry;
        return { hash: entry.hash ?? GENESIS_HASH, count: lines.length };
      } catch {
        return { hash: GENESIS_HASH, count: lines.length };
      }
    } catch {
      return { hash: GENESIS_HASH, count: 0 };
    }
  }

  /**
   * Calculate SHA-256 hash of entry + previous hash
   */
  private calculateHash(data: Omit<LedgerEntry, 'hash'>, prevHash: string): string {
    // Canonical JSON string (sorted keys recursively)
    const payload = canonicalJson(data);
    const content = `${prevHash}|${payload}`;
    return crypto.createHash('sha256').update(content, 'utf-8').digest('hex');
  }

  /**
   * Record a new entry to the ledger
   *
   * @param entryType - Type of entry (e.g., 'action', 'decision')
   * @param data - Entry payload
   * @param options - Additional options
   * @returns The entry hash, or null on failure
   */
  record<T>(
    entryType: string,
    data: T,
    options?: {
      actor?: string;
      correlationId?: string;
    }
  ): string | null {
    const timestamp = Date.now() / 1000; // Seconds for compatibility with Python
    const prevHash = this.headHash;

    // Build entry, only including optional fields if they have values
    // This ensures hash consistency when reading back from JSON
    const entry: Omit<LedgerEntry<T>, 'hash'> = {
      type: entryType,
      timestamp,
      data,
      prev_hash: prevHash,
      seq: this.entryCount,
      actor: options?.actor ?? this.config.actor,
    };

    // Only add correlationId if present
    if (options?.correlationId !== undefined) {
      entry.correlationId = options.correlationId;
    }

    // Calculate integrity hash
    const hash = this.calculateHash(entry as Omit<LedgerEntry, 'hash'>, prevHash);
    const fullEntry: LedgerEntry<T> = { ...entry, hash };

    try {
      // Append to file
      const line = JSON.stringify(fullEntry) + '\n';
      fs.appendFileSync(this.ledgerPath, line, 'utf-8');

      // Optional fsync for durability
      if (this.config.fsync) {
        const fd = fs.openSync(this.ledgerPath, 'r');
        fs.fsyncSync(fd);
        fs.closeSync(fd);
      }

      // Update state
      this.headHash = hash;
      this.entryCount++;

      // Emit bus event
      this.emitEvent('ledger.entry.recorded', fullEntry);

      return hash;
    } catch (error) {
      console.error('[DKIN Ledger] Write failed:', error);
      return null;
    }
  }

  /**
   * Read all entries from the ledger
   */
  readAll<T = unknown>(): LedgerEntry<T>[] {
    if (!fs.existsSync(this.ledgerPath)) {
      return [];
    }

    const content = fs.readFileSync(this.ledgerPath, 'utf-8');
    const lines = content.trim().split('\n').filter(Boolean);

    return lines
      .map((line) => {
        try {
          return JSON.parse(line) as LedgerEntry<T>;
        } catch {
          return null;
        }
      })
      .filter((e): e is LedgerEntry<T> => e !== null);
  }

  /**
   * Query entries with filters
   */
  query<T = unknown>(options: LedgerQueryOptions = {}): LedgerEntry<T>[] {
    let entries = this.readAll<T>();

    // Apply type filter
    if (options.type) {
      entries = entries.filter((e) => e.type === options.type);
    }
    if (options.types && options.types.length > 0) {
      entries = entries.filter((e) => options.types!.includes(e.type));
    }

    // Apply timestamp filters
    if (options.since !== undefined) {
      entries = entries.filter((e) => e.timestamp >= options.since!);
    }
    if (options.until !== undefined) {
      entries = entries.filter((e) => e.timestamp <= options.until!);
    }

    // Apply actor filter
    if (options.actor) {
      entries = entries.filter((e) => e.actor === options.actor);
    }

    // Apply correlation ID filter
    if (options.correlationId) {
      entries = entries.filter((e) => e.correlationId === options.correlationId);
    }

    // Apply custom predicate
    if (options.filter) {
      entries = entries.filter(options.filter);
    }

    // Apply ordering
    if (options.reverse) {
      entries = entries.reverse();
    }

    // Apply pagination
    const offset = options.offset ?? 0;
    const limit = options.limit ?? entries.length;
    entries = entries.slice(offset, offset + limit);

    return entries;
  }

  /**
   * Replay entries to a handler
   */
  async replay(
    handler: (entry: LedgerEntry) => void | Promise<void>,
    options: LedgerQueryOptions = {}
  ): Promise<number> {
    const entries = this.query(options);

    for (const entry of entries) {
      await handler(entry);
    }

    return entries.length;
  }

  /**
   * Verify the integrity of the ledger hash chain
   */
  verify(): VerificationResult {
    const entries = this.readAll();

    if (entries.length === 0) {
      return {
        valid: true,
        entriesVerified: 0,
        invalidAt: -1,
        genesisHash: GENESIS_HASH,
        headHash: GENESIS_HASH,
      };
    }

    let prevHash = GENESIS_HASH;
    let valid = true;
    let invalidAt = -1;
    let error: string | undefined;

    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];

      // Check prev_hash links correctly
      if (entry.prev_hash !== prevHash) {
        valid = false;
        invalidAt = i;
        error = `Entry ${i}: prev_hash mismatch. Expected ${prevHash.slice(0, 8)}..., got ${entry.prev_hash.slice(0, 8)}...`;
        break;
      }

      // Recalculate hash
      const { hash: storedHash, ...entryWithoutHash } = entry;
      const calculatedHash = this.calculateHash(entryWithoutHash, prevHash);

      if (calculatedHash !== storedHash) {
        valid = false;
        invalidAt = i;
        error = `Entry ${i}: hash mismatch. Calculated ${calculatedHash.slice(0, 8)}..., stored ${storedHash.slice(0, 8)}...`;
        break;
      }

      prevHash = storedHash;
    }

    return {
      valid,
      entriesVerified: entries.length,
      invalidAt,
      error,
      genesisHash: entries[0]?.prev_hash ?? GENESIS_HASH,
      headHash: entries[entries.length - 1]?.hash ?? GENESIS_HASH,
    };
  }

  /**
   * Get ledger statistics
   */
  stats(): LedgerStats {
    const entries = this.readAll();
    const byType: Record<string, number> = {};
    const byActor: Record<string, number> = {};

    let firstTimestamp: number | null = null;
    let lastTimestamp: number | null = null;

    for (const entry of entries) {
      // Count by type
      byType[entry.type] = (byType[entry.type] ?? 0) + 1;

      // Count by actor
      if (entry.actor) {
        byActor[entry.actor] = (byActor[entry.actor] ?? 0) + 1;
      }

      // Track timestamps
      if (firstTimestamp === null || entry.timestamp < firstTimestamp) {
        firstTimestamp = entry.timestamp;
      }
      if (lastTimestamp === null || entry.timestamp > lastTimestamp) {
        lastTimestamp = entry.timestamp;
      }
    }

    let fileSizeBytes = 0;
    if (fs.existsSync(this.ledgerPath)) {
      fileSizeBytes = fs.statSync(this.ledgerPath).size;
    }

    return {
      totalEntries: entries.length,
      byType,
      byActor,
      firstTimestamp,
      lastTimestamp,
      fileSizeBytes,
      headHash: this.headHash,
    };
  }

  /**
   * Get an entry by its hash
   */
  getByHash<T = unknown>(hash: string): LedgerEntry<T> | null {
    const entries = this.readAll<T>();
    return entries.find((e) => e.hash === hash) ?? null;
  }

  /**
   * Get an entry by sequence number
   */
  getBySeq<T = unknown>(seq: number): LedgerEntry<T> | null {
    const entries = this.readAll<T>();
    return entries.find((e) => e.seq === seq) ?? null;
  }

  /**
   * Get the current head hash
   */
  getHeadHash(): string {
    return this.headHash;
  }

  /**
   * Get the current entry count
   */
  getEntryCount(): number {
    return this.entryCount;
  }

  /**
   * Get the ledger file path
   */
  getLedgerPath(): string {
    return this.ledgerPath;
  }

  /**
   * Tail the ledger (async iterator for new entries)
   */
  async *tail(): AsyncGenerator<LedgerEntry, void, unknown> {
    let position = 0;

    // Get initial position
    if (fs.existsSync(this.ledgerPath)) {
      const stats = fs.statSync(this.ledgerPath);
      position = stats.size;
    }

    while (true) {
      // Check for new data
      if (fs.existsSync(this.ledgerPath)) {
        const stats = fs.statSync(this.ledgerPath);
        if (stats.size > position) {
          // Read new data
          const fd = fs.openSync(this.ledgerPath, 'r');
          const buffer = Buffer.alloc(stats.size - position);
          fs.readSync(fd, buffer, 0, buffer.length, position);
          fs.closeSync(fd);

          position = stats.size;

          // Parse and yield entries
          const content = buffer.toString('utf-8');
          const lines = content.trim().split('\n').filter(Boolean);

          for (const line of lines) {
            try {
              const entry = JSON.parse(line) as LedgerEntry;
              yield entry;
            } catch {
              // Skip malformed lines
            }
          }
        }
      }

      // Wait before checking again
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  /**
   * Emit a bus event
   */
  private emitEvent(topic: string, data: unknown): void {
    if (this.bus) {
      this.bus.emit(topic, data, { kind: 'event' }).catch(() => {
        // Ignore bus errors - ledger should work independently
      });
    }
  }
}

/**
 * Singleton instance management
 */
let defaultInstance: DKINLedger | null = null;

/**
 * Get or create the default ledger instance
 */
export function getLedger(config?: LedgerConfig): DKINLedger {
  if (!defaultInstance && config) {
    defaultInstance = new DKINLedger(config);
  }
  if (!defaultInstance) {
    throw new Error('Ledger not initialized. Call with config first.');
  }
  return defaultInstance;
}

/**
 * Create a new ledger instance
 */
export function createLedger(config: LedgerConfig): DKINLedger {
  return new DKINLedger(config);
}

/**
 * Reset the default instance (for testing)
 */
export function resetLedger(): void {
  defaultInstance = null;
}

// Version
export const LEDGER_VERSION = '0.1.0';
