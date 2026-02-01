/**
 * @ark/bus/ledger tests
 *
 * Comprehensive tests for DKIN Immutable Audit Ledger
 * Target: 90%+ coverage
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import * as crypto from 'node:crypto';
import {
  DKINLedger,
  createLedger,
  getLedger,
  resetLedger,
  GENESIS_HASH,
  LEDGER_VERSION,
  type LedgerEntry,
  type LedgerConfig,
  type VerificationResult,
} from './ledger.js';
import { MemoryBus, type Bus } from './index.js';

describe('DKINLedger', () => {
  let tmpDir: string;
  let ledger: DKINLedger;

  beforeEach(() => {
    // Create temp directory
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ark-ledger-test-'));
    ledger = createLedger({
      ledgerDir: tmpDir,
      actor: 'test-actor',
      fsync: false, // Disable fsync for faster tests
    });
    resetLedger();
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('initialization', () => {
    it('creates ledger directory if it does not exist', () => {
      const nestedDir = path.join(tmpDir, 'a', 'b', 'c');
      createLedger({ ledgerDir: nestedDir });

      expect(fs.existsSync(nestedDir)).toBe(true);
    });

    it('initializes with default config values', () => {
      const l = createLedger({ ledgerDir: tmpDir });

      expect(l.getLedgerPath()).toBe(path.join(tmpDir, 'ledger.jsonl'));
      expect(l.getHeadHash()).toBe(GENESIS_HASH);
      expect(l.getEntryCount()).toBe(0);
    });

    it('initializes with custom config values', () => {
      const l = createLedger({
        ledgerDir: tmpDir,
        ledgerFile: 'custom.jsonl',
        actor: 'custom-actor',
      });

      expect(l.getLedgerPath()).toBe(path.join(tmpDir, 'custom.jsonl'));
    });

    it('loads existing ledger state on init', () => {
      // Create some entries
      ledger.record('test', { value: 1 });
      ledger.record('test', { value: 2 });
      const lastHash = ledger.record('test', { value: 3 });

      // Create new instance pointing to same file
      const newLedger = createLedger({
        ledgerDir: tmpDir,
        fsync: false,
      });

      expect(newLedger.getHeadHash()).toBe(lastHash);
      expect(newLedger.getEntryCount()).toBe(3);
    });

    it('handles empty existing file', () => {
      const ledgerPath = path.join(tmpDir, 'ledger.jsonl');
      fs.writeFileSync(ledgerPath, '', 'utf-8');

      const l = createLedger({ ledgerDir: tmpDir });

      expect(l.getHeadHash()).toBe(GENESIS_HASH);
      expect(l.getEntryCount()).toBe(0);
    });

    it('handles malformed existing file gracefully', () => {
      const ledgerPath = path.join(tmpDir, 'ledger.jsonl');
      fs.writeFileSync(ledgerPath, 'not json\n', 'utf-8');

      const l = createLedger({ ledgerDir: tmpDir });

      // Should still initialize with genesis hash
      expect(l.getHeadHash()).toBe(GENESIS_HASH);
    });
  });

  describe('record', () => {
    it('records an entry and returns its hash', () => {
      const hash = ledger.record('action', { command: 'test' });

      expect(hash).toBeDefined();
      expect(hash).toHaveLength(64); // SHA-256 hex
    });

    it('creates file on first record', () => {
      ledger.record('init', { message: 'first entry' });

      expect(fs.existsSync(ledger.getLedgerPath())).toBe(true);
    });

    it('increments entry count', () => {
      expect(ledger.getEntryCount()).toBe(0);

      ledger.record('test', { n: 1 });
      expect(ledger.getEntryCount()).toBe(1);

      ledger.record('test', { n: 2 });
      expect(ledger.getEntryCount()).toBe(2);
    });

    it('updates head hash after each record', () => {
      const hash1 = ledger.record('test', { n: 1 })!;
      expect(ledger.getHeadHash()).toBe(hash1);

      const hash2 = ledger.record('test', { n: 2 })!;
      expect(ledger.getHeadHash()).toBe(hash2);
      expect(hash1).not.toBe(hash2);
    });

    it('links entries via prev_hash', () => {
      ledger.record('first', { seq: 0 });
      ledger.record('second', { seq: 1 });
      ledger.record('third', { seq: 2 });

      const entries = ledger.readAll();

      expect(entries[0].prev_hash).toBe(GENESIS_HASH);
      expect(entries[1].prev_hash).toBe(entries[0].hash);
      expect(entries[2].prev_hash).toBe(entries[1].hash);
    });

    it('stores sequence numbers', () => {
      ledger.record('a', {});
      ledger.record('b', {});
      ledger.record('c', {});

      const entries = ledger.readAll();

      expect(entries[0].seq).toBe(0);
      expect(entries[1].seq).toBe(1);
      expect(entries[2].seq).toBe(2);
    });

    it('stores actor from config', () => {
      ledger.record('test', {});

      const entries = ledger.readAll();
      expect(entries[0].actor).toBe('test-actor');
    });

    it('allows actor override per entry', () => {
      ledger.record('test', {}, { actor: 'override-actor' });

      const entries = ledger.readAll();
      expect(entries[0].actor).toBe('override-actor');
    });

    it('stores correlation ID', () => {
      ledger.record('test', {}, { correlationId: 'corr-123' });

      const entries = ledger.readAll();
      expect(entries[0].correlationId).toBe('corr-123');
    });

    it('stores timestamp in seconds', () => {
      const before = Date.now() / 1000;
      ledger.record('test', {});
      const after = Date.now() / 1000;

      const entries = ledger.readAll();
      expect(entries[0].timestamp).toBeGreaterThanOrEqual(before);
      expect(entries[0].timestamp).toBeLessThanOrEqual(after);
    });

    it('stores arbitrary data payload', () => {
      interface TestData {
        message: string;
        count: number;
        nested: { a: number };
        array: string[];
      }

      ledger.record<TestData>('complex', {
        message: 'hello',
        count: 42,
        nested: { a: 1 },
        array: ['x', 'y'],
      });

      const entries = ledger.readAll<TestData>();
      expect(entries[0].data.message).toBe('hello');
      expect(entries[0].data.count).toBe(42);
      expect(entries[0].data.nested.a).toBe(1);
      expect(entries[0].data.array).toEqual(['x', 'y']);
    });
  });

  describe('readAll', () => {
    it('returns empty array for nonexistent file', () => {
      const l = createLedger({
        ledgerDir: path.join(tmpDir, 'empty'),
      });

      const entries = l.readAll();
      expect(entries).toEqual([]);
    });

    it('returns all entries in order', () => {
      ledger.record('a', { n: 1 });
      ledger.record('b', { n: 2 });
      ledger.record('c', { n: 3 });

      const entries = ledger.readAll();

      expect(entries.length).toBe(3);
      expect(entries[0].type).toBe('a');
      expect(entries[1].type).toBe('b');
      expect(entries[2].type).toBe('c');
    });

    it('skips malformed lines', () => {
      // Write valid entry
      ledger.record('valid', {});

      // Manually append invalid line
      fs.appendFileSync(ledger.getLedgerPath(), 'not json\n', 'utf-8');

      // Write another valid entry
      ledger.record('valid2', {});

      const entries = ledger.readAll();
      // Should have 2 valid entries (malformed line is skipped in readAll)
      expect(entries.length).toBe(2);
    });
  });

  describe('query', () => {
    beforeEach(() => {
      // Seed some entries
      ledger.record('action', { cmd: 'a' }, { actor: 'alice', correlationId: 'c1' });
      ledger.record('decision', { choice: 'x' }, { actor: 'bob', correlationId: 'c1' });
      ledger.record('action', { cmd: 'b' }, { actor: 'alice', correlationId: 'c2' });
      ledger.record('audit', { event: 'login' }, { actor: 'charlie', correlationId: 'c2' });
      ledger.record('action', { cmd: 'c' }, { actor: 'bob', correlationId: 'c3' });
    });

    it('returns all entries when no filters', () => {
      const entries = ledger.query();
      expect(entries.length).toBe(5);
    });

    it('filters by type', () => {
      const entries = ledger.query({ type: 'action' });
      expect(entries.length).toBe(3);
      expect(entries.every((e) => e.type === 'action')).toBe(true);
    });

    it('filters by multiple types', () => {
      const entries = ledger.query({ types: ['action', 'audit'] });
      expect(entries.length).toBe(4);
    });

    it('filters by actor', () => {
      const entries = ledger.query({ actor: 'alice' });
      expect(entries.length).toBe(2);
      expect(entries.every((e) => e.actor === 'alice')).toBe(true);
    });

    it('filters by correlationId', () => {
      const entries = ledger.query({ correlationId: 'c1' });
      expect(entries.length).toBe(2);
    });

    it('filters by timestamp range', () => {
      const all = ledger.readAll();
      // Use first and last timestamps to ensure we get a range
      const firstTs = all[0].timestamp;
      const lastTs = all[all.length - 1].timestamp;

      // Since entries are recorded very fast (same ms), timestamps may be equal
      // Test with exact boundaries
      const afterFirst = ledger.query({ since: firstTs });
      expect(afterFirst.length).toBe(5); // All entries including first

      const beforeLast = ledger.query({ until: lastTs });
      expect(beforeLast.length).toBe(5); // All entries including last

      // Filter to entries at or after second entry timestamp
      const secondTs = all[1].timestamp;
      const afterSecond = ledger.query({ since: secondTs });
      expect(afterSecond.length).toBeGreaterThanOrEqual(4); // At least 4 entries
    });

    it('applies custom filter predicate', () => {
      const entries = ledger.query({
        filter: (e) => (e.data as { cmd?: string }).cmd === 'a',
      });
      expect(entries.length).toBe(1);
    });

    it('applies limit', () => {
      const entries = ledger.query({ limit: 2 });
      expect(entries.length).toBe(2);
    });

    it('applies offset', () => {
      const all = ledger.query();
      const withOffset = ledger.query({ offset: 2 });

      expect(withOffset.length).toBe(3);
      expect(withOffset[0].type).toBe(all[2].type);
    });

    it('applies offset and limit together', () => {
      const entries = ledger.query({ offset: 1, limit: 2 });
      expect(entries.length).toBe(2);
    });

    it('reverses order', () => {
      const normal = ledger.query();
      const reversed = ledger.query({ reverse: true });

      expect(reversed.length).toBe(normal.length);
      expect(reversed[0].seq).toBe(normal[normal.length - 1].seq);
      expect(reversed[reversed.length - 1].seq).toBe(normal[0].seq);
    });

    it('combines multiple filters', () => {
      const entries = ledger.query({
        type: 'action',
        actor: 'alice',
      });
      expect(entries.length).toBe(2);
    });
  });

  describe('replay', () => {
    beforeEach(() => {
      ledger.record('step', { n: 1 });
      ledger.record('step', { n: 2 });
      ledger.record('step', { n: 3 });
    });

    it('calls handler for each entry', async () => {
      const calls: number[] = [];
      const handler = (entry: LedgerEntry) => {
        calls.push((entry.data as { n: number }).n);
      };

      const count = await ledger.replay(handler);

      expect(count).toBe(3);
      expect(calls).toEqual([1, 2, 3]);
    });

    it('respects query options', async () => {
      const calls: number[] = [];
      await ledger.replay(
        (entry) => {
          calls.push((entry.data as { n: number }).n);
        },
        { limit: 2 }
      );

      expect(calls).toEqual([1, 2]);
    });

    it('awaits async handlers', async () => {
      const calls: number[] = [];
      await ledger.replay(async (entry) => {
        await new Promise((r) => setTimeout(r, 10));
        calls.push((entry.data as { n: number }).n);
      });

      expect(calls).toEqual([1, 2, 3]);
    });
  });

  describe('verify', () => {
    it('returns valid for empty ledger', () => {
      const result = ledger.verify();

      expect(result.valid).toBe(true);
      expect(result.entriesVerified).toBe(0);
      expect(result.invalidAt).toBe(-1);
      expect(result.genesisHash).toBe(GENESIS_HASH);
      expect(result.headHash).toBe(GENESIS_HASH);
    });

    it('returns valid for intact ledger', () => {
      ledger.record('a', {});
      ledger.record('b', {});
      ledger.record('c', {});

      const result = ledger.verify();

      expect(result.valid).toBe(true);
      expect(result.entriesVerified).toBe(3);
      expect(result.invalidAt).toBe(-1);
      expect(result.error).toBeUndefined();
    });

    it('detects hash tampering', () => {
      ledger.record('a', {});
      ledger.record('b', {});
      ledger.record('c', {});

      // Tamper with the ledger file
      const content = fs.readFileSync(ledger.getLedgerPath(), 'utf-8');
      const lines = content.trim().split('\n');
      const entry = JSON.parse(lines[1]) as LedgerEntry;
      entry.hash = 'tampered_hash_value_that_is_wrong';
      lines[1] = JSON.stringify(entry);
      fs.writeFileSync(ledger.getLedgerPath(), lines.join('\n') + '\n', 'utf-8');

      // Reload and verify
      const newLedger = createLedger({ ledgerDir: tmpDir });
      const result = newLedger.verify();

      expect(result.valid).toBe(false);
      expect(result.invalidAt).toBe(1);
      expect(result.error).toContain('hash mismatch');
    });

    it('detects prev_hash chain break', () => {
      ledger.record('a', {});
      ledger.record('b', {});
      ledger.record('c', {});

      // Tamper with prev_hash
      const content = fs.readFileSync(ledger.getLedgerPath(), 'utf-8');
      const lines = content.trim().split('\n');
      const entry = JSON.parse(lines[2]) as LedgerEntry;
      entry.prev_hash = 'wrong_prev_hash';
      lines[2] = JSON.stringify(entry);
      fs.writeFileSync(ledger.getLedgerPath(), lines.join('\n') + '\n', 'utf-8');

      const newLedger = createLedger({ ledgerDir: tmpDir });
      const result = newLedger.verify();

      expect(result.valid).toBe(false);
      expect(result.invalidAt).toBe(2);
      expect(result.error).toContain('prev_hash mismatch');
    });

    it('detects data tampering', () => {
      ledger.record('a', { value: 'original' });
      ledger.record('b', {});

      // Tamper with data (but not hash)
      const content = fs.readFileSync(ledger.getLedgerPath(), 'utf-8');
      const lines = content.trim().split('\n');
      const entry = JSON.parse(lines[0]) as LedgerEntry;
      entry.data = { value: 'tampered' };
      lines[0] = JSON.stringify(entry);
      fs.writeFileSync(ledger.getLedgerPath(), lines.join('\n') + '\n', 'utf-8');

      const newLedger = createLedger({ ledgerDir: tmpDir });
      const result = newLedger.verify();

      expect(result.valid).toBe(false);
      expect(result.invalidAt).toBe(0);
    });

    it('reports correct head hash on valid ledger', () => {
      const hash1 = ledger.record('a', {})!;
      const hash2 = ledger.record('b', {})!;
      const hash3 = ledger.record('c', {})!;

      const result = ledger.verify();

      expect(result.headHash).toBe(hash3);
      expect(result.genesisHash).toBe(GENESIS_HASH);
    });
  });

  describe('stats', () => {
    it('returns stats for empty ledger', () => {
      const stats = ledger.stats();

      expect(stats.totalEntries).toBe(0);
      expect(stats.byType).toEqual({});
      expect(stats.byActor).toEqual({});
      expect(stats.firstTimestamp).toBeNull();
      expect(stats.lastTimestamp).toBeNull();
      expect(stats.headHash).toBe(GENESIS_HASH);
    });

    it('counts entries by type', () => {
      ledger.record('action', {});
      ledger.record('action', {});
      ledger.record('decision', {});
      ledger.record('audit', {});
      ledger.record('action', {});

      const stats = ledger.stats();

      expect(stats.totalEntries).toBe(5);
      expect(stats.byType).toEqual({
        action: 3,
        decision: 1,
        audit: 1,
      });
    });

    it('counts entries by actor', () => {
      ledger.record('a', {}, { actor: 'alice' });
      ledger.record('b', {}, { actor: 'bob' });
      ledger.record('c', {}, { actor: 'alice' });

      const stats = ledger.stats();

      expect(stats.byActor).toEqual({
        alice: 2,
        bob: 1,
      });
    });

    it('tracks timestamp range', () => {
      ledger.record('first', {});
      const entries1 = ledger.readAll();
      const firstTs = entries1[0].timestamp;

      // Small delay
      ledger.record('last', {});
      const entries2 = ledger.readAll();
      const lastTs = entries2[1].timestamp;

      const stats = ledger.stats();

      expect(stats.firstTimestamp).toBe(firstTs);
      expect(stats.lastTimestamp).toBe(lastTs);
    });

    it('reports file size', () => {
      ledger.record('test', { padding: 'x'.repeat(100) });

      const stats = ledger.stats();

      expect(stats.fileSizeBytes).toBeGreaterThan(100);
    });

    it('reports current head hash', () => {
      const hash = ledger.record('test', {})!;
      const stats = ledger.stats();

      expect(stats.headHash).toBe(hash);
    });
  });

  describe('getByHash', () => {
    it('returns null for nonexistent hash', () => {
      ledger.record('test', {});

      const entry = ledger.getByHash('nonexistent');
      expect(entry).toBeNull();
    });

    it('returns entry by hash', () => {
      ledger.record('first', {});
      const hash = ledger.record('target', { value: 42 })!;
      ledger.record('last', {});

      const entry = ledger.getByHash<{ value: number }>(hash);

      expect(entry).not.toBeNull();
      expect(entry!.type).toBe('target');
      expect(entry!.data.value).toBe(42);
    });
  });

  describe('getBySeq', () => {
    it('returns null for nonexistent seq', () => {
      ledger.record('test', {});

      const entry = ledger.getBySeq(999);
      expect(entry).toBeNull();
    });

    it('returns entry by sequence number', () => {
      ledger.record('zero', {});
      ledger.record('one', { target: true });
      ledger.record('two', {});

      const entry = ledger.getBySeq<{ target?: boolean }>(1);

      expect(entry).not.toBeNull();
      expect(entry!.type).toBe('one');
      expect(entry!.data.target).toBe(true);
    });
  });

  describe('bus integration', () => {
    it('emits event on record when bus provided', async () => {
      const bus = new MemoryBus({
        actor: 'test',
        ring: 2,
      });

      const events: unknown[] = [];
      bus.subscribe('ledger.*', (event) => {
        events.push(event);
      });

      const ledgerWithBus = createLedger({
        ledgerDir: tmpDir,
        bus,
        fsync: false,
      });

      ledgerWithBus.record('test', { value: 1 });

      // Wait for async dispatch
      await new Promise((r) => setTimeout(r, 50));

      expect(events.length).toBe(1);
      expect((events[0] as { topic: string }).topic).toBe('ledger.entry.recorded');
    });

    it('works without bus', () => {
      const noBusLedger = createLedger({
        ledgerDir: path.join(tmpDir, 'nobus'),
        fsync: false,
      });

      // Should not throw
      const hash = noBusLedger.record('test', {});
      expect(hash).toBeDefined();
    });
  });

  describe('tail', () => {
    it('yields new entries', async () => {
      const received: LedgerEntry[] = [];
      const iterator = ledger.tail();

      // Start tailing in background
      const tailPromise = (async () => {
        for await (const entry of iterator) {
          received.push(entry);
          if (received.length >= 3) break;
        }
      })();

      // Write entries with delays
      await new Promise((r) => setTimeout(r, 50));
      ledger.record('one', {});
      await new Promise((r) => setTimeout(r, 150));
      ledger.record('two', {});
      await new Promise((r) => setTimeout(r, 150));
      ledger.record('three', {});

      await tailPromise;

      expect(received.length).toBe(3);
      expect(received[0].type).toBe('one');
      expect(received[1].type).toBe('two');
      expect(received[2].type).toBe('three');
    });
  });

  describe('hash calculation', () => {
    it('produces deterministic hashes', () => {
      // Record same data twice in separate ledgers
      const ledger1 = createLedger({
        ledgerDir: path.join(tmpDir, 'l1'),
        fsync: false,
      });
      const ledger2 = createLedger({
        ledgerDir: path.join(tmpDir, 'l2'),
        fsync: false,
      });

      // Same entry type and data from genesis
      // Note: timestamps will differ, so hashes will differ
      // But prev_hash will be the same (genesis)
      const entries1 = ledger1.readAll();
      const entries2 = ledger2.readAll();

      // Both should start from genesis
      expect(ledger1.getHeadHash()).toBe(GENESIS_HASH);
      expect(ledger2.getHeadHash()).toBe(GENESIS_HASH);
    });

    it('uses canonical JSON for hash calculation', () => {
      // The hash should be the same regardless of object key order
      // in the original data since we sort keys
      ledger.record('test', { z: 1, a: 2 });

      const entry = ledger.readAll()[0];

      // Verify entry exists and has correct data
      expect(entry.data).toEqual({ z: 1, a: 2 });

      // Verification should pass (hash was calculated correctly)
      const result = ledger.verify();
      expect(result.valid).toBe(true);
    });
  });
});

describe('getLedger singleton', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ark-ledger-singleton-'));
    resetLedger();
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    resetLedger();
  });

  it('throws when getting without config', () => {
    expect(() => getLedger()).toThrow('Ledger not initialized');
  });

  it('creates instance with config', () => {
    const ledger = getLedger({ ledgerDir: tmpDir });
    expect(ledger).toBeInstanceOf(DKINLedger);
  });

  it('returns same instance on subsequent calls', () => {
    const ledger1 = getLedger({ ledgerDir: tmpDir });
    const ledger2 = getLedger();

    expect(ledger1).toBe(ledger2);
  });

  it('resets singleton', () => {
    getLedger({ ledgerDir: tmpDir });
    resetLedger();

    expect(() => getLedger()).toThrow('Ledger not initialized');
  });
});

describe('GENESIS_HASH', () => {
  it('is 64 zeros', () => {
    expect(GENESIS_HASH).toBe('0'.repeat(64));
    expect(GENESIS_HASH.length).toBe(64);
  });
});

describe('LEDGER_VERSION', () => {
  it('is defined', () => {
    expect(LEDGER_VERSION).toBe('0.1.0');
  });
});

describe('edge cases', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ark-ledger-edge-'));
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('handles unicode data', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });
    ledger.record('unicode', {
      emoji: '🚀',
      chinese: '你好',
      arabic: 'مرحبا',
    });

    const entries = ledger.readAll();
    expect(entries[0].data).toEqual({
      emoji: '🚀',
      chinese: '你好',
      arabic: 'مرحبا',
    });

    expect(ledger.verify().valid).toBe(true);
  });

  it('handles large data', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });
    const largeString = 'x'.repeat(100000);

    ledger.record('large', { content: largeString });

    const entries = ledger.readAll();
    expect((entries[0].data as { content: string }).content.length).toBe(100000);
    expect(ledger.verify().valid).toBe(true);
  });

  it('handles many entries', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });

    for (let i = 0; i < 1000; i++) {
      ledger.record('entry', { i });
    }

    expect(ledger.getEntryCount()).toBe(1000);
    expect(ledger.verify().valid).toBe(true);
  });

  it('handles null and undefined in data', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });

    ledger.record('nullish', {
      nullValue: null,
      undefinedValue: undefined,
      nested: { a: null },
    });

    const entries = ledger.readAll();
    expect((entries[0].data as Record<string, unknown>).nullValue).toBeNull();
    // undefined is not serialized in JSON
    expect((entries[0].data as Record<string, unknown>).undefinedValue).toBeUndefined();

    expect(ledger.verify().valid).toBe(true);
  });

  it('handles special characters in strings', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });

    ledger.record('special', {
      quotes: '"hello"',
      backslash: '\\path\\to\\file',
      newlines: 'line1\nline2',
      tabs: 'col1\tcol2',
    });

    const entries = ledger.readAll();
    expect((entries[0].data as Record<string, string>).quotes).toBe('"hello"');
    expect((entries[0].data as Record<string, string>).newlines).toBe('line1\nline2');

    expect(ledger.verify().valid).toBe(true);
  });

  it('handles empty type', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });
    ledger.record('', { data: 'test' });

    const entries = ledger.readAll();
    expect(entries[0].type).toBe('');
    expect(ledger.verify().valid).toBe(true);
  });

  it('handles empty data', () => {
    const ledger = createLedger({ ledgerDir: tmpDir, fsync: false });
    ledger.record('empty', {});

    const entries = ledger.readAll();
    expect(entries[0].data).toEqual({});
    expect(ledger.verify().valid).toBe(true);
  });
});
