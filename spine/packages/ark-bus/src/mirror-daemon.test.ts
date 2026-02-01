/**
 * @ark/bus/mirror-daemon tests
 *
 * Tests for bus mirror daemon functionality
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import {
  loadState,
  saveState,
  readRecentIds,
  appendLines,
  mirrorOnce,
  BusMirrorDaemon,
  createMirrorDaemon,
  type MirrorState,
} from './mirror-daemon.js';

describe('mirror-daemon', () => {
  let tmpDir: string;
  let srcBusDir: string;
  let destBusDir: string;
  let srcEvents: string;
  let destEvents: string;
  let statePath: string;

  beforeEach(() => {
    // Create temp directories
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ark-bus-test-'));
    srcBusDir = path.join(tmpDir, 'src-bus');
    destBusDir = path.join(tmpDir, 'dest-bus');
    fs.mkdirSync(srcBusDir, { recursive: true });
    fs.mkdirSync(destBusDir, { recursive: true });

    srcEvents = path.join(srcBusDir, 'events.ndjson');
    destEvents = path.join(destBusDir, 'events.ndjson');
    statePath = path.join(destBusDir, 'bus_mirror_state.json');
  });

  afterEach(() => {
    // Cleanup
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  describe('loadState / saveState', () => {
    it('returns default state when file does not exist', () => {
      const state = loadState('/nonexistent/path.json');
      expect(state.offset).toBe(0);
      expect(state.srcInode).toBeNull();
    });

    it('saves and loads state correctly', () => {
      const state: MirrorState = {
        offset: 12345,
        srcInode: 67890,
      };

      saveState(statePath, state, srcEvents);
      const loaded = loadState(statePath);

      expect(loaded.offset).toBe(12345);
      expect(loaded.srcInode).toBe(67890);
    });

    it('creates parent directories if needed', () => {
      const nestedPath = path.join(tmpDir, 'a', 'b', 'c', 'state.json');
      const state: MirrorState = { offset: 100, srcInode: null };

      saveState(nestedPath, state, srcEvents);

      expect(fs.existsSync(nestedPath)).toBe(true);
    });
  });

  describe('readRecentIds', () => {
    it('returns empty set for nonexistent file', () => {
      const ids = readRecentIds('/nonexistent/events.ndjson', 1000, 100);
      expect(ids.size).toBe(0);
    });

    it('extracts event IDs from NDJSON', () => {
      const events = [
        { id: 'evt-1', topic: 'test', data: {} },
        { id: 'evt-2', topic: 'test', data: {} },
        { id: 'evt-3', topic: 'test', data: {} },
      ];

      fs.writeFileSync(
        destEvents,
        events.map((e) => JSON.stringify(e)).join('\n') + '\n',
        'utf-8'
      );

      const ids = readRecentIds(destEvents, 10000, 100);

      expect(ids.has('evt-1')).toBe(true);
      expect(ids.has('evt-2')).toBe(true);
      expect(ids.has('evt-3')).toBe(true);
      expect(ids.size).toBe(3);
    });

    it('respects maxIds limit', () => {
      const events = Array.from({ length: 10 }, (_, i) => ({
        id: `evt-${i}`,
        topic: 'test',
        data: {},
      }));

      fs.writeFileSync(
        destEvents,
        events.map((e) => JSON.stringify(e)).join('\n') + '\n',
        'utf-8'
      );

      const ids = readRecentIds(destEvents, 10000, 3);

      // Should only have last 3 IDs
      expect(ids.size).toBe(3);
      expect(ids.has('evt-7')).toBe(true);
      expect(ids.has('evt-8')).toBe(true);
      expect(ids.has('evt-9')).toBe(true);
    });

    it('handles malformed JSON lines gracefully', () => {
      fs.writeFileSync(
        destEvents,
        '{"id": "good-1"}\nnot json\n{"id": "good-2"}\n',
        'utf-8'
      );

      const ids = readRecentIds(destEvents, 10000, 100);

      expect(ids.has('good-1')).toBe(true);
      expect(ids.has('good-2')).toBe(true);
      expect(ids.size).toBe(2);
    });
  });

  describe('appendLines', () => {
    it('creates file if it does not exist', () => {
      appendLines(destEvents, [Buffer.from('line1\n')]);
      expect(fs.existsSync(destEvents)).toBe(true);
    });

    it('appends multiple lines', () => {
      appendLines(destEvents, [
        Buffer.from('line1\n'),
        Buffer.from('line2\n'),
      ]);

      const content = fs.readFileSync(destEvents, 'utf-8');
      expect(content).toBe('line1\nline2\n');
    });

    it('appends to existing content', () => {
      fs.writeFileSync(destEvents, 'existing\n', 'utf-8');
      appendLines(destEvents, [Buffer.from('new\n')]);

      const content = fs.readFileSync(destEvents, 'utf-8');
      expect(content).toBe('existing\nnew\n');
    });
  });

  describe('mirrorOnce', () => {
    it('creates source file if it does not exist', () => {
      const state: MirrorState = { offset: 0, srcInode: null };
      mirrorOnce(srcEvents, destEvents, state, 1000, 100);
      expect(fs.existsSync(srcEvents)).toBe(true);
    });

    it('mirrors new events from source to destination', () => {
      const events = [
        { id: 'evt-1', topic: 'test', data: { value: 1 } },
        { id: 'evt-2', topic: 'test', data: { value: 2 } },
      ];

      fs.writeFileSync(
        srcEvents,
        events.map((e) => JSON.stringify(e)).join('\n') + '\n',
        'utf-8'
      );

      const state: MirrorState = { offset: 0, srcInode: null };
      const { stats } = mirrorOnce(srcEvents, destEvents, state, 1000, 100);

      expect(stats.mirrored).toBe(2);
      expect(stats.skipped).toBe(0);

      const destContent = fs.readFileSync(destEvents, 'utf-8');
      expect(destContent).toContain('evt-1');
      expect(destContent).toContain('evt-2');
    });

    it('deduplicates events already in destination', () => {
      // Destination already has evt-1
      fs.writeFileSync(
        destEvents,
        JSON.stringify({ id: 'evt-1', topic: 'test' }) + '\n',
        'utf-8'
      );

      // Source has evt-1 and evt-2
      fs.writeFileSync(
        srcEvents,
        [
          JSON.stringify({ id: 'evt-1', topic: 'test' }),
          JSON.stringify({ id: 'evt-2', topic: 'test' }),
        ].join('\n') + '\n',
        'utf-8'
      );

      const state: MirrorState = { offset: 0, srcInode: null };
      const { stats } = mirrorOnce(srcEvents, destEvents, state, 1000, 100);

      expect(stats.mirrored).toBe(1); // Only evt-2
      expect(stats.skipped).toBe(1); // evt-1 was skipped
    });

    it('updates offset after mirroring', () => {
      fs.writeFileSync(
        srcEvents,
        JSON.stringify({ id: 'evt-1', topic: 'test' }) + '\n',
        'utf-8'
      );

      const state: MirrorState = { offset: 0, srcInode: null };
      const { state: newState } = mirrorOnce(srcEvents, destEvents, state, 1000, 100);

      expect(newState.offset).toBeGreaterThan(0);
    });

    it('handles incremental mirroring correctly', () => {
      // First batch
      fs.writeFileSync(
        srcEvents,
        JSON.stringify({ id: 'evt-1' }) + '\n',
        'utf-8'
      );

      let state: MirrorState = { offset: 0, srcInode: null };
      const result1 = mirrorOnce(srcEvents, destEvents, state, 1000, 100);
      state = result1.state;

      expect(result1.stats.mirrored).toBe(1);

      // Append more to source
      fs.appendFileSync(
        srcEvents,
        JSON.stringify({ id: 'evt-2' }) + '\n',
        'utf-8'
      );

      // Mirror again
      const result2 = mirrorOnce(srcEvents, destEvents, state, 1000, 100);

      expect(result2.stats.mirrored).toBe(1); // Only evt-2
    });

    it('resets offset when source file is truncated', () => {
      fs.writeFileSync(srcEvents, 'x'.repeat(1000) + '\n', 'utf-8');

      const state: MirrorState = {
        offset: 500,
        srcInode: fs.statSync(srcEvents).ino,
      };

      // Truncate source
      fs.writeFileSync(srcEvents, 'short\n', 'utf-8');

      const { state: newState } = mirrorOnce(srcEvents, destEvents, state, 1000, 100);

      // Offset should be reset (file is smaller than offset)
      expect(newState.offset).toBeLessThanOrEqual(6); // "short\n" is 6 bytes
    });
  });

  describe('BusMirrorDaemon', () => {
    it('creates daemon with default config', () => {
      const daemon = createMirrorDaemon({
        fromBusDir: srcBusDir,
        toBusDir: destBusDir,
      });

      expect(daemon).toBeInstanceOf(BusMirrorDaemon);
    });

    it('mirrorOnce returns stats', () => {
      fs.writeFileSync(
        srcEvents,
        JSON.stringify({ id: 'test-evt', topic: 'test' }) + '\n',
        'utf-8'
      );

      const daemon = createMirrorDaemon({
        fromBusDir: srcBusDir,
        toBusDir: destBusDir,
      });

      const stats = daemon.mirrorOnce();

      expect(stats.mirrored).toBe(1);
    });

    it('respects startAtEnd option', () => {
      // Write some events before daemon starts
      fs.writeFileSync(
        srcEvents,
        JSON.stringify({ id: 'old-evt', topic: 'test' }) + '\n',
        'utf-8'
      );

      const daemon = createMirrorDaemon({
        fromBusDir: srcBusDir,
        toBusDir: destBusDir,
        startAtEnd: true,
      });

      const stats = daemon.mirrorOnce();

      // Should not mirror old events
      expect(stats.mirrored).toBe(0);

      // New events should still be mirrored
      fs.appendFileSync(
        srcEvents,
        JSON.stringify({ id: 'new-evt', topic: 'test' }) + '\n',
        'utf-8'
      );

      const stats2 = daemon.mirrorOnce();
      expect(stats2.mirrored).toBe(1);
    });

    it('getState returns current state', () => {
      const daemon = createMirrorDaemon({
        fromBusDir: srcBusDir,
        toBusDir: destBusDir,
      });

      const state = daemon.getState();

      expect(state).toHaveProperty('offset');
      expect(state).toHaveProperty('srcInode');
    });

    it('stop() stops the daemon', async () => {
      const daemon = createMirrorDaemon({
        fromBusDir: srcBusDir,
        toBusDir: destBusDir,
        pollMs: 50,
      });

      // Start in background
      const startPromise = daemon.start();

      // Stop after a short delay
      setTimeout(() => daemon.stop(), 100);

      // Should resolve without hanging
      await startPromise;
    });
  });
});
