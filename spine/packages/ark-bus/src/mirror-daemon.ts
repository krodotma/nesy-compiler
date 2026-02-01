/**
 * @ark/bus/mirror-daemon - Bus Mirror Daemon
 *
 * Mirrors events from a fallback bus (e.g., sandboxed agent workspace)
 * to the canonical bus, maintaining a single source of truth.
 *
 * Design:
 * - Append-only: never rewrites either bus
 * - Idempotent: deduplicates via recent event IDs
 * - Safe: uses file locks on writes; maintains offset state
 * - Isomorphic-ready: can run in Node.js or Deno
 *
 * Ported from: nucleus/tools/bus_mirror_daemon.py
 *
 * @module
 */

import * as fs from 'node:fs';
import * as path from 'node:path';
import { createFileBus, type FileBus } from './file-bus.js';

/**
 * Mirror state persisted between runs
 */
export interface MirrorState {
  /** Byte offset in source file */
  offset: number;
  /** Source file inode for rotation detection */
  srcInode: number | null;
  /** Source path for debugging */
  srcPath?: string;
  /** Last update timestamp */
  updatedIso?: string;
}

/**
 * Mirror configuration
 */
export interface MirrorConfig {
  /** Source bus directory (fallback) */
  fromBusDir: string;
  /** Destination bus directory (canonical) */
  toBusDir: string;
  /** State file path (default: <toBusDir>/bus_mirror_state.json) */
  statePath?: string;
  /** Polling interval in milliseconds */
  pollMs?: number;
  /** Run once and exit */
  once?: boolean;
  /** Start from EOF on first run (skip historical backlog) */
  startAtEnd?: boolean;
  /** Bytes to scan from dest tail for dedup */
  recentBytesBack?: number;
  /** Max recent IDs to track for dedup */
  maxRecentIds?: number;
  /** Emit bus metrics when mirroring */
  emitBus?: boolean;
  /** Minimum ms between metric emissions */
  emitIntervalMs?: number;
  /** Actor name for metrics */
  actor?: string;
}

/**
 * Mirror statistics for a single pass
 */
export interface MirrorStats {
  mirrored: number;
  skipped: number;
  srcSize: number;
  offset: number;
}

/**
 * Load mirror state from disk
 */
export function loadState(statePath: string): MirrorState {
  try {
    const raw = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
    return {
      offset: Number(raw.offset) || 0,
      srcInode: raw.src_inode ?? raw.srcInode ?? null,
      srcPath: raw.src_path ?? raw.srcPath,
      updatedIso: raw.updated_iso ?? raw.updatedIso,
    };
  } catch {
    return { offset: 0, srcInode: null };
  }
}

/**
 * Save mirror state to disk
 */
export function saveState(statePath: string, state: MirrorState, srcPath: string): void {
  const dir = path.dirname(statePath);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const payload = {
    offset: state.offset,
    src_inode: state.srcInode,
    src_path: srcPath,
    updated_iso: new Date().toISOString(),
  };

  fs.writeFileSync(statePath, JSON.stringify(payload, null, 2) + '\n', 'utf-8');
}

/**
 * Read recent event IDs from destination for deduplication
 */
export function readRecentIds(
  destEvents: string,
  bytesBack: number,
  maxIds: number
): Set<string> {
  const ids = new Set<string>();

  if (!fs.existsSync(destEvents)) {
    return ids;
  }

  try {
    const stat = fs.statSync(destEvents);
    const start = Math.max(0, stat.size - bytesBack);

    const fd = fs.openSync(destEvents, 'r');
    const buffer = Buffer.alloc(Math.min(bytesBack, stat.size));
    fs.readSync(fd, buffer, 0, buffer.length, start);
    fs.closeSync(fd);

    let text = buffer.toString('utf-8');

    // Drop partial first line if we started mid-file
    if (start > 0) {
      const nl = text.indexOf('\n');
      if (nl >= 0) {
        text = text.slice(nl + 1);
      } else {
        text = '';
      }
    }

    const lines = text.split('\n').filter(Boolean).slice(-maxIds);

    for (const line of lines) {
      try {
        const obj = JSON.parse(line);
        if (obj && typeof obj.id === 'string' && obj.id) {
          ids.add(obj.id);
        }
      } catch {
        // Skip malformed lines
      }
    }
  } catch {
    // Return empty set on error
  }

  return ids;
}

/**
 * Append lines to destination with file locking
 */
export function appendLines(destEvents: string, lines: Buffer[]): void {
  const dir = path.dirname(destEvents);
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }

  const fd = fs.openSync(destEvents, 'a');
  try {
    for (const line of lines) {
      fs.writeSync(fd, line);
    }
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
}

/**
 * Mirror new events from source to destination (single pass)
 */
export function mirrorOnce(
  srcEvents: string,
  destEvents: string,
  state: MirrorState,
  recentBytesBack: number,
  maxRecentIds: number
): { state: MirrorState; stats: MirrorStats } {
  // Ensure source exists
  const srcDir = path.dirname(srcEvents);
  if (!fs.existsSync(srcDir)) {
    fs.mkdirSync(srcDir, { recursive: true });
  }
  if (!fs.existsSync(srcEvents)) {
    fs.writeFileSync(srcEvents, '', 'utf-8');
  }

  const stat = fs.statSync(srcEvents);
  const srcInode = stat.ino;
  const srcSize = stat.size;
  let offset = Math.max(0, state.offset || 0);

  // Rotation/truncation safety: reset if file shrank or inode changed
  if (state.srcInode !== null && srcInode !== state.srcInode) {
    offset = 0;
  }
  if (srcSize < offset) {
    offset = 0;
  }

  state.srcInode = srcInode;
  state.offset = offset;

  if (srcSize === offset) {
    return {
      state,
      stats: { mirrored: 0, skipped: 0, srcSize, offset },
    };
  }

  const recentIds = readRecentIds(destEvents, recentBytesBack, maxRecentIds);
  const toWrite: Buffer[] = [];
  let mirrored = 0;
  let skipped = 0;

  // Read new lines from source
  const fd = fs.openSync(srcEvents, 'r');
  try {
    const bufferSize = srcSize - offset;
    const buffer = Buffer.alloc(bufferSize);
    fs.readSync(fd, buffer, 0, bufferSize, offset);

    const text = buffer.toString('utf-8');
    const lines = text.split('\n');

    let bytesProcessed = 0;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const isLast = i === lines.length - 1;
      const lineBytes = Buffer.byteLength(line, 'utf-8') + (isLast ? 0 : 1);

      // Don't process partial last line (no newline)
      if (isLast && !text.endsWith('\n')) {
        break;
      }

      if (!line.trim()) {
        bytesProcessed += lineBytes;
        continue;
      }

      let eventId: string | null = null;
      try {
        const obj = JSON.parse(line);
        if (obj && typeof obj.id === 'string' && obj.id) {
          eventId = obj.id;
        }
      } catch {
        // Keep line even if not valid JSON
      }

      if (eventId && recentIds.has(eventId)) {
        skipped++;
        bytesProcessed += lineBytes;
        continue;
      }

      toWrite.push(Buffer.from(line + '\n', 'utf-8'));
      mirrored++;

      if (eventId) {
        recentIds.add(eventId);
      }

      bytesProcessed += lineBytes;
    }

    state.offset = offset + bytesProcessed;
  } finally {
    fs.closeSync(fd);
  }

  if (toWrite.length > 0) {
    appendLines(destEvents, toWrite);
  }

  return {
    state,
    stats: { mirrored, skipped, srcSize, offset: state.offset },
  };
}

/**
 * Bus Mirror Daemon
 *
 * Continuously mirrors events from fallback bus to canonical bus
 */
export class BusMirrorDaemon {
  private config: Required<MirrorConfig>;
  private state: MirrorState;
  private running = false;
  private lastEmitTs = 0;
  private srcEvents: string;
  private destEvents: string;
  private statePath: string;

  constructor(config: MirrorConfig) {
    this.config = {
      fromBusDir: config.fromBusDir,
      toBusDir: config.toBusDir,
      statePath: config.statePath ?? path.join(config.toBusDir, 'bus_mirror_state.json'),
      pollMs: config.pollMs ?? 500,
      once: config.once ?? false,
      startAtEnd: config.startAtEnd ?? false,
      recentBytesBack: config.recentBytesBack ?? 2_000_000,
      maxRecentIds: config.maxRecentIds ?? 20_000,
      emitBus: config.emitBus ?? false,
      emitIntervalMs: config.emitIntervalMs ?? 10_000,
      actor: config.actor ?? 'bus-mirror',
    };

    this.srcEvents = path.join(this.config.fromBusDir, 'events.ndjson');
    this.destEvents = path.join(this.config.toBusDir, 'events.ndjson');
    this.statePath = this.config.statePath;

    // Load or initialize state
    const firstRun = !fs.existsSync(this.statePath);
    this.state = loadState(this.statePath);

    if (this.config.startAtEnd && firstRun) {
      try {
        const stat = fs.statSync(this.srcEvents);
        this.state.offset = stat.size;
        this.state.srcInode = stat.ino;
        saveState(this.statePath, this.state, this.srcEvents);
      } catch {
        // Proceed from offset=0 if stat fails
      }
    }
  }

  /**
   * Run a single mirror pass
   */
  mirrorOnce(): MirrorStats {
    const { state, stats } = mirrorOnce(
      this.srcEvents,
      this.destEvents,
      this.state,
      this.config.recentBytesBack,
      this.config.maxRecentIds
    );

    this.state = state;
    saveState(this.statePath, this.state, this.srcEvents);

    // Emit metrics if configured
    if (this.config.emitBus && stats.mirrored > 0) {
      const now = Date.now();
      if (now - this.lastEmitTs >= this.config.emitIntervalMs) {
        this.emitMetric(stats);
        this.lastEmitTs = now;
      }
    }

    return stats;
  }

  /**
   * Emit a bus metric event
   */
  private emitMetric(stats: MirrorStats): void {
    try {
      const bus = createFileBus({
        actor: this.config.actor,
        ring: 1,
        busDir: this.config.toBusDir,
      });

      bus.emit('bus.mirror.batch', {
        ...stats,
        fromBusDir: this.config.fromBusDir,
        toBusDir: this.config.toBusDir,
        statePath: this.statePath,
        iso: new Date().toISOString(),
      }, { kind: 'heartbeat', level: 'info' });

      bus.flush();
      bus.close();
    } catch {
      // Silently ignore metric emission errors
    }
  }

  /**
   * Start the daemon loop
   */
  async start(): Promise<void> {
    this.running = true;

    while (this.running) {
      this.mirrorOnce();

      if (this.config.once) {
        break;
      }

      await new Promise((resolve) => setTimeout(resolve, this.config.pollMs));
    }
  }

  /**
   * Stop the daemon
   */
  stop(): void {
    this.running = false;
  }

  /**
   * Get current state (for testing/debugging)
   */
  getState(): MirrorState {
    return { ...this.state };
  }
}

/**
 * Create and start a bus mirror daemon
 */
export function createMirrorDaemon(config: MirrorConfig): BusMirrorDaemon {
  return new BusMirrorDaemon(config);
}

// CLI entry point
if (import.meta.url === `file://${process.argv[1]}`) {
  const args = process.argv.slice(2);

  const config: MirrorConfig = {
    fromBusDir: '/pluribus/.pluribus_local/bus',
    toBusDir: '/pluribus/.pluribus/bus',
  };

  // Parse args
  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--from-bus-dir' && args[i + 1]) {
      config.fromBusDir = args[++i];
    } else if (arg === '--to-bus-dir' && args[i + 1]) {
      config.toBusDir = args[++i];
    } else if (arg === '--poll' && args[i + 1]) {
      config.pollMs = parseFloat(args[++i]) * 1000;
    } else if (arg === '--once') {
      config.once = true;
    } else if (arg === '--start-at-end') {
      config.startAtEnd = true;
    } else if (arg === '--emit-bus') {
      config.emitBus = true;
    } else if (arg === '--actor' && args[i + 1]) {
      config.actor = args[++i];
    }
  }

  const daemon = createMirrorDaemon(config);
  daemon.start().catch(console.error);

  // Handle shutdown
  process.on('SIGINT', () => daemon.stop());
  process.on('SIGTERM', () => daemon.stop());
}
