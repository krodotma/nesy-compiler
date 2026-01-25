/**
 * Bus Bridge Server
 *
 * Node.js server that bridges NDJSON bus files to WebSocket connections.
 * Allows web and native clients to connect to the bus.
 */

import type { BusEvent } from '../state/types';
import type { WebSocket as WsWebSocket } from 'ws';

import { decodeSkyEnvelope } from '../sky/codec.node';
import { SKY_MAGIC_V1, SKY_VERSION_V1 } from '../sky/constants';
import { skyEnvelopeToBusSummary } from '../sky/summary';
import { fileURLToPath } from 'url';
import * as path from 'path';

// ESM compatibility for __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export function resolvePluribusRoot(env: NodeJS.ProcessEnv): string {
  const root = (env.PLURIBUS_ROOT || '/pluribus').trim();
  return root || '/pluribus';
}

export function resolveBusDir(busPath: string, pluribusRoot: string): string {
  const p = (busPath || '').trim() || '.pluribus/bus';
  return path.isAbsolute(p) ? p : path.join(pluribusRoot, p);
}

export function ensurePathHasDirs(currentPath: string | undefined, extraDirs: string[]): string {
  const existing = (currentPath || '').split(path.delimiter).filter(Boolean);
  const out: string[] = [];
  const seen = new Set<string>();
  for (const dir of [...extraDirs, ...existing]) {
    const d = String(dir || '').trim();
    if (!d) continue;
    if (seen.has(d)) continue;
    seen.add(d);
    out.push(d);
  }
  return out.join(path.delimiter);
}

function truthyEnv(value: string | undefined, defaultValue: boolean): boolean {
  if (value === undefined) return defaultValue;
  return ['1', 'true', 'yes', 'on'].includes(value.trim().toLowerCase());
}

export function ndjsonReadAllowed(env: NodeJS.ProcessEnv): boolean {
  const explicit = env.PLURIBUS_NDJSON_READ;
  if (explicit !== undefined) {
    return truthyEnv(explicit, false);
  }
  const mode = (env.PLURIBUS_NDJSON_MODE || '').trim().toLowerCase();
  if (!mode || ['allow', 'enabled', 'on'].includes(mode)) return true;
  if (['dr', 'disaster', 'recovery'].includes(mode)) {
    return truthyEnv(env.PLURIBUS_DR_MODE, false);
  }
  if (['off', 'disabled', 'deny', 'no'].includes(mode)) return false;
  return true;
}

export function resolveGraphApiUrl(env: NodeJS.ProcessEnv): string {
  const raw = (env.BUS_BRIDGE_GRAPH_API || env.FALKORDB_API_URL || 'http://127.0.0.1:8765').trim();
  return raw || 'http://127.0.0.1:8765';
}

export interface IngestCacheEntry {
  asset_id: string;
  filename: string;
  content_type: string;
  byte_size: number;
  stored_at: string;
  expires_at: string;
  asset_path: string;
  meta_path: string;
}

interface GraphTimelineEvent {
  id?: string;
  topic?: string;
  actor?: string;
  ts?: number;
  trace_id?: string;
}

const DEFAULT_INGEST_TTL_MS = 24 * 60 * 60 * 1000;

export function resolveIngestCacheDir(pluribusRoot: string): string {
  return path.join(pluribusRoot, '.pluribus', 'ingest_cache');
}

export function sanitizeAssetId(value: string): string {
  const cleaned = String(value || '').replace(/[^a-zA-Z0-9_-]/g, '');
  if (cleaned) return cleaned;
  return `asset_${Date.now()}`;
}

const generateFallbackId = (): string => `asset_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

const MIME_EXTENSION_MAP: Record<string, string> = {
  'image/jpeg': '.jpg',
  'image/png': '.png',
  'image/webp': '.webp',
  'image/gif': '.gif',
  'application/pdf': '.pdf',
  'text/plain': '.txt',
};

const normalizeContentType = (value: string | undefined): string => {
  if (!value) return 'application/octet-stream';
  return value.split(';')[0]?.trim() || 'application/octet-stream';
};

const resolveSafeExtension = (filename: string, contentType: string): string => {
  const fromName = filename ? path.extname(filename).toLowerCase() : '';
  const fromMime = MIME_EXTENSION_MAP[contentType] || '';
  const candidate = fromName || fromMime || '.bin';
  if (/^\.[a-z0-9]+$/.test(candidate)) return candidate;
  return '.bin';
};

export async function writeIngestAsset(
  fs: typeof import('fs/promises'),
  options: {
    root: string;
    assetId: string;
    data: Buffer;
    filename?: string;
    contentType?: string;
    createdAtMs?: number;
    ttlMs?: number;
  },
): Promise<IngestCacheEntry> {
  const cacheDir = resolveIngestCacheDir(options.root);
  const safeId = sanitizeAssetId(options.assetId);
  const rawName = options.filename ? path.basename(options.filename) : '';
  const safeName = rawName.replace(/[^\w.\-]/g, '_');
  const contentType = normalizeContentType(options.contentType);
  const ext = resolveSafeExtension(safeName, contentType);
  const nowMs = Number.isFinite(options.createdAtMs) ? options.createdAtMs! : Date.now();
  const ttlMs = Number.isFinite(options.ttlMs) ? options.ttlMs! : DEFAULT_INGEST_TTL_MS;
  const storedAt = new Date(nowMs).toISOString();
  const expiresAt = new Date(nowMs + ttlMs).toISOString();

  await fs.mkdir(cacheDir, { recursive: true });
  const assetPath = path.join(cacheDir, `${safeId}${ext}`);
  const metaPath = path.join(cacheDir, `${safeId}.json`);
  const filename = safeName || `${safeId}${ext}`;

  await fs.writeFile(assetPath, options.data);
  const meta = {
    asset_id: safeId,
    filename,
    content_type: contentType,
    byte_size: options.data.length,
    stored_at: storedAt,
    expires_at: expiresAt,
    asset_path: assetPath,
  };
  await fs.writeFile(metaPath, JSON.stringify(meta, null, 2));

  return { ...meta, meta_path: metaPath };
}

export async function pruneExpiredIngestCache(
  fs: typeof import('fs/promises'),
  cacheDir: string,
  nowMs = Date.now(),
): Promise<number> {
  let entries: string[] = [];
  try {
    entries = await fs.readdir(cacheDir);
  } catch {
    return 0;
  }

  let removed = 0;
  for (const entry of entries) {
    if (!entry.endsWith('.json')) continue;
    const metaPath = path.join(cacheDir, entry);
    try {
      const raw = await fs.readFile(metaPath, 'utf-8');
      const meta = JSON.parse(raw) as { expires_at?: string; asset_path?: string };
      const expiresMs = Date.parse(meta.expires_at || '');
      if (!Number.isFinite(expiresMs) || expiresMs > nowMs) continue;
      if (meta.asset_path) {
        await fs.unlink(meta.asset_path).catch(() => {});
      }
      await fs.unlink(metaPath).catch(() => {});
      removed += 1;
    } catch {
      // ignore malformed metadata
    }
  }
  return removed;
}

export function splitNdjsonChunk(chunk: string, carry: string): { lines: string[]; carry: string } {
  const text = `${carry}${chunk}`;
  const parts = text.split('\n');
  const nextCarry = parts.pop() ?? '';
  const lines = parts.filter((line) => line.trim().length > 0);
  return { lines, carry: nextCarry };
}

export interface BusBridgeOptions {
  busPath: string;
  wsPort: number;
  apiPort?: number;
}

export interface BusBridgeHandle {
  close: () => Promise<void>;
}

/**
 * Bus Bridge - runs as a Node.js server
 *
 * Usage:
 *   node -e "require('./bus-bridge').startBridge({ busPath: '.pluribus/bus', wsPort: 9200 })"
 */
export async function startBridge(options: BusBridgeOptions): Promise<BusBridgeHandle> {
  const { WebSocketServer } = await import('ws');
  const fs = await import('fs/promises');
  const http = await import('http');
  const chokidar = await import('chokidar');

  const { busPath, wsPort, apiPort } = options;
  const pluribusRoot = resolvePluribusRoot(process.env);
  const busDir = resolveBusDir(busPath, pluribusRoot);
  const eventsPath = path.join(busDir, 'events.ndjson');
  const ingestCacheDir = resolveIngestCacheDir(pluribusRoot);
  const ingestTtlMs = Math.max(
    60 * 1000,
    Number.parseInt(process.env.PORTAL_INGEST_TTL_MS || '', 10) || DEFAULT_INGEST_TTL_MS,
  );
  const maxIngestBytes = Math.max(
    1024 * 1024,
    Number.parseInt(process.env.PORTAL_INGEST_MAX_BYTES || '', 10) || 25 * 1024 * 1024,
  );
  const pendingPortalAssets = new Map<string, { sessionId?: string; requestedAt: number }>();
  const legacyTailEnabled = process.env.BUS_BRIDGE_LEGACY_TAIL === '1';
  const ndjsonReadEnabled = ndjsonReadAllowed(process.env) || legacyTailEnabled;
  const graphApiUrl = resolveGraphApiUrl(process.env);
  const graphPollIntervalMs = Math.max(
    1000,
    Number.parseInt(process.env.BUS_BRIDGE_GRAPH_POLL_MS || '3000', 10) || 3000,
  );
  const graphSeedLimit = Math.max(
    200,
    Number.parseInt(process.env.BUS_BRIDGE_GRAPH_SEED_LIMIT || '1200', 10) || 1200,
  );
  const seedFromArchive = process.env.BUS_BRIDGE_RECOVERY_SEED === '1' || legacyTailEnabled;
  const archiveNdjson = process.env.BUS_BRIDGE_ARCHIVE_NDJSON !== '0' && !legacyTailEnabled;
  const maxBuffer = Math.max(
    1000,
    Number.parseInt(process.env.BUS_BRIDGE_MAX_BUFFER || '6000', 10) || 6000,
  );
  const syncLimit = Math.max(
    100,
    Number.parseInt(process.env.BUS_BRIDGE_SYNC_LIMIT || '600', 10) || 600,
  );

  // Ensure busPath exists
  await fs.mkdir(busDir, { recursive: true });

  // Track connected clients and their subscriptions
  const clients = new Map<WsWebSocket, Set<string>>();
  const skyClients = new Map<string, WsWebSocket>();
  const lastByTopic = new Map<string, BusEvent>();
  const eventBuffer: BusEvent[] = [];

  // Create WebSocket server
  const wss = new WebSocketServer({ port: wsPort });
  let apiServer: import('http').Server | null = null;
  let watcher: import('chokidar').FSWatcher | null = null;

  console.log(`[bus-bridge] WebSocket server listening on ws://localhost:${wsPort}`);

  const readRequestBody = async (
    req: import('http').IncomingMessage,
    maxBytes: number,
  ): Promise<Buffer> => {
    return await new Promise((resolve, reject) => {
      const chunks: Buffer[] = [];
      let size = 0;
      req.on('data', (chunk) => {
        const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
        size += buf.length;
        if (size > maxBytes) {
          reject(new Error('payload too large'));
          req.destroy();
          return;
        }
        chunks.push(buf);
      });
      req.on('end', () => resolve(Buffer.concat(chunks, size)));
      req.on('error', reject);
    });
  };

  async function handlePortalAssetEvents(event: BusEvent): Promise<void> {
    if (!event || typeof event.topic !== 'string') return;
    if (event.topic === 'portal.ingest.asset_staged') {
      const data = (event.data || {}) as Record<string, unknown>;
      const assetId = typeof data.asset_id === 'string' ? data.asset_id : '';
      if (!assetId) return;
      const expiresMs = Date.parse(String(data.expires_at || ''));
      if (Number.isFinite(expiresMs) && expiresMs <= Date.now()) return;

      const prior = pendingPortalAssets.get(assetId);
      if (prior && Date.now() - prior.requestedAt < 10_000) return;

      const sessionId = typeof data.session_id === 'string' ? data.session_id : undefined;
      pendingPortalAssets.set(assetId, { sessionId, requestedAt: Date.now() });

      await publishEvent({
        topic: 'portal.ingest.asset_pull',
        kind: 'request',
        level: 'info',
        actor: 'bus-bridge',
        data: {
          asset_id: assetId,
          session_id: sessionId,
          upload_url: '/api/portal/assets',
          requested_at: new Date().toISOString(),
          expires_at: data.expires_at,
        },
      });
      return;
    }

    if (event.topic === 'portal.ingest.asset_cached' || event.topic === 'portal.ingest.asset_upload_failed') {
      const data = (event.data || {}) as Record<string, unknown>;
      const assetId = typeof data.asset_id === 'string' ? data.asset_id : '';
      if (assetId) pendingPortalAssets.delete(assetId);
    }
  }

  function ingestEvent(event: BusEvent): void {
    if (!event || typeof event.topic !== 'string') return;
    lastByTopic.set(event.topic, event);
    void handlePortalAssetEvents(event);
  }

  wss.on('connection', async (ws: WsWebSocket, req) => {
    // Handle Terminal Connections
    if (req.url?.startsWith('/terminal')) {
      const url = new URL(req.url, `http://localhost:${wsPort}`);
      const pin = url.searchParams.get('pin');
      
      if (pin !== '0912') {
        ws.send(JSON.stringify({ type: 'error', message: 'Invalid PIN' }));
        ws.close();
        return;
      }

      console.log('[bus-bridge] Terminal connection authenticated');
      
      // Spawn PTY Bridge
      const { spawn } = await import('child_process');
      const toolsDir = path.join(pluribusRoot, 'nucleus', 'tools');
      const tmuxSession = 'pluribus_dash';
      const ptyScript = path.join(toolsDir, 'tmux_pty_bridge.py');
      const ensureScript = path.join(toolsDir, 'ensure_tmux.sh');
      const enrichedEnv = {
        ...process.env,
        PLURIBUS_ROOT: pluribusRoot,
        PLURIBUS_BUS_DIR: busDir,
        PATH: ensurePathHasDirs(process.env.PATH, ['/usr/local/bin', '/opt/homebrew/bin', '/usr/bin', '/bin']),
      };
      
      const ptyProc = spawn('python3', [ptyScript, '--session', tmuxSession, '--ensure', ensureScript], {
        stdio: ['pipe', 'pipe', 'inherit'], // pipe stdin/stdout, inherit stderr
        cwd: pluribusRoot,
        env: enrichedEnv,
      });

      // Bridge PTY -> WebSocket
      ptyProc.stdout.on('data', (data) => {
        if (ws.readyState === 1) {
          ws.send(data); // Send raw binary/text
        }
      });

      // Bridge WebSocket -> PTY
      ws.on('message', (data) => {
        if (!ptyProc.killed) {
          // Handle resize messages (from xterm FitAddon)
          try {
            const msg = JSON.parse(data.toString());
            if (msg?.type === 'resize') {
              const cols = Number(msg.cols);
              const rows = Number(msg.rows);
              if (Number.isFinite(cols) && Number.isFinite(rows) && cols > 0 && rows > 0) {
                // Resize the controlling PTY running `tmux attach-session`.
                // Without this, tmux continues to render at the PTY's initial size (often 80x24),
                // even if the xterm.js canvas is much larger.
                try {
                  const frame = `\0PLURIBUS:resize ${Math.floor(cols)} ${Math.floor(rows)}\n`;
                  ptyProc.stdin.write(Buffer.from(frame, 'utf8'));
                } catch {
                  // Ignore resize errors
                }
                return;
              }
            }
          } catch {
            // Not JSON, forward as input
          }
          ptyProc.stdin.write(data);
        }
      });

      ws.on('close', () => {
        ptyProc.kill();
      });

      ptyProc.on('exit', () => {
        ws.close();
      });

      return;
    }

    // Handle PluriChat Terminal Connections (persistent tmux session)
    if (req.url?.startsWith('/plurichat')) {
      const url = new URL(req.url, `http://localhost:${wsPort}`);
      const pin = url.searchParams.get('pin');

      if (pin !== '0912') {
        ws.send(JSON.stringify({ type: 'error', message: 'Invalid PIN' }));
        ws.close();
        return;
      }

      console.log('[bus-bridge] PluriChat connection authenticated');

      const { spawn, spawnSync } = await import('child_process');
      const tmuxSession = 'plurichat';
      const toolsDir = path.join(pluribusRoot, 'nucleus', 'tools');
      const ptyScript = path.join(toolsDir, 'tmux_pty_bridge.py');
      const enrichedEnv = {
        ...process.env,
        PLURIBUS_ROOT: pluribusRoot,
        PLURIBUS_BUS_DIR: busDir,
        PATH: ensurePathHasDirs(process.env.PATH, ['/usr/local/bin', '/opt/homebrew/bin', '/usr/bin', '/bin']),
      };

      // Ensure tmux session exists with plurichat running
      try {
        const has = spawnSync('tmux', ['has-session', '-t', tmuxSession], { stdio: 'ignore' });
        if (has.status === 0) {
          console.log(`[bus-bridge] Attaching to existing tmux session: ${tmuxSession}`);
        } else {
          throw new Error('missing session');
        }
      } catch (e) {
        // Create new session with plurichat
        console.log(`[bus-bridge] Creating new tmux session: ${tmuxSession}`);
        const plurichatPath = path.join(pluribusRoot, 'nucleus', 'tools', 'plurichat.py');
        try {
          const created = spawnSync(
            'tmux',
            ['new-session', '-d', '-s', tmuxSession, '-x', '120', '-y', '30', 'python3', '-u', plurichatPath],
            { cwd: pluribusRoot, env: enrichedEnv, stdio: 'ignore' },
          );
          if (created.status !== 0) {
            throw new Error(`tmux new-session exit=${created.status}`);
          }
          // Seed the session with a status check (helps diagnose provider availability in WebUI).
          spawnSync('tmux', ['send-keys', '-t', `${tmuxSession}:0.0`, '/status', 'C-m'], { env: enrichedEnv, stdio: 'ignore' });
        } catch (err) {
          console.error('[bus-bridge] Failed to create tmux session:', err);
          ws.send('\x1b[31mFailed to create PluriChat session\x1b[0m\r\n');
          ws.close();
          return;
        }
      }

      // Attach to tmux session via PTY (tmux requires a real TTY)
      const ptyProc = spawn('python3', [ptyScript, '--session', tmuxSession], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: enrichedEnv,
      });

      // Bridge PTY -> WebSocket
      ptyProc.stdout.on('data', (data) => {
        if (ws.readyState === 1) {
          ws.send(data);
        }
      });

      ptyProc.stderr.on('data', (data) => {
        if (ws.readyState === 1) {
          ws.send(data);
        }
      });

      // Bridge WebSocket -> PTY
      ws.on('message', (data) => {
        if (!ptyProc.killed) {
          // Handle resize messages
          try {
            const msg = JSON.parse(data.toString());
            if (msg.type === 'resize' && msg.cols && msg.rows) {
              // Resize the controlling PTY running `tmux attach-session` (see tmux_pty_bridge.py).
              try {
                const cols = Math.floor(Number(msg.cols));
                const rows = Math.floor(Number(msg.rows));
                if (Number.isFinite(cols) && Number.isFinite(rows) && cols > 0 && rows > 0) {
                  const frame = `\0PLURIBUS:resize ${cols} ${rows}\n`;
                  ptyProc.stdin.write(Buffer.from(frame, 'utf8'));
                }
              } catch {
                // Ignore resize errors
              }
              return;
            }
          } catch {
            // Not JSON, send as raw input
          }
          ptyProc.stdin.write(data);
        }
      });

      ws.on('close', () => {
        // Don't kill the tmux session, just detach
        try {
          ptyProc.kill('SIGHUP');
        } catch {
          // Ignore
        }
      });

      ptyProc.on('exit', () => {
        ws.close();
      });

      await publishEvent({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        topic: 'plurichat.web.connected',
        kind: 'log',
        level: 'info',
        actor: 'bus-bridge',
        ts: Date.now(),
        iso: new Date().toISOString(),
        data: {
          remote: req.socket.remoteAddress ?? '',
          session: tmuxSession,
        },
      });

      return;
    }

    // Handle Crush Terminal Connections (agentic editor - persistent tmux session)
    if (req.url?.startsWith('/crush')) {
      const url = new URL(req.url, `http://localhost:${wsPort}`);
      const pin = url.searchParams.get('pin');

      if (pin !== '0912') {
        ws.send(JSON.stringify({ type: 'error', message: 'Invalid PIN' }));
        ws.close();
        return;
      }

      console.log('[bus-bridge] Crush terminal connection authenticated');

      const { spawn, spawnSync } = await import('child_process');
      const tmuxSession = 'crush';
      const toolsDir = path.join(pluribusRoot, 'nucleus', 'tools');
      const ptyScript = path.join(toolsDir, 'tmux_pty_bridge.py');
      const enrichedEnv = {
        ...process.env,
        PLURIBUS_ROOT: pluribusRoot,
        PLURIBUS_BUS_DIR: busDir,
        PATH: ensurePathHasDirs(process.env.PATH, ['/usr/local/bin', '/opt/homebrew/bin', '/usr/bin', '/bin']),
      };

      // Ensure tmux session exists (starts a shell; crush command is sent as initCommand from client)
      try {
        const has = spawnSync('tmux', ['has-session', '-t', tmuxSession], { stdio: 'ignore' });
        if (has.status === 0) {
          console.log(`[bus-bridge] Attaching to existing tmux session: ${tmuxSession}`);
        } else {
          throw new Error('missing session');
        }
      } catch {
        // Create new session with a shell
        console.log(`[bus-bridge] Creating new tmux session: ${tmuxSession}`);
        try {
          const created = spawnSync(
            'tmux',
            ['new-session', '-d', '-s', tmuxSession, '-x', '120', '-y', '30'],
            { cwd: pluribusRoot, env: enrichedEnv, stdio: 'ignore' },
          );
          if (created.status !== 0) {
            throw new Error(`tmux new-session exit=${created.status}`);
          }
        } catch (err) {
          console.error('[bus-bridge] Failed to create tmux session:', err);
          ws.send('\x1b[31mFailed to create Crush session\x1b[0m\r\n');
          ws.close();
          return;
        }
      }

      // Attach to tmux session via PTY
      const ptyProc = spawn('python3', [ptyScript, '--session', tmuxSession], {
        stdio: ['pipe', 'pipe', 'pipe'],
        env: enrichedEnv,
      });

      // Bridge PTY -> WebSocket
      ptyProc.stdout.on('data', (data) => {
        if (ws.readyState === 1) {
          ws.send(data);
        }
      });

      ptyProc.stderr.on('data', (data) => {
        if (ws.readyState === 1) {
          ws.send(data);
        }
      });

      // Bridge WebSocket -> PTY
      ws.on('message', (data) => {
        if (!ptyProc.killed) {
          // Handle resize messages
          try {
            const msg = JSON.parse(data.toString());
            if (msg.type === 'resize' && msg.cols && msg.rows) {
              try {
                const cols = Math.floor(Number(msg.cols));
                const rows = Math.floor(Number(msg.rows));
                if (Number.isFinite(cols) && Number.isFinite(rows) && cols > 0 && rows > 0) {
                  const frame = `\0PLURIBUS:resize ${cols} ${rows}\n`;
                  ptyProc.stdin.write(Buffer.from(frame, 'utf8'));
                }
              } catch {
                // Ignore resize errors
              }
              return;
            }
          } catch {
            // Not JSON, send as raw input
          }
          ptyProc.stdin.write(data);
        }
      });

      ws.on('close', () => {
        // Don't kill the tmux session, just detach
        try {
          ptyProc.kill('SIGHUP');
        } catch {
          // Ignore
        }
      });

      ptyProc.on('exit', () => {
        ws.close();
      });

      await publishEvent({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        topic: 'crush.web.connected',
        kind: 'log',
        level: 'info',
        actor: 'bus-bridge',
        ts: Date.now(),
        iso: new Date().toISOString(),
        data: {
          remote: req.socket.remoteAddress ?? '',
          session: tmuxSession,
        },
      });

      return;
    }

    // SKY WebSocket (binary protobuf signaling)
    if (req.url?.startsWith('/sky')) {
      const url = new URL(req.url, `http://localhost:${wsPort}`);
      const peerId = url.searchParams.get('peerId');

      if (!peerId) {
        ws.send(JSON.stringify({ type: 'error', message: 'Missing peerId' }));
        ws.close();
        return;
      }

      skyClients.set(peerId, ws);

      await publishEvent({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        topic: 'sky.ws.client.connected',
        kind: 'log',
        level: 'info',
        actor: 'bus-bridge',
        ts: Date.now(),
        iso: new Date().toISOString(),
        data: {
          remote: req.socket.remoteAddress ?? '',
          url: req.url,
          peerId: peerId,
        },
      });

      ws.on('message', async (data) => {
        try {
          const bytes = data instanceof Buffer ? new Uint8Array(data) : new Uint8Array(data as ArrayBuffer);
          const env = decodeSkyEnvelope(bytes);

          const magic = Number(env['magic'] ?? 0);
          const version = Number(env['version'] ?? 0);
          if (magic !== SKY_MAGIC_V1 || version !== SKY_VERSION_V1) {
            throw new Error(`unexpected SKY envelope header (magic=${magic}, version=${version})`);
          }

          const { topic, level, data: summary } = skyEnvelopeToBusSummary(env);
          const targetPeerId = (summary as Record<string, unknown>).target_peer_id as string;

          await publishEvent({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            topic,
            kind: 'log',
            level,
            actor: 'bus-bridge',
            ts: Date.now(),
            iso: new Date().toISOString(),
            data: summary,
          });

          // Relay raw frame to the target SKY client.
          const targetWs = targetPeerId ? skyClients.get(targetPeerId) : null;
          if (targetWs && targetWs.readyState === 1) {
            targetWs.send(bytes);
          }
        } catch (err) {
          await publishEvent({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            topic: 'sky.error',
            kind: 'log',
            level: 'error',
            actor: 'bus-bridge',
            ts: Date.now(),
            iso: new Date().toISOString(),
            data: { error: String(err) },
          });
        }
      });

      ws.on('close', async () => {
        skyClients.delete(peerId);
        await publishEvent({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          topic: 'sky.ws.client.disconnected',
          kind: 'log',
          level: 'info',
          actor: 'bus-bridge',
          ts: Date.now(),
          iso: new Date().toISOString(),
          data: { url: req.url, peerId: peerId },
        });
      });

      ws.on('error', async (err) => {
        skyClients.delete(peerId);
        await publishEvent({
          id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
          topic: 'sky.ws.client.error',
          kind: 'log',
          level: 'error',
          actor: 'bus-bridge',
          ts: Date.now(),
          iso: new Date().toISOString(),
          data: { error: String(err), peerId: peerId },
        });
      });

      return;
    }

    clients.set(ws, new Set(['*']));  // Subscribe to all by default

    ws.on('message', async (data) => {
      try {
        const msg = JSON.parse(data.toString());

        switch (msg.type) {
          case 'subscribe':
            clients.get(ws)?.add(msg.topic);
            break;

          case 'unsubscribe':
            clients.get(ws)?.delete(msg.topic);
            break;

          case 'publish':
            await publishEvent(msg.event);
            break;

          case 'sync_topic': {
            const topic = typeof msg.topic === 'string' ? msg.topic : '';
            await ensureLastByTopicSeeded();
            const event = topic ? (lastByTopic.get(topic) ?? null) : null;
            ws.send(JSON.stringify({ type: 'sync_topic', topic, event }));
            break;
          }

          case 'sync':
            await sendInitialSync(ws);
            break;
        }
      } catch (err) {
        console.error('[bus-bridge] Message error:', err);
      }
    });

    ws.on('close', () => {
      clients.delete(ws);
    });

    ws.on('error', (err) => {
      console.error('[bus-bridge] Client error:', err);
      clients.delete(ws);
    });
  });

  // Publish event to NDJSON file
  async function publishEvent(event: BusEvent): Promise<void> {
    const line = JSON.stringify(event) + '\n';
    await fs.appendFile(eventsPath, line, 'utf-8');
    if (!ndjsonReadEnabled) {
      ingestEvent(event);
      broadcastEvent(event);
    }
  }

  async function readTailLines(filePath: string, maxLines: number, maxBytes = 4 * 1024 * 1024): Promise<string[]> {
    try {
      const stats = await fs.stat(filePath);
      const end = stats.size;
      if (end <= 0) return [];
      const start = Math.max(0, end - maxBytes);
      const size = end - start;
      if (size <= 0) return [];

      const handle = await fs.open(filePath, 'r');
      try {
        const buffer = Buffer.alloc(size);
        await handle.read(buffer, 0, buffer.length, start);
        const text = buffer.toString('utf-8');
        let lines = text.split('\n');
        // Drop partial first line when we start mid-file.
        if (start > 0) lines = lines.slice(1);
        lines = lines.filter((l) => l.trim().length > 0);
        return lines.slice(-maxLines);
      } finally {
        await handle.close();
      }
    } catch {
      return [];
    }
  }

  async function fetchGraphJson(pathname: string, params: Record<string, string | number | undefined>): Promise<any | null> {
    const fetchFn = (globalThis as any).fetch as undefined | ((...args: any[]) => Promise<any>);
    if (typeof fetchFn !== 'function') {
      return null;
    }
    try {
      const url = new URL(pathname, graphApiUrl);
      for (const [key, value] of Object.entries(params)) {
        if (value === undefined || value === null) continue;
        url.searchParams.set(key, String(value));
      }
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 1200);
      try {
        const res = await fetchFn(url.toString(), { signal: controller.signal });
        if (!res.ok) return null;
        return await res.json();
      } finally {
        clearTimeout(timeout);
      }
    } catch {
      return null;
    }
  }

  function graphEventToBusEvent(entry: GraphTimelineEvent): BusEvent {
    const ts = typeof entry.ts === 'number' ? entry.ts : Date.now() / 1000;
    return {
      id: entry.id,
      topic: entry.topic || 'unknown',
      kind: 'log',
      level: 'info',
      actor: entry.actor || 'unknown',
      ts,
      iso: new Date(ts * 1000).toISOString(),
      data: { source: 'falkordb.timeline' },
      trace_id: entry.trace_id,
    };
  }

  async function fetchGraphTimeline(limit: number, since?: number): Promise<BusEvent[]> {
    const payload = await fetchGraphJson('/api/falkordb/bus/timeline', {
      limit,
      since,
    });
    const events = Array.isArray(payload?.events) ? payload.events as GraphTimelineEvent[] : [];
    return events.map(graphEventToBusEvent);
  }

  function computeTopicEntropy(counts: Map<string, number>): number {
    const total = Array.from(counts.values()).reduce((sum, value) => sum + value, 0);
    if (!total) return 0;
    let entropy = 0;
    for (const value of counts.values()) {
      const p = value / total;
      if (p > 0) entropy -= p * Math.log2(p);
    }
    return entropy;
  }

  let graphSinceTs: number | null = null;
  const graphSeenIds = new Set<string>();
  let graphPollTimer: ReturnType<typeof setInterval> | null = null;
  let graphPollInFlight = false;

  function rememberGraphId(id: string | undefined): boolean {
    if (!id) return true;
    if (graphSeenIds.has(id)) return false;
    graphSeenIds.add(id);
    if (graphSeenIds.size > 10_000) {
      graphSeenIds.clear();
    }
    return true;
  }

  async function seedFromGraph(): Promise<void> {
    const events = await fetchGraphTimeline(graphSeedLimit);
    if (!events.length) return;
    const sorted = [...events].sort((a, b) => (b.ts || 0) - (a.ts || 0));
    for (const event of sorted) {
      if (!event.topic || lastByTopic.has(event.topic)) continue;
      ingestEvent(event);
    }
    const maxTs = Math.max(...events.map((event) => event.ts || 0));
    if (maxTs > 0) {
      graphSinceTs = maxTs;
    }
  }

  async function pollGraphTimeline(): Promise<void> {
    if (graphPollInFlight) return;
    graphPollInFlight = true;
    try {
      const events = await fetchGraphTimeline(graphSeedLimit, graphSinceTs ?? undefined);
      if (!events.length) return;
      const ordered = [...events].sort((a, b) => (a.ts || 0) - (b.ts || 0));
      for (const event of ordered) {
        if (!rememberGraphId(event.id)) continue;
        ingestEvent(event);
        broadcastEvent(event);
      }
      const maxTs = Math.max(...events.map((event) => event.ts || 0));
      if (maxTs > 0) {
        graphSinceTs = Math.max(graphSinceTs || 0, maxTs);
      }
    } finally {
      graphPollInFlight = false;
    }
  }

  let seedPromise: Promise<void> | null = null;
  async function ensureLastByTopicSeeded(): Promise<void> {
    if (seedPromise) return seedPromise;
    seedPromise = (async () => {
      if (!ndjsonReadEnabled) {
        await seedFromGraph();
        return;
      }
      const tail = await readTailLines(eventsPath, 5000, 20 * 1024 * 1024);
      for (const line of tail) {
        try {
          const event = JSON.parse(line) as BusEvent;
          ingestEvent(event);
        } catch {
          // ignore
        }
      }
    })().catch(() => {
      // best-effort; watcher will eventually repopulate on new events
    });
    return seedPromise;
  }

  // Send initial sync to new client
  async function sendInitialSync(ws: WsWebSocket): Promise<void> {
    try {
      await ensureLastByTopicSeeded();
      if (!ndjsonReadEnabled) {
        const events = await fetchGraphTimeline(syncLimit);
        ws.send(JSON.stringify({ type: 'sync', events: events.reverse() }));
        return;
      }
      const tail = await readTailLines(eventsPath, 600, 10 * 1024 * 1024);
      const events: BusEvent[] = [];

      for (const line of tail) {
        try {
          const event = JSON.parse(line) as BusEvent;
          events.push(event);
          ingestEvent(event);
        } catch {
          // Skip malformed lines
        }
      }

      ws.send(JSON.stringify({ type: 'sync', events }));
    } catch {
      ws.send(JSON.stringify({ type: 'sync', events: [] }));
    }
  }

  // Broadcast event to subscribed clients
  function broadcastEvent(event: BusEvent): void {
    const msg = JSON.stringify({ type: 'event', event });

    for (const [ws, subscriptions] of clients) {
      let match = false;

      if (subscriptions.has('*')) {
        match = true;
      } else if (subscriptions.has(event.topic)) {
        match = true;
      } else {
        for (const sub of subscriptions) {
          if (sub.endsWith('*') && event.topic.startsWith(sub.slice(0, -1))) {
            match = true;
            break;
          }
        }
      }

      if (match) {
        ws.send(msg);
      }
    }
  }

  if (ndjsonReadEnabled) {
    // Module-level state for file watching
    let lastPosition = 0;
    let pendingLine = '';
    let changeInFlight = false;
    const maxChunkBytes = Number.parseInt(process.env.BUS_BRIDGE_MAX_CHUNK_BYTES || '1048576', 10) || 1048576;

    // Initialize lastPosition before watcher to avoid large catch-up buffers.
    try {
      const stats = await fs.stat(eventsPath);
      lastPosition = stats.size;
    } catch {
      // File doesn't exist yet
    }

    // Watch for file changes
    watcher = chokidar.watch(eventsPath, {
      persistent: true,
      usePolling: true,
      interval: 100,
    });

    watcher.on('change', async () => {
      if (changeInFlight) return;
      changeInFlight = true;
      try {
        const stats = await fs.stat(eventsPath);

        if (stats.size < lastPosition) {
          pendingLine = '';
          lastPosition = stats.size;
          seedPromise = null;
          await ensureLastByTopicSeeded();
          return;
        }

        if (stats.size > lastPosition) {
          const handle = await fs.open(eventsPath, 'r');
          try {
            let offset = lastPosition;
            while (offset < stats.size) {
              const size = Math.min(maxChunkBytes, stats.size - offset);
              const buffer = Buffer.alloc(size);
              const { bytesRead } = await handle.read(buffer, 0, size, offset);
              if (bytesRead <= 0) break;

              const chunk = buffer.subarray(0, bytesRead).toString('utf-8');
              const split = splitNdjsonChunk(chunk, pendingLine);
              pendingLine = split.carry;
              if (pendingLine.length > maxChunkBytes * 4) {
                console.warn('[bus-bridge] Dropping oversized pending line.');
                pendingLine = '';
              }

              for (const line of split.lines) {
                try {
                  const event: BusEvent = JSON.parse(line);
                  ingestEvent(event);
                  broadcastEvent(event);
                } catch {
                  // Skip malformed lines
                }
              }
              offset += bytesRead;
            }
            lastPosition = stats.size;
          } finally {
            await handle.close();
          }
        }
      } catch (err) {
        console.error('[bus-bridge] Watch error:', err);
      } finally {
        changeInFlight = false;
      }
    });
  } else {
    graphPollTimer = setInterval(() => {
      pollGraphTimeline().catch(() => {});
    }, graphPollIntervalMs);
  }

  // Optional: REST API for state queries
  if (apiPort) {
    apiServer = http.createServer(async (req, res) => {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.setHeader(
        'Access-Control-Allow-Headers',
        'Content-Type, x-portal-asset-id, x-portal-asset-name, x-portal-asset-session, x-portal-asset-created-at, x-portal-asset-expires-at, x-portal-asset-size',
      );

      if (req.method === 'OPTIONS') {
        res.writeHead(204);
        res.end();
        return;
      }

      const url = new URL(req.url || '/', `http://localhost:${apiPort}`);

      if (url.pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'healthy', clients: clients.size }));
        return;
      }

      if (url.pathname === '/portal/assets' && req.method === 'POST') {
        try {
          const rawId = String(req.headers['x-portal-asset-id'] || '');
          const assetId = sanitizeAssetId(rawId || generateFallbackId());
          const filenameHeader = req.headers['x-portal-asset-name'];
          const filename = typeof filenameHeader === 'string' ? filenameHeader : '';
          const contentType = normalizeContentType(String(req.headers['content-type'] || ''));
          const createdHeader = String(req.headers['x-portal-asset-created-at'] || '');
          const expiresHeader = String(req.headers['x-portal-asset-expires-at'] || '');
          const createdAtMs = Date.parse(createdHeader);
          const expiresAtMs = Date.parse(expiresHeader);
          const ttlMs = Number.isFinite(expiresAtMs)
            ? Math.max(expiresAtMs - (Number.isFinite(createdAtMs) ? createdAtMs : Date.now()), 1000)
            : ingestTtlMs;

          const body = await readRequestBody(req, maxIngestBytes);
          await pruneExpiredIngestCache(fs, ingestCacheDir);

          const entry = await writeIngestAsset(fs, {
            root: pluribusRoot,
            assetId,
            data: body,
            filename,
            contentType,
            createdAtMs: Number.isFinite(createdAtMs) ? createdAtMs : undefined,
            ttlMs,
          });

          const sessionHeader = req.headers['x-portal-asset-session'];
          const sessionId = typeof sessionHeader === 'string' ? sessionHeader : undefined;

          await publishEvent({
            topic: 'portal.ingest.asset_cached',
            kind: 'artifact',
            level: 'info',
            actor: 'bus-bridge',
            data: {
              asset_id: entry.asset_id,
              filename: entry.filename,
              content_type: entry.content_type,
              byte_size: entry.byte_size,
              stored_at: entry.stored_at,
              expires_at: entry.expires_at,
              asset_path: entry.asset_path,
              meta_path: entry.meta_path,
              session_id: sessionId,
            },
          });

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: true, asset_id: entry.asset_id, expires_at: entry.expires_at }));
        } catch (err) {
          const message = String(err || 'upload_failed');
          const status = message.includes('payload too large') ? 413 : 500;
          res.writeHead(status, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: message }));
        }
        return;
      }

      if (url.pathname === '/dialogos/health') {
        try {
          const lines = await readTailLines(eventsPath, 3000, 6 * 1024 * 1024);
          let payload = {
            status: 'unknown',
            records_indexed_total: 0,
            uptime_seconds: 0,
          };
          for (const line of lines.reverse()) {
            try {
              const event = JSON.parse(line);
              if (event.topic !== 'dialogos.indexer.stats') continue;
              const data = event.data || {};
              const records = Number(data.records_indexed ?? data.records_indexed_total ?? 0);
              const errors = Number(data.errors ?? 0);
              payload = {
                status: errors > 0 ? 'degraded' : 'ok',
                records_indexed_total: records,
                uptime_seconds: Number(data.uptime_s ?? data.uptime_seconds ?? 0),
              };
              break;
            } catch {
              // Skip malformed lines
            }
          }
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(payload));
        } catch {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ status: 'unknown', records_indexed_total: 0, uptime_seconds: 0 }));
        }
        return;
      }

      if (url.pathname === '/metrics/snapshot') {
        const windowSeconds = Math.max(1, Number.parseInt(url.searchParams.get('window') || '60', 10) || 60);
        try {
          const maxLines = Math.max(600, windowSeconds * 80);
          const maxBytes = Math.max(6 * 1024 * 1024, maxLines * 2048);
          const lines = await readTailLines(eventsPath, maxLines, maxBytes);
          const now = Date.now() / 1000;
          const cutoff = now - windowSeconds;
          const topicCounts = new Map<string, number>();
          const actorCounts = new Map<string, number>();
          let totalEvents = 0;
          let errorEvents = 0;
          let dialogosCompleted = 0;

          for (const line of lines) {
            try {
              const event = JSON.parse(line);
              const ts = Number(event.ts || 0);
              if (ts && ts < cutoff) continue;
              totalEvents += 1;
              const level = String(event.level || '').toLowerCase();
              if (level === 'error') errorEvents += 1;
              const topic = String(event.topic || '');
              const actor = String(event.actor || '');
              if (topic) topicCounts.set(topic, (topicCounts.get(topic) || 0) + 1);
              if (actor) actorCounts.set(actor, (actorCounts.get(actor) || 0) + 1);
              if (topic.startsWith('dialogos.cell.complete') || topic.startsWith('dialogos.cell.end')) {
                dialogosCompleted += 1;
              }
            } catch {
              // Skip malformed lines
            }
          }

          const velocity = totalEvents / Math.max(windowSeconds, 1);
          const payload = {
            ts: now,
            window_s: windowSeconds,
            agents: { count: actorCounts.size },
            topics: { count: topicCounts.size },
            queue: { completed: dialogosCompleted },
            kpis: {
              total_events: totalEvents,
              error_rate: totalEvents ? errorEvents / totalEvents : 0,
              velocity,
            },
            entropy: {
              topic_entropy: computeTopicEntropy(topicCounts),
            },
          };

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(payload));
        } catch (err) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: String(err) }));
        }
        return;
      }

      // Serve VPS session with live provider status
      if (url.pathname === '/session') {
        try {
          const sessionPath = path.join(pluribusRoot, '.pluribus', 'vps_session.json');
          const daemonPath = path.join(pluribusRoot, '.pluribus', 'browser_daemon.json');

          const sessionContent = await fs.readFile(sessionPath, 'utf-8');
          const session = JSON.parse(sessionContent);

          // Merge browser daemon status for web providers
          try {
            const daemonContent = await fs.readFile(daemonPath, 'utf-8');
            const daemon = JSON.parse(daemonContent);
            const pid = Number((daemon && daemon.pid) || 0);
            const isPidAlive = (p: number): boolean => {
              if (!Number.isFinite(p) || p <= 0) return false;
              try {
                process.kill(p, 0);
                return true;
              } catch {
                return false;
              }
            };
            const daemonAlive = Boolean(daemon && daemon.running) && isPidAlive(pid);
            const tabs = (daemon && daemon.tabs && typeof daemon.tabs === 'object') ? daemon.tabs : null;
            if (tabs) {
              for (const [tabId, tab] of Object.entries(tabs)) {
                const tabData = tab as { status?: string; error?: string | null };
                const tabStatus = String(tabData.status || '');
                const model =
                  tabId === 'chatgpt-web' ? 'gpt-5.2-turbo' :
                  tabId === 'gemini-web' ? 'gemini-3-pro' :
                  tabId === 'claude-web' ? 'claude-opus-4-5' : null;
                const baseError = tabData.error || null;
                const error = daemonAlive ? baseError : (baseError || 'browser daemon not running');
                session.providers[tabId] = {
                  available: daemonAlive && tabStatus === 'ready',
                  last_check: new Date().toISOString(),
                  error,
                  model,
                };
              }
            }
          } catch {
            // Browser daemon not running, ignore
          }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(session));
        } catch (err) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Failed to read session' }));
        }
        return;
      }

      // IO Buffer - categorized bus events with omega pair correlation (MCP/A2A/Dialogos aware)
      if (url.pathname === '/io-buffer') {
        try {
          const lines = await readTailLines(eventsPath, 500, 6 * 1024 * 1024);

          // Omega pairs from protocol spec (omega_pairs.json)
          const omegaPairs = {
            'infer_sync.request': 'infer_sync',
            'infer_sync.response': 'infer_sync',
            'dialogos.submit': 'dialogos',
            'dialogos.cell.end': 'dialogos',
            'dialogos.cell.complete': 'dialogos',
            'a2a.negotiate.request': 'a2a_negotiate',
            'a2a.negotiate.response': 'a2a_negotiate',
            'a2a.decline': 'a2a_negotiate',
            'pluribus.check.trigger': 'pluribuscheck',
            'pluribus.check.report': 'pluribuscheck',
            'operator.pbflush.request': 'pbflush',
            'operator.pbflush.ack': 'pbflush',
          };

          // Categorize events
          const input: Array<{ id: string; ts: number; topic: string; actor: string; summary: string; pairId?: string; reqId?: string }> = [];
          const processing: Array<{ id: string; ts: number; topic: string; actor: string; summary: string; pairId?: string }> = [];
          const output: Array<{ id: string; ts: number; topic: string; actor: string; kind: string; summary: string; pairId?: string }> = [];
          const a2aEvents: Array<{ id: string; ts: number; topic: string; from: string; to: string; state: string }> = [];
          const dialogosCells: Array<{ id: string; cellId: string; state: string; actor: string; ts: number }> = [];
          const mcpEvents: Array<{ id: string; ts: number; server: string; method: string; status: string }> = [];

          // Input patterns (omega pair requests + queue items)
          const inputTopics = ['infer_sync.request', 'dialogos.submit', 'strp.request', 'operator.pbflush.request', 'a2a.negotiate.request', 'pluribus.check.trigger', 'mabswarm.probe', 'oiterate.tick', 'rd.tasks.dispatch'];
          // Processing patterns (active work)
          const processingTopics = ['oiterate.action_triggered', 'strp.worker.', 'dialogos.cell.start', 'dialogos.cell.output'];
          // Output patterns (omega pair responses + state)
          const outputTopics = ['art.state', 'art.scene', 'omega.heartbeat', 'omega.queue', 'services.list', 'sota.list', 'pluribus.check.report', 'dialogos.cell.complete', 'dialogos.cell.end', 'a2a.negotiate.response', 'a2a.decline', 'operator.pbflush.ack', 'infer_sync.response', 'mcp.'];

          for (const line of lines.reverse()) {
            try {
              const e = JSON.parse(line);
              const topic = e.topic || '';
              const data = e.data || {};
              const pairId = Object.entries(omegaPairs).find(([t]) => topic.startsWith(t))?.[1];

              const item = {
                id: e.id || `${e.ts}`,
                ts: e.ts || 0,
                topic,
                actor: e.actor || 'unknown',
                kind: e.kind || 'log',
                summary: (data.goal || data.prompt || data.status || data.message || data.method || topic).toString().slice(0, 80),
                pairId,
                reqId: data.req_id || data.request_id,
              };

              // Track A2A negotiations
              if (topic.startsWith('a2a.') && a2aEvents.length < 15) {
                a2aEvents.push({
                  id: e.id, ts: e.ts, topic,
                  from: data.from || e.actor,
                  to: data.to || 'unknown',
                  state: topic.includes('request') ? 'pending' : topic.includes('decline') ? 'declined' : 'accepted',
                });
              }

              // Track Dialogos cells
              if (topic.startsWith('dialogos.cell.') && dialogosCells.length < 20) {
                const cellId = data.cell_id || data.cellId || e.id?.slice(0, 8);
                const existing = dialogosCells.find(c => c.cellId === cellId);
                if (!existing) {
                  dialogosCells.push({
                    id: e.id, cellId, ts: e.ts, actor: e.actor,
                    state: topic.includes('start') ? 'active' : topic.includes('output') ? 'streaming' : topic.includes('end') || topic.includes('complete') ? 'complete' : 'unknown',
                  });
                }
              }

              // Track MCP events
              if (topic.startsWith('mcp.') && mcpEvents.length < 10) {
                mcpEvents.push({
                  id: e.id, ts: e.ts,
                  server: data.server || topic.split('.')[1] || 'unknown',
                  method: data.method || 'unknown',
                  status: data.status || (topic.includes('error') ? 'error' : 'ok'),
                });
              }

              if (inputTopics.some(t => topic.startsWith(t)) && input.length < 25) {
                input.push(item);
              } else if (processingTopics.some(t => topic.startsWith(t)) && processing.length < 25) {
                processing.push(item);
              } else if (outputTopics.some(t => topic.startsWith(t)) && output.length < 35) {
                output.push(item);
              }
            } catch { /* skip */ }
          }

          // Get queue depth metrics
          let queueDepth = 0;
          let pendingPairs = 0;
          let pendingA2a = 0;
          try {
            const recent = lines.slice(-100);
            for (const line of recent) {
              const e = JSON.parse(line);
              if (e.topic === 'omega.queue.depth') queueDepth = e.data?.depth || 0;
              if (e.topic === 'omega.pending.pairs') pendingPairs = e.data?.count || 0;
              if (e.topic === 'omega.a2a.pending') pendingA2a = e.data?.count || 0;
            }
          } catch { /* ignore */ }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            input: input.reverse(),
            processing: processing.reverse(),
            output: output.reverse(),
            a2a: a2aEvents.reverse(),
            dialogos: dialogosCells.reverse(),
            mcp: mcpEvents.reverse(),
            metrics: { queueDepth, pendingPairs, pendingA2a, inputCount: input.length, outputCount: output.length },
          }));
        } catch (err) {
          res.writeHead(500, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: String(err) }));
        }
        return;
      }

      // Discover running agents from processes and recent bus reports
      if (url.pathname === '/agents') {
        try {
          const { execSync } = await import('child_process');

          // Get running agent processes
          const psOutput = execSync(
            "ps aux | grep -E 'claude|codex|pluribus_check|oiterate|strp_worker' | grep -v grep",
            { encoding: 'utf-8', timeout: 5000 }
          ).trim();

          const agents: Record<string, {
            actor: string;
            status: string;
            health: string;
            pid?: number;
            uptime?: string;
            current_task?: string;
          }> = {};

          // Parse ps output for running agents
          for (const line of psOutput.split('\n')) {
            if (!line.trim()) continue;
            const parts = line.trim().split(/\s+/);
            const pid = parseInt(parts[1], 10);
            const startTime = parts[8] || '';
            const cmd = parts.slice(10).join(' ');

            let actor = '';
            if (cmd.includes('claude')) actor = 'claude';
            else if (cmd.includes('codex')) actor = 'codex';
            else if (cmd.includes('pluribus_check_responder')) actor = 'pluribus-responder';
            else if (cmd.includes('oiterate')) actor = 'oiterate-operator';
            else if (cmd.includes('strp_worker')) actor = 'strp-worker';

            if (actor && !agents[actor]) {
              agents[actor] = {
                actor,
                status: 'running',
                health: 'nominal',
                pid,
                uptime: startTime,
                current_task: cmd.slice(0, 80),
              };
            }
          }

          // Also check for recent pluribus.check.report events
          if (ndjsonReadEnabled) {
            try {
              const lines = await readTailLines(eventsPath, 500, 2 * 1024 * 1024);
              for (const line of [...lines].reverse()) {
                try {
                  const event = JSON.parse(line);
                  if (event.topic === 'pluribus.check.report' && event.data) {
                    const actor = event.actor;
                    if (!agents[actor]) {
                      agents[actor] = {
                        actor,
                        status: event.data.status || 'unknown',
                        health: event.data.health || 'unknown',
                        current_task: event.data.current_task?.goal || '',
                      };
                    }
                  }
                } catch { /* skip malformed */ }
              }
            } catch { /* events file error */ }
          }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ agents: Object.values(agents) }));
        } catch (err) {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ agents: [], error: String(err) }));
        }
        return;
      }

      if (url.pathname === '/bus/events') {
        const limit = parseInt(url.searchParams.get('limit') || '100', 10);

        try {
          if (!ndjsonReadEnabled) {
            const events = await fetchGraphTimeline(limit);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(events));
            return;
          }
          const events: BusEvent[] = [];
          const tail = await readTailLines(eventsPath, limit, Math.max(2 * 1024 * 1024, limit * 8192));
          for (const line of tail) {
            try {
              events.push(JSON.parse(line));
            } catch {
              // Skip malformed lines
            }
          }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify(events.reverse()));
        } catch {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify([]));
        }
        return;
      }

      if (url.pathname === '/events') {
        const limit = parseInt(url.searchParams.get('limit') || '100', 10);

        try {
          if (!ndjsonReadEnabled) {
            const events = await fetchGraphTimeline(limit);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ events: events.reverse() }));
            return;
          }
          const events: BusEvent[] = [];
          const tail = await readTailLines(eventsPath, limit, Math.max(2 * 1024 * 1024, limit * 8192));
          for (const line of tail) {
            try {
              events.push(JSON.parse(line));
            } catch {
              // Skip malformed lines
            }
          }

          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ events }));
        } catch {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ events: [] }));
        }
        return;
      }

      if (url.pathname === '/publish' && req.method === 'POST') {
        let body = '';
        req.on('data', (chunk) => { body += chunk; });
        req.on('end', async () => {
          try {
            const event = JSON.parse(body);
            await publishEvent(event);
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ success: true }));
          } catch (err) {
            res.writeHead(400, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: String(err) }));
          }
        });
        return;
      }

      // File System API (Rhizome Access)
      if (url.pathname.startsWith('/fs')) {
        try {
          const rootDir = process.env.PLURIBUS_ROOT || process.cwd();
          const relPath = decodeURIComponent(url.pathname.slice(3)); // Strip '/fs'
          const fullPath = path.resolve(rootDir, relPath.replace(/^\//, ''));

          // Security check: ensure we are inside rootDir
          if (!fullPath.startsWith(rootDir)) {
            res.writeHead(403, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Access denied' }));
            return;
          }

          const stats = await fs.stat(fullPath);

          if (stats.isDirectory()) {
            const entries = await fs.readdir(fullPath, { withFileTypes: true });
            const listing = entries.map(e => ({
              name: e.name,
              type: e.isDirectory() ? 'dir' : 'file',
              size: e.isDirectory() ? 0 : 0, // Simplified
            }));
            // Sort: directories first, then files
            listing.sort((a, b) => {
              if (a.type === b.type) return a.name.localeCompare(b.name);
              return a.type === 'dir' ? -1 : 1;
            });
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ path: relPath, entries: listing }));
          } else {
            // Serve file content
            const content = await fs.readFile(fullPath, 'utf-8');
            res.writeHead(200, { 'Content-Type': 'text/plain' });
            res.end(content);
          }
        } catch (err) {
          res.writeHead(404, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ error: 'Not found or error reading' }));
        }
        return;
      }

      res.writeHead(404, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Not found' }));
    });

    apiServer.listen(apiPort, () => {
      console.log(`[bus-bridge] REST API listening on http://localhost:${apiPort}`);
    });
  }

  if (ndjsonReadEnabled) {
    console.log(`[bus-bridge] Watching ${eventsPath}`);
  } else {
    console.log(`[bus-bridge] NDJSON reads disabled; polling ${graphApiUrl} every ${graphPollIntervalMs}ms`);
  }

  return {
    close: async () => {
      try {
        if (watcher) await watcher.close();
      } catch {
        // ignore
      }
      try {
        if (graphPollTimer) {
          clearInterval(graphPollTimer);
          graphPollTimer = null;
        }
      } catch {
        // ignore
      }
      try {
        await new Promise<void>((resolve) => wss.close(() => resolve()));
      } catch {
        // ignore
      }
      try {
        if (apiServer) {
          await new Promise<void>((resolve) => apiServer!.close(() => resolve()));
        }
      } catch {
        // ignore
      }
    },
  };
}

// CLI entry point - works with both CommonJS and ESM
const isMainModule = typeof require !== 'undefined'
  ? require.main === module
  : import.meta.url === `file://${process.argv[1]}` || process.argv[1]?.endsWith('bus-bridge.ts');

if (isMainModule) {
  const busPath = process.env.PLURIBUS_BUS_DIR || '.pluribus/bus';
  const wsPort = parseInt(process.env.BUS_BRIDGE_WS_PORT || '9200', 10);
  const apiPort = parseInt(process.env.BUS_BRIDGE_API_PORT || '9201', 10);

  startBridge({ busPath, wsPort, apiPort }).catch((err) => {
    console.error('[bus-bridge] Fatal error:', err);
    process.exit(1);
  });
}
