/**
 * Qwik City Node.js Server Entry Point
 *
 * Production server without Vite HMR overhead.
 */
import { createReadStream, statSync } from 'node:fs';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { extname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createQwikCity } from '@builder.io/qwik-city/middleware/node';
import qwikCityPlan from '@qwik-city-plan';
import render from './entry.ssr';

const app = createQwikCity({
  render,
  qwikCityPlan: { ...qwikCityPlan, trailingSlash: false },
});

const DIST_ROOT = resolve(fileURLToPath(import.meta.url), '..', '..', 'dist');
const STATIC_REMAP: Record<string, string> = {};
const MIME_TYPES: Record<string, string> = {
  css: 'text/css',
  js: 'application/javascript',
  map: 'application/json',
  json: 'application/json',
  svg: 'image/svg+xml',
  ico: 'image/x-icon',
  png: 'image/png',
  jpg: 'image/jpeg',
  jpeg: 'image/jpeg',
  webp: 'image/webp',
  woff: 'font/woff',
  woff2: 'font/woff2',
  txt: 'text/plain',
};
const BUS_BRIDGE_API = process.env.BUS_BRIDGE_API_URL || 'http://127.0.0.1:9201';
const TOOLS_API = process.env.TOOLS_API_URL || 'http://127.0.0.1:9300';
const API_TIMEOUT_MS = 1500;
const SW_DISABLE_SCRIPT = [
  "self.addEventListener('install', (event) => {",
  '  self.skipWaiting();',
  '});',
  "self.addEventListener('activate', (event) => {",
  '  event.waitUntil(',
  '    self.registration.unregister().then(() => self.clients.matchAll()).then((clients) => {',
  '      clients.forEach((client) => client.navigate(client.url));',
  '    })',
  '  );',
  '});',
].join('\n');

type BusEvent = {
  ts?: number;
  topic?: string;
  actor?: string;
  level?: string;
  kind?: string;
};

const serveStatic = (req: IncomingMessage, res: ServerResponse) => {
  const method = (req.method || 'GET').toUpperCase();
  if (method !== 'GET' && method !== 'HEAD') return false;

  const url = new URL(req.url || '/', 'http://static');
  let pathname = url.pathname;
  if (pathname === '/vnc' || pathname.startsWith('/vnc/')) {
    res.statusCode = 302;
    res.setHeader('Location', `https://kroma.live${pathname}`);
    res.end();
    return true;
  }
  if (pathname === '/service-worker.js') {
    res.setHeader('Content-Type', 'application/javascript');
    res.setHeader('Cache-Control', 'no-store');
    res.statusCode = 200;
    res.end(SW_DISABLE_SCRIPT);
    return true;
  }
  const remap = STATIC_REMAP[pathname];
  if (remap) pathname = `/${remap}`;

  const relative = pathname.replace(/^\/+/, '');
  if (!relative) return false;

  const filePath = resolve(DIST_ROOT, relative);
  if (!filePath.startsWith(`${DIST_ROOT}/`)) return false;

  try {
    if (!statSync(filePath).isFile()) return false;
  } catch {
    return false;
  }

  const ext = extname(filePath).slice(1).toLowerCase();
  const contentType = MIME_TYPES[ext];
  if (contentType) res.setHeader('Content-Type', contentType);
  res.statusCode = 200;
  if (method === 'HEAD') {
    res.end();
    return true;
  }
  createReadStream(filePath).pipe(res as any);
  return true;
};

const respondJson = (res: { setHeader: (key: string, value: string) => void; statusCode?: number; end: (body?: string) => void }, payload: unknown) => {
  res.statusCode = 200;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(payload));
};

const readBody = (req: IncomingMessage): Promise<Buffer> => {
  const method = (req.method || 'GET').toUpperCase();
  if (method === 'GET' || method === 'HEAD') {
    return Promise.resolve(Buffer.alloc(0));
  }
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (chunk) => chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk)));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
};

const proxyRequest = async (
  req: IncomingMessage,
  res: ServerResponse,
  targetBase: string,
  overridePath?: string,
) => {
  const url = new URL(req.url || '/', 'http://proxy');
  const targetPath = overridePath ?? url.pathname.replace(/^\/api/, '');
  const target = new URL(targetPath, targetBase);
  target.search = url.search;

  const body = await readBody(req);
  const headers: Record<string, string> = {};
  for (const [key, value] of Object.entries(req.headers)) {
    if (!value) continue;
    headers[key] = Array.isArray(value) ? value.join(', ') : String(value);
  }

  const response = await fetch(target, {
    method: req.method || 'GET',
    headers,
    body: body.length ? body : undefined,
  });

  res.statusCode = response.status;
  response.headers.forEach((value, key) => {
    if (key.toLowerCase() === 'transfer-encoding') return;
    res.setHeader(key, value);
  });
  const buffer = Buffer.from(await response.arrayBuffer());
  res.end(buffer);
};

const fetchJson = async <T,>(url: string, fallback: T): Promise<T> => {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT_MS);
  try {
    const response = await fetch(url, { signal: controller.signal });
    if (!response.ok) return fallback;
    return (await response.json()) as T;
  } catch {
    return fallback;
  } finally {
    clearTimeout(timeoutId);
  }
};

const coerceEvents = (payload: unknown): BusEvent[] => {
  if (Array.isArray(payload)) return payload as BusEvent[];
  if (payload && typeof payload === 'object') {
    const events = (payload as { events?: unknown }).events;
    if (Array.isArray(events)) return events as BusEvent[];
  }
  return [];
};

const summarizeEvents = (events: BusEvent[], windowSeconds: number) => {
  const now = Date.now() / 1000;
  const window = Number.isFinite(windowSeconds) && windowSeconds > 0 ? windowSeconds : 60;
  const recent = events.filter((event) => typeof event.ts === 'number' && event.ts >= now - window);
  const total = recent.length;
  const errors = recent.filter((event) => event.level === 'error' || event.kind === 'error').length;
  const actors = new Set(
    recent.map((event) => event.actor).filter((actor): actor is string => typeof actor === 'string' && actor.length > 0)
  );
  const topics = new Set(
    recent.map((event) => event.topic).filter((topic): topic is string => typeof topic === 'string' && topic.length > 0)
  );
  const firstTs = recent.reduce((min, event) => (typeof event.ts === 'number' && event.ts < min ? event.ts : min), now);
  const elapsed = Math.max(now - firstTs, 1);
  const velocity = total > 0 ? (total / elapsed) * 60 : 0;
  const entropyBase = Math.max(total, 1);
  const dialogosCount = recent.filter((event) => typeof event.topic === 'string' && event.topic.startsWith('dialogos.')).length;
  return {
    now,
    window,
    total,
    errors,
    velocity,
    actorsCount: actors.size,
    topicsCount: topics.size,
    topicEntropy: topics.size / entropyBase,
    actorEntropy: actors.size / entropyBase,
    dialogosCount,
  };
};

const buildMetricsSnapshot = (events: BusEvent[], windowSeconds: number) => {
  const summary = summarizeEvents(events, windowSeconds);
  return {
    snapshot_id: `kroma-${Math.round(summary.now * 1000)}`,
    ts: summary.now,
    iso: new Date().toISOString(),
    window_seconds: summary.window,
    kpis: {
      total_events: summary.total,
      total_errors: summary.errors,
      velocity: summary.velocity,
      error_rate: summary.total ? summary.errors / summary.total : 0,
      latency: { avg_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0 },
    },
    agents: { count: summary.actorsCount, active: summary.actorsCount, silent: 0, details: {} },
    topics: { count: summary.topicsCount, details: {} },
    topology: { single_count: 0, star_count: 0, peer_debate_count: 0, avg_fanout: 0, coordination_budget_total: 0 },
    evolutionary: { vgt_transfers: 0, hgt_transfers: 0, total_generations: 0, avg_speciation_potential: 0, lineage_health_distribution: {} },
    entropy: { topic_entropy: summary.topicEntropy, actor_entropy: summary.actorEntropy, level_entropy: 0, causal_depth_avg: 0, reversibility_avg: 0 },
    queue: { depth: 0, pending: 0, completed: summary.dialogosCount },
  };
};

const buildDialogosHealth = (events: BusEvent[]) => {
  const dialogosEvents = events.filter((event) => typeof event.topic === 'string' && event.topic.startsWith('dialogos.'));
  const lastTs = dialogosEvents.reduce((max, event) => (typeof event.ts === 'number' && event.ts > max ? event.ts : max), 0);
  const now = Date.now() / 1000;
  const uptime = lastTs > 0 ? Math.max(now - lastTs, 1) : 0;
  return {
    status: dialogosEvents.length > 0 ? 'ok' : 'degraded',
    records_indexed_total: dialogosEvents.length,
    uptime_seconds: uptime,
    records_indexed_per_minute: uptime > 0 ? dialogosEvents.length / (uptime / 60) : 0,
  };
};

const handleApi = (req: IncomingMessage, res: ServerResponse) => {
  if (!req.url) return false;
  const url = new URL(req.url, 'http://api');

  if (url.pathname === '/api/emit' && (req.method || 'GET').toUpperCase() === 'POST') {
    void proxyRequest(req, res, BUS_BRIDGE_API, '/publish');
    return true;
  }

  if (url.pathname === '/api/portal/assets' && (req.method || 'GET').toUpperCase() === 'POST') {
    void proxyRequest(req, res, BUS_BRIDGE_API, '/portal/assets');
    return true;
  }

  if (url.pathname.startsWith('/api/browser/')) {
    void proxyRequest(req, res, TOOLS_API);
    return true;
  }

  if (url.pathname === '/api/session' || url.pathname === '/api/agents' || url.pathname === '/api/io-buffer') {
    void proxyRequest(req, res, BUS_BRIDGE_API);
    return true;
  }

  if (url.pathname === '/api/bus/events') {
    const limit = Number.parseInt(url.searchParams.get('limit') || '100', 10);
    const target = new URL('/events', BUS_BRIDGE_API);
    target.searchParams.set('limit', Number.isFinite(limit) ? String(limit) : '100');
    void fetchJson(target.toString(), { events: [] as BusEvent[] })
      .then((payload) => respondJson(res, coerceEvents(payload)));
    return true;
  }

  if (url.pathname === '/api/metrics/snapshot') {
    const windowSeconds = Number.parseInt(url.searchParams.get('window') || '60', 10);
    const target = new URL('/events', BUS_BRIDGE_API);
    target.searchParams.set('limit', '2000');
    void fetchJson(target.toString(), { events: [] as BusEvent[] })
      .then((payload) => respondJson(res, buildMetricsSnapshot(coerceEvents(payload), windowSeconds)));
    return true;
  }

  if (url.pathname === '/api/dialogos/health') {
    const target = new URL('/events', BUS_BRIDGE_API);
    target.searchParams.set('limit', '2000');
    void fetchJson(target.toString(), { events: [] as BusEvent[] })
      .then((payload) => respondJson(res, buildDialogosHealth(coerceEvents(payload))));
    return true;
  }

  return false;
};

// Start the node server when this module is executed directly.
const isMain = !!process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));
if (isMain) {
  const port = Number.parseInt(process.env.PORT || '3000', 10);
  const host = process.env.HOST || '0.0.0.0';

  createServer((req, res) => {
    if (serveStatic(req, res)) return;
    if (handleApi(req, res)) return;
    app.router(req, res, () => app.notFound(req, res));
  }).listen(port, host, () => {
    console.log(`[qwik] server ready on http://${host}:${port}`);
  });
}

/** The Qwik City Node.js middleware. */
export default app;
