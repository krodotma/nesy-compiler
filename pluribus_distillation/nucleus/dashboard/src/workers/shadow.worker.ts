/**
 * SHADOW WORKER: The Hidden App
 * =============================
 * Runs in a background thread to handle heavy lifting, data pre-fetching,
 * and state management, keeping the Main Thread (UI) buttery smooth.
 *
 * Capabilities:
 * - Data caching (SOTA, Git, Rhizome)
 * - Search indexing (Client-side FlexSearch/Fuse.js)
 * - Background compilation/processing
 */

const ctx: Worker = self as any;

// Shared State Cache
const cache = {
  sota: [],
  gitLog: [],
  gitStatus: [],
  gitBranches: [],
  gitCurrentBranch: null as string | null,
  agents: [],
  requests: [],
};

// Broadcast Channel for UI Sync
const broadcast = new BroadcastChannel('pluribus-shadow');

// Track initialization state
let initialized = false;

ctx.addEventListener('message', async (ev) => {
  const { type, payload } = ev.data;

  if (type === 'INIT') {
    if (initialized) return; // Prevent double init
    initialized = true;
    console.log('[Shadow] Initializing Hidden App...');

    // Immediately prefetch critical data in parallel (don't wait for first poll)
    Promise.all([
      fetchSOTA(),
      fetchGit(),
      fetchBrowserStatus(),
    ]).then(() => {
      broadcast.postMessage({ type: 'SHADOW_READY', ts: Date.now() });
    });

    startPolling();
  }

  if (type === 'PREFETCH') {
    const { views } = payload || {};
    // Run prefetches in parallel
    const tasks = [];
    if (views?.includes('sota')) tasks.push(fetchSOTA());
    if (views?.includes('git')) tasks.push(fetchGit());
    if (views?.includes('browser')) tasks.push(fetchBrowserStatus());
    if (tasks.length > 0) {
      await Promise.all(tasks);
      broadcast.postMessage({ type: 'PREFETCH_COMPLETE', views, ts: Date.now() });
    }
  }
});

// Helper: fetch with timeout hint (no abort to avoid request failures)
async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  timeoutMs = 5000
): Promise<Response | null> {
  const timeout = new Promise<null>((resolve) => {
    setTimeout(() => resolve(null), timeoutMs);
  });
  const fetchPromise = fetch(url, options).catch(() => null);
  return Promise.race([fetchPromise, timeout]);
}

async function fetchBrowserStatus() {
  try {
    const res = await fetchWithTimeout('/api/browser/status');
    if (res && res.ok) {
      const data = await res.json();
      broadcast.postMessage({ type: 'DATA_UPDATE', key: 'browserStatus', data });
    }
  } catch {
    // silent - timeout or network error
  }
}

async function fetchSOTA() {
  try {
    const url = `/api/sota?ts=${Date.now()}`;
    const res = await fetchWithTimeout(url, { cache: 'no-store' });
    if (res && res.ok) {
      const data = await res.json();
      cache.sota = data.items || [];
      broadcast.postMessage({ type: 'DATA_UPDATE', key: 'sota', data: cache.sota });
    }
  } catch {
    // silent - timeout or network error
  }
}

async function fetchGit() {
  try {
    const [logRes, statusRes, branchRes] = await Promise.all([
      fetchWithTimeout('/api/git/log'),
      fetchWithTimeout('/api/git/status'),
      fetchWithTimeout('/api/git/branches'),
    ]);
    if (logRes && logRes.ok) {
      const data = await logRes.json();
      cache.gitLog = data.commits || [];
      broadcast.postMessage({ type: 'DATA_UPDATE', key: 'gitLog', data: cache.gitLog });
    }
    if (statusRes && statusRes.ok) {
      const data = await statusRes.json();
      cache.gitStatus = data.status || [];
      broadcast.postMessage({ type: 'DATA_UPDATE', key: 'gitStatus', data: cache.gitStatus });
    }
    if (branchRes && branchRes.ok) {
      const data = await branchRes.json();
      cache.gitBranches = data.branches || [];
      cache.gitCurrentBranch = data.current || null;
      broadcast.postMessage({ type: 'DATA_UPDATE', key: 'gitBranches', data: cache.gitBranches });
      broadcast.postMessage({ type: 'DATA_UPDATE', key: 'gitCurrentBranch', data: cache.gitCurrentBranch });
    }
  } catch {
    // silent - timeout or network error
  }
}

function startPolling() {
  setInterval(() => {
    // Periodic background refresh
    fetchSOTA();
    fetchGit();
  }, 30000); // Every 30s
}

export {};
