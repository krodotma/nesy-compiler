/**
 * Isomorphic Bus Client
 *
 * Unified interface for bus communication across all platforms.
 * - Node.js: Direct NDJSON file access
 * - Browser: WebSocket connection to bridge
 * - React Native: WebSocket connection to bridge
 * - WASM: Stdin/stdout for NDJSON
 */

import type { BusEvent } from '../state/types';

export type Platform = 'node' | 'browser' | 'native' | 'wasm';

export interface BusClientOptions {
  platform: Platform;
  busPath?: string;  // For node/wasm
  wsUrl?: string;    // For browser/native
  pollInterval?: number;
}

export interface BusClient {
  connect(): Promise<void>;
  disconnect(): void;
  subscribe(topic: string, handler: (event: BusEvent) => void): () => void;
  publish(event: Omit<BusEvent, 'ts' | 'iso'>): Promise<void>;
  getEvents(limit?: number): Promise<BusEvent[]>;
}

/**
 * Detect current platform
 */
export function detectPlatform(): Platform {
  if (typeof globalThis.process !== 'undefined' && globalThis.process.versions?.node) {
    return 'node';
  }
  if (typeof globalThis.window !== 'undefined') {
    return 'browser';
  }
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator.product === 'ReactNative') {
    return 'native';
  }
  return 'wasm';
}

/**
 * Create a bus client for the current platform
 */
export function createBusClient(opts: BusClientOptions): BusClient {
  const platform = opts.platform || 'browser';

  if (typeof __E2E__ !== 'undefined' && __E2E__ && (platform === 'browser' || platform === 'native')) {
    return new NoopBusClient();
  }

  switch (platform) {
    case 'node':
      return new NodeBusClient(opts);
    case 'browser':
    case 'native':
      return new WebSocketBusClient({
        ...opts,
        wsUrl:
          opts.wsUrl ||
          (typeof window !== 'undefined'
            ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/bus`
            : 'ws://localhost:5173/ws/bus'),
      });
    case 'wasm':
      return new StdioBusClient(opts);
    default:
      throw new Error(`Unknown platform: ${platform}`);
  }
}

class NoopBusClient implements BusClient {
  async connect(): Promise<void> {
    return;
  }

  disconnect(): void {
    // no-op
  }

  subscribe(_topic: string, _handler: (event: BusEvent) => void): () => void {
    return () => {};
  }

  async publish(_event: Omit<BusEvent, 'ts' | 'iso'>): Promise<void> {
    return;
  }

  async getEvents(_limit = 100): Promise<BusEvent[]> {
    return [];
  }
}

/**
 * Node.js bus client - reads NDJSON files directly
 */
class NodeBusClient implements BusClient {
  private busPath: string;
  private pollInterval: number;
  private subscribers: Map<string, Set<(event: BusEvent) => void>> = new Map();
  private lastPosition = 0;
  private pollTimer: ReturnType<typeof setInterval> | null = null;

  constructor(options: Partial<BusClientOptions>) {
    this.busPath = options.busPath || process.env.PLURIBUS_BUS_DIR || '.pluribus/bus';
    this.pollInterval = options.pollInterval || 500;
  }

  async connect(): Promise<void> {
    // Start polling for new events
    this.pollTimer = setInterval(() => this.poll(), this.pollInterval);
  }

  disconnect(): void {
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
  }

  subscribe(topic: string, handler: (event: BusEvent) => void): () => void {
    if (!this.subscribers.has(topic)) {
      this.subscribers.set(topic, new Set());
    }
    this.subscribers.get(topic)!.add(handler);

    return () => {
      this.subscribers.get(topic)?.delete(handler);
    };
  }

  async publish(event: Omit<BusEvent, 'ts' | 'iso'>): Promise<void> {
    const fs = await import('fs/promises');
    const path = await import('path');

    const fullEvent: BusEvent = {
      ...event,
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
    };

    const eventsPath = path.join(this.busPath, 'events.ndjson');
    const line = JSON.stringify(fullEvent) + '\n';

    await fs.appendFile(eventsPath, line, 'utf-8');
  }

  async getEvents(limit = 100): Promise<BusEvent[]> {
    const fs = await import('fs/promises');
    const path = await import('path');

    const eventsPath = path.join(this.busPath, 'events.ndjson');

    try {
      const content = await fs.readFile(eventsPath, 'utf-8');
      const lines = content.trim().split('\n').filter(Boolean);
      const events: BusEvent[] = [];

      for (const line of lines.slice(-limit)) {
        try {
          events.push(JSON.parse(line));
        } catch {
          // Skip malformed lines
        }
      }

      return events;
    } catch {
      return [];
    }
  }

  private async poll(): Promise<void> {
    const fs = await import('fs/promises');
    const path = await import('path');

    const eventsPath = path.join(this.busPath, 'events.ndjson');

    try {
      const stats = await fs.stat(eventsPath);
      if (stats.size <= this.lastPosition) {
        if (stats.size < this.lastPosition) {
          this.lastPosition = 0;  // File was truncated
        }
        return;
      }

      const handle = await fs.open(eventsPath, 'r');
      const buffer = Buffer.alloc(stats.size - this.lastPosition);
      await handle.read(buffer, 0, buffer.length, this.lastPosition);
      await handle.close();

      this.lastPosition = stats.size;

      const lines = buffer.toString('utf-8').split('\n').filter(Boolean);

      for (const line of lines) {
        try {
          const event: BusEvent = JSON.parse(line);
          this.dispatch(event);
        } catch {
          // Skip malformed lines
        }
      }
    } catch {
      // File doesn't exist or other error
    }
  }

  private dispatch(event: BusEvent): void {
    // Dispatch to topic-specific subscribers
    this.subscribers.get(event.topic)?.forEach((handler) => handler(event));

    // Dispatch to wildcard subscribers
    this.subscribers.get('*')?.forEach((handler) => handler(event));

    // Dispatch to prefix subscribers
    for (const [topic, handlers] of this.subscribers) {
      if (topic.endsWith('*') && event.topic.startsWith(topic.slice(0, -1))) {
        handlers.forEach((handler) => handler(event));
      }
    }
  }
}

/**
 * WebSocket bus client - for browser and React Native
 */
class WebSocketBusClient implements BusClient {
  private wsUrl: string;
  private ws: WebSocket | null = null;
  private subscribers: Map<string, Set<(event: BusEvent) => void>> = new Map();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private connectTimeout: ReturnType<typeof setTimeout> | null = null;
  private eventBuffer: BusEvent[] = [];
  private reconnectEnabled = true;
  private connectPromise: Promise<void> | null = null;
  private reconnectAttempts = 0;
  private lastOpenAtMs: number | null = null;

  private shouldKeepAlive(): boolean {
    // Avoid scheduling keepalive intervals in pure Node test environments where
    // fake timers may treat intervals as infinite work (vitest runAllTimers).
    if (typeof window !== 'undefined') return true;
    if (typeof navigator !== 'undefined' && (navigator as any).product === 'ReactNative') return true;
    return false;
  }

  constructor(options: Partial<BusClientOptions>) {
    // Default to proxy path that works with both HTTP and HTTPS
    const defaultWsUrl = typeof window !== 'undefined'
      ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/bus`
      : 'ws://localhost:5173/ws/bus';
    this.wsUrl = options.wsUrl || defaultWsUrl;
  }

  async connect(): Promise<void> {
    // Re-enable reconnects for explicit connect calls.
    this.reconnectEnabled = true;

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }
    if (this.connectPromise) {
      return this.connectPromise;
    }

    // Cancel any pending reconnect timer before attempting a fresh connection.
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.connectTimeout) {
      clearTimeout(this.connectTimeout);
      this.connectTimeout = null;
    }

    this.connectPromise = new Promise((resolve, reject) => {
      let settled = false;
      const clearConnectTimeout = () => {
        if (!this.connectTimeout) return;
        clearTimeout(this.connectTimeout);
        this.connectTimeout = null;
      };
      const ws = new WebSocket(this.wsUrl);
      this.ws = ws;
      // If the connection stalls (e.g., proxy issues), fail fast so callers can
      // recover and the reconnect loop can back off. 8s is reasonable for initial
      // connection - long enough for slow networks, short enough to not block UI.
      this.connectTimeout = setTimeout(() => {
        if (settled) return;
        settled = true;
        clearConnectTimeout();
        if (this.ws === ws) {
          this.ws = null;
        }
        this.connectPromise = null;
        try {
          ws.close();
        } catch {
          // ignore
        }
        reject(new Error('WebSocket connection timed out'));
      }, 8_000);

      // Track WebSocket for telemetry (non-blocking dynamic import)
      import('../telemetry').then(({ trackWebSocket }) => {
        if (this.ws) trackWebSocket(this.ws, 'bus-client');
      }).catch(() => {/* Telemetry not critical */});

      ws.onopen = () => {
        // Request initial sync
        clearConnectTimeout();
        this.lastOpenAtMs = Date.now();
        if (this.shouldKeepAlive()) {
          if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
          }
          // Keep the connection active through idle timeouts in proxies/NATs.
          this.pingTimer = setInterval(() => {
            if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
            try {
              this.ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
            } catch {
              // ignore
            }
          }, 30_000);
        }
        try {
          ws.send(JSON.stringify({ type: 'sync' }));
        } catch {
          // ignore
        }
        // Re-send existing subscriptions after reconnect.
        this.syncSubscriptions();
        if (!settled) {
          settled = true;
          this.connectPromise = null;
          resolve();
        }
      };

      ws.onerror = (error) => {
        clearConnectTimeout();
        if (!settled) {
          settled = true;
          this.connectPromise = null;
          reject(error);
        }
        // Some environments emit `error` without an immediate `close`.
        // Best-effort: close to ensure resources are released and reconnect can happen.
        try {
          ws.close();
        } catch {
          // ignore
        }
      };

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data as string);

          if (data.type === 'event') {
            this.dispatch(data.event as BusEvent);
            this.eventBuffer.push(data.event as BusEvent);
            if (this.eventBuffer.length > 500) {
              this.eventBuffer.shift();
            }
          } else if (data.type === 'sync_topic') {
            const event = data.event as BusEvent | null | undefined;
            if (event && typeof event.topic === 'string') {
              this.dispatch(event);
              this.eventBuffer.push(event);
              if (this.eventBuffer.length > 500) {
                this.eventBuffer.shift();
              }
            }
          } else if (data.type === 'sync') {
            // Initial state sync
            if (Array.isArray(data.events)) {
              this.eventBuffer = data.events;
            }
          }
        } catch {
          // Ignore malformed messages
        }
      };

      ws.onclose = () => {
        clearConnectTimeout();
        if (this.pingTimer) {
          clearInterval(this.pingTimer);
          this.pingTimer = null;
        }
        // Clear the reference so a future connect() will create a new socket.
        if (this.ws === ws) {
          this.ws = null;
        }
        const openAt = this.lastOpenAtMs;
        this.lastOpenAtMs = null;
        // Only reset the backoff when the connection was stable for long enough;
        // otherwise flapping connections will reconnect too aggressively.
        if (openAt && Date.now() - openAt >= 30_000) {
          this.reconnectAttempts = 0;
        }

        // If we never connected, fail the pending connect() call.
        if (!settled) {
          settled = true;
          this.connectPromise = null;
          reject(new Error('WebSocket closed before connection established'));
        }

        // Attempt to reconnect (unless explicitly disconnected).
        if (!this.reconnectEnabled) {
          return;
        }

        const delayMs = this.nextReconnectDelayMs();
        if (this.reconnectTimer) {
          clearTimeout(this.reconnectTimer);
        }
        this.reconnectTimer = setTimeout(() => {
          this.connect().catch(() => {});
        }, delayMs);
      };
    });

    return this.connectPromise;
  }

  disconnect(): void {
    // Prevent the close handler from scheduling a reconnect.
    this.reconnectEnabled = false;
    this.reconnectAttempts = 0;
    this.lastOpenAtMs = null;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    if (this.connectTimeout) {
      clearTimeout(this.connectTimeout);
      this.connectTimeout = null;
    }
    this.ws?.close();
    this.ws = null;
    this.connectPromise = null;
  }

  subscribe(topic: string, handler: (event: BusEvent) => void): () => void {
    if (!this.subscribers.has(topic)) {
      this.subscribers.set(topic, new Set());
    }
    this.subscribers.get(topic)!.add(handler);

    // Subscribe on server (only when open, otherwise sync on connect).
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'subscribe', topic }));
        // Best-effort: request the last event for exact topics so new clients can bootstrap state.
        if (topic && topic !== '*' && !topic.endsWith('*')) {
          this.ws.send(JSON.stringify({ type: 'sync_topic', topic }));
        }
      } catch {
        // ignore
      }
    }

    return () => {
      this.subscribers.get(topic)?.delete(handler);
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'unsubscribe', topic }));
        } catch {
          // ignore
        }
      }
    };
  }

  async publish(event: Omit<BusEvent, 'ts' | 'iso'>): Promise<void> {
    const fullEvent: BusEvent = {
      ...event,
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
    };

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify({ type: 'publish', event: fullEvent }));
      } catch {
        // ignore
      }
    }
  }

  async getEvents(limit = 100): Promise<BusEvent[]> {
    return this.eventBuffer.slice(-limit);
  }

  private nextReconnectDelayMs(): number {
    // Exponential backoff with jitter to avoid reconnect storms.
    const baseMs = 1000;
    const maxMs = 5 * 60_000;
    const exp = Math.min(maxMs, baseMs * (2 ** Math.min(this.reconnectAttempts, 6)));
    const jitter = Math.floor(exp * (0.2 * Math.random()));
    this.reconnectAttempts = Math.min(this.reconnectAttempts + 1, 20);
    return exp + jitter;
  }

  private syncSubscriptions(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    for (const [topic, handlers] of this.subscribers.entries()) {
      if (!handlers || handlers.size === 0) continue;
      try {
        this.ws.send(JSON.stringify({ type: 'subscribe', topic }));
        if (topic && topic !== '*' && !topic.endsWith('*')) {
          this.ws.send(JSON.stringify({ type: 'sync_topic', topic }));
        }
      } catch {
        // ignore
      }
    }
  }

  private dispatch(event: BusEvent): void {
    this.subscribers.get(event.topic)?.forEach((handler) => handler(event));
    this.subscribers.get('*')?.forEach((handler) => handler(event));

    for (const [topic, handlers] of this.subscribers) {
      if (topic.endsWith('*') && event.topic.startsWith(topic.slice(0, -1))) {
        handlers.forEach((handler) => handler(event));
      }
    }
  }
}

/**
 * Stdio bus client - for WASM/Cosmopolitan
 */
class StdioBusClient implements BusClient {
  private subscribers: Map<string, Set<(event: BusEvent) => void>> = new Map();
  private eventBuffer: BusEvent[] = [];

  constructor(_options: Partial<BusClientOptions>) {}

  async connect(): Promise<void> {
    // In WASM mode, we read from stdin
    // This is a simplified implementation
  }

  disconnect(): void {}

  subscribe(topic: string, handler: (event: BusEvent) => void): () => void {
    if (!this.subscribers.has(topic)) {
      this.subscribers.set(topic, new Set());
    }
    this.subscribers.get(topic)!.add(handler);

    return () => {
      this.subscribers.get(topic)?.delete(handler);
    };
  }

  async publish(event: Omit<BusEvent, 'ts' | 'iso'>): Promise<void> {
    const fullEvent: BusEvent = {
      ...event,
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
    };

    // Write to stdout
    console.log(JSON.stringify(fullEvent));
  }

  async getEvents(limit = 100): Promise<BusEvent[]> {
    return this.eventBuffer.slice(-limit);
  }

  /**
   * Process a line of NDJSON input
   */
  processLine(line: string): void {
    try {
      const event: BusEvent = JSON.parse(line);
      this.eventBuffer.push(event);
      if (this.eventBuffer.length > 500) {
        this.eventBuffer.shift();
      }

      // Dispatch to subscribers
      this.subscribers.get(event.topic)?.forEach((handler) => handler(event));
      this.subscribers.get('*')?.forEach((handler) => handler(event));
    } catch {
      // Ignore malformed lines
    }
  }
}
