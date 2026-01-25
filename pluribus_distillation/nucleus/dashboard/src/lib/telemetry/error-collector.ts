/**
 * Error Telemetry Collector
 *
 * Captures all client-side errors, warnings, and issues and streams them
 * to the Pluribus bus for rapid pre-detection and debugging.
 *
 * Captures:
 * - JavaScript errors (window.onerror)
 * - Unhandled promise rejections
 * - Console warnings and errors (intercepted)
 * - Fetch/network failures
 * - WebSocket disconnections
 * - Resource loading errors
 * - Qwik component errors (via error boundary)
 *
 * All events are batched and sent via POST /api/emit -> bus-bridge -> events.ndjson
 */

export type ErrorSeverity = 'error' | 'warn' | 'info';
export type ErrorCategory =
  | 'js_error'
  | 'promise_rejection'
  | 'console_error'
  | 'console_warn'
  | 'fetch_error'
  | 'websocket_error'
  | 'resource_error'
  | 'component_error'
  | 'performance'
  | 'telemetry_suppressed';

export interface TelemetryEvent {
  id: string;
  ts: number;
  iso: string;
  category: ErrorCategory;
  severity: ErrorSeverity;
  message: string;
  stack?: string;
  url?: string;
  line?: number;
  column?: number;
  source?: string;
  context?: Record<string, unknown>;
  userAgent?: string;
  pathname?: string;
}

interface CollectorConfig {
  /** Emit endpoint (default: /api/emit) */
  endpoint?: string;
  /** Batch size before flush (default: 5) */
  batchSize?: number;
  /** Flush interval in ms (default: 3000) */
  flushInterval?: number;
  /** Enable console interception (default: true) */
  interceptConsole?: boolean;
  /** Enable fetch interception (default: true) */
  interceptFetch?: boolean;
  /** Sample rate 0-1 for high-frequency errors (default: 1) */
  sampleRate?: number;
  /** Max events to store before dropping oldest (default: 100) */
  maxQueueSize?: number;
  /** Burst window in ms for suppression (default: 10000) */
  burstWindowMs?: number;
  /** Max events per category per burst window (default: 20) */
  burstMaxPerCategory?: number;
  /** Max total events per burst window (default: 60) */
  burstMaxTotal?: number;
  /** Max length for message/stack fields (default: 4000) */
  maxFieldLength?: number;
  /** Actor name for bus events */
  actor?: string;
}

const DEFAULT_CONFIG: Required<CollectorConfig> = {
  endpoint: '/api/emit',
  batchSize: 5,
  flushInterval: 3000,
  interceptConsole: true,
  interceptFetch: true,
  sampleRate: 1,
  maxQueueSize: 100,
  burstWindowMs: 10000,
  burstMaxPerCategory: 20,
  burstMaxTotal: 60,
  maxFieldLength: 4000,
  actor: 'dashboard-telemetry',
};

let isInitialized = false;
let config: Required<CollectorConfig> = { ...DEFAULT_CONFIG };
let eventQueue: TelemetryEvent[] = [];
let flushTimer: ReturnType<typeof setInterval> | null = null;
let originalConsoleError: typeof console.error;
let originalConsoleWarn: typeof console.warn;
let originalFetch: typeof fetch;

// Deduplication: track recent error signatures to avoid spam
const recentErrors = new Map<string, number>();
const DEDUP_WINDOW_MS = 5000;
const burstState = {
  windowStart: 0,
  total: 0,
  perCategory: new Map<ErrorCategory, { count: number; suppressed: number; lastMessage?: string }>(),
};

function generateId(): string {
  return `err-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function getErrorSignature(event: Partial<TelemetryEvent>): string {
  return `${event.category}:${event.message?.slice(0, 100)}:${event.source || ''}:${event.line || 0}`;
}

function shouldSample(): boolean {
  return Math.random() < config.sampleRate;
}

function truncateText(value: string | undefined, maxLen: number): string | undefined {
  if (!value) return value;
  if (value.length <= maxLen) return value;
  return `${value.slice(0, maxLen)}…`;
}

function resetBurstWindow(now: number): void {
  for (const [category, entry] of burstState.perCategory.entries()) {
    if (entry.suppressed > 0) {
      emitSuppressionSummary(category, entry.suppressed, entry.lastMessage);
    }
  }
  burstState.windowStart = now;
  burstState.total = 0;
  burstState.perCategory.clear();
}

function emitSuppressionSummary(category: ErrorCategory, suppressed: number, lastMessage?: string): void {
  const now = Date.now();
  queueEvent({
    id: generateId(),
    ts: now,
    iso: new Date(now).toISOString(),
    category: 'telemetry_suppressed',
    severity: 'warn',
    message: `Suppressed ${suppressed} ${category} events in ${config.burstWindowMs}ms`,
    context: {
      suppressed,
      category,
      windowMs: config.burstWindowMs,
      lastMessage,
    },
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
  }, { force: true });
}

function shouldSuppress(event: TelemetryEvent): boolean {
  if (event.category === 'telemetry_suppressed') return false;
  const now = Date.now();
  if (!burstState.windowStart) burstState.windowStart = now;
  if (now - burstState.windowStart > config.burstWindowMs) {
    resetBurstWindow(now);
  }
  burstState.total += 1;
  const entry = burstState.perCategory.get(event.category) || { count: 0, suppressed: 0 };
  entry.count += 1;
  entry.lastMessage = truncateText(event.message, config.maxFieldLength);
  burstState.perCategory.set(event.category, entry);
  if (entry.count > config.burstMaxPerCategory || burstState.total > config.burstMaxTotal) {
    entry.suppressed += 1;
    return true;
  }
  return false;
}

function isDuplicate(event: TelemetryEvent): boolean {
  const sig = getErrorSignature(event);
  const lastSeen = recentErrors.get(sig);
  const now = Date.now();

  if (lastSeen && now - lastSeen < DEDUP_WINDOW_MS) {
    return true;
  }

  recentErrors.set(sig, now);

  // Cleanup old entries
  if (recentErrors.size > 50) {
    for (const [key, time] of recentErrors) {
      if (now - time > DEDUP_WINDOW_MS) {
        recentErrors.delete(key);
      }
    }
  }

  return false;
}

function queueEvent(event: TelemetryEvent, opts?: { force?: boolean }): void {
  if (!opts?.force) {
    if (shouldSuppress(event)) return;
    if (!shouldSample()) return;
    if (isDuplicate(event)) return;
  }

  const maxLen = config.maxFieldLength;
  event.message = truncateText(event.message, maxLen) || event.message;
  if (event.stack) {
    event.stack = truncateText(event.stack, maxLen);
  }

  eventQueue.push(event);

  // Trim queue if too large
  while (eventQueue.length > config.maxQueueSize) {
    eventQueue.shift();
  }

  // Auto-flush if batch size reached
  if (eventQueue.length >= config.batchSize) {
    flush();
  }
}

async function flush(): Promise<void> {
  if (eventQueue.length === 0) return;

  const batch = eventQueue.splice(0, config.batchSize);

  for (const event of batch) {
    try {
      // Emit as bus event
      const busEvent = {
        id: event.id,
        topic: `telemetry.client.${event.category}`,
        kind: 'log',
        level: event.severity,
        actor: config.actor,
        ts: event.ts / 1000, // Bus uses seconds
        iso: event.iso,
        data: {
          message: event.message,
          stack: event.stack,
          url: event.url,
          line: event.line,
          column: event.column,
          source: event.source,
          pathname: event.pathname,
          userAgent: event.userAgent,
          ...event.context,
        },
      };

      // Fire and forget - don't block UI
      fetch(config.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(busEvent),
      }).catch(() => {
        // Silently ignore emit failures to avoid recursion
      });
    } catch {
      // Ignore serialization errors
    }
  }
}

// ============================================================================
// ERROR HANDLERS
// ============================================================================

function handleGlobalError(
  message: Event | string,
  source?: string,
  line?: number,
  column?: number,
  error?: Error
): boolean {
  const now = Date.now();

  queueEvent({
    id: generateId(),
    ts: now,
    iso: new Date(now).toISOString(),
    category: 'js_error',
    severity: 'error',
    message: error?.message || String(message),
    stack: error?.stack,
    source,
    line,
    column,
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
  });

  return false; // Don't suppress default handling
}

function handleUnhandledRejection(event: PromiseRejectionEvent): void {
  const now = Date.now();
  const reason = event.reason;

  queueEvent({
    id: generateId(),
    ts: now,
    iso: new Date(now).toISOString(),
    category: 'promise_rejection',
    severity: 'error',
    message: reason?.message || String(reason),
    stack: reason?.stack,
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : undefined,
  });
}

function handleResourceError(event: Event): void {
  const target = event.target as HTMLElement;
  if (!target) return;

  const now = Date.now();
  const src =
    (target as HTMLImageElement).src ||
    (target as HTMLScriptElement).src ||
    (target as HTMLLinkElement).href;

  if (!src) return;

  queueEvent({
    id: generateId(),
    ts: now,
    iso: new Date(now).toISOString(),
    category: 'resource_error',
    severity: 'warn',
    message: `Failed to load resource: ${src}`,
    url: src,
    source: target.tagName.toLowerCase(),
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
  });
}

// ============================================================================
// CONSOLE INTERCEPTION
// ============================================================================

function interceptConsole(): void {
  if (!config.interceptConsole) return;

  // Store originals
  originalConsoleError = console.error;
  originalConsoleWarn = console.warn;

  // Intercept console.error
  console.error = (...args: unknown[]) => {
    // Call original first
    originalConsoleError.apply(console, args);

    const now = Date.now();
    const message = args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');

    // Skip our own telemetry errors
    if (message.includes('telemetry') || message.includes('/api/emit')) return;

    queueEvent({
      id: generateId(),
      ts: now,
      iso: new Date(now).toISOString(),
      category: 'console_error',
      severity: 'error',
      message: message.slice(0, 2000),
      pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
    });
  };

  // Intercept console.warn
  console.warn = (...args: unknown[]) => {
    // Call original first
    originalConsoleWarn.apply(console, args);

    const now = Date.now();
    const message = args.map((a) => (typeof a === 'object' ? JSON.stringify(a) : String(a))).join(' ');

    // Skip noise
    if (message.includes('telemetry') || message.includes('deprecated')) return;

    queueEvent({
      id: generateId(),
      ts: now,
      iso: new Date(now).toISOString(),
      category: 'console_warn',
      severity: 'warn',
      message: message.slice(0, 2000),
      pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
    });
  };
}

// ============================================================================
// FETCH INTERCEPTION
// ============================================================================

function interceptFetch(): void {
  if (!config.interceptFetch) return;
  if (typeof fetch === 'undefined') return;

  originalFetch = fetch;

  globalThis.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    const startTime = performance.now();

    try {
      const response = await originalFetch(input, init);

      // Log slow requests
      const duration = performance.now() - startTime;
      if (duration > 5000 && !url.includes('/api/emit')) {
        queueEvent({
          id: generateId(),
          ts: Date.now(),
          iso: new Date().toISOString(),
          category: 'performance',
          severity: 'warn',
          message: `Slow request: ${url} took ${Math.round(duration)}ms`,
          url,
          context: { duration: Math.round(duration), method: init?.method || 'GET' },
          pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
        });
      }

      // Log failed requests (4xx, 5xx)
      if (!response.ok && !url.includes('/api/emit')) {
        queueEvent({
          id: generateId(),
          ts: Date.now(),
          iso: new Date().toISOString(),
          category: 'fetch_error',
          severity: response.status >= 500 ? 'error' : 'warn',
          message: `HTTP ${response.status} ${response.statusText}: ${url}`,
          url,
          context: {
            status: response.status,
            statusText: response.statusText,
            method: init?.method || 'GET',
          },
          pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
        });
      }

      return response;
    } catch (error) {
      // Network errors, CORS, etc.
      const errName = (error as { name?: string } | null)?.name;
      const isAbort = errName === 'AbortError';
      if (!isAbort && !url.includes('/api/emit')) {
        queueEvent({
          id: generateId(),
          ts: Date.now(),
          iso: new Date().toISOString(),
          category: 'fetch_error',
          severity: 'error',
          message: `Network error: ${url} - ${error instanceof Error ? error.message : String(error)}`,
          url,
          stack: error instanceof Error ? error.stack : undefined,
          pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
        });
      }
      throw error;
    }
  };
}

// ============================================================================
// WEBSOCKET MONITORING
// ============================================================================

export function trackWebSocket(ws: WebSocket, label?: string): void {
  const originalOnError = ws.onerror;
  const originalOnClose = ws.onclose;

  ws.onerror = (event) => {
    queueEvent({
      id: generateId(),
      ts: Date.now(),
      iso: new Date().toISOString(),
      category: 'websocket_error',
      severity: 'error',
      message: `WebSocket error: ${label || ws.url}`,
      url: ws.url,
      context: { label, readyState: ws.readyState },
      pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
    });

    originalOnError?.call(ws, event);
  };

  ws.onclose = (event) => {
    // Only log abnormal closures
    const ignorable = event.code === 1005 && event.wasClean && !event.reason;
    if ((!event.wasClean || event.code !== 1000) && !ignorable) {
      queueEvent({
        id: generateId(),
        ts: Date.now(),
        iso: new Date().toISOString(),
        category: 'websocket_error',
        severity: 'warn',
        message: `WebSocket closed: ${label || ws.url} (code: ${event.code}, reason: ${event.reason || 'none'})`,
        url: ws.url,
        context: {
          label,
          code: event.code,
          reason: event.reason,
          wasClean: event.wasClean,
        },
        pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
      });
    }

    originalOnClose?.call(ws, event);
  };
}

// ============================================================================
// QWIK COMPONENT ERROR REPORTER
// ============================================================================

export function reportComponentError(
  componentName: string,
  error: Error,
  context?: Record<string, unknown>
): void {
  queueEvent({
    id: generateId(),
    ts: Date.now(),
    iso: new Date().toISOString(),
    category: 'component_error',
    severity: 'error',
    message: `Component error in ${componentName}: ${error.message}`,
    stack: error.stack,
    source: componentName,
    context,
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
  });
}

// ============================================================================
// MANUAL LOGGING API
// ============================================================================

export function logError(message: string, context?: Record<string, unknown>): void {
  queueEvent({
    id: generateId(),
    ts: Date.now(),
    iso: new Date().toISOString(),
    category: 'js_error',
    severity: 'error',
    message,
    context,
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
  });
}

export function logWarn(message: string, context?: Record<string, unknown>): void {
  queueEvent({
    id: generateId(),
    ts: Date.now(),
    iso: new Date().toISOString(),
    category: 'console_warn',
    severity: 'warn',
    message,
    context,
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
  });
}

export function logInfo(message: string, context?: Record<string, unknown>): void {
  queueEvent({
    id: generateId(),
    ts: Date.now(),
    iso: new Date().toISOString(),
    category: 'console_warn',
    severity: 'info',
    message,
    context,
    pathname: typeof window !== 'undefined' ? window.location.pathname : undefined,
  });
}

// ============================================================================
// INITIALIZATION
// ============================================================================

export function initErrorCollector(userConfig?: CollectorConfig): void {
  if (isInitialized) return;
  if (typeof window === 'undefined') return;

  config = { ...DEFAULT_CONFIG, ...userConfig };
  isInitialized = true;

  // Global error handler
  window.onerror = handleGlobalError;

  // Unhandled promise rejections
  window.addEventListener('unhandledrejection', handleUnhandledRejection);

  // Resource loading errors (images, scripts, stylesheets)
  window.addEventListener('error', handleResourceError, true);

  // Console interception
  interceptConsole();

  // Fetch interception
  interceptFetch();

  // Periodic flush
  flushTimer = setInterval(flush, config.flushInterval);

  // Flush on page unload
  window.addEventListener('beforeunload', () => {
    flush();
  });

  // Emit initialization event (info, not error)
  queueEvent({
    id: generateId(),
    ts: Date.now(),
    iso: new Date().toISOString(),
    category: 'console_warn',  // Use warn category for info-level startup events
    severity: 'info',
    message: 'Telemetry collector initialized',
    context: {
      userAgent: navigator.userAgent,
      viewport: `${window.innerWidth}x${window.innerHeight}`,
      origin: window.location.origin,
    },
    pathname: window.location.pathname,
  });

  console.log('[telemetry] Error collector initialized');
}

export function destroyErrorCollector(): void {
  if (!isInitialized) return;

  // Restore originals
  if (originalConsoleError) console.error = originalConsoleError;
  if (originalConsoleWarn) console.warn = originalConsoleWarn;
  if (originalFetch) globalThis.fetch = originalFetch;

  // Clear timer
  if (flushTimer) {
    clearInterval(flushTimer);
    flushTimer = null;
  }

  // Final flush
  flush();

  isInitialized = false;
}

// Auto-initialize if in browser
if (typeof window !== 'undefined' && typeof document !== 'undefined') {
  // Wait for DOM ready to avoid race conditions
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => initErrorCollector());
  } else {
    initErrorCollector();
  }
}
