/**
 * Load Timing Collector - Captures page load metrics and Web Vitals
 * Emits timing data to Pluribus bus for benchmarking and optimization.
 *
 * DKIN v25 compliant - emits standardized bus events with ISO timestamps.
 */

export interface LoadTimingMetrics {
  // Navigation Timing API
  ttfb: number;              // Time to First Byte
  domContentLoaded: number;  // DOMContentLoaded event
  domComplete: number;       // DOM fully parsed
  loadEvent: number;         // window.onload fired

  // Core Web Vitals
  fcp: number | null;        // First Contentful Paint
  lcp: number | null;        // Largest Contentful Paint
  cls: number | null;        // Cumulative Layout Shift

  // Qwik-specific
  hydrationStart: number | null;
  hydrationEnd: number | null;

  // Resource breakdown
  scriptCount: number;
  scriptBytes: number;
  styleCount: number;
  styleBytes: number;

  // Tags for segmentation
  tags: string[];
}

export interface TimingEvent {
  topic: string;
  kind: string;
  level: string;
  actor: string;
  data: Record<string, unknown>;
  iso: string;
  ts: number;
}

let metricsCollected = false;
let hydrationStartMark: number | null = null;
let hydrationEndMark: number | null = null;
const timingTags: string[] = [];

/**
 * Add a tag to segment timing data
 */
export function addTimingTag(tag: string): void {
  if (!timingTags.includes(tag)) {
    timingTags.push(tag);
  }
}

/**
 * Mark hydration start (call when Qwik begins resumability)
 */
export function markHydrationStart(): void {
  hydrationStartMark = performance.now();
  addTimingTag('hydration-measured');
}

/**
 * Mark hydration end (call when app is interactive)
 */
export function markHydrationEnd(): void {
  hydrationEndMark = performance.now();
  
  // Defer __pluribusReady until document is fully loaded
  // This ensures all useVisibleTask$ hooks have run
  const setReady = () => {
    console.log("%c[HYDRATION] UI is now interactive", "color: #2ecc71; font-weight: bold;");
    (window as any).__pluribusReady = true;
  };
  
  if (document.readyState === "complete") {
    // Document already loaded, add small delay for Qwik hydration
    setTimeout(setReady, 1500);
  } else {
    // Wait for load event, then add delay
    window.addEventListener("load", () => {
      setTimeout(setReady, 1500);
    }, { once: true });
  }
}

/**
 * Collect Navigation Timing API metrics
 */
function collectNavigationTiming(): Partial<LoadTimingMetrics> {
  const navEntry = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming | undefined;

  if (!navEntry) {
    return {};
  }

  return {
    ttfb: Math.round(navEntry.responseStart - navEntry.requestStart),
    domContentLoaded: Math.round(navEntry.domContentLoadedEventEnd - navEntry.startTime),
    domComplete: Math.round(navEntry.domComplete - navEntry.startTime),
    loadEvent: Math.round(navEntry.loadEventEnd - navEntry.startTime),
  };
}

/**
 * Collect resource timing breakdown
 */
function collectResourceTiming(): Pick<LoadTimingMetrics, 'scriptCount' | 'scriptBytes' | 'styleCount' | 'styleBytes'> {
  const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];

  let scriptCount = 0;
  let scriptBytes = 0;
  let styleCount = 0;
  let styleBytes = 0;

  for (const resource of resources) {
    const size = resource.transferSize || resource.encodedBodySize || 0;

    if (resource.initiatorType === 'script' || resource.name.endsWith('.js')) {
      scriptCount++;
      scriptBytes += size;
    } else if (resource.initiatorType === 'css' || resource.name.endsWith('.css')) {
      styleCount++;
      styleBytes += size;
    }
  }

  return { scriptCount, scriptBytes, styleCount, styleBytes };
}

/**
 * Emit timing event to bus via /api/emit
 */
export async function emitTimingEvent(metrics: LoadTimingMetrics | Record<string, unknown>, topic: string = 'telemetry.page.load'): Promise<void> {
  const event: TimingEvent = {
    topic,
    kind: 'metric',
    level: 'info',
    actor: 'load-timing',
    iso: new Date().toISOString(),
    ts: Date.now() / 1000,
    data: {
      ...metrics,
      url: window.location.pathname,
      userAgent: navigator.userAgent,
      connection: (navigator as any).connection?.effectiveType || 'unknown',
    },
  };

  try {
    await fetch('/api/emit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event),
    });
  } catch (err) {
    console.warn('[load-timing] Failed to emit:', err);
  }
}

/**
 * Stop component timing and log final result
 */
export function stopComponentTiming(componentName: string): number | null {
  const timer = activeTimers.get(componentName);
  if (!timer) {
    console.warn(`[${componentName}] ⚠️ no active timer found`);
    return null;
  }

  // Clear interval
  if (timer.intervalId) {
    clearInterval(timer.intervalId);
  }

  const duration = performance.now() - timer.startTime;
  const durationSec = (duration / 1000).toFixed(2);
  activeTimers.delete(componentName);

  // Store for summary
  completedTimings.push({ name: componentName, duration, timestamp: Date.now() });

  // Color based on duration
  const color = duration < 500 ? '#2ecc71' : duration < 2000 ? '#f39c12' : '#e74c3c';
  const status = duration < 500 ? '✓ FAST' : duration < 2000 ? '⚡ OK' : '🐢 SLOW';

  console.log(
    `%c[${componentName}] ${status}: ${durationSec}s done`,
    `color: ${color}; font-weight: bold; font-size: 12px;`
  );

  // Also mark in performance API
  performance.mark(`pluribus:${componentName}:end`);
  addTimingTag(`component:${componentName}:${Math.round(duration)}ms`);

  // Emit real-time bus event for the UI panel
  emitTimingEvent({
    component: componentName,
    duration: Math.round(duration),
    status: status.replace(/✓ |⚡ |🐢 /g, ''),
    timestamp: Date.now()
  }, 'telemetry.component.load');

  return duration;
}

/**
 * Collect Web Vitals using PerformanceObserver
 */
function observeWebVitals(callback: (vitals: Partial<LoadTimingMetrics>) => void): void {
  const vitals: Partial<LoadTimingMetrics> = {
    fcp: null,
    lcp: null,
    cls: 0,
  };

  let clsValue = 0;

  // First Contentful Paint
  try {
    const fcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      for (const entry of entries) {
        if (entry.name === 'first-contentful-paint') {
          vitals.fcp = Math.round(entry.startTime);
          addTimingTag('fcp-captured');
        }
      }
    });
    fcpObserver.observe({ entryTypes: ['paint'] });
  } catch (e) {
    // Not supported
  }

  // Largest Contentful Paint
  try {
    const lcpObserver = new PerformanceObserver((list) => {
      const entries = list.getEntries();
      const lastEntry = entries[entries.length - 1] as any;
      if (lastEntry) {
        vitals.lcp = Math.round(lastEntry.startTime);
        addTimingTag('lcp-captured');
      }
    });
    lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
  } catch (e) {
    // Not supported
  }

  // Cumulative Layout Shift
  try {
    const clsObserver = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        const layoutShift = entry as any;
        if (!layoutShift.hadRecentInput) {
          clsValue += layoutShift.value;
          vitals.cls = Math.round(clsValue * 1000) / 1000;
          addTimingTag('cls-captured');
        }
      }
    });
    clsObserver.observe({ entryTypes: ['layout-shift'] });
  } catch (e) {
    // Not supported
  }

  // Collect after load event + buffer time
  window.addEventListener('load', () => {
    setTimeout(() => {
      callback(vitals);
    }, 500);
  });
}

/**
 * Initialize load timing collection
 * Call this once at app boot (e.g., in root.tsx or entry.client.tsx)
 */
export function initLoadTiming(): void {
  if (typeof window === 'undefined' || metricsCollected) {
    return;
  }

  addTimingTag('load-timing-init');

  // Observe Web Vitals asynchronously
  let webVitals: Partial<LoadTimingMetrics> = {};
  observeWebVitals((vitals) => {
    webVitals = vitals;
  });

  // Collect metrics after page load
  const collectAndEmit = () => {
    if (metricsCollected) return;
    metricsCollected = true;

    const navTiming = collectNavigationTiming();
    const resourceTiming = collectResourceTiming();

    const metrics: LoadTimingMetrics = {
      ttfb: navTiming.ttfb || 0,
      domContentLoaded: navTiming.domContentLoaded || 0,
      domComplete: navTiming.domComplete || 0,
      loadEvent: navTiming.loadEvent || 0,
      fcp: webVitals.fcp || null,
      lcp: webVitals.lcp || null,
      cls: webVitals.cls ?? null,
      hydrationStart: hydrationStartMark,
      hydrationEnd: hydrationEndMark,
      scriptCount: resourceTiming.scriptCount,
      scriptBytes: resourceTiming.scriptBytes,
      styleCount: resourceTiming.styleCount,
      styleBytes: resourceTiming.styleBytes,
      tags: [...timingTags],
    };

    emitTimingEvent(metrics);
  };

  // Wait for load event + small buffer for LCP
  if (document.readyState === 'complete') {
    setTimeout(collectAndEmit, 1000);
  } else {
    window.addEventListener('load', () => {
      setTimeout(collectAndEmit, 1000);
    });
  }
}

/**
 * Manual timing mark for component-level benchmarking
 */
export function markTiming(label: string): void {
  performance.mark(`pluribus:${label}`);
  addTimingTag(`mark:${label}`);
}

/**
 * Measure between two timing marks
 */
export function measureTiming(name: string, startMark: string, endMark: string): number | null {
  try {
    performance.measure(`pluribus:${name}`, `pluribus:${startMark}`, `pluribus:${endMark}`);
    const measures = performance.getEntriesByName(`pluribus:${name}`, 'measure');
    return measures.length > 0 ? Math.round(measures[0].duration) : null;
  } catch {
    return null;
  }
}

// ============================================================================
// GRANULAR COMPONENT TIMING - Console output for real-time visibility
// ============================================================================

interface ComponentTimer {
  name: string;
  startTime: number;
  intervalId: ReturnType<typeof setInterval> | null;
}

const activeTimers = new Map<string, ComponentTimer>();
const completedTimings: Array<{ name: string; duration: number; timestamp: number }> = [];

/**
 * Start timing a component load - shows live progress in console
 * Usage: const stop = startComponentTiming('SphereAtlas');
 *        // ... load component ...
 *        stop();
 */
export function startComponentTiming(componentName: string): () => void {
  const startTime = performance.now();
  const startTimestamp = Date.now();

  // Live progress indicator
  let elapsed = 0;
  const intervalId = setInterval(() => {
    elapsed = Number(((performance.now() - startTime) / 1000).toFixed(1));
    // Update console with progress bar
    const bars = Math.min(Math.floor(Number(elapsed) * 2), 20);
    const progress = '█'.repeat(bars) + '░'.repeat(20 - bars);
    console.log(
      `%c[${componentName}] loading ${progress} ${elapsed}s`,
      'color: #888; font-family: monospace;'
    );
  }, 500);

  const timer: ComponentTimer = { name: componentName, startTime, intervalId };
  activeTimers.set(componentName, timer);

  // Log start
  console.log(
    `%c[${componentName}] ⏳ loading started @ ${new Date(startTimestamp).toISOString().slice(11, 23)}`,
    'color: #3498db; font-weight: bold;'
  );

  // Return stop function
  return () => stopComponentTiming(componentName);
}



/**
 * Async wrapper - times an async operation automatically
 * Usage: const result = await timeAsync('ThreeJS', () => import('three'));
 */
export async function timeAsync<T>(label: string, fn: () => Promise<T>): Promise<T> {
  const stop = startComponentTiming(label);
  try {
    return await fn();
  } finally {
    stop();
  }
}

/**
 * Log summary of all component timings to console
 */
export function logTimingSummary(): void {
  if (completedTimings.length === 0) {
    console.log('%c[TIMING] No component timings recorded', 'color: #888;');
    return;
  }

  const total = completedTimings.reduce((sum, t) => sum + t.duration, 0);
  const sorted = [...completedTimings].sort((a, b) => b.duration - a.duration);

  console.group('%c[TIMING SUMMARY] Component Load Times', 'color: #9b59b6; font-weight: bold;');

  console.log('%cComponent Breakdown:', 'font-weight: bold;');
  for (const timing of sorted) {
    const pct = ((timing.duration / total) * 100).toFixed(1);
    const bar = '█'.repeat(Math.round(Number(pct) / 5));
    const color = timing.duration < 500 ? '#2ecc71' : timing.duration < 2000 ? '#f39c12' : '#e74c3c';
    console.log(
      `%c  ${timing.name.padEnd(25)} ${(timing.duration / 1000).toFixed(2)}s ${bar} ${pct}%`,
      `color: ${color}; font-family: monospace;`
    );
  }

  console.log(`%c  ${'─'.repeat(50)}`, 'color: #888;');
  console.log(
    `%c  TOTAL: ${(total / 1000).toFixed(2)}s across ${completedTimings.length} components`,
    'font-weight: bold;'
  );

  console.groupEnd();
}

/**
 * Export completed timings for external analysis
 */
export function getCompletedTimings(): typeof completedTimings {
  return [...completedTimings];
}

// Auto-log summary after page is fully interactive
if (typeof window !== 'undefined') {
  window.addEventListener('load', () => {
    // Wait for all lazy-loaded components
    setTimeout(() => {
      if (completedTimings.length > 0) {
        logTimingSummary();
      }
    }, 5000);
  });
}

/**
 * Phase 5: Auto-instrumented dynamic import wrapper
 * Wraps dynamic imports to automatically capture timing
 * 
 * Usage: const Three = await instrumentedImport(() => import("three"), "three");
 */
export async function instrumentedImport<T>(
  importFn: () => Promise<T>,
  moduleName: string
): Promise<T> {
  const stop = startComponentTiming(`import:${moduleName}`);
  try {
    const module = await importFn();
    stop();
    
    // Emit to bus if available
    if (typeof window !== "undefined" && (window as any).__pluribusBus) {
      (window as any).__pluribusBus.emit("telemetry.dynamic.import", {
        module: moduleName,
        duration: completedTimings.find(t => t.name === `import:${moduleName}`)?.duration,
        timestamp: Date.now()
      });
    }
    
    return module;
  } catch (error) {
    stop();
    throw error;
  }
}

/**
 * Component boundary timing decorator for Qwik components
 * Adds timing around component render lifecycle
 */
export function withTiming<T extends (...args: any[]) => any>(
  fn: T,
  componentName: string
): T {
  return ((...args: Parameters<T>): ReturnType<T> => {
    const stop = startComponentTiming(`render:${componentName}`);
    try {
      const result = fn(...args);
      if (result instanceof Promise) {
        return result.finally(stop) as ReturnType<T>;
      }
      stop();
      return result;
    } catch (error) {
      stop();
      throw error;
    }
  }) as T;
}
