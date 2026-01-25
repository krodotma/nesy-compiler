/**
 * Performance Utilities for Lanes
 *
 * Phase 2, Iteration 14 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Virtual scrolling helpers
 * - Debouncing utilities
 * - Lazy loading support
 * - Memoization helpers
 */

// ============================================================================
// Debounce / Throttle
// ============================================================================

/**
 * Debounce a function call
 */
export function debounce<T extends (...args: any[]) => any>(
  fn: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return function (this: any, ...args: Parameters<T>) {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
    timeoutId = setTimeout(() => {
      fn.apply(this, args);
      timeoutId = null;
    }, delay);
  };
}

/**
 * Throttle a function call
 */
export function throttle<T extends (...args: any[]) => any>(
  fn: T,
  limit: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  let timeoutId: ReturnType<typeof setTimeout> | null = null;

  return function (this: any, ...args: Parameters<T>) {
    const now = Date.now();
    const remaining = limit - (now - lastCall);

    if (remaining <= 0) {
      if (timeoutId) {
        clearTimeout(timeoutId);
        timeoutId = null;
      }
      lastCall = now;
      fn.apply(this, args);
    } else if (!timeoutId) {
      timeoutId = setTimeout(() => {
        lastCall = Date.now();
        timeoutId = null;
        fn.apply(this, args);
      }, remaining);
    }
  };
}

// ============================================================================
// Virtual Scrolling
// ============================================================================

export interface VirtualScrollConfig {
  /** Total number of items */
  totalItems: number;
  /** Height of each item in pixels */
  itemHeight: number;
  /** Height of the viewport in pixels */
  viewportHeight: number;
  /** Current scroll position */
  scrollTop: number;
  /** Number of items to render beyond viewport (buffer) */
  overscan?: number;
}

export interface VirtualScrollResult {
  /** Index of first item to render */
  startIndex: number;
  /** Index of last item to render (exclusive) */
  endIndex: number;
  /** Number of items to render */
  visibleCount: number;
  /** Offset in pixels for the first item */
  offsetTop: number;
  /** Total height of the scrollable content */
  totalHeight: number;
  /** Items currently visible in viewport */
  visibleItems: number[];
}

/**
 * Calculate virtual scroll parameters
 */
export function calculateVirtualScroll(config: VirtualScrollConfig): VirtualScrollResult {
  const { totalItems, itemHeight, viewportHeight, scrollTop, overscan = 3 } = config;

  // Calculate visible range
  const startIndex = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
  const visibleCount = Math.ceil(viewportHeight / itemHeight) + overscan * 2;
  const endIndex = Math.min(totalItems, startIndex + visibleCount);

  // Calculate offset for positioning
  const offsetTop = startIndex * itemHeight;
  const totalHeight = totalItems * itemHeight;

  // List of visible item indices
  const visibleItems: number[] = [];
  for (let i = startIndex; i < endIndex; i++) {
    visibleItems.push(i);
  }

  return {
    startIndex,
    endIndex,
    visibleCount: endIndex - startIndex,
    offsetTop,
    totalHeight,
    visibleItems,
  };
}

/**
 * Create a scroll handler for virtual scrolling
 */
export function createVirtualScrollHandler(
  config: Omit<VirtualScrollConfig, 'scrollTop'>,
  onUpdate: (result: VirtualScrollResult) => void
): (scrollTop: number) => void {
  let lastResult: VirtualScrollResult | null = null;

  return throttle((scrollTop: number) => {
    const result = calculateVirtualScroll({ ...config, scrollTop });

    // Only update if range changed
    if (
      !lastResult ||
      lastResult.startIndex !== result.startIndex ||
      lastResult.endIndex !== result.endIndex
    ) {
      lastResult = result;
      onUpdate(result);
    }
  }, 16); // ~60fps
}

// ============================================================================
// Lazy Loading
// ============================================================================

export interface LazyLoadConfig {
  /** Initial number of items to load */
  initialCount: number;
  /** Number of items to load on each trigger */
  batchSize: number;
  /** Distance from bottom to trigger load (in pixels) */
  threshold: number;
}

export interface LazyLoadState<T> {
  items: T[];
  loaded: number;
  hasMore: boolean;
  loading: boolean;
}

/**
 * Check if we should trigger lazy loading
 */
export function shouldLoadMore(
  scrollTop: number,
  scrollHeight: number,
  clientHeight: number,
  threshold: number = 200
): boolean {
  return scrollTop + clientHeight >= scrollHeight - threshold;
}

/**
 * Create lazy load handler
 */
export function createLazyLoadHandler<T>(
  allItems: T[],
  config: LazyLoadConfig,
  onUpdate: (state: LazyLoadState<T>) => void
): {
  getState: () => LazyLoadState<T>;
  loadMore: () => void;
  reset: () => void;
  handleScroll: (scrollTop: number, scrollHeight: number, clientHeight: number) => void;
} {
  let state: LazyLoadState<T> = {
    items: allItems.slice(0, config.initialCount),
    loaded: Math.min(config.initialCount, allItems.length),
    hasMore: allItems.length > config.initialCount,
    loading: false,
  };

  const loadMore = () => {
    if (state.loading || !state.hasMore) return;

    state = { ...state, loading: true };
    onUpdate(state);

    // Simulate async load (can be replaced with actual async operation)
    setTimeout(() => {
      const newLoaded = Math.min(state.loaded + config.batchSize, allItems.length);
      state = {
        items: allItems.slice(0, newLoaded),
        loaded: newLoaded,
        hasMore: newLoaded < allItems.length,
        loading: false,
      };
      onUpdate(state);
    }, 0);
  };

  const reset = () => {
    state = {
      items: allItems.slice(0, config.initialCount),
      loaded: Math.min(config.initialCount, allItems.length),
      hasMore: allItems.length > config.initialCount,
      loading: false,
    };
    onUpdate(state);
  };

  const handleScroll = debounce(
    (scrollTop: number, scrollHeight: number, clientHeight: number) => {
      if (shouldLoadMore(scrollTop, scrollHeight, clientHeight, config.threshold)) {
        loadMore();
      }
    },
    100
  );

  return {
    getState: () => state,
    loadMore,
    reset,
    handleScroll,
  };
}

// ============================================================================
// Memoization
// ============================================================================

/**
 * Simple memoization for single-argument functions
 */
export function memoize<T, R>(
  fn: (arg: T) => R,
  keyFn: (arg: T) => string = (arg) => JSON.stringify(arg)
): (arg: T) => R {
  const cache = new Map<string, R>();

  return (arg: T) => {
    const key = keyFn(arg);
    if (cache.has(key)) {
      return cache.get(key)!;
    }
    const result = fn(arg);
    cache.set(key, result);
    return result;
  };
}

/**
 * LRU cache for memoization with size limit
 */
export class LRUCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;

  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    if (!this.cache.has(key)) return undefined;
    // Move to end (most recently used)
    const value = this.cache.get(key)!;
    this.cache.delete(key);
    this.cache.set(key, value);
    return value;
  }

  set(key: K, value: V): void {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    } else if (this.cache.size >= this.maxSize) {
      // Delete oldest (first) entry
      const firstKey = this.cache.keys().next().value;
      if (firstKey !== undefined) {
        this.cache.delete(firstKey);
      }
    }
    this.cache.set(key, value);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

/**
 * Memoize with LRU cache
 */
export function memoizeLRU<T, R>(
  fn: (arg: T) => R,
  maxSize: number = 100,
  keyFn: (arg: T) => string = (arg) => JSON.stringify(arg)
): (arg: T) => R {
  const cache = new LRUCache<string, R>(maxSize);

  return (arg: T) => {
    const key = keyFn(arg);
    const cached = cache.get(key);
    if (cached !== undefined) {
      return cached;
    }
    const result = fn(arg);
    cache.set(key, result);
    return result;
  };
}

// ============================================================================
// Request Animation Frame Helpers
// ============================================================================

/**
 * Schedule a function to run on the next animation frame
 */
export function scheduleFrame(fn: () => void): number {
  if (typeof requestAnimationFrame !== 'undefined') {
    return requestAnimationFrame(fn);
  }
  return setTimeout(fn, 16) as unknown as number;
}

/**
 * Cancel a scheduled frame
 */
export function cancelFrame(id: number): void {
  if (typeof cancelAnimationFrame !== 'undefined') {
    cancelAnimationFrame(id);
  } else {
    clearTimeout(id);
  }
}

/**
 * Batch multiple updates into a single frame
 */
export function batchUpdates<T>(
  updates: Array<() => T>,
  callback: (results: T[]) => void
): void {
  scheduleFrame(() => {
    const results = updates.map(fn => fn());
    callback(results);
  });
}
