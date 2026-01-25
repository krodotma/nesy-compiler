/**
 * Component Bus - Inter-Component Communication
 *
 * Phase 7, Iteration 57 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Pub/sub messaging between components
 * - Type-safe event definitions
 * - Wildcard subscriptions
 * - Event history for debugging
 * - Memory leak prevention with auto-cleanup
 */

// ============================================================================
// Types
// ============================================================================

export interface ComponentEvent<T = unknown> {
  type: string;
  payload: T;
  source: string;
  timestamp: number;
  id: string;
}

export type EventHandler<T = unknown> = (event: ComponentEvent<T>) => void;

export interface Subscription {
  id: string;
  pattern: string;
  handler: EventHandler;
  source: string;
  createdAt: number;
}

export interface ComponentBusConfig {
  /** Enable event history */
  enableHistory: boolean;
  /** Maximum history size */
  maxHistorySize: number;
  /** Enable debug logging */
  debug: boolean;
  /** Auto-cleanup interval in ms */
  cleanupInterval: number;
}

// ============================================================================
// Event Type Definitions
// ============================================================================

export interface LaneEventMap {
  'lane.selected': { laneId: string | null; previousId: string | null };
  'lane.expanded': { laneId: string; expanded: boolean };
  'lane.updated': { laneId: string; changes: Record<string, unknown> };
  'lane.hovered': { laneId: string | null };
  'lane.focused': { laneId: string; source: 'keyboard' | 'mouse' };
  'lane.action': { laneId: string; action: string; data?: unknown };

  'filter.changed': { filter: Record<string, unknown> };
  'filter.cleared': Record<string, never>;
  'sort.changed': { key: string; direction: 'asc' | 'desc' };

  'view.changed': { mode: string; previousMode: string };
  'view.scrolled': { scrollTop: number; scrollHeight: number };
  'view.resized': { width: number; height: number };

  'section.toggled': { section: string; collapsed: boolean };

  'sync.started': Record<string, never>;
  'sync.completed': { success: boolean; error?: string };
  'sync.conflict': { laneId: string; field: string };

  'undo.executed': { commandId: string };
  'redo.executed': { commandId: string };

  'modal.opened': { modalId: string; props?: unknown };
  'modal.closed': { modalId: string };

  'toast.show': { message: string; type: 'info' | 'success' | 'warning' | 'error' };

  'analytics.event': { name: string; data?: Record<string, unknown> };
}

export type LaneEventType = keyof LaneEventMap;

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: ComponentBusConfig = {
  enableHistory: true,
  maxHistorySize: 100,
  debug: false,
  cleanupInterval: 60000,
};

// ============================================================================
// Component Bus
// ============================================================================

export class ComponentBus {
  private config: ComponentBusConfig;
  private subscriptions: Map<string, Subscription> = new Map();
  private history: ComponentEvent[] = [];
  private cleanupTimer: ReturnType<typeof setInterval> | null = null;
  private nextSubscriptionId = 1;

  constructor(config: Partial<ComponentBusConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };

    if (this.config.cleanupInterval > 0) {
      this.startCleanupTimer();
    }
  }

  // ============================================================================
  // Public Methods
  // ============================================================================

  /**
   * Emit an event
   */
  emit<K extends LaneEventType>(
    type: K,
    payload: LaneEventMap[K],
    source: string = 'unknown'
  ): void {
    const event: ComponentEvent<LaneEventMap[K]> = {
      type,
      payload,
      source,
      timestamp: Date.now(),
      id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    };

    this.dispatchEvent(event);
  }

  /**
   * Emit a custom event (not in LaneEventMap)
   */
  emitCustom<T>(type: string, payload: T, source: string = 'unknown'): void {
    const event: ComponentEvent<T> = {
      type,
      payload,
      source,
      timestamp: Date.now(),
      id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    };

    this.dispatchEvent(event);
  }

  /**
   * Subscribe to events matching a pattern
   */
  subscribe<K extends LaneEventType>(
    pattern: K | string,
    handler: EventHandler<K extends LaneEventType ? LaneEventMap[K] : unknown>,
    source: string = 'unknown'
  ): () => void {
    const id = `sub-${this.nextSubscriptionId++}`;

    const subscription: Subscription = {
      id,
      pattern,
      handler: handler as EventHandler,
      source,
      createdAt: Date.now(),
    };

    this.subscriptions.set(id, subscription);

    if (this.config.debug) {
      console.log(`[ComponentBus] Subscription added: ${pattern} from ${source}`);
    }

    // Return unsubscribe function
    return () => {
      this.subscriptions.delete(id);
      if (this.config.debug) {
        console.log(`[ComponentBus] Subscription removed: ${pattern}`);
      }
    };
  }

  /**
   * Subscribe to multiple patterns
   */
  subscribeMany(
    patterns: string[],
    handler: EventHandler,
    source: string = 'unknown'
  ): () => void {
    const unsubscribers = patterns.map(pattern =>
      this.subscribe(pattern as LaneEventType, handler, source)
    );

    return () => {
      unsubscribers.forEach(unsub => unsub());
    };
  }

  /**
   * Subscribe once (auto-unsubscribe after first event)
   */
  once<K extends LaneEventType>(
    pattern: K,
    handler: EventHandler<LaneEventMap[K]>,
    source: string = 'unknown'
  ): () => void {
    let unsubscribe: (() => void) | null = null;

    unsubscribe = this.subscribe(
      pattern,
      (event) => {
        handler(event as ComponentEvent<LaneEventMap[K]>);
        if (unsubscribe) {
          unsubscribe();
        }
      },
      source
    );

    return unsubscribe;
  }

  /**
   * Get event history
   */
  getHistory(): ComponentEvent[] {
    return [...this.history];
  }

  /**
   * Get filtered history
   */
  getHistoryByType(type: string): ComponentEvent[] {
    return this.history.filter(e => this.matchPattern(type, e.type));
  }

  /**
   * Get history by source
   */
  getHistoryBySource(source: string): ComponentEvent[] {
    return this.history.filter(e => e.source === source);
  }

  /**
   * Clear event history
   */
  clearHistory(): void {
    this.history = [];
  }

  /**
   * Get subscription count
   */
  getSubscriptionCount(): number {
    return this.subscriptions.size;
  }

  /**
   * Get all subscriptions (for debugging)
   */
  getSubscriptions(): Array<{ id: string; pattern: string; source: string }> {
    return Array.from(this.subscriptions.values()).map(sub => ({
      id: sub.id,
      pattern: sub.pattern,
      source: sub.source,
    }));
  }

  /**
   * Dispose the bus
   */
  dispose(): void {
    this.subscriptions.clear();
    this.history = [];
    if (this.cleanupTimer) {
      clearInterval(this.cleanupTimer);
      this.cleanupTimer = null;
    }
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private dispatchEvent(event: ComponentEvent): void {
    // Add to history
    if (this.config.enableHistory) {
      this.history.push(event);
      while (this.history.length > this.config.maxHistorySize) {
        this.history.shift();
      }
    }

    // Debug logging
    if (this.config.debug) {
      console.log(`[ComponentBus] Event: ${event.type}`, event.payload);
    }

    // Dispatch to matching subscribers
    for (const subscription of this.subscriptions.values()) {
      if (this.matchPattern(subscription.pattern, event.type)) {
        try {
          subscription.handler(event);
        } catch (err) {
          console.error(`[ComponentBus] Handler error for ${event.type}:`, err);
        }
      }
    }
  }

  private matchPattern(pattern: string, eventType: string): boolean {
    // Exact match
    if (pattern === eventType) return true;

    // Wildcard patterns
    if (pattern === '*') return true;

    // Prefix wildcard (e.g., "lane.*")
    if (pattern.endsWith('.*')) {
      const prefix = pattern.slice(0, -2);
      return eventType.startsWith(prefix + '.');
    }

    // Suffix wildcard (e.g., "*.changed")
    if (pattern.startsWith('*.')) {
      const suffix = pattern.slice(2);
      return eventType.endsWith('.' + suffix);
    }

    // Regex pattern (if wrapped in slashes)
    if (pattern.startsWith('/') && pattern.endsWith('/')) {
      try {
        const regex = new RegExp(pattern.slice(1, -1));
        return regex.test(eventType);
      } catch {
        return false;
      }
    }

    return false;
  }

  private startCleanupTimer(): void {
    this.cleanupTimer = setInterval(() => {
      // Check for stale subscriptions
      const now = Date.now();
      const staleThreshold = 3600000; // 1 hour

      // Note: We don't auto-remove subscriptions by age since components
      // may legitimately hold long-lived subscriptions. This is just a
      // placeholder for future cleanup logic if needed.

      if (this.config.debug) {
        console.log(`[ComponentBus] Cleanup check: ${this.subscriptions.size} subscriptions`);
      }
    }, this.config.cleanupInterval);
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalBus: ComponentBus | null = null;

export function getGlobalComponentBus(config?: Partial<ComponentBusConfig>): ComponentBus {
  if (!globalBus) {
    globalBus = new ComponentBus(config);
  }
  return globalBus;
}

export function resetGlobalComponentBus(): void {
  if (globalBus) {
    globalBus.dispose();
  }
  globalBus = null;
}

// ============================================================================
// Helper Hooks (for use in components)
// ============================================================================

/**
 * Create a namespaced emitter
 */
export function createNamespacedEmitter(bus: ComponentBus, namespace: string) {
  return {
    emit: <K extends LaneEventType>(
      type: K,
      payload: LaneEventMap[K]
    ) => bus.emit(type, payload, namespace),

    emitCustom: <T>(type: string, payload: T) =>
      bus.emitCustom(type, payload, namespace),
  };
}

/**
 * Create a subscription manager that auto-cleans on disposal
 */
export function createSubscriptionManager(bus: ComponentBus, source: string) {
  const unsubscribers: Array<() => void> = [];

  return {
    subscribe: <K extends LaneEventType>(
      pattern: K,
      handler: EventHandler<LaneEventMap[K]>
    ) => {
      const unsub = bus.subscribe(pattern, handler, source);
      unsubscribers.push(unsub);
      return unsub;
    },

    subscribeMany: (patterns: string[], handler: EventHandler) => {
      const unsub = bus.subscribeMany(patterns, handler, source);
      unsubscribers.push(unsub);
      return unsub;
    },

    disposeAll: () => {
      unsubscribers.forEach(unsub => unsub());
      unsubscribers.length = 0;
    },
  };
}

export default ComponentBus;
