/**
 * @ark/bus - Isomorphic Event Bus for Neo Pluribus
 *
 * This package provides an isomorphic event bus that works identically
 * in browser and Node.js environments. It implements:
 * - NDJSON event streaming
 * - Topic-based publish/subscribe
 * - DKIN v29 protocol compliance
 * - Append-only evidence trail
 * - Bus partitioning with proper topic bucketing
 *
 * @module
 */

import type { Holon, Ring } from '@ark/core';

/**
 * Event levels matching DKIN v29
 */
export type EventLevel = 'info' | 'warn' | 'error' | 'debug' | 'trace';

/**
 * Event kinds matching DKIN v29
 */
export type EventKind =
  | 'event'
  | 'request'
  | 'response'
  | 'heartbeat'
  | 'handshake'
  | 'dissociation';

/**
 * Bus event structure (DKIN v29 compliant)
 */
export interface BusEvent<T = unknown> {
  /** Event ID (UUID) */
  id: string;
  /** Topic for routing (e.g., 'agent.action', 'a2a.handshake') */
  topic: string;
  /** Event kind */
  kind: EventKind;
  /** Event level */
  level: EventLevel;
  /** Actor emitting the event */
  actor: string;
  /** Timestamp in milliseconds */
  timestamp: number;
  /** Protocol version */
  protocol: string;
  /** Ring level of actor */
  ring: Ring;
  /** Event payload */
  data: T;
  /** Correlation ID for request/response chains */
  correlationId?: string;
  /** Parent event ID for causal chains */
  parentId?: string;
  /** Associated Holon ID */
  holonId?: string;
}

/**
 * Topic subscription handler
 */
export type TopicHandler<T = unknown> = (event: BusEvent<T>) => void | Promise<void>;

/**
 * Topic filter function
 */
export type TopicFilter = (topic: string) => boolean;

/**
 * Subscription handle for cleanup
 */
export interface Subscription {
  /** Unsubscribe from this topic */
  unsubscribe: () => void;
  /** Topic pattern subscribed to */
  pattern: string;
  /** Handler function */
  handler: TopicHandler;
}

/**
 * Bus configuration
 */
export interface BusConfig {
  /** Actor name for this bus instance */
  actor: string;
  /** Ring level for this actor */
  ring: Ring;
  /** Protocol version */
  protocol?: string;
  /** Enable debug logging */
  debug?: boolean;
  /** Bus directory for file-based storage (Node.js only) */
  busDir?: string;
  /** Maximum events to buffer before flushing */
  bufferSize?: number;
  /** Flush interval in milliseconds */
  flushInterval?: number;
}

/**
 * Topic partitions for proper bucketing
 * Includes antigravity.* topics per Gemini's analysis
 */
export const TOPIC_PARTITIONS = {
  // Ring 0 - Kernel topics
  kernel: ['axiom.*', 'constitution.*', 'dna.*'],

  // Ring 1 - Infrastructure topics
  infrastructure: [
    'a2a.*',
    'mcp.*',
    'bus.*',
    'daemon.*',
    'antigravity.*', // Critical: Gemini found this was missing
  ],

  // Ring 2 - Agent topics
  agent: ['agent.*', 'task.*', 'tool.*', 'iterate.*'],

  // Ring 3 - Ephemeral topics
  ephemeral: ['paip.*', 'clone.*', 'sandbox.*'],
} as const;

/**
 * Create a topic matcher for glob patterns
 */
export function createTopicMatcher(pattern: string): TopicFilter {
  // Convert glob to regex
  const regexPattern = pattern
    .replace(/\./g, '\\.')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');

  const regex = new RegExp(`^${regexPattern}$`);
  return (topic: string) => regex.test(topic);
}

/**
 * Abstract Bus class - base for isomorphic implementations
 */
export abstract class Bus {
  protected config: Required<BusConfig>;
  protected subscriptions: Map<string, Set<Subscription>> = new Map();
  protected eventBuffer: BusEvent[] = [];

  constructor(config: BusConfig) {
    this.config = {
      actor: config.actor,
      ring: config.ring,
      protocol: config.protocol ?? 'DKIN v29',
      debug: config.debug ?? false,
      busDir: config.busDir ?? '',
      bufferSize: config.bufferSize ?? 100,
      flushInterval: config.flushInterval ?? 1000,
    };
  }

  /**
   * Emit an event to the bus
   */
  abstract emit<T>(topic: string, data: T, options?: Partial<BusEvent>): Promise<BusEvent<T>>;

  /**
   * Subscribe to a topic pattern
   */
  subscribe<T = unknown>(pattern: string, handler: TopicHandler<T>): Subscription {
    const matcher = createTopicMatcher(pattern);

    const subscription: Subscription = {
      pattern,
      handler: handler as TopicHandler,
      unsubscribe: () => {
        const subs = this.subscriptions.get(pattern);
        if (subs) {
          subs.delete(subscription);
          if (subs.size === 0) {
            this.subscriptions.delete(pattern);
          }
        }
      },
    };

    if (!this.subscriptions.has(pattern)) {
      this.subscriptions.set(pattern, new Set());
    }
    this.subscriptions.get(pattern)!.add(subscription);

    return subscription;
  }

  /**
   * Dispatch event to matching subscribers
   */
  protected async dispatch(event: BusEvent): Promise<void> {
    for (const [pattern, subs] of this.subscriptions) {
      const matcher = createTopicMatcher(pattern);
      if (matcher(event.topic)) {
        for (const sub of subs) {
          try {
            await sub.handler(event);
          } catch (error) {
            if (this.config.debug) {
              console.error(`Handler error for ${pattern}:`, error);
            }
          }
        }
      }
    }
  }

  /**
   * Create a new event with all required fields
   */
  protected createEvent<T>(
    topic: string,
    data: T,
    options?: Partial<BusEvent>
  ): BusEvent<T> {
    return {
      id: crypto.randomUUID(),
      topic,
      kind: options?.kind ?? 'event',
      level: options?.level ?? 'info',
      actor: this.config.actor,
      timestamp: Date.now(),
      protocol: this.config.protocol,
      ring: this.config.ring,
      data,
      correlationId: options?.correlationId,
      parentId: options?.parentId,
      holonId: options?.holonId,
    };
  }

  /**
   * Get the topic partition for a given topic
   */
  getPartition(topic: string): string | null {
    for (const [partition, patterns] of Object.entries(TOPIC_PARTITIONS)) {
      for (const pattern of patterns) {
        const matcher = createTopicMatcher(pattern);
        if (matcher(topic)) {
          return partition;
        }
      }
    }
    return null;
  }

  /**
   * Flush buffered events
   */
  abstract flush(): Promise<void>;

  /**
   * Close the bus and cleanup resources
   */
  abstract close(): Promise<void>;
}

/**
 * In-memory bus implementation (works in browser and Node.js)
 */
export class MemoryBus extends Bus {
  private events: BusEvent[] = [];
  private flushTimer?: ReturnType<typeof setInterval>;

  constructor(config: BusConfig) {
    super(config);

    // Start periodic flush if configured
    if (this.config.flushInterval > 0) {
      this.flushTimer = setInterval(() => {
        this.flush();
      }, this.config.flushInterval);
    }
  }

  async emit<T>(
    topic: string,
    data: T,
    options?: Partial<BusEvent>
  ): Promise<BusEvent<T>> {
    const event = this.createEvent(topic, data, options);

    // Store event
    this.events.push(event);
    this.eventBuffer.push(event);

    // Dispatch to subscribers
    await this.dispatch(event);

    // Auto-flush if buffer is full
    if (this.eventBuffer.length >= this.config.bufferSize) {
      await this.flush();
    }

    return event;
  }

  async flush(): Promise<void> {
    this.eventBuffer = [];
  }

  async close(): Promise<void> {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    await this.flush();
  }

  /**
   * Get all events (for testing/debugging)
   */
  getEvents(): BusEvent[] {
    return [...this.events];
  }

  /**
   * Query events by topic pattern
   */
  query(pattern: string, limit?: number): BusEvent[] {
    const matcher = createTopicMatcher(pattern);
    const matching = this.events.filter((e) => matcher(e.topic));
    return limit ? matching.slice(-limit) : matching;
  }
}

// Export default memory bus factory
export function createBus(config: BusConfig): Bus {
  return new MemoryBus(config);
}

// Re-export types
export type { Holon, Ring };

// Version
export const VERSION = '0.1.0';
