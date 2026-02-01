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
export type EventKind = 'event' | 'request' | 'response' | 'heartbeat' | 'handshake' | 'dissociation';
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
export declare const TOPIC_PARTITIONS: {
    readonly kernel: readonly ["axiom.*", "constitution.*", "dna.*"];
    readonly infrastructure: readonly ["a2a.*", "mcp.*", "bus.*", "daemon.*", "antigravity.*"];
    readonly agent: readonly ["agent.*", "task.*", "tool.*", "iterate.*"];
    readonly ephemeral: readonly ["paip.*", "clone.*", "sandbox.*"];
};
/**
 * Create a topic matcher for glob patterns
 */
export declare function createTopicMatcher(pattern: string): TopicFilter;
/**
 * Abstract Bus class - base for isomorphic implementations
 */
export declare abstract class Bus {
    protected config: Required<BusConfig>;
    protected subscriptions: Map<string, Set<Subscription>>;
    protected eventBuffer: BusEvent[];
    constructor(config: BusConfig);
    /**
     * Emit an event to the bus
     */
    abstract emit<T>(topic: string, data: T, options?: Partial<BusEvent>): Promise<BusEvent<T>>;
    /**
     * Subscribe to a topic pattern
     */
    subscribe<T = unknown>(pattern: string, handler: TopicHandler<T>): Subscription;
    /**
     * Dispatch event to matching subscribers
     */
    protected dispatch(event: BusEvent): Promise<void>;
    /**
     * Create a new event with all required fields
     */
    protected createEvent<T>(topic: string, data: T, options?: Partial<BusEvent>): BusEvent<T>;
    /**
     * Get the topic partition for a given topic
     */
    getPartition(topic: string): string | null;
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
export declare class MemoryBus extends Bus {
    private events;
    private flushTimer?;
    constructor(config: BusConfig);
    emit<T>(topic: string, data: T, options?: Partial<BusEvent>): Promise<BusEvent<T>>;
    flush(): Promise<void>;
    close(): Promise<void>;
    /**
     * Get all events (for testing/debugging)
     */
    getEvents(): BusEvent[];
    /**
     * Query events by topic pattern
     */
    query(pattern: string, limit?: number): BusEvent[];
}
export declare function createBus(config: BusConfig): Bus;
export type { Holon, Ring };
//# sourceMappingURL=index.d.ts.map