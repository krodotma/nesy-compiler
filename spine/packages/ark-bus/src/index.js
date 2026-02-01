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
};
/**
 * Create a topic matcher for glob patterns
 */
export function createTopicMatcher(pattern) {
    // Convert glob to regex
    const regexPattern = pattern
        .replace(/\./g, '\\.')
        .replace(/\*/g, '.*')
        .replace(/\?/g, '.');
    const regex = new RegExp(`^${regexPattern}$`);
    return (topic) => regex.test(topic);
}
/**
 * Abstract Bus class - base for isomorphic implementations
 */
export class Bus {
    config;
    subscriptions = new Map();
    eventBuffer = [];
    constructor(config) {
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
     * Subscribe to a topic pattern
     */
    subscribe(pattern, handler) {
        const matcher = createTopicMatcher(pattern);
        const subscription = {
            pattern,
            handler: handler,
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
        this.subscriptions.get(pattern).add(subscription);
        return subscription;
    }
    /**
     * Dispatch event to matching subscribers
     */
    async dispatch(event) {
        for (const [pattern, subs] of this.subscriptions) {
            const matcher = createTopicMatcher(pattern);
            if (matcher(event.topic)) {
                for (const sub of subs) {
                    try {
                        await sub.handler(event);
                    }
                    catch (error) {
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
    createEvent(topic, data, options) {
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
    getPartition(topic) {
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
}
/**
 * In-memory bus implementation (works in browser and Node.js)
 */
export class MemoryBus extends Bus {
    events = [];
    flushTimer;
    constructor(config) {
        super(config);
        // Start periodic flush if configured
        if (this.config.flushInterval > 0) {
            this.flushTimer = setInterval(() => {
                this.flush();
            }, this.config.flushInterval);
        }
    }
    async emit(topic, data, options) {
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
    async flush() {
        this.eventBuffer = [];
    }
    async close() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
        }
        await this.flush();
    }
    /**
     * Get all events (for testing/debugging)
     */
    getEvents() {
        return [...this.events];
    }
    /**
     * Query events by topic pattern
     */
    query(pattern, limit) {
        const matcher = createTopicMatcher(pattern);
        const matching = this.events.filter((e) => matcher(e.topic));
        return limit ? matching.slice(-limit) : matching;
    }
}
// Export default memory bus factory
export function createBus(config) {
    return new MemoryBus(config);
}
//# sourceMappingURL=index.js.map