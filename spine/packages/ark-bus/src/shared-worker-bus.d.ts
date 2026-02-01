/**
 * @ark/bus/shared-worker-bus - SharedWorker Event Bus
 *
 * Browser-native event bus using SharedWorker for cross-tab communication.
 * Provides true multi-tab event coordination without external dependencies.
 *
 * @module
 */
import { Bus, type BusConfig, type BusEvent } from './index.js';
/**
 * SharedWorker bus configuration
 */
export interface SharedWorkerBusConfig extends BusConfig {
    /** Worker script URL */
    workerUrl?: string;
    /** Worker name for identification */
    workerName?: string;
    /** Enable IndexedDB persistence */
    persist?: boolean;
    /** IndexedDB database name */
    dbName?: string;
}
/**
 * SharedWorker-based bus for browser multi-tab coordination
 */
export declare class SharedWorkerBus extends Bus {
    private worker;
    private port;
    private workerConfig;
    private connected;
    private pendingMessages;
    private localEvents;
    constructor(config: SharedWorkerBusConfig);
    /**
     * Initialize the SharedWorker connection
     */
    private initialize;
    /**
     * Send message to worker
     */
    private sendToWorker;
    /**
     * Handle messages from worker
     */
    private handleWorkerMessage;
    /**
     * Emit an event to the bus
     */
    emit<T>(topic: string, data: T, options?: Partial<BusEvent>): Promise<BusEvent<T>>;
    /**
     * Flush buffered events
     */
    flush(): Promise<void>;
    /**
     * Close the bus and cleanup resources
     */
    close(): Promise<void>;
    /**
     * Get local events (for debugging)
     */
    getEvents(): BusEvent[];
    /**
     * Query local events by topic pattern
     */
    query(pattern: string, limit?: number): BusEvent[];
    /**
     * Request sync from worker
     */
    requestSync(since?: number): void;
}
/**
 * SharedWorker script code (to be served as bus-worker.js)
 *
 * This is the code that runs inside the SharedWorker.
 * In production, this would be bundled separately.
 */
export declare const SHARED_WORKER_SCRIPT = "\n// @ark/bus SharedWorker\n\nconst connections = new Map();\nconst events = [];\nconst MAX_EVENTS = 10000;\n\nself.onconnect = (e) => {\n  const port = e.ports[0];\n  let actorId = null;\n\n  port.onmessage = (event) => {\n    const message = event.data;\n\n    switch (message.type) {\n      case 'connect':\n        actorId = message.payload.actor;\n        connections.set(actorId, port);\n        port.postMessage({ type: 'connect', payload: { success: true } });\n        break;\n\n      case 'disconnect':\n        if (actorId) {\n          connections.delete(actorId);\n        }\n        break;\n\n      case 'emit':\n        const busEvent = message.payload;\n        events.push(busEvent);\n\n        // Trim old events\n        while (events.length > MAX_EVENTS) {\n          events.shift();\n        }\n\n        // Broadcast to all other connections\n        for (const [id, p] of connections) {\n          if (id !== actorId) {\n            p.postMessage({ type: 'event', payload: busEvent });\n          }\n        }\n        break;\n\n      case 'sync':\n        const since = message.payload.since || 0;\n        const syncEvents = events.filter(e => e.timestamp >= since);\n        port.postMessage({ type: 'sync', payload: syncEvents });\n        break;\n    }\n  };\n\n  port.start();\n};\n";
/**
 * Create a SharedWorker bus
 */
export declare function createSharedWorkerBus(config: SharedWorkerBusConfig): SharedWorkerBus;
/**
 * Generate a data URL for the SharedWorker script
 */
export declare function getWorkerDataUrl(): string;
//# sourceMappingURL=shared-worker-bus.d.ts.map