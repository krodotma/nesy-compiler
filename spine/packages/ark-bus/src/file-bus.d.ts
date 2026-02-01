/**
 * @ark/bus/file-bus - File-based Event Bus
 *
 * NDJSON append-only event storage for Node.js.
 * Provides durable event persistence with append-only semantics.
 *
 * @module
 */
import { Bus, type BusConfig, type BusEvent } from './index.js';
/**
 * File bus configuration
 */
export interface FileBusConfig extends BusConfig {
    /** Directory for event files */
    busDir: string;
    /** Main event file name */
    eventFile?: string;
    /** Enable topic-based partitioning */
    partitionByTopic?: boolean;
    /** Maximum file size before rotation (bytes) */
    maxFileSize?: number;
    /** Compress rotated files */
    compressRotated?: boolean;
}
/**
 * File-based bus implementation for Node.js
 * Persists events as NDJSON to an append-only log
 */
export declare class FileBus extends Bus {
    private fileConfig;
    private writeStream;
    private flushTimer?;
    private currentFileSize;
    constructor(config: FileBusConfig);
    /**
     * Initialize the file bus
     */
    private initialize;
    /**
     * Get the event file path
     */
    private getEventFilePath;
    /**
     * Rotate the event file if needed
     */
    private maybeRotate;
    /**
     * Close the write stream
     */
    private closeStream;
    /**
     * Emit an event to the bus
     */
    emit<T>(topic: string, data: T, options?: Partial<BusEvent>): Promise<BusEvent<T>>;
    /**
     * Flush buffered events to disk
     */
    flush(): Promise<void>;
    /**
     * Close the bus and cleanup resources
     */
    close(): Promise<void>;
    /**
     * Read events from the log file
     */
    readEvents(options?: {
        topic?: string;
        since?: number;
        limit?: number;
        filter?: (event: BusEvent) => boolean;
    }): Promise<BusEvent[]>;
    /**
     * Tail the event log (returns async iterator)
     */
    tail(topic?: string): AsyncGenerator<BusEvent, void, unknown>;
    /**
     * Replay events to a handler
     */
    replay(handler: (event: BusEvent) => void | Promise<void>, options?: {
        topic?: string;
        since?: number;
        until?: number;
    }): Promise<number>;
}
/**
 * Create a file-based bus
 */
export declare function createFileBus(config: FileBusConfig): FileBus;
//# sourceMappingURL=file-bus.d.ts.map