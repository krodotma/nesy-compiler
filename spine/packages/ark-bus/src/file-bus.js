/**
 * @ark/bus/file-bus - File-based Event Bus
 *
 * NDJSON append-only event storage for Node.js.
 * Provides durable event persistence with append-only semantics.
 *
 * @module
 */
import { Bus, createTopicMatcher } from './index.js';
import * as fs from 'node:fs';
import * as path from 'node:path';
/**
 * File-based bus implementation for Node.js
 * Persists events as NDJSON to an append-only log
 */
export class FileBus extends Bus {
    fileConfig;
    writeStream = null;
    flushTimer;
    currentFileSize = 0;
    constructor(config) {
        super(config);
        this.fileConfig = {
            actor: config.actor,
            ring: config.ring,
            protocol: config.protocol ?? 'DKIN v29',
            debug: config.debug ?? false,
            busDir: config.busDir,
            eventFile: config.eventFile ?? 'events.ndjson',
            partitionByTopic: config.partitionByTopic ?? false,
            maxFileSize: config.maxFileSize ?? 100 * 1024 * 1024, // 100MB
            compressRotated: config.compressRotated ?? true,
            bufferSize: config.bufferSize ?? 100,
            flushInterval: config.flushInterval ?? 1000,
        };
        this.initialize();
    }
    /**
     * Initialize the file bus
     */
    initialize() {
        // Ensure bus directory exists
        if (!fs.existsSync(this.fileConfig.busDir)) {
            fs.mkdirSync(this.fileConfig.busDir, { recursive: true });
        }
        // Get current file size
        const filePath = this.getEventFilePath();
        if (fs.existsSync(filePath)) {
            const stats = fs.statSync(filePath);
            this.currentFileSize = stats.size;
        }
        // Open write stream
        this.writeStream = fs.createWriteStream(filePath, { flags: 'a' });
        // Start periodic flush
        if (this.fileConfig.flushInterval > 0) {
            this.flushTimer = setInterval(() => {
                this.flush();
            }, this.fileConfig.flushInterval);
        }
    }
    /**
     * Get the event file path
     */
    getEventFilePath(topic) {
        if (this.fileConfig.partitionByTopic && topic) {
            const partition = this.getPartition(topic) ?? 'default';
            return path.join(this.fileConfig.busDir, `${partition}_events.ndjson`);
        }
        return path.join(this.fileConfig.busDir, this.fileConfig.eventFile);
    }
    /**
     * Rotate the event file if needed
     */
    async maybeRotate() {
        if (this.currentFileSize < this.fileConfig.maxFileSize)
            return;
        // Close current stream
        await this.closeStream();
        // Rename current file with timestamp
        const filePath = this.getEventFilePath();
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const rotatedPath = filePath.replace('.ndjson', `_${timestamp}.ndjson`);
        fs.renameSync(filePath, rotatedPath);
        // Optionally compress
        if (this.fileConfig.compressRotated) {
            // Compression would be implemented here with zlib
            // For now, just leave as-is
        }
        // Reset size and open new stream
        this.currentFileSize = 0;
        this.writeStream = fs.createWriteStream(filePath, { flags: 'a' });
    }
    /**
     * Close the write stream
     */
    async closeStream() {
        if (!this.writeStream)
            return;
        return new Promise((resolve, reject) => {
            this.writeStream.end((err) => {
                if (err)
                    reject(err);
                else
                    resolve();
            });
        });
    }
    /**
     * Emit an event to the bus
     */
    async emit(topic, data, options) {
        const event = this.createEvent(topic, data, options);
        // Write to buffer
        this.eventBuffer.push(event);
        // Dispatch to in-memory subscribers
        await this.dispatch(event);
        // Auto-flush if buffer is full
        if (this.eventBuffer.length >= this.config.bufferSize) {
            await this.flush();
        }
        return event;
    }
    /**
     * Flush buffered events to disk
     */
    async flush() {
        if (this.eventBuffer.length === 0)
            return;
        if (!this.writeStream)
            return;
        // Check rotation
        await this.maybeRotate();
        // Write events as NDJSON
        const lines = this.eventBuffer.map((e) => JSON.stringify(e) + '\n');
        const data = lines.join('');
        return new Promise((resolve, reject) => {
            this.writeStream.write(data, (err) => {
                if (err) {
                    reject(err);
                }
                else {
                    this.currentFileSize += Buffer.byteLength(data);
                    this.eventBuffer = [];
                    resolve();
                }
            });
        });
    }
    /**
     * Close the bus and cleanup resources
     */
    async close() {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
        }
        await this.flush();
        await this.closeStream();
    }
    /**
     * Read events from the log file
     */
    async readEvents(options = {}) {
        const filePath = this.getEventFilePath(options.topic);
        if (!fs.existsSync(filePath))
            return [];
        const content = fs.readFileSync(filePath, 'utf-8');
        const lines = content.trim().split('\n').filter(Boolean);
        let events = lines.map((line) => JSON.parse(line));
        // Apply filters
        if (options.topic) {
            const matcher = createTopicMatcher(options.topic);
            events = events.filter((e) => matcher(e.topic));
        }
        if (options.since) {
            events = events.filter((e) => e.timestamp >= options.since);
        }
        if (options.filter) {
            events = events.filter(options.filter);
        }
        if (options.limit) {
            events = events.slice(-options.limit);
        }
        return events;
    }
    /**
     * Tail the event log (returns async iterator)
     */
    async *tail(topic) {
        const filePath = this.getEventFilePath(topic);
        let position = 0;
        // Get initial position
        if (fs.existsSync(filePath)) {
            const stats = fs.statSync(filePath);
            position = stats.size;
        }
        const matcher = topic ? createTopicMatcher(topic) : null;
        while (true) {
            // Check for new data
            if (fs.existsSync(filePath)) {
                const stats = fs.statSync(filePath);
                if (stats.size > position) {
                    // Read new data
                    const fd = fs.openSync(filePath, 'r');
                    const buffer = Buffer.alloc(stats.size - position);
                    fs.readSync(fd, buffer, 0, buffer.length, position);
                    fs.closeSync(fd);
                    position = stats.size;
                    // Parse and yield events
                    const content = buffer.toString('utf-8');
                    const lines = content.trim().split('\n').filter(Boolean);
                    for (const line of lines) {
                        try {
                            const event = JSON.parse(line);
                            if (!matcher || matcher(event.topic)) {
                                yield event;
                            }
                        }
                        catch {
                            // Skip malformed lines
                        }
                    }
                }
            }
            // Wait before checking again
            await new Promise((resolve) => setTimeout(resolve, 100));
        }
    }
    /**
     * Replay events to a handler
     */
    async replay(handler, options = {}) {
        const events = await this.readEvents({
            topic: options.topic,
            since: options.since,
            filter: options.until ? (e) => e.timestamp <= options.until : undefined,
        });
        for (const event of events) {
            await handler(event);
        }
        return events.length;
    }
}
/**
 * Create a file-based bus
 */
export function createFileBus(config) {
    return new FileBus(config);
}
//# sourceMappingURL=file-bus.js.map