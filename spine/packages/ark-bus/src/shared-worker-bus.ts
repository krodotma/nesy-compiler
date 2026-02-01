/**
 * @ark/bus/shared-worker-bus - SharedWorker Event Bus
 *
 * Browser-native event bus using SharedWorker for cross-tab communication.
 * Provides true multi-tab event coordination without external dependencies.
 *
 * @module
 */

import { Bus, type BusConfig, type BusEvent, createTopicMatcher } from './index.js';

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
 * Message types for worker communication
 */
type WorkerMessageType =
  | 'connect'
  | 'disconnect'
  | 'emit'
  | 'subscribe'
  | 'unsubscribe'
  | 'event'
  | 'sync';

interface WorkerMessage {
  type: WorkerMessageType;
  payload: unknown;
}

/**
 * SharedWorker-based bus for browser multi-tab coordination
 */
export class SharedWorkerBus extends Bus {
  private worker: SharedWorker | null = null;
  private port: MessagePort | null = null;
  private workerConfig: Required<SharedWorkerBusConfig>;
  private connected: boolean = false;
  private pendingMessages: WorkerMessage[] = [];
  private localEvents: BusEvent[] = [];

  constructor(config: SharedWorkerBusConfig) {
    super(config);
    this.workerConfig = {
      actor: config.actor,
      ring: config.ring,
      protocol: config.protocol ?? 'DKIN v29',
      debug: config.debug ?? false,
      busDir: config.busDir ?? '',
      bufferSize: config.bufferSize ?? 100,
      flushInterval: config.flushInterval ?? 1000,
      workerUrl: config.workerUrl ?? '/bus-worker.js',
      workerName: config.workerName ?? 'ark-bus',
      persist: config.persist ?? false,
      dbName: config.dbName ?? 'ark-bus-events',
    };

    this.initialize();
  }

  /**
   * Initialize the SharedWorker connection
   */
  private initialize(): void {
    // Check for SharedWorker support
    if (typeof SharedWorker === 'undefined') {
      console.warn('@ark/bus: SharedWorker not supported, falling back to local-only mode');
      return;
    }

    try {
      this.worker = new SharedWorker(
        this.workerConfig.workerUrl,
        { name: this.workerConfig.workerName }
      );

      this.port = this.worker.port;
      this.port.start();

      // Handle messages from worker
      this.port.onmessage = (event: MessageEvent<WorkerMessage>) => {
        this.handleWorkerMessage(event.data);
      };

      // Handle errors
      this.worker.onerror = (error) => {
        console.error('@ark/bus: SharedWorker error:', error);
      };

      // Send connect message
      this.sendToWorker({
        type: 'connect',
        payload: {
          actor: this.config.actor,
          ring: this.config.ring,
        },
      });
    } catch (error) {
      console.error('@ark/bus: Failed to initialize SharedWorker:', error);
    }
  }

  /**
   * Send message to worker
   */
  private sendToWorker(message: WorkerMessage): void {
    if (!this.port || !this.connected) {
      this.pendingMessages.push(message);
      return;
    }

    this.port.postMessage(message);
  }

  /**
   * Handle messages from worker
   */
  private handleWorkerMessage(message: WorkerMessage): void {
    switch (message.type) {
      case 'connect':
        this.connected = true;
        // Send pending messages
        for (const msg of this.pendingMessages) {
          this.port?.postMessage(msg);
        }
        this.pendingMessages = [];
        break;

      case 'event':
        // Event from another tab
        const event = message.payload as BusEvent;
        this.localEvents.push(event);
        this.dispatch(event);
        break;

      case 'sync':
        // Sync events from worker
        const events = message.payload as BusEvent[];
        for (const e of events) {
          if (!this.localEvents.find((le) => le.id === e.id)) {
            this.localEvents.push(e);
          }
        }
        break;

      default:
        if (this.config.debug) {
          console.log('@ark/bus: Unknown worker message:', message.type);
        }
    }
  }

  /**
   * Emit an event to the bus
   */
  async emit<T>(
    topic: string,
    data: T,
    options?: Partial<BusEvent>
  ): Promise<BusEvent<T>> {
    const event = this.createEvent(topic, data, options);

    // Store locally
    this.localEvents.push(event);
    this.eventBuffer.push(event);

    // Dispatch to local subscribers
    await this.dispatch(event);

    // Send to worker for cross-tab broadcast
    this.sendToWorker({
      type: 'emit',
      payload: event,
    });

    // Auto-flush if buffer is full
    if (this.eventBuffer.length >= this.config.bufferSize) {
      await this.flush();
    }

    return event;
  }

  /**
   * Flush buffered events
   */
  async flush(): Promise<void> {
    this.eventBuffer = [];
  }

  /**
   * Close the bus and cleanup resources
   */
  async close(): Promise<void> {
    if (this.port) {
      this.sendToWorker({
        type: 'disconnect',
        payload: { actor: this.config.actor },
      });
      this.port.close();
    }

    this.worker = null;
    this.port = null;
    this.connected = false;
  }

  /**
   * Get local events (for debugging)
   */
  getEvents(): BusEvent[] {
    return [...this.localEvents];
  }

  /**
   * Query local events by topic pattern
   */
  query(pattern: string, limit?: number): BusEvent[] {
    const matcher = createTopicMatcher(pattern);
    const matching = this.localEvents.filter((e) => matcher(e.topic));
    return limit ? matching.slice(-limit) : matching;
  }

  /**
   * Request sync from worker
   */
  requestSync(since?: number): void {
    this.sendToWorker({
      type: 'sync',
      payload: { since: since ?? 0 },
    });
  }
}

/**
 * SharedWorker script code (to be served as bus-worker.js)
 *
 * This is the code that runs inside the SharedWorker.
 * In production, this would be bundled separately.
 */
export const SHARED_WORKER_SCRIPT = `
// @ark/bus SharedWorker

const connections = new Map();
const events = [];
const MAX_EVENTS = 10000;

self.onconnect = (e) => {
  const port = e.ports[0];
  let actorId = null;

  port.onmessage = (event) => {
    const message = event.data;

    switch (message.type) {
      case 'connect':
        actorId = message.payload.actor;
        connections.set(actorId, port);
        port.postMessage({ type: 'connect', payload: { success: true } });
        break;

      case 'disconnect':
        if (actorId) {
          connections.delete(actorId);
        }
        break;

      case 'emit':
        const busEvent = message.payload;
        events.push(busEvent);

        // Trim old events
        while (events.length > MAX_EVENTS) {
          events.shift();
        }

        // Broadcast to all other connections
        for (const [id, p] of connections) {
          if (id !== actorId) {
            p.postMessage({ type: 'event', payload: busEvent });
          }
        }
        break;

      case 'sync':
        const since = message.payload.since || 0;
        const syncEvents = events.filter(e => e.timestamp >= since);
        port.postMessage({ type: 'sync', payload: syncEvents });
        break;
    }
  };

  port.start();
};
`;

/**
 * Create a SharedWorker bus
 */
export function createSharedWorkerBus(config: SharedWorkerBusConfig): SharedWorkerBus {
  return new SharedWorkerBus(config);
}

/**
 * Generate a data URL for the SharedWorker script
 */
export function getWorkerDataUrl(): string {
  const blob = new Blob([SHARED_WORKER_SCRIPT], { type: 'application/javascript' });
  return URL.createObjectURL(blob);
}
