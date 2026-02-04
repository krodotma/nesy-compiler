/**
 * Bus Integration - Pluribus Event Bus Connection
 *
 * Phase 4 Step 32: Event Bus Integration
 *
 * Connects NeSy compilation to the Pluribus event bus:
 * - Emit compilation events (started, completed, failed)
 * - Listen for compilation requests
 * - Integrate with Holon lifecycle
 * - Support distributed compilation
 */

import type { CompiledHolon, CompilationRequest, TrustLevel } from '@nesy/core';
import type { CompilationResult } from '@nesy/pipeline';
import type { OrchestrationResult, TelemetryData } from './orchestration-engine.js';

/** Event types for NeSy compilation */
export type NeSyEventType =
  | 'nesy.compile.requested'
  | 'nesy.compile.started'
  | 'nesy.compile.phase.started'
  | 'nesy.compile.phase.completed'
  | 'nesy.compile.completed'
  | 'nesy.compile.failed'
  | 'nesy.holon.actualized'
  | 'nesy.holon.verified'
  | 'nesy.telemetry.report';

/** Base event structure */
export interface BusEvent<T extends NeSyEventType = NeSyEventType, P = unknown> {
  type: T;
  timestamp: number;
  correlationId: string;
  source: string;
  payload: P;
  metadata?: EventMetadata;
}

/** Event metadata */
export interface EventMetadata {
  version: string;
  trustLevel: TrustLevel;
  traceId?: string;
  spanId?: string;
  parentSpanId?: string;
}

/** Compile requested event */
export interface CompileRequestedEvent extends BusEvent<'nesy.compile.requested'> {
  payload: {
    request: CompilationRequest;
    priority: 'low' | 'normal' | 'high' | 'critical';
    deadline?: number;
  };
}

/** Compile started event */
export interface CompileStartedEvent extends BusEvent<'nesy.compile.started'> {
  payload: {
    requestId: string;
    strategy: string;
    phases: string[];
  };
}

/** Phase completed event */
export interface PhaseCompletedEvent extends BusEvent<'nesy.compile.phase.completed'> {
  payload: {
    requestId: string;
    phase: string;
    duration: number;
    success: boolean;
    metrics: Record<string, number>;
  };
}

/** Compile completed event */
export interface CompileCompletedEvent extends BusEvent<'nesy.compile.completed'> {
  payload: {
    requestId: string;
    result: CompilationResult;
    telemetry: TelemetryData;
  };
}

/** Compile failed event */
export interface CompileFailedEvent extends BusEvent<'nesy.compile.failed'> {
  payload: {
    requestId: string;
    error: string;
    phase?: string;
    retryable: boolean;
  };
}

/** Holon actualized event */
export interface HolonActualizedEvent extends BusEvent<'nesy.holon.actualized'> {
  payload: {
    holonId: string;
    holon: CompiledHolon;
    compilationId: string;
  };
}

/** Bus connection interface */
export interface BusConnection {
  /** Publish event to bus */
  publish(event: BusEvent): Promise<void>;
  /** Subscribe to event type */
  subscribe(type: NeSyEventType, handler: EventHandler): Subscription;
  /** Request-reply pattern */
  request(event: BusEvent, timeout?: number): Promise<BusEvent>;
  /** Check connection status */
  isConnected(): boolean;
  /** Close connection */
  close(): Promise<void>;
}

/** Event handler */
export type EventHandler = (event: BusEvent) => void | Promise<void>;

/** Subscription handle */
export interface Subscription {
  unsubscribe(): void;
  readonly active: boolean;
}

/** Bus configuration */
export interface BusConfig {
  /** Bus endpoint URL */
  endpoint: string;
  /** Client identifier */
  clientId: string;
  /** Authentication token */
  token?: string;
  /** Reconnect on disconnect */
  autoReconnect: boolean;
  /** Maximum reconnect attempts */
  maxReconnectAttempts: number;
  /** Event queue size */
  queueSize: number;
  /** Default event timeout */
  timeout: number;
}

const DEFAULT_CONFIG: BusConfig = {
  endpoint: 'ws://localhost:9876/bus',
  clientId: 'nesy-compiler',
  autoReconnect: true,
  maxReconnectAttempts: 5,
  queueSize: 1000,
  timeout: 30000,
};

/**
 * BusIntegration: Connect NeSy compiler to Pluribus bus.
 */
export class BusIntegration {
  private config: BusConfig;
  private connection: BusConnection | null = null;
  private subscriptions: Map<string, Subscription> = new Map();
  private pendingEvents: BusEvent[] = [];
  private correlationCounter = 0;

  constructor(config?: Partial<BusConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Connect to the bus.
   */
  async connect(): Promise<void> {
    // In a real implementation, this would establish WebSocket connection
    // For now, we create an in-memory mock connection
    this.connection = this.createMockConnection();

    // Flush pending events
    for (const event of this.pendingEvents) {
      await this.connection.publish(event);
    }
    this.pendingEvents = [];
  }

  /**
   * Disconnect from the bus.
   */
  async disconnect(): Promise<void> {
    // Unsubscribe all
    for (const [, sub] of this.subscriptions) {
      sub.unsubscribe();
    }
    this.subscriptions.clear();

    if (this.connection) {
      await this.connection.close();
      this.connection = null;
    }
  }

  /**
   * Emit compilation requested event.
   */
  async emitCompileRequested(
    request: CompilationRequest,
    priority: 'low' | 'normal' | 'high' | 'critical' = 'normal'
  ): Promise<string> {
    const correlationId = this.nextCorrelationId();

    const event: CompileRequestedEvent = {
      type: 'nesy.compile.requested',
      timestamp: Date.now(),
      correlationId,
      source: this.config.clientId,
      payload: { request, priority },
    };

    await this.publish(event);
    return correlationId;
  }

  /**
   * Emit compilation started event.
   */
  async emitCompileStarted(
    requestId: string,
    strategy: string,
    phases: string[]
  ): Promise<void> {
    const event: CompileStartedEvent = {
      type: 'nesy.compile.started',
      timestamp: Date.now(),
      correlationId: requestId,
      source: this.config.clientId,
      payload: { requestId, strategy, phases },
    };

    await this.publish(event);
  }

  /**
   * Emit phase completed event.
   */
  async emitPhaseCompleted(
    requestId: string,
    phase: string,
    duration: number,
    success: boolean,
    metrics: Record<string, number> = {}
  ): Promise<void> {
    const event: PhaseCompletedEvent = {
      type: 'nesy.compile.phase.completed',
      timestamp: Date.now(),
      correlationId: requestId,
      source: this.config.clientId,
      payload: { requestId, phase, duration, success, metrics },
    };

    await this.publish(event);
  }

  /**
   * Emit compilation completed event.
   */
  async emitCompileCompleted(
    requestId: string,
    result: CompilationResult,
    telemetry: TelemetryData
  ): Promise<void> {
    const event: CompileCompletedEvent = {
      type: 'nesy.compile.completed',
      timestamp: Date.now(),
      correlationId: requestId,
      source: this.config.clientId,
      payload: { requestId, result, telemetry },
    };

    await this.publish(event);
  }

  /**
   * Emit compilation failed event.
   */
  async emitCompileFailed(
    requestId: string,
    error: string,
    phase?: string,
    retryable: boolean = true
  ): Promise<void> {
    const event: CompileFailedEvent = {
      type: 'nesy.compile.failed',
      timestamp: Date.now(),
      correlationId: requestId,
      source: this.config.clientId,
      payload: { requestId, error, phase, retryable },
    };

    await this.publish(event);
  }

  /**
   * Emit holon actualized event.
   */
  async emitHolonActualized(
    holonId: string,
    holon: CompiledHolon,
    compilationId: string
  ): Promise<void> {
    const event: HolonActualizedEvent = {
      type: 'nesy.holon.actualized',
      timestamp: Date.now(),
      correlationId: compilationId,
      source: this.config.clientId,
      payload: { holonId, holon, compilationId },
    };

    await this.publish(event);
  }

  /**
   * Subscribe to compilation requests.
   */
  onCompileRequested(handler: (event: CompileRequestedEvent) => void | Promise<void>): Subscription {
    return this.subscribe('nesy.compile.requested', handler as EventHandler);
  }

  /**
   * Subscribe to event type.
   */
  subscribe(type: NeSyEventType, handler: EventHandler): Subscription {
    if (!this.connection) {
      throw new Error('Not connected to bus');
    }

    const subscription = this.connection.subscribe(type, handler);
    this.subscriptions.set(`${type}-${Date.now()}`, subscription);
    return subscription;
  }

  /**
   * Publish event to bus.
   */
  private async publish(event: BusEvent): Promise<void> {
    if (!this.connection) {
      // Queue event for later
      if (this.pendingEvents.length < this.config.queueSize) {
        this.pendingEvents.push(event);
      }
      return;
    }

    await this.connection.publish(event);
  }

  /**
   * Generate next correlation ID.
   */
  private nextCorrelationId(): string {
    return `${this.config.clientId}-${Date.now()}-${++this.correlationCounter}`;
  }

  /**
   * Create mock connection for testing.
   */
  private createMockConnection(): BusConnection {
    const handlers: Map<NeSyEventType, Set<EventHandler>> = new Map();
    let connected = true;

    return {
      async publish(event: BusEvent): Promise<void> {
        // Notify local handlers
        const eventHandlers = handlers.get(event.type);
        if (eventHandlers) {
          for (const handler of eventHandlers) {
            try {
              await handler(event);
            } catch (e) {
              console.error('Event handler error:', e);
            }
          }
        }
      },

      subscribe(type: NeSyEventType, handler: EventHandler): Subscription {
        if (!handlers.has(type)) {
          handlers.set(type, new Set());
        }
        handlers.get(type)!.add(handler);

        let active = true;
        return {
          unsubscribe() {
            handlers.get(type)?.delete(handler);
            active = false;
          },
          get active() { return active; },
        };
      },

      async request(event: BusEvent, timeout?: number): Promise<BusEvent> {
        // For mock, just return acknowledgment
        return {
          type: event.type,
          timestamp: Date.now(),
          correlationId: event.correlationId,
          source: 'mock-bus',
          payload: { acknowledged: true },
        };
      },

      isConnected(): boolean {
        return connected;
      },

      async close(): Promise<void> {
        connected = false;
        handlers.clear();
      },
    };
  }

  /**
   * Check if connected to bus.
   */
  isConnected(): boolean {
    return this.connection?.isConnected() ?? false;
  }
}

/**
 * Create and connect bus integration.
 */
export async function createBusIntegration(
  config?: Partial<BusConfig>
): Promise<BusIntegration> {
  const integration = new BusIntegration(config);
  await integration.connect();
  return integration;
}
