/**
 * @ark/core/events - Event System Types
 *
 * Defines the event types and bus interfaces for the ARK system.
 * These are the foundational event abstractions used by @ark/bus.
 *
 * @module
 */

import type { Id, Timestamp, Metadata, Priority, JsonValue } from './types.js';

/**
 * Event types in the system
 */
export type EventType =
  | 'task.created'
  | 'task.started'
  | 'task.completed'
  | 'task.failed'
  | 'task.blocked'
  | 'task.unblocked'
  | 'agent.spawned'
  | 'agent.ready'
  | 'agent.busy'
  | 'agent.idle'
  | 'agent.error'
  | 'agent.terminated'
  | 'message.sent'
  | 'message.received'
  | 'message.ack'
  | 'resource.created'
  | 'resource.acquired'
  | 'resource.released'
  | 'resource.destroyed'
  | 'checkpoint.created'
  | 'checkpoint.restored'
  | 'pattern.detected'
  | 'pattern.matched'
  | 'thrash.detected'
  | 'thrash.resolved'
  | 'health.check'
  | 'health.degraded'
  | 'health.recovered'
  | 'system.startup'
  | 'system.shutdown'
  | 'system.error'
  | string; // Allow custom event types

/**
 * Event header - common fields for all events
 */
export interface EventHeader {
  /** Unique event ID */
  id: Id;
  /** Event type */
  type: EventType;
  /** Topic/channel */
  topic: string;
  /** Event timestamp */
  timestamp: Timestamp;
  /** Source agent/component ID */
  source: Id;
  /** Correlation ID for tracing */
  correlationId?: Id;
  /** Causation ID (event that caused this one) */
  causationId?: Id;
  /** Event priority */
  priority: Priority;
  /** Event version (for schema evolution) */
  version: number;
}

/**
 * Base event interface
 */
export interface Event<T = unknown> {
  /** Event header */
  header: EventHeader;
  /** Event payload */
  payload: T;
  /** Additional metadata */
  metadata?: Metadata;
}

/**
 * Task event payloads
 */
export interface TaskCreatedPayload {
  taskId: Id;
  name: string;
  description?: string;
  assignedTo?: Id;
  parentTask?: Id;
  dependencies: Id[];
  priority: Priority;
}

export interface TaskCompletedPayload {
  taskId: Id;
  result?: JsonValue;
  duration: number;
}

export interface TaskFailedPayload {
  taskId: Id;
  error: string;
  stack?: string;
  retryable: boolean;
}

/**
 * Agent event payloads
 */
export interface AgentSpawnedPayload {
  agentId: Id;
  name: string;
  role: string;
  capabilities: string[];
  parentAgent?: Id;
}

export interface AgentStateChangePayload {
  agentId: Id;
  previousState: string;
  newState: string;
  reason?: string;
}

/**
 * Message event payloads
 */
export interface MessagePayload {
  messageId: Id;
  from: Id;
  to: Id;
  content: JsonValue;
  replyTo?: Id;
}

/**
 * Pattern event payloads
 */
export interface PatternDetectedPayload {
  patternId: Id;
  patternType: string;
  confidence: number;
  context: Metadata;
  recommendations?: string[];
}

/**
 * Health event payloads
 */
export interface HealthCheckPayload {
  component: string;
  healthy: boolean;
  message: string;
  metrics?: Metadata;
}

/**
 * Event subscription filter
 */
export interface EventFilter {
  /** Filter by event types */
  types?: EventType[];
  /** Filter by topics */
  topics?: string[];
  /** Filter by source */
  sources?: Id[];
  /** Filter by correlation ID */
  correlationId?: Id;
  /** Filter by priority */
  minPriority?: Priority;
  /** Custom filter predicate */
  predicate?: (event: Event) => boolean;
}

/**
 * Event handler function type
 */
export type EventHandler<T = unknown> = (event: Event<T>) => void | Promise<void>;

/**
 * Subscription handle for unsubscribing
 */
export interface Subscription {
  /** Unique subscription ID */
  id: Id;
  /** Topic subscribed to */
  topic: string;
  /** Filter applied */
  filter?: EventFilter;
  /** Unsubscribe function */
  unsubscribe: () => void;
}

/**
 * Event bus interface
 */
export interface EventBus {
  /** Publish an event */
  publish<T>(event: Event<T>): Promise<void>;
  /** Subscribe to events */
  subscribe<T>(
    topic: string,
    handler: EventHandler<T>,
    filter?: EventFilter
  ): Subscription;
  /** Unsubscribe from events */
  unsubscribe(subscriptionId: Id): void;
  /** Get event history */
  getHistory(filter?: EventFilter, limit?: number): Promise<Event[]>;
}

/**
 * Create an event header
 */
export function createEventHeader(
  type: EventType,
  topic: string,
  source: Id,
  options: Partial<Omit<EventHeader, 'id' | 'type' | 'topic' | 'timestamp' | 'source'>> = {}
): EventHeader {
  return {
    id: crypto.randomUUID(),
    type,
    topic,
    timestamp: Date.now(),
    source,
    priority: options.priority ?? 'normal',
    version: options.version ?? 1,
    ...options,
  };
}

/**
 * Create an event
 */
export function createEvent<T>(
  type: EventType,
  topic: string,
  source: Id,
  payload: T,
  options: Partial<Omit<EventHeader, 'id' | 'type' | 'topic' | 'timestamp' | 'source'>> = {}
): Event<T> {
  return {
    header: createEventHeader(type, topic, source, options),
    payload,
  };
}

/**
 * Check if an event matches a filter
 */
export function matchesFilter(event: Event, filter: EventFilter): boolean {
  if (filter.types && !filter.types.includes(event.header.type)) {
    return false;
  }

  if (filter.topics && !filter.topics.includes(event.header.topic)) {
    return false;
  }

  if (filter.sources && !filter.sources.includes(event.header.source)) {
    return false;
  }

  if (filter.correlationId && event.header.correlationId !== filter.correlationId) {
    return false;
  }

  if (filter.minPriority) {
    const priorityOrder: Priority[] = ['background', 'low', 'normal', 'high', 'critical'];
    const eventIndex = priorityOrder.indexOf(event.header.priority);
    const minIndex = priorityOrder.indexOf(filter.minPriority);
    if (eventIndex < minIndex) {
      return false;
    }
  }

  if (filter.predicate && !filter.predicate(event)) {
    return false;
  }

  return true;
}

/**
 * Serialize an event to JSON
 */
export function serializeEvent(event: Event): string {
  return JSON.stringify(event);
}

/**
 * Deserialize an event from JSON
 */
export function deserializeEvent(json: string): Event {
  return JSON.parse(json) as Event;
}

/**
 * Event batch for bulk operations
 */
export interface EventBatch {
  events: Event[];
  timestamp: Timestamp;
  batchId: Id;
}

/**
 * Create an event batch
 */
export function createEventBatch(events: Event[]): EventBatch {
  return {
    events,
    timestamp: Date.now(),
    batchId: crypto.randomUUID(),
  };
}
