/**
 * @ark/core/events - Event System Tests
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  createEventHeader,
  createEvent,
  matchesFilter,
  serializeEvent,
  deserializeEvent,
  createEventBatch,
  type Event,
  type EventFilter,
  type Priority,
} from './events.js';

describe('Events', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
  });

  describe('createEventHeader()', () => {
    it('should create event header with required fields', () => {
      const header = createEventHeader('task.created', 'tasks', 'agent-1');

      expect(header.type).toBe('task.created');
      expect(header.topic).toBe('tasks');
      expect(header.source).toBe('agent-1');
      expect(header.timestamp).toBe(Date.now());
      expect(header.priority).toBe('normal');
      expect(header.version).toBe(1);
      expect(header.id).toMatch(/^[0-9a-f-]{36}$/);
    });

    it('should accept optional fields', () => {
      const header = createEventHeader('task.created', 'tasks', 'agent-1', {
        priority: 'high',
        version: 2,
        correlationId: 'corr-123',
        causationId: 'cause-456',
      });

      expect(header.priority).toBe('high');
      expect(header.version).toBe(2);
      expect(header.correlationId).toBe('corr-123');
      expect(header.causationId).toBe('cause-456');
    });
  });

  describe('createEvent()', () => {
    it('should create complete event', () => {
      const event = createEvent(
        'task.completed',
        'tasks',
        'agent-1',
        { taskId: 'task-123', result: 'success' }
      );

      expect(event.header.type).toBe('task.completed');
      expect(event.header.topic).toBe('tasks');
      expect(event.payload).toEqual({ taskId: 'task-123', result: 'success' });
    });

    it('should preserve type information in payload', () => {
      interface MyPayload { value: number }
      const event = createEvent<MyPayload>(
        'custom.event',
        'custom',
        'source',
        { value: 42 }
      );

      expect(event.payload.value).toBe(42);
    });
  });

  describe('matchesFilter()', () => {
    const baseEvent: Event = {
      header: {
        id: 'event-1',
        type: 'task.created',
        topic: 'tasks',
        timestamp: Date.now(),
        source: 'agent-1',
        priority: 'normal',
        version: 1,
        correlationId: 'corr-123',
      },
      payload: {},
    };

    it('should match when filter is empty', () => {
      expect(matchesFilter(baseEvent, {})).toBe(true);
    });

    it('should filter by event types', () => {
      expect(matchesFilter(baseEvent, { types: ['task.created'] })).toBe(true);
      expect(matchesFilter(baseEvent, { types: ['task.completed'] })).toBe(false);
      expect(matchesFilter(baseEvent, { types: ['task.created', 'task.completed'] })).toBe(true);
    });

    it('should filter by topics', () => {
      expect(matchesFilter(baseEvent, { topics: ['tasks'] })).toBe(true);
      expect(matchesFilter(baseEvent, { topics: ['agents'] })).toBe(false);
    });

    it('should filter by sources', () => {
      expect(matchesFilter(baseEvent, { sources: ['agent-1'] })).toBe(true);
      expect(matchesFilter(baseEvent, { sources: ['agent-2'] })).toBe(false);
    });

    it('should filter by correlationId', () => {
      expect(matchesFilter(baseEvent, { correlationId: 'corr-123' })).toBe(true);
      expect(matchesFilter(baseEvent, { correlationId: 'other' })).toBe(false);
    });

    it('should filter by minPriority', () => {
      const priorities: Priority[] = ['background', 'low', 'normal', 'high', 'critical'];

      expect(matchesFilter(baseEvent, { minPriority: 'normal' })).toBe(true);
      expect(matchesFilter(baseEvent, { minPriority: 'low' })).toBe(true);
      expect(matchesFilter(baseEvent, { minPriority: 'high' })).toBe(false);
    });

    it('should filter by custom predicate', () => {
      expect(matchesFilter(baseEvent, {
        predicate: (e) => e.header.source === 'agent-1',
      })).toBe(true);

      expect(matchesFilter(baseEvent, {
        predicate: (e) => e.header.source === 'agent-2',
      })).toBe(false);
    });

    it('should combine multiple filters with AND logic', () => {
      expect(matchesFilter(baseEvent, {
        types: ['task.created'],
        topics: ['tasks'],
        sources: ['agent-1'],
      })).toBe(true);

      expect(matchesFilter(baseEvent, {
        types: ['task.created'],
        topics: ['tasks'],
        sources: ['agent-2'], // wrong source
      })).toBe(false);
    });
  });

  describe('serializeEvent()', () => {
    it('should serialize event to JSON string', () => {
      const event = createEvent('test', 'topic', 'source', { data: 123 });
      const json = serializeEvent(event);

      expect(typeof json).toBe('string');
      expect(json).toContain('"type":"test"');
      expect(json).toContain('"data":123');
    });
  });

  describe('deserializeEvent()', () => {
    it('should deserialize JSON string to event', () => {
      const original = createEvent('test', 'topic', 'source', { data: 123 });
      const json = serializeEvent(original);
      const deserialized = deserializeEvent(json);

      expect(deserialized.header.type).toBe('test');
      expect(deserialized.payload).toEqual({ data: 123 });
    });

    it('should roundtrip correctly', () => {
      const event = createEvent('task.created', 'tasks', 'agent-1', {
        taskId: 'task-123',
        name: 'Test Task',
      });

      const roundtripped = deserializeEvent(serializeEvent(event));
      expect(roundtripped).toEqual(event);
    });
  });

  describe('createEventBatch()', () => {
    it('should create batch with events', () => {
      const events = [
        createEvent('a', 'topic', 'src', {}),
        createEvent('b', 'topic', 'src', {}),
      ];

      const batch = createEventBatch(events);

      expect(batch.events).toHaveLength(2);
      expect(batch.timestamp).toBe(Date.now());
      expect(batch.batchId).toMatch(/^[0-9a-f-]{36}$/);
    });

    it('should handle empty batch', () => {
      const batch = createEventBatch([]);
      expect(batch.events).toHaveLength(0);
    });
  });

  vi.useRealTimers();
});
