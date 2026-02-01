/**
 * @ark/core/types - Core Types Tests
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  isEntity,
  isTask,
  createEntity,
  touchEntity,
  type Entity,
  type Task,
  type Agent,
  type Priority,
  type TaskStatus,
  type AgentState,
} from './types.js';

describe('Types', () => {
  describe('isEntity()', () => {
    it('should return true for valid entities', () => {
      const entity = {
        id: 'test-id',
        createdAt: Date.now(),
        updatedAt: Date.now(),
      };
      expect(isEntity(entity)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isEntity(null)).toBe(false);
    });

    it('should return false for undefined', () => {
      expect(isEntity(undefined)).toBe(false);
    });

    it('should return false for missing id', () => {
      expect(isEntity({ createdAt: 1, updatedAt: 1 })).toBe(false);
    });

    it('should return false for missing createdAt', () => {
      expect(isEntity({ id: 'x', updatedAt: 1 })).toBe(false);
    });

    it('should return false for missing updatedAt', () => {
      expect(isEntity({ id: 'x', createdAt: 1 })).toBe(false);
    });

    it('should return false for primitive values', () => {
      expect(isEntity('string')).toBe(false);
      expect(isEntity(123)).toBe(false);
      expect(isEntity(true)).toBe(false);
    });
  });

  describe('isTask()', () => {
    it('should return true for valid tasks', () => {
      const task = {
        id: 'task-1',
        name: 'Test Task',
        createdAt: Date.now(),
        updatedAt: Date.now(),
        status: 'pending' as TaskStatus,
        priority: 'normal' as Priority,
        tags: [],
        subtasks: [],
        dependencies: [],
        data: {},
      };
      expect(isTask(task)).toBe(true);
    });

    it('should return false for entities without task fields', () => {
      const entity = {
        id: 'x',
        createdAt: 1,
        updatedAt: 1,
      };
      expect(isTask(entity)).toBe(false);
    });

    it('should return false for non-entities', () => {
      expect(isTask(null)).toBe(false);
      expect(isTask({})).toBe(false);
    });
  });

  describe('createEntity()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
    });

    it('should create entity with generated UUID', () => {
      const entity = createEntity();
      expect(entity.id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    });

    it('should create entity with provided ID', () => {
      const entity = createEntity('my-custom-id');
      expect(entity.id).toBe('my-custom-id');
    });

    it('should set createdAt to current time', () => {
      const entity = createEntity();
      expect(entity.createdAt).toBe(Date.now());
    });

    it('should set updatedAt to current time', () => {
      const entity = createEntity();
      expect(entity.updatedAt).toBe(Date.now());
    });

    it('should set same createdAt and updatedAt initially', () => {
      const entity = createEntity();
      expect(entity.createdAt).toBe(entity.updatedAt);
    });

    vi.useRealTimers();
  });

  describe('touchEntity()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    it('should update updatedAt timestamp', () => {
      vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
      const entity = createEntity('test');
      const originalUpdatedAt = entity.updatedAt;

      vi.setSystemTime(new Date('2025-01-28T13:00:00Z'));
      const touched = touchEntity(entity);

      expect(touched.updatedAt).toBeGreaterThan(originalUpdatedAt);
    });

    it('should preserve id', () => {
      const entity = createEntity('my-id');
      const touched = touchEntity(entity);
      expect(touched.id).toBe('my-id');
    });

    it('should preserve createdAt', () => {
      const entity = createEntity('test');
      const originalCreatedAt = entity.createdAt;

      vi.advanceTimersByTime(1000);
      const touched = touchEntity(entity);

      expect(touched.createdAt).toBe(originalCreatedAt);
    });

    it('should preserve additional properties', () => {
      const entity = { ...createEntity('test'), extra: 'data', count: 42 };
      const touched = touchEntity(entity);
      expect(touched.extra).toBe('data');
      expect(touched.count).toBe(42);
    });

    vi.useRealTimers();
  });

  describe('Type definitions', () => {
    it('should allow valid TaskStatus values', () => {
      const statuses: TaskStatus[] = ['pending', 'running', 'completed', 'failed', 'cancelled', 'blocked'];
      expect(statuses).toHaveLength(6);
    });

    it('should allow valid AgentState values', () => {
      const states: AgentState[] = ['idle', 'thinking', 'acting', 'waiting', 'error', 'terminated'];
      expect(states).toHaveLength(6);
    });

    it('should allow valid Priority values', () => {
      const priorities: Priority[] = ['critical', 'high', 'normal', 'low', 'background'];
      expect(priorities).toHaveLength(5);
    });
  });
});
