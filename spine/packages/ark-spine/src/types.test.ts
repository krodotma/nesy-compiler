/**
 * @ark/spine/types tests
 */

import { describe, it, expect } from 'vitest';
import {
  isModuleDefinition,
  isModuleInstance,
  createDefaultMetadata,
  createDefaultInstance,
  type ModuleDefinition,
  type ModuleInstance,
  type ModuleMetadata,
  type ModuleState,
  type ModuleHealth,
  type DependencyStatus,
} from './types.js';

describe('types', () => {
  describe('isModuleDefinition', () => {
    it('returns true for valid module definition', () => {
      const def: ModuleDefinition = {
        id: 'test-module',
        metadata: {
          name: '@ark/test',
          version: '1.0.0',
          ring: 2,
          capabilities: [],
          tags: [],
        },
        dependencies: [],
        exports: [],
      };

      expect(isModuleDefinition(def)).toBe(true);
    });

    it('returns false for null', () => {
      expect(isModuleDefinition(null)).toBe(false);
    });

    it('returns false for undefined', () => {
      expect(isModuleDefinition(undefined)).toBe(false);
    });

    it('returns false for primitive values', () => {
      expect(isModuleDefinition('string')).toBe(false);
      expect(isModuleDefinition(123)).toBe(false);
      expect(isModuleDefinition(true)).toBe(false);
    });

    it('returns false for object missing id', () => {
      expect(isModuleDefinition({ metadata: {}, dependencies: [] })).toBe(false);
    });

    it('returns false for object missing metadata', () => {
      expect(isModuleDefinition({ id: 'test', dependencies: [] })).toBe(false);
    });

    it('returns false for object missing dependencies', () => {
      expect(isModuleDefinition({ id: 'test', metadata: {} })).toBe(false);
    });
  });

  describe('isModuleInstance', () => {
    it('returns true for valid module instance', () => {
      const instance: ModuleInstance = {
        definitionId: 'test-module',
        instanceId: 'instance-1',
        state: 'unloaded',
        health: 'unknown',
        config: {},
      };

      expect(isModuleInstance(instance)).toBe(true);
    });

    it('returns false for null', () => {
      expect(isModuleInstance(null)).toBe(false);
    });

    it('returns false for undefined', () => {
      expect(isModuleInstance(undefined)).toBe(false);
    });

    it('returns false for primitive values', () => {
      expect(isModuleInstance('string')).toBe(false);
      expect(isModuleInstance(123)).toBe(false);
    });

    it('returns false for object missing definitionId', () => {
      expect(isModuleInstance({ instanceId: 'i1', state: 'unloaded' })).toBe(false);
    });

    it('returns false for object missing instanceId', () => {
      expect(isModuleInstance({ definitionId: 'd1', state: 'unloaded' })).toBe(false);
    });

    it('returns false for object missing state', () => {
      expect(isModuleInstance({ definitionId: 'd1', instanceId: 'i1' })).toBe(false);
    });
  });

  describe('createDefaultMetadata', () => {
    it('creates metadata with provided name', () => {
      const metadata = createDefaultMetadata('@ark/test');

      expect(metadata.name).toBe('@ark/test');
    });

    it('sets default version to 0.0.0', () => {
      const metadata = createDefaultMetadata('@ark/test');

      expect(metadata.version).toBe('0.0.0');
    });

    it('sets default ring to 2', () => {
      const metadata = createDefaultMetadata('@ark/test');

      expect(metadata.ring).toBe(2);
    });

    it('sets empty capabilities array', () => {
      const metadata = createDefaultMetadata('@ark/test');

      expect(metadata.capabilities).toEqual([]);
    });

    it('sets empty tags array', () => {
      const metadata = createDefaultMetadata('@ark/test');

      expect(metadata.tags).toEqual([]);
    });
  });

  describe('createDefaultInstance', () => {
    it('creates instance with provided definitionId', () => {
      const instance = createDefaultInstance('test-module');

      expect(instance.definitionId).toBe('test-module');
    });

    it('generates instanceId if not provided', () => {
      const instance = createDefaultInstance('test-module');

      expect(instance.instanceId).toBeDefined();
      expect(instance.instanceId.length).toBeGreaterThan(0);
    });

    it('uses provided instanceId', () => {
      const instance = createDefaultInstance('test-module', 'custom-id');

      expect(instance.instanceId).toBe('custom-id');
    });

    it('sets initial state to unloaded', () => {
      const instance = createDefaultInstance('test-module');

      expect(instance.state).toBe('unloaded');
    });

    it('sets initial health to unknown', () => {
      const instance = createDefaultInstance('test-module');

      expect(instance.health).toBe('unknown');
    });

    it('sets empty config object', () => {
      const instance = createDefaultInstance('test-module');

      expect(instance.config).toEqual({});
    });
  });

  describe('type definitions', () => {
    it('ModuleState has expected values', () => {
      const states: ModuleState[] = [
        'unloaded',
        'loading',
        'loaded',
        'initializing',
        'ready',
        'starting',
        'running',
        'stopping',
        'stopped',
        'error',
        'disposed',
      ];

      // Verify these compile without error
      expect(states.length).toBe(11);
    });

    it('ModuleHealth has expected values', () => {
      const healths: ModuleHealth[] = ['unknown', 'healthy', 'degraded', 'unhealthy'];

      expect(healths.length).toBe(4);
    });

    it('DependencyStatus has expected values', () => {
      const statuses: DependencyStatus[] = [
        'unresolved',
        'resolving',
        'resolved',
        'failed',
        'circular',
      ];

      expect(statuses.length).toBe(5);
    });
  });
});
