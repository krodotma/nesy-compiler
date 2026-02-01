/**
 * @ark/spine/module-registry tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { ModuleRegistry, createModuleRegistry } from './module-registry.js';
import type { ModuleDefinition, ModuleInstance } from './types.js';

describe('ModuleRegistry', () => {
  let registry: ModuleRegistry;

  beforeEach(() => {
    registry = createModuleRegistry();
  });

  describe('createModuleRegistry', () => {
    it('creates a new registry instance', () => {
      const reg = createModuleRegistry();
      expect(reg).toBeInstanceOf(ModuleRegistry);
    });

    it('accepts configuration options', () => {
      const reg = createModuleRegistry({
        enableCache: true,
        maxCacheSize: 500,
        cacheTtl: 30000,
      });
      expect(reg).toBeInstanceOf(ModuleRegistry);
    });
  });

  describe('definition management', () => {
    const testDef: ModuleDefinition = {
      id: 'test-module',
      metadata: {
        name: '@ark/test',
        version: '1.0.0',
        ring: 2,
        capabilities: ['event-emitter'],
        tags: ['core'],
      },
      dependencies: [],
      exports: ['TestClass'],
    };

    describe('registerDefinition', () => {
      it('registers a new definition', () => {
        const success = registry.registerDefinition(testDef);
        expect(success).toBe(true);
      });

      it('returns false for duplicate registration', () => {
        registry.registerDefinition(testDef);
        const success = registry.registerDefinition(testDef);
        expect(success).toBe(false);
      });

      it('stores definition correctly', () => {
        registry.registerDefinition(testDef);
        const def = registry.getDefinition('test-module');
        expect(def).toBeDefined();
        expect(def!.id).toBe('test-module');
      });
    });

    describe('unregisterDefinition', () => {
      it('unregisters an existing definition', () => {
        registry.registerDefinition(testDef);
        const success = registry.unregisterDefinition('test-module');
        expect(success).toBe(true);
      });

      it('returns false for non-existent definition', () => {
        const success = registry.unregisterDefinition('non-existent');
        expect(success).toBe(false);
      });

      it('throws error when running instances exist', () => {
        registry.registerDefinition(testDef);
        const instance = registry.createInstance('test-module');
        registry.updateInstanceState(instance.instanceId, 'running');

        expect(() => registry.unregisterDefinition('test-module')).toThrow(
          /running instances/
        );
      });

      it('removes definition from registry', () => {
        registry.registerDefinition(testDef);
        registry.unregisterDefinition('test-module');
        expect(registry.getDefinition('test-module')).toBeUndefined();
      });
    });

    describe('getDefinition', () => {
      it('returns definition by ID', () => {
        registry.registerDefinition(testDef);
        const def = registry.getDefinition('test-module');
        expect(def).toBeDefined();
        expect(def!.metadata.name).toBe('@ark/test');
      });

      it('returns undefined for non-existent ID', () => {
        expect(registry.getDefinition('non-existent')).toBeUndefined();
      });
    });

    describe('hasDefinition', () => {
      it('returns true for registered definition', () => {
        registry.registerDefinition(testDef);
        expect(registry.hasDefinition('test-module')).toBe(true);
      });

      it('returns false for non-existent definition', () => {
        expect(registry.hasDefinition('non-existent')).toBe(false);
      });
    });

    describe('getAllDefinitions', () => {
      it('returns empty array initially', () => {
        expect(registry.getAllDefinitions()).toEqual([]);
      });

      it('returns all registered definitions', () => {
        registry.registerDefinition(testDef);
        registry.registerDefinition({
          ...testDef,
          id: 'test-module-2',
        });

        const defs = registry.getAllDefinitions();
        expect(defs.length).toBe(2);
      });
    });

    describe('getDefinitionsByCapability', () => {
      it('filters by capability', () => {
        registry.registerDefinition(testDef);
        registry.registerDefinition({
          ...testDef,
          id: 'other-module',
          metadata: { ...testDef.metadata, capabilities: ['storage'] },
        });

        const defs = registry.getDefinitionsByCapability('event-emitter');
        expect(defs.length).toBe(1);
        expect(defs[0].id).toBe('test-module');
      });
    });

    describe('getDefinitionsByTag', () => {
      it('filters by tag', () => {
        registry.registerDefinition(testDef);
        registry.registerDefinition({
          ...testDef,
          id: 'other-module',
          metadata: { ...testDef.metadata, tags: ['experimental'] },
        });

        const defs = registry.getDefinitionsByTag('core');
        expect(defs.length).toBe(1);
        expect(defs[0].id).toBe('test-module');
      });
    });

    describe('findDefinitions', () => {
      it('filters with custom predicate', () => {
        registry.registerDefinition(testDef);
        registry.registerDefinition({
          ...testDef,
          id: 'ark-bus',
          metadata: { ...testDef.metadata, name: '@ark/bus' },
        });

        const defs = registry.findDefinitions((d) =>
          d.metadata.name.startsWith('@ark/')
        );
        expect(defs.length).toBe(2);
      });
    });

    describe('createDefinition', () => {
      it('creates definition with minimal input', () => {
        const def = registry.createDefinition('new-module', {
          name: '@ark/new',
        });

        expect(def.id).toBe('new-module');
        expect(def.metadata.name).toBe('@ark/new');
        expect(def.metadata.version).toBe('0.0.0');
        expect(def.dependencies).toEqual([]);
        expect(def.exports).toEqual([]);
      });

      it('accepts additional options', () => {
        const def = registry.createDefinition(
          'new-module',
          { name: '@ark/new', version: '2.0.0' },
          {
            dependencies: [{ name: '@ark/core' }],
            exports: ['NewClass'],
            entryPoint: './src/index.js',
          }
        );

        expect(def.metadata.version).toBe('2.0.0');
        expect(def.dependencies.length).toBe(1);
        expect(def.exports).toEqual(['NewClass']);
        expect(def.entryPoint).toBe('./src/index.js');
      });
    });
  });

  describe('instance management', () => {
    const testDef: ModuleDefinition = {
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

    beforeEach(() => {
      registry.registerDefinition(testDef);
    });

    describe('createInstance', () => {
      it('creates a new instance', () => {
        const instance = registry.createInstance('test-module');

        expect(instance).toBeDefined();
        expect(instance.definitionId).toBe('test-module');
        expect(instance.state).toBe('unloaded');
      });

      it('throws for non-existent definition', () => {
        expect(() => registry.createInstance('non-existent')).toThrow(
          /not found/
        );
      });

      it('accepts initial config', () => {
        const instance = registry.createInstance('test-module', { port: 3000 });

        expect(instance.config).toEqual({ port: 3000 });
      });

      it('generates unique instance ID', () => {
        const i1 = registry.createInstance('test-module');
        const i2 = registry.createInstance('test-module');

        expect(i1.instanceId).not.toBe(i2.instanceId);
      });
    });

    describe('getInstance', () => {
      it('returns instance by ID', () => {
        const created = registry.createInstance('test-module');
        const retrieved = registry.getInstance(created.instanceId);

        expect(retrieved).toBeDefined();
        expect(retrieved!.instanceId).toBe(created.instanceId);
      });

      it('returns undefined for non-existent ID', () => {
        expect(registry.getInstance('non-existent')).toBeUndefined();
      });
    });

    describe('hasInstance', () => {
      it('returns true for existing instance', () => {
        const instance = registry.createInstance('test-module');
        expect(registry.hasInstance(instance.instanceId)).toBe(true);
      });

      it('returns false for non-existent instance', () => {
        expect(registry.hasInstance('non-existent')).toBe(false);
      });
    });

    describe('updateInstanceState', () => {
      it('updates instance state', () => {
        const instance = registry.createInstance('test-module');
        const success = registry.updateInstanceState(instance.instanceId, 'loading');

        expect(success).toBe(true);
        expect(registry.getInstance(instance.instanceId)!.state).toBe('loading');
      });

      it('returns false for non-existent instance', () => {
        expect(registry.updateInstanceState('non-existent', 'loading')).toBe(false);
      });

      it('sets startedAt when transitioning to running', () => {
        const instance = registry.createInstance('test-module');
        registry.updateInstanceState(instance.instanceId, 'running');

        expect(registry.getInstance(instance.instanceId)!.startedAt).toBeDefined();
      });

      it('sets stoppedAt when transitioning to stopped', () => {
        const instance = registry.createInstance('test-module');
        registry.updateInstanceState(instance.instanceId, 'stopped');

        expect(registry.getInstance(instance.instanceId)!.stoppedAt).toBeDefined();
      });
    });

    describe('updateInstanceHealth', () => {
      it('updates health status', () => {
        const instance = registry.createInstance('test-module');
        const success = registry.updateInstanceHealth(instance.instanceId, 'healthy');

        expect(success).toBe(true);
        expect(registry.getInstance(instance.instanceId)!.health).toBe('healthy');
      });

      it('updates lastHealthCheck timestamp', () => {
        const instance = registry.createInstance('test-module');
        registry.updateInstanceHealth(instance.instanceId, 'healthy');

        expect(registry.getInstance(instance.instanceId)!.lastHealthCheck).toBeDefined();
      });

      it('sets error message if provided', () => {
        const instance = registry.createInstance('test-module');
        registry.updateInstanceHealth(instance.instanceId, 'unhealthy', 'Connection failed');

        expect(registry.getInstance(instance.instanceId)!.error).toBe('Connection failed');
      });
    });

    describe('updateInstanceConfig', () => {
      it('merges config', () => {
        const instance = registry.createInstance('test-module', { a: 1 });
        registry.updateInstanceConfig(instance.instanceId, { b: 2 });

        expect(registry.getInstance(instance.instanceId)!.config).toEqual({
          a: 1,
          b: 2,
        });
      });
    });

    describe('setInstanceError', () => {
      it('sets error and updates state/health', () => {
        const instance = registry.createInstance('test-module');
        registry.setInstanceError(instance.instanceId, 'Fatal error');

        const updated = registry.getInstance(instance.instanceId)!;
        expect(updated.state).toBe('error');
        expect(updated.health).toBe('unhealthy');
        expect(updated.error).toBe('Fatal error');
      });
    });

    describe('clearInstanceError', () => {
      it('clears error message', () => {
        const instance = registry.createInstance('test-module');
        registry.setInstanceError(instance.instanceId, 'Error');
        registry.clearInstanceError(instance.instanceId);

        expect(registry.getInstance(instance.instanceId)!.error).toBeUndefined();
      });
    });

    describe('setInstanceExports', () => {
      it('sets instance exports', () => {
        const instance = registry.createInstance('test-module');
        registry.setInstanceExports(instance.instanceId, { MyClass: {} });

        expect(registry.getInstance(instance.instanceId)!.exports).toEqual({
          MyClass: {},
        });
      });
    });

    describe('deleteInstance', () => {
      it('removes instance', () => {
        const instance = registry.createInstance('test-module');
        const success = registry.deleteInstance(instance.instanceId);

        expect(success).toBe(true);
        expect(registry.getInstance(instance.instanceId)).toBeUndefined();
      });

      it('returns false for non-existent instance', () => {
        expect(registry.deleteInstance('non-existent')).toBe(false);
      });
    });

    describe('query methods', () => {
      beforeEach(() => {
        const i1 = registry.createInstance('test-module');
        registry.updateInstanceState(i1.instanceId, 'running');
        registry.updateInstanceHealth(i1.instanceId, 'healthy');

        const i2 = registry.createInstance('test-module');
        registry.updateInstanceState(i2.instanceId, 'stopped');

        const i3 = registry.createInstance('test-module');
        registry.updateInstanceHealth(i3.instanceId, 'unhealthy');
      });

      it('getAllInstances returns all instances', () => {
        expect(registry.getAllInstances().length).toBe(3);
      });

      it('getInstancesByDefinition filters by definition', () => {
        const instances = registry.getInstancesByDefinition('test-module');
        expect(instances.length).toBe(3);
      });

      it('getInstancesByState filters by state', () => {
        const running = registry.getInstancesByState('running');
        expect(running.length).toBe(1);
      });

      it('getInstancesByHealth filters by health', () => {
        const healthy = registry.getInstancesByHealth('healthy');
        expect(healthy.length).toBe(1);
      });

      it('getRunningInstances returns running instances', () => {
        const running = registry.getRunningInstances();
        expect(running.length).toBe(1);
        expect(running[0].state).toBe('running');
      });

      it('getUnhealthyInstances returns unhealthy/degraded instances', () => {
        const unhealthy = registry.getUnhealthyInstances();
        expect(unhealthy.length).toBe(1);
      });

      it('findInstances with custom predicate', () => {
        const found = registry.findInstances((i) => i.state !== 'unloaded');
        expect(found.length).toBe(2);
      });
    });
  });

  describe('cache management', () => {
    it('getCached returns undefined for missing key', () => {
      expect(registry.getCached('non-existent')).toBeUndefined();
    });

    it('setCached and getCached work together', () => {
      registry.setCached('key', { value: 42 });
      expect(registry.getCached('key')).toEqual({ value: 42 });
    });

    it('invalidateCache removes entry', () => {
      registry.setCached('key', { value: 42 });
      registry.invalidateCache('key');
      expect(registry.getCached('key')).toBeUndefined();
    });

    it('clearCache removes all entries', () => {
      registry.setCached('key1', 1);
      registry.setCached('key2', 2);
      registry.clearCache();

      expect(registry.getCached('key1')).toBeUndefined();
      expect(registry.getCached('key2')).toBeUndefined();
    });

    it('cache respects TTL', async () => {
      const shortTtlRegistry = createModuleRegistry({ cacheTtl: 50 });
      shortTtlRegistry.setCached('key', 'value');

      expect(shortTtlRegistry.getCached('key')).toBe('value');

      await new Promise((r) => setTimeout(r, 60));

      expect(shortTtlRegistry.getCached('key')).toBeUndefined();
    });
  });

  describe('getStats', () => {
    it('returns correct statistics', () => {
      const def: ModuleDefinition = {
        id: 'test',
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

      registry.registerDefinition(def);

      const i1 = registry.createInstance('test');
      registry.updateInstanceState(i1.instanceId, 'running');
      registry.updateInstanceHealth(i1.instanceId, 'healthy');

      const i2 = registry.createInstance('test');
      registry.updateInstanceState(i2.instanceId, 'error');
      registry.updateInstanceHealth(i2.instanceId, 'unhealthy');

      const stats = registry.getStats();

      expect(stats.definitions).toBe(1);
      expect(stats.instances).toBe(2);
      expect(stats.byState.running).toBe(1);
      expect(stats.byState.error).toBe(1);
      expect(stats.byHealth.healthy).toBe(1);
      expect(stats.byHealth.unhealthy).toBe(1);
    });
  });

  describe('clear', () => {
    it('clears all data', () => {
      const def: ModuleDefinition = {
        id: 'test',
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

      registry.registerDefinition(def);
      registry.createInstance('test');
      registry.setCached('key', 'value');

      registry.clear();

      expect(registry.getAllDefinitions().length).toBe(0);
      expect(registry.getAllInstances().length).toBe(0);
      expect(registry.getCached('key')).toBeUndefined();
    });
  });
});
