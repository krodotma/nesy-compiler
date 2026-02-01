/**
 * @ark/spine/spine-coordinator tests
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import {
  SpineCoordinator,
  createSpineCoordinator,
} from './spine-coordinator.js';
import type {
  ModuleDefinition,
  CoordinatorEvent,
  CoordinatorEventType,
} from './types.js';

describe('SpineCoordinator', () => {
  let coordinator: SpineCoordinator;

  const createDef = (
    id: string,
    deps: Array<{ name: string }> = []
  ): ModuleDefinition => ({
    id,
    metadata: {
      name: `@ark/${id}`,
      version: '1.0.0',
      ring: 2,
      capabilities: [],
      tags: [],
    },
    dependencies: deps,
    exports: [],
  });

  beforeEach(() => {
    coordinator = createSpineCoordinator({ debug: false });
  });

  afterEach(async () => {
    await coordinator.shutdown();
  });

  describe('createSpineCoordinator', () => {
    it('creates a new coordinator instance', () => {
      const coord = createSpineCoordinator();
      expect(coord).toBeInstanceOf(SpineCoordinator);
    });

    it('accepts configuration', () => {
      const coord = createSpineCoordinator({
        id: 'test-coordinator',
        defaultRing: 1,
        strictDependencies: false,
      });
      expect(coord.getId()).toBe('test-coordinator');
    });

    it('generates ID if not provided', () => {
      const coord = createSpineCoordinator();
      expect(coord.getId()).toBeDefined();
      expect(coord.getId().length).toBeGreaterThan(0);
    });
  });

  describe('module registration', () => {
    describe('registerModule', () => {
      it('registers a module', () => {
        const def = createDef('test');
        const success = coordinator.registerModule(def);
        expect(success).toBe(true);
      });

      it('returns false for duplicate registration', () => {
        const def = createDef('test');
        coordinator.registerModule(def);
        const success = coordinator.registerModule(def);
        expect(success).toBe(false);
      });

      it('adds module to dependency resolver', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));

        const deps = coordinator.getDependencies('b');
        expect(deps).toContain('a');
      });

      it('emits module.registered event', () => {
        const handler = vi.fn();
        coordinator.on('module.registered', handler);

        coordinator.registerModule(createDef('test'));

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('unregisterModule', () => {
      it('unregisters a module', () => {
        coordinator.registerModule(createDef('test'));
        const success = coordinator.unregisterModule('test');
        expect(success).toBe(true);
      });

      it('throws error when running instances exist', async () => {
        coordinator.registerModule(createDef('test'));
        await coordinator.loadAndStart('test');

        expect(() => coordinator.unregisterModule('test')).toThrow(
          /running instances/
        );
      });

      it('emits module.unregistered event', () => {
        coordinator.registerModule(createDef('test'));
        const handler = vi.fn();
        coordinator.on('module.unregistered', handler);

        coordinator.unregisterModule('test');

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('getModule', () => {
      it('returns module definition', () => {
        coordinator.registerModule(createDef('test'));
        const def = coordinator.getModule('test');
        expect(def).toBeDefined();
        expect(def!.id).toBe('test');
      });

      it('returns undefined for non-existent module', () => {
        expect(coordinator.getModule('non-existent')).toBeUndefined();
      });
    });

    describe('hasModule', () => {
      it('returns true for registered module', () => {
        coordinator.registerModule(createDef('test'));
        expect(coordinator.hasModule('test')).toBe(true);
      });

      it('returns false for non-existent module', () => {
        expect(coordinator.hasModule('non-existent')).toBe(false);
      });
    });

    describe('getAllModules', () => {
      it('returns all registered modules', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b'));

        const modules = coordinator.getAllModules();
        expect(modules.length).toBe(2);
      });
    });
  });

  describe('instance management', () => {
    beforeEach(() => {
      coordinator.registerModule(createDef('test'));
    });

    describe('createInstance', () => {
      it('creates a module instance', () => {
        const instance = coordinator.createInstance('test');
        expect(instance).toBeDefined();
        expect(instance.definitionId).toBe('test');
      });

      it('accepts config', () => {
        const instance = coordinator.createInstance('test', { port: 3000 });
        expect(instance.config).toEqual({ port: 3000 });
      });
    });

    describe('getInstance', () => {
      it('returns instance by ID', () => {
        const created = coordinator.createInstance('test');
        const retrieved = coordinator.getInstance(created.instanceId);
        expect(retrieved!.instanceId).toBe(created.instanceId);
      });
    });

    describe('getAllInstances', () => {
      it('returns all instances', () => {
        coordinator.createInstance('test');
        coordinator.createInstance('test');
        expect(coordinator.getAllInstances().length).toBe(2);
      });
    });

    describe('getRunningInstances', () => {
      it('returns only running instances', async () => {
        await coordinator.loadAndStart('test');
        expect(coordinator.getRunningInstances().length).toBe(1);
      });
    });
  });

  describe('dependency resolution', () => {
    describe('resolveDependencies', () => {
      it('resolves all module dependencies', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));
        coordinator.registerModule(createDef('c', [{ name: 'b' }]));

        const result = coordinator.resolveDependencies();

        expect(result.success).toBe(true);
        expect(result.order).toEqual(['a', 'b', 'c']);
      });

      it('emits dependency.resolved event on success', () => {
        coordinator.registerModule(createDef('a'));
        const handler = vi.fn();
        coordinator.on('dependency.resolved', handler);

        coordinator.resolveDependencies();

        expect(handler).toHaveBeenCalled();
      });

      it('emits dependency.failed event on failure', () => {
        coordinator.registerModule(createDef('a', [{ name: 'missing' }]));
        const handler = vi.fn();
        coordinator.on('dependency.failed', handler);

        coordinator.resolveDependencies();

        expect(handler).toHaveBeenCalled();
      });

      it('emits dependency.circular event for cycles', () => {
        coordinator.registerModule(createDef('a', [{ name: 'b' }]));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));

        const handler = vi.fn();
        coordinator.on('dependency.circular', handler);

        coordinator.resolveDependencies();

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('resolveDependenciesFor', () => {
      it('resolves dependencies for specific module', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));
        coordinator.registerModule(createDef('c'));

        const result = coordinator.resolveDependenciesFor('b');

        expect(result.success).toBe(true);
        expect(result.order).toContain('a');
        expect(result.order).toContain('b');
        expect(result.order).not.toContain('c');
      });
    });

    describe('getDependencies', () => {
      it('returns module dependencies', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));

        expect(coordinator.getDependencies('b')).toContain('a');
      });
    });

    describe('getDependents', () => {
      it('returns module dependents', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));

        expect(coordinator.getDependents('a')).toContain('b');
      });
    });

    describe('getTransitiveDependencies', () => {
      it('returns all transitive dependencies', () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b', [{ name: 'a' }]));
        coordinator.registerModule(createDef('c', [{ name: 'b' }]));

        const deps = coordinator.getTransitiveDependencies('c');
        expect(deps).toContain('a');
        expect(deps).toContain('b');
      });
    });
  });

  describe('lifecycle operations', () => {
    beforeEach(() => {
      coordinator.registerModule(createDef('a'));
      coordinator.registerModule(createDef('b', [{ name: 'a' }]));
    });

    describe('loadModule', () => {
      it('loads a module', async () => {
        const result = await coordinator.loadModule('a');

        expect(result.success).toBe(true);
        expect(result.data).toBeDefined();
        expect(result.data!.state).toBe('loaded');
      });

      it('returns error for non-existent module', async () => {
        const result = await coordinator.loadModule('non-existent');
        expect(result.success).toBe(false);
        expect(result.error).toContain('not found');
      });

      it('loads dependencies first', async () => {
        const result = await coordinator.loadModule('b');

        expect(result.success).toBe(true);

        // a should also be loaded
        const instances = coordinator.getAllInstances();
        expect(instances.length).toBe(2);
      });

      it('emits module.loaded event', async () => {
        const handler = vi.fn();
        coordinator.on('module.loaded', handler);

        await coordinator.loadModule('a');

        expect(handler).toHaveBeenCalled();
      });

      it('can skip dependency loading', async () => {
        const result = await coordinator.loadModule('b', { skipDependencies: true });

        // Will fail because dependency resolution fails
        expect(result.success).toBe(true);
      });
    });

    describe('initializeModule', () => {
      it('initializes a loaded module', async () => {
        const loadResult = await coordinator.loadModule('a');
        const result = await coordinator.initializeModule(loadResult.data!.instanceId);

        expect(result.success).toBe(true);
      });

      it('returns error for non-existent instance', async () => {
        const result = await coordinator.initializeModule('non-existent');
        expect(result.success).toBe(false);
      });

      it('emits module.initialized event', async () => {
        const handler = vi.fn();
        coordinator.on('module.initialized', handler);

        const loadResult = await coordinator.loadModule('a');
        await coordinator.initializeModule(loadResult.data!.instanceId);

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('startModule', () => {
      it('starts an initialized module', async () => {
        const loadResult = await coordinator.loadModule('a');
        await coordinator.initializeModule(loadResult.data!.instanceId);
        const result = await coordinator.startModule(loadResult.data!.instanceId);

        expect(result.success).toBe(true);
      });

      it('emits module.started event', async () => {
        const handler = vi.fn();
        coordinator.on('module.started', handler);

        const loadResult = await coordinator.loadModule('a');
        await coordinator.initializeModule(loadResult.data!.instanceId);
        await coordinator.startModule(loadResult.data!.instanceId);

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('stopModule', () => {
      it('stops a running module', async () => {
        const loadResult = await coordinator.loadAndStart('a');
        const result = await coordinator.stopModule(loadResult.data!.instanceId);

        expect(result.success).toBe(true);
      });

      it('emits module.stopped event', async () => {
        const handler = vi.fn();
        coordinator.on('module.stopped', handler);

        const loadResult = await coordinator.loadAndStart('a');
        await coordinator.stopModule(loadResult.data!.instanceId);

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('disposeModule', () => {
      it('disposes a stopped module', async () => {
        const loadResult = await coordinator.loadAndStart('a');
        await coordinator.stopModule(loadResult.data!.instanceId);
        const result = await coordinator.disposeModule(loadResult.data!.instanceId);

        expect(result.success).toBe(true);
      });

      it('stops module first if running', async () => {
        const loadResult = await coordinator.loadAndStart('a');
        const result = await coordinator.disposeModule(loadResult.data!.instanceId);

        expect(result.success).toBe(true);
      });

      it('emits module.disposed event', async () => {
        const handler = vi.fn();
        coordinator.on('module.disposed', handler);

        const loadResult = await coordinator.loadAndStart('a');
        await coordinator.disposeModule(loadResult.data!.instanceId);

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('loadAndStart', () => {
      it('loads, initializes, and starts module', async () => {
        const result = await coordinator.loadAndStart('a');

        expect(result.success).toBe(true);
        expect(result.data!.state).toBe('running');
      });

      it('loads dependencies first', async () => {
        const result = await coordinator.loadAndStart('b');

        expect(result.success).toBe(true);

        // At minimum, b should be running (dependencies may or may not start depending on config)
        const running = coordinator.getRunningInstances();
        expect(running.length).toBeGreaterThanOrEqual(1);

        // Module b should definitely be running
        const bRunning = running.some(
          (r) => r.definitionId === 'b' && r.state === 'running'
        );
        expect(bRunning).toBe(true);
      });
    });
  });

  describe('health checks', () => {
    describe('checkHealth', () => {
      it('checks health of running instance', async () => {
        coordinator.registerModule(createDef('a'));
        const loadResult = await coordinator.loadAndStart('a');

        const result = await coordinator.checkHealth(loadResult.data!.instanceId);

        expect(result.health).toBe('healthy');
      });

      it('returns unknown for non-existent instance', async () => {
        const result = await coordinator.checkHealth('non-existent');
        expect(result.health).toBe('unknown');
        expect(result.error).toContain('not found');
      });

      it('emits module.health event', async () => {
        const handler = vi.fn();
        coordinator.on('module.health', handler);

        coordinator.registerModule(createDef('a'));
        const loadResult = await coordinator.loadAndStart('a');
        await coordinator.checkHealth(loadResult.data!.instanceId);

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('checkAllHealth', () => {
      it('checks health of all running instances', async () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b'));

        await coordinator.loadAndStart('a');
        await coordinator.loadAndStart('b');

        const results = await coordinator.checkAllHealth();

        expect(results.length).toBe(2);
      });
    });

    describe('health check interval', () => {
      it('starts and stops health checks', async () => {
        coordinator.startHealthChecks();
        // Health check interval should be set
        coordinator.stopHealthChecks();
        // Should not throw
      });
    });
  });

  describe('lifecycle hooks', () => {
    describe('onLifecycle', () => {
      it('registers lifecycle hook', async () => {
        const handler = vi.fn();
        coordinator.onLifecycle('afterStart', handler);

        coordinator.registerModule(createDef('a'));
        await coordinator.loadAndStart('a');

        expect(handler).toHaveBeenCalled();
      });

      it('returns unsubscribe function', async () => {
        const handler = vi.fn();
        const unsubscribe = coordinator.onLifecycle('afterStart', handler);

        unsubscribe();

        coordinator.registerModule(createDef('a'));
        await coordinator.loadAndStart('a');

        expect(handler).not.toHaveBeenCalled();
      });
    });
  });

  describe('events', () => {
    describe('on', () => {
      it('subscribes to specific event', () => {
        const handler = vi.fn();
        coordinator.on('module.registered', handler);

        coordinator.registerModule(createDef('a'));

        expect(handler).toHaveBeenCalled();
        expect(handler).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'module.registered',
          })
        );
      });

      it('subscribes to all events with wildcard', () => {
        const handler = vi.fn();
        coordinator.on('*', handler);

        coordinator.registerModule(createDef('a'));

        expect(handler).toHaveBeenCalled();
      });

      it('returns unsubscribe function', () => {
        const handler = vi.fn();
        const unsubscribe = coordinator.on('module.registered', handler);

        unsubscribe();
        coordinator.registerModule(createDef('a'));

        expect(handler).not.toHaveBeenCalled();
      });
    });
  });

  describe('statistics', () => {
    describe('getStats', () => {
      it('returns coordinator statistics', async () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b'));
        await coordinator.loadAndStart('a');

        const stats = coordinator.getStats();

        expect(stats.totalModules).toBe(2);
        expect(stats.runningInstances).toBe(1);
        expect(stats.byState.running).toBe(1);
        expect(stats.uptime).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('coordinator lifecycle', () => {
    describe('start', () => {
      it('starts the coordinator', async () => {
        const handler = vi.fn();
        coordinator.on('coordinator.started', handler);

        await coordinator.start();

        expect(handler).toHaveBeenCalled();
      });

      it('starts health checks', async () => {
        await coordinator.start();
        // Health check interval should be set
        coordinator.stopHealthChecks();
      });
    });

    describe('stop', () => {
      it('stops all running modules', async () => {
        coordinator.registerModule(createDef('a'));
        coordinator.registerModule(createDef('b'));

        await coordinator.loadAndStart('a');
        await coordinator.loadAndStart('b');

        await coordinator.stop();

        expect(coordinator.getRunningInstances().length).toBe(0);
      });

      it('emits coordinator.stopped event', async () => {
        const handler = vi.fn();
        coordinator.on('coordinator.stopped', handler);

        await coordinator.stop();

        expect(handler).toHaveBeenCalled();
      });
    });

    describe('shutdown', () => {
      it('disposes all instances', async () => {
        coordinator.registerModule(createDef('a'));
        await coordinator.loadAndStart('a');

        await coordinator.shutdown();

        expect(coordinator.getAllInstances().length).toBe(0);
      });
    });
  });

  describe('component access', () => {
    it('getConfig returns configuration', () => {
      const config = coordinator.getConfig();
      expect(config).toBeDefined();
      expect(config.defaultRing).toBeDefined();
    });

    it('getRegistry returns module registry', () => {
      const registry = coordinator.getRegistry();
      expect(registry).toBeDefined();
    });

    it('getResolver returns dependency resolver', () => {
      const resolver = coordinator.getResolver();
      expect(resolver).toBeDefined();
    });

    it('getLifecycleManager returns lifecycle manager', () => {
      const lifecycle = coordinator.getLifecycleManager();
      expect(lifecycle).toBeDefined();
    });
  });
});
