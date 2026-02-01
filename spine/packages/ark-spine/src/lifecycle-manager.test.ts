/**
 * @ark/spine/lifecycle-manager tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  LifecycleManager,
  createLifecycleManager,
} from './lifecycle-manager.js';
import type { ModuleInstance, ModuleState, LifecycleHook } from './types.js';

describe('LifecycleManager', () => {
  let manager: LifecycleManager;

  const createInstance = (state: ModuleState = 'unloaded'): ModuleInstance => ({
    definitionId: 'test-module',
    instanceId: 'test-instance',
    state,
    health: 'unknown',
    config: {},
  });

  beforeEach(() => {
    manager = createLifecycleManager();
  });

  describe('createLifecycleManager', () => {
    it('creates a new manager instance', () => {
      const mgr = createLifecycleManager();
      expect(mgr).toBeInstanceOf(LifecycleManager);
    });

    it('accepts configuration', () => {
      const mgr = createLifecycleManager({
        defaultTimeout: 5000,
        retryCount: 5,
        retryDelay: 500,
      });
      expect(mgr.getConfig().defaultTimeout).toBe(5000);
    });
  });

  describe('hook management', () => {
    describe('onHook', () => {
      it('registers a hook handler', async () => {
        const handler = vi.fn();
        manager.onHook('afterLoad', handler);

        await manager.triggerHook('afterLoad', createInstance());

        expect(handler).toHaveBeenCalled();
      });

      it('returns unsubscribe function', async () => {
        const handler = vi.fn();
        const unsubscribe = manager.onHook('afterLoad', handler);

        unsubscribe();
        await manager.triggerHook('afterLoad', createInstance());

        expect(handler).not.toHaveBeenCalled();
      });

      it('allows multiple handlers per hook', async () => {
        const handler1 = vi.fn();
        const handler2 = vi.fn();

        manager.onHook('afterLoad', handler1);
        manager.onHook('afterLoad', handler2);

        await manager.triggerHook('afterLoad', createInstance());

        expect(handler1).toHaveBeenCalled();
        expect(handler2).toHaveBeenCalled();
      });
    });

    describe('offHook', () => {
      it('removes a specific handler', async () => {
        const handler = vi.fn();
        manager.onHook('afterLoad', handler);
        const removed = manager.offHook('afterLoad', handler);

        expect(removed).toBe(true);

        await manager.triggerHook('afterLoad', createInstance());
        expect(handler).not.toHaveBeenCalled();
      });

      it('returns false for non-existent handler', () => {
        const handler = vi.fn();
        expect(manager.offHook('afterLoad', handler)).toBe(false);
      });
    });

    describe('clearHook', () => {
      it('clears all handlers for a hook', async () => {
        const handler1 = vi.fn();
        const handler2 = vi.fn();

        manager.onHook('afterLoad', handler1);
        manager.onHook('afterLoad', handler2);

        manager.clearHook('afterLoad');

        await manager.triggerHook('afterLoad', createInstance());

        expect(handler1).not.toHaveBeenCalled();
        expect(handler2).not.toHaveBeenCalled();
      });
    });

    describe('clearAllHooks', () => {
      it('clears all hooks', async () => {
        manager.onHook('afterLoad', vi.fn());
        manager.onHook('afterStart', vi.fn());

        manager.clearAllHooks();

        expect(manager.getRegisteredHooks().length).toBe(0);
      });
    });

    describe('triggerHook', () => {
      it('passes instance to handler', async () => {
        const handler = vi.fn();
        const instance = createInstance();

        manager.onHook('afterLoad', handler);
        await manager.triggerHook('afterLoad', instance);

        expect(handler).toHaveBeenCalledWith(instance, expect.any(Object));
      });

      it('passes context to handler', async () => {
        const handler = vi.fn();

        manager.onHook('afterLoad', handler);
        await manager.triggerHook('afterLoad', createInstance());

        expect(handler).toHaveBeenCalledWith(
          expect.any(Object),
          expect.objectContaining({ timestamp: expect.any(Number) })
        );
      });

      it('throws AggregateError on handler failures', async () => {
        const handler1 = vi.fn().mockRejectedValue(new Error('Error 1'));
        const handler2 = vi.fn().mockRejectedValue(new Error('Error 2'));

        manager.onHook('afterLoad', handler1);
        manager.onHook('afterLoad', handler2);

        await expect(
          manager.triggerHook('afterLoad', createInstance())
        ).rejects.toThrow(AggregateError);
      });

      it('does nothing for hooks with no handlers', async () => {
        // Should not throw
        await manager.triggerHook('afterLoad', createInstance());
      });
    });
  });

  describe('state transitions', () => {
    describe('canTransition', () => {
      it('allows valid transitions', () => {
        expect(manager.canTransition('unloaded', 'loading')).toBe(true);
        expect(manager.canTransition('loading', 'loaded')).toBe(true);
        expect(manager.canTransition('loaded', 'initializing')).toBe(true);
        expect(manager.canTransition('initializing', 'ready')).toBe(true);
        expect(manager.canTransition('ready', 'starting')).toBe(true);
        expect(manager.canTransition('starting', 'running')).toBe(true);
        expect(manager.canTransition('running', 'stopping')).toBe(true);
        expect(manager.canTransition('stopping', 'stopped')).toBe(true);
        expect(manager.canTransition('stopped', 'starting')).toBe(true);
      });

      it('rejects invalid transitions', () => {
        expect(manager.canTransition('unloaded', 'running')).toBe(false);
        expect(manager.canTransition('loading', 'running')).toBe(false);
        expect(manager.canTransition('stopped', 'loading')).toBe(false);
      });

      it('disposed is terminal', () => {
        expect(manager.canTransition('disposed', 'loading')).toBe(false);
        expect(manager.canTransition('disposed', 'unloaded')).toBe(false);
      });

      it('allows error recovery', () => {
        expect(manager.canTransition('error', 'unloaded')).toBe(true);
        expect(manager.canTransition('error', 'disposed')).toBe(true);
      });
    });

    describe('getAllowedTransitions', () => {
      it('returns allowed states from unloaded', () => {
        const allowed = manager.getAllowedTransitions('unloaded');
        expect(allowed).toContain('loading');
        expect(allowed.length).toBe(1);
      });

      it('returns empty array for disposed', () => {
        const allowed = manager.getAllowedTransitions('disposed');
        expect(allowed).toEqual([]);
      });
    });

    describe('transition', () => {
      it('performs valid transition', async () => {
        const instance = createInstance('unloaded');
        let currentState: ModuleState = 'unloaded';

        const result = await manager.transition(
          instance,
          'loading',
          (state) => {
            currentState = state;
          }
        );

        expect(result.success).toBe(true);
        expect(currentState).toBe('loading');
      });

      it('rejects invalid transition', async () => {
        const instance = createInstance('unloaded');

        const result = await manager.transition(instance, 'running', vi.fn());

        expect(result.success).toBe(false);
        expect(result.error).toContain('Invalid transition');
      });

      it('triggers before/after hooks', async () => {
        const beforeHandler = vi.fn();
        const afterHandler = vi.fn();

        manager.onHook('beforeLoad', beforeHandler);
        manager.onHook('afterLoad', afterHandler);

        const instance = createInstance('unloaded');

        // loading triggers beforeLoad
        await manager.transition(instance, 'loading', vi.fn());
        expect(beforeHandler).toHaveBeenCalled();

        // loaded triggers afterLoad - need to transition from loading
        await manager.transition(
          { ...instance, state: 'loading' },
          'loaded',
          vi.fn()
        );
        expect(afterHandler).toHaveBeenCalled();
      });

      it('records duration', async () => {
        const instance = createInstance('unloaded');
        const result = await manager.transition(instance, 'loading', vi.fn());

        expect(result.duration).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('lifecycle operations', () => {
    describe('load', () => {
      it('transitions through loading to loaded', async () => {
        const instance = createInstance('unloaded');
        const states: ModuleState[] = [];

        const result = await manager.load(
          instance,
          async () => {},
          (state) => states.push(state)
        );

        expect(result.success).toBe(true);
        expect(states).toContain('loading');
        expect(states).toContain('loaded');
      });

      it('calls load function', async () => {
        const loadFn = vi.fn();
        const instance = createInstance('unloaded');

        await manager.load(instance, loadFn, vi.fn());

        expect(loadFn).toHaveBeenCalled();
      });

      it('handles load function errors', async () => {
        const instance = createInstance('unloaded');
        let finalState: ModuleState = 'unloaded';

        const result = await manager.load(
          instance,
          async () => {
            throw new Error('Load failed');
          },
          (state) => {
            finalState = state;
          }
        );

        expect(result.success).toBe(false);
        expect(result.error).toContain('Load failed');
        expect(finalState).toBe('error');
      });
    });

    describe('initialize', () => {
      it('transitions from loaded to ready', async () => {
        const instance = createInstance('loaded');
        const states: ModuleState[] = [];

        const result = await manager.initialize(
          instance,
          async () => {},
          (state) => states.push(state)
        );

        expect(result.success).toBe(true);
        expect(states).toContain('initializing');
        expect(states).toContain('ready');
      });
    });

    describe('start', () => {
      it('transitions from ready to running', async () => {
        const instance = createInstance('ready');
        const states: ModuleState[] = [];

        const result = await manager.start(
          instance,
          async () => {},
          (state) => states.push(state)
        );

        expect(result.success).toBe(true);
        expect(states).toContain('starting');
        expect(states).toContain('running');
      });

      it('can start from stopped state', async () => {
        const instance = createInstance('stopped');
        const states: ModuleState[] = [];

        const result = await manager.start(
          instance,
          async () => {},
          (state) => states.push(state)
        );

        expect(result.success).toBe(true);
      });
    });

    describe('stop', () => {
      it('transitions from running to stopped', async () => {
        const instance = createInstance('running');
        const states: ModuleState[] = [];

        const result = await manager.stop(
          instance,
          async () => {},
          (state) => states.push(state)
        );

        expect(result.success).toBe(true);
        expect(states).toContain('stopping');
        expect(states).toContain('stopped');
      });

      it('retries on failure', async () => {
        const instance = createInstance('running');
        let attempts = 0;

        const mgr = createLifecycleManager({ retryCount: 2, retryDelay: 10 });

        const result = await mgr.stop(
          instance,
          async () => {
            attempts++;
            if (attempts < 3) {
              throw new Error('Stop failed');
            }
          },
          vi.fn()
        );

        expect(result.success).toBe(true);
        expect(attempts).toBe(3);
      });
    });

    describe('dispose', () => {
      it('transitions to disposed', async () => {
        const instance = createInstance('stopped');
        let finalState: ModuleState = 'stopped';

        const result = await manager.dispose(
          instance,
          async () => {},
          (state) => {
            finalState = state;
          }
        );

        expect(result.success).toBe(true);
        expect(finalState).toBe('disposed');
      });

      it('returns success for already disposed', async () => {
        const instance = createInstance('disposed');

        const result = await manager.dispose(instance, async () => {}, vi.fn());

        expect(result.success).toBe(true);
        expect(result.duration).toBe(0);
      });

      it('triggers dispose hooks', async () => {
        const beforeHandler = vi.fn();
        const afterHandler = vi.fn();

        manager.onHook('beforeDispose', beforeHandler);
        manager.onHook('afterDispose', afterHandler);

        const instance = createInstance('stopped');
        await manager.dispose(instance, async () => {}, vi.fn());

        expect(beforeHandler).toHaveBeenCalled();
        expect(afterHandler).toHaveBeenCalled();
      });
    });

    describe('healthCheck', () => {
      it('executes health check function', async () => {
        const instance = createInstance('running');

        const result = await manager.healthCheck(instance, async () => true);

        expect(result.success).toBe(true);
        expect(result.data).toBe(true);
      });

      it('triggers health check hook', async () => {
        const handler = vi.fn();
        manager.onHook('onHealthCheck', handler);

        const instance = createInstance('running');
        await manager.healthCheck(instance, async () => true);

        expect(handler).toHaveBeenCalled();
      });

      it('handles check failure', async () => {
        const instance = createInstance('running');

        const result = await manager.healthCheck(instance, async () => {
          throw new Error('Health check failed');
        });

        expect(result.success).toBe(false);
        expect(result.data).toBe(false);
        expect(result.error).toContain('Health check failed');
      });

      it('uses shorter timeout for health checks', async () => {
        const instance = createInstance('running');

        const result = await manager.healthCheck(
          instance,
          async () => {
            await new Promise((r) => setTimeout(r, 100));
            return true;
          },
          { timeout: 50 }
        );

        expect(result.success).toBe(false);
        expect(result.error).toContain('timed out');
      });
    });
  });

  describe('timeout handling', () => {
    it('times out long operations', async () => {
      const mgr = createLifecycleManager({ defaultTimeout: 50 });
      const instance = createInstance('unloaded');

      const result = await mgr.load(
        instance,
        async () => {
          await new Promise((r) => setTimeout(r, 200));
        },
        vi.fn()
      );

      expect(result.success).toBe(false);
      expect(result.error).toContain('timed out');
    });
  });

  describe('getConfig', () => {
    it('returns configuration', () => {
      const config = manager.getConfig();

      expect(config.defaultTimeout).toBeDefined();
      expect(config.retryCount).toBeDefined();
      expect(config.retryDelay).toBeDefined();
    });
  });

  describe('getRegisteredHooks', () => {
    it('returns list of registered hooks', () => {
      manager.onHook('afterLoad', vi.fn());
      manager.onHook('afterStart', vi.fn());

      const hooks = manager.getRegisteredHooks();

      expect(hooks).toContain('afterLoad');
      expect(hooks).toContain('afterStart');
    });
  });

  describe('getHandlerCount', () => {
    it('returns handler count for hook', () => {
      manager.onHook('afterLoad', vi.fn());
      manager.onHook('afterLoad', vi.fn());

      expect(manager.getHandlerCount('afterLoad')).toBe(2);
    });

    it('returns 0 for unregistered hook', () => {
      expect(manager.getHandlerCount('afterLoad')).toBe(0);
    });
  });
});
