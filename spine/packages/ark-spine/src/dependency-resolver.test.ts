/**
 * @ark/spine/dependency-resolver tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  DependencyResolver,
  createDependencyResolver,
} from './dependency-resolver.js';
import type { ModuleDefinition } from './types.js';

describe('DependencyResolver', () => {
  let resolver: DependencyResolver;

  beforeEach(() => {
    resolver = createDependencyResolver();
  });

  const createDef = (
    id: string,
    deps: Array<{ name: string; optional?: boolean }> = []
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

  describe('createDependencyResolver', () => {
    it('creates a new resolver instance', () => {
      const res = createDependencyResolver();
      expect(res).toBeInstanceOf(DependencyResolver);
    });

    it('accepts configuration', () => {
      const res = createDependencyResolver({
        maxDepth: 50,
        allowCircular: true,
        timeout: 5000,
      });
      expect(res).toBeInstanceOf(DependencyResolver);
    });
  });

  describe('buildGraph', () => {
    it('builds graph from definitions', () => {
      const defs = [
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
      ];

      resolver.buildGraph(defs);

      expect(resolver.getNode('a')).toBeDefined();
      expect(resolver.getNode('b')).toBeDefined();
      expect(resolver.getNode('c')).toBeDefined();
    });

    it('sets up dependency edges correctly', () => {
      const defs = [
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
      ];

      resolver.buildGraph(defs);

      expect(resolver.getDependencies('b')).toContain('a');
      expect(resolver.getDependents('a')).toContain('b');
    });

    it('marks node as failed for missing required dependency', () => {
      const defs = [
        createDef('a', [{ name: 'non-existent' }]),
      ];

      resolver.buildGraph(defs);

      expect(resolver.getNode('a')!.status).toBe('failed');
    });

    it('allows missing optional dependencies', () => {
      const defs = [
        createDef('a', [{ name: 'non-existent', optional: true }]),
      ];

      resolver.buildGraph(defs);

      expect(resolver.getNode('a')!.status).not.toBe('failed');
    });
  });

  describe('addModule', () => {
    it('adds a module to existing graph', () => {
      resolver.buildGraph([createDef('a')]);
      resolver.addModule(createDef('b', [{ name: 'a' }]));

      expect(resolver.getNode('b')).toBeDefined();
      expect(resolver.getDependencies('b')).toContain('a');
    });

    it('does not add duplicate module', () => {
      resolver.buildGraph([createDef('a')]);
      resolver.addModule(createDef('a'));

      // Should not throw or create duplicate
      expect(resolver.getNode('a')).toBeDefined();
    });
  });

  describe('removeModule', () => {
    it('removes a module from graph', () => {
      resolver.buildGraph([createDef('a'), createDef('b')]);
      const success = resolver.removeModule('a');

      expect(success).toBe(true);
      expect(resolver.getNode('a')).toBeUndefined();
    });

    it('returns false for non-existent module', () => {
      const success = resolver.removeModule('non-existent');
      expect(success).toBe(false);
    });

    it('cleans up dependency edges', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
      ]);

      resolver.removeModule('a');

      expect(resolver.getDependencies('b')).not.toContain('a');
    });

    it('cleans up dependent edges', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
      ]);

      resolver.removeModule('b');

      expect(resolver.getDependents('a')).not.toContain('b');
    });
  });

  describe('resolve', () => {
    it('resolves simple chain', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
      ]);

      const result = resolver.resolve();

      expect(result.success).toBe(true);
      expect(result.order).toEqual(['a', 'b', 'c']);
    });

    it('resolves diamond dependency', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'a' }]),
        createDef('d', [{ name: 'b' }, { name: 'c' }]),
      ]);

      const result = resolver.resolve();

      expect(result.success).toBe(true);
      expect(result.order.indexOf('a')).toBeLessThan(result.order.indexOf('b'));
      expect(result.order.indexOf('a')).toBeLessThan(result.order.indexOf('c'));
      expect(result.order.indexOf('b')).toBeLessThan(result.order.indexOf('d'));
      expect(result.order.indexOf('c')).toBeLessThan(result.order.indexOf('d'));
    });

    it('handles modules with no dependencies', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b'),
        createDef('c'),
      ]);

      const result = resolver.resolve();

      expect(result.success).toBe(true);
      expect(result.order.length).toBe(3);
    });

    it('detects circular dependencies', () => {
      resolver.buildGraph([
        createDef('a', [{ name: 'c' }]),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
      ]);

      const result = resolver.resolve();

      expect(result.success).toBe(false);
      expect(result.circular.length).toBeGreaterThan(0);
    });

    it('reports failed dependencies', () => {
      resolver.buildGraph([
        createDef('a', [{ name: 'missing' }]),
      ]);

      const result = resolver.resolve();

      expect(result.success).toBe(false);
      expect(result.failed).toContain('a');
    });

    it('allows circular with config', () => {
      const circularResolver = createDependencyResolver({ allowCircular: true });

      circularResolver.buildGraph([
        createDef('a', [{ name: 'b' }]),
        createDef('b', [{ name: 'a' }]),
      ]);

      const result = circularResolver.resolve();

      expect(result.success).toBe(true);
      expect(result.circular.length).toBeGreaterThan(0);
      expect(result.order.length).toBe(2);
    });

    it('records resolution duration', () => {
      resolver.buildGraph([createDef('a')]);
      const result = resolver.resolve();

      expect(result.duration).toBeGreaterThanOrEqual(0);
    });
  });

  describe('resolveFor', () => {
    it('resolves dependencies for specific module', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
        createDef('d'),
      ]);

      const result = resolver.resolveFor('c');

      expect(result.success).toBe(true);
      expect(result.order).toContain('a');
      expect(result.order).toContain('b');
      expect(result.order).toContain('c');
      expect(result.order).not.toContain('d');
    });

    it('returns failure for non-existent module', () => {
      const result = resolver.resolveFor('non-existent');

      expect(result.success).toBe(false);
      expect(result.failed).toContain('non-existent');
    });

    it('detects circular in subgraph', () => {
      resolver.buildGraph([
        createDef('a', [{ name: 'c' }]),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
        createDef('d'),
      ]);

      const result = resolver.resolveFor('c');

      expect(result.success).toBe(false);
      expect(result.circular.length).toBeGreaterThan(0);
    });
  });

  describe('getDependencies', () => {
    it('returns direct dependencies', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b'),
        createDef('c', [{ name: 'a' }, { name: 'b' }]),
      ]);

      const deps = resolver.getDependencies('c');

      expect(deps).toContain('a');
      expect(deps).toContain('b');
      expect(deps.length).toBe(2);
    });

    it('returns empty array for no dependencies', () => {
      resolver.buildGraph([createDef('a')]);
      expect(resolver.getDependencies('a')).toEqual([]);
    });

    it('returns empty array for non-existent module', () => {
      expect(resolver.getDependencies('non-existent')).toEqual([]);
    });
  });

  describe('getTransitiveDependencies', () => {
    it('returns all transitive dependencies', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
      ]);

      const deps = resolver.getTransitiveDependencies('c');

      expect(deps).toContain('a');
      expect(deps).toContain('b');
      expect(deps).not.toContain('c');
    });
  });

  describe('getDependents', () => {
    it('returns direct dependents', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'a' }]),
      ]);

      const dependents = resolver.getDependents('a');

      expect(dependents).toContain('b');
      expect(dependents).toContain('c');
    });
  });

  describe('getTransitiveDependents', () => {
    it('returns all transitive dependents', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
      ]);

      const dependents = resolver.getTransitiveDependents('a');

      expect(dependents).toContain('b');
      expect(dependents).toContain('c');
    });
  });

  describe('hasDependencies', () => {
    it('returns true if module has dependencies', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
      ]);

      expect(resolver.hasDependencies('b')).toBe(true);
    });

    it('returns false if module has no dependencies', () => {
      resolver.buildGraph([createDef('a')]);
      expect(resolver.hasDependencies('a')).toBe(false);
    });
  });

  describe('hasDependents', () => {
    it('returns true if module has dependents', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
      ]);

      expect(resolver.hasDependents('a')).toBe(true);
    });

    it('returns false if module has no dependents', () => {
      resolver.buildGraph([createDef('a')]);
      expect(resolver.hasDependents('a')).toBe(false);
    });
  });

  describe('wouldCreateCycle', () => {
    it('returns true if edge would create cycle', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'b' }]),
      ]);

      expect(resolver.wouldCreateCycle('a', 'c')).toBe(true);
    });

    it('returns false if edge would not create cycle', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b'),
      ]);

      expect(resolver.wouldCreateCycle('a', 'b')).toBe(false);
    });
  });

  describe('getStats', () => {
    it('returns graph statistics', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
        createDef('c', [{ name: 'missing' }]),
      ]);

      resolver.resolve();
      const stats = resolver.getStats();

      expect(stats.nodeCount).toBe(3);
      expect(stats.edgeCount).toBe(1); // b->a only (c->missing doesn't count since missing doesn't exist)
      expect(stats.resolvedCount).toBe(2);
      expect(stats.failedCount).toBe(1);
    });
  });

  describe('clear', () => {
    it('clears the graph', () => {
      resolver.buildGraph([createDef('a'), createDef('b')]);
      resolver.clear();

      expect(resolver.getNode('a')).toBeUndefined();
      expect(resolver.getStats().nodeCount).toBe(0);
    });
  });

  describe('toDot', () => {
    it('generates DOT format output', () => {
      resolver.buildGraph([
        createDef('a'),
        createDef('b', [{ name: 'a' }]),
      ]);

      resolver.resolve();
      const dot = resolver.toDot();

      expect(dot).toContain('digraph Dependencies');
      expect(dot).toContain('"a"');
      expect(dot).toContain('"b"');
      expect(dot).toContain('"b" -> "a"');
    });

    it('includes status styling', () => {
      resolver.buildGraph([createDef('a')]);
      resolver.resolve();
      const dot = resolver.toDot();

      expect(dot).toContain('lightgreen'); // resolved
    });
  });
});
