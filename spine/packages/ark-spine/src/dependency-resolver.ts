/**
 * @ark/spine/dependency-resolver - Dependency Resolution for Module Loading
 *
 * Resolves module dependencies using topological sort with cycle detection.
 *
 * @module
 */

import type {
  ModuleDefinition,
  ModuleDependency,
  DependencyNode,
  DependencyStatus,
  ResolverConfig,
  ResolutionResult,
} from './types.js';

/**
 * Dependency Resolver - Resolves module load order
 */
export class DependencyResolver {
  private nodes: Map<string, DependencyNode> = new Map();
  private config: Required<ResolverConfig>;

  constructor(config: ResolverConfig = {}) {
    this.config = {
      maxDepth: config.maxDepth ?? 100,
      allowCircular: config.allowCircular ?? false,
      timeout: config.timeout ?? 30000, // 30 seconds default
    };
  }

  /**
   * Build dependency graph from module definitions
   */
  buildGraph(definitions: ModuleDefinition[]): void {
    this.nodes.clear();

    // First pass: create all nodes
    for (const def of definitions) {
      this.nodes.set(def.id, {
        moduleId: def.id,
        dependencies: [],
        dependents: [],
        status: 'unresolved',
      });
    }

    // Second pass: add edges
    for (const def of definitions) {
      const node = this.nodes.get(def.id)!;

      for (const dep of def.dependencies) {
        // Check if dependency exists
        if (this.nodes.has(dep.name)) {
          node.dependencies.push(dep.name);

          // Add reverse edge (dependent)
          const depNode = this.nodes.get(dep.name)!;
          depNode.dependents.push(def.id);
        } else if (!dep.optional) {
          // Non-optional dependency not found
          node.status = 'failed';
        }
      }
    }
  }

  /**
   * Add a single module to the graph
   */
  addModule(definition: ModuleDefinition): void {
    if (this.nodes.has(definition.id)) {
      return;
    }

    const node: DependencyNode = {
      moduleId: definition.id,
      dependencies: [],
      dependents: [],
      status: 'unresolved',
    };

    // Add dependencies
    for (const dep of definition.dependencies) {
      if (this.nodes.has(dep.name)) {
        node.dependencies.push(dep.name);

        // Add reverse edge
        const depNode = this.nodes.get(dep.name)!;
        depNode.dependents.push(definition.id);
      } else if (!dep.optional) {
        node.status = 'failed';
      }
    }

    this.nodes.set(definition.id, node);
  }

  /**
   * Remove a module from the graph
   */
  removeModule(moduleId: string): boolean {
    const node = this.nodes.get(moduleId);
    if (!node) {
      return false;
    }

    // Remove from dependents' dependency lists
    for (const depId of node.dependencies) {
      const depNode = this.nodes.get(depId);
      if (depNode) {
        depNode.dependents = depNode.dependents.filter((d) => d !== moduleId);
      }
    }

    // Remove from dependencies' dependent lists
    for (const dependentId of node.dependents) {
      const dependentNode = this.nodes.get(dependentId);
      if (dependentNode) {
        dependentNode.dependencies = dependentNode.dependencies.filter(
          (d) => d !== moduleId
        );
      }
    }

    this.nodes.delete(moduleId);
    return true;
  }

  /**
   * Resolve dependency order using topological sort (Kahn's algorithm)
   */
  resolve(): ResolutionResult {
    const startTime = Date.now();
    const order: string[] = [];
    const failed: string[] = [];
    const circular: string[][] = [];

    // Reset status
    for (const node of this.nodes.values()) {
      if (node.status !== 'failed') {
        node.status = 'unresolved';
      }
    }

    // Calculate in-degree for each node
    const inDegree = new Map<string, number>();
    for (const [id, node] of this.nodes) {
      if (node.status === 'failed') {
        failed.push(id);
        continue;
      }
      inDegree.set(id, node.dependencies.length);
    }

    // Start with nodes that have no dependencies
    const queue: string[] = [];
    for (const [id, degree] of inDegree) {
      if (degree === 0) {
        queue.push(id);
      }
    }

    let orderIndex = 0;

    while (queue.length > 0) {
      const moduleId = queue.shift()!;
      order.push(moduleId);

      const node = this.nodes.get(moduleId)!;
      node.status = 'resolved';
      node.order = orderIndex++;

      // Reduce in-degree of dependents
      for (const dependentId of node.dependents) {
        const degree = inDegree.get(dependentId);
        if (degree === undefined) continue;

        const newDegree = degree - 1;
        inDegree.set(dependentId, newDegree);

        if (newDegree === 0) {
          queue.push(dependentId);
        }
      }
    }

    // Check for cycles (nodes with remaining in-degree > 0)
    const unresolved: string[] = [];
    for (const [id, degree] of inDegree) {
      if (degree > 0) {
        unresolved.push(id);
        const node = this.nodes.get(id)!;
        node.status = 'circular';
      }
    }

    // Find actual cycles if there are unresolved nodes
    if (unresolved.length > 0) {
      const cycles = this.detectCycles(unresolved);
      circular.push(...cycles);

      if (!this.config.allowCircular) {
        failed.push(...unresolved);
      } else {
        // If circular deps allowed, add them in arbitrary order
        order.push(...unresolved);
        for (const id of unresolved) {
          const node = this.nodes.get(id);
          if (node) {
            node.order = orderIndex++;
          }
        }
      }
    }

    return {
      success: failed.length === 0 && (this.config.allowCircular || circular.length === 0),
      order,
      failed,
      circular,
      duration: Date.now() - startTime,
    };
  }

  /**
   * Resolve dependencies for a specific module
   */
  resolveFor(moduleId: string): ResolutionResult {
    const startTime = Date.now();

    if (!this.nodes.has(moduleId)) {
      return {
        success: false,
        order: [],
        failed: [moduleId],
        circular: [],
        duration: Date.now() - startTime,
      };
    }

    // Get all transitive dependencies
    const visited = new Set<string>();
    const order: string[] = [];
    const failed: string[] = [];
    const circular: string[][] = [];
    const inStack = new Set<string>();

    const visit = (id: string, path: string[]): boolean => {
      if (visited.has(id)) {
        return true;
      }

      if (inStack.has(id)) {
        // Cycle detected
        const cycleStart = path.indexOf(id);
        const cycle = [...path.slice(cycleStart), id];
        circular.push(cycle);

        if (!this.config.allowCircular) {
          return false;
        }
        return true;
      }

      const node = this.nodes.get(id);
      if (!node) {
        failed.push(id);
        return false;
      }

      if (node.status === 'failed') {
        failed.push(id);
        return false;
      }

      inStack.add(id);

      // Visit dependencies first
      for (const depId of node.dependencies) {
        if (!visit(depId, [...path, id])) {
          if (!this.config.allowCircular) {
            return false;
          }
        }
      }

      inStack.delete(id);
      visited.add(id);
      order.push(id);

      return true;
    };

    visit(moduleId, []);

    return {
      success: failed.length === 0 && (this.config.allowCircular || circular.length === 0),
      order,
      failed,
      circular,
      duration: Date.now() - startTime,
    };
  }

  /**
   * Get dependencies of a module (direct only)
   */
  getDependencies(moduleId: string): string[] {
    const node = this.nodes.get(moduleId);
    return node ? [...node.dependencies] : [];
  }

  /**
   * Get all transitive dependencies of a module
   */
  getTransitiveDependencies(moduleId: string): string[] {
    const result = this.resolveFor(moduleId);
    // Remove the module itself from the order
    return result.order.filter((id) => id !== moduleId);
  }

  /**
   * Get dependents of a module (direct only)
   */
  getDependents(moduleId: string): string[] {
    const node = this.nodes.get(moduleId);
    return node ? [...node.dependents] : [];
  }

  /**
   * Get all transitive dependents of a module
   */
  getTransitiveDependents(moduleId: string): string[] {
    const dependents: string[] = [];
    const visited = new Set<string>();

    const collect = (id: string) => {
      const node = this.nodes.get(id);
      if (!node) return;

      for (const depId of node.dependents) {
        if (!visited.has(depId)) {
          visited.add(depId);
          dependents.push(depId);
          collect(depId);
        }
      }
    };

    collect(moduleId);
    return dependents;
  }

  /**
   * Get node information
   */
  getNode(moduleId: string): DependencyNode | undefined {
    return this.nodes.get(moduleId);
  }

  /**
   * Check if a module has any dependencies
   */
  hasDependencies(moduleId: string): boolean {
    const node = this.nodes.get(moduleId);
    return node ? node.dependencies.length > 0 : false;
  }

  /**
   * Check if a module has any dependents
   */
  hasDependents(moduleId: string): boolean {
    const node = this.nodes.get(moduleId);
    return node ? node.dependents.length > 0 : false;
  }

  /**
   * Check if adding a dependency would create a cycle
   */
  wouldCreateCycle(from: string, to: string): boolean {
    // If 'to' can reach 'from', adding from->to would create a cycle
    const transitiveDeps = this.getTransitiveDependencies(to);
    return transitiveDeps.includes(from);
  }

  /**
   * Detect cycles in a set of nodes using Tarjan's algorithm
   */
  private detectCycles(nodeIds: string[]): string[][] {
    const cycles: string[][] = [];
    const visited = new Set<string>();
    const recStack = new Set<string>();

    const findCyclesFrom = (id: string, path: string[]): void => {
      if (recStack.has(id)) {
        // Found a cycle
        const cycleStart = path.indexOf(id);
        if (cycleStart >= 0) {
          cycles.push([...path.slice(cycleStart), id]);
        }
        return;
      }

      if (visited.has(id)) {
        return;
      }

      visited.add(id);
      recStack.add(id);

      const node = this.nodes.get(id);
      if (node) {
        for (const depId of node.dependencies) {
          if (nodeIds.includes(depId)) {
            findCyclesFrom(depId, [...path, id]);
          }
        }
      }

      recStack.delete(id);
    };

    for (const id of nodeIds) {
      if (!visited.has(id)) {
        findCyclesFrom(id, []);
      }
    }

    // Deduplicate cycles (they might be found from different starting points)
    const uniqueCycles: string[][] = [];
    const seen = new Set<string>();

    for (const cycle of cycles) {
      // Normalize cycle by finding canonical representation
      const normalized = this.normalizeCycle(cycle);
      const key = normalized.join(',');
      if (!seen.has(key)) {
        seen.add(key);
        uniqueCycles.push(normalized);
      }
    }

    return uniqueCycles;
  }

  /**
   * Normalize a cycle to canonical form (start with smallest element)
   */
  private normalizeCycle(cycle: string[]): string[] {
    if (cycle.length <= 1) return cycle;

    // Remove duplicate ending element if present
    const cleanCycle =
      cycle[0] === cycle[cycle.length - 1] ? cycle.slice(0, -1) : cycle;

    // Find index of smallest element
    let minIndex = 0;
    for (let i = 1; i < cleanCycle.length; i++) {
      if (cleanCycle[i] < cleanCycle[minIndex]) {
        minIndex = i;
      }
    }

    // Rotate to start with smallest
    return [...cleanCycle.slice(minIndex), ...cleanCycle.slice(0, minIndex)];
  }

  /**
   * Get graph statistics
   */
  getStats(): {
    nodeCount: number;
    edgeCount: number;
    resolvedCount: number;
    failedCount: number;
    circularCount: number;
  } {
    let edgeCount = 0;
    let resolvedCount = 0;
    let failedCount = 0;
    let circularCount = 0;

    for (const node of this.nodes.values()) {
      edgeCount += node.dependencies.length;

      switch (node.status) {
        case 'resolved':
          resolvedCount++;
          break;
        case 'failed':
          failedCount++;
          break;
        case 'circular':
          circularCount++;
          break;
      }
    }

    return {
      nodeCount: this.nodes.size,
      edgeCount,
      resolvedCount,
      failedCount,
      circularCount,
    };
  }

  /**
   * Clear the graph
   */
  clear(): void {
    this.nodes.clear();
  }

  /**
   * Export graph as DOT format (for visualization)
   */
  toDot(): string {
    const lines: string[] = ['digraph Dependencies {'];

    for (const [id, node] of this.nodes) {
      // Node styling based on status
      let style = '';
      switch (node.status) {
        case 'resolved':
          style = 'style=filled, fillcolor=lightgreen';
          break;
        case 'failed':
          style = 'style=filled, fillcolor=lightcoral';
          break;
        case 'circular':
          style = 'style=filled, fillcolor=lightyellow';
          break;
        default:
          style = 'style=filled, fillcolor=lightgray';
      }

      lines.push(`  "${id}" [${style}];`);

      // Edges
      for (const depId of node.dependencies) {
        lines.push(`  "${id}" -> "${depId}";`);
      }
    }

    lines.push('}');
    return lines.join('\n');
  }
}

/**
 * Create a new dependency resolver
 */
export function createDependencyResolver(
  config?: ResolverConfig
): DependencyResolver {
  return new DependencyResolver(config);
}
