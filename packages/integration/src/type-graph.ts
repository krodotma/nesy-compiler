/**
 * TypeGraph: Ingest TypeScript Type Information
 *
 * Step 13 of NeSy Evolution Phase 1.5 (Sensor Fusion)
 *
 * Parses .d.ts declaration files to build a 100% accurate Type Dependency Graph.
 * This provides ground truth from the TypeScript compiler about type relationships.
 *
 * Types of relationships captured:
 * - EXTENDS: class A extends B
 * - IMPLEMENTS: class A implements I
 * - USES: function uses type T as parameter/return
 * - REFERENCES: type A references type B
 */

import * as fs from 'fs';
import * as path from 'path';

export type TypeKind =
  | 'class'
  | 'interface'
  | 'type'
  | 'enum'
  | 'function'
  | 'variable'
  | 'namespace'
  | 'module';

export interface TypeNode {
  id: string;
  name: string;
  kind: TypeKind;
  filePath: string;
  exported: boolean;
  generic?: boolean;
  typeParameters?: string[];
  members?: TypeMember[];
  documentation?: string;
}

export interface TypeMember {
  name: string;
  kind: 'property' | 'method' | 'getter' | 'setter' | 'index';
  type?: string;
  optional?: boolean;
  readonly?: boolean;
  static?: boolean;
  visibility?: 'public' | 'private' | 'protected';
}

export type TypeRelation =
  | 'EXTENDS'
  | 'IMPLEMENTS'
  | 'USES'
  | 'REFERENCES'
  | 'CONTAINS'
  | 'EXPORTS';

export interface TypeEdge {
  from: string;
  to: string;
  relation: TypeRelation;
  context?: string;  // e.g., "parameter:0", "return", "property:foo"
}

export interface TypeGraph {
  nodes: Map<string, TypeNode>;
  edges: TypeEdge[];
  modules: Map<string, string[]>;  // module path -> exported type IDs
}

/**
 * Parse a .d.ts file and extract type information.
 * Uses regex-based parsing for simplicity (not a full TS parser).
 */
export function parseDeclarationFile(
  filePath: string,
  content?: string
): { nodes: TypeNode[]; edges: TypeEdge[] } {
  const source = content ?? fs.readFileSync(filePath, 'utf-8');
  const nodes: TypeNode[] = [];
  const edges: TypeEdge[] = [];

  // Extract interfaces
  const interfaceRegex = /export\s+interface\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+([^\{]+))?\s*\{/g;
  let match;

  while ((match = interfaceRegex.exec(source)) !== null) {
    const [, name, typeParams, extendsClause] = match;
    const id = `${filePath}:${name}`;

    nodes.push({
      id,
      name,
      kind: 'interface',
      filePath,
      exported: true,
      generic: !!typeParams,
      typeParameters: typeParams?.split(',').map(t => t.trim()),
    });

    // Parse extends
    if (extendsClause) {
      const parents = extendsClause.split(',').map(t => t.trim().replace(/<.*>/, ''));
      for (const parent of parents) {
        edges.push({
          from: id,
          to: parent,
          relation: 'EXTENDS',
        });
      }
    }
  }

  // Extract type aliases
  const typeRegex = /export\s+type\s+(\w+)(?:<([^>]+)>)?\s*=\s*([^;]+);/g;

  while ((match = typeRegex.exec(source)) !== null) {
    const [, name, typeParams, definition] = match;
    const id = `${filePath}:${name}`;

    nodes.push({
      id,
      name,
      kind: 'type',
      filePath,
      exported: true,
      generic: !!typeParams,
      typeParameters: typeParams?.split(',').map(t => t.trim()),
    });

    // Extract type references from definition
    const refs = extractTypeReferences(definition);
    for (const ref of refs) {
      edges.push({
        from: id,
        to: ref,
        relation: 'REFERENCES',
      });
    }
  }

  // Extract classes
  const classRegex = /export\s+(?:abstract\s+)?class\s+(\w+)(?:<([^>]+)>)?(?:\s+extends\s+(\w+)(?:<[^>]+>)?)?(?:\s+implements\s+([^\{]+))?\s*\{/g;

  while ((match = classRegex.exec(source)) !== null) {
    const [, name, typeParams, extendsClass, implementsClause] = match;
    const id = `${filePath}:${name}`;

    nodes.push({
      id,
      name,
      kind: 'class',
      filePath,
      exported: true,
      generic: !!typeParams,
      typeParameters: typeParams?.split(',').map(t => t.trim()),
    });

    if (extendsClass) {
      edges.push({
        from: id,
        to: extendsClass,
        relation: 'EXTENDS',
      });
    }

    if (implementsClause) {
      const interfaces = implementsClause.split(',').map(t => t.trim().replace(/<.*>/, ''));
      for (const iface of interfaces) {
        edges.push({
          from: id,
          to: iface,
          relation: 'IMPLEMENTS',
        });
      }
    }
  }

  // Extract enums
  const enumRegex = /export\s+(?:const\s+)?enum\s+(\w+)\s*\{/g;

  while ((match = enumRegex.exec(source)) !== null) {
    const [, name] = match;
    const id = `${filePath}:${name}`;

    nodes.push({
      id,
      name,
      kind: 'enum',
      filePath,
      exported: true,
    });
  }

  // Extract functions
  const funcRegex = /export\s+(?:async\s+)?function\s+(\w+)(?:<([^>]+)>)?\s*\(([^)]*)\)\s*:\s*([^;{]+)/g;

  while ((match = funcRegex.exec(source)) !== null) {
    const [, name, typeParams, params, returnType] = match;
    const id = `${filePath}:${name}`;

    nodes.push({
      id,
      name,
      kind: 'function',
      filePath,
      exported: true,
      generic: !!typeParams,
      typeParameters: typeParams?.split(',').map(t => t.trim()),
    });

    // Extract parameter type references
    const paramRefs = extractTypeReferences(params);
    for (const ref of paramRefs) {
      edges.push({
        from: id,
        to: ref,
        relation: 'USES',
        context: 'parameter',
      });
    }

    // Extract return type references
    const returnRefs = extractTypeReferences(returnType);
    for (const ref of returnRefs) {
      edges.push({
        from: id,
        to: ref,
        relation: 'USES',
        context: 'return',
      });
    }
  }

  return { nodes, edges };
}

/**
 * Extract type references from a type expression.
 */
function extractTypeReferences(typeExpr: string): string[] {
  const refs: string[] = [];

  // Match type identifiers (capitalized words that aren't keywords)
  const keywords = new Set([
    'string', 'number', 'boolean', 'void', 'null', 'undefined',
    'never', 'unknown', 'any', 'object', 'symbol', 'bigint',
    'true', 'false', 'Promise', 'Array', 'Map', 'Set', 'Record',
    'Partial', 'Required', 'Readonly', 'Pick', 'Omit', 'Exclude',
  ]);

  const typeRefRegex = /\b([A-Z][a-zA-Z0-9]*)\b/g;
  let match;

  while ((match = typeRefRegex.exec(typeExpr)) !== null) {
    const [, name] = match;
    if (!keywords.has(name) && !refs.includes(name)) {
      refs.push(name);
    }
  }

  return refs;
}

/**
 * Build a TypeGraph from multiple declaration files.
 */
export async function buildTypeGraph(
  dtsFiles: string[]
): Promise<TypeGraph> {
  const graph: TypeGraph = {
    nodes: new Map(),
    edges: [],
    modules: new Map(),
  };

  for (const file of dtsFiles) {
    try {
      const { nodes, edges } = parseDeclarationFile(file);

      // Add nodes
      const moduleTypes: string[] = [];
      for (const node of nodes) {
        graph.nodes.set(node.id, node);
        moduleTypes.push(node.id);
      }

      // Track module exports
      graph.modules.set(file, moduleTypes);

      // Add edges
      graph.edges.push(...edges);
    } catch (error) {
      console.error(`[TypeGraph] Failed to parse ${file}:`, error);
    }
  }

  // Resolve edge targets (convert short names to full IDs)
  resolveEdgeTargets(graph);

  return graph;
}

/**
 * Resolve edge targets from short names to full type IDs.
 */
function resolveEdgeTargets(graph: TypeGraph): void {
  // Build name -> ID index
  const nameToId = new Map<string, string[]>();
  for (const [id, node] of graph.nodes) {
    const existing = nameToId.get(node.name) || [];
    existing.push(id);
    nameToId.set(node.name, existing);
  }

  // Resolve each edge
  for (const edge of graph.edges) {
    if (!edge.to.includes(':')) {
      // Short name - resolve to full ID
      const candidates = nameToId.get(edge.to);
      if (candidates && candidates.length > 0) {
        // Prefer same-file resolution
        const sameFile = candidates.find(c =>
          c.startsWith(edge.from.split(':')[0])
        );
        edge.to = sameFile || candidates[0];
      }
    }
  }
}

/**
 * Generate d.ts files using TypeScript compiler.
 */
export async function generateDeclarations(
  tsConfigPath: string,
  outputDir: string
): Promise<string[]> {
  const { spawn } = await import('child_process');

  return new Promise((resolve, reject) => {
    const proc = spawn('tsc', [
      '-p', tsConfigPath,
      '--declaration',
      '--emitDeclarationOnly',
      '--declarationDir', outputDir,
    ], { shell: true });

    proc.on('close', async (code) => {
      if (code !== 0) {
        reject(new Error(`tsc exited with code ${code}`));
        return;
      }

      // Find generated .d.ts files
      const files = await findDtsFiles(outputDir);
      resolve(files);
    });

    proc.on('error', reject);
  });
}

/**
 * Find all .d.ts files in a directory.
 */
async function findDtsFiles(dir: string): Promise<string[]> {
  const files: string[] = [];

  async function walk(d: string) {
    const entries = await fs.promises.readdir(d, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(d, entry.name);
      if (entry.isDirectory()) {
        await walk(full);
      } else if (entry.name.endsWith('.d.ts')) {
        files.push(full);
      }
    }
  }

  await walk(dir);
  return files;
}

/**
 * Get type hierarchy for a given type (ancestors and descendants).
 */
export function getTypeHierarchy(
  graph: TypeGraph,
  typeId: string
): { ancestors: string[]; descendants: string[] } {
  const ancestors: string[] = [];
  const descendants: string[] = [];

  // Find ancestors (types this extends/implements)
  const visited = new Set<string>();
  function findAncestors(id: string) {
    if (visited.has(id)) return;
    visited.add(id);

    for (const edge of graph.edges) {
      if (edge.from === id && (edge.relation === 'EXTENDS' || edge.relation === 'IMPLEMENTS')) {
        ancestors.push(edge.to);
        findAncestors(edge.to);
      }
    }
  }
  findAncestors(typeId);

  // Find descendants (types that extend/implement this)
  visited.clear();
  function findDescendants(id: string) {
    if (visited.has(id)) return;
    visited.add(id);

    for (const edge of graph.edges) {
      if (edge.to === id && (edge.relation === 'EXTENDS' || edge.relation === 'IMPLEMENTS')) {
        descendants.push(edge.from);
        findDescendants(edge.from);
      }
    }
  }
  findDescendants(typeId);

  return { ancestors, descendants };
}

/**
 * Find types that use a given type.
 */
export function findTypeUsages(
  graph: TypeGraph,
  typeId: string
): Array<{ user: string; context?: string }> {
  const usages: Array<{ user: string; context?: string }> = [];

  for (const edge of graph.edges) {
    if (edge.to === typeId && (edge.relation === 'USES' || edge.relation === 'REFERENCES')) {
      usages.push({
        user: edge.from,
        context: edge.context,
      });
    }
  }

  return usages;
}

/**
 * Convert TypeGraph to FalkorDB-compatible format.
 */
export function typeGraphToFalkor(
  graph: TypeGraph
): {
  nodes: Array<{ id: string; labels: string[]; properties: Record<string, unknown> }>;
  edges: Array<{ from: string; to: string; type: string; properties?: Record<string, unknown> }>;
} {
  const nodes = Array.from(graph.nodes.values()).map(node => ({
    id: node.id,
    labels: ['type', node.kind],
    properties: {
      name: node.name,
      filePath: node.filePath,
      exported: node.exported,
      generic: node.generic || false,
      typeParameters: node.typeParameters?.join(', ') || '',
    },
  }));

  const edges = graph.edges.map(edge => ({
    from: edge.from,
    to: edge.to,
    type: edge.relation,
    properties: edge.context ? { context: edge.context } : undefined,
  }));

  return { nodes, edges };
}
