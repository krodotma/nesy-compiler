/**
 * CodeGraphParser - Git Diff to Code Graph Extraction
 *
 * Step 3 & 16-17: Code Graph Extraction & Performance Optimization
 *
 * Features:
 * - Parse `git diff --stat` output
 * - Extract file paths and line counts
 * - Build CodeNode[] with visual weights
 * - Detect import relationships (basic heuristic)
 * - Web Worker compatible for off-main-thread processing
 */

import type { CodeGraph, CodeNode } from './types';

// =============================================================================
// Types
// =============================================================================

export interface DiffStatEntry {
  /** File path */
  path: string;
  /** Lines added */
  additions: number;
  /** Lines removed */
  deletions: number;
  /** Binary file indicator */
  binary: boolean;
}

export interface ParseOptions {
  /** Include binary files in graph (default: false) */
  includeBinary?: boolean;
  /** Minimum lines changed to include (default: 1) */
  minLinesChanged?: number;
  /** Maximum nodes in graph (default: 100) */
  maxNodes?: number;
  /** Detect dependencies via imports (default: true) */
  detectDependencies?: boolean;
  /** Root path for graph (default: '/pluribus') */
  rootPath?: string;
}

// =============================================================================
// Diff Stat Parser
// =============================================================================

/**
 * Parse `git diff --stat` output into structured entries
 *
 * Example input:
 * ```
 *  nucleus/tools/agent_bus.py | 25 ++++++++++++-
 *  dashboard/src/App.tsx      | 10 ++---
 *  2 files changed, 30 insertions(+), 5 deletions(-)
 * ```
 */
export function parseDiffStat(diffStatOutput: string): DiffStatEntry[] {
  const entries: DiffStatEntry[] = [];
  const lines = diffStatOutput.split('\n');

  for (const line of lines) {
    // Skip summary line
    if (line.includes('files changed') || line.includes('file changed')) {
      continue;
    }

    // Skip empty lines
    if (!line.trim()) {
      continue;
    }

    // Parse file stat line: " path/to/file | N +++ --"
    const match = line.match(/^\s*(.+?)\s*\|\s*(\d+|Bin)/);
    if (!match) continue;

    const path = match[1].trim();
    const changeStr = match[2];

    if (changeStr === 'Bin') {
      entries.push({
        path,
        additions: 0,
        deletions: 0,
        binary: true,
      });
      continue;
    }

    // Count + and - in the bar
    const barMatch = line.match(/\|\s*\d+\s*([+-]+)/);
    let additions = 0;
    let deletions = 0;

    if (barMatch) {
      const bar = barMatch[1];
      additions = (bar.match(/\+/g) || []).length;
      deletions = (bar.match(/-/g) || []).length;
    } else {
      // If no bar, assume all changes are additions (for simple diffs)
      additions = parseInt(changeStr, 10) || 0;
    }

    entries.push({
      path,
      additions,
      deletions,
      binary: false,
    });
  }

  return entries;
}

/**
 * Parse `git diff --numstat` output (more precise)
 *
 * Example input:
 * ```
 * 25   5   nucleus/tools/agent_bus.py
 * 10   3   dashboard/src/App.tsx
 * -    -   binary/file.png
 * ```
 */
export function parseNumstat(numstatOutput: string): DiffStatEntry[] {
  const entries: DiffStatEntry[] = [];
  const lines = numstatOutput.split('\n');

  for (const line of lines) {
    if (!line.trim()) continue;

    const parts = line.split('\t');
    if (parts.length < 3) continue;

    const [addStr, delStr, path] = parts;

    if (addStr === '-' && delStr === '-') {
      entries.push({
        path: path.trim(),
        additions: 0,
        deletions: 0,
        binary: true,
      });
    } else {
      entries.push({
        path: path.trim(),
        additions: parseInt(addStr, 10) || 0,
        deletions: parseInt(delStr, 10) || 0,
        binary: false,
      });
    }
  }

  return entries;
}

// =============================================================================
// Import Detection
// =============================================================================

/**
 * Detect import relationships between files (basic heuristics)
 *
 * This is a simplified approach that:
 * 1. Groups files by directory
 * 2. Links index files to siblings
 * 3. Links components to types/utils
 */
export function detectDependencies(nodes: CodeNode[]): [string, string][] {
  const edges: [string, string][] = [];
  const pathSet = new Set(nodes.map((n) => n.path));

  // Group by directory
  const byDir = new Map<string, CodeNode[]>();
  for (const node of nodes) {
    const dir = node.path.substring(0, node.path.lastIndexOf('/') || 0);
    if (!byDir.has(dir)) {
      byDir.set(dir, []);
    }
    byDir.get(dir)!.push(node);
  }

  for (const node of nodes) {
    const dir = node.path.substring(0, node.path.lastIndexOf('/') || 0);
    const filename = node.path.substring(node.path.lastIndexOf('/') + 1);
    const basename = filename.replace(/\.[^.]+$/, '');

    // Pattern 1: index files import siblings
    if (filename.startsWith('index.')) {
      const siblings = byDir.get(dir) || [];
      for (const sibling of siblings) {
        if (sibling.path !== node.path) {
          edges.push([node.path, sibling.path]);
        }
      }
    }

    // Pattern 2: Components import types
    if (node.node_type === 'file' && (filename.endsWith('.tsx') || filename.endsWith('.ts'))) {
      // Look for types.ts in same directory
      const typesPath = `${dir}/types.ts`;
      if (pathSet.has(typesPath) && typesPath !== node.path) {
        edges.push([node.path, typesPath]);
      }

      // Look for utils in same or parent directory
      const utilsPatterns = ['utils.ts', 'util.ts', 'helpers.ts'];
      for (const pattern of utilsPatterns) {
        const utilPath = `${dir}/${pattern}`;
        if (pathSet.has(utilPath)) {
          edges.push([node.path, utilPath]);
        }
      }
    }

    // Pattern 3: Test files import source
    if (filename.includes('.test.') || filename.includes('.spec.')) {
      const sourceFile = filename
        .replace('.test.', '.')
        .replace('.spec.', '.');
      const sourcePath = `${dir}/${sourceFile}`;
      if (pathSet.has(sourcePath)) {
        edges.push([node.path, sourcePath]);
      }
    }

    // Pattern 4: Infer from file naming conventions
    // e.g., ChromaticVisualizer.tsx imports types.ts, utils/colorMap.ts
    if (basename.includes('Visualizer') || basename.includes('Component')) {
      // Look for related utils
      const utilsDir = `${dir}/utils`;
      const siblings = byDir.get(utilsDir) || [];
      for (const util of siblings) {
        edges.push([node.path, util.path]);
      }
    }
  }

  // Deduplicate edges
  const seen = new Set<string>();
  return edges.filter(([from, to]) => {
    const key = `${from}:${to}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// =============================================================================
// Node Type Detection
// =============================================================================

/**
 * Infer node type from file path
 */
export function inferNodeType(path: string): 'file' | 'module' | 'directory' {
  const filename = path.substring(path.lastIndexOf('/') + 1);

  // Directory indicators
  if (path.endsWith('/') || !filename.includes('.')) {
    return 'directory';
  }

  // Module indicators (entry points)
  if (
    filename === 'index.ts' ||
    filename === 'index.tsx' ||
    filename === 'index.js' ||
    filename === 'mod.ts' ||
    filename === '__init__.py'
  ) {
    return 'module';
  }

  return 'file';
}

/**
 * Calculate visual weight from lines changed
 * Uses sqrt for sub-linear scaling (prevents huge nodes)
 */
export function calculateVisualWeight(linesChanged: number): number {
  return Math.sqrt(Math.max(1, linesChanged));
}

// =============================================================================
// Code Graph Builder
// =============================================================================

/**
 * Build a CodeGraph from diff stat entries
 */
export function buildCodeGraph(
  entries: DiffStatEntry[],
  options: ParseOptions = {}
): CodeGraph {
  const {
    includeBinary = false,
    minLinesChanged = 1,
    maxNodes = 100,
    detectDependencies: shouldDetectDeps = true,
    rootPath = '/pluribus',
  } = options;

  const now = new Date().toISOString();

  // Filter and sort entries
  let filtered = entries
    .filter((e) => includeBinary || !e.binary)
    .filter((e) => e.additions + e.deletions >= minLinesChanged);

  // Sort by total changes descending
  filtered.sort((a, b) => {
    const totalA = a.additions + a.deletions;
    const totalB = b.additions + b.deletions;
    return totalB - totalA;
  });

  // Limit to maxNodes
  filtered = filtered.slice(0, maxNodes);

  // Build nodes
  const nodes: CodeNode[] = filtered.map((entry) => ({
    path: entry.path,
    lines_changed: entry.additions + entry.deletions,
    node_type: inferNodeType(entry.path),
    dependencies: [],
    last_modified_iso: now,
    visual_weight: calculateVisualWeight(entry.additions + entry.deletions),
  }));

  // Detect dependencies
  let edges: [string, string][] = [];
  if (shouldDetectDeps && nodes.length > 0) {
    edges = detectDependencies(nodes);

    // Update node dependencies
    const depMap = new Map<string, string[]>();
    for (const [from, to] of edges) {
      if (!depMap.has(from)) {
        depMap.set(from, []);
      }
      depMap.get(from)!.push(to);
    }

    for (const node of nodes) {
      node.dependencies = depMap.get(node.path) || [];
    }
  }

  return {
    root: rootPath,
    nodes,
    edges,
    timestamp_iso: now,
  };
}

/**
 * Parse diff output and build code graph
 */
export function parseAndBuildGraph(
  diffOutput: string,
  options: ParseOptions = {}
): CodeGraph {
  // Detect format and parse
  let entries: DiffStatEntry[];

  if (diffOutput.includes('\t')) {
    // Likely numstat format
    entries = parseNumstat(diffOutput);
  } else {
    // Standard --stat format
    entries = parseDiffStat(diffOutput);
  }

  return buildCodeGraph(entries, options);
}

// =============================================================================
// Graph Manipulation
// =============================================================================

/**
 * Merge two code graphs (for incremental updates)
 */
export function mergeGraphs(
  existing: CodeGraph,
  incoming: CodeGraph,
  maxAge: number = 300000 // 5 minutes
): CodeGraph {
  const now = Date.now();
  const nodeMap = new Map<string, CodeNode>();

  // Add existing nodes (filter out old ones)
  for (const node of existing.nodes) {
    const nodeTime = new Date(node.last_modified_iso).getTime();
    if (now - nodeTime < maxAge) {
      nodeMap.set(node.path, node);
    }
  }

  // Add/update incoming nodes
  for (const node of incoming.nodes) {
    const existing = nodeMap.get(node.path);
    if (existing) {
      // Merge: keep higher line count, update timestamp
      nodeMap.set(node.path, {
        ...existing,
        lines_changed: Math.max(existing.lines_changed, node.lines_changed),
        visual_weight: Math.max(existing.visual_weight, node.visual_weight),
        last_modified_iso: node.last_modified_iso,
        dependencies: [...new Set([...existing.dependencies, ...node.dependencies])],
      });
    } else {
      nodeMap.set(node.path, node);
    }
  }

  const nodes = Array.from(nodeMap.values());

  // Merge edges
  const edgeSet = new Set<string>();
  for (const [from, to] of existing.edges) {
    if (nodeMap.has(from) && nodeMap.has(to)) {
      edgeSet.add(`${from}:${to}`);
    }
  }
  for (const [from, to] of incoming.edges) {
    edgeSet.add(`${from}:${to}`);
  }

  const edges: [string, string][] = Array.from(edgeSet).map((key) => {
    const [from, to] = key.split(':');
    return [from, to];
  });

  return {
    root: incoming.root || existing.root,
    nodes,
    edges,
    timestamp_iso: new Date().toISOString(),
  };
}

/**
 * Calculate graph statistics
 */
export function getGraphStats(graph: CodeGraph): {
  nodeCount: number;
  edgeCount: number;
  totalLinesChanged: number;
  avgWeight: number;
  maxWeight: number;
  topFiles: string[];
} {
  const totalLines = graph.nodes.reduce((sum, n) => sum + n.lines_changed, 0);
  const avgWeight = graph.nodes.length > 0
    ? graph.nodes.reduce((sum, n) => sum + n.visual_weight, 0) / graph.nodes.length
    : 0;
  const maxWeight = graph.nodes.length > 0
    ? Math.max(...graph.nodes.map((n) => n.visual_weight))
    : 0;

  // Top 5 files by lines changed
  const topFiles = [...graph.nodes]
    .sort((a, b) => b.lines_changed - a.lines_changed)
    .slice(0, 5)
    .map((n) => n.path);

  return {
    nodeCount: graph.nodes.length,
    edgeCount: graph.edges.length,
    totalLinesChanged: totalLines,
    avgWeight,
    maxWeight,
    topFiles,
  };
}

// =============================================================================
// Web Worker Support
// =============================================================================

/**
 * Message types for Web Worker communication
 */
export interface WorkerMessage {
  type: 'parse' | 'merge' | 'stats';
  id: string;
  payload: unknown;
}

export interface WorkerResponse {
  type: 'result' | 'error';
  id: string;
  payload: unknown;
}

/**
 * Handle worker message (for use in Web Worker)
 */
export function handleWorkerMessage(message: WorkerMessage): WorkerResponse {
  try {
    switch (message.type) {
      case 'parse': {
        const { diffOutput, options } = message.payload as {
          diffOutput: string;
          options?: ParseOptions;
        };
        const graph = parseAndBuildGraph(diffOutput, options);
        return { type: 'result', id: message.id, payload: graph };
      }
      case 'merge': {
        const { existing, incoming, maxAge } = message.payload as {
          existing: CodeGraph;
          incoming: CodeGraph;
          maxAge?: number;
        };
        const merged = mergeGraphs(existing, incoming, maxAge);
        return { type: 'result', id: message.id, payload: merged };
      }
      case 'stats': {
        const graph = message.payload as CodeGraph;
        const stats = getGraphStats(graph);
        return { type: 'result', id: message.id, payload: stats };
      }
      default:
        return {
          type: 'error',
          id: message.id,
          payload: `Unknown message type: ${message.type}`,
        };
    }
  } catch (err) {
    return {
      type: 'error',
      id: message.id,
      payload: err instanceof Error ? err.message : String(err),
    };
  }
}

export default {
  parseDiffStat,
  parseNumstat,
  buildCodeGraph,
  parseAndBuildGraph,
  mergeGraphs,
  getGraphStats,
  detectDependencies,
  handleWorkerMessage,
};
