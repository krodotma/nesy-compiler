/**
 * BranchingVisualization - Delineation Branching Visuals
 *
 * Visualizes decision branch points in the Pluribus workflow:
 * - Decision branch points (where choices were made)
 * - Possible futures (potential paths - dashed)
 * - Selected paths (actualized - solid)
 * - Counterfactual paths (not taken - faded)
 *
 * Integrates with the Holon Pentad (where/when context) and
 * Entelechesis phases (potential -> actualizing -> actualized -> decaying).
 *
 * Architecture:
 * - DAG layout with collapsible branches
 * - SVG rendering for crisp edges
 * - Path highlighting on hover
 * - Phase-based coloring (Aristotelian entelechesis)
 */

import {
  component$,
  useSignal,
  useComputed$,
  useStore,
  $,
  type QRL,
} from '@builder.io/qwik';
import type { EntelexisPhase, HysteresisTrace } from '../lib/state/types';

// M3 Components - BranchingVisualization
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';

// =============================================================================
// Types
// =============================================================================

/** Node states in the decision DAG */
export type BranchNodeState =
  | 'potential'    // Future possibility (not yet reached)
  | 'actualizing'  // Currently being evaluated
  | 'actualized'   // Path was taken (selected)
  | 'counterfactual' // Path not taken
  | 'decaying';    // Obsolete branch

/** The Holon Pentad context for each node */
export interface HolonContext {
  why?: string;    // Etymon - justification
  where?: string;  // Locus - physical binding
  who?: string;    // Lineage - agent identity
  when?: string;   // Kairos - opportune moment
  what?: string;   // Artifact - latent shape
}

/** A single node in the branching DAG */
export interface BranchNode {
  id: string;
  label: string;
  state: BranchNodeState;
  phase: EntelexisPhase;
  parentId?: string;
  childIds: string[];
  depth: number;
  holon?: HolonContext;
  hysteresis?: HysteresisTrace;
  metadata?: {
    probability?: number;    // Likelihood of this path (0-1)
    entropy?: number;        // Decision uncertainty
    timestamp?: string;      // When decision was made
    agent?: string;          // Which agent made the decision
    reasoning?: string;      // Why this path was chosen/rejected
  };
  collapsed?: boolean;       // For collapsible branches
}

/** Edge connecting two nodes */
export interface BranchEdge {
  sourceId: string;
  targetId: string;
  weight?: number;           // Visual thickness
  probability?: number;      // Path probability
  selected?: boolean;        // Was this path taken?
}

/** Complete branch graph data */
export interface BranchGraphData {
  nodes: BranchNode[];
  edges: BranchEdge[];
  rootId: string;
  title?: string;
  timestamp?: string;
}

/** Component props */
interface BranchingVisualizationProps {
  graph?: BranchGraphData;
  width?: number;
  height?: number;
  onNodeClick$?: QRL<(node: BranchNode) => void>;
  onNodeHover$?: QRL<(node: BranchNode | null) => void>;
  enableCollapse?: boolean;
  showProbabilities?: boolean;
  showHolonContext?: boolean;
  animateTransitions?: boolean;
}

// =============================================================================
// Constants
// =============================================================================

/** Colors for each entelechesis phase */
const PHASE_COLORS: Record<EntelexisPhase, { fill: string; stroke: string; glow: string }> = {
  potential: {
    fill: '#1e3a5f',
    stroke: '#38bdf8',
    glow: 'rgba(56, 189, 248, 0.3)',
  },
  actualizing: {
    fill: '#3b2f5f',
    stroke: '#a855f7',
    glow: 'rgba(168, 85, 247, 0.4)',
  },
  actualized: {
    fill: '#1e4d3d',
    stroke: '#22c55e',
    glow: 'rgba(34, 197, 94, 0.3)',
  },
  decaying: {
    fill: '#4d3d1e',
    stroke: '#f59e0b',
    glow: 'rgba(245, 158, 11, 0.2)',
  },
};

/** Colors for node states */
const STATE_COLORS: Record<BranchNodeState, { opacity: number; dashArray: string }> = {
  potential: { opacity: 0.5, dashArray: '4 4' },
  actualizing: { opacity: 0.9, dashArray: '8 2' },
  actualized: { opacity: 1.0, dashArray: 'none' },
  counterfactual: { opacity: 0.3, dashArray: '2 6' },
  decaying: { opacity: 0.4, dashArray: '1 3' },
};

/** Layout constants */
const NODE_RADIUS = 20;
const NODE_SPACING_X = 120;
const NODE_SPACING_Y = 80;
const PADDING = 60;

// =============================================================================
// Layout Algorithm
// =============================================================================

interface LayoutNode {
  node: BranchNode;
  x: number;
  y: number;
  subtreeWidth: number;
}

interface LayoutResult {
  nodes: LayoutNode[];
  edges: { edge: BranchEdge; x1: number; y1: number; x2: number; y2: number }[];
  width: number;
  height: number;
}

function computeLayout(graph: BranchGraphData, collapsedIds: Set<string>): LayoutResult {
  const nodeMap = new Map<string, BranchNode>();
  graph.nodes.forEach(n => nodeMap.set(n.id, n));

  // Build tree structure respecting collapsed state
  function getVisibleChildren(nodeId: string): string[] {
    const node = nodeMap.get(nodeId);
    if (!node || collapsedIds.has(nodeId)) return [];
    return node.childIds.filter(id => nodeMap.has(id));
  }

  // Calculate subtree widths (bottom-up)
  const subtreeWidths = new Map<string, number>();

  function calcSubtreeWidth(nodeId: string): number {
    const children = getVisibleChildren(nodeId);
    if (children.length === 0) {
      subtreeWidths.set(nodeId, 1);
      return 1;
    }
    const width = children.reduce((sum, cid) => sum + calcSubtreeWidth(cid), 0);
    subtreeWidths.set(nodeId, width);
    return width;
  }

  calcSubtreeWidth(graph.rootId);

  // Position nodes (top-down)
  const layoutNodes: LayoutNode[] = [];
  const positions = new Map<string, { x: number; y: number }>();

  function positionNode(nodeId: string, x: number, y: number): void {
    const node = nodeMap.get(nodeId);
    if (!node) return;

    positions.set(nodeId, { x, y });
    layoutNodes.push({
      node,
      x,
      y,
      subtreeWidth: subtreeWidths.get(nodeId) || 1,
    });

    const children = getVisibleChildren(nodeId);
    if (children.length === 0) return;

    // Calculate starting x for children
    const totalWidth = children.reduce((sum, cid) => sum + (subtreeWidths.get(cid) || 1), 0);
    let childX = x - ((totalWidth - 1) * NODE_SPACING_X) / 2;

    children.forEach(childId => {
      const childWidth = subtreeWidths.get(childId) || 1;
      const childCenterX = childX + ((childWidth - 1) * NODE_SPACING_X) / 2;
      positionNode(childId, childCenterX, y + NODE_SPACING_Y);
      childX += childWidth * NODE_SPACING_X;
    });
  }

  // Start layout from root
  const rootWidth = subtreeWidths.get(graph.rootId) || 1;
  const startX = PADDING + ((rootWidth - 1) * NODE_SPACING_X) / 2;
  positionNode(graph.rootId, startX, PADDING);

  // Compute edges
  const layoutEdges = graph.edges
    .filter(edge => {
      const sourcePos = positions.get(edge.sourceId);
      const targetPos = positions.get(edge.targetId);
      return sourcePos && targetPos;
    })
    .map(edge => {
      const sourcePos = positions.get(edge.sourceId)!;
      const targetPos = positions.get(edge.targetId)!;
      return {
        edge,
        x1: sourcePos.x,
        y1: sourcePos.y,
        x2: targetPos.x,
        y2: targetPos.y,
      };
    });

  // Calculate bounds
  let maxX = 0, maxY = 0;
  layoutNodes.forEach(ln => {
    maxX = Math.max(maxX, ln.x);
    maxY = Math.max(maxY, ln.y);
  });

  return {
    nodes: layoutNodes,
    edges: layoutEdges,
    width: maxX + PADDING,
    height: maxY + PADDING,
  };
}

// =============================================================================
// Demo Data Generator
// =============================================================================

function generateDemoGraph(): BranchGraphData {
  const now = new Date().toISOString();

  const nodes: BranchNode[] = [
    {
      id: 'root',
      label: 'Task Received',
      state: 'actualized',
      phase: 'actualized',
      childIds: ['branch-a', 'branch-b'],
      depth: 0,
      holon: {
        why: 'ASL-2025: Sequential Collapse Law',
        where: 'nucleus/tools/agent_bus.py',
        who: 'codex-v2',
        when: 'Kairos: Implementation Phase',
        what: 'VerificationModule Interface',
      },
      metadata: {
        timestamp: now,
        agent: 'coordinator',
        reasoning: 'Initial task routing decision',
      },
    },
    {
      id: 'branch-a',
      label: 'Single Agent Path',
      state: 'actualized',
      phase: 'actualized',
      parentId: 'root',
      childIds: ['leaf-a1', 'leaf-a2'],
      depth: 1,
      metadata: {
        probability: 0.7,
        agent: 'codex',
        reasoning: 'Etymon recommends single topology',
      },
    },
    {
      id: 'branch-b',
      label: 'Swarm Path',
      state: 'counterfactual',
      phase: 'potential',
      parentId: 'root',
      childIds: ['leaf-b1'],
      depth: 1,
      metadata: {
        probability: 0.3,
        reasoning: 'Rejected: -70% performance penalty',
      },
    },
    {
      id: 'leaf-a1',
      label: 'Implement Core',
      state: 'actualized',
      phase: 'actualized',
      parentId: 'branch-a',
      childIds: [],
      depth: 2,
      metadata: {
        probability: 0.9,
        agent: 'codex',
      },
    },
    {
      id: 'leaf-a2',
      label: 'Write Tests',
      state: 'actualizing',
      phase: 'actualizing',
      parentId: 'branch-a',
      childIds: ['future-1', 'future-2'],
      depth: 2,
      metadata: {
        probability: 0.85,
        agent: 'codex',
      },
    },
    {
      id: 'leaf-b1',
      label: 'Distribute Work',
      state: 'counterfactual',
      phase: 'potential',
      parentId: 'branch-b',
      childIds: [],
      depth: 2,
      metadata: {
        probability: 0.2,
        reasoning: 'Never evaluated',
      },
    },
    {
      id: 'future-1',
      label: 'Unit Tests',
      state: 'potential',
      phase: 'potential',
      parentId: 'leaf-a2',
      childIds: [],
      depth: 3,
      metadata: {
        probability: 0.6,
      },
    },
    {
      id: 'future-2',
      label: 'Integration Tests',
      state: 'potential',
      phase: 'potential',
      parentId: 'leaf-a2',
      childIds: [],
      depth: 3,
      metadata: {
        probability: 0.4,
      },
    },
  ];

  const edges: BranchEdge[] = [
    { sourceId: 'root', targetId: 'branch-a', selected: true, probability: 0.7 },
    { sourceId: 'root', targetId: 'branch-b', selected: false, probability: 0.3 },
    { sourceId: 'branch-a', targetId: 'leaf-a1', selected: true, probability: 0.9 },
    { sourceId: 'branch-a', targetId: 'leaf-a2', selected: true, probability: 0.85 },
    { sourceId: 'branch-b', targetId: 'leaf-b1', selected: false, probability: 0.2 },
    { sourceId: 'leaf-a2', targetId: 'future-1', probability: 0.6 },
    { sourceId: 'leaf-a2', targetId: 'future-2', probability: 0.4 },
  ];

  return {
    nodes,
    edges,
    rootId: 'root',
    title: 'Task Decision Tree',
    timestamp: now,
  };
}

// =============================================================================
// Sub-Components
// =============================================================================

interface NodeTooltipProps {
  node: BranchNode;
  x: number;
  y: number;
  showHolon?: boolean;
}

const NodeTooltip = component$<NodeTooltipProps>(({ node, x, y, showHolon }) => {
  return (
    <foreignObject
      x={x + NODE_RADIUS + 10}
      y={y - 60}
      width={280}
      height={200}
      class="pointer-events-none"
    >
      <div class="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 shadow-xl text-xs">
        <div class="font-semibold text-foreground mb-2">{node.label}</div>

        {/* State and Phase */}
        <div class="flex items-center gap-2 mb-2">
          <span
            class="px-2 py-0.5 rounded text-[10px] uppercase"
            style={{
              backgroundColor: PHASE_COLORS[node.phase].fill,
              color: PHASE_COLORS[node.phase].stroke,
            }}
          >
            {node.phase}
          </span>
          <span class="text-muted-foreground">{node.state}</span>
        </div>

        {/* Metadata */}
        {node.metadata && (
          <div class="space-y-1 text-muted-foreground">
            {node.metadata.probability !== undefined && (
              <div class="flex justify-between">
                <span>Probability:</span>
                <span class="text-foreground">
                  {(node.metadata.probability * 100).toFixed(0)}%
                </span>
              </div>
            )}
            {node.metadata.agent && (
              <div class="flex justify-between">
                <span>Agent:</span>
                <span class="text-cyan-400">{node.metadata.agent}</span>
              </div>
            )}
            {node.metadata.reasoning && (
              <div class="mt-2 text-[10px] italic opacity-80">
                "{node.metadata.reasoning}"
              </div>
            )}
          </div>
        )}

        {/* Holon Context */}
        {showHolon && node.holon && (
          <div class="mt-3 pt-2 border-t border-border/50">
            <div class="text-[10px] uppercase text-muted-foreground mb-1">
              Holon Pentad
            </div>
            <div class="space-y-0.5 text-[10px]">
              {node.holon.why && (
                <div>
                  <span class="text-purple-400">WHY:</span> {node.holon.why}
                </div>
              )}
              {node.holon.where && (
                <div>
                  <span class="text-blue-400">WHERE:</span> {node.holon.where}
                </div>
              )}
              {node.holon.who && (
                <div>
                  <span class="text-green-400">WHO:</span> {node.holon.who}
                </div>
              )}
              {node.holon.when && (
                <div>
                  <span class="text-yellow-400">WHEN:</span> {node.holon.when}
                </div>
              )}
              {node.holon.what && (
                <div>
                  <span class="text-cyan-400">WHAT:</span> {node.holon.what}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </foreignObject>
  );
});

// =============================================================================
// Main Component
// =============================================================================

export const BranchingVisualization = component$<BranchingVisualizationProps>((props) => {
  // State
  const collapsedIds = useStore<Set<string>>(new Set());
  const hoveredNodeId = useSignal<string | null>(null);
  const selectedNodeId = useSignal<string | null>(null);

  // Use demo data if none provided
  const graphData = useComputed$(() => props.graph ?? generateDemoGraph());

  // Compute layout
  const layout = useComputed$(() => {
    return computeLayout(graphData.value, new Set(Object.keys(collapsedIds)));
  });

  // Computed dimensions
  const viewWidth = useComputed$(() => props.width ?? Math.max(600, layout.value.width));
  const viewHeight = useComputed$(() => props.height ?? Math.max(400, layout.value.height));

  // Event handlers
  const handleNodeClick = $((node: BranchNode) => {
    if (props.enableCollapse !== false && node.childIds.length > 0) {
      if (collapsedIds.has(node.id)) {
        collapsedIds.delete(node.id);
      } else {
        collapsedIds.add(node.id);
      }
    }
    selectedNodeId.value = node.id;
    props.onNodeClick$?.(node);
  });

  const handleNodeHover = $((node: BranchNode | null) => {
    hoveredNodeId.value = node?.id ?? null;
    props.onNodeHover$?.(node);
  });

  // Find hovered node for tooltip
  const hoveredNode = useComputed$(() => {
    if (!hoveredNodeId.value) return null;
    const ln = layout.value.nodes.find(ln => ln.node.id === hoveredNodeId.value);
    return ln ? { node: ln.node, x: ln.x, y: ln.y } : null;
  });

  // Highlighted path (from hovered node to root)
  const highlightedPath = useComputed$(() => {
    if (!hoveredNodeId.value) return new Set<string>();
    const path = new Set<string>();
    let current = hoveredNodeId.value;
    const nodeMap = new Map(graphData.value.nodes.map(n => [n.id, n]));

    while (current) {
      path.add(current);
      const node = nodeMap.get(current);
      current = node?.parentId ?? '';
    }
    return path;
  });

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="p-3 border-b border-border flex items-center justify-between bg-muted/20">
        <div class="flex items-center gap-2">
          <svg class="w-5 h-5 text-purple-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M6 3v18M6 9a6 6 0 0 0 6-6M6 15a6 6 0 0 1 6 6M18 3v18M18 9a6 6 0 0 1-6-6M18 15a6 6 0 0 0-6 6" />
          </svg>
          <span class="font-semibold">Delineation Branching</span>
          {graphData.value.title && (
            <span class="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">
              {graphData.value.title}
            </span>
          )}
        </div>
        <div class="flex items-center gap-4 text-xs text-muted-foreground">
          <span>{layout.value.nodes.length} nodes</span>
          <span>{layout.value.edges.length} edges</span>
        </div>
      </div>

      {/* Legend */}
      <div class="px-3 py-2 border-b border-border/50 flex items-center gap-6 text-xs bg-muted/10">
        <div class="text-muted-foreground">Entelechesis:</div>
        {(['potential', 'actualizing', 'actualized', 'decaying'] as EntelexisPhase[]).map(phase => (
          <div key={phase} class="flex items-center gap-1.5">
            <span
              class="w-3 h-3 rounded-full border-2"
              style={{
                backgroundColor: PHASE_COLORS[phase].fill,
                borderColor: PHASE_COLORS[phase].stroke,
              }}
            />
            <span class="capitalize">{phase}</span>
          </div>
        ))}
      </div>

      {/* SVG Canvas */}
      <div class="relative overflow-auto" style={{ maxHeight: `${viewHeight.value + 100}px` }}>
        <svg
          width={viewWidth.value}
          height={viewHeight.value}
          viewBox={`0 0 ${viewWidth.value} ${viewHeight.value}`}
          class="block"
        >
          {/* Background grid */}
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" stroke-width="1" />
            </pattern>
            {/* Glow filters */}
            {(['potential', 'actualizing', 'actualized', 'decaying'] as EntelexisPhase[]).map(phase => (
              <filter key={`glow-${phase}`} id={`glow-${phase}`} x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                <feMerge>
                  <feMergeNode in="coloredBlur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            ))}
            {/* Arrow marker */}
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" opacity="0.5" />
            </marker>
          </defs>

          <rect width="100%" height="100%" fill="url(#grid)" />

          {/* Edges */}
          <g class="edges">
            {layout.value.edges.map((e, idx) => {
              const sourceNode = graphData.value.nodes.find(n => n.id === e.edge.sourceId);
              const targetNode = graphData.value.nodes.find(n => n.id === e.edge.targetId);
              if (!sourceNode || !targetNode) return null;

              const isHighlighted = highlightedPath.value.has(e.edge.sourceId) &&
                                   highlightedPath.value.has(e.edge.targetId);
              const stateStyle = STATE_COLORS[targetNode.state];
              const phaseColor = PHASE_COLORS[targetNode.phase];

              // Curved path for tree edges
              const midY = (e.y1 + e.y2) / 2;
              const pathD = e.x1 === e.x2
                ? `M ${e.x1} ${e.y1 + NODE_RADIUS} L ${e.x2} ${e.y2 - NODE_RADIUS}`
                : `M ${e.x1} ${e.y1 + NODE_RADIUS} C ${e.x1} ${midY}, ${e.x2} ${midY}, ${e.x2} ${e.y2 - NODE_RADIUS}`;

              return (
                <g key={`edge-${idx}`}>
                  {/* Edge glow for highlighted paths */}
                  {isHighlighted && (
                    <path
                      d={pathD}
                      fill="none"
                      stroke={phaseColor.stroke}
                      stroke-width="6"
                      opacity="0.3"
                      stroke-linecap="round"
                    />
                  )}
                  {/* Main edge */}
                  <path
                    d={pathD}
                    fill="none"
                    stroke={e.edge.selected ? phaseColor.stroke : '#64748b'}
                    stroke-width={isHighlighted ? 3 : 2}
                    stroke-dasharray={stateStyle.dashArray}
                    opacity={isHighlighted ? 1 : stateStyle.opacity}
                    stroke-linecap="round"
                    marker-end="url(#arrow)"
                    class={props.animateTransitions !== false ? 'transition-all duration-300' : ''}
                  />
                  {/* Probability label */}
                  {props.showProbabilities && e.edge.probability !== undefined && (
                    <text
                      x={(e.x1 + e.x2) / 2 + 8}
                      y={midY}
                      font-size="10"
                      fill="rgba(148, 163, 184, 0.7)"
                    >
                      {(e.edge.probability * 100).toFixed(0)}%
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* Nodes */}
          <g class="nodes">
            {layout.value.nodes.map((ln) => {
              const { node, x, y } = ln;
              const phaseStyle = PHASE_COLORS[node.phase];
              const stateStyle = STATE_COLORS[node.state];
              const isHovered = hoveredNodeId.value === node.id;
              const isSelected = selectedNodeId.value === node.id;
              const isOnPath = highlightedPath.value.has(node.id);
              const hasChildren = node.childIds.length > 0;
              const isCollapsed = collapsedIds.has(node.id);

              return (
                <g
                  key={node.id}
                  class="cursor-pointer"
                  onMouseEnter$={() => handleNodeHover(node)}
                  onMouseLeave$={() => handleNodeHover(null)}
                  onClick$={() => handleNodeClick(node)}
                >
                  {/* Node glow */}
                  {(isHovered || isOnPath || node.state === 'actualizing') && (
                    <circle
                      cx={x}
                      cy={y}
                      r={NODE_RADIUS + 8}
                      fill={phaseStyle.glow}
                      filter={`url(#glow-${node.phase})`}
                      class={props.animateTransitions !== false ? 'transition-all duration-300' : ''}
                    />
                  )}

                  {/* Main node circle */}
                  <circle
                    cx={x}
                    cy={y}
                    r={NODE_RADIUS}
                    fill={phaseStyle.fill}
                    stroke={isSelected ? '#fff' : phaseStyle.stroke}
                    stroke-width={isSelected ? 3 : 2}
                    opacity={isOnPath ? 1 : stateStyle.opacity}
                    class={props.animateTransitions !== false ? 'transition-all duration-300' : ''}
                  />

                  {/* Inner pulse for actualizing state */}
                  {node.state === 'actualizing' && (
                    <circle
                      cx={x}
                      cy={y}
                      r={NODE_RADIUS - 4}
                      fill="none"
                      stroke={phaseStyle.stroke}
                      stroke-width="2"
                      opacity="0.5"
                      class="animate-pulse"
                    />
                  )}

                  {/* Collapse indicator */}
                  {hasChildren && (
                    <g transform={`translate(${x + NODE_RADIUS - 6}, ${y + NODE_RADIUS - 6})`}>
                      <circle r="8" fill="#1e293b" stroke={phaseStyle.stroke} stroke-width="1" />
                      <text
                        font-size="10"
                        fill={phaseStyle.stroke}
                        text-anchor="middle"
                        dominant-baseline="central"
                      >
                        {isCollapsed ? '+' : '-'}
                      </text>
                    </g>
                  )}

                  {/* Node label */}
                  <text
                    x={x}
                    y={y + NODE_RADIUS + 16}
                    font-size="11"
                    fill={isOnPath ? '#fff' : 'rgba(226, 232, 240, 0.8)'}
                    text-anchor="middle"
                    class="pointer-events-none"
                  >
                    {node.label.length > 16 ? node.label.slice(0, 14) + '...' : node.label}
                  </text>

                  {/* State icon in center */}
                  <text
                    x={x}
                    y={y + 4}
                    font-size="14"
                    text-anchor="middle"
                    fill={phaseStyle.stroke}
                  >
                    {node.state === 'actualized' ? '\u2713' :
                     node.state === 'actualizing' ? '\u2022' :
                     node.state === 'counterfactual' ? '\u2717' :
                     node.state === 'potential' ? '\u25CB' : '\u2606'}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Tooltip */}
          {hoveredNode.value && (
            <NodeTooltip
              node={hoveredNode.value.node}
              x={hoveredNode.value.x}
              y={hoveredNode.value.y}
              showHolon={props.showHolonContext}
            />
          )}
        </svg>
      </div>

      {/* Footer with selected node details */}
      {selectedNodeId.value && (
        <div class="p-3 border-t border-border bg-muted/10 text-xs">
          {(() => {
            const node = graphData.value.nodes.find(n => n.id === selectedNodeId.value);
            if (!node) return null;

            return (
              <div class="flex items-start justify-between gap-4">
                <div>
                  <div class="font-semibold text-foreground">{node.label}</div>
                  <div class="text-muted-foreground mt-1">
                    Phase: <span class="text-foreground">{node.phase}</span> |
                    State: <span class="text-foreground">{node.state}</span> |
                    Depth: <span class="text-foreground">{node.depth}</span>
                  </div>
                </div>
                {node.metadata?.reasoning && (
                  <div class="text-muted-foreground italic max-w-xs">
                    "{node.metadata.reasoning}"
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
});

export default BranchingVisualization;
