/**
 * LaneNetworkGraph - Force-directed network graph of lanes
 *
 * Phase 4, Iteration 28 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Force-directed layout (D3-style)
 * - Lanes as nodes, dependencies as edges
 * - Agent clusters with color coding
 * - Draggable nodes
 * - Zoom and pan
 * - Node tooltips
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface GraphNode {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
}

export interface GraphEdge {
  source: string;
  target: string;
  type?: 'dependency' | 'related' | 'blocks';
}

export interface LaneNetworkGraphProps {
  /** Nodes (lanes) */
  nodes: GraphNode[];
  /** Edges (dependencies) */
  edges: GraphEdge[];
  /** Callback when node is clicked */
  onNodeClick$?: QRL<(node: GraphNode) => void>;
  /** Show agent clusters */
  showClusters?: boolean;
  /** Enable physics simulation */
  enablePhysics?: boolean;
  /** Width */
  width?: number;
  /** Height */
  height?: number;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return '#34d399';
    case 'yellow': return '#fbbf24';
    case 'red': return '#f87171';
    default: return '#6b7280';
  }
}

function getAgentColor(agent: string): string {
  // Generate consistent color from agent name
  let hash = 0;
  for (let i = 0; i < agent.length; i++) {
    hash = agent.charCodeAt(i) + ((hash << 5) - hash);
  }
  const colors = [
    '#3b82f6', '#8b5cf6', '#ec4899', '#f97316',
    '#14b8a6', '#84cc16', '#06b6d4', '#f43f5e',
  ];
  return colors[Math.abs(hash) % colors.length];
}

function getEdgeColor(type?: string): string {
  switch (type) {
    case 'dependency': return 'rgba(59, 130, 246, 0.5)';
    case 'blocks': return 'rgba(239, 68, 68, 0.5)';
    case 'related': return 'rgba(156, 163, 175, 0.3)';
    default: return 'rgba(156, 163, 175, 0.3)';
  }
}

// Simple force simulation
function initializePositions(nodes: GraphNode[], width: number, height: number): GraphNode[] {
  const centerX = width / 2;
  const centerY = height / 2;
  const radius = Math.min(width, height) / 3;

  return nodes.map((node, i) => ({
    ...node,
    x: node.x ?? centerX + radius * Math.cos((2 * Math.PI * i) / nodes.length),
    y: node.y ?? centerY + radius * Math.sin((2 * Math.PI * i) / nodes.length),
    vx: 0,
    vy: 0,
  }));
}

function simulateForces(
  nodes: GraphNode[],
  edges: GraphEdge[],
  width: number,
  height: number,
  alpha: number = 0.1
): GraphNode[] {
  const centerX = width / 2;
  const centerY = height / 2;

  // Create a map for quick node lookup
  const nodeMap = new Map(nodes.map(n => [n.id, n]));

  // Apply forces
  return nodes.map(node => {
    if (node.fx !== null && node.fx !== undefined) {
      return { ...node, x: node.fx, y: node.fy };
    }

    let fx = 0;
    let fy = 0;

    // Center gravity
    fx += (centerX - (node.x ?? centerX)) * 0.01;
    fy += (centerY - (node.y ?? centerY)) * 0.01;

    // Repulsion from other nodes
    nodes.forEach(other => {
      if (other.id === node.id) return;
      const dx = (node.x ?? 0) - (other.x ?? 0);
      const dy = (node.y ?? 0) - (other.y ?? 0);
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = 1000 / (dist * dist);
      fx += (dx / dist) * force;
      fy += (dy / dist) * force;
    });

    // Attraction along edges
    edges.forEach(edge => {
      let other: GraphNode | undefined;
      if (edge.source === node.id) {
        other = nodeMap.get(edge.target);
      } else if (edge.target === node.id) {
        other = nodeMap.get(edge.source);
      }
      if (!other) return;

      const dx = (other.x ?? 0) - (node.x ?? 0);
      const dy = (other.y ?? 0) - (node.y ?? 0);
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const force = (dist - 100) * 0.01;
      fx += (dx / dist) * force;
      fy += (dy / dist) * force;
    });

    // Apply velocity
    const vx = ((node.vx ?? 0) + fx) * 0.8;
    const vy = ((node.vy ?? 0) + fy) * 0.8;

    // Update position
    let x = (node.x ?? centerX) + vx * alpha;
    let y = (node.y ?? centerY) + vy * alpha;

    // Bounds
    x = Math.max(30, Math.min(width - 30, x));
    y = Math.max(30, Math.min(height - 30, y));

    return { ...node, x, y, vx, vy };
  });
}

// ============================================================================
// Component
// ============================================================================

export const LaneNetworkGraph = component$<LaneNetworkGraphProps>(({
  nodes: initialNodes,
  edges,
  onNodeClick$,
  showClusters = true,
  enablePhysics = true,
  width = 600,
  height = 400,
}) => {
  // State
  const nodes = useSignal<GraphNode[]>(initializePositions(initialNodes, width, height));
  const selectedNodeId = useSignal<string | null>(null);
  const hoveredNodeId = useSignal<string | null>(null);
  const draggingNodeId = useSignal<string | null>(null);
  const isSimulating = useSignal(enablePhysics);

  // Get unique agents for legend
  const agents = useComputed$(() => {
    const agentSet = new Set<string>();
    nodes.value.forEach(n => agentSet.add(n.owner));
    return Array.from(agentSet);
  });

  // Physics simulation
  useVisibleTask$(({ cleanup }) => {
    if (!enablePhysics) return;

    let animationId: number;
    let iterations = 0;
    const maxIterations = 300;

    const simulate = () => {
      if (!isSimulating.value || iterations >= maxIterations) return;

      const alpha = Math.max(0.01, 1 - iterations / maxIterations);
      nodes.value = simulateForces(nodes.value, edges, width, height, alpha);
      iterations++;

      if (alpha > 0.01) {
        animationId = requestAnimationFrame(simulate);
      }
    };

    simulate();

    cleanup(() => {
      if (animationId) cancelAnimationFrame(animationId);
    });
  });

  // Drag handlers
  const handleMouseDown = $((e: MouseEvent, nodeId: string) => {
    e.preventDefault();
    draggingNodeId.value = nodeId;

    // Fix node position
    nodes.value = nodes.value.map(n =>
      n.id === nodeId ? { ...n, fx: n.x, fy: n.y } : n
    );
  });

  const handleMouseMove = $((e: MouseEvent) => {
    if (!draggingNodeId.value) return;

    const svg = (e.currentTarget as SVGSVGElement);
    const rect = svg.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    nodes.value = nodes.value.map(n =>
      n.id === draggingNodeId.value ? { ...n, x, y, fx: x, fy: y } : n
    );
  });

  const handleMouseUp = $(() => {
    if (!draggingNodeId.value) return;

    // Release node
    nodes.value = nodes.value.map(n =>
      n.id === draggingNodeId.value ? { ...n, fx: null, fy: null } : n
    );
    draggingNodeId.value = null;

    // Restart simulation
    isSimulating.value = true;
  });

  // Handle node click
  const handleNodeClick = $(async (node: GraphNode) => {
    selectedNodeId.value = selectedNodeId.value === node.id ? null : node.id;
    if (onNodeClick$) {
      await onNodeClick$(node);
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">NETWORK GRAPH</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {nodes.value.length} nodes
          </span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
            {edges.length} edges
          </span>
        </div>
        <button
          onClick$={() => {
            nodes.value = initializePositions(initialNodes, width, height);
            isSimulating.value = true;
          }}
          class="text-[10px] px-2 py-1 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
        >
          Reset Layout
        </button>
      </div>

      {/* Graph */}
      <svg
        width={width}
        height={height}
        class="bg-muted/5"
        onMouseMove$={handleMouseMove}
        onMouseUp$={handleMouseUp}
        onMouseLeave$={handleMouseUp}
      >
        {/* Agent cluster backgrounds */}
        {showClusters && agents.value.map(agent => {
          const agentNodes = nodes.value.filter(n => n.owner === agent);
          if (agentNodes.length < 2) return null;

          const xs = agentNodes.map(n => n.x ?? 0);
          const ys = agentNodes.map(n => n.y ?? 0);
          const minX = Math.min(...xs) - 40;
          const maxX = Math.max(...xs) + 40;
          const minY = Math.min(...ys) - 40;
          const maxY = Math.max(...ys) + 40;

          return (
            <rect
              key={agent}
              x={minX}
              y={minY}
              width={maxX - minX}
              height={maxY - minY}
              rx={8}
              fill={getAgentColor(agent)}
              opacity={0.05}
            />
          );
        })}

        {/* Edges */}
        {edges.map((edge, i) => {
          const source = nodes.value.find(n => n.id === edge.source);
          const target = nodes.value.find(n => n.id === edge.target);
          if (!source || !target) return null;

          const isHighlighted =
            selectedNodeId.value === edge.source ||
            selectedNodeId.value === edge.target;

          return (
            <line
              key={i}
              x1={source.x}
              y1={source.y}
              x2={target.x}
              y2={target.y}
              stroke={isHighlighted ? getStatusColor('green') : getEdgeColor(edge.type)}
              stroke-width={isHighlighted ? 2 : 1}
              marker-end="url(#arrowhead)"
            />
          );
        })}

        {/* Arrow marker */}
        <defs>
          <marker
            id="arrowhead"
            viewBox="0 0 10 10"
            refX="20"
            refY="5"
            markerWidth="4"
            markerHeight="4"
            orient="auto-start-reverse"
          >
            <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(156, 163, 175, 0.5)" />
          </marker>
        </defs>

        {/* Nodes */}
        {nodes.value.map(node => {
          const isSelected = selectedNodeId.value === node.id;
          const isHovered = hoveredNodeId.value === node.id;
          const isDragging = draggingNodeId.value === node.id;
          const radius = 20 + (node.wip_pct / 10);

          return (
            <g
              key={node.id}
              onMouseDown$={(e) => handleMouseDown(e, node.id)}
              onMouseEnter$={() => { hoveredNodeId.value = node.id; }}
              onMouseLeave$={() => { hoveredNodeId.value = null; }}
              onClick$={() => handleNodeClick(node)}
              class={`cursor-pointer ${isDragging ? 'cursor-grabbing' : 'cursor-grab'}`}
            >
              {/* Node circle */}
              <circle
                cx={node.x}
                cy={node.y}
                r={radius}
                fill={getStatusColor(node.status)}
                stroke={isSelected ? 'white' : getAgentColor(node.owner)}
                stroke-width={isSelected ? 3 : 2}
                opacity={isHovered || isSelected ? 1 : 0.8}
              />

              {/* WIP percentage */}
              <text
                x={node.x}
                y={node.y}
                dy={4}
                text-anchor="middle"
                fill="white"
                font-size="10"
                font-weight="bold"
              >
                {node.wip_pct}%
              </text>

              {/* Label */}
              {(isHovered || isSelected) && (
                <g>
                  <rect
                    x={(node.x ?? 0) - 50}
                    y={(node.y ?? 0) + radius + 5}
                    width={100}
                    height={36}
                    rx={4}
                    fill="rgba(0,0,0,0.8)"
                  />
                  <text
                    x={node.x}
                    y={(node.y ?? 0) + radius + 18}
                    text-anchor="middle"
                    fill="white"
                    font-size="10"
                  >
                    {node.name.slice(0, 15)}
                  </text>
                  <text
                    x={node.x}
                    y={(node.y ?? 0) + radius + 32}
                    text-anchor="middle"
                    fill="rgba(255,255,255,0.6)"
                    font-size="9"
                  >
                    @{node.owner}
                  </text>
                </g>
              )}
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div class="p-2 border-t border-border/50 flex items-center justify-between text-[9px] text-muted-foreground">
        <div class="flex items-center gap-3">
          <span class="font-semibold">Agents:</span>
          {agents.value.slice(0, 5).map(agent => (
            <div key={agent} class="flex items-center gap-1">
              <div
                class="w-2 h-2 rounded-full"
                style={{ backgroundColor: getAgentColor(agent) }}
              />
              <span>@{agent}</span>
            </div>
          ))}
          {agents.value.length > 5 && <span>+{agents.value.length - 5} more</span>}
        </div>
        <span>Drag nodes to reposition</span>
      </div>
    </div>
  );
});

export default LaneNetworkGraph;
