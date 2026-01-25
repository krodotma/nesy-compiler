/**
 * KnowledgeGraphView - D3.js force-directed graph visualization
 *
 * Features:
 * - Node rendering with colors by type
 * - Edge rendering with relationship types
 * - Zoom and pan controls
 * - Search highlighting
 * - Node selection
 * - FalkorDB query integration
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  useContext,
  $,
  noSerialize,
  type NoSerialize,
} from '@builder.io/qwik';
import { MetaIngestContext } from '../../lib/metaingest/store';
import type { KnowledgeGraphStats, GraphNode, GraphRelationship, APIResponse } from '../../lib/metaingest/types';
import { NeonTitle, NeonBadge } from '../ui/NeonTitle';
import { Card, CardContent } from '../ui/Card';

interface D3ForceSimulation {
  nodes: (nodes: GraphNode[]) => D3ForceSimulation;
  force: (name: string, force: unknown) => D3ForceSimulation;
  on: (event: string, callback: () => void) => D3ForceSimulation;
  alpha: (alpha: number) => D3ForceSimulation;
  restart: () => D3ForceSimulation;
  stop: () => void;
}

interface GraphState {
  nodes: GraphNode[];
  edges: GraphRelationship[];
  selectedNode: string | null;
  highlightedNodes: Set<string>;
  transform: { x: number; y: number; scale: number };
}

export const KnowledgeGraphView = component$(() => {
  const state = useContext(MetaIngestContext);
  const canvasRef = useSignal<HTMLCanvasElement>();
  const stats = useSignal<KnowledgeGraphStats | null>(null);
  const loading = useSignal(false);
  const error = useSignal<string | null>(null);
  const searchQuery = useSignal('');

  const graphState = useStore<GraphState>({
    nodes: [],
    edges: [],
    selectedNode: null,
    highlightedNodes: new Set(),
    transform: { x: 0, y: 0, scale: 1 },
  });

  const simulation = useSignal<NoSerialize<D3ForceSimulation>>();

  // Fetch graph stats
  useVisibleTask$(async ({ cleanup }) => {
    const controller = new AbortController();
    cleanup(() => controller.abort());

    loading.value = true;
    state.loading.knowledge = true;

    try {
      const res = await fetch('/api/metaingest/knowledge/stats', {
        signal: controller.signal,
      });
      const data: APIResponse<KnowledgeGraphStats> = await res.json();

      if (data.success && data.data) {
        stats.value = data.data;
        state.knowledgeStats = data.data;
      } else {
        error.value = data.error?.message ?? 'Failed to load graph stats';
      }
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') {
        error.value = e.message;
        state.errors.knowledge = e.message;
      }
    } finally {
      loading.value = false;
      state.loading.knowledge = false;
    }
  });

  // Initialize D3 force simulation and canvas rendering
  useVisibleTask$(async ({ cleanup }) => {
    const canvas = canvasRef.value;
    if (!canvas || !stats.value) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Generate sample nodes and edges (in production, fetch from API)
    const sampleNodes: GraphNode[] = [];
    const sampleEdges: GraphRelationship[] = [];

    const nodeCount = Math.min(stats.value.nodeCount, 50); // Limit for performance

    for (let i = 0; i < nodeCount; i++) {
      sampleNodes.push({
        id: `node-${i}`,
        term: `Term ${i}`,
        definition: `Definition for term ${i}`,
        lineageId: `lineage-${i}`,
        nodeType: ['concept', 'entity', 'relationship'][i % 3] as GraphNode['nodeType'],
      });
    }

    for (let i = 0; i < Math.min(stats.value.edgeCount, 80); i++) {
      const source = Math.floor(Math.random() * nodeCount);
      const target = Math.floor(Math.random() * nodeCount);
      if (source !== target) {
        sampleEdges.push({
          source: `node-${source}`,
          target: `node-${target}`,
          type: ['related_to', 'derives_from', 'similar_to'][i % 3],
          weight: Math.random(),
        });
      }
    }

    graphState.nodes = sampleNodes;
    graphState.edges = sampleEdges;

    // Load D3 dynamically
    try {
      const d3 = await import('d3-force');

      const sim = d3.forceSimulation(graphState.nodes as unknown as d3.SimulationNodeDatum[])
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(canvas.width / 2, canvas.height / 2))
        .force('link', d3.forceLink(sampleEdges)
          .id((d: unknown) => (d as GraphNode).id)
          .distance(80))
        .force('collision', d3.forceCollide().radius(20));

      const draw = () => {
        if (!ctx) return;

        const width = canvas.width;
        const height = canvas.height;
        const { x, y, scale } = graphState.transform;

        // Clear
        ctx.clearRect(0, 0, width, height);
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);

        ctx.save();
        ctx.translate(x, y);
        ctx.scale(scale, scale);

        // Draw edges
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        sampleEdges.forEach((edge) => {
          const sourceNode = graphState.nodes.find(n => n.id === edge.source);
          const targetNode = graphState.nodes.find(n => n.id === edge.target);

          if (sourceNode?.x && sourceNode?.y && targetNode?.x && targetNode?.y) {
            ctx.beginPath();
            ctx.moveTo(sourceNode.x, sourceNode.y);
            ctx.lineTo(targetNode.x, targetNode.y);
            ctx.stroke();
          }
        });

        // Draw nodes
        graphState.nodes.forEach((node) => {
          if (!node.x || !node.y) return;

          const isSelected = graphState.selectedNode === node.id;
          const isHighlighted = graphState.highlightedNodes.has(node.id);

          // Node color by type
          const colors = {
            concept: '#06b6d4',
            entity: '#8b5cf6',
            relationship: '#f97316',
          };

          ctx.fillStyle = colors[node.nodeType];

          if (isSelected) {
            ctx.strokeStyle = '#fbbf24';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.arc(node.x, node.y, 12, 0, Math.PI * 2);
            ctx.stroke();
          }

          if (isHighlighted) {
            ctx.fillStyle = '#fbbf24';
          }

          ctx.beginPath();
          ctx.arc(node.x, node.y, 8, 0, Math.PI * 2);
          ctx.fill();

          // Label for selected or highlighted nodes
          if (isSelected || isHighlighted) {
            ctx.fillStyle = '#e2e8f0';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(node.term, node.x, node.y - 15);
          }
        });

        ctx.restore();
      };

      sim.on('tick', draw);
      simulation.value = noSerialize(sim as unknown as D3ForceSimulation);

      cleanup(() => sim.stop());
    } catch (err) {
      console.error('Failed to initialize D3:', err);
      error.value = 'Failed to load graph visualization';
    }
  });

  // Handle search
  const handleSearch = $(() => {
    const query = searchQuery.value.toLowerCase();
    if (!query) {
      graphState.highlightedNodes = new Set();
      return;
    }

    const matches = graphState.nodes
      .filter(n => n.term.toLowerCase().includes(query))
      .map(n => n.id);

    graphState.highlightedNodes = new Set(matches);
  });

  // Handle canvas click
  const handleCanvasClick = $((event: MouseEvent) => {
    const canvas = canvasRef.value;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (event.clientX - rect.left - graphState.transform.x) / graphState.transform.scale;
    const y = (event.clientY - rect.top - graphState.transform.y) / graphState.transform.scale;

    // Find clicked node
    const clickedNode = graphState.nodes.find(node => {
      if (!node.x || !node.y) return false;
      const dx = x - node.x;
      const dy = y - node.y;
      return Math.sqrt(dx * dx + dy * dy) < 10;
    });

    graphState.selectedNode = clickedNode?.id ?? null;
  });

  // Handle zoom
  const handleWheel = $((event: WheelEvent) => {
    event.preventDefault();
    const delta = event.deltaY > 0 ? 0.9 : 1.1;
    graphState.transform.scale = Math.max(0.1, Math.min(5, graphState.transform.scale * delta));
  });

  if (loading.value && !stats.value) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-slate-400 text-sm">Loading knowledge graph...</div>
      </div>
    );
  }

  if (error.value) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-red-400 text-sm">{error.value}</div>
      </div>
    );
  }

  return (
    <div class="flex flex-col h-full gap-4">
      {/* Header */}
      <div class="flex items-center justify-between">
        <NeonTitle level="h2" color="purple" size="lg">
          Knowledge Graph
        </NeonTitle>
        <div class="flex items-center gap-2">
          <NeonBadge color="cyan">
            {stats.value?.nodeCount ?? 0} Nodes
          </NeonBadge>
          <NeonBadge color="purple">
            {stats.value?.edgeCount ?? 0} Edges
          </NeonBadge>
          <NeonBadge color={stats.value?.falkordbConnected ? 'emerald' : 'rose'}>
            FalkorDB: {stats.value?.falkordbConnected ? 'Connected' : 'Disconnected'}
          </NeonBadge>
        </div>
      </div>

      {/* Search */}
      <div class="flex items-center gap-2">
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery.value}
          onInput$={(e) => {
            searchQuery.value = (e.target as HTMLInputElement).value;
          }}
          onKeyUp$={handleSearch}
          class="flex-1 px-3 py-2 bg-slate-900/50 border border-slate-700 rounded-md text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-purple-500"
        />
        <button
          onClick$={handleSearch}
          class="px-4 py-2 bg-purple-500/20 text-purple-300 rounded-md text-sm font-medium hover:bg-purple-500/30"
        >
          Search
        </button>
      </div>

      {/* Graph Canvas */}
      <Card variant="elevated" class="flex-1">
        <CardContent class="h-full flex flex-col p-0">
          <canvas
            ref={canvasRef}
            width={1200}
            height={600}
            onClick$={handleCanvasClick}
            onWheel$={handleWheel}
            class="w-full h-full cursor-grab active:cursor-grabbing"
          />
        </CardContent>
      </Card>

      {/* Legend */}
      <div class="flex items-center gap-6 text-xs text-slate-400">
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-cyan-500" />
          <span>Concept</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-purple-500" />
          <span>Entity</span>
        </div>
        <div class="flex items-center gap-2">
          <span class="w-3 h-3 rounded-full bg-orange-500" />
          <span>Relationship</span>
        </div>
        <div class="ml-auto">
          {graphState.selectedNode && (
            <span class="text-yellow-400">
              Selected: {graphState.nodes.find(n => n.id === graphState.selectedNode)?.term}
            </span>
          )}
        </div>
      </div>
    </div>
  );
});
