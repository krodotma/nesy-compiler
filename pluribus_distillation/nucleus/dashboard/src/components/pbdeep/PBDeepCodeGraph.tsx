import { component$, useComputed$ } from '@builder.io/qwik';
import type { PBDeepGraphData } from './PBDeepGraph';

// M3 Components - PBDeepCodeGraph
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';

interface PBDeepCodeGraphProps {
  graph?: PBDeepGraphData | null;
  height?: number;
}

const TYPE_COLORS: Record<string, string> = {
  root: '#f8fafc',
  branch: '#38bdf8',
  lost: '#f59e0b',
  untracked: '#f43f5e',
  drift: '#a3e635',
};

export const PBDeepCodeGraph = component$<PBDeepCodeGraphProps>((props) => {
  const layout = useComputed$(() => {
    const graph = props.graph ?? { nodes: [], edges: [] };
    const width = 680;
    const height = props.height ?? 260;
    const centerX = width / 2;
    const centerY = height / 2;
    const positions = new Map<string, { x: number; y: number }>();
    const golden = Math.PI * (3 - Math.sqrt(5));

    graph.nodes.forEach((node, idx) => {
      if (node.type === 'root') {
        positions.set(node.id, { x: centerX, y: centerY });
        return;
      }
      const angle = idx * golden;
      const radius = 30 + Math.sqrt(idx + 1) * 22;
      const bias = node.type === 'branch' ? 1.2 : node.type === 'lost' ? 0.9 : 1.05;
      const x = centerX + Math.cos(angle) * radius * bias;
      const y = centerY + Math.sin(angle) * radius * bias;
      positions.set(node.id, { x, y });
    });

    const edges = graph.edges
      .map((edge) => {
        const from = positions.get(edge.source);
        const to = positions.get(edge.target);
        if (!from || !to) return null;
        return { from, to };
      })
      .filter(Boolean) as { from: { x: number; y: number }; to: { x: number; y: number } }[];

    const nodes = graph.nodes.map((node, idx) => {
      const pos = positions.get(node.id);
      const color = TYPE_COLORS[node.type] ?? '#94a3b8';
      const radius = Math.max(3.5, 3 + Math.sqrt(node.weight ?? 1) * 1.6);
      return { node, pos, color, radius, idx };
    });

    return { width, height, edges, nodes };
  });

  const nodeCount = props.graph?.nodes?.length ?? 0;
  const edgeCount = props.graph?.edges?.length ?? 0;

  return (
    <div class="glass-surface glass-surface-1 glass-gradient-border p-3">
      <div class="flex items-center justify-between pb-2">
        <div class="text-xs uppercase tracking-[0.2em] text-muted-foreground glass-chromatic-subtle">PBDEEP Codegraph</div>
        <div class="text-[10px] text-muted-foreground">nodes {nodeCount} | edges {edgeCount}</div>
      </div>
      <svg class="w-full block rounded-md bg-black/40" viewBox={`0 0 ${layout.value.width} ${layout.value.height}`} aria-label="PBDEEP codegraph">
        <g stroke="rgba(148, 163, 184, 0.35)" stroke-width="1.2">
          {layout.value.edges.map((edge, idx) => (
            <line
              key={`edge-${idx}`}
              x1={edge.from.x}
              y1={edge.from.y}
              x2={edge.to.x}
              y2={edge.to.y}
            />
          ))}
        </g>
        <g>
          {layout.value.nodes.map((item) => {
            if (!item.pos) return null;
            const label = item.node.type === 'root' || item.node.type === 'branch' ? item.node.label : '';
            return (
              <g key={item.node.id}>
                <circle cx={item.pos.x} cy={item.pos.y} r={item.radius} fill={item.color} opacity="0.85" />
                {label && (
                  <text
                    x={item.pos.x + item.radius + 6}
                    y={item.pos.y + 4}
                    font-size="10"
                    fill="rgba(226, 232, 240, 0.75)"
                  >
                    {label}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
});
