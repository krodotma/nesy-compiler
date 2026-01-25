/**
 * Git Graph Visualizer (Neon DAG)
 * Renders the commit history as a multi-lane directed acyclic graph.
 */
import { component$, useComputed$, $ } from '@builder.io/qwik';

// M3 Components - GitGraph
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';

interface Commit {
  sha: string;
  message: string;
  author: string;
  date: string;
  parents: string[];
}

interface Props {
  commits: Commit[];
  onSelectCommit$: (sha: string) => void;
  selectedSha?: string | null;
}

const COLORS = [
  '#06b6d4', // Cyan (Primary)
  '#a855f7', // Purple
  '#22c55e', // Green
  '#f59e0b', // Orange
  '#ef4444', // Red
  '#ec4899', // Pink
  '#3b82f6', // Blue
  '#eab308', // Yellow
];

export const GitGraph = component$<Props>(({ commits, onSelectCommit$, selectedSha }) => {
  // Compute graph layout (lanes)
  const graph = useComputed$(() => {
    const nodes: any[] = [];
    const edges: any[] = [];
    const lanes: (string | null)[] = []; // Occupied lanes (commit shas)
    
    // Map sha -> { row, lane }
    const pos = new Map<string, { row: number, lane: number }>();

    commits.forEach((commit, row) => {
      // 1. Assign Lane
      let lane = -1;

      // Check if any existing lane is pointing to us (we are the parent of a previous commit)
      // Wait, git log is usually newest first. So we are the *parent* of a previous row.
      // So if a previous row's commit listed us as a parent, that lane continues to us.
      
      // Let's look at previous rows to find who points to us.
      // Actually, standard DAG algo for git log (newest -> oldest):
      // Keep track of "active branches" flowing down.
      
      // Simplified approach:
      // Try to continue the lane of our first child (the commit that listed us as parent[0]).
      // Since we iterate newest to oldest, we check the *active lanes*.
      
      // Let's track active parents in lanes.
      // lanes[i] = "sha_of_parent_we_are_hunting"
      
      // When we process a commit, we check if we match any active lane target.
      
      // Initialize lanes if empty (first commit processed)
      // But we don't know children yet.
      
      // Better approach: Topological sort is already roughly done by git log.
      // We just need visual stability.
      
      // Algorithm:
      // lanes[] array of active parent SHAs looking for a home.
      // For each commit C:
      //   1. Check if C.sha is in lanes[]. 
      //      If yes, take that lane. (If multiple, merge logic).
      //      If no, allocate new lane (new branch tip).
      //   2. Update lanes[]: replace C.sha with C.parents.
      //      If multiple parents, fork lanes.
      
      // Find our lane
      let laneIdx = lanes.indexOf(commit.sha);
      
      if (laneIdx === -1) {
        // New branch tip or disjoint history
        // Find first empty lane or append
        laneIdx = lanes.findIndex(l => l === null);
        if (laneIdx === -1) laneIdx = lanes.length;
      }
      
      lane = laneIdx;
      lanes[lane] = null; // We occupied it
      
      // Register position
      nodes.push({ commit, row, lane });
      pos.set(commit.sha, { row, lane });

      // Propagate parents to lanes
      commit.parents.forEach((pSha, idx) => {
        if (idx === 0) {
          // Primary parent inherits our lane (or fills it back in)
          if (lanes[lane] === null) {
             lanes[lane] = pSha;
          } else {
             // Lane occupied (merge case?), find another
             // This is simplistic but works for basic graphs
             let sub = lanes.indexOf(pSha);
             if (sub === -1) {
                 // Find empty
                 let empty = lanes.findIndex(l => l === null);
                 if (empty === -1) empty = lanes.length;
                 lanes[empty] = pSha;
             }
          }
        } else {
          // Secondary parents (merge sources) get their own lanes
          // Check if already tracked
          if (!lanes.includes(pSha)) {
             let empty = lanes.findIndex(l => l === null);
             if (empty === -1) empty = lanes.length;
             lanes[empty] = pSha;
          }
        }
        
        // We defer edge drawing until we know parent positions? 
        // No, we can draw edges backwards if we know positions, but we assume parents are below.
        // We'll calculate edges in a second pass or on-the-fly if we only need start/end lanes.
        // For SVG, we need start (x1, y1) and end (x2, y2).
        // Since we iterate down, we know (x1, y1) = (lane, row).
        // We don't know (x2, y2) yet.
        // So we record the *requirement* for an edge: from (lane, row) to pSha.
      });
    });

    // Pass 2: Generate Edges
    nodes.forEach(node => {
        node.commit.parents.forEach((pSha: string) => {
            const parentPos = pos.get(pSha);
            if (parentPos) {
                edges.push({
                    x1: node.lane, y1: node.row,
                    x2: parentPos.lane, y2: parentPos.row,
                    color: COLORS[node.lane % COLORS.length]
                });
            } else {
                // Parent off-screen or too deep. Draw trailing edge?
                edges.push({
                    x1: node.lane, y1: node.row,
                    x2: node.lane, y2: node.row + 1, // Just down
                    color: COLORS[node.lane % COLORS.length],
                    fade: true
                });
            }
        });
    });

    return { nodes, edges, height: commits.length * 50 };
  });

  return (
    <div class="relative pl-4 h-full">
      {/* SVG Layer for lines */}
      <svg 
        class="absolute top-0 left-0 w-full h-full pointer-events-none overflow-visible"
        style={{ height: `${Math.max(100, graph.value.height)}px` }}
      >
        {graph.value.edges.map((e, i) => {
            // Bezier curve
            const x1 = 20 + e.x1 * 24;
            const y1 = 35 + e.y1 * 72;
            const x2 = 20 + e.x2 * 24;
            const y2 = 35 + e.y2 * 72;
            
            // If same lane, straight line
            if (e.x1 === e.x2) {
                return <line key={`e-${i}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke={e.color} stroke-width="2" opacity="0.6" />;
            }
            
            // Curved connection
            const midY = (y1 + y2) / 2;
            return (
                <path 
                    key={`e-${i}`}
                    d={`M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`}
                    fill="none"
                    stroke={e.color}
                    stroke-width="2"
                    opacity="0.6"
                />
            );
        })}
        
        {graph.value.nodes.map((n, i) => (
             <circle 
                key={`n-${n.commit.sha}`} 
                cx={20 + n.lane * 24} 
                cy={35 + n.row * 72} 
                r="4" 
                fill={COLORS[n.lane % COLORS.length]} 
                class="stroke-background stroke-2"
             />
        ))}
      </svg>

      {/* Commit List */}
      <div class="space-y-4 pt-2">
        {graph.value.nodes.map(({ commit, lane }) => (
          <div 
            key={commit.sha}
            onClick$={() => onSelectCommit$(commit.sha)}
            class={`
              relative p-3 ml-[${40 + lane * 24}px] rounded-lg border transition-all cursor-pointer group
              ${selectedSha === commit.sha 
                ? 'bg-primary/10 border-primary shadow-[0_0_15px_-5px_rgba(6,182,212,0.5)] translate-x-2' 
                : 'bg-card border-border/50 hover:bg-muted/30 hover:border-primary/30'}
            `}
            style={{ marginLeft: `${30 + lane * 24}px` }}
          >
            <div class="flex items-start justify-between gap-4">
              <div class="min-w-0">
                <div class="font-medium text-sm truncate pr-2 text-foreground/90 group-hover:text-primary transition-colors">
                  {commit.message}
                </div>
                <div class="flex items-center gap-3 mt-1.5 text-xs text-muted-foreground">
                  <span class="flex items-center gap-1">
                    <span 
                        class="w-4 h-4 rounded-full flex items-center justify-center text-[8px] text-white font-bold"
                        style={{ backgroundColor: COLORS[lane % COLORS.length] }}
                    >
                        {commit.author[0]?.toUpperCase() || '?'}
                    </span>
                    {commit.author}
                  </span>
                  <span>•</span>
                  <span>{commit.date.slice(0, 10)}</span>
                </div>
              </div>
              <div class="font-mono text-[10px] text-muted-foreground/70 bg-muted/30 px-1.5 py-0.5 rounded">
                {commit.sha.slice(0, 7)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});