/**
 * LaneKanbanBoard - Kanban board view for lanes
 *
 * Phase 4, Iteration 27 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Drag lanes between columns
 * - Columns: Backlog, Active, Review, Done
 * - Swim lanes by agent
 * - WIP limits per column
 * - Card details on hover
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export type KanbanColumn = 'backlog' | 'active' | 'review' | 'done';

export interface KanbanLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  column: KanbanColumn;
  priority?: number;
  blockers?: number;
  dueDate?: string;
}

export interface KanbanMoveEvent {
  laneId: string;
  fromColumn: KanbanColumn;
  toColumn: KanbanColumn;
  actor: string;
  timestamp: string;
}

export interface LaneKanbanBoardProps {
  /** Lanes to display */
  lanes: KanbanLane[];
  /** Callback when lane is moved */
  onMove$?: QRL<(event: KanbanMoveEvent) => void>;
  /** Callback when lane is clicked */
  onLaneClick$?: QRL<(lane: KanbanLane) => void>;
  /** WIP limits per column */
  wipLimits?: Partial<Record<KanbanColumn, number>>;
  /** Group by agent (swim lanes) */
  groupByAgent?: boolean;
  /** Current actor */
  actor?: string;
}

// ============================================================================
// Helpers
// ============================================================================

const COLUMNS: { id: KanbanColumn; label: string; color: string }[] = [
  { id: 'backlog', label: 'Backlog', color: 'border-gray-500/30' },
  { id: 'active', label: 'Active', color: 'border-blue-500/30' },
  { id: 'review', label: 'Review', color: 'border-amber-500/30' },
  { id: 'done', label: 'Done', color: 'border-emerald-500/30' },
];

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500/20 border-emerald-500/30';
    case 'yellow': return 'bg-amber-500/20 border-amber-500/30';
    case 'red': return 'bg-red-500/20 border-red-500/30';
    default: return 'bg-muted/20 border-border/30';
  }
}

function getColumnHeaderColor(column: KanbanColumn): string {
  switch (column) {
    case 'backlog': return 'bg-gray-500/20 text-gray-400';
    case 'active': return 'bg-blue-500/20 text-blue-400';
    case 'review': return 'bg-amber-500/20 text-amber-400';
    case 'done': return 'bg-emerald-500/20 text-emerald-400';
    default: return 'bg-muted/20 text-muted-foreground';
  }
}

function formatDueDate(dateStr: string): { text: string; urgent: boolean } {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = date.getTime() - now.getTime();
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffDays < 0) return { text: 'Overdue', urgent: true };
    if (diffDays === 0) return { text: 'Today', urgent: true };
    if (diffDays === 1) return { text: 'Tomorrow', urgent: true };
    if (diffDays < 7) return { text: `${diffDays}d`, urgent: false };
    return { text: date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), urgent: false };
  } catch {
    return { text: dateStr.slice(0, 10), urgent: false };
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneKanbanBoard = component$<LaneKanbanBoardProps>(({
  lanes: initialLanes,
  onMove$,
  onLaneClick$,
  wipLimits = {},
  groupByAgent = false,
  actor = 'dashboard',
}) => {
  // State
  const lanes = useSignal<KanbanLane[]>(initialLanes);
  const draggedLaneId = useSignal<string | null>(null);
  const dragOverColumn = useSignal<KanbanColumn | null>(null);
  const expandedCardId = useSignal<string | null>(null);

  // Computed: lanes by column
  const lanesByColumn = useComputed$(() => {
    const result: Record<KanbanColumn, KanbanLane[]> = {
      backlog: [],
      active: [],
      review: [],
      done: [],
    };

    lanes.value.forEach(lane => {
      result[lane.column].push(lane);
    });

    // Sort by priority within each column
    Object.keys(result).forEach(col => {
      result[col as KanbanColumn].sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999));
    });

    return result;
  });

  // Computed: lanes by agent (for swim lanes)
  const agents = useComputed$(() => {
    const agentSet = new Set<string>();
    lanes.value.forEach(lane => agentSet.add(lane.owner));
    return Array.from(agentSet).sort();
  });

  // Computed: column stats
  const columnStats = useComputed$(() => {
    const stats: Record<KanbanColumn, { count: number; overLimit: boolean }> = {
      backlog: { count: 0, overLimit: false },
      active: { count: 0, overLimit: false },
      review: { count: 0, overLimit: false },
      done: { count: 0, overLimit: false },
    };

    Object.entries(lanesByColumn.value).forEach(([col, colLanes]) => {
      const column = col as KanbanColumn;
      stats[column].count = colLanes.length;
      const limit = wipLimits[column];
      stats[column].overLimit = limit !== undefined && colLanes.length > limit;
    });

    return stats;
  });

  // Drag handlers
  const handleDragStart = $((laneId: string) => {
    draggedLaneId.value = laneId;
  });

  const handleDragOver = $((e: DragEvent, column: KanbanColumn) => {
    e.preventDefault();
    dragOverColumn.value = column;
  });

  const handleDragEnd = $(() => {
    draggedLaneId.value = null;
    dragOverColumn.value = null;
  });

  const handleDrop = $(async (targetColumn: KanbanColumn) => {
    if (!draggedLaneId.value) return;

    const laneIdx = lanes.value.findIndex(l => l.id === draggedLaneId.value);
    if (laneIdx === -1) return;

    const lane = lanes.value[laneIdx];
    const fromColumn = lane.column;

    if (fromColumn === targetColumn) {
      handleDragEnd();
      return;
    }

    // Update lane
    const updatedLanes = [...lanes.value];
    updatedLanes[laneIdx] = { ...lane, column: targetColumn };
    lanes.value = updatedLanes;

    // Emit event
    if (onMove$) {
      await onMove$({
        laneId: lane.id,
        fromColumn,
        toColumn: targetColumn,
        actor,
        timestamp: new Date().toISOString(),
      });
    }

    handleDragEnd();
  });

  // Render lane card
  const renderCard = $((lane: KanbanLane) => {
    const isDragging = draggedLaneId.value === lane.id;
    const isExpanded = expandedCardId.value === lane.id;
    const dueInfo = lane.dueDate ? formatDueDate(lane.dueDate) : null;

    return (
      <div
        key={lane.id}
        draggable
        onDragStart$={() => handleDragStart(lane.id)}
        onDragEnd$={handleDragEnd}
        onClick$={() => {
          expandedCardId.value = isExpanded ? null : lane.id;
          if (onLaneClick$) onLaneClick$(lane);
        }}
        class={`p-3 rounded-lg border cursor-grab active:cursor-grabbing transition-all ${
          getStatusColor(lane.status)
        } ${
          isDragging ? 'opacity-50 scale-95' : ''
        } hover:border-primary/50`}
      >
        {/* Card header */}
        <div class="flex items-start justify-between gap-2">
          <div class="flex-grow min-w-0">
            <div class="text-xs font-medium text-foreground truncate">{lane.name}</div>
            <div class="text-[9px] text-muted-foreground">@{lane.owner}</div>
          </div>
          <div class="flex-shrink-0 flex flex-col items-end gap-1">
            <span class={`text-[10px] font-bold ${
              lane.wip_pct >= 90 ? 'text-emerald-400' :
              lane.wip_pct >= 50 ? 'text-cyan-400' :
              lane.wip_pct >= 20 ? 'text-amber-400' :
              'text-red-400'
            }`}>
              {lane.wip_pct}%
            </span>
            {lane.blockers && lane.blockers > 0 && (
              <span class="text-[9px] px-1 py-0.5 rounded bg-red-500/30 text-red-400">
                🚫 {lane.blockers}
              </span>
            )}
          </div>
        </div>

        {/* Progress bar */}
        <div class="mt-2 h-1 rounded-full bg-muted/30 overflow-hidden">
          <div
            class={`h-full rounded-full ${
              lane.status === 'green' ? 'bg-emerald-400' :
              lane.status === 'yellow' ? 'bg-amber-400' :
              'bg-red-400'
            }`}
            style={{ width: `${lane.wip_pct}%` }}
          />
        </div>

        {/* Footer */}
        <div class="mt-2 flex items-center justify-between">
          {dueInfo && (
            <span class={`text-[9px] ${dueInfo.urgent ? 'text-red-400' : 'text-muted-foreground'}`}>
              📅 {dueInfo.text}
            </span>
          )}
          {lane.priority !== undefined && (
            <span class="text-[9px] text-muted-foreground">
              P{lane.priority}
            </span>
          )}
        </div>

        {/* Expanded details */}
        {isExpanded && (
          <div class="mt-3 pt-3 border-t border-border/30 text-[9px] space-y-1">
            <div class="flex justify-between">
              <span class="text-muted-foreground">Status:</span>
              <span class="text-foreground">{lane.status}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted-foreground">Column:</span>
              <span class="text-foreground">{lane.column}</span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted-foreground">ID:</span>
              <span class="text-foreground font-mono">{lane.id.slice(0, 12)}...</span>
            </div>
          </div>
        )}
      </div>
    );
  });

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">KANBAN BOARD</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {lanes.value.length} lanes
          </span>
        </div>
        <div class="text-[9px] text-muted-foreground">
          Drag cards to move between columns
        </div>
      </div>

      {/* Board */}
      <div class="flex gap-3 p-3 overflow-x-auto min-h-[400px]">
        {COLUMNS.map(column => {
          const columnLanes = lanesByColumn.value[column.id];
          const stats = columnStats.value[column.id];
          const isDropTarget = dragOverColumn.value === column.id && draggedLaneId.value;
          const limit = wipLimits[column.id];

          return (
            <div
              key={column.id}
              onDragOver$={(e) => handleDragOver(e, column.id)}
              onDrop$={() => handleDrop(column.id)}
              class={`flex-1 min-w-[220px] rounded-lg border ${column.color} ${
                isDropTarget ? 'bg-primary/10 border-primary/50' : 'bg-muted/5'
              } transition-colors`}
            >
              {/* Column header */}
              <div class={`p-2 rounded-t-lg ${getColumnHeaderColor(column.id)}`}>
                <div class="flex items-center justify-between">
                  <span class="text-xs font-semibold">{column.label}</span>
                  <div class="flex items-center gap-1">
                    <span class={`text-[10px] px-1.5 py-0.5 rounded ${
                      stats.overLimit ? 'bg-red-500/30 text-red-400' : 'bg-black/20'
                    }`}>
                      {stats.count}{limit !== undefined ? `/${limit}` : ''}
                    </span>
                  </div>
                </div>
              </div>

              {/* Column content */}
              <div class="p-2 space-y-2 max-h-[500px] overflow-y-auto">
                {groupByAgent ? (
                  // Grouped by agent
                  agents.value.map(agentId => {
                    const agentLanes = columnLanes.filter(l => l.owner === agentId);
                    if (agentLanes.length === 0) return null;

                    return (
                      <div key={agentId} class="space-y-2">
                        <div class="text-[9px] text-muted-foreground font-medium px-1">
                          @{agentId} ({agentLanes.length})
                        </div>
                        {agentLanes.map(lane => renderCard(lane))}
                      </div>
                    );
                  })
                ) : (
                  // Flat list
                  columnLanes.map(lane => renderCard(lane))
                )}

                {columnLanes.length === 0 && (
                  <div class="p-4 text-center text-[10px] text-muted-foreground/50">
                    Drop lanes here
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 flex items-center justify-between text-[9px] text-muted-foreground">
        <div class="flex items-center gap-4">
          {COLUMNS.map(col => (
            <span key={col.id}>
              {col.label}: {columnStats.value[col.id].count}
            </span>
          ))}
        </div>
        {groupByAgent && <span>{agents.value.length} agents</span>}
      </div>
    </div>
  );
});

export default LaneKanbanBoard;
