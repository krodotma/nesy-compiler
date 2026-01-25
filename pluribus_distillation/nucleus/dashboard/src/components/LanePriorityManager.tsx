/**
 * LanePriorityManager - Drag-to-reorder lane priorities
 *
 * Phase 3, Iteration 18 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Drag and drop to reorder lanes
 * - Priority number input
 * - Visual priority indicators
 * - Emit bus events on priority change
 * - Keyboard accessible (up/down arrows)
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

interface Lane {
  id: string;
  name: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  owner: string;
  priority?: number;
}

export interface PriorityChangeEvent {
  laneId: string;
  oldPriority: number;
  newPriority: number;
  timestamp: string;
}

export interface LanePriorityManagerProps {
  /** Lanes to manage */
  lanes: Lane[];
  /** Callback when priority changes */
  onPriorityChange$?: QRL<(event: PriorityChangeEvent) => void>;
  /** Callback when order changes (provides new order) */
  onReorder$?: QRL<(orderedLaneIds: string[]) => void>;
  /** Enable drag and drop */
  enableDrag?: boolean;
  /** Show priority numbers */
  showNumbers?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function statusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'yellow': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'red': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function wipColor(pct: number): string {
  if (pct >= 90) return 'text-emerald-400';
  if (pct >= 60) return 'text-cyan-400';
  if (pct >= 30) return 'text-amber-400';
  return 'text-red-400';
}

// ============================================================================
// Component
// ============================================================================

export const LanePriorityManager = component$<LanePriorityManagerProps>(({
  lanes: initialLanes,
  onPriorityChange$,
  onReorder$,
  enableDrag = true,
  showNumbers = true,
}) => {
  // State
  const orderedLanes = useSignal<Lane[]>(
    [...initialLanes].sort((a, b) => (a.priority ?? 999) - (b.priority ?? 999))
  );
  const draggedIndex = useSignal<number | null>(null);
  const dragOverIndex = useSignal<number | null>(null);
  const editingId = useSignal<string | null>(null);
  const editingValue = useSignal<string>('');

  // Computed stats
  const stats = useComputed$(() => ({
    total: orderedLanes.value.length,
    highPriority: orderedLanes.value.filter((_, i) => i < 3).length,
  }));

  // Handle drag start
  const handleDragStart = $((index: number) => {
    if (!enableDrag) return;
    draggedIndex.value = index;
  });

  // Handle drag over
  const handleDragOver = $((e: DragEvent, index: number) => {
    e.preventDefault();
    if (draggedIndex.value !== null && draggedIndex.value !== index) {
      dragOverIndex.value = index;
    }
  });

  // Handle drag end
  const handleDragEnd = $(() => {
    draggedIndex.value = null;
    dragOverIndex.value = null;
  });

  // Handle drop
  const handleDrop = $(async (targetIndex: number) => {
    if (draggedIndex.value === null || draggedIndex.value === targetIndex) {
      handleDragEnd();
      return;
    }

    const lanes = [...orderedLanes.value];
    const [draggedLane] = lanes.splice(draggedIndex.value, 1);
    lanes.splice(targetIndex, 0, draggedLane);

    // Update priorities
    const updatedLanes = lanes.map((lane, i) => ({
      ...lane,
      priority: i + 1,
    }));

    orderedLanes.value = updatedLanes;

    // Emit events
    if (onPriorityChange$) {
      await onPriorityChange$({
        laneId: draggedLane.id,
        oldPriority: draggedIndex.value + 1,
        newPriority: targetIndex + 1,
        timestamp: new Date().toISOString(),
      });
    }

    if (onReorder$) {
      await onReorder$(updatedLanes.map(l => l.id));
    }

    handleDragEnd();
  });

  // Handle keyboard navigation
  const handleKeyDown = $(async (e: KeyboardEvent, index: number) => {
    const lanes = orderedLanes.value;

    if (e.key === 'ArrowUp' && index > 0) {
      e.preventDefault();
      // Move up
      const newLanes = [...lanes];
      [newLanes[index - 1], newLanes[index]] = [newLanes[index], newLanes[index - 1]];
      const updated = newLanes.map((l, i) => ({ ...l, priority: i + 1 }));
      orderedLanes.value = updated;

      if (onPriorityChange$) {
        await onPriorityChange$({
          laneId: lanes[index].id,
          oldPriority: index + 1,
          newPriority: index,
          timestamp: new Date().toISOString(),
        });
      }
    } else if (e.key === 'ArrowDown' && index < lanes.length - 1) {
      e.preventDefault();
      // Move down
      const newLanes = [...lanes];
      [newLanes[index], newLanes[index + 1]] = [newLanes[index + 1], newLanes[index]];
      const updated = newLanes.map((l, i) => ({ ...l, priority: i + 1 }));
      orderedLanes.value = updated;

      if (onPriorityChange$) {
        await onPriorityChange$({
          laneId: lanes[index].id,
          oldPriority: index + 1,
          newPriority: index + 2,
          timestamp: new Date().toISOString(),
        });
      }
    }
  });

  // Start editing priority number
  const startEditing = $((laneId: string, currentPriority: number) => {
    editingId.value = laneId;
    editingValue.value = String(currentPriority);
  });

  // Save edited priority
  const saveEditing = $(async () => {
    if (!editingId.value) return;

    const newPriority = parseInt(editingValue.value, 10);
    if (isNaN(newPriority) || newPriority < 1) {
      editingId.value = null;
      return;
    }

    const lanes = [...orderedLanes.value];
    const currentIndex = lanes.findIndex(l => l.id === editingId.value);
    if (currentIndex === -1) {
      editingId.value = null;
      return;
    }

    const targetIndex = Math.min(Math.max(newPriority - 1, 0), lanes.length - 1);

    if (currentIndex !== targetIndex) {
      const [lane] = lanes.splice(currentIndex, 1);
      lanes.splice(targetIndex, 0, lane);

      const updated = lanes.map((l, i) => ({ ...l, priority: i + 1 }));
      orderedLanes.value = updated;

      if (onPriorityChange$) {
        await onPriorityChange$({
          laneId: editingId.value,
          oldPriority: currentIndex + 1,
          newPriority: targetIndex + 1,
          timestamp: new Date().toISOString(),
        });
      }

      if (onReorder$) {
        await onReorder$(updated.map(l => l.id));
      }
    }

    editingId.value = null;
  });

  // Cancel editing
  const cancelEditing = $(() => {
    editingId.value = null;
    editingValue.value = '';
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">PRIORITY ORDER</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {stats.value.total} lanes
          </span>
        </div>
        <div class="text-[9px] text-muted-foreground">
          {enableDrag ? 'Drag to reorder • ' : ''}
          ↑↓ keys to move
        </div>
      </div>

      {/* Lane list */}
      <div class="max-h-[400px] overflow-y-auto">
        {orderedLanes.value.map((lane, index) => (
          <div
            key={lane.id}
            draggable={enableDrag}
            onDragStart$={() => handleDragStart(index)}
            onDragOver$={(e) => handleDragOver(e, index)}
            onDragEnd$={handleDragEnd}
            onDrop$={() => handleDrop(index)}
            onKeyDown$={(e) => handleKeyDown(e, index)}
            tabIndex={0}
            class={`flex items-center gap-3 p-3 border-b border-border/30 transition-all ${
              enableDrag ? 'cursor-grab active:cursor-grabbing' : ''
            } ${
              draggedIndex.value === index ? 'opacity-50 bg-primary/10' : ''
            } ${
              dragOverIndex.value === index ? 'bg-primary/20 border-primary/50' : ''
            } ${
              index < 3 ? 'bg-gradient-to-r from-blue-500/5 to-transparent' : ''
            } hover:bg-muted/10 focus:outline-none focus:ring-1 focus:ring-primary/50`}
          >
            {/* Priority number */}
            {showNumbers && (
              <div class="flex-shrink-0 w-8">
                {editingId.value === lane.id ? (
                  <input
                    type="number"
                    min="1"
                    max={orderedLanes.value.length}
                    value={editingValue.value}
                    onChange$={(e) => { editingValue.value = (e.target as HTMLInputElement).value; }}
                    onKeyDown$={(e) => {
                      if (e.key === 'Enter') saveEditing();
                      if (e.key === 'Escape') cancelEditing();
                    }}
                    onBlur$={saveEditing}
                    class="w-8 h-6 text-[10px] text-center rounded bg-card border border-primary/50 text-foreground focus:outline-none"
                    autoFocus
                  />
                ) : (
                  <button
                    onClick$={() => startEditing(lane.id, index + 1)}
                    class={`w-6 h-6 text-[10px] font-bold rounded-full flex items-center justify-center ${
                      index < 3
                        ? 'bg-blue-500/30 text-blue-400 border border-blue-500/40'
                        : 'bg-muted/30 text-muted-foreground border border-border/30'
                    } hover:bg-primary/20 transition-colors`}
                    title="Click to edit priority"
                  >
                    {index + 1}
                  </button>
                )}
              </div>
            )}

            {/* Drag handle */}
            {enableDrag && (
              <div class="flex-shrink-0 text-muted-foreground/50 text-sm">
                ⋮⋮
              </div>
            )}

            {/* Lane info */}
            <div class="flex-grow min-w-0">
              <div class="flex items-center gap-2">
                <span class={`text-[10px] px-1.5 py-0.5 rounded border ${statusColor(lane.status)}`}>
                  {lane.status}
                </span>
                <span class="text-xs font-medium text-foreground truncate">
                  {lane.name}
                </span>
              </div>
              <div class="flex items-center gap-2 mt-0.5">
                <span class="text-[9px] text-muted-foreground">@{lane.owner}</span>
                <span class={`text-[9px] font-bold ${wipColor(lane.wip_pct)}`}>
                  {lane.wip_pct}%
                </span>
              </div>
            </div>

            {/* Quick actions */}
            <div class="flex-shrink-0 flex items-center gap-1">
              <button
                onClick$={() => handleKeyDown({ key: 'ArrowUp', preventDefault: () => {} } as KeyboardEvent, index)}
                disabled={index === 0}
                class="w-6 h-6 flex items-center justify-center rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-[10px]"
                title="Move up"
              >
                ▲
              </button>
              <button
                onClick$={() => handleKeyDown({ key: 'ArrowDown', preventDefault: () => {} } as KeyboardEvent, index)}
                disabled={index === orderedLanes.value.length - 1}
                class="w-6 h-6 flex items-center justify-center rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 disabled:opacity-30 disabled:cursor-not-allowed transition-colors text-[10px]"
                title="Move down"
              >
                ▼
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        Top 3 lanes highlighted as high priority
      </div>
    </div>
  );
});

export default LanePriorityManager;
