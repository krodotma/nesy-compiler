/**
 * NextActionsEditor - Manage next actions for lanes
 *
 * Phase 3, Iteration 20 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Add/edit/remove next action items
 * - Reorder actions via drag-and-drop
 * - Mark actions as done
 * - Quick-add with keyboard shortcuts
 * - Estimate effort (small/medium/large)
 * - Emit bus events for action changes
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

export type ActionEffort = 'small' | 'medium' | 'large';
export type ActionStatus = 'pending' | 'in_progress' | 'done';

export interface NextAction {
  id: string;
  title: string;
  description?: string;
  effort: ActionEffort;
  status: ActionStatus;
  createdAt: string;
  completedAt?: string;
  assignedTo?: string;
}

export interface ActionEvent {
  type: 'add' | 'update' | 'complete' | 'delete' | 'reorder';
  laneId: string;
  action?: NextAction;
  actions?: NextAction[];
  actor: string;
  timestamp: string;
}

export interface NextActionsEditorProps {
  /** Lane ID */
  laneId: string;
  /** Lane name for display */
  laneName: string;
  /** Current actions */
  actions: NextAction[];
  /** Callback when actions change */
  onActionChange$?: QRL<(event: ActionEvent) => void>;
  /** Actor performing changes */
  actor?: string;
  /** Compact mode (inline) */
  compact?: boolean;
  /** Max visible actions before scrolling */
  maxVisible?: number;
}

// ============================================================================
// Helpers
// ============================================================================

function getEffortColor(effort: ActionEffort): string {
  switch (effort) {
    case 'small': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'large': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getEffortIcon(effort: ActionEffort): string {
  switch (effort) {
    case 'small': return 'S';
    case 'medium': return 'M';
    case 'large': return 'L';
    default: return '?';
  }
}

function getStatusColor(status: ActionStatus): string {
  switch (status) {
    case 'pending': return 'border-muted-foreground/30';
    case 'in_progress': return 'border-amber-400 bg-amber-400/10';
    case 'done': return 'border-emerald-400 bg-emerald-400/10';
    default: return 'border-border/30';
  }
}

function generateId(): string {
  return `act-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

// ============================================================================
// Component
// ============================================================================

export const NextActionsEditor = component$<NextActionsEditorProps>(({
  laneId,
  laneName,
  actions: initialActions,
  onActionChange$,
  actor = 'dashboard',
  compact = false,
  maxVisible = 5,
}) => {
  // State
  const actions = useSignal<NextAction[]>(initialActions);
  const showAddForm = useSignal(false);
  const editingId = useSignal<string | null>(null);
  const draggedIndex = useSignal<number | null>(null);
  const dragOverIndex = useSignal<number | null>(null);
  const showCompleted = useSignal(false);

  // New action form state
  const newTitle = useSignal('');
  const newDescription = useSignal('');
  const newEffort = useSignal<ActionEffort>('medium');

  // Computed
  const pendingActions = useComputed$(() =>
    actions.value.filter(a => a.status !== 'done').sort((a, b) => {
      // In progress first, then pending
      if (a.status === 'in_progress' && b.status !== 'in_progress') return -1;
      if (a.status !== 'in_progress' && b.status === 'in_progress') return 1;
      return 0;
    })
  );

  const completedActions = useComputed$(() =>
    actions.value.filter(a => a.status === 'done')
  );

  const stats = useComputed$(() => ({
    total: actions.value.length,
    pending: actions.value.filter(a => a.status === 'pending').length,
    inProgress: actions.value.filter(a => a.status === 'in_progress').length,
    done: actions.value.filter(a => a.status === 'done').length,
  }));

  // Emit event helper
  const emitEvent = $(async (type: ActionEvent['type'], action?: NextAction, allActions?: NextAction[]) => {
    const event: ActionEvent = {
      type,
      laneId,
      action,
      actions: allActions,
      actor,
      timestamp: new Date().toISOString(),
    };

    console.log(`[NextActionsEditor] Emitting: operator.lanes.action.${type}`, event);

    if (onActionChange$) {
      await onActionChange$(event);
    }
  });

  // Add new action
  const addAction = $(async () => {
    if (!newTitle.value.trim()) return;

    const newAction: NextAction = {
      id: generateId(),
      title: newTitle.value.trim(),
      description: newDescription.value.trim() || undefined,
      effort: newEffort.value,
      status: 'pending',
      createdAt: new Date().toISOString(),
    };

    actions.value = [...actions.value, newAction];

    await emitEvent('add', newAction);

    // Reset form
    newTitle.value = '';
    newDescription.value = '';
    newEffort.value = 'medium';
    showAddForm.value = false;
  });

  // Quick add (just title)
  const quickAdd = $(async () => {
    if (!newTitle.value.trim()) return;

    const newAction: NextAction = {
      id: generateId(),
      title: newTitle.value.trim(),
      effort: 'medium',
      status: 'pending',
      createdAt: new Date().toISOString(),
    };

    actions.value = [...actions.value, newAction];
    await emitEvent('add', newAction);
    newTitle.value = '';
  });

  // Update action status
  const updateStatus = $(async (actionId: string, newStatus: ActionStatus) => {
    const idx = actions.value.findIndex(a => a.id === actionId);
    if (idx === -1) return;

    const updated = { ...actions.value[idx], status: newStatus };
    if (newStatus === 'done') {
      updated.completedAt = new Date().toISOString();
    }

    const newActions = [...actions.value];
    newActions[idx] = updated;
    actions.value = newActions;

    await emitEvent(newStatus === 'done' ? 'complete' : 'update', updated);
  });

  // Delete action
  const deleteAction = $(async (actionId: string) => {
    const action = actions.value.find(a => a.id === actionId);
    if (!action) return;

    actions.value = actions.value.filter(a => a.id !== actionId);
    await emitEvent('delete', action);
  });

  // Drag handlers
  const handleDragStart = $((index: number) => {
    draggedIndex.value = index;
  });

  const handleDragOver = $((e: DragEvent, index: number) => {
    e.preventDefault();
    if (draggedIndex.value !== null && draggedIndex.value !== index) {
      dragOverIndex.value = index;
    }
  });

  const handleDragEnd = $(() => {
    draggedIndex.value = null;
    dragOverIndex.value = null;
  });

  const handleDrop = $(async (targetIndex: number) => {
    if (draggedIndex.value === null || draggedIndex.value === targetIndex) {
      handleDragEnd();
      return;
    }

    const pending = [...pendingActions.value];
    const [dragged] = pending.splice(draggedIndex.value, 1);
    pending.splice(targetIndex, 0, dragged);

    // Reconstruct full actions list with reordered pending + completed
    actions.value = [...pending, ...completedActions.value];

    await emitEvent('reorder', undefined, actions.value);
    handleDragEnd();
  });

  // Handle keyboard shortcut (Enter to add)
  const handleKeyDown = $((e: KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && newTitle.value.trim()) {
      e.preventDefault();
      quickAdd();
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">NEXT ACTIONS</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {stats.value.pending + stats.value.inProgress} pending
          </span>
          {stats.value.inProgress > 0 && (
            <span class="text-[10px] px-2 py-0.5 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30">
              {stats.value.inProgress} in progress
            </span>
          )}
        </div>
        <button
          onClick$={() => { showAddForm.value = !showAddForm.value; }}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
        >
          + Add Action
        </button>
      </div>

      {/* Quick add bar (always visible in compact mode) */}
      {compact && (
        <div class="p-2 border-b border-border/30 flex gap-2">
          <input
            type="text"
            placeholder="Quick add action (Enter to add)..."
            value={newTitle.value}
            onInput$={(e) => { newTitle.value = (e.target as HTMLInputElement).value; }}
            onKeyDown$={handleKeyDown}
            class="flex-1 px-2 py-1.5 text-[10px] rounded bg-muted/10 border border-border/30 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
          />
          <button
            onClick$={quickAdd}
            disabled={!newTitle.value.trim()}
            class="px-3 py-1.5 text-[10px] rounded bg-primary text-primary-foreground font-medium disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Add
          </button>
        </div>
      )}

      {/* Add form (expanded) */}
      {showAddForm.value && !compact && (
        <div class="p-3 border-b border-border/30 bg-muted/10 space-y-3">
          <div class="text-[10px] font-medium text-foreground">Add Next Action</div>

          {/* Title */}
          <input
            type="text"
            placeholder="Action title..."
            value={newTitle.value}
            onInput$={(e) => { newTitle.value = (e.target as HTMLInputElement).value; }}
            onKeyDown$={handleKeyDown}
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50"
          />

          {/* Description */}
          <textarea
            placeholder="Description (optional)..."
            value={newDescription.value}
            onInput$={(e) => { newDescription.value = (e.target as HTMLTextAreaElement).value; }}
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 resize-none"
            rows={2}
          />

          {/* Effort */}
          <div>
            <label class="text-[9px] text-muted-foreground block mb-1">Effort Estimate</label>
            <div class="flex gap-2">
              {(['small', 'medium', 'large'] as const).map(effort => (
                <button
                  key={effort}
                  onClick$={() => { newEffort.value = effort; }}
                  class={`flex-1 px-2 py-1.5 text-[10px] rounded border transition-colors ${
                    newEffort.value === effort
                      ? getEffortColor(effort)
                      : 'bg-muted/10 text-muted-foreground border-border/30 hover:bg-muted/20'
                  }`}
                >
                  {getEffortIcon(effort)} {effort}
                </button>
              ))}
            </div>
          </div>

          {/* Actions */}
          <div class="flex gap-2 justify-end">
            <button
              onClick$={() => { showAddForm.value = false; }}
              class="px-3 py-1.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick$={addAction}
              disabled={!newTitle.value.trim()}
              class="px-3 py-1.5 text-[10px] rounded bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add Action
            </button>
          </div>
        </div>
      )}

      {/* Action list */}
      <div class={`overflow-y-auto ${compact ? 'max-h-[150px]' : `max-h-[${maxVisible * 60}px]`}`}>
        {pendingActions.value.length === 0 ? (
          <div class="p-6 text-center">
            <div class="text-2xl mb-2">✓</div>
            <div class="text-xs text-muted-foreground">No pending actions</div>
          </div>
        ) : (
          pendingActions.value.map((action, index) => (
            <div
              key={action.id}
              draggable={!compact}
              onDragStart$={() => handleDragStart(index)}
              onDragOver$={(e) => handleDragOver(e, index)}
              onDragEnd$={handleDragEnd}
              onDrop$={() => handleDrop(index)}
              class={`flex items-start gap-2 p-3 border-b border-border/30 transition-all ${
                !compact ? 'cursor-grab active:cursor-grabbing' : ''
              } ${
                draggedIndex.value === index ? 'opacity-50 bg-primary/10' : ''
              } ${
                dragOverIndex.value === index ? 'bg-primary/20 border-primary/50' : ''
              } hover:bg-muted/10`}
            >
              {/* Checkbox */}
              <button
                onClick$={() => updateStatus(action.id, action.status === 'done' ? 'pending' : 'done')}
                class={`w-4 h-4 rounded border flex-shrink-0 flex items-center justify-center transition-colors ${getStatusColor(action.status)}`}
              >
                {action.status === 'done' && (
                  <span class="text-emerald-400 text-[10px]">✓</span>
                )}
                {action.status === 'in_progress' && (
                  <span class="text-amber-400 text-[8px]">▶</span>
                )}
              </button>

              {/* Content */}
              <div class="flex-grow min-w-0">
                <div class={`text-xs ${action.status === 'done' ? 'text-muted-foreground line-through' : 'text-foreground'}`}>
                  {action.title}
                </div>
                {action.description && (
                  <div class="text-[9px] text-muted-foreground mt-0.5 truncate">
                    {action.description}
                  </div>
                )}
                <div class="flex items-center gap-2 mt-1">
                  <span class={`text-[8px] px-1.5 py-0.5 rounded border ${getEffortColor(action.effort)}`}>
                    {getEffortIcon(action.effort)}
                  </span>
                  {action.status === 'in_progress' && (
                    <span class="text-[8px] text-amber-400">In Progress</span>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div class="flex-shrink-0 flex items-center gap-1">
                {action.status === 'pending' && (
                  <button
                    onClick$={() => updateStatus(action.id, 'in_progress')}
                    class="w-5 h-5 flex items-center justify-center rounded bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors text-[10px]"
                    title="Start"
                  >
                    ▶
                  </button>
                )}
                {action.status === 'in_progress' && (
                  <button
                    onClick$={() => updateStatus(action.id, 'pending')}
                    class="w-5 h-5 flex items-center justify-center rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors text-[10px]"
                    title="Pause"
                  >
                    ⏸
                  </button>
                )}
                <button
                  onClick$={() => deleteAction(action.id)}
                  class="w-5 h-5 flex items-center justify-center rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors text-[10px]"
                  title="Delete"
                >
                  ×
                </button>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Completed toggle */}
      {completedActions.value.length > 0 && (
        <div class="border-t border-border/30">
          <button
            onClick$={() => { showCompleted.value = !showCompleted.value; }}
            class="w-full p-2 text-[9px] text-muted-foreground hover:bg-muted/10 flex items-center justify-center gap-1 transition-colors"
          >
            <span>{showCompleted.value ? '▼' : '▶'}</span>
            <span>{completedActions.value.length} completed</span>
          </button>

          {showCompleted.value && (
            <div class="max-h-[100px] overflow-y-auto bg-muted/5">
              {completedActions.value.map(action => (
                <div
                  key={action.id}
                  class="flex items-center gap-2 px-3 py-2 border-b border-border/20 opacity-60"
                >
                  <span class="text-emerald-400 text-[10px]">✓</span>
                  <span class="text-[10px] text-muted-foreground line-through flex-grow truncate">
                    {action.title}
                  </span>
                  <button
                    onClick$={() => updateStatus(action.id, 'pending')}
                    class="text-[9px] text-muted-foreground hover:text-foreground"
                    title="Reopen"
                  >
                    ↺
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {stats.value.done}/{stats.value.total} completed • Lane: {laneName}
      </div>
    </div>
  );
});

export default NextActionsEditor;
