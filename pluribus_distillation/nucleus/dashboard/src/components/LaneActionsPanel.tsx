/**
 * LaneActionsPanel - Action buttons for lane control
 *
 * Phase 3, Iteration 16 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Pause/Resume lane execution
 * - Reassign to different agent
 * - Mark as blocked/unblocked
 * - Archive lane
 * - Emit bus events for all actions
 * - Confirmation modals for destructive actions
 */

import {
  component$,
  useSignal,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export type LaneAction =
  | 'pause'
  | 'resume'
  | 'reassign'
  | 'block'
  | 'unblock'
  | 'archive'
  | 'prioritize';

export interface LaneActionEvent {
  action: LaneAction;
  laneId: string;
  laneName: string;
  actor: string;
  timestamp: string;
  data?: Record<string, unknown>;
}

export interface LaneActionsPanelProps {
  /** Lane ID */
  laneId: string;
  /** Lane name for display */
  laneName: string;
  /** Current lane status */
  status: 'green' | 'yellow' | 'red';
  /** Current WIP percentage */
  wipPct: number;
  /** Current owner */
  owner: string;
  /** Whether lane is currently paused */
  isPaused?: boolean;
  /** Whether lane has blockers */
  hasBlockers?: boolean;
  /** Callback when action is triggered */
  onAction$?: QRL<(event: LaneActionEvent) => void>;
  /** Available agents for reassignment */
  availableAgents?: string[];
  /** Actor performing actions (current user/agent) */
  actor?: string;
  /** Compact mode (fewer buttons) */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getActionColor(action: LaneAction): string {
  switch (action) {
    case 'pause':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30 hover:bg-amber-500/30';
    case 'resume':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/30';
    case 'block':
      return 'bg-red-500/20 text-red-400 border-red-500/30 hover:bg-red-500/30';
    case 'unblock':
      return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30 hover:bg-cyan-500/30';
    case 'archive':
      return 'bg-rose-500/20 text-rose-400 border-rose-500/30 hover:bg-rose-500/30';
    case 'reassign':
      return 'bg-purple-500/20 text-purple-400 border-purple-500/30 hover:bg-purple-500/30';
    case 'prioritize':
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30 hover:bg-blue-500/30';
    default:
      return 'bg-muted/20 text-muted-foreground border-border/30 hover:bg-muted/40';
  }
}

function getActionIcon(action: LaneAction): string {
  switch (action) {
    case 'pause': return '⏸';
    case 'resume': return '▶';
    case 'block': return '🚫';
    case 'unblock': return '✓';
    case 'archive': return '📦';
    case 'reassign': return '👤';
    case 'prioritize': return '⬆';
    default: return '•';
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneActionsPanel = component$<LaneActionsPanelProps>(({
  laneId,
  laneName,
  status,
  wipPct,
  owner,
  isPaused = false,
  hasBlockers = false,
  onAction$,
  availableAgents = [],
  actor = 'dashboard',
  compact = false,
}) => {
  // State
  const showConfirmModal = useSignal(false);
  const pendingAction = useSignal<LaneAction | null>(null);
  const showReassignDropdown = useSignal(false);
  const selectedAgent = useSignal<string>('');

  // Create action event
  const createEvent = (action: LaneAction, data?: Record<string, unknown>): LaneActionEvent => ({
    action,
    laneId,
    laneName,
    actor,
    timestamp: new Date().toISOString(),
    data,
  });

  // Execute action
  const executeAction = $(async (action: LaneAction, data?: Record<string, unknown>) => {
    const event = createEvent(action, data);

    // Emit to bus (simulated - would use actual bus client)
    console.log(`[LaneActionsPanel] Emitting: operator.lanes.${action}`, event);

    // Call callback if provided
    if (onAction$) {
      await onAction$(event);
    }

    // Reset state
    showConfirmModal.value = false;
    pendingAction.value = null;
    showReassignDropdown.value = false;
  });

  // Request confirmation for destructive actions
  const requestConfirmation = $((action: LaneAction) => {
    pendingAction.value = action;
    showConfirmModal.value = true;
  });

  // Handle action button click
  const handleAction = $((action: LaneAction) => {
    // Actions requiring confirmation
    if (['archive', 'block'].includes(action)) {
      requestConfirmation(action);
      return;
    }

    // Reassign needs dropdown
    if (action === 'reassign') {
      showReassignDropdown.value = !showReassignDropdown.value;
      return;
    }

    // Execute immediately
    executeAction(action);
  });

  // Confirm pending action
  const confirmAction = $(() => {
    if (pendingAction.value) {
      executeAction(pendingAction.value);
    }
  });

  // Cancel confirmation
  const cancelConfirmation = $(() => {
    showConfirmModal.value = false;
    pendingAction.value = null;
  });

  // Handle reassignment
  const handleReassign = $(() => {
    if (selectedAgent.value && selectedAgent.value !== owner) {
      executeAction('reassign', {
        previousOwner: owner,
        newOwner: selectedAgent.value
      });
    }
    showReassignDropdown.value = false;
  });

  return (
    <div class="rounded-lg border border-border bg-card p-3">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">ACTIONS</span>
          <span class={`text-[10px] px-1.5 py-0.5 rounded border ${
            status === 'green' ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' :
            status === 'yellow' ? 'bg-amber-500/20 text-amber-400 border-amber-500/30' :
            'bg-red-500/20 text-red-400 border-red-500/30'
          }`}>
            {laneName}
          </span>
        </div>
        <span class="text-[9px] text-muted-foreground">
          @{owner} · {wipPct}%
        </span>
      </div>

      {/* Action buttons */}
      <div class={`grid gap-2 ${compact ? 'grid-cols-3' : 'grid-cols-2 sm:grid-cols-4'}`}>
        {/* Pause/Resume */}
        <button
          onClick$={() => handleAction(isPaused ? 'resume' : 'pause')}
          class={`flex items-center justify-center gap-1.5 px-3 py-2 rounded border text-[10px] font-medium transition-colors ${
            getActionColor(isPaused ? 'resume' : 'pause')
          }`}
        >
          <span>{getActionIcon(isPaused ? 'resume' : 'pause')}</span>
          <span>{isPaused ? 'Resume' : 'Pause'}</span>
        </button>

        {/* Block/Unblock */}
        <button
          onClick$={() => handleAction(hasBlockers ? 'unblock' : 'block')}
          class={`flex items-center justify-center gap-1.5 px-3 py-2 rounded border text-[10px] font-medium transition-colors ${
            getActionColor(hasBlockers ? 'unblock' : 'block')
          }`}
        >
          <span>{getActionIcon(hasBlockers ? 'unblock' : 'block')}</span>
          <span>{hasBlockers ? 'Unblock' : 'Block'}</span>
        </button>

        {/* Reassign */}
        <button
          onClick$={() => handleAction('reassign')}
          class={`flex items-center justify-center gap-1.5 px-3 py-2 rounded border text-[10px] font-medium transition-colors ${
            getActionColor('reassign')
          } ${showReassignDropdown.value ? 'ring-1 ring-purple-400' : ''}`}
        >
          <span>{getActionIcon('reassign')}</span>
          <span>Reassign</span>
        </button>

        {/* Prioritize */}
        {!compact && (
          <button
            onClick$={() => handleAction('prioritize')}
            class={`flex items-center justify-center gap-1.5 px-3 py-2 rounded border text-[10px] font-medium transition-colors ${
              getActionColor('prioritize')
            }`}
          >
            <span>{getActionIcon('prioritize')}</span>
            <span>Priority</span>
          </button>
        )}

        {/* Archive (destructive) */}
        {!compact && (
          <button
            onClick$={() => handleAction('archive')}
            class={`flex items-center justify-center gap-1.5 px-3 py-2 rounded border text-[10px] font-medium transition-colors ${
              getActionColor('archive')
            }`}
          >
            <span>{getActionIcon('archive')}</span>
            <span>Archive</span>
          </button>
        )}
      </div>

      {/* Reassign dropdown */}
      {showReassignDropdown.value && (
        <div class="mt-3 p-2 rounded border border-purple-500/30 bg-purple-500/10">
          <div class="text-[10px] text-purple-400 mb-2">Reassign to:</div>
          <div class="flex gap-2">
            <select
              value={selectedAgent.value}
              onChange$={(e) => { selectedAgent.value = (e.target as HTMLSelectElement).value; }}
              class="flex-1 text-[10px] px-2 py-1.5 rounded bg-card border border-border/50 text-foreground"
            >
              <option value="">Select agent...</option>
              {availableAgents.filter(a => a !== owner).map(agent => (
                <option key={agent} value={agent}>@{agent}</option>
              ))}
            </select>
            <button
              onClick$={handleReassign}
              disabled={!selectedAgent.value}
              class="px-3 py-1.5 rounded bg-purple-500/30 text-purple-400 border border-purple-500/30 text-[10px] font-medium disabled:opacity-50 disabled:cursor-not-allowed hover:bg-purple-500/40 transition-colors"
            >
              Confirm
            </button>
            <button
              onClick$={() => { showReassignDropdown.value = false; }}
              class="px-3 py-1.5 rounded bg-muted/30 text-muted-foreground border border-border/30 text-[10px] hover:bg-muted/50 transition-colors"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Confirmation modal */}
      {showConfirmModal.value && pendingAction.value && (
        <div class="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div class="bg-card border border-border rounded-lg p-4 max-w-sm w-full mx-4 shadow-xl">
            <div class="text-sm font-semibold text-foreground mb-2">
              Confirm {pendingAction.value}?
            </div>
            <div class="text-xs text-muted-foreground mb-4">
              {pendingAction.value === 'archive' && (
                <>Are you sure you want to archive <strong>{laneName}</strong>? This will remove it from active lanes.</>
              )}
              {pendingAction.value === 'block' && (
                <>Are you sure you want to mark <strong>{laneName}</strong> as blocked? This will pause all work on this lane.</>
              )}
            </div>
            <div class="flex gap-2 justify-end">
              <button
                onClick$={cancelConfirmation}
                class="px-3 py-1.5 rounded bg-muted/30 text-muted-foreground border border-border/30 text-[10px] hover:bg-muted/50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick$={confirmAction}
                class={`px-3 py-1.5 rounded text-[10px] font-medium transition-colors ${
                  getActionColor(pendingAction.value)
                }`}
              >
                Confirm {pendingAction.value}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status bar */}
      <div class="mt-3 pt-2 border-t border-border/30 flex items-center justify-between text-[9px] text-muted-foreground">
        <span>
          {isPaused && <span class="text-amber-400 mr-2">⏸ Paused</span>}
          {hasBlockers && <span class="text-red-400 mr-2">🚫 Blocked</span>}
          {!isPaused && !hasBlockers && <span class="text-emerald-400">● Active</span>}
        </span>
        <span>Actor: @{actor}</span>
      </div>
    </div>
  );
});

export default LaneActionsPanel;
