/**
 * BlockerManagement - Track and manage lane blockers
 *
 * Phase 3, Iteration 19 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Add/remove blockers with descriptions
 * - Categorize blockers (dependency, resource, external, technical)
 * - Set blocker severity (critical, high, medium, low)
 * - Track blocker age/duration
 * - Assign blocker resolution to agents
 * - Emit bus events for blocker state changes
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - BlockerManagement
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';

// ============================================================================
// Types
// ============================================================================

export type BlockerCategory = 'dependency' | 'resource' | 'external' | 'technical';
export type BlockerSeverity = 'critical' | 'high' | 'medium' | 'low';
export type BlockerStatus = 'open' | 'in_progress' | 'resolved';

export interface Blocker {
  id: string;
  description: string;
  category: BlockerCategory;
  severity: BlockerSeverity;
  status: BlockerStatus;
  createdAt: string;
  resolvedAt?: string;
  assignedTo?: string;
  notes?: string;
}

export interface BlockerEvent {
  type: 'add' | 'update' | 'resolve' | 'reopen';
  laneId: string;
  blocker: Blocker;
  actor: string;
  timestamp: string;
}

export interface BlockerManagementProps {
  /** Lane ID */
  laneId: string;
  /** Lane name for display */
  laneName: string;
  /** Current blockers */
  blockers: Blocker[];
  /** Available agents for assignment */
  availableAgents?: string[];
  /** Callback when blocker state changes */
  onBlockerChange$?: QRL<(event: BlockerEvent) => void>;
  /** Actor performing changes */
  actor?: string;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getCategoryColor(category: BlockerCategory): string {
  switch (category) {
    case 'dependency': return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
    case 'resource': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'external': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'technical': return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getSeverityColor(severity: BlockerSeverity): string {
  switch (severity) {
    case 'critical': return 'bg-red-500/30 text-red-400 border-red-500/40';
    case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    case 'medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'low': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getStatusColor(status: BlockerStatus): string {
  switch (status) {
    case 'open': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'in_progress': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'resolved': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getCategoryIcon(category: BlockerCategory): string {
  switch (category) {
    case 'dependency': return '🔗';
    case 'resource': return '💼';
    case 'external': return '🌐';
    case 'technical': return '⚙️';
    default: return '•';
  }
}

function getSeverityIcon(severity: BlockerSeverity): string {
  switch (severity) {
    case 'critical': return '🔴';
    case 'high': return '🟠';
    case 'medium': return '🟡';
    case 'low': return '🟢';
    default: return '⚪';
  }
}

function formatDuration(createdAt: string): string {
  try {
    const created = new Date(createdAt);
    const now = new Date();
    const diffMs = now.getTime() - created.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffHours / 24);

    if (diffDays > 0) return `${diffDays}d ${diffHours % 24}h`;
    if (diffHours > 0) return `${diffHours}h`;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    return `${diffMins}m`;
  } catch {
    return '?';
  }
}

function generateId(): string {
  return `blk-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

// ============================================================================
// Component
// ============================================================================

export const BlockerManagement = component$<BlockerManagementProps>(({
  laneId,
  laneName,
  blockers: initialBlockers,
  availableAgents = [],
  onBlockerChange$,
  actor = 'dashboard',
  compact = false,
}) => {
  // State
  const blockers = useSignal<Blocker[]>(initialBlockers);
  const showAddForm = useSignal(false);
  const editingId = useSignal<string | null>(null);
  const filterStatus = useSignal<'all' | BlockerStatus>('all');

  // New blocker form state
  const newDescription = useSignal('');
  const newCategory = useSignal<BlockerCategory>('technical');
  const newSeverity = useSignal<BlockerSeverity>('medium');
  const newAssignee = useSignal('');

  // Computed stats
  const stats = useComputed$(() => {
    const all = blockers.value;
    return {
      total: all.length,
      open: all.filter(b => b.status === 'open').length,
      inProgress: all.filter(b => b.status === 'in_progress').length,
      resolved: all.filter(b => b.status === 'resolved').length,
      critical: all.filter(b => b.severity === 'critical' && b.status !== 'resolved').length,
    };
  });

  // Filtered blockers
  const filteredBlockers = useComputed$(() => {
    if (filterStatus.value === 'all') {
      return blockers.value.filter(b => b.status !== 'resolved');
    }
    return blockers.value.filter(b => b.status === filterStatus.value);
  });

  // Emit event helper
  const emitEvent = $(async (type: BlockerEvent['type'], blocker: Blocker) => {
    const event: BlockerEvent = {
      type,
      laneId,
      blocker,
      actor,
      timestamp: new Date().toISOString(),
    };

    console.log(`[BlockerManagement] Emitting: operator.lanes.blocker.${type}`, event);

    if (onBlockerChange$) {
      await onBlockerChange$(event);
    }
  });

  // Add new blocker
  const addBlocker = $(async () => {
    if (!newDescription.value.trim()) return;

    const newBlocker: Blocker = {
      id: generateId(),
      description: newDescription.value.trim(),
      category: newCategory.value,
      severity: newSeverity.value,
      status: 'open',
      createdAt: new Date().toISOString(),
      assignedTo: newAssignee.value || undefined,
    };

    blockers.value = [...blockers.value, newBlocker];

    await emitEvent('add', newBlocker);

    // Reset form
    newDescription.value = '';
    newCategory.value = 'technical';
    newSeverity.value = 'medium';
    newAssignee.value = '';
    showAddForm.value = false;
  });

  // Update blocker status
  const updateStatus = $(async (blockerId: string, newStatus: BlockerStatus) => {
    const idx = blockers.value.findIndex(b => b.id === blockerId);
    if (idx === -1) return;

    const updated = { ...blockers.value[idx], status: newStatus };
    if (newStatus === 'resolved') {
      updated.resolvedAt = new Date().toISOString();
    }

    const newBlockers = [...blockers.value];
    newBlockers[idx] = updated;
    blockers.value = newBlockers;

    await emitEvent(newStatus === 'resolved' ? 'resolve' : 'update', updated);
  });

  // Reopen resolved blocker
  const reopenBlocker = $(async (blockerId: string) => {
    const idx = blockers.value.findIndex(b => b.id === blockerId);
    if (idx === -1) return;

    const updated = {
      ...blockers.value[idx],
      status: 'open' as BlockerStatus,
      resolvedAt: undefined,
    };

    const newBlockers = [...blockers.value];
    newBlockers[idx] = updated;
    blockers.value = newBlockers;

    await emitEvent('reopen', updated);
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">BLOCKERS</span>
          {stats.value.critical > 0 && (
            <span class="text-[10px] px-2 py-0.5 rounded bg-red-500/30 text-red-400 border border-red-500/40 animate-pulse">
              {stats.value.critical} critical
            </span>
          )}
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {stats.value.open + stats.value.inProgress} active
          </span>
        </div>
        <button
          onClick$={() => { showAddForm.value = !showAddForm.value; }}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 transition-colors"
        >
          + Add Blocker
        </button>
      </div>

      {/* Lane info bar */}
      <div class="px-3 py-2 border-b border-border/30 text-[9px] text-muted-foreground flex items-center justify-between">
        <span>Lane: <span class="text-foreground">{laneName}</span></span>
        <span>{stats.value.resolved} resolved</span>
      </div>

      {/* Add form */}
      {showAddForm.value && (
        <div class="p-3 border-b border-border/30 bg-muted/10 space-y-3">
          <div class="text-[10px] font-medium text-foreground">Add New Blocker</div>

          {/* Description */}
          <textarea
            value={newDescription.value}
            onInput$={(e) => { newDescription.value = (e.target as HTMLTextAreaElement).value; }}
            placeholder="Describe the blocker..."
            class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 text-foreground placeholder:text-muted-foreground/50 focus:outline-none focus:border-primary/50 resize-none"
            rows={2}
          />

          {/* Category and Severity */}
          <div class="flex gap-2">
            <div class="flex-1">
              <label class="text-[9px] text-muted-foreground block mb-1">Category</label>
              <select
                value={newCategory.value}
                onChange$={(e) => { newCategory.value = (e.target as HTMLSelectElement).value as BlockerCategory; }}
                class="w-full px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground"
              >
                <option value="dependency">🔗 Dependency</option>
                <option value="resource">💼 Resource</option>
                <option value="external">🌐 External</option>
                <option value="technical">⚙️ Technical</option>
              </select>
            </div>
            <div class="flex-1">
              <label class="text-[9px] text-muted-foreground block mb-1">Severity</label>
              <select
                value={newSeverity.value}
                onChange$={(e) => { newSeverity.value = (e.target as HTMLSelectElement).value as BlockerSeverity; }}
                class="w-full px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground"
              >
                <option value="critical">🔴 Critical</option>
                <option value="high">🟠 High</option>
                <option value="medium">🟡 Medium</option>
                <option value="low">🟢 Low</option>
              </select>
            </div>
          </div>

          {/* Assignee */}
          {availableAgents.length > 0 && (
            <div>
              <label class="text-[9px] text-muted-foreground block mb-1">Assign To (optional)</label>
              <select
                value={newAssignee.value}
                onChange$={(e) => { newAssignee.value = (e.target as HTMLSelectElement).value; }}
                class="w-full px-2 py-1.5 text-[10px] rounded bg-card border border-border/50 text-foreground"
              >
                <option value="">Unassigned</option>
                {availableAgents.map(agent => (
                  <option key={agent} value={agent}>@{agent}</option>
                ))}
              </select>
            </div>
          )}

          {/* Actions */}
          <div class="flex gap-2 justify-end">
            <button
              onClick$={() => { showAddForm.value = false; }}
              class="px-3 py-1.5 text-[10px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick$={addBlocker}
              disabled={!newDescription.value.trim()}
              class="px-3 py-1.5 text-[10px] rounded bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Add Blocker
            </button>
          </div>
        </div>
      )}

      {/* Filter tabs */}
      {!compact && (
        <div class="flex gap-1 p-2 border-b border-border/30">
          {(['all', 'open', 'in_progress', 'resolved'] as const).map(status => (
            <button
              key={status}
              onClick$={() => { filterStatus.value = status; }}
              class={`px-2 py-1 text-[9px] rounded transition-colors ${
                filterStatus.value === status
                  ? 'bg-primary/20 text-primary border border-primary/30'
                  : 'bg-muted/10 text-muted-foreground hover:bg-muted/20 border border-transparent'
              }`}
            >
              {status === 'all' ? 'Active' : status === 'in_progress' ? 'In Progress' : status.charAt(0).toUpperCase() + status.slice(1)}
              {status === 'open' && stats.value.open > 0 && (
                <span class="ml-1 text-red-400">({stats.value.open})</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Blocker list */}
      <div class="max-h-[300px] overflow-y-auto">
        {filteredBlockers.value.length === 0 ? (
          <div class="p-6 text-center">
            <div class="text-2xl mb-2">✓</div>
            <div class="text-xs text-muted-foreground">
              {filterStatus.value === 'all' ? 'No active blockers' : `No ${filterStatus.value} blockers`}
            </div>
          </div>
        ) : (
          filteredBlockers.value.map(blocker => (
            <div
              key={blocker.id}
              class={`p-3 border-b border-border/30 ${
                blocker.severity === 'critical' && blocker.status !== 'resolved'
                  ? 'bg-red-500/5 border-l-2 border-l-red-500'
                  : ''
              }`}
            >
              {/* Blocker header */}
              <div class="flex items-start gap-2">
                {/* Severity indicator */}
                <span class="text-sm flex-shrink-0">{getSeverityIcon(blocker.severity)}</span>

                {/* Content */}
                <div class="flex-grow min-w-0">
                  <div class="text-xs text-foreground">{blocker.description}</div>
                  <div class="flex items-center gap-2 mt-1 flex-wrap">
                    <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getCategoryColor(blocker.category)}`}>
                      {getCategoryIcon(blocker.category)} {blocker.category}
                    </span>
                    <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getSeverityColor(blocker.severity)}`}>
                      {blocker.severity}
                    </span>
                    <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getStatusColor(blocker.status)}`}>
                      {blocker.status === 'in_progress' ? 'in progress' : blocker.status}
                    </span>
                    {blocker.assignedTo && (
                      <span class="text-[9px] text-muted-foreground">
                        → @{blocker.assignedTo}
                      </span>
                    )}
                  </div>
                </div>

                {/* Duration */}
                <div class="flex-shrink-0 text-right">
                  <div class="text-[9px] text-muted-foreground">
                    {blocker.status === 'resolved' ? 'Resolved' : formatDuration(blocker.createdAt)}
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div class="flex gap-1 mt-2 ml-6">
                {blocker.status === 'open' && (
                  <>
                    <button
                      onClick$={() => updateStatus(blocker.id, 'in_progress')}
                      class="px-2 py-0.5 text-[9px] rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30 transition-colors"
                    >
                      Start Work
                    </button>
                    <button
                      onClick$={() => updateStatus(blocker.id, 'resolved')}
                      class="px-2 py-0.5 text-[9px] rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30 transition-colors"
                    >
                      Resolve
                    </button>
                  </>
                )}
                {blocker.status === 'in_progress' && (
                  <>
                    <button
                      onClick$={() => updateStatus(blocker.id, 'open')}
                      class="px-2 py-0.5 text-[9px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
                    >
                      Pause
                    </button>
                    <button
                      onClick$={() => updateStatus(blocker.id, 'resolved')}
                      class="px-2 py-0.5 text-[9px] rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 hover:bg-emerald-500/30 transition-colors"
                    >
                      Resolve
                    </button>
                  </>
                )}
                {blocker.status === 'resolved' && (
                  <button
                    onClick$={() => reopenBlocker(blocker.id)}
                    class="px-2 py-0.5 text-[9px] rounded bg-red-500/20 text-red-400 border border-red-500/30 hover:bg-red-500/30 transition-colors"
                  >
                    Reopen
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {stats.value.total} total blockers • Actor: @{actor}
      </div>
    </div>
  );
});

export default BlockerManagement;
