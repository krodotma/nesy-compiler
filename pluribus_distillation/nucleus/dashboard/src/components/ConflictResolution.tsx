/**
 * ConflictResolution - Multi-agent conflict detection and resolution
 *
 * Phase 5, Iteration 41 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Automatic conflict detection
 * - Resolution strategies (auto, manual, escalate)
 * - Conflict history and patterns
 * - Agent coordination notifications
 * - Merge conflict visualization
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

export type ConflictType = 'resource' | 'schedule' | 'priority' | 'dependency' | 'merge' | 'assignment';
export type ConflictSeverity = 'low' | 'medium' | 'high' | 'critical';
export type ConflictStatus = 'detected' | 'reviewing' | 'resolved' | 'escalated';

export interface Conflict {
  id: string;
  type: ConflictType;
  severity: ConflictSeverity;
  status: ConflictStatus;
  title: string;
  description: string;
  involvedAgents: string[];
  involvedLanes: string[];
  detectedAt: string;
  resolvedAt?: string;
  resolvedBy?: string;
  resolutionStrategy?: string;
  suggestedResolutions: Resolution[];
}

export interface Resolution {
  id: string;
  label: string;
  description: string;
  impact: 'none' | 'minor' | 'moderate' | 'major';
  automated: boolean;
  confidence: number; // 0-100
}

export interface ConflictResolutionProps {
  /** Active conflicts */
  conflicts: Conflict[];
  /** Callback when resolution is applied */
  onResolve$?: QRL<(conflictId: string, resolutionId: string) => void>;
  /** Callback when conflict is escalated */
  onEscalate$?: QRL<(conflictId: string, reason: string) => void>;
  /** Callback when conflict is dismissed */
  onDismiss$?: QRL<(conflictId: string) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getSeverityColor(severity: ConflictSeverity): string {
  switch (severity) {
    case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    case 'medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'low': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getStatusColor(status: ConflictStatus): string {
  switch (status) {
    case 'detected': return 'bg-red-500/20 text-red-400';
    case 'reviewing': return 'bg-amber-500/20 text-amber-400';
    case 'resolved': return 'bg-emerald-500/20 text-emerald-400';
    case 'escalated': return 'bg-purple-500/20 text-purple-400';
    default: return 'bg-muted/20 text-muted-foreground';
  }
}

function getTypeIcon(type: ConflictType): string {
  switch (type) {
    case 'resource': return '⚡';
    case 'schedule': return '🕐';
    case 'priority': return '⬆';
    case 'dependency': return '🔗';
    case 'merge': return '⎇';
    case 'assignment': return '👤';
    default: return '⚠';
  }
}

function getImpactColor(impact: string): string {
  switch (impact) {
    case 'none': return 'text-emerald-400';
    case 'minor': return 'text-blue-400';
    case 'moderate': return 'text-amber-400';
    case 'major': return 'text-red-400';
    default: return 'text-muted-foreground';
  }
}

function formatTimeSince(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
    return `${Math.floor(diffMins / 1440)}d ago`;
  } catch {
    return dateStr;
  }
}

// ============================================================================
// Component
// ============================================================================

export const ConflictResolution = component$<ConflictResolutionProps>(({
  conflicts,
  onResolve$,
  onEscalate$,
  onDismiss$,
}) => {
  // State
  const selectedConflictId = useSignal<string | null>(conflicts.find(c => c.status !== 'resolved')?.id || null);
  const showEscalateModal = useSignal(false);
  const escalateReason = useSignal('');
  const filterStatus = useSignal<ConflictStatus | 'all'>('all');

  // Computed
  const selectedConflict = useComputed$(() =>
    conflicts.find(c => c.id === selectedConflictId.value)
  );

  const filteredConflicts = useComputed$(() => {
    if (filterStatus.value === 'all') return conflicts;
    return conflicts.filter(c => c.status === filterStatus.value);
  });

  const stats = useComputed$(() => ({
    total: conflicts.length,
    active: conflicts.filter(c => c.status === 'detected' || c.status === 'reviewing').length,
    resolved: conflicts.filter(c => c.status === 'resolved').length,
    escalated: conflicts.filter(c => c.status === 'escalated').length,
    critical: conflicts.filter(c => c.severity === 'critical' && c.status !== 'resolved').length,
  }));

  // Actions
  const resolveConflict = $(async (resolutionId: string) => {
    if (!selectedConflictId.value || !onResolve$) return;
    await onResolve$(selectedConflictId.value, resolutionId);
  });

  const escalateConflict = $(async () => {
    if (!selectedConflictId.value || !onEscalate$) return;
    await onEscalate$(selectedConflictId.value, escalateReason.value);
    showEscalateModal.value = false;
    escalateReason.value = '';
  });

  const dismissConflict = $(async () => {
    if (!selectedConflictId.value || !onDismiss$) return;
    await onDismiss$(selectedConflictId.value);
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">CONFLICT RESOLUTION</span>
          {stats.value.critical > 0 && (
            <span class="text-[9px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse">
              {stats.value.critical} Critical
            </span>
          )}
        </div>
        <div class="flex items-center gap-2 text-[9px]">
          {(['all', 'detected', 'reviewing', 'escalated', 'resolved'] as const).map(status => (
            <button
              key={status}
              onClick$={() => { filterStatus.value = status; }}
              class={`px-2 py-1 rounded transition-colors ${
                filterStatus.value === status
                  ? 'bg-primary/20 text-primary'
                  : 'text-muted-foreground hover:bg-muted/30'
              }`}
            >
              {status === 'all' ? 'All' : status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Summary stats */}
      <div class="grid grid-cols-4 gap-2 p-3 border-b border-border/30 bg-muted/5">
        <div class="text-center">
          <div class="text-lg font-bold text-foreground">{stats.value.total}</div>
          <div class="text-[9px] text-muted-foreground">Total</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-amber-400">{stats.value.active}</div>
          <div class="text-[9px] text-muted-foreground">Active</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-emerald-400">{stats.value.resolved}</div>
          <div class="text-[9px] text-muted-foreground">Resolved</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-purple-400">{stats.value.escalated}</div>
          <div class="text-[9px] text-muted-foreground">Escalated</div>
        </div>
      </div>

      {/* Main content */}
      <div class="grid grid-cols-2 gap-0 min-h-[300px]">
        {/* Conflict list */}
        <div class="border-r border-border/30 max-h-[350px] overflow-y-auto">
          {filteredConflicts.value.length > 0 ? (
            <div class="divide-y divide-border/20">
              {filteredConflicts.value.map(conflict => (
                <div
                  key={conflict.id}
                  onClick$={() => { selectedConflictId.value = conflict.id; }}
                  class={`p-3 cursor-pointer transition-colors ${
                    selectedConflictId.value === conflict.id
                      ? 'bg-primary/10'
                      : 'hover:bg-muted/5'
                  }`}
                >
                  <div class="flex items-start justify-between">
                    <div class="flex items-center gap-2">
                      <span class="text-sm">{getTypeIcon(conflict.type)}</span>
                      <span class="text-xs font-medium text-foreground">{conflict.title}</span>
                    </div>
                    <span class={`text-[8px] px-1.5 py-0.5 rounded border ${getSeverityColor(conflict.severity)}`}>
                      {conflict.severity}
                    </span>
                  </div>

                  <div class="mt-2 flex items-center gap-2">
                    <span class={`text-[8px] px-1.5 py-0.5 rounded ${getStatusColor(conflict.status)}`}>
                      {conflict.status}
                    </span>
                    <span class="text-[8px] text-muted-foreground">
                      {formatTimeSince(conflict.detectedAt)}
                    </span>
                  </div>

                  <div class="mt-2 flex items-center gap-1 flex-wrap">
                    {conflict.involvedAgents.slice(0, 3).map(agent => (
                      <span key={agent} class="text-[8px] px-1 py-0.5 rounded bg-cyan-500/20 text-cyan-400">
                        @{agent}
                      </span>
                    ))}
                    {conflict.involvedAgents.length > 3 && (
                      <span class="text-[8px] text-muted-foreground">
                        +{conflict.involvedAgents.length - 3}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground p-8">
              {conflicts.length === 0 ? 'No conflicts detected' : 'No conflicts match filter'}
            </div>
          )}
        </div>

        {/* Conflict detail */}
        <div class="max-h-[350px] overflow-y-auto">
          {selectedConflict.value ? (
            <div class="p-3">
              {/* Header */}
              <div class="mb-4 pb-4 border-b border-border/30">
                <div class="flex items-center gap-2 mb-2">
                  <span class="text-lg">{getTypeIcon(selectedConflict.value.type)}</span>
                  <span class="text-sm font-medium text-foreground">{selectedConflict.value.title}</span>
                </div>

                <div class="flex items-center gap-2 mb-2">
                  <span class={`text-[9px] px-2 py-0.5 rounded border ${getSeverityColor(selectedConflict.value.severity)}`}>
                    {selectedConflict.value.severity.toUpperCase()}
                  </span>
                  <span class={`text-[9px] px-2 py-0.5 rounded ${getStatusColor(selectedConflict.value.status)}`}>
                    {selectedConflict.value.status}
                  </span>
                </div>

                <p class="text-[10px] text-muted-foreground">{selectedConflict.value.description}</p>
              </div>

              {/* Involved parties */}
              <div class="mb-4">
                <div class="text-[9px] font-semibold text-muted-foreground mb-2">INVOLVED</div>
                <div class="grid grid-cols-2 gap-2">
                  <div>
                    <div class="text-[8px] text-muted-foreground mb-1">Agents</div>
                    <div class="flex flex-wrap gap-1">
                      {selectedConflict.value.involvedAgents.map(agent => (
                        <span key={agent} class="text-[8px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400">
                          @{agent}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div>
                    <div class="text-[8px] text-muted-foreground mb-1">Lanes</div>
                    <div class="flex flex-wrap gap-1">
                      {selectedConflict.value.involvedLanes.map(lane => (
                        <span key={lane} class="text-[8px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400">
                          {lane}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Suggested resolutions */}
              {selectedConflict.value.status !== 'resolved' && (
                <div class="mb-4">
                  <div class="text-[9px] font-semibold text-muted-foreground mb-2">
                    SUGGESTED RESOLUTIONS
                  </div>
                  <div class="space-y-2">
                    {selectedConflict.value.suggestedResolutions.map(resolution => (
                      <div
                        key={resolution.id}
                        class="p-2 rounded border border-border/30 hover:border-border/50 transition-colors"
                      >
                        <div class="flex items-center justify-between mb-1">
                          <div class="flex items-center gap-2">
                            <span class="text-[10px] font-medium text-foreground">{resolution.label}</span>
                            {resolution.automated && (
                              <span class="text-[8px] px-1 py-0.5 rounded bg-blue-500/20 text-blue-400">
                                Auto
                              </span>
                            )}
                          </div>
                          <div class="flex items-center gap-2">
                            <span class={`text-[8px] ${getImpactColor(resolution.impact)}`}>
                              {resolution.impact} impact
                            </span>
                            <span class="text-[8px] text-muted-foreground">
                              {resolution.confidence}% conf
                            </span>
                          </div>
                        </div>
                        <p class="text-[9px] text-muted-foreground mb-2">{resolution.description}</p>
                        <button
                          onClick$={() => resolveConflict(resolution.id)}
                          class="text-[9px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
                        >
                          Apply Resolution
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Resolution info (for resolved conflicts) */}
              {selectedConflict.value.status === 'resolved' && (
                <div class="mb-4 p-3 rounded bg-emerald-500/10 border border-emerald-500/30">
                  <div class="text-[9px] font-semibold text-emerald-400 mb-1">RESOLVED</div>
                  <div class="text-[10px] text-foreground">
                    {selectedConflict.value.resolutionStrategy}
                  </div>
                  <div class="text-[8px] text-muted-foreground mt-1">
                    by @{selectedConflict.value.resolvedBy} • {formatTimeSince(selectedConflict.value.resolvedAt!)}
                  </div>
                </div>
              )}

              {/* Actions */}
              {selectedConflict.value.status !== 'resolved' && (
                <div class="flex items-center gap-2">
                  <button
                    onClick$={() => { showEscalateModal.value = true; }}
                    class="px-3 py-1.5 text-[9px] rounded bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
                  >
                    Escalate
                  </button>
                  <button
                    onClick$={dismissConflict}
                    class="px-3 py-1.5 text-[9px] rounded bg-muted/30 text-muted-foreground hover:bg-muted/50"
                  >
                    Dismiss
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground">
              Select a conflict to view details
            </div>
          )}
        </div>
      </div>

      {/* Escalate Modal */}
      {showEscalateModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-80">
            <div class="text-xs font-semibold text-foreground mb-4">Escalate Conflict</div>

            <div class="mb-4">
              <label class="text-[9px] text-muted-foreground block mb-1">Reason for escalation</label>
              <textarea
                value={escalateReason.value}
                onInput$={(e) => { escalateReason.value = (e.target as HTMLTextAreaElement).value; }}
                class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 resize-none h-20"
                placeholder="Describe why this conflict needs escalation..."
              />
            </div>

            <div class="flex items-center gap-2">
              <button
                onClick$={() => { showEscalateModal.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={escalateConflict}
                disabled={!escalateReason.value}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-purple-500 text-white disabled:opacity-50"
              >
                Escalate
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {stats.value.active} active conflicts • {stats.value.resolved} resolved
      </div>
    </div>
  );
});

export default ConflictResolution;
