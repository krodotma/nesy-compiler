/**
 * LaneOrchestrator - Central Coordination Component
 *
 * Phase 7, Iteration 58 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Central coordination for multi-component operations
 * - Conflict detection between concurrent edits
 * - Operation queuing for sequential execution
 * - Agent coordination display
 * - Activity monitoring
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
} from '@builder.io/qwik';

import { getGlobalComponentBus } from '../lib/lanes/component-bus';
import type { ComponentEvent, LaneEventMap } from '../lib/lanes/component-bus';

// ============================================================================
// Types
// ============================================================================

export interface PendingOperation {
  id: string;
  type: string;
  laneId?: string;
  description: string;
  source: string;
  timestamp: number;
  status: 'pending' | 'executing' | 'completed' | 'failed';
  error?: string;
}

export interface Conflict {
  id: string;
  laneId: string;
  field: string;
  sources: string[];
  timestamp: number;
  resolved: boolean;
  resolution?: 'local' | 'remote' | 'merged';
}

export interface AgentActivity {
  agentId: string;
  lastAction: string;
  timestamp: number;
  laneId?: string;
}

export interface LaneOrchestratorProps {
  /** Show detailed operation log */
  showLog?: boolean;
  /** Maximum operations to display */
  maxOperations?: number;
  /** Enable conflict detection */
  detectConflicts?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function formatTimestamp(ts: number): string {
  const date = new Date(ts);
  return date.toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'pending': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'executing': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'completed': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneOrchestrator = component$<LaneOrchestratorProps>(({
  showLog = true,
  maxOperations = 20,
  detectConflicts = true,
}) => {
  // State
  const operations = useSignal<PendingOperation[]>([]);
  const conflicts = useSignal<Conflict[]>([]);
  const agentActivity = useSignal<Map<string, AgentActivity>>(new Map());
  const isPaused = useSignal(false);
  const showConflicts = useSignal(false);

  // Get component bus
  const bus = getGlobalComponentBus();

  // Computed stats
  const stats = useComputed$(() => {
    const ops = operations.value;
    return {
      total: ops.length,
      pending: ops.filter(o => o.status === 'pending').length,
      executing: ops.filter(o => o.status === 'executing').length,
      completed: ops.filter(o => o.status === 'completed').length,
      failed: ops.filter(o => o.status === 'failed').length,
    };
  });

  const activeConflicts = useComputed$(() =>
    conflicts.value.filter(c => !c.resolved)
  );

  // Subscribe to bus events
  useVisibleTask$(({ cleanup }) => {
    const unsubscribers: Array<() => void> = [];

    // Track lane updates
    unsubscribers.push(
      bus.subscribe('lane.updated', (event: ComponentEvent<LaneEventMap['lane.updated']>) => {
        const op: PendingOperation = {
          id: event.id,
          type: 'update',
          laneId: event.payload.laneId,
          description: `Update lane ${event.payload.laneId}`,
          source: event.source,
          timestamp: event.timestamp,
          status: 'completed',
        };

        operations.value = [op, ...operations.value].slice(0, maxOperations);

        // Check for conflicts
        if (detectConflicts) {
          checkForConflict(event.payload.laneId, event.source);
        }

        // Update agent activity
        updateAgentActivity(event.source, 'lane.update', event.payload.laneId);
      })
    );

    // Track lane selections
    unsubscribers.push(
      bus.subscribe('lane.selected', (event: ComponentEvent<LaneEventMap['lane.selected']>) => {
        updateAgentActivity(event.source, 'lane.select', event.payload.laneId || undefined);
      })
    );

    // Track sync events
    unsubscribers.push(
      bus.subscribe('sync.started', (event) => {
        const op: PendingOperation = {
          id: event.id,
          type: 'sync',
          description: 'Syncing with server...',
          source: event.source,
          timestamp: event.timestamp,
          status: 'executing',
        };
        operations.value = [op, ...operations.value].slice(0, maxOperations);
      })
    );

    unsubscribers.push(
      bus.subscribe('sync.completed', (event: ComponentEvent<LaneEventMap['sync.completed']>) => {
        operations.value = operations.value.map(op => {
          if (op.type === 'sync' && op.status === 'executing') {
            return {
              ...op,
              status: event.payload.success ? 'completed' : 'failed',
              error: event.payload.error,
            };
          }
          return op;
        });
      })
    );

    // Track conflicts
    unsubscribers.push(
      bus.subscribe('sync.conflict', (event: ComponentEvent<LaneEventMap['sync.conflict']>) => {
        const conflict: Conflict = {
          id: `conflict-${Date.now()}`,
          laneId: event.payload.laneId,
          field: event.payload.field,
          sources: [event.source],
          timestamp: event.timestamp,
          resolved: false,
        };
        conflicts.value = [conflict, ...conflicts.value].slice(0, 10);
      })
    );

    cleanup(() => {
      unsubscribers.forEach(unsub => unsub());
    });
  });

  // Check for conflicts
  const checkForConflict = (laneId: string, source: string) => {
    // Check if multiple sources are editing the same lane
    const recentOps = operations.value
      .filter(op =>
        op.laneId === laneId &&
        op.timestamp > Date.now() - 5000 &&
        op.source !== source
      );

    if (recentOps.length > 0) {
      const existingConflict = conflicts.value.find(
        c => c.laneId === laneId && !c.resolved
      );

      if (existingConflict) {
        // Add source to existing conflict
        if (!existingConflict.sources.includes(source)) {
          conflicts.value = conflicts.value.map(c =>
            c.id === existingConflict.id
              ? { ...c, sources: [...c.sources, source] }
              : c
          );
        }
      } else {
        // Create new conflict
        const conflict: Conflict = {
          id: `conflict-${Date.now()}`,
          laneId,
          field: 'multiple',
          sources: [source, ...recentOps.map(op => op.source)],
          timestamp: Date.now(),
          resolved: false,
        };
        conflicts.value = [conflict, ...conflicts.value].slice(0, 10);
      }
    }
  };

  // Update agent activity
  const updateAgentActivity = (agentId: string, action: string, laneId?: string) => {
    const newActivity = new Map(agentActivity.value);
    newActivity.set(agentId, {
      agentId,
      lastAction: action,
      timestamp: Date.now(),
      laneId,
    });
    agentActivity.value = newActivity;
  };

  // Resolve conflict
  const resolveConflict = $((conflictId: string, resolution: 'local' | 'remote' | 'merged') => {
    conflicts.value = conflicts.value.map(c =>
      c.id === conflictId
        ? { ...c, resolved: true, resolution }
        : c
    );
  });

  // Clear operations
  const clearOperations = $(() => {
    operations.value = [];
  });

  // Pause/resume
  const togglePause = $(() => {
    isPaused.value = !isPaused.value;
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-2 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-[10px] font-semibold text-muted-foreground">ORCHESTRATOR</span>
          {activeConflicts.value.length > 0 && (
            <span class="text-[9px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse">
              {activeConflicts.value.length} conflict{activeConflicts.value.length > 1 ? 's' : ''}
            </span>
          )}
        </div>
        <div class="flex items-center gap-1">
          <button
            onClick$={togglePause}
            class={`text-[9px] px-2 py-0.5 rounded transition-colors ${
              isPaused.value
                ? 'bg-amber-500/20 text-amber-400'
                : 'bg-muted/20 text-muted-foreground hover:bg-muted/40'
            }`}
          >
            {isPaused.value ? 'Resume' : 'Pause'}
          </button>
          <button
            onClick$={() => { showConflicts.value = !showConflicts.value; }}
            class="text-[9px] px-2 py-0.5 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
          >
            Conflicts
          </button>
          <button
            onClick$={clearOperations}
            class="text-[9px] px-2 py-0.5 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Stats bar */}
      <div class="grid grid-cols-5 gap-1 p-2 border-b border-border/30 bg-muted/5">
        <div class="text-center">
          <div class="text-sm font-bold text-foreground">{stats.value.total}</div>
          <div class="text-[8px] text-muted-foreground">Total</div>
        </div>
        <div class="text-center">
          <div class="text-sm font-bold text-amber-400">{stats.value.pending}</div>
          <div class="text-[8px] text-muted-foreground">Pending</div>
        </div>
        <div class="text-center">
          <div class="text-sm font-bold text-blue-400">{stats.value.executing}</div>
          <div class="text-[8px] text-muted-foreground">Executing</div>
        </div>
        <div class="text-center">
          <div class="text-sm font-bold text-emerald-400">{stats.value.completed}</div>
          <div class="text-[8px] text-muted-foreground">Done</div>
        </div>
        <div class="text-center">
          <div class="text-sm font-bold text-red-400">{stats.value.failed}</div>
          <div class="text-[8px] text-muted-foreground">Failed</div>
        </div>
      </div>

      {/* Conflicts panel */}
      {showConflicts.value && activeConflicts.value.length > 0 && (
        <div class="p-2 border-b border-border/30 bg-red-500/5">
          <div class="text-[9px] font-semibold text-red-400 mb-2">ACTIVE CONFLICTS</div>
          <div class="space-y-2">
            {activeConflicts.value.map(conflict => (
              <div
                key={conflict.id}
                class="p-2 rounded bg-red-500/10 border border-red-500/30"
              >
                <div class="flex items-center justify-between mb-1">
                  <span class="text-[10px] font-medium text-red-300">
                    Lane: {conflict.laneId}
                  </span>
                  <span class="text-[8px] text-red-400/50">
                    {formatTimestamp(conflict.timestamp)}
                  </span>
                </div>
                <div class="text-[9px] text-red-300/80 mb-2">
                  Conflicting sources: {conflict.sources.join(', ')}
                </div>
                <div class="flex items-center gap-1">
                  <button
                    onClick$={() => resolveConflict(conflict.id, 'local')}
                    class="text-[8px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
                  >
                    Keep Local
                  </button>
                  <button
                    onClick$={() => resolveConflict(conflict.id, 'remote')}
                    class="text-[8px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
                  >
                    Use Remote
                  </button>
                  <button
                    onClick$={() => resolveConflict(conflict.id, 'merged')}
                    class="text-[8px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                  >
                    Merge
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Agent activity */}
      {agentActivity.value.size > 0 && (
        <div class="p-2 border-b border-border/30">
          <div class="text-[9px] font-semibold text-muted-foreground mb-2">ACTIVE AGENTS</div>
          <div class="flex flex-wrap gap-1">
            {Array.from(agentActivity.value.values())
              .filter(a => Date.now() - a.timestamp < 60000)
              .map(activity => (
                <span
                  key={activity.agentId}
                  class="text-[9px] px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                  title={`${activity.lastAction} ${activity.laneId ? `on ${activity.laneId}` : ''}`}
                >
                  @{activity.agentId}
                  {activity.laneId && (
                    <span class="text-cyan-400/50 ml-1">[{activity.laneId.slice(0, 8)}]</span>
                  )}
                </span>
              ))}
          </div>
        </div>
      )}

      {/* Operations log */}
      {showLog && (
        <div class="max-h-[200px] overflow-y-auto">
          {operations.value.length > 0 ? (
            <div class="p-2 space-y-1">
              {operations.value.map(op => (
                <div
                  key={op.id}
                  class="flex items-center gap-2 p-1.5 rounded bg-muted/5 border border-border/20 text-[10px]"
                >
                  <span class={`px-1.5 py-0.5 rounded border ${getStatusColor(op.status)}`}>
                    {op.status}
                  </span>
                  <span class="flex-1 text-foreground truncate">
                    {op.description}
                  </span>
                  <span class="text-muted-foreground/50">
                    {op.source}
                  </span>
                  <span class="text-muted-foreground/30">
                    {formatTimestamp(op.timestamp)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <div class="p-4 text-center text-[10px] text-muted-foreground">
              No operations recorded. Activity will appear here.
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 flex items-center justify-between text-[9px] text-muted-foreground">
        <span>
          {isPaused.value && (
            <span class="text-amber-400 mr-2">PAUSED</span>
          )}
          {operations.value.length} operations
        </span>
        <span>
          {conflicts.value.filter(c => c.resolved).length} resolved /
          {activeConflicts.value.length} active conflicts
        </span>
      </div>
    </div>
  );
});

export default LaneOrchestrator;
