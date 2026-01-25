/**
 * AgentHandoff - Agent task handoff management
 *
 * Phase 5, Iteration 42 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Seamless task handoff between agents
 * - Context transfer tracking
 * - Handoff history
 * - Recovery from failed handoffs
 * - Handoff templates
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - AgentHandoff
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { Input } from './ui/Input';

// ============================================================================
// Types
// ============================================================================

export type HandoffStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'cancelled';

export interface HandoffContext {
  key: string;
  value: string;
  transferred: boolean;
}

export interface Handoff {
  id: string;
  taskId: string;
  taskTitle: string;
  fromAgent: string;
  toAgent: string;
  status: HandoffStatus;
  reason: string;
  context: HandoffContext[];
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  error?: string;
  progress: number; // 0-100
}

export interface AgentHandoffProps {
  /** Active and recent handoffs */
  handoffs: Handoff[];
  /** Available agents for handoff */
  agents: { id: string; name: string; status: string }[];
  /** Callback when handoff is initiated */
  onInitiateHandoff$?: QRL<(taskId: string, fromAgent: string, toAgent: string, reason: string) => void>;
  /** Callback when handoff is cancelled */
  onCancelHandoff$?: QRL<(handoffId: string) => void>;
  /** Callback when handoff is retried */
  onRetryHandoff$?: QRL<(handoffId: string) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: HandoffStatus): string {
  switch (status) {
    case 'pending': return 'bg-md-warning/10 text-md-warning border-md-warning/30';
    case 'in_progress': return 'bg-md-primary/10 text-md-primary border-md-primary/30';
    case 'completed': return 'bg-md-success/10 text-md-success border-md-success/30';
    case 'failed': return 'bg-md-error/10 text-md-error border-md-error/30';
    case 'cancelled': return 'bg-md-surface-variant text-md-on-surface-variant border-md-outline/30';
    default: return 'bg-md-surface-variant border-md-outline/30';
  }
}

function formatTime(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  } catch { return dateStr; }
}

function formatDuration(startStr: string, endStr?: string): string {
  try {
    const start = new Date(startStr).getTime();
    const end = endStr ? new Date(endStr).getTime() : Date.now();
    const diffSecs = Math.floor((end - start) / 1000);
    if (diffSecs < 60) return `${diffSecs}s`;
    return `${Math.floor(diffSecs / 60)}m ${diffSecs % 60}s`;
  } catch { return '-'; }
}

// ============================================================================
// Component
// ============================================================================

export const AgentHandoff = component$<AgentHandoffProps>(({
  handoffs,
  agents,
  onInitiateHandoff$,
  onCancelHandoff$,
  onRetryHandoff$,
}) => {
  const selectedHandoffId = useSignal<string | null>(handoffs.find(h => h.status === 'in_progress')?.id || null);
  const showInitiateModal = useSignal(false);

  const newHandoff = useSignal({
    taskId: '', taskTitle: '', fromAgent: '', toAgent: '', reason: '',
  });

  const selectedHandoff = useComputed$(() => handoffs.find(h => h.id === selectedHandoffId.value));
  const activeHandoffs = useComputed$(() => handoffs.filter(h => h.status === 'pending' || h.status === 'in_progress'));

  const stats = useComputed$(() => ({
    total: handoffs.length,
    active: handoffs.filter(h => h.status === 'pending' || h.status === 'in_progress').length,
    completed: handoffs.filter(h => h.status === 'completed').length,
    failed: handoffs.filter(h => h.status === 'failed').length,
  }));

  const initiateHandoff = $(async () => {
    if (!newHandoff.value.taskId || !newHandoff.value.fromAgent || !newHandoff.value.toAgent) return;
    if (onInitiateHandoff$) {
      await onInitiateHandoff$(newHandoff.value.taskId, newHandoff.value.fromAgent, newHandoff.value.toAgent, newHandoff.value.reason);
    }
    showInitiateModal.value = false;
    newHandoff.value = { taskId: '', taskTitle: '', fromAgent: '', toAgent: '', reason: '' };
  });

  const cancelHandoff = $(async (handoffId: string) => { if (onCancelHandoff$) await onCancelHandoff$(handoffId); });
  const retryHandoff = $(async (handoffId: string) => { if (onRetryHandoff$) await onRetryHandoff$(handoffId); });

  return (
    <Card variant="outlined" padding="p-0" class="overflow-hidden bg-md-surface flex flex-col h-full">
      {/* Header */}
      <div class="p-4 border-b border-md-outline/10 bg-md-surface-container-low flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="text-[11px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/60">Cross-Agent Handoff</span>
          {activeHandoffs.value.length > 0 && (
            <span class="text-[9px] px-2 py-0.5 rounded-full bg-md-primary/10 text-md-primary font-bold border border-md-primary/20 animate-pulse">
              {activeHandoffs.value.length} IN TRANSIT
            </span>
          )}
        </div>
        <Button variant="primary" icon="add" class="h-9" onClick$={() => { showInitiateModal.value = true; }}>
          Initiate
        </Button>
      </div>

      {/* Summary stats */}
      <div class="grid grid-cols-4 gap-0 border-b border-md-outline/5 bg-md-surface-container-lowest divide-x divide-md-outline/5">
        <div class="py-3 text-center">
          <div class="text-xl font-black text-md-on-surface">{stats.value.total}</div>
          <div class="text-[9px] font-bold uppercase text-md-on-surface-variant/40">Total</div>
        </div>
        <div class="py-3 text-center">
          <div class="text-xl font-black text-md-primary">{stats.value.active}</div>
          <div class="text-[9px] font-bold uppercase text-md-on-surface-variant/40">Active</div>
        </div>
        <div class="py-3 text-center">
          <div class="text-xl font-black text-md-success">{stats.value.completed}</div>
          <div class="text-[9px] font-bold uppercase text-md-on-surface-variant/40">Done</div>
        </div>
        <div class="py-3 text-center">
          <div class="text-xl font-black text-md-error">{stats.value.failed}</div>
          <div class="text-[9px] font-bold uppercase text-md-on-surface-variant/40">Fail</div>
        </div>
      </div>

      {/* Main content */}
      <div class="grid grid-cols-2 gap-0 min-h-[300px] flex-1">
        {/* Handoff list */}
        <div class="border-r border-md-outline/10 overflow-y-auto no-scrollbar bg-md-surface-container-low/30">
          {handoffs.length > 0 ? (
            <div class="divide-y divide-md-outline/5">
              {handoffs.map(handoff => (
                <div
                  key={handoff.id}
                  onClick$={() => { selectedHandoffId.value = handoff.id; }}
                  class={`p-4 cursor-pointer transition-all ${
                    selectedHandoffId.value === handoff.id ? 'bg-md-primary/10 shadow-inner' : 'hover:bg-md-surface-variant/40'
                  }`}
                >
                  <div class="flex items-center justify-between mb-3">
                    <span class="text-xs font-black text-md-on-surface uppercase tracking-tight truncate mr-2">
                      {handoff.taskTitle}
                    </span>
                    <span class={`text-[8px] font-black px-2 py-0.5 rounded-full border uppercase tracking-tighter ${getStatusColor(handoff.status)}`}>
                      {handoff.status}
                    </span>
                  </div>

                  <div class="flex items-center gap-2 text-[10px] font-bold uppercase">
                    <span class="text-md-secondary">@{handoff.fromAgent}</span>
                    <md-icon class="text-[10px] text-md-on-surface-variant/30">arrow_forward</md-icon>
                    <span class="text-md-tertiary">@{handoff.toAgent}</span>
                  </div>

                  {handoff.status === 'in_progress' && (
                    <div class="mt-3 space-y-1">
                      <div class="h-1 rounded-full bg-md-surface-container-highest overflow-hidden">
                        <div
                          class="h-full rounded-full bg-md-primary transition-all duration-500"
                          style={{ width: `${handoff.progress}%` }}
                        />
                      </div>
                      <div class="text-[8px] font-black text-md-primary text-right uppercase tracking-widest">{handoff.progress}% Sync</div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div class="flex flex-col items-center justify-center h-full text-md-on-surface-variant/30 space-y-2 opacity-50">
              <md-icon class="text-4xl">swap_horiz</md-icon>
              <div class="text-[10px] font-black uppercase tracking-[0.2em]">No Handoff Data</div>
            </div>
          )}
        </div>

        {/* Handoff detail */}
        <div class="overflow-y-auto no-scrollbar">
          {selectedHandoff.value ? (
            <div class="p-6 space-y-6">
              <div class="space-y-2">
                <div class="text-lg font-black text-md-on-surface tracking-tight leading-tight">
                  {selectedHandoff.value.taskTitle}
                </div>
                <span class={`text-[9px] font-black px-3 py-1 rounded-full border uppercase tracking-widest ${getStatusColor(selectedHandoff.value.status)}`}>
                  {selectedHandoff.value.status}
                </span>
              </div>

              {/* Transfer visualization */}
              <Card variant="filled" padding="p-6" class="bg-md-surface-container-high border border-md-outline/5 rounded-3xl shadow-sm">
                <div class="flex items-center justify-between text-[10px] font-black uppercase tracking-widest">
                  <div class="text-center">
                    <div class="text-md-secondary mb-1">@{selectedHandoff.value.fromAgent}</div>
                    <div class="text-[8px] text-md-on-surface-variant/40">Egress</div>
                  </div>
                  <div class="flex-1 mx-6 h-0.5 bg-md-outline/10 relative rounded-full overflow-hidden">
                    <div class="absolute inset-0 bg-gradient-to-r from-md-secondary to-md-tertiary opacity-40" />
                    {selectedHandoff.value.status === 'in_progress' && (
                      <div
                        class="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-white shadow-lg animate-pulse"
                        style={{ left: `${selectedHandoff.value.progress}%` }}
                      />
                    )}
                  </div>
                  <div class="text-center">
                    <div class="text-md-tertiary mb-1">@{selectedHandoff.value.toAgent}</div>
                    <div class="text-[8px] text-md-on-surface-variant/40">Ingress</div>
                  </div>
                </div>
              </Card>

              {/* Reason */}
              <div class="space-y-2">
                <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/40 ml-1">Transfer Logic</div>
                <div class="text-sm font-medium text-md-on-surface bg-md-surface-container-low p-4 rounded-2xl border border-md-outline/5">
                  {selectedHandoff.value.reason}
                </div>
              </div>

              {/* Context transfer */}
              <div class="space-y-3">
                <div class="text-[10px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/40 ml-1">Context Mapping</div>
                <div class="grid grid-cols-1 gap-2">
                  {selectedHandoff.value.context.map((ctx, i) => (
                    <div
                      key={i}
                      class={`flex items-center justify-between p-3 rounded-xl text-[10px] font-bold border transition-all ${
                        ctx.transferred ? 'bg-md-success/5 border-md-success/20' : 'bg-md-surface-container border-md-outline/10'
                      }`}
                    >
                      <span class="text-md-on-surface-variant/60 uppercase tracking-tight">{ctx.key}</span>
                      <div class="flex items-center gap-3">
                        <span class="text-md-on-surface truncate max-w-[120px] font-mono">{ctx.value}</span>
                        {ctx.transferred ? (
                          <md-icon class="text-md-success text-xs">check_circle</md-icon>
                        ) : (
                          <div class="w-2 h-2 rounded-full bg-md-outline/30" />
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Timing & Error */}
              <div class="space-y-4">
                {selectedHandoff.value.error && (
                  <Card variant="filled" class="bg-md-error/10 border border-md-error/20 p-4">
                    <div class="text-[9px] font-black text-md-error uppercase tracking-widest mb-1">Fault Detected</div>
                    <div class="text-[11px] font-medium text-md-on-error-container">{selectedHandoff.value.error}</div>
                  </Card>
                )}

                <div class="grid grid-cols-2 gap-4 text-[10px] font-black uppercase tracking-tighter text-md-on-surface-variant/40 px-1">
                  <div class="flex items-center gap-2">
                    <md-icon class="text-[10px]">schedule</md-icon>
                    <span>Created {formatTime(selectedHandoff.value.createdAt)}</span>
                  </div>
                  {selectedHandoff.value.completedAt && (
                    <div class="flex items-center gap-2">
                      <md-icon class="text-[10px]">timer</md-icon>
                      <span>Delta: {formatDuration(selectedHandoff.value.startedAt!, selectedHandoff.value.completedAt)}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* Actions */}
              <div class="flex items-center gap-3 pt-4 border-t border-md-outline/10">
                {(selectedHandoff.value.status === 'pending' || selectedHandoff.value.status === 'in_progress') && (
                  <Button variant="tonal" class="flex-1 rounded-xl h-11" onClick$={() => cancelHandoff(selectedHandoff.value!.id)}>
                    Cancel Transit
                  </Button>
                )}
                {selectedHandoff.value.status === 'failed' && (
                  <Button variant="primary" class="flex-1 rounded-xl h-11" icon="refresh" onClick$={() => retryHandoff(selectedHandoff.value!.id)}>
                    Retry Link
                  </Button>
                )}
              </div>
            </div>
          ) : (
            <div class="flex flex-col items-center justify-center h-full text-md-on-surface-variant/20 italic p-12 text-center">
              <md-icon class="text-6xl mb-4">info</md-icon>
              <div class="text-sm font-bold uppercase tracking-widest">Awaiting Node Selection</div>
            </div>
          )}
        </div>
      </div>

      {/* Initiate Modal (Simulated with fixed overlay) */}
      {showInitiateModal.value && (
        <div class="fixed inset-0 bg-md-scrim/60 backdrop-blur-sm flex items-center justify-center z-[70] p-4 animate-in fade-in duration-300">
          <Card variant="elevated" padding="p-6" class="w-full max-w-md bg-md-surface rounded-[2rem] shadow-2xl space-y-6">
            <div class="flex items-center justify-between">
              <h2 class="text-xl font-black tracking-tight text-md-on-surface">Initiate Swarm Handoff</h2>
              <Button variant="tonal" icon="close" class="h-10 w-10 min-w-0 rounded-full" onClick$={() => { showInitiateModal.value = false; }} />
            </div>

            <div class="space-y-4">
              <Input
                label="Task Identity"
                placeholder="Target task summary..."
                value={newHandoff.value.taskTitle}
                onInput$={(_, el) => { newHandoff.value = { ...newHandoff.value, taskTitle: el.value }; }}
              />

              <div class="space-y-1.5">
                <label class="text-[10px] font-black uppercase tracking-widest text-md-on-surface-variant/60 ml-2">Source Node</label>
                <select
                  value={newHandoff.value.fromAgent}
                  class="w-full h-12 px-4 rounded-2xl bg-md-surface-container-high border border-md-outline/20 text-sm font-bold text-md-on-surface"
                  onChange$={(e) => { newHandoff.value = { ...newHandoff.value, fromAgent: (e.target as HTMLSelectElement).value }; }}
                >
                  <option value="">Select Egress Agent</option>
                  {agents.map(a => (
                    <option key={a.id} value={a.id}>{`${a.name} (${a.status})`}</option>
                  ))}
                </select>
              </div>

              <div class="space-y-1.5">
                <label class="text-[10px] font-black uppercase tracking-widest text-md-on-surface-variant/60 ml-2">Target Node</label>
                <select
                  value={newHandoff.value.toAgent}
                  class="w-full h-12 px-4 rounded-2xl bg-md-surface-container-high border border-md-outline/20 text-sm font-bold text-md-on-surface"
                  onChange$={(e) => { newHandoff.value = { ...newHandoff.value, toAgent: (e.target as HTMLSelectElement).value }; }}
                >
                  <option value="">Select Ingress Agent</option>
                  {agents.filter(a => a.id !== newHandoff.value.fromAgent).map(a => (
                    <option key={a.id} value={a.id}>{`${a.name} (${a.status})`}</option>
                  ))}
                </select>
              </div>

              <Input
                label="Transfer Reason"
                placeholder="Escalation / Specialization / Context Shift..."
                value={newHandoff.value.reason}
                onInput$={(_, el) => { newHandoff.value = { ...newHandoff.value, reason: el.value }; }}
              />
            </div>

            <div class="flex gap-3 pt-4">
              <Button variant="tonal" class="flex-1 h-12 rounded-2xl" onClick$={() => { showInitiateModal.value = false; }}>Discard</Button>
              <Button
                variant="primary"
                class="flex-1 h-12 rounded-2xl"
                disabled={!newHandoff.value.fromAgent || !newHandoff.value.toAgent || !newHandoff.value.taskTitle}
                onClick$={initiateHandoff}
              >
                Start Transfer
              </Button>
            </div>
          </Card>
        </div>
      )}

      <div class="p-3 border-t border-md-outline/10 text-[9px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/30 text-center bg-md-surface-container-low">
        {stats.value.active} Transit • {stats.value.completed} Actualized • {stats.value.failed} Faulted
      </div>
    </Card>
  );
});

export default AgentHandoff;