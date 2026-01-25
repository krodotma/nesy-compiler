/**
 * AgentAssignmentUI - Lane Agent Assignment Component
 *
 * Iteration 17 of OITERATE lanes-widget-enhancement
 * Provides UI for reassigning lane ownership between agents.
 *
 * Features:
 * - Dropdown to select/reassign lane owner from available agents
 * - Shows agent status (active/idle/offline) with visual indicators
 * - Displays agent's current lane assignment if any
 * - Emits bus event on assignment: topic "operator.lanes.reassign"
 * - Confirmation dialog before reassignment
 *
 * @see IsotopeLanesGrid.tsx for lane grid integration
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  noSerialize,
  type NoSerialize,
  type QRL,
} from '@builder.io/qwik';
import { createBusClient, type BusClient } from '../lib/bus/bus-client';

// ============================================================================
// Types
// ============================================================================

export interface Agent {
  id: string;
  status: 'active' | 'idle' | 'offline';
  lane: string | null;
  last_seen: string;
}

export interface AgentAssignmentUIProps {
  /** Lane ID being assigned */
  laneId: string;
  /** Human-readable lane name */
  laneName: string;
  /** Current owner agent ID */
  currentOwner: string;
  /** List of available agents */
  agents: Agent[];
  /** Callback when assignment is made */
  onAssign$?: QRL<(newOwner: string) => void>;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Get CSS classes for agent status indicator
 */
function statusIndicatorClass(status: Agent['status']): string {
  switch (status) {
    case 'active':
      return 'bg-emerald-400 shadow-emerald-400/50';
    case 'idle':
      return 'bg-amber-400 shadow-amber-400/50';
    case 'offline':
      return 'bg-red-400 shadow-red-400/50';
    default:
      return 'bg-muted-foreground';
  }
}

/**
 * Get CSS classes for agent status badge
 */
function statusBadgeClass(status: Agent['status']): string {
  switch (status) {
    case 'active':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'idle':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'offline':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    default:
      return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

/**
 * Format last seen timestamp to human-readable relative time
 */
function formatLastSeen(isoString: string): string {
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffSecs < 60) return `${diffSecs}s ago`;
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return `${diffDays}d ago`;
  } catch {
    return 'unknown';
  }
}

// ============================================================================
// Component
// ============================================================================

export const AgentAssignmentUI = component$<AgentAssignmentUIProps>(
  ({ laneId, laneName, currentOwner, agents, onAssign$ }) => {
    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    const isOpen = useSignal(false);
    const selectedAgent = useSignal<string | null>(null);
    const showConfirmDialog = useSignal(false);
    const isSubmitting = useSignal(false);
    const errorMessage = useSignal<string | null>(null);

    const state = useStore<{
      busClient: NoSerialize<BusClient> | null;
      connected: boolean;
    }>({
      busClient: null,
      connected: false,
    });

    // -------------------------------------------------------------------------
    // Bus Client Setup
    // -------------------------------------------------------------------------
    useVisibleTask$(async ({ cleanup }) => {
      try {
        const client = createBusClient({ platform: 'browser' });
        await client.connect();
        state.busClient = noSerialize(client);
        state.connected = true;
      } catch (err) {
        console.warn('[AgentAssignmentUI] Bus connection failed:', err);
        state.connected = false;
      }

      cleanup(() => {
        if (state.busClient) {
          (state.busClient as unknown as BusClient).disconnect();
          state.busClient = null;
        }
      });
    });

    // -------------------------------------------------------------------------
    // Actions
    // -------------------------------------------------------------------------
    const openDropdown = $(() => {
      isOpen.value = true;
      errorMessage.value = null;
    });

    const closeDropdown = $(() => {
      isOpen.value = false;
      selectedAgent.value = null;
    });

    const selectAgent = $((agentId: string) => {
      if (agentId === currentOwner) {
        // Same owner selected, just close
        closeDropdown();
        return;
      }
      selectedAgent.value = agentId;
      showConfirmDialog.value = true;
    });

    const cancelConfirm = $(() => {
      showConfirmDialog.value = false;
      selectedAgent.value = null;
    });

    const confirmAssignment = $(async () => {
      if (!selectedAgent.value) return;

      isSubmitting.value = true;
      errorMessage.value = null;

      try {
        // Emit bus event for lane reassignment
        const client = state.busClient as unknown as BusClient | null;
        if (client) {
          await client.publish({
            topic: 'operator.lanes.reassign',
            kind: 'request',
            level: 'info',
            actor: 'agent-assignment-ui',
            data: {
              lane_id: laneId,
              lane_name: laneName,
              previous_owner: currentOwner,
              new_owner: selectedAgent.value,
              timestamp: new Date().toISOString(),
            },
          });
        }

        // Call the callback if provided
        if (onAssign$) {
          await onAssign$(selectedAgent.value);
        }

        // Reset state
        showConfirmDialog.value = false;
        isOpen.value = false;
        selectedAgent.value = null;
      } catch (err: any) {
        errorMessage.value = err?.message || 'Failed to assign agent';
      } finally {
        isSubmitting.value = false;
      }
    });

    // -------------------------------------------------------------------------
    // Computed Values
    // -------------------------------------------------------------------------
    const currentAgent = agents.find((a) => a.id === currentOwner);
    const availableAgents = agents.filter((a) => a.status !== 'offline' || a.id === currentOwner);
    const offlineAgents = agents.filter((a) => a.status === 'offline' && a.id !== currentOwner);
    const selectedAgentData = selectedAgent.value
      ? agents.find((a) => a.id === selectedAgent.value)
      : null;

    // -------------------------------------------------------------------------
    // Render
    // -------------------------------------------------------------------------
    return (
      <div class="relative inline-block">
        {/* Current Owner Button (Trigger) */}
        <button
          onClick$={openDropdown}
          class="flex items-center gap-2 px-2 py-1 rounded border border-border/30 bg-muted/20 hover:bg-muted/40 transition-colors text-[11px]"
          title={`Current owner: @${currentOwner}${currentAgent ? ` (${currentAgent.status})` : ''}`}
        >
          {/* Status indicator dot */}
          {currentAgent && (
            <span
              class={`w-2 h-2 rounded-full shadow-sm ${statusIndicatorClass(currentAgent.status)}`}
            />
          )}
          <span class="text-muted-foreground">@{currentOwner}</span>
          <svg
            class="w-3 h-3 text-muted-foreground"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              stroke-linecap="round"
              stroke-linejoin="round"
              stroke-width="2"
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </button>

        {/* Dropdown Menu */}
        {isOpen.value && (
          <>
            {/* Backdrop */}
            <div
              class="fixed inset-0 z-40"
              onClick$={closeDropdown}
            />

            {/* Dropdown Panel */}
            <div class="absolute right-0 mt-1 z-50 w-64 rounded-lg border border-border bg-card shadow-lg overflow-hidden">
              {/* Header */}
              <div class="px-3 py-2 border-b border-border/50 bg-muted/20">
                <div class="text-[10px] text-muted-foreground uppercase tracking-wide">
                  Reassign Lane
                </div>
                <div class="text-xs text-foreground truncate">{laneName}</div>
              </div>

              {/* Agent List */}
              <div class="max-h-64 overflow-y-auto">
                {/* Available Agents Section */}
                {availableAgents.length > 0 && (
                  <div class="p-1">
                    <div class="px-2 py-1 text-[9px] text-muted-foreground uppercase tracking-wide">
                      Available
                    </div>
                    {availableAgents.map((agent) => (
                      <button
                        key={agent.id}
                        onClick$={() => selectAgent(agent.id)}
                        class={`w-full flex items-center gap-2 px-2 py-1.5 rounded text-left transition-colors ${
                          agent.id === currentOwner
                            ? 'bg-primary/10 border border-primary/30'
                            : 'hover:bg-muted/40'
                        }`}
                      >
                        {/* Status dot */}
                        <span
                          class={`w-2 h-2 rounded-full shadow-sm flex-shrink-0 ${statusIndicatorClass(agent.status)}`}
                        />

                        {/* Agent info */}
                        <div class="flex-1 min-w-0">
                          <div class="flex items-center gap-1.5">
                            <span class="text-xs text-foreground">@{agent.id}</span>
                            {agent.id === currentOwner && (
                              <span class="text-[8px] px-1 py-0.5 rounded bg-primary/20 text-primary border border-primary/30">
                                current
                              </span>
                            )}
                          </div>
                          <div class="flex items-center gap-2 text-[9px] text-muted-foreground">
                            <span class={`px-1 py-0.5 rounded border ${statusBadgeClass(agent.status)}`}>
                              {agent.status}
                            </span>
                            {agent.lane && agent.lane !== laneId && (
                              <span class="truncate" title={`Currently assigned to: ${agent.lane}`}>
                                [{agent.lane.slice(0, 12)}...]
                              </span>
                            )}
                          </div>
                        </div>

                        {/* Last seen */}
                        <span class="text-[9px] text-muted-foreground/70 flex-shrink-0">
                          {formatLastSeen(agent.last_seen)}
                        </span>
                      </button>
                    ))}
                  </div>
                )}

                {/* Offline Agents Section */}
                {offlineAgents.length > 0 && (
                  <div class="p-1 border-t border-border/30">
                    <div class="px-2 py-1 text-[9px] text-muted-foreground/70 uppercase tracking-wide">
                      Offline
                    </div>
                    {offlineAgents.map((agent) => (
                      <button
                        key={agent.id}
                        onClick$={() => selectAgent(agent.id)}
                        class="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left opacity-60 hover:opacity-80 hover:bg-muted/20 transition-all"
                      >
                        <span
                          class={`w-2 h-2 rounded-full flex-shrink-0 ${statusIndicatorClass(agent.status)}`}
                        />
                        <div class="flex-1 min-w-0">
                          <span class="text-xs text-foreground/70">@{agent.id}</span>
                          <div class="text-[9px] text-muted-foreground/60">
                            last: {formatLastSeen(agent.last_seen)}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}

                {/* Empty state */}
                {agents.length === 0 && (
                  <div class="px-3 py-4 text-center text-[10px] text-muted-foreground">
                    No agents available
                  </div>
                )}
              </div>

              {/* Footer */}
              <div class="px-3 py-2 border-t border-border/50 bg-muted/10">
                <div class="flex items-center justify-between text-[9px] text-muted-foreground">
                  <span>{agents.length} agents total</span>
                  <span class={state.connected ? 'text-emerald-400' : 'text-red-400'}>
                    bus: {state.connected ? 'connected' : 'disconnected'}
                  </span>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Confirmation Dialog */}
        {showConfirmDialog.value && selectedAgentData && (
          <>
            {/* Dialog Backdrop */}
            <div
              class="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm"
              onClick$={cancelConfirm}
            />

            {/* Dialog Panel */}
            <div class="fixed left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 z-50 w-80 rounded-lg border border-border bg-card shadow-xl">
              {/* Dialog Header */}
              <div class="px-4 py-3 border-b border-border/50">
                <h3 class="text-sm font-semibold text-foreground">Confirm Assignment</h3>
                <p class="text-[10px] text-muted-foreground mt-0.5">
                  Lane ownership will be transferred
                </p>
              </div>

              {/* Dialog Body */}
              <div class="px-4 py-4 space-y-3">
                {/* Lane info */}
                <div class="text-[11px]">
                  <div class="text-muted-foreground">Lane:</div>
                  <div class="text-foreground font-medium truncate">{laneName}</div>
                  <div class="text-[9px] text-muted-foreground/70 mono">{laneId}</div>
                </div>

                {/* Transfer visualization */}
                <div class="flex items-center gap-2 py-2">
                  {/* From */}
                  <div class="flex-1 p-2 rounded border border-border/30 bg-muted/10">
                    <div class="text-[9px] text-muted-foreground uppercase">From</div>
                    <div class="flex items-center gap-1.5 mt-1">
                      {currentAgent && (
                        <span
                          class={`w-2 h-2 rounded-full ${statusIndicatorClass(currentAgent.status)}`}
                        />
                      )}
                      <span class="text-xs text-foreground">@{currentOwner}</span>
                    </div>
                  </div>

                  {/* Arrow */}
                  <svg class="w-5 h-5 text-muted-foreground flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7l5 5m0 0l-5 5m5-5H6" />
                  </svg>

                  {/* To */}
                  <div class="flex-1 p-2 rounded border border-primary/30 bg-primary/10">
                    <div class="text-[9px] text-muted-foreground uppercase">To</div>
                    <div class="flex items-center gap-1.5 mt-1">
                      <span
                        class={`w-2 h-2 rounded-full ${statusIndicatorClass(selectedAgentData.status)}`}
                      />
                      <span class="text-xs text-foreground">@{selectedAgentData.id}</span>
                    </div>
                  </div>
                </div>

                {/* Warning for offline agent */}
                {selectedAgentData.status === 'offline' && (
                  <div class="flex items-start gap-2 p-2 rounded bg-amber-500/10 border border-amber-500/30">
                    <svg class="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <div class="text-[10px] text-amber-300">
                      <strong>Warning:</strong> This agent is currently offline.
                      The lane may remain unworked until the agent comes back online.
                    </div>
                  </div>
                )}

                {/* Warning for already assigned agent */}
                {selectedAgentData.lane && selectedAgentData.lane !== laneId && (
                  <div class="flex items-start gap-2 p-2 rounded bg-blue-500/10 border border-blue-500/30">
                    <svg class="w-4 h-4 text-blue-400 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div class="text-[10px] text-blue-300">
                      <strong>Note:</strong> This agent is already assigned to lane "{selectedAgentData.lane}".
                      They will now work on both lanes.
                    </div>
                  </div>
                )}

                {/* Error message */}
                {errorMessage.value && (
                  <div class="text-[10px] text-red-400 p-2 rounded bg-red-500/10 border border-red-500/30">
                    {errorMessage.value}
                  </div>
                )}
              </div>

              {/* Dialog Footer */}
              <div class="px-4 py-3 border-t border-border/50 flex items-center justify-end gap-2">
                <button
                  onClick$={cancelConfirm}
                  disabled={isSubmitting.value}
                  class="px-3 py-1.5 text-[11px] rounded border border-border/50 bg-muted/20 hover:bg-muted/40 text-muted-foreground transition-colors disabled:opacity-50"
                >
                  Cancel
                </button>
                <button
                  onClick$={confirmAssignment}
                  disabled={isSubmitting.value}
                  class="px-3 py-1.5 text-[11px] rounded bg-primary hover:bg-primary/90 text-primary-foreground font-medium transition-colors disabled:opacity-50 flex items-center gap-2"
                >
                  {isSubmitting.value && (
                    <svg class="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                  )}
                  {isSubmitting.value ? 'Assigning...' : 'Confirm'}
                </button>
              </div>
            </div>
          </>
        )}
      </div>
    );
  }
);

export default AgentAssignmentUI;
