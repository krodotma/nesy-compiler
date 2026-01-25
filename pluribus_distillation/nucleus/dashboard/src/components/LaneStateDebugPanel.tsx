/**
 * LaneStateDebugPanel - Dev Tools for State Inspection
 *
 * Phase 7, Iteration 60 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - State inspection and visualization
 * - Action log with filtering
 * - Performance metrics
 * - Event bus monitor
 * - Export/import state
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
} from '@builder.io/qwik';

// M3 Components - LaneStateDebugPanel
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

import { getGlobalComponentBus } from '../lib/lanes/component-bus';
import type { ComponentEvent } from '../lib/lanes/component-bus';
import { getGlobalUndoRedoManager } from '../lib/lanes/undo-redo';
import { getGlobalEventBridge } from '../lib/lanes/event-bridge';
import type { LanesStoreState, selectStats } from '../lib/lanes/store';

// ============================================================================
// Types
// ============================================================================

export interface DebugAction {
  id: string;
  type: string;
  timestamp: number;
  payload?: unknown;
  source: string;
  duration?: number;
}

export interface LaneStateDebugPanelProps {
  /** Store state to inspect */
  state?: LanesStoreState;
  /** Show by default */
  defaultOpen?: boolean;
  /** Enable action logging */
  logActions?: boolean;
  /** Maximum actions to keep */
  maxActions?: number;
  /** Show performance metrics */
  showPerformance?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function formatTimestamp(ts: number): string {
  return new Date(ts).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    fractionalSecondDigits: 3,
  });
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function truncateJson(obj: unknown, maxLength: number = 100): string {
  const str = JSON.stringify(obj);
  if (str.length <= maxLength) return str;
  return str.slice(0, maxLength) + '...';
}

// ============================================================================
// Component
// ============================================================================

export const LaneStateDebugPanel = component$<LaneStateDebugPanelProps>(({
  state,
  defaultOpen = false,
  logActions = true,
  maxActions = 100,
  showPerformance = true,
}) => {
  // State
  const isOpen = useSignal(defaultOpen);
  const activeTab = useSignal<'state' | 'actions' | 'bus' | 'perf'>('state');
  const actions = useSignal<DebugAction[]>([]);
  const busEvents = useSignal<ComponentEvent[]>([]);
  const actionFilter = useSignal('');
  const stateExpanded = useSignal<Record<string, boolean>>({});

  // Performance metrics
  const perfMetrics = useSignal({
    renderCount: 0,
    lastRenderTime: 0,
    avgRenderTime: 0,
    stateSize: 0,
    actionCount: 0,
    busEventCount: 0,
  });

  // Get singletons
  const bus = getGlobalComponentBus();
  const undoManager = getGlobalUndoRedoManager();
  const eventBridge = getGlobalEventBridge();

  // Subscribe to bus events
  useVisibleTask$(({ cleanup }) => {
    if (!logActions) return;

    // Subscribe to all events
    const unsubBus = bus.subscribe('*', (event) => {
      busEvents.value = [event, ...busEvents.value].slice(0, maxActions);
      perfMetrics.value = {
        ...perfMetrics.value,
        busEventCount: perfMetrics.value.busEventCount + 1,
      };
    });

    // Subscribe to action events
    const unsubAction = bus.subscribe('/.*(update|create|delete|change).*/i' as any, (event) => {
      const action: DebugAction = {
        id: event.id,
        type: event.type,
        timestamp: event.timestamp,
        payload: event.payload,
        source: event.source,
      };
      actions.value = [action, ...actions.value].slice(0, maxActions);
      perfMetrics.value = {
        ...perfMetrics.value,
        actionCount: perfMetrics.value.actionCount + 1,
      };
    });

    cleanup(() => {
      unsubBus();
      unsubAction();
    });
  });

  // Calculate state size
  useVisibleTask$(({ track }) => {
    track(() => state);
    if (state) {
      const stateStr = JSON.stringify(state);
      perfMetrics.value = {
        ...perfMetrics.value,
        stateSize: stateStr.length,
        renderCount: perfMetrics.value.renderCount + 1,
        lastRenderTime: Date.now(),
      };
    }
  });

  // Filtered actions
  const filteredActions = useComputed$(() => {
    if (!actionFilter.value) return actions.value;
    const filter = actionFilter.value.toLowerCase();
    return actions.value.filter(a =>
      a.type.toLowerCase().includes(filter) ||
      a.source.toLowerCase().includes(filter)
    );
  });

  // Export state
  const exportState = $(() => {
    if (!state) return;

    const data = JSON.stringify(state, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `lanes-state-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Import state
  const importState = $(() => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const text = await file.text();
      try {
        const imported = JSON.parse(text);
        console.log('[Debug] Imported state:', imported);
        // Would need to dispatch to store to actually apply
      } catch (err) {
        console.error('[Debug] Failed to parse imported state:', err);
      }
    };
    input.click();
  });

  // Clear actions
  const clearActions = $(() => {
    actions.value = [];
  });

  // Clear bus events
  const clearBusEvents = $(() => {
    busEvents.value = [];
    bus.clearHistory();
  });

  // Toggle state section
  const toggleStateSection = $((key: string) => {
    stateExpanded.value = {
      ...stateExpanded.value,
      [key]: !stateExpanded.value[key],
    };
  });

  // Don't render if closed
  if (!isOpen.value) {
    return (
      <button
        onClick$={() => { isOpen.value = true; }}
        class="fixed bottom-4 right-4 z-50 px-3 py-2 rounded-lg bg-purple-500/20 text-purple-400 border border-purple-500/30 text-xs hover:bg-purple-500/30 transition-colors"
      >
        Debug Panel
      </button>
    );
  }

  return (
    <div class="fixed bottom-4 right-4 z-50 w-96 max-h-[500px] rounded-lg border border-purple-500/30 bg-card shadow-xl overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-2 bg-purple-500/10 border-b border-purple-500/30">
        <div class="flex items-center gap-2">
          <span class="text-[10px] font-bold text-purple-400">DEBUG PANEL</span>
          <span class="text-[8px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
            DEV
          </span>
        </div>
        <button
          onClick$={() => { isOpen.value = false; }}
          class="text-purple-400 hover:text-purple-300"
        >
          \u2715
        </button>
      </div>

      {/* Tabs */}
      <div class="flex border-b border-border/50">
        {(['state', 'actions', 'bus', 'perf'] as const).map(tab => (
          <button
            key={tab}
            onClick$={() => { activeTab.value = tab; }}
            class={`flex-1 p-2 text-[10px] transition-colors ${
              activeTab.value === tab
                ? 'bg-purple-500/10 text-purple-400 border-b-2 border-purple-500'
                : 'text-muted-foreground hover:bg-muted/10'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      <div class="max-h-[350px] overflow-y-auto">
        {/* State Tab */}
        {activeTab.value === 'state' && (
          <div class="p-2 space-y-2">
            {/* Export/Import buttons */}
            <div class="flex items-center gap-2 mb-2">
              <button
                onClick$={exportState}
                class="flex-1 text-[9px] px-2 py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
              >
                Export State
              </button>
              <button
                onClick$={importState}
                class="flex-1 text-[9px] px-2 py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
              >
                Import State
              </button>
            </div>

            {state ? (
              <>
                {/* Data section */}
                <div class="rounded bg-muted/10 border border-border/30">
                  <button
                    onClick$={() => toggleStateSection('data')}
                    class="w-full flex items-center justify-between p-2 text-[10px] hover:bg-muted/10"
                  >
                    <span class="text-cyan-400">data</span>
                    <span class="text-muted-foreground">
                      {state.data ? `${state.data.lanes.length} lanes` : 'null'}
                    </span>
                  </button>
                  {stateExpanded.value['data'] && state.data && (
                    <div class="p-2 border-t border-border/30 text-[9px] text-muted-foreground">
                      <div>version: {state.data.version}</div>
                      <div>updated: {state.data.updated}</div>
                      <div>lanes: {state.data.lanes.length}</div>
                      <div>agents: {state.data.agents.length}</div>
                    </div>
                  )}
                </div>

                {/* UI section */}
                <div class="rounded bg-muted/10 border border-border/30">
                  <button
                    onClick$={() => toggleStateSection('ui')}
                    class="w-full flex items-center justify-between p-2 text-[10px] hover:bg-muted/10"
                  >
                    <span class="text-purple-400">ui</span>
                    <span class="text-muted-foreground">
                      {state.ui.viewMode}
                    </span>
                  </button>
                  {stateExpanded.value['ui'] && (
                    <div class="p-2 border-t border-border/30 text-[9px] text-muted-foreground">
                      <div>selectedLaneId: {state.ui.selectedLaneId || 'null'}</div>
                      <div>viewMode: {state.ui.viewMode}</div>
                      <div>sort: {state.ui.sort.key} ({state.ui.sort.direction})</div>
                      <div>filter: {JSON.stringify(state.ui.filter)}</div>
                    </div>
                  )}
                </div>

                {/* Sync section */}
                <div class="rounded bg-muted/10 border border-border/30">
                  <button
                    onClick$={() => toggleStateSection('sync')}
                    class="w-full flex items-center justify-between p-2 text-[10px] hover:bg-muted/10"
                  >
                    <span class="text-amber-400">sync</span>
                    <span class="text-muted-foreground">
                      {state.syncStatus}
                    </span>
                  </button>
                  {stateExpanded.value['sync'] && (
                    <div class="p-2 border-t border-border/30 text-[9px] text-muted-foreground">
                      <div>status: {state.syncStatus}</div>
                      <div>pending: {state.pendingChanges.length}</div>
                      <div>lastSync: {state.lastSync || 'null'}</div>
                    </div>
                  )}
                </div>

                {/* Undo/Redo section */}
                <div class="rounded bg-muted/10 border border-border/30">
                  <button
                    onClick$={() => toggleStateSection('history')}
                    class="w-full flex items-center justify-between p-2 text-[10px] hover:bg-muted/10"
                  >
                    <span class="text-emerald-400">history</span>
                    <span class="text-muted-foreground">
                      {state.undoStack.length} undo / {state.redoStack.length} redo
                    </span>
                  </button>
                </div>
              </>
            ) : (
              <div class="text-[10px] text-muted-foreground text-center py-4">
                No state provided
              </div>
            )}
          </div>
        )}

        {/* Actions Tab */}
        {activeTab.value === 'actions' && (
          <div class="p-2">
            <div class="flex items-center gap-2 mb-2">
              <input
                type="text"
                value={actionFilter.value}
                onInput$={(e) => { actionFilter.value = (e.target as HTMLInputElement).value; }}
                placeholder="Filter actions..."
                class="flex-1 px-2 py-1 text-[10px] rounded bg-card border border-border/50"
              />
              <button
                onClick$={clearActions}
                class="text-[9px] px-2 py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
              >
                Clear
              </button>
            </div>

            <div class="space-y-1 max-h-[250px] overflow-y-auto">
              {filteredActions.value.length > 0 ? (
                filteredActions.value.map(action => (
                  <div
                    key={action.id}
                    class="p-1.5 rounded bg-muted/10 border border-border/30 text-[9px]"
                  >
                    <div class="flex items-center justify-between">
                      <span class="text-cyan-400">{action.type}</span>
                      <span class="text-muted-foreground/50">
                        {formatTimestamp(action.timestamp)}
                      </span>
                    </div>
                    <div class="text-muted-foreground truncate">
                      {truncateJson(action.payload, 60)}
                    </div>
                  </div>
                ))
              ) : (
                <div class="text-[10px] text-muted-foreground text-center py-4">
                  No actions recorded
                </div>
              )}
            </div>
          </div>
        )}

        {/* Bus Tab */}
        {activeTab.value === 'bus' && (
          <div class="p-2">
            <div class="flex items-center justify-between mb-2">
              <span class="text-[9px] text-muted-foreground">
                {busEvents.value.length} events
              </span>
              <button
                onClick$={clearBusEvents}
                class="text-[9px] px-2 py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
              >
                Clear
              </button>
            </div>

            <div class="space-y-1 max-h-[250px] overflow-y-auto">
              {busEvents.value.length > 0 ? (
                busEvents.value.map(event => (
                  <div
                    key={event.id}
                    class="p-1.5 rounded bg-muted/10 border border-border/30 text-[9px]"
                  >
                    <div class="flex items-center justify-between">
                      <span class="text-purple-400">{event.type}</span>
                      <span class="text-muted-foreground/50">
                        {formatTimestamp(event.timestamp)}
                      </span>
                    </div>
                    <div class="flex items-center justify-between text-[8px]">
                      <span class="text-muted-foreground">from: {event.source}</span>
                    </div>
                  </div>
                ))
              ) : (
                <div class="text-[10px] text-muted-foreground text-center py-4">
                  No bus events
                </div>
              )}
            </div>
          </div>
        )}

        {/* Performance Tab */}
        {activeTab.value === 'perf' && showPerformance && (
          <div class="p-2 space-y-2">
            <div class="grid grid-cols-2 gap-2">
              <div class="p-2 rounded bg-muted/10 border border-border/30">
                <div class="text-[9px] text-muted-foreground">Render Count</div>
                <div class="text-sm font-bold text-foreground">
                  {perfMetrics.value.renderCount}
                </div>
              </div>
              <div class="p-2 rounded bg-muted/10 border border-border/30">
                <div class="text-[9px] text-muted-foreground">State Size</div>
                <div class="text-sm font-bold text-foreground">
                  {formatBytes(perfMetrics.value.stateSize)}
                </div>
              </div>
              <div class="p-2 rounded bg-muted/10 border border-border/30">
                <div class="text-[9px] text-muted-foreground">Actions</div>
                <div class="text-sm font-bold text-foreground">
                  {perfMetrics.value.actionCount}
                </div>
              </div>
              <div class="p-2 rounded bg-muted/10 border border-border/30">
                <div class="text-[9px] text-muted-foreground">Bus Events</div>
                <div class="text-sm font-bold text-foreground">
                  {perfMetrics.value.busEventCount}
                </div>
              </div>
            </div>

            <div class="p-2 rounded bg-muted/10 border border-border/30">
              <div class="text-[9px] text-muted-foreground mb-1">Undo/Redo Stack</div>
              <div class="text-[10px]">
                <span class="text-cyan-400">Undo: {undoManager.getUndoStack().length}</span>
                <span class="text-muted-foreground mx-2">|</span>
                <span class="text-purple-400">Redo: {undoManager.getRedoStack().length}</span>
              </div>
            </div>

            <div class="p-2 rounded bg-muted/10 border border-border/30">
              <div class="text-[9px] text-muted-foreground mb-1">Event Bridge</div>
              <div class="text-[10px]">
                <span class="text-amber-400">Pending: {eventBridge.getPendingCount()}</span>
              </div>
            </div>

            <div class="p-2 rounded bg-muted/10 border border-border/30">
              <div class="text-[9px] text-muted-foreground mb-1">Component Bus</div>
              <div class="text-[10px]">
                <span class="text-emerald-400">Subscriptions: {bus.getSubscriptionCount()}</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[8px] text-muted-foreground text-center">
        Lanes Widget Debug Panel v1.0 | Phase 7 Iteration 60
      </div>
    </div>
  );
});

export default LaneStateDebugPanel;
