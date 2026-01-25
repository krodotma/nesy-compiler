/**
 * LaneRealtime - Real-time lane updates via WebSocket
 *
 * Phase 4, Iteration 31 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - WebSocket subscription to lane changes
 * - Animated transitions on update
 * - Optimistic UI updates
 * - Connection status indicator
 * - Reconnection handling
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

// M3 Components - LaneRealtime
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/progress/circular-progress.js';

// ============================================================================
// Types
// ============================================================================

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface RealtimeLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  lastUpdate: string;
}

export interface LaneUpdateEvent {
  type: 'update' | 'create' | 'delete';
  laneId: string;
  data?: Partial<RealtimeLane>;
  timestamp: string;
}

export interface LaneRealtimeProps {
  /** Initial lanes */
  lanes: RealtimeLane[];
  /** WebSocket URL */
  wsUrl?: string;
  /** Topics to subscribe to */
  topics?: string[];
  /** Callback when lane updates */
  onLaneUpdate$?: QRL<(event: LaneUpdateEvent) => void>;
  /** Show connection status */
  showConnectionStatus?: boolean;
  /** Auto-reconnect */
  autoReconnect?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'yellow': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'red': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getConnectionColor(status: ConnectionStatus): string {
  switch (status) {
    case 'connected': return 'bg-emerald-400';
    case 'connecting': return 'bg-amber-400 animate-pulse';
    case 'disconnected': return 'bg-red-400';
    case 'error': return 'bg-red-400';
    default: return 'bg-gray-400';
  }
}

function formatTime(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return dateStr;
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneRealtime = component$<LaneRealtimeProps>(({
  lanes: initialLanes,
  wsUrl = 'wss://kroma.live/ws/bus',
  topics = ['operator.lanes.*'],
  onLaneUpdate$,
  showConnectionStatus = true,
  autoReconnect = true,
}) => {
  // State
  const lanes = useSignal<RealtimeLane[]>(initialLanes);
  const recentUpdates = useSignal<LaneUpdateEvent[]>([]);
  const animatingLaneIds = useSignal<Set<string>>(new Set());

  const state = useStore<{
    connectionStatus: ConnectionStatus;
    lastMessage: string | null;
    reconnectAttempts: number;
    ws: NoSerialize<WebSocket> | null;
  }>({
    connectionStatus: 'disconnected',
    lastMessage: null,
    reconnectAttempts: 0,
    ws: null,
  });

  // Connect to WebSocket
  const connect = $(() => {
    if (typeof window === 'undefined') return;

    state.connectionStatus = 'connecting';

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        state.connectionStatus = 'connected';
        state.reconnectAttempts = 0;

        // Subscribe to topics
        topics.forEach(topic => {
          ws.send(JSON.stringify({ type: 'subscribe', topic }));
        });
      };

      ws.onmessage = async (event) => {
        try {
          const message = JSON.parse(event.data);
          state.lastMessage = event.data;

          if (message.topic?.startsWith('operator.lanes.')) {
            const updateEvent: LaneUpdateEvent = {
              type: message.data?.action || 'update',
              laneId: message.data?.lane_id,
              data: message.data,
              timestamp: message.timestamp || new Date().toISOString(),
            };

            // Apply update
            if (updateEvent.laneId && updateEvent.data) {
              applyUpdate(updateEvent);
            }

            // Track recent updates
            recentUpdates.value = [updateEvent, ...recentUpdates.value].slice(0, 10);

            if (onLaneUpdate$) {
              await onLaneUpdate$(updateEvent);
            }
          }
        } catch (e) {
          console.warn('[LaneRealtime] Failed to parse message:', e);
        }
      };

      ws.onclose = () => {
        state.connectionStatus = 'disconnected';
        state.ws = null;

        // Auto-reconnect
        if (autoReconnect && state.reconnectAttempts < 5) {
          state.reconnectAttempts++;
          setTimeout(() => connect(), 1000 * state.reconnectAttempts);
        }
      };

      ws.onerror = () => {
        state.connectionStatus = 'error';
      };

      state.ws = noSerialize(ws);
    } catch (e) {
      state.connectionStatus = 'error';
      console.error('[LaneRealtime] WebSocket error:', e);
    }
  });

  // Apply update to lanes
  const applyUpdate = $((event: LaneUpdateEvent) => {
    const laneIdx = lanes.value.findIndex(l => l.id === event.laneId);

    if (event.type === 'delete' && laneIdx !== -1) {
      lanes.value = lanes.value.filter(l => l.id !== event.laneId);
      return;
    }

    if (event.type === 'create' && laneIdx === -1 && event.data) {
      const newLane: RealtimeLane = {
        id: event.laneId,
        name: event.data.name || 'New Lane',
        owner: event.data.owner || 'unknown',
        status: event.data.status || 'yellow',
        wip_pct: event.data.wip_pct || 0,
        lastUpdate: event.timestamp,
      };
      lanes.value = [...lanes.value, newLane];
      triggerAnimation(event.laneId);
      return;
    }

    if (event.type === 'update' && laneIdx !== -1 && event.data) {
      const updated = [...lanes.value];
      updated[laneIdx] = {
        ...updated[laneIdx],
        ...event.data,
        lastUpdate: event.timestamp,
      };
      lanes.value = updated;
      triggerAnimation(event.laneId);
    }
  });

  // Trigger animation for lane
  const triggerAnimation = $((laneId: string) => {
    const newSet = new Set(animatingLaneIds.value);
    newSet.add(laneId);
    animatingLaneIds.value = newSet;

    setTimeout(() => {
      const cleared = new Set(animatingLaneIds.value);
      cleared.delete(laneId);
      animatingLaneIds.value = cleared;
    }, 1000);
  });

  // Disconnect
  const disconnect = $(() => {
    if (state.ws) {
      (state.ws as unknown as WebSocket).close();
      state.ws = null;
    }
    state.connectionStatus = 'disconnected';
  });

  // Connect on mount
  useVisibleTask$(({ cleanup }) => {
    connect();
    cleanup(() => disconnect());
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">REAL-TIME LANES</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {lanes.value.length} lanes
          </span>
        </div>

        {/* Connection status */}
        {showConnectionStatus && (
          <div class="flex items-center gap-2">
            <div class={`w-2 h-2 rounded-full ${getConnectionColor(state.connectionStatus)}`} />
            <span class="text-[9px] text-muted-foreground capitalize">
              {state.connectionStatus}
            </span>
            {state.connectionStatus === 'disconnected' && (
              <button
                onClick$={connect}
                class="text-[9px] px-2 py-0.5 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
              >
                Reconnect
              </button>
            )}
          </div>
        )}
      </div>

      {/* Lane list */}
      <div class="max-h-[300px] overflow-y-auto">
        {lanes.value.map(lane => {
          const isAnimating = animatingLaneIds.value.has(lane.id);

          return (
            <div
              key={lane.id}
              class={`p-3 border-b border-border/30 transition-all duration-500 ${
                isAnimating ? 'bg-primary/10 scale-[1.02]' : ''
              }`}
            >
              <div class="flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <span class={`text-[10px] px-1.5 py-0.5 rounded border ${getStatusColor(lane.status)}`}>
                    {lane.status}
                  </span>
                  <span class="text-xs font-medium text-foreground">{lane.name}</span>
                </div>
                <div class="flex items-center gap-2">
                  <span class={`text-[10px] font-bold ${
                    lane.wip_pct >= 90 ? 'text-emerald-400' :
                    lane.wip_pct >= 50 ? 'text-cyan-400' :
                    'text-amber-400'
                  }`}>
                    {lane.wip_pct}%
                  </span>
                  {isAnimating && (
                    <span class="text-[9px] text-primary animate-pulse">● Updated</span>
                  )}
                </div>
              </div>
              <div class="mt-1 flex items-center justify-between text-[9px] text-muted-foreground">
                <span>@{lane.owner}</span>
                <span>{formatTime(lane.lastUpdate)}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Recent updates */}
      {recentUpdates.value.length > 0 && (
        <div class="p-2 border-t border-border/50 bg-muted/5">
          <div class="text-[9px] text-muted-foreground mb-1">Recent Updates</div>
          <div class="space-y-1 max-h-[80px] overflow-y-auto">
            {recentUpdates.value.slice(0, 5).map((update, i) => (
              <div key={i} class="text-[9px] flex items-center gap-2">
                <span class={`px-1 py-0.5 rounded ${
                  update.type === 'create' ? 'bg-emerald-500/20 text-emerald-400' :
                  update.type === 'delete' ? 'bg-red-500/20 text-red-400' :
                  'bg-blue-500/20 text-blue-400'
                }`}>
                  {update.type}
                </span>
                <span class="text-muted-foreground">{update.laneId?.slice(0, 12)}...</span>
                <span class="text-muted-foreground/50 ml-auto">{formatTime(update.timestamp)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        Subscribed to: {topics.join(', ')}
      </div>
    </div>
  );
});

export default LaneRealtime;
