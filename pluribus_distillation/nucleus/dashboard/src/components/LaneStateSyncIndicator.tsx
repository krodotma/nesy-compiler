/**
 * LaneStateSyncIndicator - Sync Status Indicator
 *
 * Phase 7, Iteration 59 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Visual sync status indicator
 * - Connection state display
 * - Pending changes count
 * - Conflict resolution UI
 * - Last sync timestamp
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  $,
} from '@builder.io/qwik';

// M3 Components - LaneStateSyncIndicator
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/progress/circular-progress.js';

import { getGlobalEventBridge } from '../lib/lanes/event-bridge';
import { getGlobalComponentBus } from '../lib/lanes/component-bus';

// ============================================================================
// Types
// ============================================================================

export type SyncStatus = 'synced' | 'syncing' | 'pending' | 'error' | 'offline';

export interface LaneStateSyncIndicatorProps {
  /** Current sync status */
  status?: SyncStatus;
  /** Is connected to server */
  connected?: boolean;
  /** Is attempting to reconnect */
  reconnecting?: boolean;
  /** Number of pending changes */
  pendingCount?: number;
  /** Last successful sync time */
  lastSync?: string | null;
  /** Show expanded details */
  showDetails?: boolean;
  /** Compact mode */
  compact?: boolean;
  /** Callback for retry */
  onRetry$?: () => void;
  /** Callback for force sync */
  onForceSync$?: () => void;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: SyncStatus): {
  bg: string;
  text: string;
  border: string;
  pulse: boolean;
} {
  switch (status) {
    case 'synced':
      return {
        bg: 'bg-emerald-500/20',
        text: 'text-emerald-400',
        border: 'border-emerald-500/30',
        pulse: false,
      };
    case 'syncing':
      return {
        bg: 'bg-blue-500/20',
        text: 'text-blue-400',
        border: 'border-blue-500/30',
        pulse: true,
      };
    case 'pending':
      return {
        bg: 'bg-amber-500/20',
        text: 'text-amber-400',
        border: 'border-amber-500/30',
        pulse: true,
      };
    case 'error':
      return {
        bg: 'bg-red-500/20',
        text: 'text-red-400',
        border: 'border-red-500/30',
        pulse: false,
      };
    case 'offline':
      return {
        bg: 'bg-gray-500/20',
        text: 'text-gray-400',
        border: 'border-gray-500/30',
        pulse: false,
      };
    default:
      return {
        bg: 'bg-muted/20',
        text: 'text-muted-foreground',
        border: 'border-border/30',
        pulse: false,
      };
  }
}

function getStatusLabel(status: SyncStatus): string {
  switch (status) {
    case 'synced': return 'Synced';
    case 'syncing': return 'Syncing...';
    case 'pending': return 'Pending';
    case 'error': return 'Error';
    case 'offline': return 'Offline';
    default: return 'Unknown';
  }
}

function getStatusIcon(status: SyncStatus): string {
  switch (status) {
    case 'synced': return '\u2713';
    case 'syncing': return '\u21BB';
    case 'pending': return '\u2022';
    case 'error': return '!';
    case 'offline': return '\u2715';
    default: return '?';
  }
}

function formatLastSync(lastSync: string | null): string {
  if (!lastSync) return 'Never';

  const date = new Date(lastSync);
  const now = new Date();
  const diff = now.getTime() - date.getTime();

  if (diff < 60000) return 'Just now';
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m ago`;
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h ago`;
  return date.toLocaleDateString();
}

// ============================================================================
// Component
// ============================================================================

export const LaneStateSyncIndicator = component$<LaneStateSyncIndicatorProps>(({
  status: propStatus,
  connected: propConnected,
  reconnecting: propReconnecting,
  pendingCount: propPendingCount,
  lastSync: propLastSync,
  showDetails = false,
  compact = false,
  onRetry$,
  onForceSync$,
}) => {
  // Internal state (used if props not provided)
  const internalStatus = useSignal<SyncStatus>('synced');
  const internalConnected = useSignal(true);
  const internalReconnecting = useSignal(false);
  const internalPendingCount = useSignal(0);
  const internalLastSync = useSignal<string | null>(null);

  // Use props or internal state
  const status = propStatus ?? internalStatus.value;
  const connected = propConnected ?? internalConnected.value;
  const reconnecting = propReconnecting ?? internalReconnecting.value;
  const pendingCount = propPendingCount ?? internalPendingCount.value;
  const lastSync = propLastSync ?? internalLastSync.value;

  const detailsExpanded = useSignal(showDetails);

  // Subscribe to event bridge for updates
  useVisibleTask$(({ cleanup }) => {
    if (propStatus !== undefined) return; // Skip if using props

    const bridge = getGlobalEventBridge();
    const bus = getGlobalComponentBus();

    // Track pending count
    const updatePending = () => {
      internalPendingCount.value = bridge.getPendingCount();
      if (internalPendingCount.value > 0) {
        internalStatus.value = 'pending';
      }
    };

    // Subscribe to bridge events
    const unsubBridge = bridge.subscribe((event) => {
      if (event.confirmed) {
        internalLastSync.value = new Date().toISOString();
        updatePending();
        if (internalPendingCount.value === 0) {
          internalStatus.value = 'synced';
        }
      }
    });

    // Subscribe to sync events
    const unsubSync = bus.subscribe('sync.started', () => {
      internalStatus.value = 'syncing';
    });

    const unsubSyncDone = bus.subscribe('sync.completed', (event) => {
      if (event.payload.success) {
        internalStatus.value = 'synced';
        internalLastSync.value = new Date().toISOString();
      } else {
        internalStatus.value = 'error';
      }
    });

    // Initial pending count
    updatePending();

    cleanup(() => {
      unsubBridge();
      unsubSync();
      unsubSyncDone();
    });
  });

  // Handlers
  const handleRetry = $(() => {
    if (onRetry$) {
      onRetry$();
    }
  });

  const handleForceSync = $(() => {
    if (onForceSync$) {
      onForceSync$();
    }
  });

  const colors = getStatusColor(status);

  // Compact mode - just show indicator dot
  if (compact) {
    return (
      <div
        class="flex items-center gap-1.5"
        title={`${getStatusLabel(status)} ${pendingCount > 0 ? `(${pendingCount} pending)` : ''}`}
      >
        <div
          class={`w-2 h-2 rounded-full ${colors.bg} ${colors.pulse ? 'animate-pulse' : ''}`}
        />
        {pendingCount > 0 && (
          <span class="text-[9px] text-muted-foreground">{pendingCount}</span>
        )}
      </div>
    );
  }

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Main status bar */}
      <div
        class="flex items-center justify-between p-2 cursor-pointer hover:bg-muted/5 transition-colors"
        onClick$={() => { detailsExpanded.value = !detailsExpanded.value; }}
      >
        <div class="flex items-center gap-2">
          {/* Status indicator */}
          <div
            class={`w-6 h-6 rounded-full flex items-center justify-center ${colors.bg} ${colors.pulse ? 'animate-pulse' : ''}`}
          >
            <span class={`text-xs ${colors.text}`}>
              {getStatusIcon(status)}
            </span>
          </div>

          {/* Status text */}
          <div>
            <div class={`text-xs font-medium ${colors.text}`}>
              {getStatusLabel(status)}
            </div>
            <div class="text-[9px] text-muted-foreground">
              Last sync: {formatLastSync(lastSync)}
            </div>
          </div>
        </div>

        {/* Right side indicators */}
        <div class="flex items-center gap-2">
          {/* Connection indicator */}
          <div class="flex items-center gap-1">
            <div
              class={`w-1.5 h-1.5 rounded-full ${
                connected ? 'bg-emerald-500' : reconnecting ? 'bg-amber-500 animate-pulse' : 'bg-red-500'
              }`}
            />
            <span class="text-[9px] text-muted-foreground">
              {connected ? 'Connected' : reconnecting ? 'Reconnecting' : 'Disconnected'}
            </span>
          </div>

          {/* Pending count badge */}
          {pendingCount > 0 && (
            <span class={`text-[9px] px-1.5 py-0.5 rounded border ${colors.bg} ${colors.text} ${colors.border}`}>
              {pendingCount} pending
            </span>
          )}

          {/* Expand indicator */}
          <span class="text-[9px] text-muted-foreground">
            {detailsExpanded.value ? '\u25BC' : '\u25B6'}
          </span>
        </div>
      </div>

      {/* Expanded details */}
      {detailsExpanded.value && (
        <div class="border-t border-border/50 p-2 space-y-2">
          {/* Connection details */}
          <div class="flex items-center justify-between text-[10px]">
            <span class="text-muted-foreground">WebSocket:</span>
            <span class={connected ? 'text-emerald-400' : 'text-red-400'}>
              {connected ? 'Connected' : reconnecting ? 'Reconnecting...' : 'Disconnected'}
            </span>
          </div>

          <div class="flex items-center justify-between text-[10px]">
            <span class="text-muted-foreground">Pending Changes:</span>
            <span class={pendingCount > 0 ? 'text-amber-400' : 'text-foreground'}>
              {pendingCount}
            </span>
          </div>

          <div class="flex items-center justify-between text-[10px]">
            <span class="text-muted-foreground">Last Sync:</span>
            <span class="text-foreground">
              {lastSync ? new Date(lastSync).toLocaleString() : 'Never'}
            </span>
          </div>

          {/* Status-specific messages */}
          {status === 'error' && (
            <div class="p-2 rounded bg-red-500/10 border border-red-500/30 text-[10px] text-red-300">
              Sync failed. Check your connection and try again.
            </div>
          )}

          {status === 'offline' && (
            <div class="p-2 rounded bg-gray-500/10 border border-gray-500/30 text-[10px] text-gray-300">
              Working offline. Changes will sync when connected.
            </div>
          )}

          {/* Action buttons */}
          <div class="flex items-center gap-2 pt-2">
            {(status === 'error' || status === 'offline') && onRetry$ && (
              <button
                onClick$={handleRetry}
                class="flex-1 text-[10px] px-2 py-1.5 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
              >
                Retry Connection
              </button>
            )}

            {pendingCount > 0 && onForceSync$ && (
              <button
                onClick$={handleForceSync}
                class="flex-1 text-[10px] px-2 py-1.5 rounded bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 transition-colors"
              >
                Force Sync ({pendingCount})
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
});

export default LaneStateSyncIndicator;
