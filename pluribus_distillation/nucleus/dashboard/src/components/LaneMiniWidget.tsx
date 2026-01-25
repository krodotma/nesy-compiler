/**
 * LaneMiniWidget - Compact lane widgets for sidebars and embeds
 *
 * Phase 4, Iteration 30 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Mini lane status widget for sidebar
 * - Embeddable status badges
 * - Slack/Discord embed format
 * - Configurable size/style
 * - Real-time status indicator
 */

import {
  component$,
  useComputed$,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface MiniLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  blockers?: number;
  lastUpdate?: string;
}

export interface LaneMiniWidgetProps {
  /** Lane data */
  lane: MiniLane;
  /** Widget size */
  size?: 'xs' | 'sm' | 'md';
  /** Show owner */
  showOwner?: boolean;
  /** Show progress bar */
  showProgress?: boolean;
  /** Callback when clicked */
  onClick$?: QRL<(lane: MiniLane) => void>;
}

export interface LaneStatusBadgeProps {
  /** Status */
  status: 'green' | 'yellow' | 'red';
  /** WIP percentage */
  wip_pct: number;
  /** Label text */
  label?: string;
  /** Size */
  size?: 'xs' | 'sm' | 'md';
}

export interface LaneSummaryCardProps {
  /** Lanes data */
  lanes: MiniLane[];
  /** Title */
  title?: string;
  /** Max lanes to show */
  maxShow?: number;
  /** Callback when lane is clicked */
  onLaneClick$?: QRL<(lane: MiniLane) => void>;
}

export interface SlackEmbedProps {
  /** Lane data */
  lane: MiniLane;
  /** Include blockers info */
  showBlockers?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500 shadow-[0_0_6px_rgba(52,211,153,0.5)]';
    case 'yellow': return 'bg-amber-500 shadow-[0_0_6px_rgba(251,191,36,0.5)]';
    case 'red': return 'bg-red-500 shadow-[0_0_6px_rgba(248,113,113,0.5)]';
    default: return 'bg-gray-500';
  }
}

function getStatusBgColor(status: string): string {
  switch (status) {
    case 'green': return 'glass-status-healthy';
    case 'yellow': return 'glass-status-warning';
    case 'red': return 'glass-status-critical';
    default: return 'glass-chip';
  }
}

function getStatusText(status: string): string {
  switch (status) {
    case 'green': return 'On Track';
    case 'yellow': return 'At Risk';
    case 'red': return 'Blocked';
    default: return 'Unknown';
  }
}

function formatTimeAgo(dateStr?: string): string {
  if (!dateStr) return '';
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

// ============================================================================
// Mini Widget Component
// ============================================================================

export const LaneMiniWidget = component$<LaneMiniWidgetProps>(({
  lane,
  size = 'sm',
  showOwner = true,
  showProgress = true,
  onClick$,
}) => {
  const sizeClasses = {
    xs: 'p-1.5 text-[9px]',
    sm: 'p-2 text-[10px]',
    md: 'p-3 text-xs',
  };

  return (
    <div
      onClick$={() => onClick$ && onClick$(lane)}
      class={`rounded-lg ${getStatusBgColor(lane.status)} ${sizeClasses[size]} ${
        onClick$ ? 'cursor-pointer glass-hover-glow' : ''
      } glass-transition-colors`}
    >
      {/* Header */}
      <div class="flex items-center justify-between gap-2">
        <div class="flex items-center gap-1.5 min-w-0">
          <div class={`w-2 h-2 rounded-full ${getStatusColor(lane.status)} flex-shrink-0`} />
          <span class="font-medium text-foreground truncate">{lane.name}</span>
        </div>
        <span class={`font-bold flex-shrink-0 ${
          lane.wip_pct >= 90 ? 'text-emerald-400' :
          lane.wip_pct >= 50 ? 'text-cyan-400' :
          'text-amber-400'
        }`}>
          {lane.wip_pct}%
        </span>
      </div>

      {/* Progress bar */}
      {showProgress && (
        <div class="mt-1.5 h-1 rounded-full bg-glass-bg-muted/30 overflow-hidden">
          <div
            class={`h-full rounded-full glass-transition-all ${getStatusColor(lane.status)}`}
            style={{ width: `${lane.wip_pct}%` }}
          />
        </div>
      )}

      {/* Footer */}
      {(showOwner || lane.blockers) && (
        <div class="mt-1.5 flex items-center justify-between text-muted-foreground">
          {showOwner && <span>@{lane.owner}</span>}
          {lane.blockers && lane.blockers > 0 && (
            <span class="text-red-400">🚫 {lane.blockers}</span>
          )}
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Status Badge Component
// ============================================================================

export const LaneStatusBadge = component$<LaneStatusBadgeProps>(({
  status,
  wip_pct,
  label,
  size = 'sm',
}) => {
  const sizeClasses = {
    xs: 'px-1.5 py-0.5 text-[8px]',
    sm: 'px-2 py-1 text-[9px]',
    md: 'px-3 py-1.5 text-[10px]',
  };

  return (
    <span class={`inline-flex items-center gap-1.5 rounded-full border ${getStatusBgColor(status)} ${sizeClasses[size]}`}>
      <span class={`w-1.5 h-1.5 rounded-full ${getStatusColor(status)}`} />
      <span class="font-medium">{label || getStatusText(status)}</span>
      <span class="opacity-70">{wip_pct}%</span>
    </span>
  );
});

// ============================================================================
// Summary Card Component
// ============================================================================

export const LaneSummaryCard = component$<LaneSummaryCardProps>(({
  lanes,
  title = 'Lanes',
  maxShow = 5,
  onLaneClick$,
}) => {
  const stats = useComputed$(() => ({
    total: lanes.length,
    green: lanes.filter(l => l.status === 'green').length,
    yellow: lanes.filter(l => l.status === 'yellow').length,
    red: lanes.filter(l => l.status === 'red').length,
    avgWip: lanes.length > 0
      ? Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / lanes.length)
      : 0,
  }));

  const displayLanes = lanes.slice(0, maxShow);
  const hasMore = lanes.length > maxShow;

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="p-3 border-b border-border/50">
        <div class="flex items-center justify-between">
          <span class="text-xs font-semibold text-muted-foreground">{title}</span>
          <div class="flex items-center gap-2">
            <span class="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">
              {stats.value.green}
            </span>
            <span class="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-400">
              {stats.value.yellow}
            </span>
            <span class="text-[9px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400">
              {stats.value.red}
            </span>
          </div>
        </div>
        <div class="mt-1 text-[10px] text-muted-foreground">
          {stats.value.total} lanes • {stats.value.avgWip}% avg WIP
        </div>
      </div>

      {/* Lane list */}
      <div class="p-2 space-y-1.5">
        {displayLanes.map(lane => (
          <LaneMiniWidget
            key={lane.id}
            lane={lane}
            size="xs"
            showOwner={false}
            showProgress={false}
            onClick$={onLaneClick$}
          />
        ))}
        {hasMore && (
          <div class="text-center text-[9px] text-muted-foreground py-1">
            +{lanes.length - maxShow} more lanes
          </div>
        )}
      </div>
    </div>
  );
});

// ============================================================================
// Slack/Discord Embed Format
// ============================================================================

export const LaneSlackEmbed = component$<SlackEmbedProps>(({
  lane,
  showBlockers = true,
}) => {
  const statusEmoji = lane.status === 'green' ? '🟢' :
                      lane.status === 'yellow' ? '🟡' : '🔴';

  return (
    <div class="rounded border-l-4 bg-muted/10 p-3" style={{
      borderLeftColor: lane.status === 'green' ? '#22c55e' :
                       lane.status === 'yellow' ? '#eab308' : '#ef4444'
    }}>
      <div class="flex items-start gap-2">
        <span class="text-lg">{statusEmoji}</span>
        <div class="flex-grow">
          <div class="font-semibold text-sm text-foreground">{lane.name}</div>
          <div class="text-xs text-muted-foreground mt-0.5">
            <span class="font-mono">@{lane.owner}</span>
          </div>

          <div class="mt-2 grid grid-cols-2 gap-2 text-[10px]">
            <div>
              <span class="text-muted-foreground">Progress:</span>
              <span class="ml-1 font-bold">{lane.wip_pct}%</span>
            </div>
            <div>
              <span class="text-muted-foreground">Status:</span>
              <span class="ml-1">{getStatusText(lane.status)}</span>
            </div>
            {showBlockers && lane.blockers !== undefined && (
              <div>
                <span class="text-muted-foreground">Blockers:</span>
                <span class={`ml-1 ${lane.blockers > 0 ? 'text-red-400' : ''}`}>
                  {lane.blockers}
                </span>
              </div>
            )}
            {lane.lastUpdate && (
              <div>
                <span class="text-muted-foreground">Updated:</span>
                <span class="ml-1">{formatTimeAgo(lane.lastUpdate)}</span>
              </div>
            )}
          </div>

          {/* Progress bar */}
          <div class="mt-2 h-1.5 rounded-full bg-muted/30 overflow-hidden">
            <div
              class={`h-full rounded-full ${getStatusColor(lane.status)}`}
              style={{ width: `${lane.wip_pct}%` }}
            />
          </div>
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Inline Status Component
// ============================================================================

export interface InlineStatusProps {
  lane: MiniLane;
}

export const LaneInlineStatus = component$<InlineStatusProps>(({ lane }) => {
  return (
    <span class="inline-flex items-center gap-1 text-[10px]">
      <span class={`w-1.5 h-1.5 rounded-full ${getStatusColor(lane.status)}`} />
      <span class="text-foreground">{lane.name}</span>
      <span class="text-muted-foreground">({lane.wip_pct}%)</span>
    </span>
  );
});

export default LaneMiniWidget;
