/**
 * LaneActivityFeed - Unified activity timeline for lanes
 *
 * Phase 3, Iteration 23 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Chronological event timeline
 * - Filter by event type
 * - Group by day
 * - Expandable event details
 * - Actor avatars/initials
 * - Real-time updates from bus
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

export type ActivityType =
  | 'status_change'
  | 'wip_update'
  | 'blocker_added'
  | 'blocker_resolved'
  | 'action_added'
  | 'action_completed'
  | 'assignment_change'
  | 'priority_change'
  | 'note_added'
  | 'approval_requested'
  | 'approval_granted'
  | 'approval_rejected';

export interface ActivityEvent {
  id: string;
  type: ActivityType;
  laneId: string;
  laneName: string;
  actor: string;
  timestamp: string;
  data: Record<string, unknown>;
  summary: string;
}

export interface LaneActivityFeedProps {
  /** Activity events */
  activities: ActivityEvent[];
  /** Filter to specific lane */
  laneId?: string;
  /** Callback when activity is clicked */
  onActivityClick$?: QRL<(activity: ActivityEvent) => void>;
  /** Max visible events */
  maxVisible?: number;
  /** Group by day */
  groupByDay?: boolean;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getActivityColor(type: ActivityType): string {
  switch (type) {
    case 'status_change':
      return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'wip_update':
      return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
    case 'blocker_added':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'blocker_resolved':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'action_added':
      return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
    case 'action_completed':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'assignment_change':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'priority_change':
      return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    case 'note_added':
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    case 'approval_requested':
      return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    case 'approval_granted':
      return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'approval_rejected':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    default:
      return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getActivityIcon(type: ActivityType): string {
  switch (type) {
    case 'status_change': return '🔄';
    case 'wip_update': return '📊';
    case 'blocker_added': return '🚫';
    case 'blocker_resolved': return '✓';
    case 'action_added': return '📋';
    case 'action_completed': return '✅';
    case 'assignment_change': return '👤';
    case 'priority_change': return '⬆';
    case 'note_added': return '📝';
    case 'approval_requested': return '⏳';
    case 'approval_granted': return '✓';
    case 'approval_rejected': return '✕';
    default: return '•';
  }
}

function getActivityLabel(type: ActivityType): string {
  switch (type) {
    case 'status_change': return 'Status Change';
    case 'wip_update': return 'WIP Update';
    case 'blocker_added': return 'Blocker Added';
    case 'blocker_resolved': return 'Blocker Resolved';
    case 'action_added': return 'Action Added';
    case 'action_completed': return 'Action Completed';
    case 'assignment_change': return 'Assignment Change';
    case 'priority_change': return 'Priority Change';
    case 'note_added': return 'Note Added';
    case 'approval_requested': return 'Approval Requested';
    case 'approval_granted': return 'Approval Granted';
    case 'approval_rejected': return 'Approval Rejected';
    default: return type;
  }
}

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return ts.slice(11, 16);
  }
}

function formatDate(ts: string): string {
  try {
    const date = new Date(ts);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === yesterday.toDateString()) return 'Yesterday';
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  } catch {
    return ts.slice(0, 10);
  }
}

function getInitials(name: string): string {
  return name.slice(0, 2).toUpperCase();
}

function hashColor(str: string): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const colors = [
    'bg-blue-500', 'bg-emerald-500', 'bg-purple-500', 'bg-amber-500',
    'bg-cyan-500', 'bg-pink-500', 'bg-indigo-500', 'bg-rose-500',
  ];
  return colors[Math.abs(hash) % colors.length];
}

// ============================================================================
// Component
// ============================================================================

export const LaneActivityFeed = component$<LaneActivityFeedProps>(({
  activities,
  laneId,
  onActivityClick$,
  maxVisible = 50,
  groupByDay = true,
  compact = false,
}) => {
  // State
  const expandedIds = useSignal<Set<string>>(new Set());
  const filterType = useSignal<ActivityType | 'all'>('all');

  // Computed
  const filteredActivities = useComputed$(() => {
    let result = [...activities];

    // Filter by lane if specified
    if (laneId) {
      result = result.filter(a => a.laneId === laneId);
    }

    // Filter by type
    if (filterType.value !== 'all') {
      result = result.filter(a => a.type === filterType.value);
    }

    // Sort by timestamp (newest first)
    result.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    return result.slice(0, maxVisible);
  });

  // Group by day
  const groupedActivities = useComputed$(() => {
    if (!groupByDay) {
      return [{ date: '', activities: filteredActivities.value }];
    }

    const groups: { date: string; activities: ActivityEvent[] }[] = [];
    let currentDate = '';

    filteredActivities.value.forEach(activity => {
      const date = activity.timestamp.slice(0, 10);
      if (date !== currentDate) {
        currentDate = date;
        groups.push({ date: activity.timestamp, activities: [] });
      }
      groups[groups.length - 1].activities.push(activity);
    });

    return groups;
  });

  // Stats
  const stats = useComputed$(() => {
    const typeCounts: Partial<Record<ActivityType, number>> = {};
    activities.forEach(a => {
      typeCounts[a.type] = (typeCounts[a.type] || 0) + 1;
    });
    return {
      total: activities.length,
      filtered: filteredActivities.value.length,
      types: typeCounts,
    };
  });

  // Toggle expand
  const toggleExpand = $((id: string) => {
    const newSet = new Set(expandedIds.value);
    if (newSet.has(id)) {
      newSet.delete(id);
    } else {
      newSet.add(id);
    }
    expandedIds.value = newSet;
  });

  // Handle click
  const handleClick = $(async (activity: ActivityEvent) => {
    if (onActivityClick$) {
      await onActivityClick$(activity);
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">ACTIVITY FEED</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {stats.value.filtered} events
          </span>
        </div>
        {laneId && (
          <span class="text-[9px] text-muted-foreground">Lane: {laneId.slice(0, 12)}...</span>
        )}
      </div>

      {/* Type filter */}
      {!compact && (
        <div class="flex gap-1 p-2 border-b border-border/30 overflow-x-auto">
          <button
            onClick$={() => { filterType.value = 'all'; }}
            class={`px-2 py-1 text-[9px] rounded whitespace-nowrap transition-colors ${
              filterType.value === 'all'
                ? 'bg-primary/20 text-primary border border-primary/30'
                : 'bg-muted/10 text-muted-foreground hover:bg-muted/20 border border-transparent'
            }`}
          >
            All ({stats.value.total})
          </button>
          {Object.entries(stats.value.types).map(([type, count]) => (
            <button
              key={type}
              onClick$={() => { filterType.value = type as ActivityType; }}
              class={`flex items-center gap-1 px-2 py-1 text-[9px] rounded whitespace-nowrap transition-colors ${
                filterType.value === type
                  ? getActivityColor(type as ActivityType)
                  : 'bg-muted/10 text-muted-foreground hover:bg-muted/20 border border-transparent'
              }`}
            >
              <span>{getActivityIcon(type as ActivityType)}</span>
              <span>{count}</span>
            </button>
          ))}
        </div>
      )}

      {/* Activity list */}
      <div class={`overflow-y-auto ${compact ? 'max-h-[200px]' : 'max-h-[400px]'}`}>
        {filteredActivities.value.length === 0 ? (
          <div class="p-6 text-center">
            <div class="text-2xl mb-2">📋</div>
            <div class="text-xs text-muted-foreground">No activity yet</div>
          </div>
        ) : (
          groupedActivities.value.map((group, groupIdx) => (
            <div key={groupIdx}>
              {/* Date header */}
              {groupByDay && group.date && (
                <div class="sticky top-0 px-3 py-1.5 bg-muted/20 border-b border-border/30 text-[10px] font-medium text-muted-foreground z-10">
                  {formatDate(group.date)}
                </div>
              )}

              {/* Events */}
              {group.activities.map((activity, idx) => (
                <div
                  key={activity.id}
                  class="relative pl-8 pr-3 py-3 border-b border-border/20 hover:bg-muted/5 transition-colors"
                >
                  {/* Timeline connector */}
                  {idx < group.activities.length - 1 && (
                    <div class="absolute left-[15px] top-8 bottom-0 w-px bg-border/30" />
                  )}

                  {/* Avatar */}
                  <div
                    class={`absolute left-2 top-3 w-6 h-6 rounded-full flex items-center justify-center text-[9px] font-bold text-white ${hashColor(activity.actor)}`}
                  >
                    {getInitials(activity.actor)}
                  </div>

                  {/* Content */}
                  <div
                    onClick$={() => handleClick(activity)}
                    class="cursor-pointer"
                  >
                    <div class="flex items-center gap-2">
                      <span class={`text-[9px] px-1.5 py-0.5 rounded border ${getActivityColor(activity.type)}`}>
                        {getActivityIcon(activity.type)} {getActivityLabel(activity.type)}
                      </span>
                      <span class="text-[9px] text-muted-foreground">
                        {formatTimestamp(activity.timestamp)}
                      </span>
                    </div>

                    <div class="text-xs text-foreground mt-1">
                      {activity.summary}
                    </div>

                    <div class="flex items-center gap-2 mt-1 text-[9px] text-muted-foreground">
                      <span>@{activity.actor}</span>
                      {!laneId && (
                        <>
                          <span>•</span>
                          <span>{activity.laneName}</span>
                        </>
                      )}
                    </div>

                    {/* Expandable details */}
                    {Object.keys(activity.data).length > 0 && (
                      <>
                        <button
                          onClick$={(e) => {
                            e.stopPropagation();
                            toggleExpand(activity.id);
                          }}
                          class="text-[9px] text-primary hover:underline mt-1"
                        >
                          {expandedIds.value.has(activity.id) ? '▼ Hide details' : '▶ Show details'}
                        </button>

                        {expandedIds.value.has(activity.id) && (
                          <div class="mt-2 p-2 rounded bg-muted/10 border border-border/30">
                            <pre class="text-[9px] text-muted-foreground overflow-x-auto">
                              {JSON.stringify(activity.data, null, 2)}
                            </pre>
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ))
        )}
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        Showing {filteredActivities.value.length} of {stats.value.total} events
      </div>
    </div>
  );
});

export default LaneActivityFeed;
