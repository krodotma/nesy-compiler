/**
 * LaneNotifications - Real-time notifications for lane events
 *
 * Phase 3, Iteration 22 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Show notifications for lane events (status changes, blockers, etc.)
 * - Categorize by type (info, warning, error, success)
 * - Mark as read/unread
 * - Clear notifications
 * - Subscribe to bus events
 * - Notification badges
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export type NotificationType = 'info' | 'warning' | 'error' | 'success';
export type NotificationCategory = 'status' | 'blocker' | 'action' | 'assignment' | 'priority' | 'note';

export interface Notification {
  id: string;
  type: NotificationType;
  category: NotificationCategory;
  title: string;
  message: string;
  laneId?: string;
  laneName?: string;
  timestamp: string;
  read: boolean;
  actor?: string;
}

export interface LaneNotificationsProps {
  /** Initial notifications */
  notifications?: Notification[];
  /** Callback when notification is clicked */
  onNotificationClick$?: QRL<(notification: Notification) => void>;
  /** Callback when notifications change */
  onNotificationsChange$?: QRL<(notifications: Notification[]) => void>;
  /** Show only unread */
  showOnlyUnread?: boolean;
  /** Max visible before scrolling */
  maxVisible?: number;
  /** Auto-dismiss after ms (0 = never) */
  autoDismiss?: number;
  /** Compact mode (dropdown style) */
  compact?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getTypeColor(type: NotificationType): string {
  switch (type) {
    case 'success': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'warning': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'error': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'info':
    default: return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  }
}

function getTypeIcon(type: NotificationType): string {
  switch (type) {
    case 'success': return '✓';
    case 'warning': return '⚠';
    case 'error': return '✕';
    case 'info':
    default: return 'ℹ';
  }
}

function getCategoryIcon(category: NotificationCategory): string {
  switch (category) {
    case 'status': return '🔄';
    case 'blocker': return '🚫';
    case 'action': return '📋';
    case 'assignment': return '👤';
    case 'priority': return '⬆';
    case 'note': return '📝';
    default: return '•';
  }
}

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffMins = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMins / 60);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  } catch {
    return ts.slice(0, 16);
  }
}

function generateId(): string {
  return `notif-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

// ============================================================================
// Component
// ============================================================================

export const LaneNotifications = component$<LaneNotificationsProps>(({
  notifications: initialNotifications = [],
  onNotificationClick$,
  onNotificationsChange$,
  showOnlyUnread = false,
  maxVisible = 10,
  autoDismiss = 0,
  compact = false,
}) => {
  // State
  const notifications = useSignal<Notification[]>(initialNotifications);
  const isExpanded = useSignal(!compact);
  const filterCategory = useSignal<NotificationCategory | 'all'>('all');

  // Auto-dismiss effect
  useVisibleTask$(({ cleanup }) => {
    if (autoDismiss <= 0) return;

    const interval = setInterval(() => {
      const now = Date.now();
      notifications.value = notifications.value.filter(n => {
        const notifTime = new Date(n.timestamp).getTime();
        return now - notifTime < autoDismiss;
      });
    }, 1000);

    cleanup(() => clearInterval(interval));
  });

  // Computed
  const unreadCount = useComputed$(() =>
    notifications.value.filter(n => !n.read).length
  );

  const filteredNotifications = useComputed$(() => {
    let result = [...notifications.value];

    if (showOnlyUnread) {
      result = result.filter(n => !n.read);
    }

    if (filterCategory.value !== 'all') {
      result = result.filter(n => n.category === filterCategory.value);
    }

    // Sort by timestamp (newest first)
    result.sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());

    return result;
  });

  const categoryCounts = useComputed$(() => {
    const counts: Record<NotificationCategory | 'all', number> = {
      all: notifications.value.length,
      status: 0,
      blocker: 0,
      action: 0,
      assignment: 0,
      priority: 0,
      note: 0,
    };

    notifications.value.forEach(n => {
      counts[n.category]++;
    });

    return counts;
  });

  // Mark as read
  const markAsRead = $(async (notifId: string) => {
    const idx = notifications.value.findIndex(n => n.id === notifId);
    if (idx === -1) return;

    const updated = [...notifications.value];
    updated[idx] = { ...updated[idx], read: true };
    notifications.value = updated;

    if (onNotificationsChange$) {
      await onNotificationsChange$(updated);
    }
  });

  // Mark all as read
  const markAllAsRead = $(async () => {
    const updated = notifications.value.map(n => ({ ...n, read: true }));
    notifications.value = updated;

    if (onNotificationsChange$) {
      await onNotificationsChange$(updated);
    }
  });

  // Clear notification
  const clearNotification = $(async (notifId: string) => {
    const updated = notifications.value.filter(n => n.id !== notifId);
    notifications.value = updated;

    if (onNotificationsChange$) {
      await onNotificationsChange$(updated);
    }
  });

  // Clear all
  const clearAll = $(async () => {
    notifications.value = [];

    if (onNotificationsChange$) {
      await onNotificationsChange$([]);
    }
  });

  // Handle click
  const handleClick = $(async (notif: Notification) => {
    await markAsRead(notif.id);

    if (onNotificationClick$) {
      await onNotificationClick$(notif);
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <button
            onClick$={() => { isExpanded.value = !isExpanded.value; }}
            class="flex items-center gap-2"
          >
            <span class="text-xs font-semibold text-muted-foreground">NOTIFICATIONS</span>
            {unreadCount.value > 0 && (
              <span class="min-w-[18px] h-[18px] px-1 flex items-center justify-center text-[10px] font-bold rounded-full bg-red-500 text-white">
                {unreadCount.value > 99 ? '99+' : unreadCount.value}
              </span>
            )}
          </button>
        </div>
        <div class="flex items-center gap-1">
          {unreadCount.value > 0 && (
            <button
              onClick$={markAllAsRead}
              class="text-[9px] px-2 py-1 rounded bg-muted/20 text-muted-foreground hover:bg-muted/40 transition-colors"
            >
              Mark all read
            </button>
          )}
          {notifications.value.length > 0 && (
            <button
              onClick$={clearAll}
              class="text-[9px] px-2 py-1 rounded bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-colors"
            >
              Clear all
            </button>
          )}
        </div>
      </div>

      {/* Category filters */}
      {isExpanded.value && !compact && (
        <div class="flex gap-1 p-2 border-b border-border/30 overflow-x-auto">
          {(['all', 'status', 'blocker', 'action', 'assignment', 'priority', 'note'] as const).map(cat => (
            <button
              key={cat}
              onClick$={() => { filterCategory.value = cat; }}
              class={`flex items-center gap-1 px-2 py-1 text-[9px] rounded whitespace-nowrap transition-colors ${
                filterCategory.value === cat
                  ? 'bg-primary/20 text-primary border border-primary/30'
                  : 'bg-muted/10 text-muted-foreground hover:bg-muted/20 border border-transparent'
              }`}
            >
              {cat !== 'all' && <span>{getCategoryIcon(cat)}</span>}
              <span>{cat === 'all' ? 'All' : cat}</span>
              {categoryCounts.value[cat] > 0 && (
                <span class="opacity-70">({categoryCounts.value[cat]})</span>
              )}
            </button>
          ))}
        </div>
      )}

      {/* Notification list */}
      {isExpanded.value && (
        <div class={`overflow-y-auto ${compact ? 'max-h-[200px]' : 'max-h-[350px]'}`}>
          {filteredNotifications.value.length === 0 ? (
            <div class="p-6 text-center">
              <div class="text-2xl mb-2">🔔</div>
              <div class="text-xs text-muted-foreground">
                {showOnlyUnread ? 'No unread notifications' : 'No notifications'}
              </div>
            </div>
          ) : (
            filteredNotifications.value.slice(0, maxVisible).map(notif => (
              <div
                key={notif.id}
                onClick$={() => handleClick(notif)}
                class={`p-3 border-b border-border/30 cursor-pointer transition-colors ${
                  notif.read ? 'opacity-60' : 'bg-muted/5'
                } hover:bg-muted/10`}
              >
                <div class="flex items-start gap-2">
                  {/* Type icon */}
                  <div class={`w-6 h-6 rounded flex items-center justify-center flex-shrink-0 ${getTypeColor(notif.type)}`}>
                    <span class="text-[10px]">{getTypeIcon(notif.type)}</span>
                  </div>

                  {/* Content */}
                  <div class="flex-grow min-w-0">
                    <div class="flex items-center gap-2">
                      <span class={`text-xs font-medium ${notif.read ? 'text-muted-foreground' : 'text-foreground'}`}>
                        {notif.title}
                      </span>
                      {!notif.read && (
                        <span class="w-2 h-2 rounded-full bg-primary flex-shrink-0" />
                      )}
                    </div>
                    <div class="text-[10px] text-muted-foreground mt-0.5 truncate">
                      {notif.message}
                    </div>
                    <div class="flex items-center gap-2 mt-1">
                      <span class="text-[8px] text-muted-foreground/70">
                        {getCategoryIcon(notif.category)} {notif.category}
                      </span>
                      {notif.laneName && (
                        <span class="text-[8px] text-muted-foreground/70">
                          • {notif.laneName}
                        </span>
                      )}
                      {notif.actor && (
                        <span class="text-[8px] text-muted-foreground/70">
                          • @{notif.actor}
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Timestamp and clear */}
                  <div class="flex-shrink-0 flex flex-col items-end gap-1">
                    <span class="text-[9px] text-muted-foreground">
                      {formatTimestamp(notif.timestamp)}
                    </span>
                    <button
                      onClick$={(e) => {
                        e.stopPropagation();
                        clearNotification(notif.id);
                      }}
                      class="w-4 h-4 flex items-center justify-center rounded text-[10px] text-muted-foreground/50 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                    >
                      ×
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}

          {/* Show more indicator */}
          {filteredNotifications.value.length > maxVisible && (
            <div class="p-2 text-center text-[9px] text-muted-foreground">
              +{filteredNotifications.value.length - maxVisible} more notifications
            </div>
          )}
        </div>
      )}

      {/* Collapsed state */}
      {!isExpanded.value && (
        <button
          onClick$={() => { isExpanded.value = true; }}
          class="w-full p-2 text-[9px] text-muted-foreground hover:bg-muted/10 transition-colors"
        >
          {notifications.value.length} notifications ({unreadCount.value} unread)
        </button>
      )}
    </div>
  );
});

// ============================================================================
// Notification Badge Component (for use elsewhere)
// ============================================================================

export interface NotificationBadgeProps {
  count: number;
  onClick$?: QRL<() => void>;
}

export const NotificationBadge = component$<NotificationBadgeProps>(({ count, onClick$ }) => {
  if (count === 0) return null;

  return (
    <button
      onClick$={onClick$}
      class="relative inline-flex items-center justify-center w-8 h-8 rounded-full bg-muted/20 hover:bg-muted/40 transition-colors"
    >
      <span class="text-sm">🔔</span>
      <span class="absolute -top-1 -right-1 min-w-[16px] h-[16px] px-1 flex items-center justify-center text-[9px] font-bold rounded-full bg-red-500 text-white">
        {count > 99 ? '99+' : count}
      </span>
    </button>
  );
});

export default LaneNotifications;
