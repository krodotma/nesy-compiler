/**
 * LaneHistoryTimeline - Vertical timeline showing WIP changes over time for a lane
 *
 * Features:
 * - Vertical timeline with visual connector lines
 * - WIP percentage with color coding
 * - Formatted timestamps
 * - Note text for each entry
 * - Graceful handling of empty history
 */

import { component$ } from '@builder.io/qwik';

export interface LaneHistory {
  ts: string;
  wip_pct: number;
  note: string;
}

export interface Lane {
  id: string;
  name: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  owner: string;
  description: string;
  commits: string[];
  blockers: string[];
  next_actions: string[];
  history: LaneHistory[];
  tier?: string;
  host?: string;
  slot?: number;
}

export interface LaneHistoryTimelineProps {
  lane: Lane;
  maxEntries?: number;
}

/**
 * Format a timestamp into a human-readable date/time
 * e.g., "2026-01-17T14:30:00Z" -> "Jan 17, 2026 14:30"
 */
function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    if (isNaN(date.getTime())) {
      // Fallback for invalid dates
      return ts.slice(0, 16).replace('T', ' ');
    }
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[date.getMonth()];
    const day = date.getDate();
    const year = date.getFullYear();
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${month} ${day}, ${year} ${hours}:${minutes}`;
  } catch {
    return ts.slice(0, 16).replace('T', ' ');
  }
}

/**
 * Get color class based on WIP percentage
 */
function wipColor(pct: number): string {
  if (pct >= 90) return 'text-emerald-400';
  if (pct >= 60) return 'text-cyan-400';
  if (pct >= 30) return 'text-amber-400';
  return 'text-red-400';
}

/**
 * Get background color class based on WIP percentage
 */
function wipBgColor(pct: number): string {
  if (pct >= 90) return 'bg-emerald-500/20 border-emerald-500/30';
  if (pct >= 60) return 'bg-cyan-500/20 border-cyan-500/30';
  if (pct >= 30) return 'bg-amber-500/20 border-amber-500/30';
  return 'bg-red-500/20 border-red-500/30';
}

/**
 * Get connector line color based on WIP percentage
 */
function connectorColor(pct: number): string {
  if (pct >= 90) return 'bg-emerald-500/40';
  if (pct >= 60) return 'bg-cyan-500/40';
  if (pct >= 30) return 'bg-amber-500/40';
  return 'bg-red-500/40';
}

export const LaneHistoryTimeline = component$<LaneHistoryTimelineProps>(({
  lane,
  maxEntries = 10,
}) => {
  const history = lane.history || [];
  const displayHistory = history.slice(0, maxEntries);
  const hasMore = history.length > maxEntries;

  // Empty state
  if (displayHistory.length === 0) {
    return (
      <div class="rounded-lg border border-border bg-card p-4">
        <div class="flex items-center gap-2 mb-3">
          <h4 class="text-xs font-semibold text-muted-foreground">HISTORY TIMELINE</h4>
          <span class="text-[10px] px-2 py-0.5 rounded bg-muted/30 text-muted-foreground border border-border/30">
            {lane.name}
          </span>
        </div>
        <div class="flex flex-col items-center justify-center py-6 text-muted-foreground">
          <div class="text-[10px] mb-1">No history recorded</div>
          <div class="text-[9px] text-muted-foreground/60">
            History entries will appear here as the lane progresses
          </div>
        </div>
      </div>
    );
  }

  return (
    <div class="rounded-lg border border-border bg-card p-4">
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <h4 class="text-xs font-semibold text-muted-foreground">HISTORY TIMELINE</h4>
          <span class="text-[10px] px-2 py-0.5 rounded bg-muted/30 text-muted-foreground border border-border/30">
            {lane.name}
          </span>
        </div>
        <span class="text-[9px] text-muted-foreground">
          {displayHistory.length} entries{hasMore && ` (+${history.length - maxEntries} more)`}
        </span>
      </div>

      {/* Timeline */}
      <div class="relative">
        {displayHistory.map((entry, index) => {
          const isLast = index === displayHistory.length - 1;
          const isFirst = index === 0;

          return (
            <div key={`${entry.ts}-${index}`} class="relative flex gap-3">
              {/* Timeline connector column */}
              <div class="flex flex-col items-center w-4">
                {/* Top connector line (not for first entry) */}
                {!isFirst && (
                  <div class={`w-0.5 h-2 ${connectorColor(displayHistory[index - 1].wip_pct)}`} />
                )}

                {/* Node dot */}
                <div
                  class={`w-3 h-3 rounded-full border-2 flex-shrink-0 ${wipBgColor(entry.wip_pct)}`}
                  title={`${entry.wip_pct}%`}
                />

                {/* Bottom connector line (not for last entry) */}
                {!isLast && (
                  <div class={`w-0.5 flex-grow min-h-[16px] ${connectorColor(entry.wip_pct)}`} />
                )}
              </div>

              {/* Content column */}
              <div class={`flex-grow pb-3 ${isLast ? 'pb-0' : ''}`}>
                {/* Timestamp and WIP */}
                <div class="flex items-center gap-2 mb-0.5">
                  <span class="text-[10px] text-muted-foreground/80 mono">
                    {formatTimestamp(entry.ts)}
                  </span>
                  <span class={`text-[10px] font-bold px-1.5 py-0.5 rounded ${wipBgColor(entry.wip_pct)} ${wipColor(entry.wip_pct)}`}>
                    {entry.wip_pct}%
                  </span>
                  {/* Delta indicator */}
                  {index > 0 && (
                    <span class={`text-[9px] ${
                      entry.wip_pct > displayHistory[index - 1].wip_pct
                        ? 'text-emerald-400'
                        : entry.wip_pct < displayHistory[index - 1].wip_pct
                        ? 'text-red-400'
                        : 'text-muted-foreground/60'
                    }`}>
                      {entry.wip_pct > displayHistory[index - 1].wip_pct && '+'}
                      {entry.wip_pct - displayHistory[index - 1].wip_pct}%
                    </span>
                  )}
                  {isFirst && (
                    <span class="text-[9px] px-1 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                      latest
                    </span>
                  )}
                </div>

                {/* Note */}
                {entry.note && (
                  <div class="text-[10px] text-foreground/80 leading-relaxed">
                    {entry.note}
                  </div>
                )}
                {!entry.note && (
                  <div class="text-[9px] text-muted-foreground/50 italic">
                    No note provided
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* More indicator */}
      {hasMore && (
        <div class="mt-2 pt-2 border-t border-border/30 text-center">
          <span class="text-[9px] text-muted-foreground/60">
            {history.length - maxEntries} earlier entries not shown
          </span>
        </div>
      )}

      {/* Current WIP summary */}
      <div class="mt-3 pt-3 border-t border-border/50">
        <div class="flex items-center justify-between">
          <span class="text-[9px] text-muted-foreground">Current WIP</span>
          <span class={`text-xs font-bold ${wipColor(lane.wip_pct)}`}>
            {lane.wip_pct}%
          </span>
        </div>
        {displayHistory.length >= 2 && (
          <div class="flex items-center justify-between mt-1">
            <span class="text-[9px] text-muted-foreground">Total Progress</span>
            <span class={`text-[10px] ${
              lane.wip_pct >= displayHistory[displayHistory.length - 1].wip_pct
                ? 'text-emerald-400'
                : 'text-red-400'
            }`}>
              {lane.wip_pct >= displayHistory[displayHistory.length - 1].wip_pct ? '+' : ''}
              {lane.wip_pct - displayHistory[displayHistory.length - 1].wip_pct}% since first entry
            </span>
          </div>
        )}
      </div>
    </div>
  );
});

export default LaneHistoryTimeline;
