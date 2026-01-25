/**
 * CrossLaneHistoryView - Combined timeline showing history across all lanes
 *
 * Phase 2, Iteration 11 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Unified timeline of all lane history entries
 * - Filter by lane, agent, date range
 * - Sort by recency or lane
 * - Lane badges with color coding
 * - Infinite scroll support
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type Signal,
} from '@builder.io/qwik';

// M3 Components - CrossLaneHistoryView
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/button/text-button.js';

// ============================================================================
// Types
// ============================================================================

interface LaneHistory {
  ts: string;
  wip_pct: number;
  note: string;
}

interface Lane {
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
}

interface AggregatedEntry {
  ts: string;
  laneId: string;
  laneName: string;
  laneStatus: 'green' | 'yellow' | 'red';
  owner: string;
  wip_pct: number;
  note: string;
}

export interface CrossLaneHistoryViewProps {
  /** All lanes data */
  lanes: Lane[];
  /** Maximum entries to display initially */
  initialLimit?: number;
  /** Page size for load more */
  pageSize?: number;
  /** External filter signal */
  externalFilter?: Signal<{
    laneId?: string;
    owner?: string;
    dateStart?: string;
    dateEnd?: string;
  }>;
}

// ============================================================================
// Utilities
// ============================================================================

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    if (isNaN(date.getTime())) return ts.slice(0, 16).replace('T', ' ');
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const month = months[date.getMonth()];
    const day = date.getDate();
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    return `${month} ${day} ${hours}:${minutes}`;
  } catch {
    return ts.slice(0, 16).replace('T', ' ');
  }
}

function wipColor(pct: number): string {
  if (pct >= 90) return 'text-emerald-400';
  if (pct >= 60) return 'text-cyan-400';
  if (pct >= 30) return 'text-amber-400';
  return 'text-red-400';
}

function wipBg(pct: number): string {
  if (pct >= 90) return 'bg-emerald-500/20 border-emerald-500/30';
  if (pct >= 60) return 'bg-cyan-500/20 border-cyan-500/30';
  if (pct >= 30) return 'bg-amber-500/20 border-amber-500/30';
  return 'bg-red-500/20 border-red-500/30';
}

function statusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'yellow': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'red': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

// ============================================================================
// Component
// ============================================================================

export const CrossLaneHistoryView = component$<CrossLaneHistoryViewProps>(({
  lanes,
  initialLimit = 50,
  pageSize = 25,
  externalFilter,
}) => {
  // State
  const displayLimit = useSignal(initialLimit);
  const sortBy = useSignal<'recency' | 'lane'>('recency');
  const filterLane = useSignal<string>('all');
  const filterOwner = useSignal<string>('all');
  const filterDateStart = useSignal<string>('');
  const filterDateEnd = useSignal<string>('');
  const showFilters = useSignal(false);

  // Get unique values for filters
  const uniqueLanes = useComputed$(() => {
    return lanes.map(l => ({ id: l.id, name: l.name }));
  });

  const uniqueOwners = useComputed$(() => {
    return [...new Set(lanes.map(l => l.owner))].sort();
  });

  // Aggregate all history entries
  const allEntries = useComputed$(() => {
    const entries: AggregatedEntry[] = [];
    for (const lane of lanes) {
      for (const h of lane.history || []) {
        entries.push({
          ts: h.ts,
          laneId: lane.id,
          laneName: lane.name,
          laneStatus: lane.status,
          owner: lane.owner,
          wip_pct: h.wip_pct,
          note: h.note,
        });
      }
    }
    return entries;
  });

  // Apply filters
  const filteredEntries = useComputed$(() => {
    let entries = [...allEntries.value];

    // Apply external filter if provided
    const ext = externalFilter?.value;
    if (ext) {
      if (ext.laneId) entries = entries.filter(e => e.laneId === ext.laneId);
      if (ext.owner) entries = entries.filter(e => e.owner === ext.owner);
      if (ext.dateStart) entries = entries.filter(e => e.ts >= ext.dateStart!);
      if (ext.dateEnd) entries = entries.filter(e => e.ts <= ext.dateEnd!);
    }

    // Apply local filters
    if (filterLane.value !== 'all') {
      entries = entries.filter(e => e.laneId === filterLane.value);
    }
    if (filterOwner.value !== 'all') {
      entries = entries.filter(e => e.owner === filterOwner.value);
    }
    if (filterDateStart.value) {
      entries = entries.filter(e => e.ts >= filterDateStart.value);
    }
    if (filterDateEnd.value) {
      entries = entries.filter(e => e.ts <= filterDateEnd.value);
    }

    // Sort
    if (sortBy.value === 'recency') {
      entries.sort((a, b) => b.ts.localeCompare(a.ts));
    } else {
      entries.sort((a, b) => {
        const laneCompare = a.laneName.localeCompare(b.laneName);
        if (laneCompare !== 0) return laneCompare;
        return b.ts.localeCompare(a.ts);
      });
    }

    return entries;
  });

  // Display entries (with limit)
  const displayEntries = useComputed$(() => {
    return filteredEntries.value.slice(0, displayLimit.value);
  });

  const hasMore = useComputed$(() => {
    return filteredEntries.value.length > displayLimit.value;
  });

  const loadMore = $(() => {
    displayLimit.value += pageSize;
  });

  const resetFilters = $(() => {
    filterLane.value = 'all';
    filterOwner.value = 'all';
    filterDateStart.value = '';
    filterDateEnd.value = '';
  });

  // Stats
  const stats = useComputed$(() => {
    const total = filteredEntries.value.length;
    const lanesCount = new Set(filteredEntries.value.map(e => e.laneId)).size;
    return { total, lanesCount };
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <h4 class="text-xs font-semibold text-muted-foreground">ALL HISTORY</h4>
          <span class="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
            {stats.value.total} entries
          </span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            {stats.value.lanesCount} lanes
          </span>
        </div>
        <div class="flex items-center gap-2">
          {/* Sort toggle */}
          <button
            onClick$={() => { sortBy.value = sortBy.value === 'recency' ? 'lane' : 'recency'; }}
            class="text-[10px] px-2 py-1 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30"
          >
            Sort: {sortBy.value === 'recency' ? 'Recent' : 'Lane'}
          </button>
          {/* Filter toggle */}
          <button
            onClick$={() => { showFilters.value = !showFilters.value; }}
            class={`text-[10px] px-2 py-1 rounded border transition-colors ${
              showFilters.value
                ? 'bg-primary/20 text-primary border-primary/30'
                : 'bg-muted/30 text-muted-foreground border-border/30 hover:bg-muted/50'
            }`}
          >
            Filters {showFilters.value ? '▲' : '▼'}
          </button>
        </div>
      </div>

      {/* Filter panel */}
      {showFilters.value && (
        <div class="p-3 border-b border-border/50 bg-muted/10">
          <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
            {/* Lane filter */}
            <div>
              <label class="text-[9px] text-muted-foreground block mb-1">Lane</label>
              <select
                value={filterLane.value}
                onChange$={(e) => { filterLane.value = (e.target as HTMLSelectElement).value; }}
                class="w-full text-[10px] px-2 py-1 rounded bg-card border border-border/50 text-foreground"
              >
                <option value="all">All Lanes</option>
                {uniqueLanes.value.map(l => (
                  <option key={l.id} value={l.id}>{l.name}</option>
                ))}
              </select>
            </div>

            {/* Owner filter */}
            <div>
              <label class="text-[9px] text-muted-foreground block mb-1">Owner</label>
              <select
                value={filterOwner.value}
                onChange$={(e) => { filterOwner.value = (e.target as HTMLSelectElement).value; }}
                class="w-full text-[10px] px-2 py-1 rounded bg-card border border-border/50 text-foreground"
              >
                <option value="all">All Owners</option>
                {uniqueOwners.value.map(o => (
                  <option key={o} value={o}>@{o}</option>
                ))}
              </select>
            </div>

            {/* Date range */}
            <div>
              <label class="text-[9px] text-muted-foreground block mb-1">From</label>
              <input
                type="date"
                value={filterDateStart.value}
                onChange$={(e) => { filterDateStart.value = (e.target as HTMLInputElement).value; }}
                class="w-full text-[10px] px-2 py-1 rounded bg-card border border-border/50 text-foreground"
              />
            </div>
            <div>
              <label class="text-[9px] text-muted-foreground block mb-1">To</label>
              <input
                type="date"
                value={filterDateEnd.value}
                onChange$={(e) => { filterDateEnd.value = (e.target as HTMLInputElement).value; }}
                class="w-full text-[10px] px-2 py-1 rounded bg-card border border-border/50 text-foreground"
              />
            </div>
          </div>

          {/* Reset button */}
          <button
            onClick$={resetFilters}
            class="mt-2 text-[10px] px-2 py-1 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground"
          >
            Reset Filters
          </button>
        </div>
      )}

      {/* Timeline content */}
      <div class="max-h-[500px] overflow-y-auto">
        {displayEntries.value.length === 0 ? (
          <div class="p-8 text-center text-xs text-muted-foreground">
            No history entries found
            {(filterLane.value !== 'all' || filterOwner.value !== 'all') && (
              <div class="mt-2">
                <button
                  onClick$={resetFilters}
                  class="text-[10px] underline hover:no-underline text-primary"
                >
                  Clear filters
                </button>
              </div>
            )}
          </div>
        ) : (
          <div class="divide-y divide-border/30">
            {displayEntries.value.map((entry, index) => (
              <div
                key={`${entry.laneId}-${entry.ts}-${index}`}
                class="p-3 hover:bg-muted/10 transition-colors"
              >
                <div class="flex items-start justify-between gap-2">
                  {/* Left: Lane badge + timestamp */}
                  <div class="flex items-center gap-2 min-w-0">
                    <span class={`text-[9px] px-1.5 py-0.5 rounded border flex-shrink-0 ${statusColor(entry.laneStatus)}`}>
                      {entry.laneName}
                    </span>
                    <span class="text-[10px] text-muted-foreground/70 mono">
                      {formatTimestamp(entry.ts)}
                    </span>
                  </div>

                  {/* Right: WIP + owner */}
                  <div class="flex items-center gap-2 flex-shrink-0">
                    <span class={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${wipBg(entry.wip_pct)} ${wipColor(entry.wip_pct)}`}>
                      {entry.wip_pct}%
                    </span>
                    <span class="text-[9px] text-muted-foreground">
                      @{entry.owner}
                    </span>
                  </div>
                </div>

                {/* Note */}
                {entry.note && (
                  <div class="mt-1 text-[10px] text-foreground/80 leading-relaxed pl-0">
                    {entry.note}
                  </div>
                )}
                {!entry.note && (
                  <div class="mt-1 text-[9px] text-muted-foreground/50 italic">
                    No note
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Load more */}
        {hasMore.value && (
          <div class="p-3 border-t border-border/30">
            <button
              onClick$={loadMore}
              class="w-full text-[10px] py-2 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground border border-border/30 transition-colors"
            >
              Load More ({filteredEntries.value.length - displayLimit.value} remaining)
            </button>
          </div>
        )}
      </div>
    </div>
  );
});

export default CrossLaneHistoryView;
