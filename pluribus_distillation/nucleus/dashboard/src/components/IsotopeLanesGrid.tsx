/**
 * IsotopeLanesGrid - Isotope-powered filterable/sortable lanes grid
 *
 * Phase 2 of OITERATE lanes-widget-enhancement
 * Uses Metafizzy Isotope for filtering/sorting with masonry layout
 *
 * Qwik Integration Pattern:
 * - Dynamic import in useVisibleTask$ (browser-only)
 * - Container ref with useSignal
 * - MutationObserver for reactivity
 * - Proper cleanup on unmount
 *
 * @see MetafizzyDeck.tsx for reference implementation
 * @see https://isotope.metafizzy.co/
 *
 * Iteration 6: isotope-layout package installed and integration enabled.
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
  type Signal,
} from '@builder.io/qwik';

// M3 Components - IsotopeLanesGrid
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';

import type { BusEvent } from '../lib/state/types';

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
  tier?: string;
  host?: string;
  slot?: number;
}

interface Agent {
  id: string;
  status: 'active' | 'idle' | 'offline';
  lane: string | null;
  last_seen: string;
}

interface LanesState {
  version: string;
  generated: string;
  updated: string;
  lanes: Lane[];
  agents: Agent[];
}

export interface IsotopeLanesGridProps {
  /** Refresh interval in ms (default: 30000) */
  refreshIntervalMs?: number;
  /** Bus events signal for reactive updates */
  events?: Signal<BusEvent[]>;
  /** Initial filter: 'all' | 'green' | 'yellow' | 'red' */
  initialFilter?: string;
  /** Initial sort: 'wip' | 'name' | 'status' | 'owner' */
  initialSort?: string;
  /** Enable drag-to-reorder (requires Draggabilly) */
  enableDrag?: boolean;
}

// ============================================================================
// Isotope Configuration
// ============================================================================

/**
 * Isotope filter functions
 * Maps filter names to CSS selectors
 */
const ISOTOPE_FILTERS: Record<string, string> = {
  all: '*',
  green: '.status-green',
  yellow: '.status-yellow',
  red: '.status-red',
  blocked: '.has-blockers',
  active: '.wip-active', // WIP > 0 and < 100
  complete: '.wip-complete', // WIP = 100
  stalled: '.wip-stalled', // WIP = 0
};

/**
 * Isotope sort functions
 * These will be passed to Isotope's getSortData option
 */
const ISOTOPE_SORT_KEYS = ['wip', 'name', 'status', 'owner', 'blockers'] as const;
type IsotopeSortKey = (typeof ISOTOPE_SORT_KEYS)[number];

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Generate a text-based WIP meter bar
 * e.g., 75% -> "######..  " (6 filled, 2 empty)
 */
function wipMeter(pct: number, width: number = 10): string {
  const clamped = Math.max(0, Math.min(100, pct));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
}

/**
 * Get CSS class for WIP percentage
 */
function wipColorClass(pct: number): string {
  if (pct >= 90) return 'text-emerald-400';
  if (pct >= 60) return 'text-cyan-400';
  if (pct >= 30) return 'text-amber-400';
  return 'text-red-400';
}

/**
 * Get CSS class for lane status
 */
function statusColorClass(status: string): string {
  switch (status) {
    case 'green':
      return 'text-emerald-400';
    case 'yellow':
      return 'text-amber-400';
    case 'red':
      return 'text-red-400';
    default:
      return 'text-muted-foreground';
  }
}

/**
 * Get background CSS class for lane status
 */
function statusBgClass(status: string): string {
  switch (status) {
    case 'green':
      return 'bg-emerald-500/20 border-emerald-500/30';
    case 'yellow':
      return 'bg-amber-500/20 border-amber-500/30';
    case 'red':
      return 'bg-red-500/20 border-red-500/30';
    default:
      return 'bg-muted/20 border-border/30';
  }
}

/**
 * Build CSS classes for a lane item for Isotope filtering
 */
function laneFilterClasses(lane: Lane): string {
  const classes: string[] = ['lane-item'];
  classes.push(`status-${lane.status}`);
  if (lane.blockers.length > 0) classes.push('has-blockers');
  if (lane.wip_pct === 0) classes.push('wip-stalled');
  else if (lane.wip_pct === 100) classes.push('wip-complete');
  else classes.push('wip-active');
  return classes.join(' ');
}

// ============================================================================
// Component
// ============================================================================

export const IsotopeLanesGrid = component$<IsotopeLanesGridProps>(
  ({
    refreshIntervalMs = 30000,
    events,
    initialFilter = 'all',
    initialSort = 'wip',
    enableDrag = false,
  }) => {
    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------
    const containerRef = useSignal<HTMLDivElement>();
    const lanesState = useSignal<LanesState | null>(null);
    const loading = useSignal(true);
    const error = useSignal<string | null>(null);
    const lastFetch = useSignal<string | null>(null);

    // Isotope state
    const isotopeStatus = useSignal<'idle' | 'ready' | 'unavailable'>('idle');
    const isotopeError = useSignal<string | null>(null);
    const activeFilter = useSignal(initialFilter);
    const activeSort = useSignal(initialSort);
    const sortAscending = useSignal(true);

    // Expanded lane (for detail view)
    const expandedLaneId = useSignal<string | null>(null);

    // -------------------------------------------------------------------------
    // Data Fetching
    // -------------------------------------------------------------------------
    const fetchLanes = $(async () => {
      try {
        const res = await fetch('/api/fs/nucleus/state/lanes.json');
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        lanesState.value = data;
        lastFetch.value = new Date().toISOString();
        error.value = null;
      } catch (err: any) {
        error.value = err?.message || 'Failed to fetch lanes';
      } finally {
        loading.value = false;
      }
    });

    // Watch for bus events
    useVisibleTask$(({ track }) => {
      if (!events) return;
      track(() => events.value);

      const recent = events.value.slice(-10).reverse();
      const update = recent.find((event) => event.topic === 'operator.lanes.state');

      if (update?.data && !loading.value) {
        fetchLanes();
      }
    });

    // Initial fetch and periodic refresh
    useVisibleTask$(({ cleanup, track }) => {
      track(() => refreshIntervalMs);

      fetchLanes();
      const interval = setInterval(fetchLanes, refreshIntervalMs);
      cleanup(() => clearInterval(interval));
    });

    // -------------------------------------------------------------------------
    // Isotope Integration
    // -------------------------------------------------------------------------
    useVisibleTask$(async ({ cleanup }) => {
      const container = containerRef.value;
      if (!container) return;

      // Dynamic import - Isotope is browser-only
      let Isotope: any = null;
      let Draggabilly: any = null;

      try {
        // Dynamic import isotope-layout (browser-only)
        const modIsotope: any = await import('isotope-layout');
        Isotope = modIsotope?.default || modIsotope;

        if (enableDrag) {
          const modDraggabilly: any = await import('draggabilly');
          Draggabilly = modDraggabilly?.default || modDraggabilly;
        }

        const isotope = new Isotope(container, {
          itemSelector: '.lane-item',
          layoutMode: 'masonry',
          masonry: {
            columnWidth: '.lane-sizer',
            gutter: 12,
          },
          percentPosition: true,
          transitionDuration: '0.3s',
          getSortData: {
            wip: '[data-wip] parseInt',
            name: '[data-name]',
            status: '[data-status]',
            owner: '[data-owner]',
            blockers: '[data-blockers] parseInt',
          },
        });

        (container as any).__isotope = isotope;

        // Bind Draggabilly if enabled
        const draggies: any[] = [];
        if (enableDrag && Draggabilly) {
          const elems: HTMLElement[] = Array.from(container.querySelectorAll('.lane-item'));
          for (const elem of elems) {
            if ((elem as any).__draggie) continue;
            const draggie = new Draggabilly(elem);
            (elem as any).__draggie = draggie;
            // isotope.bindDraggabillyEvents is not available - use position events
            draggies.push(draggie);
          }
        }

        // Initial layout
        isotope.layout();
        isotopeStatus.value = 'ready';

        // Watch for DOM changes
        const observer = new MutationObserver(() => {
          isotope.reloadItems();
          isotope.layout();
        });
        observer.observe(container, { childList: true, subtree: false });

        // Resize handler
        const onResize = () => isotope.layout();
        window.addEventListener('resize', onResize);

        cleanup(() => {
          window.removeEventListener('resize', onResize);
          observer.disconnect();
          for (const d of draggies) {
            try { d?.destroy?.(); } catch {}
          }
          try { isotope?.destroy?.(); } catch {}
          try { delete (container as any).__isotope; } catch {}
        });
      } catch (e) {
        isotopeStatus.value = 'unavailable';
        isotopeError.value = String(e);
      }
    });

    // -------------------------------------------------------------------------
    // Filter/Sort Actions
    // -------------------------------------------------------------------------
    const applyFilter = $((filterName: string) => {
      activeFilter.value = filterName;
      const container = containerRef.value;
      const isotope = (container as any)?.__isotope;
      if (isotope) {
        isotope.arrange({ filter: ISOTOPE_FILTERS[filterName] || '*' });
      }
    });

    const applySort = $((sortKey: string) => {
      // Toggle ascending/descending if same key clicked
      if (activeSort.value === sortKey) {
        sortAscending.value = !sortAscending.value;
      } else {
        activeSort.value = sortKey;
        sortAscending.value = true;
      }

      const container = containerRef.value;
      const isotope = (container as any)?.__isotope;
      if (isotope) {
        isotope.arrange({
          sortBy: sortKey,
          sortAscending: sortAscending.value,
        });
      }
    });

    const relayout = $(() => {
      const container = containerRef.value;
      const isotope = (container as any)?.__isotope;
      isotope?.layout?.();
    });

    // -------------------------------------------------------------------------
    // Computed Values
    // -------------------------------------------------------------------------
    const stats = useComputed$(() => {
      if (!lanesState.value) {
        return {
          totalLanes: 0,
          avgWip: 0,
          greenCount: 0,
          yellowCount: 0,
          redCount: 0,
          activeAgents: 0,
        };
      }
      const lanes = lanesState.value.lanes || [];
      const agents = lanesState.value.agents || [];

      const totalLanes = lanes.length;
      const avgWip =
        totalLanes > 0
          ? Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / totalLanes)
          : 0;
      const greenCount = lanes.filter((l) => l.status === 'green').length;
      const yellowCount = lanes.filter((l) => l.status === 'yellow').length;
      const redCount = lanes.filter((l) => l.status === 'red').length;
      const activeAgents = agents.filter((a) => a.status === 'active').length;

      return { totalLanes, avgWip, greenCount, yellowCount, redCount, activeAgents };
    });

    // -------------------------------------------------------------------------
    // Render
    // -------------------------------------------------------------------------
    return (
      <div class="rounded-lg border border-border bg-card p-4">
        {/* Header */}
        <div class="flex items-center justify-between mb-3">
          <div class="flex items-center gap-2">
            <h3 class="text-sm font-semibold text-muted-foreground">LANES GRID</h3>
            <span class="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
              {stats.value.totalLanes} lanes
            </span>
            <span class="text-[10px] px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
              isotope
            </span>
          </div>
          <div class="flex items-center gap-2">
            <span
              class={`text-[10px] px-2 py-0.5 rounded border ${
                isotopeStatus.value === 'ready'
                  ? 'border-green-500/30 text-green-400 bg-green-500/10'
                  : isotopeStatus.value === 'unavailable'
                    ? 'border-rose-500/30 text-rose-400 bg-rose-500/10'
                    : 'border-border bg-muted/30'
              }`}
            >
              {isotopeStatus.value}
            </span>
            {lastFetch.value && (
              <span class="text-[9px] text-muted-foreground mono">
                {lastFetch.value.slice(11, 19)}
              </span>
            )}
            <button
              onClick$={fetchLanes}
              class="text-[10px] px-2 py-1 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground"
              title="Refresh lanes data"
            >
              Refresh
            </button>
            <button
              onClick$={relayout}
              class="text-[10px] px-2 py-1 rounded bg-muted/30 hover:bg-muted/50 text-muted-foreground"
              title="Force Isotope relayout"
              disabled={isotopeStatus.value !== 'ready'}
            >
              Relayout
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div class="flex flex-wrap gap-1 mb-3">
          <span class="text-[10px] text-muted-foreground mr-1">Filter:</span>
          {Object.keys(ISOTOPE_FILTERS).map((filterName) => (
            <button
              key={filterName}
              onClick$={() => applyFilter(filterName)}
              class={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                activeFilter.value === filterName
                  ? 'border-primary/50 bg-primary/20 text-primary'
                  : 'border-border/30 bg-muted/20 text-muted-foreground hover:bg-muted/40'
              }`}
            >
              {filterName}
            </button>
          ))}
        </div>

        {/* Sort Bar */}
        <div class="flex flex-wrap gap-1 mb-3">
          <span class="text-[10px] text-muted-foreground mr-1">Sort:</span>
          {ISOTOPE_SORT_KEYS.map((sortKey) => (
            <button
              key={sortKey}
              onClick$={() => applySort(sortKey)}
              class={`text-[10px] px-2 py-0.5 rounded border transition-colors ${
                activeSort.value === sortKey
                  ? 'border-primary/50 bg-primary/20 text-primary'
                  : 'border-border/30 bg-muted/20 text-muted-foreground hover:bg-muted/40'
              }`}
            >
              {sortKey} {activeSort.value === sortKey && (sortAscending.value ? '↑' : '↓')}
            </button>
          ))}
        </div>

        {/* Isotope Unavailable Warning */}
        {isotopeStatus.value === 'unavailable' && isotopeError.value && (
          <div class="text-xs text-amber-300 border border-amber-500/30 bg-amber-500/10 rounded p-2 mb-3">
            <strong>Isotope unavailable:</strong> {isotopeError.value}
            <br />
            <span class="text-[10px] text-muted-foreground">
              Falling back to basic grid layout. Install isotope-layout for full functionality.
            </span>
          </div>
        )}

        {/* Stats Summary */}
        <div class="flex flex-wrap gap-2 mb-3 text-[10px]">
          <span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            {stats.value.greenCount} green
          </span>
          <span class="px-2 py-0.5 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
            {stats.value.yellowCount} yellow
          </span>
          <span class="px-2 py-0.5 rounded bg-red-500/10 text-red-400 border border-red-500/20">
            {stats.value.redCount} red
          </span>
          <span class="px-2 py-0.5 rounded bg-purple-500/10 text-purple-400 border border-purple-500/20">
            {stats.value.activeAgents} agents
          </span>
          <span class="px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
            {stats.value.avgWip}% avg
          </span>
        </div>

        {/* Loading/Error States */}
        {loading.value ? (
          <div class="text-xs text-muted-foreground animate-pulse">Loading lanes...</div>
        ) : error.value ? (
          <div class="text-xs text-red-400">{error.value}</div>
        ) : !lanesState.value?.lanes?.length ? (
          <div class="text-xs text-muted-foreground">No lanes found.</div>
        ) : (
          /* Isotope Container */
          <div
            ref={containerRef}
            class="relative w-full min-h-[400px] rounded-lg border border-border/30 bg-black/10 p-3"
          >
            {/* Sizer element for Isotope masonry */}
            <div class="lane-sizer w-full sm:w-[calc(50%-6px)] lg:w-[calc(33.333%-8px)]" />

            {/* Lane Items */}
            {lanesState.value.lanes.map((lane) => (
              <div
                key={lane.id}
                class={`${laneFilterClasses(lane)} w-full sm:w-[calc(50%-6px)] lg:w-[calc(33.333%-8px)] mb-3 rounded-md border p-2 ${statusBgClass(lane.status)} transition-all`}
                data-wip={lane.wip_pct}
                data-name={lane.name}
                data-status={lane.status}
                data-owner={lane.owner}
                data-blockers={lane.blockers.length}
              >
                {/* Lane Header */}
                <div class="flex items-center justify-between gap-2">
                  <div class="flex items-center gap-2 min-w-0">
                    <span class={`text-[10px] font-bold ${statusColorClass(lane.status)}`}>
                      {'\u25CF'}
                    </span>
                    <span class="text-xs font-medium text-foreground truncate">{lane.name}</span>
                  </div>
                  <span class="text-[10px] px-1.5 py-0.5 rounded bg-muted/30 text-muted-foreground">
                    @{lane.owner}
                  </span>
                </div>

                {/* WIP Meter */}
                <div class="mt-1.5 flex items-center gap-2">
                  <code class={`text-[11px] mono ${wipColorClass(lane.wip_pct)}`}>
                    {wipMeter(lane.wip_pct, 12)}
                  </code>
                  <span class={`text-[10px] font-bold ${wipColorClass(lane.wip_pct)}`}>
                    {lane.wip_pct}%
                  </span>
                </div>

                {/* Description */}
                {lane.description && (
                  <div class="mt-1 text-[10px] text-muted-foreground line-clamp-1">
                    {lane.description}
                  </div>
                )}

                {/* Blockers */}
                {lane.blockers.length > 0 && (
                  <div class="mt-1 flex items-center gap-1">
                    <span class="text-[9px] text-red-400">BLOCKED:</span>
                    <span class="text-[9px] text-red-300 truncate">{lane.blockers[0]}</span>
                  </div>
                )}

                {/* Expand Button */}
                <button
                  onClick$={() => {
                    expandedLaneId.value =
                      expandedLaneId.value === lane.id ? null : lane.id;
                  }}
                  class="mt-2 w-full text-[9px] py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground border border-border/30"
                >
                  {expandedLaneId.value === lane.id ? 'Collapse ▲' : 'Details ▼'}
                </button>

                {/* Expanded Details */}
                {expandedLaneId.value === lane.id && (
                  <div class="mt-2 pt-2 border-t border-border/30 space-y-2 text-[10px]">
                    {/* Next Actions */}
                    {lane.next_actions.length > 0 && (
                      <div>
                        <div class="text-muted-foreground mb-1">Next Actions:</div>
                        <ul class="list-disc list-inside text-foreground/80">
                          {lane.next_actions.map((action, i) => (
                            <li key={i}>{action}</li>
                          ))}
                        </ul>
                      </div>
                    )}

                    {/* Commits */}
                    {lane.commits.length > 0 && (
                      <div>
                        <div class="text-muted-foreground mb-1">Commits:</div>
                        <div class="mono text-[9px] text-foreground/60">
                          {lane.commits.slice(0, 3).join(', ')}
                          {lane.commits.length > 3 && ` +${lane.commits.length - 3} more`}
                        </div>
                      </div>
                    )}

                    {/* History Preview */}
                    {lane.history.length > 0 && (
                      <div>
                        <div class="text-muted-foreground mb-1">Recent History:</div>
                        {lane.history.slice(0, 3).map((h, i) => (
                          <div key={i} class="flex items-center gap-2 text-[9px]">
                            <span class="text-muted-foreground/60">
                              {h.ts.slice(0, 10)}
                            </span>
                            <span class={wipColorClass(h.wip_pct)}>{h.wip_pct}%</span>
                            <span class="text-foreground/70 truncate">{h.note}</span>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* Tier/Host info */}
                    {(lane.tier || lane.host) && (
                      <div class="flex gap-2 text-[9px] text-muted-foreground/70">
                        {lane.tier && <span>tier: {lane.tier}</span>}
                        {lane.host && <span>host: {lane.host}</span>}
                        {lane.slot !== undefined && <span>slot: {lane.slot}</span>}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Agent Membership */}
        {lanesState.value?.agents && lanesState.value.agents.length > 0 && (
          <div class="mt-3 pt-3 border-t border-border/50">
            <div class="text-[10px] text-muted-foreground mb-2">AGENTS</div>
            <div class="flex flex-wrap gap-1">
              {lanesState.value.agents.map((agent) => (
                <span
                  key={agent.id}
                  class={`text-[9px] px-1.5 py-0.5 rounded border ${
                    agent.status === 'active'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : 'bg-muted/20 text-muted-foreground border-border/30'
                  }`}
                  title={agent.lane ? `Lane: ${agent.lane}` : 'No lane assigned'}
                >
                  @{agent.id}
                  {agent.lane && (
                    <span class="text-muted-foreground/60 ml-1">[{agent.lane.slice(0, 8)}]</span>
                  )}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }
);

export default IsotopeLanesGrid;
