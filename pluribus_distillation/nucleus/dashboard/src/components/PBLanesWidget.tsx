/**
 * PBLanesWidget - Displays PBLANES lanes progress with WIP meters
 *
 * Fetches lanes state from /api/fs/nucleus/state/lanes.json and displays:
 * - Lane progress bars in text meter format
 * - Agent membership
 * - Status indicators
 * - Periodic refresh
 */

import { component$, useSignal, useVisibleTask$, useComputed$, $, type Signal } from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';
import { NeonTitle, NeonBadge, NeonSectionHeader } from './ui/NeonTitle';
import { FreshnessBadge } from './ui/FreshnessBadge';
import { AccessibleLaneCard } from './LaneAccessibility';

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

export interface PBLanesWidgetProps {
  refreshIntervalMs?: number;
  maxLanes?: number;
  defaultShowAll?: boolean;
  events?: Signal<BusEvent[]>;
}

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

export const PBLanesWidget = component$<PBLanesWidgetProps>(({
  refreshIntervalMs = 30000,
  maxLanes = 100,
  defaultShowAll = true,
  events,
}) => {
  const lanesState = useSignal<LanesState | null>(null);
  const loading = useSignal(true);
  const error = useSignal<string | null>(null);
  const lastFetch = useSignal<string | null>(null);
  const showAll = useSignal(defaultShowAll);

  // Watch for bus events
  useVisibleTask$(({ track }) => {
    if (!events) return;
    track(() => events.value);

    const recent = events.value.slice(-10).reverse();
    const update = recent.find((event) => event.topic === 'operator.lanes.state');

    if (update?.data && !loading.value) {
      fetch('/api/fs/nucleus/state/lanes.json')
        .then((res) => res.json())
        .then((data) => {
          lanesState.value = data;
          lastFetch.value = update.iso;
        })
        .catch(() => {});
    }
  });

  // Initial fetch and periodic refresh
  useVisibleTask$(({ cleanup, track }) => {
    // Track signals explicitly
    track(() => refreshIntervalMs);

    const doFetch = async () => {
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
    };

    doFetch();
    const interval = setInterval(doFetch, refreshIntervalMs);
    cleanup(() => clearInterval(interval));
  });

  // Computed stats
  const stats = useComputed$(() => {
    if (!lanesState.value) {
      return { totalLanes: 0, avgWip: 0, greenCount: 0, yellowCount: 0, redCount: 0, activeAgents: 0 };
    }
    const lanes = lanesState.value.lanes || [];
    const agents = lanesState.value.agents || [];

    const totalLanes = lanes.length;
    const avgWip = totalLanes > 0
      ? Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / totalLanes)
      : 0;
    const greenCount = lanes.filter(l => l.status === 'green').length;
    const yellowCount = lanes.filter(l => l.status === 'yellow').length;
    const redCount = lanes.filter(l => l.status === 'red').length;
    const activeAgents = agents.filter(a => a.status === 'active').length;

    return { totalLanes, avgWip, greenCount, yellowCount, redCount, activeAgents };
  });

  // Sorted lanes by WIP (incomplete first)
  const sortedLanes = useComputed$(() => {
    if (!lanesState.value) return [];
    const sorted = [...lanesState.value.lanes]
      .sort((a, b) => a.wip_pct - b.wip_pct);
    return showAll.value ? sorted : sorted.slice(0, Math.min(maxLanes, 8));
  });

  // Check if there are more lanes than displayed
  const hasMoreLanes = useComputed$(() => {
    if (!lanesState.value || showAll.value) return false;
    return lanesState.value.lanes.length > 8;
  });

  const hiddenCount = useComputed$(() => {
    if (!lanesState.value || showAll.value) return 0;
    return Math.max(0, lanesState.value.lanes.length - 8);
  });

  // Grouped lanes by status for collapsible sections
  const groupedLanes = useComputed$(() => {
    if (!lanesState.value) return { red: [], yellow: [], green: [] };
    const lanes = lanesState.value.lanes;
    return {
      red: lanes.filter(l => l.status === 'red').sort((a, b) => a.wip_pct - b.wip_pct),
      yellow: lanes.filter(l => l.status === 'yellow').sort((a, b) => a.wip_pct - b.wip_pct),
      green: lanes.filter(l => l.status === 'green').sort((a, b) => a.wip_pct - b.wip_pct),
    };
  });

  // Collapsed state for each section (persisted in localStorage)
  const collapsedSections = useSignal<Record<string, boolean>>({});

  // Expanded lane for detail view (Iteration 4)
  const expandedLaneId = useSignal<string | null>(null);
  const focusedLaneIndex = useSignal<number>(-1);

  // Toggle section collapse
  const toggleSection = $((section: string) => {
    collapsedSections.value = {
      ...collapsedSections.value,
      [section]: !collapsedSections.value[section]
    };
    // Persist to localStorage
    if (typeof window !== 'undefined') {
      localStorage.setItem('pblanes-collapsed', JSON.stringify(collapsedSections.value));
    }
  });

  // Load collapsed state from localStorage on mount
  useVisibleTask$(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('pblanes-collapsed');
      if (saved) {
        try {
          collapsedSections.value = JSON.parse(saved);
        } catch {}
      }
    }
  });

  // Toggle lane expansion (Iteration 4)
  const toggleLaneExpand = $((laneId: string) => {
    expandedLaneId.value = expandedLaneId.value === laneId ? null : laneId;
  });

  // Get flat list of visible lanes for keyboard nav
  const visibleLanes = useComputed$(() => {
    const lanes: Lane[] = [];
    if (!collapsedSections.value['red']) lanes.push(...groupedLanes.value.red);
    if (!collapsedSections.value['yellow']) lanes.push(...groupedLanes.value.yellow);
    if (!collapsedSections.value['green']) lanes.push(...groupedLanes.value.green);
    return lanes;
  });

  // Keyboard navigation (j/k to move, Enter to expand)
  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const handleKeyDown = (e: KeyboardEvent) => {
      // Only handle if widget is focused or no other input is focused
      const activeTag = document.activeElement?.tagName?.toLowerCase();
      if (activeTag === 'input' || activeTag === 'textarea') return;

      const lanes = visibleLanes.value;
      if (lanes.length === 0) return;

      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        focusedLaneIndex.value = Math.min(focusedLaneIndex.value + 1, lanes.length - 1);
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        focusedLaneIndex.value = Math.max(focusedLaneIndex.value - 1, 0);
      } else if (e.key === 'Enter' && focusedLaneIndex.value >= 0) {
        e.preventDefault();
        const lane = lanes[focusedLaneIndex.value];
        if (lane) {
          expandedLaneId.value = expandedLaneId.value === lane.id ? null : lane.id;
        }
      } else if (e.key === 'Escape') {
        expandedLaneId.value = null;
        focusedLaneIndex.value = -1;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    cleanup(() => window.removeEventListener('keydown', handleKeyDown));
  });

  // Check if a lane is focused by keyboard nav
  const isLaneFocused = (laneId: string): boolean => {
    const idx = focusedLaneIndex.value;
    if (idx < 0) return false;
    const lanes = visibleLanes.value;
    return lanes[idx]?.id === laneId;
  };

  const statusColor = (status: string) => {
    switch (status) {
      case 'green': return 'text-emerald-400 drop-shadow-[0_0_4px_rgba(52,211,153,0.4)]';
      case 'yellow': return 'text-amber-400 drop-shadow-[0_0_4px_rgba(251,191,36,0.4)]';
      case 'red': return 'text-red-400 drop-shadow-[0_0_4px_rgba(248,113,113,0.4)]';
      default: return 'text-glass-text-muted';
    }
  };

  const statusBg = (status: string) => {
    switch (status) {
      case 'green': return 'glass-status-healthy';
      case 'yellow': return 'glass-status-warning';
      case 'red': return 'glass-status-critical';
      default: return 'glass-chip';
    }
  };

  const wipColor = (pct: number) => {
    if (pct >= 90) return 'text-emerald-400';
    if (pct >= 60) return 'text-cyan-400';
    if (pct >= 30) return 'text-amber-400';
    return 'text-red-400';
  };

  return (
    <div class="glass-surface-elevated glass-animate-enter p-4">
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <NeonTitle level="h3" color="cyan" size="sm" animation="flicker">SWE LANES</NeonTitle>
          <NeonBadge color="purple" glow>{stats.value.totalLanes} lanes</NeonBadge>
          <NeonBadge color="cyan" glow>{stats.value.avgWip}% avg</NeonBadge>
        </div>
        <div class="flex items-center gap-2">
          <FreshnessBadge
            timestamp={lanesState.value?.updated}
            source="lanes.json"
            ttlFresh={120}
            ttlRecent={600}
            ttlStale={3600}
          />
          <button
            onClick$={$(async () => {
              loading.value = true;
              try {
                const res = await fetch('/api/fs/nucleus/state/lanes.json');
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                const data = await res.json();
                lanesState.value = data;
                lastFetch.value = new Date().toISOString();
                error.value = null;
              } catch (err: any) {
                error.value = err?.message || 'Failed to fetch lanes';
              } finally {
                loading.value = false;
              }
            })}
            class="text-[10px] px-2 py-1 rounded glass-interactive glass-hover-glow text-glass-text-muted"
            title="Refresh lanes"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Stats summary */}
      <div class="flex flex-wrap gap-2 mb-3 text-[10px]">
        <NeonBadge color="emerald" glow>{stats.value.greenCount} green</NeonBadge>
        <NeonBadge color="amber" glow>{stats.value.yellowCount} yellow</NeonBadge>
        <NeonBadge color="rose" glow>{stats.value.redCount} red</NeonBadge>
        <NeonBadge color="purple" glow>{stats.value.activeAgents} agents</NeonBadge>
      </div>

      {loading.value ? (
        <div class="text-xs text-muted-foreground animate-pulse">Loading lanes...</div>
      ) : error.value ? (
        <div class="text-xs text-red-400">{error.value}</div>
      ) : sortedLanes.value.length === 0 ? (
        <div class="text-xs text-muted-foreground">No lanes found.</div>
      ) : (
        <div class="space-y-3 max-h-[500px] overflow-y-auto pr-1">
          {/* Red lanes section - Blocked/Critical */}
          {groupedLanes.value.red.length > 0 && (
            <div class="rounded-lg border border-red-500/30 overflow-hidden">
              <button
                onClick$={() => toggleSection('red')}
                class="w-full flex items-center justify-between px-3 py-2 bg-red-500/10 hover:bg-red-500/20 transition-colors"
              >
                <div class="flex items-center gap-2">
                  <span class="text-red-400 text-[10px] font-bold">●</span>
                  <NeonTitle level="span" color="rose" size="xs">CRITICAL</NeonTitle>
                  <NeonBadge color="rose" glow>{groupedLanes.value.red.length}</NeonBadge>
                </div>
                <span class="text-red-400 text-[10px]">{collapsedSections.value['red'] ? '▶' : '▼'}</span>
              </button>
              {!collapsedSections.value['red'] && (
                <div class="p-2 space-y-1.5">
                  {groupedLanes.value.red.map((lane) => (
                    <AccessibleLaneCard
                      key={lane.id}
                      lane={{
                        ...lane,
                        blockers: lane.blockers.length
                      }}
                      isSelected={expandedLaneId.value === lane.id}
                      onClick$={() => toggleLaneExpand(lane.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Yellow lanes section - In Progress */}
          {groupedLanes.value.yellow.length > 0 && (
            <div class="rounded-lg border border-amber-500/30 overflow-hidden">
              <button
                onClick$={() => toggleSection('yellow')}
                class="w-full flex items-center justify-between px-3 py-2 bg-amber-500/10 hover:bg-amber-500/20 transition-colors"
              >
                <div class="flex items-center gap-2">
                  <span class="text-amber-400 text-[10px] font-bold">●</span>
                  <NeonTitle level="span" color="amber" size="xs">IN PROGRESS</NeonTitle>
                  <NeonBadge color="amber" glow>{groupedLanes.value.yellow.length}</NeonBadge>
                </div>
                <span class="text-amber-400 text-[10px]">{collapsedSections.value['yellow'] ? '▶' : '▼'}</span>
              </button>
              {!collapsedSections.value['yellow'] && (
                <div class="p-2 space-y-1.5">
                  {groupedLanes.value.yellow.map((lane) => (
                    <AccessibleLaneCard
                      key={lane.id}
                      lane={{
                        ...lane,
                        blockers: lane.blockers.length
                      }}
                      isSelected={expandedLaneId.value === lane.id}
                      onClick$={() => toggleLaneExpand(lane.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Green lanes section - Complete/Healthy */}
          {groupedLanes.value.green.length > 0 && (
            <div class="rounded-lg border border-emerald-500/30 overflow-hidden">
              <button
                onClick$={() => toggleSection('green')}
                class="w-full flex items-center justify-between px-3 py-2 bg-emerald-500/10 hover:bg-emerald-500/20 transition-colors"
              >
                <div class="flex items-center gap-2">
                  <span class="text-emerald-400 text-[10px] font-bold">●</span>
                  <NeonTitle level="span" color="emerald" size="xs">COMPLETE</NeonTitle>
                  <NeonBadge color="emerald" glow>{groupedLanes.value.green.length}</NeonBadge>
                </div>
                <span class="text-emerald-400 text-[10px]">{collapsedSections.value['green'] ? '▶' : '▼'}</span>
              </button>
              {!collapsedSections.value['green'] && (
                <div class="p-2 space-y-1.5">
                  {groupedLanes.value.green.map((lane) => (
                    <AccessibleLaneCard
                      key={lane.id}
                      lane={{
                        ...lane,
                        blockers: lane.blockers.length
                      }}
                      isSelected={expandedLaneId.value === lane.id}
                      onClick$={() => toggleLaneExpand(lane.id)}
                    />
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Show All / Show Less toggle */}
      {!loading.value && !error.value && lanesState.value && lanesState.value.lanes.length > 8 && (
        <button
          onClick$={() => { showAll.value = !showAll.value; }}
          class="mt-2 w-full text-[10px] py-1.5 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground border border-border/30 transition-colors"
        >
          {showAll.value ? (
            <span>Show Less ▲</span>
          ) : (
            <span>Show All ({hiddenCount.value} more) ▼</span>
          )}
        </button>
      )}

      {/* Agent membership */}
      {lanesState.value?.agents && lanesState.value.agents.length > 0 && (
        <div class="mt-3 pt-3 border-t border-border/50">
          <NeonTitle level="div" color="purple" size="xs" class="mb-2">AGENTS</NeonTitle>
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
                {agent.lane && <span class="text-muted-foreground/60 ml-1">[{agent.lane.slice(0, 8)}]</span>}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

export default PBLanesWidget;
