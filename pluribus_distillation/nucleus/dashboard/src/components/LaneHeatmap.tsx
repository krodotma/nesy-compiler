/**
 * LaneHeatmap - Activity heatmap for lanes (GitHub-style)
 *
 * Phase 4, Iteration 29 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - GitHub-style contribution heatmap
 * - Color by WIP progress or activity count
 * - Click to drill down to specific day
 * - Tooltip with details
 * - Multiple lanes stacked or aggregated
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

export interface HeatmapData {
  date: string;
  laneId?: string;
  laneName?: string;
  value: number;
  events?: number;
  note?: string;
}

export interface LaneHeatmapProps {
  /** Heatmap data points */
  data: HeatmapData[];
  /** Number of weeks to show */
  weeks?: number;
  /** Color mode */
  colorMode?: 'wip' | 'activity';
  /** Callback when cell is clicked */
  onCellClick$?: QRL<(data: HeatmapData) => void>;
  /** Show lane labels */
  showLaneLabels?: boolean;
  /** Aggregate all lanes */
  aggregateLanes?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

function getWipColor(value: number): string {
  if (value >= 90) return '#22c55e';  // green-500
  if (value >= 70) return '#84cc16';  // lime-500
  if (value >= 50) return '#eab308';  // yellow-500
  if (value >= 30) return '#f97316';  // orange-500
  if (value >= 10) return '#ef4444';  // red-500
  if (value > 0) return '#fca5a5';    // red-300
  return 'rgba(255,255,255,0.05)';    // empty
}

function getActivityColor(count: number): string {
  if (count >= 20) return '#22c55e';
  if (count >= 15) return '#84cc16';
  if (count >= 10) return '#a3e635';
  if (count >= 5) return '#d9f99d';
  if (count >= 1) return '#ecfccb';
  return 'rgba(255,255,255,0.05)';
}

function formatDate(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

function getMonthLabel(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short' });
  } catch {
    return '';
  }
}

function getDayOfWeek(dateStr: string): number {
  return new Date(dateStr).getDay();
}

function generateDateRange(weeks: number): string[] {
  const dates: string[] = [];
  const today = new Date();
  const startDate = new Date(today);
  startDate.setDate(startDate.getDate() - (weeks * 7) + 1);

  // Adjust to start on Sunday
  const dayOffset = startDate.getDay();
  startDate.setDate(startDate.getDate() - dayOffset);

  for (let i = 0; i < weeks * 7; i++) {
    const date = new Date(startDate);
    date.setDate(date.getDate() + i);
    dates.push(date.toISOString().slice(0, 10));
  }

  return dates;
}

// ============================================================================
// Component
// ============================================================================

export const LaneHeatmap = component$<LaneHeatmapProps>(({
  data,
  weeks = 12,
  colorMode = 'wip',
  onCellClick$,
  showLaneLabels = false,
  aggregateLanes = true,
}) => {
  // State
  const hoveredCell = useSignal<{ date: string; laneId?: string } | null>(null);
  const selectedDate = useSignal<string | null>(null);

  // Generate date range
  const dates = useComputed$(() => generateDateRange(weeks));

  // Get unique lanes
  const lanes = useComputed$(() => {
    if (aggregateLanes) return [{ id: 'all', name: 'All Lanes' }];
    const laneMap = new Map<string, string>();
    data.forEach(d => {
      if (d.laneId && d.laneName) {
        laneMap.set(d.laneId, d.laneName);
      }
    });
    return Array.from(laneMap.entries()).map(([id, name]) => ({ id, name }));
  });

  // Create data lookup
  const dataMap = useComputed$(() => {
    const map = new Map<string, HeatmapData>();

    if (aggregateLanes) {
      // Aggregate by date
      const aggregated = new Map<string, { total: number; count: number; events: number }>();
      data.forEach(d => {
        const existing = aggregated.get(d.date) || { total: 0, count: 0, events: 0 };
        aggregated.set(d.date, {
          total: existing.total + d.value,
          count: existing.count + 1,
          events: existing.events + (d.events || 0),
        });
      });

      aggregated.forEach((agg, date) => {
        map.set(`all-${date}`, {
          date,
          value: Math.round(agg.total / agg.count),
          events: agg.events,
        });
      });
    } else {
      data.forEach(d => {
        if (d.laneId) {
          map.set(`${d.laneId}-${d.date}`, d);
        }
      });
    }

    return map;
  });

  // Group dates by week and calculate month labels
  const weekGroups = useComputed$(() => {
    const groups: string[][] = [];
    let currentWeek: string[] = [];

    dates.value.forEach((date, i) => {
      currentWeek.push(date);
      if ((i + 1) % 7 === 0) {
        groups.push(currentWeek);
        currentWeek = [];
      }
    });

    return groups;
  });

  // Month labels with positions
  const monthLabels = useComputed$(() => {
    const labels: { label: string; weekIndex: number }[] = [];
    let lastMonth = '';

    weekGroups.value.forEach((week, weekIndex) => {
      const firstDay = week[0];
      const month = getMonthLabel(firstDay);
      if (month !== lastMonth) {
        labels.push({ label: month, weekIndex });
        lastMonth = month;
      }
    });

    return labels;
  });

  // Stats
  const stats = useComputed$(() => {
    const values = data.map(d => d.value);
    const events = data.reduce((sum, d) => sum + (d.events || 0), 0);
    return {
      avg: values.length > 0 ? Math.round(values.reduce((a, b) => a + b, 0) / values.length) : 0,
      max: values.length > 0 ? Math.max(...values) : 0,
      min: values.length > 0 ? Math.min(...values) : 0,
      events,
      days: new Set(data.map(d => d.date)).size,
    };
  });

  // Get cell data
  const getCellData = $((laneId: string, date: string): HeatmapData | null => {
    return dataMap.value.get(`${laneId}-${date}`) || null;
  });

  // Handle cell click
  const handleCellClick = $(async (laneId: string, date: string) => {
    selectedDate.value = selectedDate.value === date ? null : date;
    const cellData = dataMap.value.get(`${laneId}-${date}`);
    if (cellData && onCellClick$) {
      await onCellClick$(cellData);
    }
  });

  const cellSize = 12;
  const cellGap = 3;
  const dayLabelWidth = 20;
  const monthLabelHeight = 20;
  const laneLabelWidth = showLaneLabels ? 100 : 0;

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">ACTIVITY HEATMAP</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {weeks} weeks
          </span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
            {stats.value.days} active days
          </span>
        </div>
        <div class="text-[9px] text-muted-foreground">
          {colorMode === 'wip' ? 'Color by WIP %' : 'Color by activity count'}
        </div>
      </div>

      {/* Heatmap */}
      <div class="p-3 overflow-x-auto">
        {lanes.value.map((lane, laneIndex) => (
          <div key={lane.id} class="flex items-start gap-2 mb-2">
            {/* Lane label */}
            {showLaneLabels && (
              <div
                class="flex-shrink-0 text-[9px] text-muted-foreground truncate py-1"
                style={{ width: `${laneLabelWidth}px` }}
              >
                {lane.name}
              </div>
            )}

            <div>
              {/* Month labels */}
              {laneIndex === 0 && (
                <div class="flex mb-1" style={{ marginLeft: `${dayLabelWidth}px` }}>
                  {monthLabels.value.map((m, i) => (
                    <div
                      key={i}
                      class="text-[9px] text-muted-foreground"
                      style={{
                        position: 'absolute',
                        left: `${m.weekIndex * (cellSize + cellGap) + dayLabelWidth + laneLabelWidth}px`,
                      }}
                    >
                      {m.label}
                    </div>
                  ))}
                  <div style={{ height: `${monthLabelHeight}px` }} />
                </div>
              )}

              {/* Grid */}
              <div class="flex">
                {/* Day labels */}
                <div class="flex flex-col gap-[3px] mr-1" style={{ width: `${dayLabelWidth}px` }}>
                  {['', 'M', '', 'W', '', 'F', ''].map((day, i) => (
                    <div
                      key={i}
                      class="text-[8px] text-muted-foreground text-right pr-1"
                      style={{ height: `${cellSize}px`, lineHeight: `${cellSize}px` }}
                    >
                      {day}
                    </div>
                  ))}
                </div>

                {/* Cells */}
                <div class="flex gap-[3px]">
                  {weekGroups.value.map((week, weekIndex) => (
                    <div key={weekIndex} class="flex flex-col gap-[3px]">
                      {week.map(date => {
                        const cellData = dataMap.value.get(`${lane.id}-${date}`);
                        const value = cellData?.value ?? 0;
                        const events = cellData?.events ?? 0;
                        const isHovered =
                          hoveredCell.value?.date === date &&
                          hoveredCell.value?.laneId === lane.id;
                        const isSelected = selectedDate.value === date;

                        const color = colorMode === 'wip'
                          ? getWipColor(value)
                          : getActivityColor(events);

                        return (
                          <div
                            key={date}
                            onMouseEnter$={() => { hoveredCell.value = { date, laneId: lane.id }; }}
                            onMouseLeave$={() => { hoveredCell.value = null; }}
                            onClick$={() => handleCellClick(lane.id, date)}
                            class={`rounded-sm cursor-pointer transition-all ${
                              isSelected ? 'ring-2 ring-primary' : ''
                            } ${isHovered ? 'ring-1 ring-white/50' : ''}`}
                            style={{
                              width: `${cellSize}px`,
                              height: `${cellSize}px`,
                              backgroundColor: color,
                            }}
                            title={`${formatDate(date)}: ${value}% WIP, ${events} events`}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tooltip */}
      {hoveredCell.value && (
        <div class="fixed z-50 pointer-events-none">
          {/* Tooltip would be positioned via JS in a real implementation */}
        </div>
      )}

      {/* Legend */}
      <div class="p-2 border-t border-border/50 flex items-center justify-between text-[9px] text-muted-foreground">
        <div class="flex items-center gap-2">
          <span>Less</span>
          <div class="flex gap-[2px]">
            {colorMode === 'wip' ? (
              <>
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#fca5a5' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#f97316' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#eab308' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#84cc16' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#22c55e' }} />
              </>
            ) : (
              <>
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: 'rgba(255,255,255,0.05)' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#ecfccb' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#d9f99d' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#a3e635' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#84cc16' }} />
                <div class="w-3 h-3 rounded-sm" style={{ backgroundColor: '#22c55e' }} />
              </>
            )}
          </div>
          <span>More</span>
        </div>
        <div class="flex items-center gap-4">
          <span>Avg: {stats.value.avg}%</span>
          <span>Events: {stats.value.events}</span>
        </div>
      </div>
    </div>
  );
});

export default LaneHeatmap;
