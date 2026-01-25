/**
 * LaneGanttChart - Timeline visualization of lanes
 *
 * Phase 4, Iteration 26 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Timeline visualization with lanes as bars
 * - Dependencies between lanes (arrows)
 * - Critical path highlighting
 * - Zoom and pan controls
 * - Today marker
 * - Milestone markers
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import { Button } from './ui/Button';

// ============================================================================
// Types
// ============================================================================

export interface GanttLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  startDate: string;
  endDate?: string;
  estimatedEndDate?: string;
  dependencies?: string[];
  isCriticalPath?: boolean;
  milestones?: { date: string; label: string }[];
}

export interface LaneGanttChartProps {
  /** Lanes to display */
  lanes: GanttLane[];
  /** Start date of the chart view */
  startDate?: string;
  /** End date of the chart view */
  endDate?: string;
  /** Callback when lane is clicked */
  onLaneClick$?: QRL<(lane: GanttLane) => void>;
  /** Show dependencies */
  showDependencies?: boolean;
  /** Highlight critical path */
  highlightCriticalPath?: boolean;
  /** Time unit for grid */
  timeUnit?: 'day' | 'week' | 'month';
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return '#34d399';
    case 'yellow': return '#fbbf24';
    case 'red': return '#f87171';
    default: return '#6b7280';
  }
}

function getStatusBgColor(status: string): string {
  switch (status) {
    case 'green': return 'rgba(52, 211, 153, 0.3)';
    case 'yellow': return 'rgba(251, 191, 36, 0.3)';
    case 'red': return 'rgba(248, 113, 113, 0.3)';
    default: return 'rgba(107, 114, 128, 0.3)';
  }
}

function parseDate(dateStr: string): Date {
  return new Date(dateStr);
}

function formatDate(date: Date): string {
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function daysBetween(start: Date, end: Date): number {
  const diffMs = end.getTime() - start.getTime();
  return Math.ceil(diffMs / (1000 * 60 * 60 * 24));
}

function addDays(date: Date, days: number): Date {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}

// ============================================================================
// Component
// ============================================================================

export const LaneGanttChart = component$<LaneGanttChartProps>(({
  lanes,
  startDate: propStartDate,
  endDate: propEndDate,
  onLaneClick$,
  showDependencies = true,
  highlightCriticalPath = true,
  timeUnit = 'day',
}) => {
  // State
  const hoveredLaneId = useSignal<string | null>(null);
  const zoomLevel = useSignal(1);
  const scrollLeft = useSignal(0);

  // Calculate chart bounds
  const chartBounds = useComputed$(() => {
    let minDate = propStartDate ? parseDate(propStartDate) : new Date();
    let maxDate = propEndDate ? parseDate(propEndDate) : addDays(new Date(), 30);

    // Find actual min/max from lanes
    lanes.forEach(lane => {
      const start = parseDate(lane.startDate);
      const end = lane.endDate ? parseDate(lane.endDate) :
                  lane.estimatedEndDate ? parseDate(lane.estimatedEndDate) :
                  addDays(start, 14);

      if (start < minDate) minDate = start;
      if (end > maxDate) maxDate = end;
    });

    // Add padding
    minDate = addDays(minDate, -2);
    maxDate = addDays(maxDate, 5);

    const totalDays = daysBetween(minDate, maxDate);

    return { minDate, maxDate, totalDays };
  });

  // Generate time markers
  const timeMarkers = useComputed$(() => {
    const { minDate, totalDays } = chartBounds.value;
    const markers: { date: Date; label: string; isWeekStart: boolean; isMonthStart: boolean }[] = [];

    for (let i = 0; i <= totalDays; i++) {
      const date = addDays(minDate, i);
      const isWeekStart = date.getDay() === 1; // Monday
      const isMonthStart = date.getDate() === 1;

      if (timeUnit === 'day' ||
          (timeUnit === 'week' && isWeekStart) ||
          (timeUnit === 'month' && isMonthStart)) {
        markers.push({
          date,
          label: formatDate(date),
          isWeekStart,
          isMonthStart,
        });
      }
    }

    return markers;
  });

  // Calculate lane positions
  const lanePositions = useComputed$(() => {
    const { minDate, totalDays } = chartBounds.value;
    const dayWidth = 40 * zoomLevel.value;

    return lanes.map(lane => {
      const start = parseDate(lane.startDate);
      const end = lane.endDate ? parseDate(lane.endDate) :
                  lane.estimatedEndDate ? parseDate(lane.estimatedEndDate) :
                  addDays(start, 14);

      const startOffset = daysBetween(minDate, start);
      const duration = daysBetween(start, end);

      return {
        ...lane,
        x: startOffset * dayWidth,
        width: Math.max(duration * dayWidth, dayWidth),
        startOffset,
        duration,
      };
    });
  });

  // Today marker position
  const todayPosition = useComputed$(() => {
    const { minDate } = chartBounds.value;
    const today = new Date();
    const dayWidth = 40 * zoomLevel.value;
    return daysBetween(minDate, today) * dayWidth;
  });

  // Chart dimensions
  const chartWidth = useComputed$(() => {
    const dayWidth = 40 * zoomLevel.value;
    return chartBounds.value.totalDays * dayWidth;
  });

  const rowHeight = 48;
  const headerHeight = 60;
  const chartHeight = lanes.length * rowHeight + headerHeight + 20;

  // Zoom controls
  const zoomIn = $(() => {
    zoomLevel.value = Math.min(zoomLevel.value * 1.25, 3);
  });

  const zoomOut = $(() => {
    zoomLevel.value = Math.max(zoomLevel.value / 1.25, 0.5);
  });

  const resetZoom = $(() => {
    zoomLevel.value = 1;
  });

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">GANTT CHART</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {lanes.length} lanes
          </span>
          {highlightCriticalPath && lanes.some(l => l.isCriticalPath) && (
            <span class="text-[10px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">
              Critical Path
            </span>
          )}
        </div>
        <div class="flex items-center gap-1">
          <Button
            variant="tonal"
            onClick$={zoomOut}
            class="w-6 h-6 p-0 min-w-0 text-xs"
          >
            −
          </Button>
          <Button
            variant="tonal"
            onClick$={resetZoom}
            class="px-2 h-6 text-[10px]"
          >
            {Math.round(zoomLevel.value * 100)}%
          </Button>
          <Button
            variant="tonal"
            onClick$={zoomIn}
            class="w-6 h-6 p-0 min-w-0 text-xs"
          >
            +
          </Button>
        </div>
      </div>

      {/* Chart area */}
      <div class="overflow-x-auto overflow-y-hidden">
        <svg
          width={Math.max(chartWidth.value, 600)}
          height={chartHeight}
          class="select-none"
        >
          {/* Background grid */}
          <g class="grid">
            {timeMarkers.value.map((marker, i) => {
              const x = i * 40 * zoomLevel.value;
              return (
                <g key={i}>
                  <line
                    x1={x}
                    y1={headerHeight}
                    x2={x}
                    y2={chartHeight}
                    stroke={marker.isMonthStart ? 'rgba(255,255,255,0.15)' :
                            marker.isWeekStart ? 'rgba(255,255,255,0.08)' :
                            'rgba(255,255,255,0.03)'}
                    stroke-width="1"
                  />
                  {/* Date labels */}
                  {(timeUnit === 'day' || marker.isWeekStart || marker.isMonthStart) && (
                    <text
                      x={x + 4}
                      y={20}
                      fill="rgba(255,255,255,0.4)"
                      font-size="9"
                    >
                      {marker.label}
                    </text>
                  )}
                </g>
              );
            })}
          </g>

          {/* Today marker */}
          <line
            x1={todayPosition.value}
            y1={headerHeight}
            x2={todayPosition.value}
            y2={chartHeight}
            stroke="#ef4444"
            stroke-width="2"
            stroke-dasharray="4,4"
          />
          <text
            x={todayPosition.value + 4}
            y={headerHeight - 5}
            fill="#ef4444"
            font-size="9"
            font-weight="bold"
          >
            Today
          </text>

          {/* Lane rows */}
          {lanePositions.value.map((lane, index) => {
            const y = headerHeight + index * rowHeight;
            const isHovered = hoveredLaneId.value === lane.id;
            const isCritical = highlightCriticalPath && lane.isCriticalPath;

            return (
              <g key={lane.id}>
                {/* Row background */}
                <rect
                  x={0}
                  y={y}
                  width={chartWidth.value}
                  height={rowHeight}
                  fill={isHovered ? 'rgba(255,255,255,0.03)' : 'transparent'}
                />

                {/* Lane name (fixed position) */}
                <text
                  x={8}
                  y={y + rowHeight / 2 + 4}
                  fill="rgba(255,255,255,0.6)"
                  font-size="10"
                  class="pointer-events-none"
                >
                  {lane.name.slice(0, 20)}{lane.name.length > 20 ? '...' : ''}
                </text>

                {/* Lane bar */}
                <g
                  onMouseEnter$={() => { hoveredLaneId.value = lane.id; }}
                  onMouseLeave$={() => { hoveredLaneId.value = null; }}
                  onClick$={() => onLaneClick$ && onLaneClick$(lane)}
                  class="cursor-pointer"
                >
                  {/* Bar background */}
                  <rect
                    x={lane.x + 150}
                    y={y + 8}
                    width={lane.width}
                    height={rowHeight - 16}
                    rx={4}
                    fill={getStatusBgColor(lane.status)}
                    stroke={isCritical ? '#ef4444' : getStatusColor(lane.status)}
                    stroke-width={isCritical ? 2 : 1}
                  />

                  {/* Progress fill */}
                  <rect
                    x={lane.x + 150}
                    y={y + 8}
                    width={lane.width * (lane.wip_pct / 100)}
                    height={rowHeight - 16}
                    rx={4}
                    fill={getStatusColor(lane.status)}
                    opacity={0.6}
                  />

                  {/* WIP percentage */}
                  <text
                    x={lane.x + 150 + lane.width / 2}
                    y={y + rowHeight / 2 + 4}
                    fill="white"
                    font-size="10"
                    font-weight="bold"
                    text-anchor="middle"
                  >
                    {lane.wip_pct}%
                  </text>

                  {/* Owner */}
                  <text
                    x={lane.x + 150 + lane.width + 8}
                    y={y + rowHeight / 2 + 4}
                    fill="rgba(255,255,255,0.4)"
                    font-size="9"
                  >
                    @{lane.owner}
                  </text>
                </g>

                {/* Milestones */}
                {lane.milestones?.map((milestone, mIdx) => {
                  const mDate = parseDate(milestone.date);
                  const mOffset = daysBetween(chartBounds.value.minDate, mDate);
                  const mX = mOffset * 40 * zoomLevel.value + 150;

                  return (
                    <g key={mIdx}>
                      <polygon
                        points={`${mX},${y + rowHeight / 2 - 6} ${mX + 6},${y + rowHeight / 2} ${mX},${y + rowHeight / 2 + 6} ${mX - 6},${y + rowHeight / 2}`}
                        fill="#a855f7"
                        stroke="#a855f7"
                      />
                      <title>{milestone.label}</title>
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* Dependencies */}
          {showDependencies && lanePositions.value.map(lane => {
            if (!lane.dependencies) return null;

            return lane.dependencies.map(depId => {
              const depLane = lanePositions.value.find(l => l.id === depId);
              if (!depLane) return null;

              const fromIndex = lanePositions.value.findIndex(l => l.id === depId);
              const toIndex = lanePositions.value.findIndex(l => l.id === lane.id);

              const fromY = headerHeight + fromIndex * rowHeight + rowHeight / 2;
              const toY = headerHeight + toIndex * rowHeight + rowHeight / 2;
              const fromX = depLane.x + 150 + depLane.width;
              const toX = lane.x + 150;

              const midX = (fromX + toX) / 2;

              return (
                <g key={`${depId}-${lane.id}`}>
                  <path
                    d={`M ${fromX} ${fromY} C ${midX} ${fromY}, ${midX} ${toY}, ${toX} ${toY}`}
                    fill="none"
                    stroke="rgba(255,255,255,0.2)"
                    stroke-width="1.5"
                    marker-end="url(#arrow)"
                  />
                </g>
              );
            });
          })}

          {/* Arrow marker definition */}
          <defs>
            <marker
              id="arrow"
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="6"
              markerHeight="6"
              orient="auto-start-reverse"
            >
              <path d="M 0 0 L 10 5 L 0 10 z" fill="rgba(255,255,255,0.3)" />
            </marker>
          </defs>
        </svg>
      </div>

      {/* Legend */}
      <div class="p-2 border-t border-border/50 flex items-center justify-between text-[9px] text-muted-foreground">
        <div class="flex items-center gap-4">
          <div class="flex items-center gap-1">
            <div class="w-3 h-3 rounded bg-emerald-500/50" />
            <span>Green</span>
          </div>
          <div class="flex items-center gap-1">
            <div class="w-3 h-3 rounded bg-amber-500/50" />
            <span>Yellow</span>
          </div>
          <div class="flex items-center gap-1">
            <div class="w-3 h-3 rounded bg-red-500/50" />
            <span>Red</span>
          </div>
          <div class="flex items-center gap-1">
            <div class="w-2 h-2 rotate-45 bg-purple-500" />
            <span>Milestone</span>
          </div>
        </div>
        <span>Drag to scroll • Scroll to pan</span>
      </div>
    </div>
  );
});

export default LaneGanttChart;