/**
 * WIPSparkline - Mini chart showing WIP progress over time
 *
 * Phase 2, Iteration 10 of OITERATE lanes-widget-enhancement
 * Pure CSS/SVG sparkline with no external dependencies
 *
 * Features:
 * - SVG-based sparkline graph
 * - Color-coded by final WIP percentage
 * - Hover tooltips with details
 * - Configurable dimensions
 * - Gradient fill under the line
 */

import { component$, useSignal, useComputed$ } from '@builder.io/qwik';

// M3 Components - WIPSparkline
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';

export interface SparklineDataPoint {
  ts: string;
  value: number;
  label?: string;
}

export interface WIPSparklineProps {
  /** Data points for the sparkline */
  data: SparklineDataPoint[];
  /** Width of the sparkline in pixels */
  width?: number;
  /** Height of the sparkline in pixels */
  height?: number;
  /** Show dots on data points */
  showDots?: boolean;
  /** Show hover tooltips */
  showTooltip?: boolean;
  /** Fill under the line */
  showFill?: boolean;
  /** Current value to highlight */
  currentValue?: number;
}

/**
 * Get color based on WIP percentage (0-100)
 */
function getColor(value: number): { stroke: string; fill: string; text: string } {
  if (value >= 90) return { stroke: '#34d399', fill: 'rgba(52, 211, 153, 0.2)', text: 'text-emerald-400' };
  if (value >= 60) return { stroke: '#22d3ee', fill: 'rgba(34, 211, 238, 0.2)', text: 'text-cyan-400' };
  if (value >= 30) return { stroke: '#fbbf24', fill: 'rgba(251, 191, 36, 0.2)', text: 'text-amber-400' };
  return { stroke: '#f87171', fill: 'rgba(248, 113, 113, 0.2)', text: 'text-red-400' };
}

/**
 * Format timestamp for tooltip
 */
function formatDate(ts: string): string {
  try {
    const date = new Date(ts);
    if (isNaN(date.getTime())) return ts.slice(0, 10);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return ts.slice(0, 10);
  }
}

export const WIPSparkline = component$<WIPSparklineProps>(({
  data,
  width = 120,
  height = 32,
  showDots = false,
  showTooltip = true,
  showFill = true,
  currentValue,
}) => {
  const hoveredIndex = useSignal<number | null>(null);

  // Calculate sparkline path
  const sparkline = useComputed$(() => {
    if (data.length === 0) return null;
    if (data.length === 1) {
      // Single point - draw a horizontal line
      const y = height - (data[0].value / 100) * height;
      return {
        path: `M 0 ${y} L ${width} ${y}`,
        fillPath: `M 0 ${y} L ${width} ${y} L ${width} ${height} L 0 ${height} Z`,
        points: [{ x: width / 2, y, value: data[0].value, ts: data[0].ts, label: data[0].label }],
        minValue: data[0].value,
        maxValue: data[0].value,
        avgValue: data[0].value,
        finalValue: data[0].value,
      };
    }

    const padding = 2;
    const effectiveWidth = width - padding * 2;
    const effectiveHeight = height - padding * 2;

    // Calculate points
    const points = data.map((d, i) => {
      const x = padding + (i / (data.length - 1)) * effectiveWidth;
      const y = padding + effectiveHeight - (d.value / 100) * effectiveHeight;
      return { x, y, value: d.value, ts: d.ts, label: d.label };
    });

    // Build SVG path
    const linePath = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`)
      .join(' ');

    // Build fill path (closed polygon)
    const fillPath = linePath +
      ` L ${points[points.length - 1].x.toFixed(2)} ${height - padding}` +
      ` L ${points[0].x.toFixed(2)} ${height - padding} Z`;

    // Calculate stats
    const values = data.map(d => d.value);
    const minValue = Math.min(...values);
    const maxValue = Math.max(...values);
    const avgValue = Math.round(values.reduce((a, b) => a + b, 0) / values.length);
    const finalValue = currentValue ?? data[data.length - 1].value;

    return { path: linePath, fillPath, points, minValue, maxValue, avgValue, finalValue };
  });

  // Get colors based on final value
  const colors = useComputed$(() => {
    const finalValue = sparkline.value?.finalValue ?? 0;
    return getColor(finalValue);
  });

  // Empty state
  if (data.length === 0) {
    return (
      <div
        class="flex items-center justify-center bg-muted/20 rounded text-[9px] text-muted-foreground/50"
        style={{ width: `${width}px`, height: `${height}px` }}
      >
        No data
      </div>
    );
  }

  return (
    <div class="relative inline-block" style={{ width: `${width}px`, height: `${height}px` }}>
      <svg
        width={width}
        height={height}
        class="overflow-visible"
        viewBox={`0 0 ${width} ${height}`}
      >
        {/* Gradient definition */}
        <defs>
          <linearGradient id={`sparkline-gradient-${width}-${height}`} x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor={colors.value.stroke} stopOpacity="0.4" />
            <stop offset="100%" stopColor={colors.value.stroke} stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Fill area */}
        {showFill && sparkline.value?.fillPath && (
          <path
            d={sparkline.value.fillPath}
            fill={`url(#sparkline-gradient-${width}-${height})`}
          />
        )}

        {/* Line */}
        {sparkline.value?.path && (
          <path
            d={sparkline.value.path}
            fill="none"
            stroke={colors.value.stroke}
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        )}

        {/* Data points */}
        {showDots && sparkline.value?.points.map((point, i) => (
          <circle
            key={i}
            cx={point.x}
            cy={point.y}
            r={hoveredIndex.value === i ? 3 : 2}
            fill={hoveredIndex.value === i ? colors.value.stroke : 'transparent'}
            stroke={colors.value.stroke}
            stroke-width="1"
          />
        ))}

        {/* Interactive hover areas */}
        {showTooltip && sparkline.value?.points.map((point, i) => (
          <rect
            key={`hover-${i}`}
            x={point.x - width / data.length / 2}
            y={0}
            width={width / data.length}
            height={height}
            fill="transparent"
            onMouseEnter$={() => { hoveredIndex.value = i; }}
            onMouseLeave$={() => { hoveredIndex.value = null; }}
            style={{ cursor: 'crosshair' }}
          />
        ))}

        {/* Highlight current point */}
        {sparkline.value?.points && sparkline.value.points.length > 0 && (
          <circle
            cx={sparkline.value.points[sparkline.value.points.length - 1].x}
            cy={sparkline.value.points[sparkline.value.points.length - 1].y}
            r="3"
            fill={colors.value.stroke}
          />
        )}
      </svg>

      {/* Tooltip */}
      {showTooltip && hoveredIndex.value !== null && sparkline.value?.points[hoveredIndex.value] && (
        <div
          class="absolute z-10 px-2 py-1 rounded bg-card border border-border shadow-lg text-[9px] whitespace-nowrap pointer-events-none"
          style={{
            left: `${Math.min(sparkline.value.points[hoveredIndex.value].x, width - 60)}px`,
            top: '-24px',
            transform: 'translateX(-50%)',
          }}
        >
          <span class={colors.value.text}>
            {sparkline.value.points[hoveredIndex.value].value}%
          </span>
          <span class="text-muted-foreground mx-1">·</span>
          <span class="text-muted-foreground">
            {formatDate(sparkline.value.points[hoveredIndex.value].ts)}
          </span>
          {sparkline.value.points[hoveredIndex.value].label && (
            <>
              <span class="text-muted-foreground mx-1">·</span>
              <span class="text-foreground/70">
                {sparkline.value.points[hoveredIndex.value].label}
              </span>
            </>
          )}
        </div>
      )}

      {/* Stats overlay (on hover) */}
      {hoveredIndex.value === null && sparkline.value && (
        <div
          class="absolute bottom-0 right-0 text-[8px] px-1 rounded bg-card/80"
          style={{ lineHeight: 1 }}
        >
          <span class={colors.value.text}>{sparkline.value.finalValue}%</span>
        </div>
      )}
    </div>
  );
});

/**
 * Compact sparkline specifically for lane history
 */
export interface LaneHistorySparklineProps {
  history: Array<{ ts: string; wip_pct: number; note: string }>;
  currentWip: number;
  width?: number;
  height?: number;
}

export const LaneHistorySparkline = component$<LaneHistorySparklineProps>(({
  history,
  currentWip,
  width = 80,
  height = 24,
}) => {
  // Convert history to sparkline data points
  const data = useComputed$(() => {
    if (!history || history.length === 0) return [];
    return history.map(h => ({
      ts: h.ts,
      value: h.wip_pct,
      label: h.note,
    }));
  });

  return (
    <WIPSparkline
      data={data.value}
      width={width}
      height={height}
      showDots={false}
      showTooltip={true}
      showFill={true}
      currentValue={currentWip}
    />
  );
});

export default WIPSparkline;
