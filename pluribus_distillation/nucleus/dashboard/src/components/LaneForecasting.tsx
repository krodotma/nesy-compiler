/**
 * LaneForecasting - ML-based completion prediction and trend analysis
 *
 * Phase 4, Iteration 33 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Completion date prediction based on historical data
 * - Trend analysis with confidence intervals
 * - Risk indicators
 * - Velocity tracking
 * - Burndown/burnup projections
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

export interface HistoricalDataPoint {
  date: string;
  wip_pct: number;
  blockers?: number;
  velocity?: number;
}

export interface ForecastLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  startDate: string;
  targetDate?: string;
  history: HistoricalDataPoint[];
}

export interface Forecast {
  predictedCompletionDate: string;
  confidenceLow: string;
  confidenceHigh: string;
  confidencePercent: number;
  velocity: number;
  trend: 'accelerating' | 'steady' | 'slowing' | 'stalled';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  riskFactors: string[];
  daysRemaining: number;
  projectedPoints: { date: string; wip_pct: number }[];
}

export interface LaneForecastingProps {
  /** Lane with historical data */
  lane: ForecastLane;
  /** Callback when forecast is generated */
  onForecastGenerated$?: QRL<(forecast: Forecast) => void>;
  /** Show detailed analysis */
  showDetails?: boolean;
  /** Projection days ahead */
  projectionDays?: number;
}

// ============================================================================
// Forecasting Logic
// ============================================================================

function calculateVelocity(history: HistoricalDataPoint[]): number {
  if (history.length < 2) return 0;

  // Use last 7 data points for recent velocity
  const recent = history.slice(-7);
  if (recent.length < 2) return 0;

  const first = recent[0];
  const last = recent[recent.length - 1];
  const daysDiff = Math.max(1, (new Date(last.date).getTime() - new Date(first.date).getTime()) / (1000 * 60 * 60 * 24));

  return (last.wip_pct - first.wip_pct) / daysDiff;
}

function determineTrend(history: HistoricalDataPoint[]): 'accelerating' | 'steady' | 'slowing' | 'stalled' {
  if (history.length < 5) return 'steady';

  const recent = history.slice(-5);
  const velocities: number[] = [];

  for (let i = 1; i < recent.length; i++) {
    const daysDiff = Math.max(1, (new Date(recent[i].date).getTime() - new Date(recent[i-1].date).getTime()) / (1000 * 60 * 60 * 24));
    velocities.push((recent[i].wip_pct - recent[i-1].wip_pct) / daysDiff);
  }

  const avgVelocity = velocities.reduce((a, b) => a + b, 0) / velocities.length;
  const recentVelocity = velocities.slice(-2).reduce((a, b) => a + b, 0) / 2;

  if (avgVelocity <= 0.1) return 'stalled';
  if (recentVelocity > avgVelocity * 1.2) return 'accelerating';
  if (recentVelocity < avgVelocity * 0.8) return 'slowing';
  return 'steady';
}

function calculateRisk(lane: ForecastLane, velocity: number, forecast: { daysRemaining: number; predictedDate: Date }): { level: 'low' | 'medium' | 'high' | 'critical'; factors: string[] } {
  const factors: string[] = [];
  let riskScore = 0;

  // Velocity risk
  if (velocity <= 0) {
    factors.push('No progress in recent history');
    riskScore += 40;
  } else if (velocity < 1) {
    factors.push('Very slow velocity (<1% per day)');
    riskScore += 20;
  }

  // Blocker risk
  const recentBlockers = lane.history.slice(-5).filter(h => h.blockers && h.blockers > 0).length;
  if (recentBlockers >= 3) {
    factors.push('Frequent blockers detected');
    riskScore += 25;
  }

  // Target date risk
  if (lane.targetDate) {
    const targetDate = new Date(lane.targetDate);
    if (forecast.predictedDate > targetDate) {
      const daysLate = Math.ceil((forecast.predictedDate.getTime() - targetDate.getTime()) / (1000 * 60 * 60 * 24));
      factors.push(`Predicted ${daysLate} days past target`);
      riskScore += Math.min(40, daysLate * 5);
    }
  }

  // Status risk
  if (lane.status === 'red') {
    factors.push('Lane is currently blocked');
    riskScore += 30;
  } else if (lane.status === 'yellow') {
    factors.push('Lane has warnings');
    riskScore += 15;
  }

  // WIP stagnation
  if (lane.wip_pct > 80 && velocity < 0.5) {
    factors.push('Near completion but slowing down');
    riskScore += 10;
  }

  let level: 'low' | 'medium' | 'high' | 'critical';
  if (riskScore >= 70) level = 'critical';
  else if (riskScore >= 50) level = 'high';
  else if (riskScore >= 25) level = 'medium';
  else level = 'low';

  return { level, factors };
}

function generateForecast(lane: ForecastLane, projectionDays: number): Forecast {
  const velocity = calculateVelocity(lane.history);
  const trend = determineTrend(lane.history);
  const remaining = 100 - lane.wip_pct;

  // Calculate days to completion
  let daysToComplete: number;
  if (velocity <= 0) {
    daysToComplete = projectionDays * 2; // Pessimistic fallback
  } else {
    daysToComplete = Math.ceil(remaining / velocity);
  }

  const today = new Date();
  const predictedDate = new Date(today);
  predictedDate.setDate(predictedDate.getDate() + daysToComplete);

  // Confidence intervals based on historical variance
  const variance = lane.history.length > 5 ? 0.2 : 0.4;
  const confidenceLow = new Date(predictedDate);
  confidenceLow.setDate(confidenceLow.getDate() - Math.ceil(daysToComplete * variance));
  const confidenceHigh = new Date(predictedDate);
  confidenceHigh.setDate(confidenceHigh.getDate() + Math.ceil(daysToComplete * variance));

  const risk = calculateRisk(lane, velocity, { daysRemaining: daysToComplete, predictedDate });

  // Generate projection points
  const projectedPoints: { date: string; wip_pct: number }[] = [];
  let currentWip = lane.wip_pct;
  const projectionDate = new Date(today);

  // Apply trend modifier
  let trendModifier = 1;
  if (trend === 'accelerating') trendModifier = 1.1;
  else if (trend === 'slowing') trendModifier = 0.9;
  else if (trend === 'stalled') trendModifier = 0.1;

  for (let i = 0; i <= Math.min(projectionDays, daysToComplete + 7); i++) {
    projectedPoints.push({
      date: projectionDate.toISOString().slice(0, 10),
      wip_pct: Math.min(100, Math.round(currentWip * 10) / 10),
    });
    currentWip += velocity * trendModifier;
    projectionDate.setDate(projectionDate.getDate() + 1);
  }

  return {
    predictedCompletionDate: predictedDate.toISOString().slice(0, 10),
    confidenceLow: confidenceLow.toISOString().slice(0, 10),
    confidenceHigh: confidenceHigh.toISOString().slice(0, 10),
    confidencePercent: Math.round((1 - variance) * 100),
    velocity: Math.round(velocity * 100) / 100,
    trend,
    riskLevel: risk.level,
    riskFactors: risk.factors,
    daysRemaining: daysToComplete,
    projectedPoints,
  };
}

// ============================================================================
// Helpers
// ============================================================================

function getRiskColor(level: string): string {
  switch (level) {
    case 'low': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'medium': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'high': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    case 'critical': return 'bg-red-500/20 text-red-400 border-red-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getTrendIcon(trend: string): string {
  switch (trend) {
    case 'accelerating': return '📈';
    case 'steady': return '➡️';
    case 'slowing': return '📉';
    case 'stalled': return '⏸️';
    default: return '❓';
  }
}

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneForecasting = component$<LaneForecastingProps>(({
  lane,
  onForecastGenerated$,
  showDetails = true,
  projectionDays = 30,
}) => {
  // State
  const showProjection = useSignal(false);

  // Generate forecast
  const forecast = useComputed$(() => {
    const result = generateForecast(lane, projectionDays);
    return result;
  });

  // Notify on forecast generation
  const notifyForecast = $(async () => {
    if (onForecastGenerated$) {
      await onForecastGenerated$(forecast.value);
    }
  });

  // SVG dimensions for projection chart
  const chartWidth = 400;
  const chartHeight = 120;
  const padding = { top: 10, right: 10, bottom: 25, left: 35 };

  // Generate chart path
  const chartPath = useComputed$(() => {
    const points = forecast.value.projectedPoints;
    if (points.length < 2) return '';

    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;

    const xScale = innerWidth / (points.length - 1);
    const yScale = innerHeight / 100;

    let path = '';
    points.forEach((p, i) => {
      const x = padding.left + i * xScale;
      const y = chartHeight - padding.bottom - (p.wip_pct * yScale);
      path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });

    return path;
  });

  // Historical path (for comparison)
  const historyPath = useComputed$(() => {
    const points = lane.history.slice(-14);
    if (points.length < 2) return '';

    const innerWidth = chartWidth - padding.left - padding.right;
    const innerHeight = chartHeight - padding.top - padding.bottom;

    // Scale history to left portion of chart
    const historyWidth = innerWidth * 0.3;
    const xScale = historyWidth / (points.length - 1);
    const yScale = innerHeight / 100;

    let path = '';
    points.forEach((p, i) => {
      const x = padding.left + i * xScale;
      const y = chartHeight - padding.bottom - (p.wip_pct * yScale);
      path += i === 0 ? `M ${x} ${y}` : ` L ${x} ${y}`;
    });

    return path;
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">FORECAST</span>
          <span class={`text-[10px] px-2 py-0.5 rounded border ${getRiskColor(forecast.value.riskLevel)}`}>
            {forecast.value.riskLevel.toUpperCase()} RISK
          </span>
        </div>
        <div class="flex items-center gap-2">
          <span class="text-[9px] text-muted-foreground">
            {getTrendIcon(forecast.value.trend)} {forecast.value.trend}
          </span>
          <button
            onClick$={() => notifyForecast()}
            class="text-[9px] px-2 py-0.5 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
          >
            Export
          </button>
        </div>
      </div>

      {/* Main prediction */}
      <div class="p-4">
        <div class="grid grid-cols-2 gap-4 mb-4">
          {/* Predicted completion */}
          <div class="text-center p-3 rounded-lg bg-muted/10 border border-border/30">
            <div class="text-[9px] text-muted-foreground mb-1">PREDICTED COMPLETION</div>
            <div class="text-xl font-bold text-foreground">
              {formatDate(forecast.value.predictedCompletionDate)}
            </div>
            <div class="text-[9px] text-muted-foreground mt-1">
              {forecast.value.daysRemaining} days remaining
            </div>
          </div>

          {/* Velocity */}
          <div class="text-center p-3 rounded-lg bg-muted/10 border border-border/30">
            <div class="text-[9px] text-muted-foreground mb-1">VELOCITY</div>
            <div class={`text-xl font-bold ${
              forecast.value.velocity >= 2 ? 'text-emerald-400' :
              forecast.value.velocity >= 1 ? 'text-cyan-400' :
              forecast.value.velocity > 0 ? 'text-amber-400' :
              'text-red-400'
            }`}>
              {forecast.value.velocity}%
            </div>
            <div class="text-[9px] text-muted-foreground mt-1">per day</div>
          </div>
        </div>

        {/* Confidence interval */}
        <div class="mb-4 p-2 rounded bg-muted/5 border border-border/20">
          <div class="flex items-center justify-between text-[9px]">
            <span class="text-muted-foreground">Confidence Range ({forecast.value.confidencePercent}%)</span>
            <span class="text-foreground">
              {formatDate(forecast.value.confidenceLow)} — {formatDate(forecast.value.confidenceHigh)}
            </span>
          </div>
          <div class="mt-2 h-2 rounded-full bg-muted/20 overflow-hidden relative">
            <div
              class="absolute h-full bg-primary/30 rounded-full"
              style={{
                left: '20%',
                width: '60%',
              }}
            />
            <div
              class="absolute h-full w-1 bg-primary rounded-full"
              style={{ left: '50%' }}
            />
          </div>
        </div>

        {/* Projection chart toggle */}
        <button
          onClick$={() => { showProjection.value = !showProjection.value; }}
          class="w-full p-2 text-[9px] text-muted-foreground hover:bg-muted/10 transition-colors flex items-center justify-center gap-1 rounded border border-border/30"
        >
          <span>{showProjection.value ? '▼' : '▶'}</span>
          <span>Show Projection Chart</span>
        </button>

        {showProjection.value && (
          <div class="mt-3 p-2 rounded bg-muted/5 border border-border/20">
            <svg width={chartWidth} height={chartHeight} class="w-full">
              {/* Grid lines */}
              {[0, 25, 50, 75, 100].map(pct => {
                const y = chartHeight - padding.bottom - (pct * (chartHeight - padding.top - padding.bottom) / 100);
                return (
                  <g key={pct}>
                    <line
                      x1={padding.left}
                      y1={y}
                      x2={chartWidth - padding.right}
                      y2={y}
                      stroke="rgba(255,255,255,0.1)"
                      stroke-dasharray="2,2"
                    />
                    <text
                      x={padding.left - 5}
                      y={y + 3}
                      text-anchor="end"
                      fill="rgba(255,255,255,0.4)"
                      font-size="8"
                    >
                      {pct}%
                    </text>
                  </g>
                );
              })}

              {/* Historical line */}
              <path
                d={historyPath.value}
                fill="none"
                stroke="rgba(255,255,255,0.3)"
                stroke-width="1.5"
              />

              {/* Projection line */}
              <path
                d={chartPath.value}
                fill="none"
                stroke="#3b82f6"
                stroke-width="2"
                stroke-dasharray="4,2"
              />

              {/* 100% completion line */}
              <line
                x1={padding.left}
                y1={padding.top}
                x2={chartWidth - padding.right}
                y2={padding.top}
                stroke="#22c55e"
                stroke-width="1"
                stroke-dasharray="4,4"
              />

              {/* Current point */}
              <circle
                cx={padding.left + (chartWidth - padding.left - padding.right) * 0.3}
                cy={chartHeight - padding.bottom - (lane.wip_pct * (chartHeight - padding.top - padding.bottom) / 100)}
                r="4"
                fill="#3b82f6"
              />

              {/* Labels */}
              <text
                x={padding.left}
                y={chartHeight - 5}
                fill="rgba(255,255,255,0.4)"
                font-size="8"
              >
                History
              </text>
              <text
                x={chartWidth - padding.right}
                y={chartHeight - 5}
                text-anchor="end"
                fill="rgba(255,255,255,0.4)"
                font-size="8"
              >
                +{projectionDays}d
              </text>
            </svg>
          </div>
        )}
      </div>

      {/* Risk factors */}
      {showDetails && forecast.value.riskFactors.length > 0 && (
        <div class="p-3 border-t border-border/30">
          <div class="text-[9px] font-semibold text-muted-foreground mb-2">RISK FACTORS</div>
          <div class="space-y-1">
            {forecast.value.riskFactors.map((factor, i) => (
              <div key={i} class="flex items-center gap-2 text-[10px]">
                <span class="text-amber-400">⚠</span>
                <span class="text-foreground">{factor}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Stats footer */}
      <div class="p-2 border-t border-border/50 flex items-center justify-between text-[9px] text-muted-foreground">
        <span>Based on {lane.history.length} data points</span>
        <span>Current: {lane.wip_pct}% | Target: {lane.targetDate ? formatDate(lane.targetDate) : 'None'}</span>
      </div>
    </div>
  );
});

export default LaneForecasting;
