/**
 * PortalMetricsPanel.tsx - Portal Ingest Metrics Dashboard
 *
 * Displays key Portal metrics:
 * 1. Total ingests (AM vs SM breakdown)
 * 2. Holon conversion rate (potential -> actualized)
 * 3. HITL interaction frequency
 * 4. Average processing time
 * 5. Quality gate pass/fail rates
 *
 * Listens to bus events for real-time updates.
 */

import { component$, useSignal, useComputed$, useVisibleTask$ } from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';

// ─────────────────────────────────────────────────────────────────────────────
// Types
// ─────────────────────────────────────────────────────────────────────────────

export interface PortalMetricsPanelProps {
  events: BusEvent[];
  refreshIntervalMs?: number;
}

interface IngestStats {
  total: number;
  am: number;  // Actualized Mode
  sm: number;  // Shadow Mode
  rejected: number;
  pruned: number;
}

interface HolonStats {
  potential: number;
  actualizing: number;
  actualized: number;
  decaying: number;
  conversionRate: number;
}

interface HITLStats {
  totalInteractions: number;
  approvals: number;
  rejections: number;
  pending: number;
  avgResponseTimeMs: number;
}

interface ProcessingStats {
  avgTimeMs: number;
  minTimeMs: number;
  maxTimeMs: number;
  p95TimeMs: number;
  throughputPerMin: number;
}

interface QualityGateStats {
  total: number;
  passed: number;
  failed: number;
  warnings: number;
  passRate: number;
  gateBreakdown: {
    P: { passed: number; failed: number };  // Provenance
    E: { passed: number; failed: number };  // Effects
    L: { passed: number; failed: number };  // Liveness
    R: { passed: number; failed: number };  // Recovery
    Q: { passed: number; failed: number };  // Quality
    omega: { passed: number; failed: number }; // Omega alignment
  };
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

function wipMeter(pct: number, width: number = 10): string {
  const clamped = Math.max(0, Math.min(100, pct));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function rateColor(rate: number): string {
  if (rate >= 90) return 'text-emerald-400';
  if (rate >= 70) return 'text-cyan-400';
  if (rate >= 50) return 'text-amber-400';
  return 'text-red-400';
}

// ─────────────────────────────────────────────────────────────────────────────
// Event Classification
// ─────────────────────────────────────────────────────────────────────────────

function classifyPortalEvents(events: BusEvent[]): {
  ingestStats: IngestStats;
  holonStats: HolonStats;
  hitlStats: HITLStats;
  processingStats: ProcessingStats;
  qualityGateStats: QualityGateStats;
} {
  const ingestStats: IngestStats = { total: 0, am: 0, sm: 0, rejected: 0, pruned: 0 };
  const holonStats: HolonStats = { potential: 0, actualizing: 0, actualized: 0, decaying: 0, conversionRate: 0 };
  const hitlStats: HITLStats = { totalInteractions: 0, approvals: 0, rejections: 0, pending: 0, avgResponseTimeMs: 0 };
  const processingTimes: number[] = [];
  const qualityGateStats: QualityGateStats = {
    total: 0,
    passed: 0,
    failed: 0,
    warnings: 0,
    passRate: 0,
    gateBreakdown: {
      P: { passed: 0, failed: 0 },
      E: { passed: 0, failed: 0 },
      L: { passed: 0, failed: 0 },
      R: { passed: 0, failed: 0 },
      Q: { passed: 0, failed: 0 },
      omega: { passed: 0, failed: 0 },
    },
  };

  const hitlResponseTimes: number[] = [];
  const recentEvents = events.slice(-1000); // Look at last 1000 events

  for (const event of recentEvents) {
    const data = event.data as Record<string, unknown> | undefined;
    const topic = event.topic;

    // Portal ingest events
    if (topic.startsWith('portal.') || topic.startsWith('entelexis.') || topic.startsWith('hysteresis.')) {
      ingestStats.total++;

      if (topic === 'entelexis.actualize' || data?.mode === 'AM') {
        ingestStats.am++;
      } else if (topic === 'hysteresis.shadow' || data?.mode === 'SM') {
        ingestStats.sm++;
      }

      if (data?.status === 'rejected') {
        ingestStats.rejected++;
      } else if (data?.status === 'pruned') {
        ingestStats.pruned++;
      }

      // Processing time tracking
      if (typeof data?.processing_time_ms === 'number') {
        processingTimes.push(data.processing_time_ms);
      } else if (typeof data?.duration_ms === 'number') {
        processingTimes.push(data.duration_ms);
      }
    }

    // Holon lifecycle events
    if (topic.startsWith('holon.')) {
      const phase = event.entelexis?.phase || (data?.phase as string);
      if (phase === 'potential' || topic === 'holon.potential') {
        holonStats.potential++;
      } else if (phase === 'actualizing' || topic === 'holon.actualizing') {
        holonStats.actualizing++;
      } else if (phase === 'actualized' || topic === 'holon.created' || topic === 'holon.actualized') {
        holonStats.actualized++;
      } else if (phase === 'decaying' || topic === 'holon.decaying') {
        holonStats.decaying++;
      }
    }

    // HITL (Human-In-The-Loop) events
    if (topic.startsWith('hitl.') || topic.startsWith('pbpair.') || topic === 'user.approve' || topic === 'user.reject') {
      hitlStats.totalInteractions++;

      if (topic.includes('approve') || data?.action === 'approve') {
        hitlStats.approvals++;
      } else if (topic.includes('reject') || data?.action === 'reject') {
        hitlStats.rejections++;
      } else if (topic.includes('pending') || data?.status === 'pending') {
        hitlStats.pending++;
      }

      if (typeof data?.response_time_ms === 'number') {
        hitlResponseTimes.push(data.response_time_ms);
      }
    }

    // Quality gate / Sextet validation events
    if (topic.startsWith('sextet.') || topic.startsWith('gate.') || topic.startsWith('verify.')) {
      qualityGateStats.total++;

      const verdict = data?.verdict || data?.result || event.level;
      if (verdict === 'passed' || verdict === 'pass' || verdict === 'info') {
        qualityGateStats.passed++;
      } else if (verdict === 'failed' || verdict === 'fail' || verdict === 'error') {
        qualityGateStats.failed++;
      } else if (verdict === 'warn' || verdict === 'warning') {
        qualityGateStats.warnings++;
      }

      // Gate-specific breakdown
      const gate = (data?.gate as string) || '';
      if (gate === 'P' || topic.includes('.provenance')) {
        if (verdict === 'passed' || verdict === 'pass') qualityGateStats.gateBreakdown.P.passed++;
        else qualityGateStats.gateBreakdown.P.failed++;
      } else if (gate === 'E' || topic.includes('.effects')) {
        if (verdict === 'passed' || verdict === 'pass') qualityGateStats.gateBreakdown.E.passed++;
        else qualityGateStats.gateBreakdown.E.failed++;
      } else if (gate === 'L' || topic.includes('.liveness')) {
        if (verdict === 'passed' || verdict === 'pass') qualityGateStats.gateBreakdown.L.passed++;
        else qualityGateStats.gateBreakdown.L.failed++;
      } else if (gate === 'R' || topic.includes('.recovery')) {
        if (verdict === 'passed' || verdict === 'pass') qualityGateStats.gateBreakdown.R.passed++;
        else qualityGateStats.gateBreakdown.R.failed++;
      } else if (gate === 'Q' || topic.includes('.quality')) {
        if (verdict === 'passed' || verdict === 'pass') qualityGateStats.gateBreakdown.Q.passed++;
        else qualityGateStats.gateBreakdown.Q.failed++;
      } else if (gate === 'omega' || gate === 'O' || topic.includes('.omega')) {
        if (verdict === 'passed' || verdict === 'pass') qualityGateStats.gateBreakdown.omega.passed++;
        else qualityGateStats.gateBreakdown.omega.failed++;
      }
    }
  }

  // Calculate derived metrics
  holonStats.conversionRate = holonStats.potential > 0
    ? Math.round((holonStats.actualized / holonStats.potential) * 100)
    : 0;

  qualityGateStats.passRate = qualityGateStats.total > 0
    ? Math.round((qualityGateStats.passed / qualityGateStats.total) * 100)
    : 0;

  hitlStats.avgResponseTimeMs = hitlResponseTimes.length > 0
    ? Math.round(hitlResponseTimes.reduce((a, b) => a + b, 0) / hitlResponseTimes.length)
    : 0;

  // Processing time stats
  const sortedTimes = [...processingTimes].sort((a, b) => a - b);
  const processingStats: ProcessingStats = {
    avgTimeMs: processingTimes.length > 0
      ? Math.round(processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length)
      : 0,
    minTimeMs: sortedTimes[0] || 0,
    maxTimeMs: sortedTimes[sortedTimes.length - 1] || 0,
    p95TimeMs: sortedTimes[Math.floor(sortedTimes.length * 0.95)] || 0,
    throughputPerMin: 0, // Calculated separately
  };

  // Calculate throughput from recent events
  const now = Date.now();
  const oneMinuteAgo = now - 60000;
  const recentIngests = recentEvents.filter(
    e => e.ts * 1000 > oneMinuteAgo &&
    (e.topic.startsWith('portal.') || e.topic.startsWith('entelexis.') || e.topic.startsWith('hysteresis.'))
  );
  processingStats.throughputPerMin = recentIngests.length;

  return { ingestStats, holonStats, hitlStats, processingStats, qualityGateStats };
}

// ─────────────────────────────────────────────────────────────────────────────
// Component
// ─────────────────────────────────────────────────────────────────────────────

export const PortalMetricsPanel = component$<PortalMetricsPanelProps>(({
  events,
  refreshIntervalMs = 5000,
}) => {
  const activeTab = useSignal<'overview' | 'gates' | 'holon' | 'hitl'>('overview');
  const lastUpdate = useSignal<string>('');

  // Track update times
  useVisibleTask$(({ track }) => {
    track(() => events);
    lastUpdate.value = new Date().toISOString().slice(11, 19);
  });

  // Compute metrics from events
  const metrics = useComputed$(() => classifyPortalEvents(events));

  const { ingestStats, holonStats, hitlStats, processingStats, qualityGateStats } = metrics.value;

  // AM/SM ratio for visualization
  const amRatio = ingestStats.total > 0
    ? Math.round((ingestStats.am / ingestStats.total) * 100)
    : 0;

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="p-3 border-b border-border flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span class="text-lg">Portal Metrics</span>
          <span class="text-[9px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
            {ingestStats.total} ingests
          </span>
          <span class="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
            {processingStats.throughputPerMin}/min
          </span>
        </div>
        <span class="text-[10px] text-muted-foreground font-mono">{lastUpdate.value}</span>
      </div>

      {/* Tab Navigation */}
      <div class="px-3 py-2 border-b border-border/50 flex gap-1">
        {(['overview', 'gates', 'holon', 'hitl'] as const).map((tab) => (
          <button
            key={tab}
            onClick$={() => activeTab.value = tab}
            class={`text-[10px] px-2 py-1 rounded transition-colors capitalize ${
              activeTab.value === tab
                ? 'bg-primary/20 text-primary border border-primary/30'
                : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Content */}
      <div class="p-3">
        {/* Overview Tab */}
        {activeTab.value === 'overview' && (
          <div class="space-y-4">
            {/* Ingest Breakdown */}
            <div class="p-3 rounded-lg border border-border/50 bg-muted/10">
              <div class="text-[11px] text-muted-foreground uppercase tracking-wide mb-2">
                Ingest Mode Breakdown
              </div>
              <div class="flex items-center gap-3 mb-2">
                <div class="flex-1">
                  <div class="flex justify-between text-[10px] mb-1">
                    <span class="text-emerald-400">AM (Actualized)</span>
                    <span class="text-emerald-400">{ingestStats.am}</span>
                  </div>
                  <div class="h-2 bg-muted/30 rounded overflow-hidden">
                    <div
                      class="h-full bg-emerald-500 transition-all duration-300"
                      style={{ width: `${amRatio}%` }}
                    />
                  </div>
                </div>
                <div class="text-[10px] text-muted-foreground">{amRatio}%</div>
              </div>
              <div class="flex items-center gap-3">
                <div class="flex-1">
                  <div class="flex justify-between text-[10px] mb-1">
                    <span class="text-purple-400">SM (Shadow)</span>
                    <span class="text-purple-400">{ingestStats.sm}</span>
                  </div>
                  <div class="h-2 bg-muted/30 rounded overflow-hidden">
                    <div
                      class="h-full bg-purple-500 transition-all duration-300"
                      style={{ width: `${100 - amRatio}%` }}
                    />
                  </div>
                </div>
                <div class="text-[10px] text-muted-foreground">{100 - amRatio}%</div>
              </div>
              <div class="flex gap-2 mt-2 text-[9px]">
                <span class="px-1.5 py-0.5 rounded bg-red-500/10 text-red-400">
                  {ingestStats.rejected} rejected
                </span>
                <span class="px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400">
                  {ingestStats.pruned} pruned
                </span>
              </div>
            </div>

            {/* Key Metrics Grid */}
            <div class="grid grid-cols-2 gap-3">
              {/* Holon Conversion */}
              <div class="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5">
                <div class="text-[10px] text-muted-foreground uppercase mb-1">Holon Conversion</div>
                <div class={`text-xl font-bold ${rateColor(holonStats.conversionRate)}`}>
                  {holonStats.conversionRate}%
                </div>
                <div class="text-[9px] text-muted-foreground">
                  {holonStats.actualized} / {holonStats.potential} potential
                </div>
              </div>

              {/* Quality Gate Pass Rate */}
              <div class="p-3 rounded-lg border border-cyan-500/30 bg-cyan-500/5">
                <div class="text-[10px] text-muted-foreground uppercase mb-1">Gate Pass Rate</div>
                <div class={`text-xl font-bold ${rateColor(qualityGateStats.passRate)}`}>
                  {qualityGateStats.passRate}%
                </div>
                <div class="text-[9px] text-muted-foreground">
                  {qualityGateStats.passed} / {qualityGateStats.total} checks
                </div>
              </div>

              {/* Avg Processing Time */}
              <div class="p-3 rounded-lg border border-amber-500/30 bg-amber-500/5">
                <div class="text-[10px] text-muted-foreground uppercase mb-1">Avg Processing</div>
                <div class="text-xl font-bold text-amber-400">
                  {formatDuration(processingStats.avgTimeMs)}
                </div>
                <div class="text-[9px] text-muted-foreground">
                  p95: {formatDuration(processingStats.p95TimeMs)}
                </div>
              </div>

              {/* HITL Frequency */}
              <div class="p-3 rounded-lg border border-purple-500/30 bg-purple-500/5">
                <div class="text-[10px] text-muted-foreground uppercase mb-1">HITL Interactions</div>
                <div class="text-xl font-bold text-purple-400">
                  {hitlStats.totalInteractions}
                </div>
                <div class="text-[9px] text-muted-foreground">
                  {hitlStats.pending} pending
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Quality Gates Tab */}
        {activeTab.value === 'gates' && (
          <div class="space-y-3">
            <div class="p-3 rounded-lg border border-border/50 bg-muted/10">
              <div class="flex items-center justify-between mb-3">
                <span class="text-[11px] text-muted-foreground uppercase tracking-wide">Sextet Gate Status</span>
                <span class={`text-sm font-bold ${rateColor(qualityGateStats.passRate)}`}>
                  {qualityGateStats.passRate}% overall
                </span>
              </div>

              <div class="space-y-2">
                {([
                  { key: 'P', name: 'Provenance', desc: 'Lineage validation' },
                  { key: 'E', name: 'Effects', desc: 'Side effect bounds' },
                  { key: 'L', name: 'Liveness', desc: 'Timing feasibility' },
                  { key: 'R', name: 'Recovery', desc: 'Rollback capability' },
                  { key: 'Q', name: 'Quality', desc: 'Test coverage' },
                  { key: 'omega', name: 'Omega', desc: 'ASL alignment' },
                ] as const).map((gate) => {
                  const stats = qualityGateStats.gateBreakdown[gate.key];
                  const total = stats.passed + stats.failed;
                  const rate = total > 0 ? Math.round((stats.passed / total) * 100) : 100;
                  return (
                    <div key={gate.key} class="flex items-center gap-2">
                      <div class="w-16 text-[10px]">
                        <span class="font-bold text-primary">{gate.key === 'omega' ? '\u03A9' : gate.key}</span>
                        <span class="text-muted-foreground ml-1">{gate.name}</span>
                      </div>
                      <div class="flex-1">
                        <code class={`text-[10px] font-mono ${rateColor(rate)}`}>
                          {wipMeter(rate, 12)}
                        </code>
                      </div>
                      <div class={`text-[10px] font-mono w-10 text-right ${rateColor(rate)}`}>
                        {rate}%
                      </div>
                      <div class="text-[9px] text-muted-foreground w-12 text-right">
                        {stats.passed}/{total}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Gate Summary */}
            <div class="flex gap-2 text-[9px]">
              <span class="px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                {qualityGateStats.passed} passed
              </span>
              <span class="px-2 py-1 rounded bg-red-500/10 text-red-400 border border-red-500/20">
                {qualityGateStats.failed} failed
              </span>
              <span class="px-2 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">
                {qualityGateStats.warnings} warnings
              </span>
            </div>
          </div>
        )}

        {/* Holon Lifecycle Tab */}
        {activeTab.value === 'holon' && (
          <div class="space-y-3">
            <div class="p-3 rounded-lg border border-border/50 bg-muted/10">
              <div class="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">
                Holon State Distribution
              </div>

              <div class="space-y-2">
                {([
                  { phase: 'potential', icon: '\u25CB', color: 'text-blue-400 bg-blue-500' },
                  { phase: 'actualizing', icon: '\u25D4', color: 'text-amber-400 bg-amber-500' },
                  { phase: 'actualized', icon: '\u25CF', color: 'text-emerald-400 bg-emerald-500' },
                  { phase: 'decaying', icon: '\u25CC', color: 'text-red-400 bg-red-500' },
                ] as const).map((state) => {
                  const count = holonStats[state.phase as keyof HolonStats] as number;
                  const total = holonStats.potential + holonStats.actualizing + holonStats.actualized + holonStats.decaying;
                  const pct = total > 0 ? Math.round((count / total) * 100) : 0;
                  return (
                    <div key={state.phase} class="flex items-center gap-2">
                      <span class={`text-sm ${state.color.split(' ')[0]}`}>{state.icon}</span>
                      <span class="w-20 text-[10px] capitalize text-muted-foreground">{state.phase}</span>
                      <div class="flex-1 h-2 bg-muted/30 rounded overflow-hidden">
                        <div
                          class={`h-full transition-all duration-300 ${state.color.split(' ')[1]}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                      <span class="text-[10px] font-mono w-8 text-right text-muted-foreground">{count}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Conversion Funnel */}
            <div class="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5">
              <div class="text-[11px] text-muted-foreground uppercase tracking-wide mb-2">
                Potential to Actualized Conversion
              </div>
              <div class="flex items-center gap-3">
                <div class="text-2xl font-bold text-emerald-400">{holonStats.conversionRate}%</div>
                <div class="flex-1 text-[10px] text-muted-foreground">
                  {holonStats.actualized} of {holonStats.potential} holons fully actualized
                </div>
              </div>
            </div>
          </div>
        )}

        {/* HITL Tab */}
        {activeTab.value === 'hitl' && (
          <div class="space-y-3">
            <div class="p-3 rounded-lg border border-border/50 bg-muted/10">
              <div class="text-[11px] text-muted-foreground uppercase tracking-wide mb-3">
                Human-In-The-Loop Metrics
              </div>

              <div class="grid grid-cols-2 gap-3">
                <div class="p-2 rounded bg-muted/20">
                  <div class="text-[9px] text-muted-foreground uppercase">Total Interactions</div>
                  <div class="text-lg font-bold text-purple-400">{hitlStats.totalInteractions}</div>
                </div>
                <div class="p-2 rounded bg-muted/20">
                  <div class="text-[9px] text-muted-foreground uppercase">Avg Response Time</div>
                  <div class="text-lg font-bold text-cyan-400">
                    {formatDuration(hitlStats.avgResponseTimeMs)}
                  </div>
                </div>
              </div>
            </div>

            {/* Approval/Rejection Breakdown */}
            <div class="p-3 rounded-lg border border-border/50 bg-muted/10">
              <div class="text-[11px] text-muted-foreground uppercase tracking-wide mb-2">
                Decision Breakdown
              </div>
              <div class="flex gap-2">
                <div class="flex-1 p-2 rounded bg-emerald-500/10 border border-emerald-500/20 text-center">
                  <div class="text-lg font-bold text-emerald-400">{hitlStats.approvals}</div>
                  <div class="text-[9px] text-muted-foreground">Approved</div>
                </div>
                <div class="flex-1 p-2 rounded bg-red-500/10 border border-red-500/20 text-center">
                  <div class="text-lg font-bold text-red-400">{hitlStats.rejections}</div>
                  <div class="text-[9px] text-muted-foreground">Rejected</div>
                </div>
                <div class="flex-1 p-2 rounded bg-amber-500/10 border border-amber-500/20 text-center">
                  <div class="text-lg font-bold text-amber-400">{hitlStats.pending}</div>
                  <div class="text-[9px] text-muted-foreground">Pending</div>
                </div>
              </div>
            </div>

            {/* Approval Rate */}
            {hitlStats.totalInteractions > 0 && (
              <div class="p-3 rounded-lg border border-purple-500/30 bg-purple-500/5">
                <div class="text-[11px] text-muted-foreground uppercase tracking-wide mb-1">
                  Approval Rate
                </div>
                <div class="flex items-center gap-3">
                  <div class={`text-xl font-bold ${rateColor(
                    Math.round((hitlStats.approvals / (hitlStats.approvals + hitlStats.rejections || 1)) * 100)
                  )}`}>
                    {Math.round((hitlStats.approvals / (hitlStats.approvals + hitlStats.rejections || 1)) * 100)}%
                  </div>
                  <code class="text-[10px] font-mono text-purple-400">
                    {wipMeter(Math.round((hitlStats.approvals / (hitlStats.approvals + hitlStats.rejections || 1)) * 100), 12)}
                  </code>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});

export default PortalMetricsPanel;
