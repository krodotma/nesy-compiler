/**
 * DriftMonitor - Semantic drift detection and timeline visualization
 *
 * Features:
 * - Timeline visualization using canvas
 * - Alert list with severity badges
 * - Real-time bus subscription
 * - Drift magnitude visualization
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useContext,
  $,
} from '@builder.io/qwik';
import { MetaIngestContext } from '../../lib/metaingest/store';
import type { DriftReport, DriftAlert, APIResponse } from '../../lib/metaingest/types';
import { NeonTitle, NeonBadge } from '../ui/NeonTitle';
import { Card, CardContent } from '../ui/Card';

export const DriftMonitor = component$(() => {
  const state = useContext(MetaIngestContext);
  const reports = useSignal<DriftReport[]>([]);
  const alerts = useSignal<DriftAlert[]>([]);
  const loading = useSignal(false);
  const error = useSignal<string | null>(null);
  const canvasRef = useSignal<HTMLCanvasElement>();
  const selectedTerm = useSignal<string | null>(null);

  // Fetch drift reports and alerts
  useVisibleTask$(async ({ cleanup }) => {
    const controller = new AbortController();
    cleanup(() => controller.abort());

    loading.value = true;
    state.loading.drift = true;

    try {
      // Fetch reports
      const reportsRes = await fetch('/api/metaingest/drift', {
        signal: controller.signal,
      });
      const reportsData: APIResponse<DriftReport[]> = await reportsRes.json();

      if (reportsData.success && reportsData.data) {
        reports.value = reportsData.data;
        state.driftReports = reportsData.data;
      }

      // Fetch alerts
      const alertsRes = await fetch('/api/metaingest/drift/alerts', {
        signal: controller.signal,
      });
      const alertsData: APIResponse<DriftAlert[]> = await alertsRes.json();

      if (alertsData.success && alertsData.data) {
        alerts.value = alertsData.data;
        state.driftAlerts = alertsData.data;
      }
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') {
        error.value = e.message;
        state.errors.drift = e.message;
      }
    } finally {
      loading.value = false;
      state.loading.drift = false;
    }
  });

  // Canvas timeline visualization
  useVisibleTask$(({ cleanup }) => {
    const canvas = canvasRef.value;
    if (!canvas || reports.value.length === 0) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;

      // Clear canvas
      ctx.fillStyle = '#0f172a';
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1;

      for (let i = 0; i <= 10; i++) {
        const y = (height / 10) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }

      // Draw timeline
      const sortedReports = [...reports.value].sort(
        (a, b) => new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime()
      );

      const maxMagnitude = Math.max(...sortedReports.map(r => r.driftMagnitude), 0.1);
      const xStep = width / Math.max(sortedReports.length - 1, 1);

      // Draw lines
      ctx.beginPath();
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;

      sortedReports.forEach((report, i) => {
        const x = i * xStep;
        const y = height - (report.driftMagnitude / maxMagnitude) * (height - 20);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      // Draw points
      sortedReports.forEach((report, i) => {
        const x = i * xStep;
        const y = height - (report.driftMagnitude / maxMagnitude) * (height - 20);

        // Point color based on severity
        if (report.driftMagnitude > 0.25) {
          ctx.fillStyle = '#ef4444'; // critical
        } else if (report.driftMagnitude > 0.15) {
          ctx.fillStyle = '#f97316'; // warning
        } else {
          ctx.fillStyle = '#06b6d4'; // normal
        }

        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();

        // Highlight selected term
        if (selectedTerm.value === report.term) {
          ctx.strokeStyle = '#fbbf24';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, Math.PI * 2);
          ctx.stroke();
        }
      });
    };

    draw();

    // Redraw on resize
    const observer = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      draw();
    });
    observer.observe(canvas);

    cleanup(() => observer.disconnect());
  });

  const selectTerm = $((term: string) => {
    selectedTerm.value = term;
  });

  if (loading.value && reports.value.length === 0) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-slate-400 text-sm">Loading drift data...</div>
      </div>
    );
  }

  if (error.value) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-red-400 text-sm">{error.value}</div>
      </div>
    );
  }

  return (
    <div class="flex flex-col h-full gap-4">
      {/* Header */}
      <div class="flex items-center justify-between">
        <NeonTitle level="h2" color="cyan" size="lg">
          Semantic Drift Monitor
        </NeonTitle>
        <div class="flex items-center gap-2">
          <NeonBadge color="cyan">
            {reports.value.length} Terms
          </NeonBadge>
          <NeonBadge color="rose" glow={alerts.value.length > 0}>
            {alerts.value.length} Alerts
          </NeonBadge>
        </div>
      </div>

      {/* Timeline visualization */}
      <Card variant="elevated" class="flex-1">
        <CardContent class="h-full flex flex-col">
          <div class="text-sm font-medium text-slate-300 mb-2">Drift Timeline</div>
          <canvas
            ref={canvasRef}
            class="flex-1 w-full rounded-lg"
            width={800}
            height={200}
          />
        </CardContent>
      </Card>

      {/* Split view: Reports and Alerts */}
      <div class="flex gap-4 flex-1 min-h-0">
        {/* Drift Reports */}
        <Card variant="elevated" class="flex-1 overflow-hidden">
          <CardContent class="h-full flex flex-col">
            <div class="text-sm font-medium text-slate-300 mb-3">Drift Reports</div>
            <div class="flex-1 overflow-y-auto space-y-2">
              {reports.value.length === 0 ? (
                <div class="text-slate-500 text-sm text-center py-8">
                  No drift reports available
                </div>
              ) : (
                reports.value.map((report) => (
                  <button
                    key={report.term}
                    onClick$={() => selectTerm(report.term)}
                    class={`w-full p-3 rounded-lg border transition-all text-left ${
                      selectedTerm.value === report.term
                        ? 'bg-cyan-500/10 border-cyan-500/30'
                        : 'bg-slate-800/30 border-slate-700 hover:bg-slate-800/50'
                    }`}
                  >
                    <div class="flex items-center justify-between mb-1">
                      <span class="text-sm font-medium text-slate-200">{report.term}</span>
                      <span class={`text-xs px-2 py-0.5 rounded ${
                        report.isDrifting
                          ? 'bg-yellow-500/20 text-yellow-300'
                          : 'bg-emerald-500/20 text-emerald-300'
                      }`}>
                        {report.driftDirection}
                      </span>
                    </div>
                    <div class="flex items-center gap-3 text-xs text-slate-400">
                      <span>Magnitude: {(report.driftMagnitude * 100).toFixed(1)}%</span>
                      <span>Observations: {report.observationCount}</span>
                      <span>Confidence: {(report.confidence * 100).toFixed(0)}%</span>
                    </div>
                    {/* Magnitude bar */}
                    <div class="mt-2 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        class={`h-full rounded-full ${
                          report.driftMagnitude > 0.25
                            ? 'bg-red-500'
                            : report.driftMagnitude > 0.15
                            ? 'bg-orange-500'
                            : 'bg-cyan-500'
                        }`}
                        style={{ width: `${Math.min(report.driftMagnitude * 100, 100)}%` }}
                      />
                    </div>
                  </button>
                ))
              )}
            </div>
          </CardContent>
        </Card>

        {/* Alerts */}
        <Card variant="elevated" class="w-80 overflow-hidden">
          <CardContent class="h-full flex flex-col">
            <div class="text-sm font-medium text-slate-300 mb-3">Active Alerts</div>
            <div class="flex-1 overflow-y-auto space-y-2">
              {alerts.value.length === 0 ? (
                <div class="text-slate-500 text-sm text-center py-8">
                  No active alerts
                </div>
              ) : (
                alerts.value.map((alert, idx) => (
                  <div
                    key={idx}
                    class={`p-3 rounded-lg border ${
                      alert.severity === 'critical'
                        ? 'bg-red-500/10 border-red-500/30'
                        : 'bg-orange-500/10 border-orange-500/30'
                    }`}
                  >
                    <div class="flex items-center gap-2 mb-1">
                      <span class={`w-2 h-2 rounded-full ${
                        alert.severity === 'critical' ? 'bg-red-400' : 'bg-orange-400'
                      }`} />
                      <span class="text-sm font-medium text-slate-200">{alert.term}</span>
                    </div>
                    <div class="text-xs text-slate-400 space-y-1">
                      <div>Magnitude: {(alert.magnitude * 100).toFixed(1)}%</div>
                      <div>Direction: {alert.direction}</div>
                      <div>
                        Detected: {new Date(alert.detectedAt).toLocaleString()}
                      </div>
                    </div>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
});
