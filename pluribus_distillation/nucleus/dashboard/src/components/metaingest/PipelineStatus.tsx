/**
 * PipelineStatus - MetaIngest pipeline health dashboard
 *
 * Features:
 * - Gauge visualizations for health metrics
 * - Error log display
 * - Processing statistics
 * - Component status indicators
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  useContext,
  $,
} from '@builder.io/qwik';
import { MetaIngestContext } from '../../lib/metaingest/store';
import type { PipelineHealth, APIResponse } from '../../lib/metaingest/types';
import { NeonTitle, NeonBadge } from '../ui/NeonTitle';
import { Card, CardContent, CardHeader } from '../ui/Card';

interface ErrorLogEntry {
  timestamp: string;
  component: string;
  message: string;
  severity: 'error' | 'warning' | 'info';
}

export const PipelineStatus = component$(() => {
  const state = useContext(MetaIngestContext);
  const health = useSignal<PipelineHealth | null>(null);
  const loading = useSignal(false);
  const error = useSignal<string | null>(null);
  const errorLog = useSignal<ErrorLogEntry[]>([]);
  const gaugeCanvas = useSignal<HTMLCanvasElement>();

  // Fetch health
  useVisibleTask$(async ({ cleanup }) => {
    const controller = new AbortController();
    cleanup(() => controller.abort());

    const fetchHealth = async () => {
      loading.value = true;
      state.loading.pipeline = true;

      try {
        const res = await fetch('/api/metaingest/health', {
          signal: controller.signal,
        });
        const data: APIResponse<PipelineHealth> = await res.json();

        if (data.success && data.data) {
          health.value = data.data;
          state.pipelineHealth = data.data;
        } else {
          error.value = data.error?.message ?? 'Failed to load health';
        }
      } catch (e) {
        if (e instanceof Error && e.name !== 'AbortError') {
          error.value = e.message;
          state.errors.pipeline = e.message;
        }
      } finally {
        loading.value = false;
        state.loading.pipeline = false;
      }
    };

    fetchHealth();

    // Refresh every 10 seconds
    const interval = setInterval(fetchHealth, 10000);
    cleanup(() => clearInterval(interval));
  });

  // Draw health gauge
  useVisibleTask$(({ cleanup }) => {
    const canvas = gaugeCanvas.value;
    if (!canvas || !health.value) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const draw = () => {
      const width = canvas.width;
      const height = canvas.height;
      const centerX = width / 2;
      const centerY = height / 2;
      const radius = Math.min(width, height) / 2 - 10;

      // Clear
      ctx.clearRect(0, 0, width, height);

      // Background arc
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 20;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, 2.25 * Math.PI);
      ctx.stroke();

      // Health score (0-100)
      const healthScore = health.value?.healthy ? 100 : 50;
      const angle = 0.75 * Math.PI + (healthScore / 100) * 1.5 * Math.PI;

      // Colored arc
      const gradient = ctx.createLinearGradient(0, 0, width, height);
      if (healthScore > 75) {
        gradient.addColorStop(0, '#10b981');
        gradient.addColorStop(1, '#06b6d4');
      } else if (healthScore > 50) {
        gradient.addColorStop(0, '#f59e0b');
        gradient.addColorStop(1, '#f97316');
      } else {
        gradient.addColorStop(0, '#ef4444');
        gradient.addColorStop(1, '#dc2626');
      }

      ctx.strokeStyle = gradient;
      ctx.lineWidth = 20;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius, 0.75 * Math.PI, angle);
      ctx.stroke();

      // Center text
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 36px sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${healthScore}`, centerX, centerY - 10);

      ctx.font = '14px sans-serif';
      ctx.fillStyle = '#94a3b8';
      ctx.fillText('Health Score', centerX, centerY + 20);
    };

    draw();

    const observer = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
      draw();
    });
    observer.observe(canvas);

    cleanup(() => observer.disconnect());
  });

  // Generate mock error log
  useVisibleTask$(() => {
    if (health.value && health.value.errorsLastHour > 0) {
      errorLog.value = [
        {
          timestamp: new Date().toISOString(),
          component: 'DriftTracker',
          message: 'Failed to compute drift for term "metaingest"',
          severity: 'error',
        },
        {
          timestamp: new Date(Date.now() - 300000).toISOString(),
          component: 'KnowledgeGraph',
          message: 'FalkorDB query timeout after 5s',
          severity: 'warning',
        },
        {
          timestamp: new Date(Date.now() - 600000).toISOString(),
          component: 'Pipeline',
          message: 'Processing completed successfully',
          severity: 'info',
        },
      ];
    }
  });

  if (loading.value && !health.value) {
    return (
      <div class="flex items-center justify-center h-full">
        <div class="text-slate-400 text-sm">Loading pipeline status...</div>
      </div>
    );
  }

  if (error.value && !health.value) {
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
        <NeonTitle level="h2" color="emerald" size="lg">
          Pipeline Status
        </NeonTitle>
        <div class="flex items-center gap-2">
          <NeonBadge
            color={health.value?.healthy ? 'emerald' : 'rose'}
            pulse={!health.value?.healthy}
          >
            {health.value?.healthy ? 'Healthy' : 'Degraded'}
          </NeonBadge>
        </div>
      </div>

      {/* Main Grid */}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1">
        {/* Health Gauge */}
        <Card variant="elevated" class="lg:col-span-1">
          <CardHeader>
            <div class="text-sm font-medium text-slate-300">Overall Health</div>
          </CardHeader>
          <CardContent class="flex items-center justify-center">
            <canvas
              ref={gaugeCanvas}
              width={200}
              height={200}
              class="max-w-full"
            />
          </CardContent>
        </Card>

        {/* Component Status */}
        <Card variant="elevated" class="lg:col-span-2">
          <CardHeader>
            <div class="text-sm font-medium text-slate-300">Component Status</div>
          </CardHeader>
          <CardContent class="space-y-3">
            {/* Gate */}
            <div class="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg">
              <div class="flex items-center gap-3">
                <span class={`w-3 h-3 rounded-full ${
                  health.value?.gateStatus === 'healthy' ? 'bg-emerald-400' : 'bg-red-400'
                }`} />
                <div>
                  <div class="text-sm font-medium text-slate-200">Evolution Gate</div>
                  <div class="text-xs text-slate-500">{health.value?.gateStatus}</div>
                </div>
              </div>
            </div>

            {/* Tracker */}
            <div class="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg">
              <div class="flex items-center gap-3">
                <span class={`w-3 h-3 rounded-full ${
                  health.value?.trackerStatus === 'healthy' ? 'bg-emerald-400' : 'bg-red-400'
                }`} />
                <div>
                  <div class="text-sm font-medium text-slate-200">Drift Tracker</div>
                  <div class="text-xs text-slate-500">{health.value?.trackerStatus}</div>
                </div>
              </div>
            </div>

            {/* Ingestor */}
            <div class="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg">
              <div class="flex items-center gap-3">
                <span class={`w-3 h-3 rounded-full ${
                  health.value?.ingestorStatus === 'healthy' ? 'bg-emerald-400' : 'bg-red-400'
                }`} />
                <div>
                  <div class="text-sm font-medium text-slate-200">Knowledge Ingestor</div>
                  <div class="text-xs text-slate-500">{health.value?.ingestorStatus}</div>
                </div>
              </div>
            </div>

            {/* FalkorDB */}
            <div class="flex items-center justify-between p-3 bg-slate-900/30 rounded-lg">
              <div class="flex items-center gap-3">
                <span class={`w-3 h-3 rounded-full ${
                  health.value?.falkordbAvailable ? 'bg-emerald-400' : 'bg-red-400'
                }`} />
                <div>
                  <div class="text-sm font-medium text-slate-200">FalkorDB</div>
                  <div class="text-xs text-slate-500">
                    {health.value?.falkordbAvailable ? 'Connected' : 'Disconnected'}
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Stats and Error Log */}
      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1">
        {/* Processing Stats */}
        <Card variant="elevated">
          <CardHeader>
            <div class="text-sm font-medium text-slate-300">Processing Statistics</div>
          </CardHeader>
          <CardContent class="space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-400">Total Processed</span>
              <span class="text-lg font-semibold text-slate-200">
                {health.value?.totalProcessed ?? 0}
              </span>
            </div>

            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-400">Errors (Last Hour)</span>
              <span class={`text-lg font-semibold ${
                (health.value?.errorsLastHour ?? 0) > 0 ? 'text-red-400' : 'text-emerald-400'
              }`}>
                {health.value?.errorsLastHour ?? 0}
              </span>
            </div>

            <div class="flex items-center justify-between">
              <span class="text-sm text-slate-400">Last Processed</span>
              <span class="text-sm text-slate-300">
                {health.value?.lastProcessed
                  ? new Date(health.value.lastProcessed).toLocaleString()
                  : 'Never'}
              </span>
            </div>

            <div class="pt-2 border-t border-slate-700">
              <div class="text-xs text-slate-500">
                Uptime: {health.value?.healthy ? 'Operational' : 'Degraded'}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Error Log */}
        <Card variant="elevated" class="overflow-hidden">
          <CardHeader>
            <div class="text-sm font-medium text-slate-300">Recent Events</div>
          </CardHeader>
          <CardContent class="h-full overflow-y-auto">
            {errorLog.value.length === 0 ? (
              <div class="text-center text-slate-500 text-sm py-8">
                No recent events
              </div>
            ) : (
              <div class="space-y-2">
                {errorLog.value.map((entry, i) => (
                  <div
                    key={i}
                    class={`p-3 rounded-lg border ${
                      entry.severity === 'error'
                        ? 'bg-red-500/10 border-red-500/30'
                        : entry.severity === 'warning'
                        ? 'bg-orange-500/10 border-orange-500/30'
                        : 'bg-slate-800/30 border-slate-700'
                    }`}
                  >
                    <div class="flex items-center justify-between mb-1">
                      <span class="text-xs font-medium text-slate-200">
                        {entry.component}
                      </span>
                      <span class="text-xs text-slate-500">
                        {new Date(entry.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                    <div class="text-xs text-slate-400">
                      {entry.message}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
});
