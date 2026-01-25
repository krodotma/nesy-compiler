/**
 * LanePerformanceMonitor - Real-time Performance Metrics
 *
 * Phase 9, Iteration 75 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Real-time FPS monitoring
 * - Memory usage tracking
 * - Render performance metrics
 * - Performance warnings
 * - Historical trends
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface PerformanceMetrics {
  fps: number;
  frameTime: number;
  memoryUsed: number;
  memoryTotal: number;
  renderCount: number;
  lastRenderTime: number;
  jankFrames: number;
  timestamp: number;
}

export interface PerformanceHistory {
  timestamps: number[];
  fps: number[];
  memory: number[];
  renderTimes: number[];
}

export interface PerformanceThresholds {
  minFps: number;
  maxFrameTime: number;
  maxMemory: number;
  maxJankPercentage: number;
}

export interface LanePerformanceMonitorProps {
  /** Show expanded view */
  expanded?: boolean;
  /** Enable monitoring */
  enabled?: boolean;
  /** Sampling interval in ms */
  sampleInterval?: number;
  /** History length (samples) */
  historyLength?: number;
  /** Performance thresholds */
  thresholds?: Partial<PerformanceThresholds>;
  /** Position */
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
}

// ============================================================================
// Default Values
// ============================================================================

const DEFAULT_THRESHOLDS: PerformanceThresholds = {
  minFps: 30,
  maxFrameTime: 33.33, // 30fps
  maxMemory: 500 * 1024 * 1024, // 500MB
  maxJankPercentage: 10,
};

// ============================================================================
// Helper Functions
// ============================================================================

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function getHealthColor(value: number, min: number, max: number, inverse: boolean = false): string {
  const normalized = (value - min) / (max - min);
  const score = inverse ? 1 - normalized : normalized;

  if (score >= 0.8) return 'text-emerald-400';
  if (score >= 0.6) return 'text-lime-400';
  if (score >= 0.4) return 'text-yellow-400';
  if (score >= 0.2) return 'text-orange-400';
  return 'text-red-400';
}

function getFpsColor(fps: number): string {
  if (fps >= 55) return 'text-emerald-400';
  if (fps >= 45) return 'text-lime-400';
  if (fps >= 30) return 'text-yellow-400';
  if (fps >= 20) return 'text-orange-400';
  return 'text-red-400';
}

function getMemoryColor(used: number, total: number): string {
  const ratio = used / total;
  if (ratio <= 0.5) return 'text-emerald-400';
  if (ratio <= 0.7) return 'text-yellow-400';
  if (ratio <= 0.85) return 'text-orange-400';
  return 'text-red-400';
}

// ============================================================================
// Performance Monitor Hook
// ============================================================================

function usePerformanceMonitor(sampleInterval: number, historyLength: number) {
  const metrics = useSignal<PerformanceMetrics>({
    fps: 60,
    frameTime: 16.67,
    memoryUsed: 0,
    memoryTotal: 0,
    renderCount: 0,
    lastRenderTime: 0,
    jankFrames: 0,
    timestamp: Date.now(),
  });

  const history = useSignal<PerformanceHistory>({
    timestamps: [],
    fps: [],
    memory: [],
    renderTimes: [],
  });

  useVisibleTask$(({ cleanup }) => {
    let frameCount = 0;
    let lastFrameTime = performance.now();
    let lastSampleTime = performance.now();
    let jankCount = 0;
    let animationId: number;

    const measureFrame = () => {
      const now = performance.now();
      const delta = now - lastFrameTime;
      lastFrameTime = now;
      frameCount++;

      // Count jank frames (>16.67ms)
      if (delta > 16.67) {
        jankCount++;
      }

      // Sample at interval
      if (now - lastSampleTime >= sampleInterval) {
        const fps = (frameCount / (now - lastSampleTime)) * 1000;
        const frameTime = (now - lastSampleTime) / frameCount;

        // Get memory if available
        let memoryUsed = 0;
        let memoryTotal = 0;
        if (typeof performance !== 'undefined' && (performance as any).memory) {
          memoryUsed = (performance as any).memory.usedJSHeapSize;
          memoryTotal = (performance as any).memory.jsHeapSizeLimit;
        }

        // Update metrics
        metrics.value = {
          fps: Math.round(fps),
          frameTime: Math.round(frameTime * 100) / 100,
          memoryUsed,
          memoryTotal,
          renderCount: metrics.value.renderCount + 1,
          lastRenderTime: frameTime,
          jankFrames: jankCount,
          timestamp: now,
        };

        // Update history
        const newHistory = { ...history.value };
        newHistory.timestamps.push(now);
        newHistory.fps.push(fps);
        newHistory.memory.push(memoryUsed);
        newHistory.renderTimes.push(frameTime);

        // Trim history
        while (newHistory.timestamps.length > historyLength) {
          newHistory.timestamps.shift();
          newHistory.fps.shift();
          newHistory.memory.shift();
          newHistory.renderTimes.shift();
        }

        history.value = newHistory;

        // Reset counters
        frameCount = 0;
        jankCount = 0;
        lastSampleTime = now;
      }

      animationId = requestAnimationFrame(measureFrame);
    };

    animationId = requestAnimationFrame(measureFrame);

    cleanup(() => {
      cancelAnimationFrame(animationId);
    });
  });

  return { metrics, history };
}

// ============================================================================
// Component
// ============================================================================

export const LanePerformanceMonitor = component$<LanePerformanceMonitorProps>(({
  expanded: initialExpanded = false,
  enabled = true,
  sampleInterval = 1000,
  historyLength = 60,
  thresholds: customThresholds = {},
  position = 'bottom-right',
}) => {
  const expanded = useSignal(initialExpanded);
  const thresholds = { ...DEFAULT_THRESHOLDS, ...customThresholds };

  const { metrics, history } = usePerformanceMonitor(sampleInterval, historyLength);

  // Calculate health score
  const healthScore = useComputed$(() => {
    const m = metrics.value;
    let score = 100;

    // FPS penalty
    if (m.fps < thresholds.minFps) {
      score -= Math.min(40, (thresholds.minFps - m.fps) * 2);
    }

    // Frame time penalty
    if (m.frameTime > thresholds.maxFrameTime) {
      score -= Math.min(30, (m.frameTime - thresholds.maxFrameTime));
    }

    // Memory penalty
    if (m.memoryTotal > 0) {
      const memoryRatio = m.memoryUsed / m.memoryTotal;
      if (memoryRatio > 0.8) {
        score -= Math.min(20, (memoryRatio - 0.8) * 100);
      }
    }

    // Jank penalty
    const jankRatio = m.jankFrames / Math.max(1, m.renderCount);
    if (jankRatio > thresholds.maxJankPercentage / 100) {
      score -= Math.min(10, jankRatio * 50);
    }

    return Math.max(0, Math.round(score));
  });

  // Warnings
  const warnings = useComputed$(() => {
    const w: string[] = [];
    const m = metrics.value;

    if (m.fps < thresholds.minFps) {
      w.push(`Low FPS: ${m.fps} (min: ${thresholds.minFps})`);
    }
    if (m.frameTime > thresholds.maxFrameTime) {
      w.push(`High frame time: ${m.frameTime.toFixed(1)}ms`);
    }
    if (m.memoryTotal > 0 && m.memoryUsed > thresholds.maxMemory) {
      w.push(`High memory: ${formatBytes(m.memoryUsed)}`);
    }

    return w;
  });

  // Position classes
  const positionClass = {
    'top-left': 'top-2 left-2',
    'top-right': 'top-2 right-2',
    'bottom-left': 'bottom-2 left-2',
    'bottom-right': 'bottom-2 right-2',
  }[position];

  if (!enabled) return null;

  return (
    <div class={`fixed ${positionClass} z-50`}>
      {/* Collapsed View */}
      {!expanded.value && (
        <button
          onClick$={() => { expanded.value = true; }}
          class="flex items-center gap-2 px-2 py-1 rounded bg-black/80 border border-border/50 backdrop-blur-sm"
        >
          <span class={`text-xs font-mono ${getFpsColor(metrics.value.fps)}`}>
            {metrics.value.fps} FPS
          </span>
          <span class={`w-2 h-2 rounded-full ${
            healthScore.value >= 80 ? 'bg-emerald-500' :
            healthScore.value >= 50 ? 'bg-yellow-500' :
            'bg-red-500'
          }`} />
        </button>
      )}

      {/* Expanded View */}
      {expanded.value && (
        <div class="w-72 rounded-lg bg-black/90 border border-border/50 backdrop-blur-sm overflow-hidden">
          {/* Header */}
          <div class="flex items-center justify-between p-2 border-b border-border/30">
            <div class="flex items-center gap-2">
              <span class="text-[10px] font-semibold text-muted-foreground">PERFORMANCE</span>
              <span class={`px-1.5 py-0.5 rounded text-[9px] font-medium ${
                healthScore.value >= 80 ? 'bg-emerald-500/20 text-emerald-400' :
                healthScore.value >= 50 ? 'bg-yellow-500/20 text-yellow-400' :
                'bg-red-500/20 text-red-400'
              }`}>
                {healthScore.value}%
              </span>
            </div>
            <button
              onClick$={() => { expanded.value = false; }}
              class="text-muted-foreground hover:text-foreground text-xs"
            >
              ✕
            </button>
          </div>

          {/* Metrics Grid */}
          <div class="grid grid-cols-2 gap-2 p-2">
            {/* FPS */}
            <div class="p-2 rounded bg-muted/10">
              <div class="text-[9px] text-muted-foreground">FPS</div>
              <div class={`text-lg font-mono ${getFpsColor(metrics.value.fps)}`}>
                {metrics.value.fps}
              </div>
            </div>

            {/* Frame Time */}
            <div class="p-2 rounded bg-muted/10">
              <div class="text-[9px] text-muted-foreground">Frame Time</div>
              <div class={`text-lg font-mono ${
                metrics.value.frameTime <= 16.67 ? 'text-emerald-400' :
                metrics.value.frameTime <= 33.33 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {metrics.value.frameTime.toFixed(1)}ms
              </div>
            </div>

            {/* Memory */}
            <div class="p-2 rounded bg-muted/10">
              <div class="text-[9px] text-muted-foreground">Memory</div>
              <div class={`text-lg font-mono ${getMemoryColor(metrics.value.memoryUsed, metrics.value.memoryTotal)}`}>
                {formatBytes(metrics.value.memoryUsed)}
              </div>
              {metrics.value.memoryTotal > 0 && (
                <div class="text-[8px] text-muted-foreground">
                  / {formatBytes(metrics.value.memoryTotal)}
                </div>
              )}
            </div>

            {/* Jank */}
            <div class="p-2 rounded bg-muted/10">
              <div class="text-[9px] text-muted-foreground">Jank Frames</div>
              <div class={`text-lg font-mono ${
                metrics.value.jankFrames === 0 ? 'text-emerald-400' :
                metrics.value.jankFrames < 5 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {metrics.value.jankFrames}
              </div>
            </div>
          </div>

          {/* FPS History Sparkline */}
          <div class="px-2 pb-2">
            <div class="text-[9px] text-muted-foreground mb-1">FPS History</div>
            <div class="h-8 flex items-end gap-px">
              {history.value.fps.slice(-30).map((fps, i) => (
                <div
                  key={i}
                  class={`flex-1 rounded-t ${
                    fps >= 55 ? 'bg-emerald-500' :
                    fps >= 30 ? 'bg-yellow-500' :
                    'bg-red-500'
                  }`}
                  style={{ height: `${Math.min(100, (fps / 60) * 100)}%` }}
                />
              ))}
            </div>
          </div>

          {/* Warnings */}
          {warnings.value.length > 0 && (
            <div class="px-2 pb-2">
              <div class="text-[9px] text-orange-400 mb-1">Warnings</div>
              <div class="space-y-1">
                {warnings.value.map((warning, i) => (
                  <div key={i} class="text-[9px] text-orange-300">
                    ⚠ {warning}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div class="flex items-center gap-1 p-2 border-t border-border/30">
            <button
              onClick$={() => {
                // Force garbage collection if available
                if ((window as any).gc) {
                  (window as any).gc();
                }
              }}
              class="flex-1 px-2 py-1 text-[9px] rounded bg-muted/20 text-muted-foreground hover:bg-muted/40"
            >
              Force GC
            </button>
            <button
              onClick$={() => {
                console.log('Performance Metrics:', metrics.value);
                console.log('History:', history.value);
              }}
              class="flex-1 px-2 py-1 text-[9px] rounded bg-muted/20 text-muted-foreground hover:bg-muted/40"
            >
              Log Metrics
            </button>
          </div>
        </div>
      )}
    </div>
  );
});

export default LanePerformanceMonitor;
