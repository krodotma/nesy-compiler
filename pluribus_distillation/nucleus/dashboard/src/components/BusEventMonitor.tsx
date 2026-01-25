/**
 * BusEventMonitor - Real-time Pluribus event bus monitoring
 *
 * Phase 5, Iteration 39 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Real-time event stream display
 * - Topic filtering
 * - Event search
 * - Event detail inspection
 * - Event rate statistics
 * - Replay capability
 */

import {
  component$,
  useSignal,
  useStore,
  useComputed$,
  useVisibleTask$,
  $,
  noSerialize,
  type NoSerialize,
  type QRL,
} from '@builder.io/qwik';
import { NeonTitle, NeonBadge, NeonSectionHeader } from './ui/NeonTitle';

// M3 Components - BusEventMonitor
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/textfield/outlined-text-field.js';

// ============================================================================
// Types
// ============================================================================

export interface BusEvent {
  id: string;
  topic: string;
  timestamp: string;
  source: string;
  data: Record<string, unknown>;
  correlationId?: string;
  sequenceNum?: number;
}

export interface TopicSubscription {
  topic: string;
  pattern: string;
  enabled: boolean;
  color: string;
}

export interface EventStats {
  total: number;
  perSecond: number;
  byTopic: Record<string, number>;
  bySource: Record<string, number>;
}

interface PulseHistory {
  total: number[];
  errors: number[];
  metrics: number[];
  a2a: number[];
}

export interface BusEventMonitorProps {
  /** Initial events (from history) */
  initialEvents?: BusEvent[];
  /** WebSocket URL */
  wsUrl?: string;
  /** Default topic subscriptions */
  defaultTopics?: string[];
  /** Max events to keep in buffer */
  maxEvents?: number;
  /** Callback when event is selected */
  onEventSelect$?: QRL<(event: BusEvent) => void>;
  /** Callback when replay is requested */
  onReplayRequest$?: QRL<(fromTimestamp: string, topic?: string) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

const TOPIC_COLORS: Record<string, string> = {
  'agent': '#8b5cf6',
  'operator': '#3b82f6',
  'cascade': '#f97316',
  'lane': '#22c55e',
  'secops': '#ef4444',
  'bus': '#06b6d4',
  'system': '#6b7280',
};

const PULSE_HISTORY_MAX = 120;
const PULSE_LANES = [
  { key: 'total' as const, label: 'Total', color: '#22c55e' },
  { key: 'errors' as const, label: 'Errors', color: '#f97316' },
  { key: 'metrics' as const, label: 'Metrics', color: '#38bdf8' },
  { key: 'a2a' as const, label: 'A2A', color: '#a855f7' },
];

function getTopicColor(topic: string): string {
  const prefix = topic.split('.')[0];
  return TOPIC_COLORS[prefix] || '#6b7280';
}

function classifyEvent(event: BusEvent) {
  const topic = event.topic?.toLowerCase?.() || '';
  const data = event.data as Record<string, unknown>;
  const kind = typeof data?.kind === 'string' ? data.kind.toLowerCase() : '';
  const level = typeof data?.level === 'string' ? data.level.toLowerCase() : '';
  const isError = level === 'error' || topic.includes('error') || topic.includes('fail') || topic.includes('panic');
  const isMetric = kind === 'metric' || topic.includes('metric') || topic.startsWith('telemetry.');
  const isA2a = topic.startsWith('a2a.');
  return { isError, isMetric, isA2a };
}

function formatTimestamp(ts: string): string {
  try {
    const date = new Date(ts);
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      fractionalSecondDigits: 3,
    });
  } catch {
    return ts;
  }
}

function formatData(data: Record<string, unknown>): string {
  try {
    return JSON.stringify(data, null, 2);
  } catch {
    return String(data);
  }
}

function matchTopic(pattern: string, topic: string): boolean {
  const regex = new RegExp('^' + pattern.replace(/\*/g, '.*') + '$');
  return regex.test(topic);
}

function drawPulseCanvas(canvas: HTMLCanvasElement, history: PulseHistory) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const rect = canvas.getBoundingClientRect();
  const width = rect.width;
  const height = rect.height;
  if (width <= 0 || height <= 0) return;
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(width * dpr);
  canvas.height = Math.floor(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const paddingX = 10;
  const paddingY = 8;
  const laneHeight = (height - paddingY * 2) / PULSE_LANES.length;

  ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
  ctx.lineWidth = 1;
  PULSE_LANES.forEach((_, idx) => {
    const y = paddingY + idx * laneHeight + laneHeight * 0.8;
    ctx.beginPath();
    ctx.moveTo(paddingX, y);
    ctx.lineTo(width - paddingX, y);
    ctx.stroke();
  });

  PULSE_LANES.forEach((lane, idx) => {
    const values = history[lane.key];
    const maxValue = Math.max(1, ...values);
    const laneTop = paddingY + idx * laneHeight + laneHeight * 0.1;
    const laneSpan = laneHeight * 0.7;
    const points = values.length;
    if (points < 2) return;
    const xScale = (width - paddingX * 2) / (points - 1);

    ctx.save();
    ctx.strokeStyle = lane.color;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = paddingX + i * xScale;
      const y = laneTop + laneSpan - (v / maxValue) * laneSpan;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    ctx.globalAlpha = 0.25;
    ctx.lineWidth = 3;
    ctx.stroke();
    ctx.restore();

    const lastValue = values[values.length - 1] ?? 0;
    const lastX = paddingX + (points - 1) * xScale;
    const lastY = laneTop + laneSpan - (lastValue / maxValue) * laneSpan;
    ctx.fillStyle = lane.color;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2, 0, Math.PI * 2);
    ctx.fill();
  });
}

// ============================================================================
// Component
// ============================================================================

export const BusEventMonitor = component$<BusEventMonitorProps>(({
  initialEvents = [],
  wsUrl = 'wss://kroma.live/ws/bus',
  defaultTopics = ['*'],
  maxEvents = 500,
  onEventSelect$,
  onReplayRequest$,
}) => {
  // State
  const events = useSignal<BusEvent[]>(initialEvents);
  const selectedEventId = useSignal<string | null>(null);
  const searchQuery = useSignal('');
  const isPaused = useSignal(false);
  const showFilters = useSignal(false);
  const autoScroll = useSignal(true);

  const subscriptions = useSignal<TopicSubscription[]>(
    defaultTopics.map(t => ({
      topic: t,
      pattern: t,
      enabled: true,
      color: getTopicColor(t),
    }))
  );

  const wsState = useStore<{
    connected: boolean;
    ws: NoSerialize<WebSocket> | null;
    error: string | null;
  }>({
    connected: false,
    ws: null,
    error: null,
  });

  const stats = useStore<EventStats>({
    total: 0,
    perSecond: 0,
    byTopic: {},
    bySource: {},
  });

  // Rate tracking
  const eventTimes = useSignal<number[]>([]);
  const errorTimes = useSignal<number[]>([]);
  const metricTimes = useSignal<number[]>([]);
  const a2aTimes = useSignal<number[]>([]);

  const pulseCanvasRef = useSignal<HTMLCanvasElement>();
  const pulseHistory = useSignal<PulseHistory>({
    total: new Array(PULSE_HISTORY_MAX).fill(0),
    errors: new Array(PULSE_HISTORY_MAX).fill(0),
    metrics: new Array(PULSE_HISTORY_MAX).fill(0),
    a2a: new Array(PULSE_HISTORY_MAX).fill(0),
  });

  // Computed
  const selectedEvent = useComputed$(() =>
    events.value.find(e => e.id === selectedEventId.value)
  );

  const filteredEvents = useComputed$(() => {
    let result = events.value;

    // Apply topic filters
    const enabledPatterns = subscriptions.value.filter(s => s.enabled).map(s => s.pattern);
    if (enabledPatterns.length > 0 && !enabledPatterns.includes('*')) {
      result = result.filter(e =>
        enabledPatterns.some(p => matchTopic(p, e.topic))
      );
    }

    // Apply search
    if (searchQuery.value) {
      const query = searchQuery.value.toLowerCase();
      result = result.filter(e =>
        e.topic.toLowerCase().includes(query) ||
        e.source.toLowerCase().includes(query) ||
        JSON.stringify(e.data).toLowerCase().includes(query)
      );
    }

    return result;
  });

  const topicStats = useComputed$(() => {
    const topicCounts: Record<string, number> = {};
    events.value.forEach(e => {
      const prefix = e.topic.split('.')[0];
      topicCounts[prefix] = (topicCounts[prefix] || 0) + 1;
    });
    return Object.entries(topicCounts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8);
  });

  const latestTotal = pulseHistory.value.total[pulseHistory.value.total.length - 1] ?? 0;
  const latestErrors = pulseHistory.value.errors[pulseHistory.value.errors.length - 1] ?? 0;
  const latestMetrics = pulseHistory.value.metrics[pulseHistory.value.metrics.length - 1] ?? 0;
  const latestA2a = pulseHistory.value.a2a[pulseHistory.value.a2a.length - 1] ?? 0;
  const enabledTopics = subscriptions.value.filter(s => s.enabled).length;

  // WebSocket connection
  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const connect = () => {
      try {
        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
          wsState.connected = true;
          wsState.error = null;

          // Subscribe to topics
          subscriptions.value.forEach(sub => {
            if (sub.enabled) {
              ws.send(JSON.stringify({ type: 'subscribe', topic: sub.pattern }));
            }
          });
        };

        ws.onmessage = (e) => {
          if (isPaused.value) return;

          try {
            const event: BusEvent = JSON.parse(e.data);

            // Add to events buffer
            events.value = [event, ...events.value].slice(0, maxEvents);

            // Update stats
            stats.total++;
            stats.byTopic[event.topic] = (stats.byTopic[event.topic] || 0) + 1;
            stats.bySource[event.source] = (stats.bySource[event.source] || 0) + 1;

            // Track rate
            const now = Date.now();
            eventTimes.value = [...eventTimes.value.filter(t => now - t < 1000), now];
            stats.perSecond = eventTimes.value.length;
            const classification = classifyEvent(event);
            if (classification.isError) {
              errorTimes.value = [...errorTimes.value.filter(t => now - t < 1000), now];
            }
            if (classification.isMetric) {
              metricTimes.value = [...metricTimes.value.filter(t => now - t < 1000), now];
            }
            if (classification.isA2a) {
              a2aTimes.value = [...a2aTimes.value.filter(t => now - t < 1000), now];
            }
          } catch (err) {
            console.warn('[BusMonitor] Parse error:', err);
          }
        };

        ws.onclose = () => {
          wsState.connected = false;
          wsState.ws = null;

          // Auto-reconnect
          setTimeout(connect, 3000);
        };

        ws.onerror = () => {
          wsState.error = 'Connection error';
          wsState.connected = false;
        };

        wsState.ws = noSerialize(ws);
      } catch (err) {
        wsState.error = String(err);
      }
    };

    connect();

    cleanup(() => {
      if (wsState.ws) {
        (wsState.ws as unknown as WebSocket).close();
      }
    });
  });

  // Pulse history + canvas rendering
  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const trimHistory = (values: number[], nextValue: number) => {
      const sliced = values.length >= PULSE_HISTORY_MAX
        ? values.slice(values.length - (PULSE_HISTORY_MAX - 1))
        : values;
      return [...sliced, nextValue];
    };

    const updateHistory = () => {
      const now = Date.now();
      const filterTimes = (times: number[]) => times.filter(t => now - t < 1000);
      eventTimes.value = filterTimes(eventTimes.value);
      errorTimes.value = filterTimes(errorTimes.value);
      metricTimes.value = filterTimes(metricTimes.value);
      a2aTimes.value = filterTimes(a2aTimes.value);
      stats.perSecond = eventTimes.value.length;

      pulseHistory.value = {
        total: trimHistory(pulseHistory.value.total, eventTimes.value.length),
        errors: trimHistory(pulseHistory.value.errors, errorTimes.value.length),
        metrics: trimHistory(pulseHistory.value.metrics, metricTimes.value.length),
        a2a: trimHistory(pulseHistory.value.a2a, a2aTimes.value.length),
      };

      const canvas = pulseCanvasRef.value;
      if (canvas) {
        drawPulseCanvas(canvas, pulseHistory.value);
      }
    };

    const interval = window.setInterval(updateHistory, 1000);
    const handleResize = () => {
      const canvas = pulseCanvasRef.value;
      if (canvas) {
        drawPulseCanvas(canvas, pulseHistory.value);
      }
    };

    updateHistory();
    window.addEventListener('resize', handleResize);

    cleanup(() => {
      window.clearInterval(interval);
      window.removeEventListener('resize', handleResize);
    });
  });

  // Actions
  const toggleSubscription = $((index: number) => {
    const updated = [...subscriptions.value];
    updated[index].enabled = !updated[index].enabled;
    subscriptions.value = updated;

    // Update WebSocket subscription
    const sub = updated[index];
    if (wsState.ws) {
      const ws = wsState.ws as unknown as WebSocket;
      ws.send(JSON.stringify({
        type: sub.enabled ? 'subscribe' : 'unsubscribe',
        topic: sub.pattern,
      }));
    }
  });

  const addSubscription = $((pattern: string) => {
    if (subscriptions.value.some(s => s.pattern === pattern)) return;
    subscriptions.value = [
      ...subscriptions.value,
      {
        topic: pattern,
        pattern,
        enabled: true,
        color: getTopicColor(pattern),
      },
    ];
  });

  const clearEvents = $(() => {
    events.value = [];
    stats.total = 0;
    stats.byTopic = {};
    stats.bySource = {};
  });

  const selectEvent = $(async (event: BusEvent) => {
    selectedEventId.value = event.id;
    if (onEventSelect$) {
      await onEventSelect$(event);
    }
  });

  const requestReplay = $(async () => {
    if (!onReplayRequest$) return;
    const oldest = events.value[events.value.length - 1];
    if (oldest) {
      await onReplayRequest$(oldest.timestamp);
    }
  });

  return (
    <div class="rounded-lg glass-surface-elevated glass-animate-enter">
      {/* Header */}
      <div class="flex items-center justify-between px-4 py-3 border-b border-glass-border-subtle glass-surface-subtle">
        <div class="flex items-center gap-3">
          <NeonTitle level="span" color="cyan" size="xs" animation="flicker">Bus Event Monitor</NeonTitle>
          <div class={`w-2.5 h-2.5 rounded-full ${
            wsState.connected ? 'bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.6)] glass-pulse' : 'bg-red-400 animate-pulse'
          }`} />
          <span class="text-[10px] text-muted-foreground">
            {wsState.connected ? 'Live stream' : 'Disconnected'}
          </span>
        </div>
        <div class="flex items-center gap-2">
          <button
            onClick$={() => { isPaused.value = !isPaused.value; }}
            class={`text-[11px] px-3 py-1.5 rounded transition-colors ${
              isPaused.value
                ? 'bg-amber-500/20 text-amber-400'
                : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
            }`}
          >
            {isPaused.value ? '▶ Resume' : '⏸ Pause'}
          </button>
          <button
            onClick$={clearEvents}
            class="text-[11px] px-3 py-1.5 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
          >
            Clear
          </button>
          <button
            onClick$={() => { showFilters.value = !showFilters.value; }}
            class="text-[11px] px-3 py-1.5 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
          >
            Filters
          </button>
        </div>
      </div>

      {/* Signal deck */}
      <div class="px-4 py-4 border-b border-glass-border-subtle glass-surface-subtle">
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div class="lg:col-span-2">
            <div class="flex items-center justify-between">
              <NeonTitle level="span" color="magenta" size="xs">Signal Matrix</NeonTitle>
              <NeonBadge color="emerald" glow pulse>{stats.perSecond}/s live</NeonBadge>
            </div>
            <div class="mt-2 h-28 rounded-md border border-[var(--glass-border)] bg-gradient-to-br from-slate-950/80 via-slate-900/40 to-cyan-950/30 overflow-hidden relative">
              <canvas ref={pulseCanvasRef} class="w-full h-full block" />
              <div class="absolute bottom-2 left-2 flex items-center gap-3 text-[9px] text-slate-200/70">
                {PULSE_LANES.map((lane) => (
                  <div key={lane.key} class="flex items-center gap-1">
                    <span class="w-2 h-2 rounded-full" style={{ backgroundColor: lane.color }} />
                    <span>{lane.label}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <div class="grid grid-cols-2 gap-2">
            <div class="rounded-md border border-[var(--glass-border)] bg-slate-950/40 p-3">
              <div class="text-[9px] uppercase tracking-[0.24em] text-slate-400">Total /s</div>
              <div class="text-lg font-semibold" style={{ color: PULSE_LANES[0].color }}>{latestTotal}</div>
              <div class="text-[9px] text-slate-500">{stats.total} total</div>
            </div>
            <div class="rounded-md border border-[var(--glass-border)] bg-slate-950/40 p-3">
              <div class="text-[9px] uppercase tracking-[0.24em] text-slate-400">Errors /s</div>
              <div class="text-lg font-semibold" style={{ color: PULSE_LANES[1].color }}>{latestErrors}</div>
              <div class="text-[9px] text-slate-500">{enabledTopics} topics</div>
            </div>
            <div class="rounded-md border border-[var(--glass-border)] bg-slate-950/40 p-3">
              <div class="text-[9px] uppercase tracking-[0.24em] text-slate-400">Metrics /s</div>
              <div class="text-lg font-semibold" style={{ color: PULSE_LANES[2].color }}>{latestMetrics}</div>
              <div class="text-[9px] text-slate-500">{events.value.length}/{maxEvents} buffered</div>
            </div>
            <div class="rounded-md border border-[var(--glass-border)] bg-slate-950/40 p-3">
              <div class="text-[9px] uppercase tracking-[0.24em] text-slate-400">A2A /s</div>
              <div class="text-lg font-semibold" style={{ color: PULSE_LANES[3].color }}>{latestA2a}</div>
              <div class="text-[9px] text-slate-500">{filteredEvents.value.length} visible</div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div class="flex items-center gap-3 px-4 py-3 border-b border-glass-border-subtle glass-surface-subtle overflow-x-auto">
        {topicStats.value.map(([topic, count]) => (
          <div key={topic} class="flex items-center gap-2 text-[10px] whitespace-nowrap">
            <div
              class="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: getTopicColor(topic) }}
            />
            <span class="text-muted-foreground">{topic}</span>
            <span class="text-foreground font-semibold">{count}</span>
          </div>
        ))}
        <div class="flex items-center gap-2 text-[10px] ml-auto">
          <span class="text-muted-foreground">Total</span>
          <span class="text-foreground font-semibold">{stats.total}</span>
        </div>
      </div>

      {/* Filter panel */}
      {showFilters.value && (
        <div class="px-4 py-4 border-b border-[var(--glass-border)] bg-muted/5">
          <div class="flex items-center gap-2 mb-3">
            <input
              type="text"
              value={searchQuery.value}
              onInput$={(e) => { searchQuery.value = (e.target as HTMLInputElement).value; }}
              class="flex-grow px-3 py-2 text-[11px] rounded-md bg-card border border-[var(--glass-border)]"
              placeholder="Search events..."
            />
          </div>
          <div class="flex flex-wrap gap-2">
            {subscriptions.value.map((sub, i) => (
              <button
                key={sub.pattern}
                onClick$={() => toggleSubscription(i)}
                class={`text-[10px] px-3 py-1.5 rounded transition-colors ${
                  sub.enabled
                    ? 'border'
                    : 'bg-muted/20 text-muted-foreground border border-transparent'
                }`}
                style={sub.enabled ? {
                  backgroundColor: `${sub.color}20`,
                  borderColor: `${sub.color}50`,
                  color: sub.color,
                } : {}}
              >
                {sub.pattern}
              </button>
            ))}
            <button
              onClick$={() => {
                const pattern = prompt('Enter topic pattern (e.g., agent.*)');
                if (pattern) addSubscription(pattern);
              }}
              class="text-[10px] px-3 py-1.5 rounded bg-muted/20 text-muted-foreground hover:bg-muted/30"
            >
              + Add
            </button>
          </div>
        </div>
      )}

      {/* Main content */}
      <div class="grid grid-cols-2 gap-0 min-h-[300px]">
        {/* Event list */}
        <div class="border-r border-[var(--glass-border)] max-h-[420px] overflow-y-auto">
          {filteredEvents.value.length > 0 ? (
            <div class="divide-y divide-[var(--glass-border-subtle)]">
              {filteredEvents.value.map((event, idx) => (
                <div
                  key={event.id}
                  onClick$={() => selectEvent(event)}
                  class={`p-3 cursor-pointer glass-transition-colors glass-hover-glow ${
                    selectedEventId.value === event.id
                      ? 'glass-surface ring-1 ring-glass-accent/30'
                      : ''
                  }`}
                  style={{ '--stagger': idx % 20 } as any}
                >
                  <div class="flex items-center gap-2">
                    <div
                      class="w-2 h-2 rounded-full flex-shrink-0"
                      style={{ backgroundColor: getTopicColor(event.topic) }}
                    />
                    <span class="text-[10px] font-mono text-foreground truncate flex-grow">
                      {event.topic}
                    </span>
                    <span class="text-[9px] text-muted-foreground flex-shrink-0">
                      {formatTimestamp(event.timestamp)}
                    </span>
                  </div>
                  <div class="flex items-center gap-2 mt-1">
                    <span class="text-[9px] text-muted-foreground">from:</span>
                    <span class="text-[9px] text-cyan-400">@{event.source}</span>
                    {event.correlationId && (
                      <span class="text-[9px] text-purple-400/50 font-mono">
                        #{event.correlationId.slice(0, 8)}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[11px] text-muted-foreground">
              {events.value.length === 0 ? 'Waiting for events...' : 'No matching events'}
            </div>
          )}
        </div>

        {/* Event detail */}
        <div class="max-h-[420px] overflow-y-auto">
          {selectedEvent.value ? (
            <div class="p-4">
              {/* Event header */}
              <div class="mb-4 pb-4 border-b border-[var(--glass-border)]">
                <div class="flex items-center gap-2">
                  <div
                    class="w-3.5 h-3.5 rounded-full"
                    style={{ backgroundColor: getTopicColor(selectedEvent.value.topic) }}
                  />
                  <span class="text-sm font-mono font-medium text-foreground">
                    {selectedEvent.value.topic}
                  </span>
                </div>
                <div class="mt-3 grid grid-cols-2 gap-2 text-[10px]">
                  <div>
                    <span class="text-muted-foreground">Time:</span>
                    <span class="ml-1 text-foreground font-mono">
                      {formatTimestamp(selectedEvent.value.timestamp)}
                    </span>
                  </div>
                  <div>
                    <span class="text-muted-foreground">Source:</span>
                    <span class="ml-1 text-cyan-400">@{selectedEvent.value.source}</span>
                  </div>
                  {selectedEvent.value.correlationId && (
                    <div>
                      <span class="text-muted-foreground">Correlation:</span>
                      <span class="ml-1 text-purple-400 font-mono">
                        {selectedEvent.value.correlationId}
                      </span>
                    </div>
                  )}
                  {selectedEvent.value.sequenceNum !== undefined && (
                    <div>
                      <span class="text-muted-foreground">Seq:</span>
                      <span class="ml-1 text-foreground font-mono">
                        #{selectedEvent.value.sequenceNum}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Event data */}
              <div>
                <div class="text-[10px] font-semibold text-muted-foreground mb-2">PAYLOAD</div>
                <pre class="text-[10px] font-mono text-foreground bg-muted/10 p-3 rounded-md overflow-x-auto whitespace-pre-wrap">
                  {formatData(selectedEvent.value.data)}
                </pre>
              </div>

              {/* Actions */}
              <div class="mt-4 flex items-center gap-2">
                <button
                  onClick$={() => {
                    navigator.clipboard.writeText(JSON.stringify(selectedEvent.value, null, 2));
                  }}
                  class="text-[10px] px-3 py-1.5 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50"
                >
                  Copy JSON
                </button>
                <button
                  onClick$={() => {
                    // Filter to show only events with same correlation ID
                    if (selectedEvent.value?.correlationId) {
                      searchQuery.value = selectedEvent.value.correlationId;
                    }
                  }}
                  disabled={!selectedEvent.value?.correlationId}
                  class="text-[10px] px-3 py-1.5 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 disabled:opacity-50"
                >
                  Follow Chain
                </button>
              </div>
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[11px] text-muted-foreground">
              Select an event to view details
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div class="px-4 py-3 border-t border-glass-border-subtle flex items-center justify-between text-[10px] text-glass-text-muted glass-surface-subtle">
        <div class="flex items-center gap-2">
          <label class="flex items-center gap-1">
            <input
              type="checkbox"
              checked={autoScroll.value}
              onChange$={(e) => { autoScroll.value = (e.target as HTMLInputElement).checked; }}
              class="w-4 h-4"
            />
            Auto-scroll
          </label>
        </div>
        <div>
          {filteredEvents.value.length} of {events.value.length} events
          {isPaused.value && <span class="ml-2 text-amber-400">(paused)</span>}
        </div>
      </div>
    </div>
  );
});

export default BusEventMonitor;
