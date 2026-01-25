import { component$, useComputed$, useSignal, useVisibleTask$, $, type Signal } from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';
import { Button } from './ui/Button';
import { FreshnessBadge } from './ui/FreshnessBadge';

// M3 Components - BusPulseWidget
import '@material/web/ripple/ripple.js';

type PulseFilter = 'all' | 'error' | 'warn' | 'metric' | 'artifact' | 'a2a';
type ActorFilter = string | null;

type BucketMeta = {
  buckets: number;
  bucketMs: number;
  windowMs: number;
  now: number;
};

type BucketSelection = {
  index: number;
  startMs: number;
  endMs: number;
  events: BusEvent[];
};

export interface BusPulseWidgetProps {
  events: BusEvent[];
  title?: string;
  heightClass?: string;
  maxEvents?: number;
}

function safeIsoToMs(iso: string | undefined): number | null {
  if (!iso) return null;
  const ms = Date.parse(iso);
  return Number.isFinite(ms) ? ms : null;
}

function domainOf(topic: string): string {
  const idx = topic.indexOf('.');
  return idx === -1 ? topic : topic.slice(0, idx);
}

// Unified color scheme for heatmap - consistent palette
// Cyan primary, Purple secondary, Gold accent, Red errors only
function topicToColor(topic: string): string {
  const domain = domainOf(topic);
  const colors: Record<string, string> = {
    // Primary actions - Cyan family (hsl 185)
    'a2a': 'rgba(0, 210, 210, 0.7)',       // bright cyan - agent coordination
    'dialogos': 'rgba(0, 190, 200, 0.6)',  // cyan - dialogos
    'operator': 'rgba(0, 180, 190, 0.6)',  // cyan - operators
    'pbresume': 'rgba(0, 170, 180, 0.6)',  // cyan - pbresume
    'metric': 'rgba(0, 200, 200, 0.6)',    // cyan - metrics
    'health': 'rgba(0, 190, 185, 0.6)',    // cyan-teal - health
    // Secondary - Purple family (hsl 270)
    'graphiti': 'rgba(140, 100, 200, 0.6)', // purple - graphiti/kg
    'kg': 'rgba(140, 100, 200, 0.6)',       // purple - kg
    'omega': 'rgba(160, 90, 180, 0.6)',     // purple-magenta - omega
    'lineage': 'rgba(130, 90, 190, 0.6)',   // purple - lineage
    // Accent - Gold (hsl 45)
    'qa': 'rgba(210, 170, 50, 0.6)',        // gold - QA
    // Error only - Red
    'secops': 'rgba(200, 80, 80, 0.6)',     // muted red - security
  };
  return colors[domain] || 'rgba(0, 200, 210, 0.55)'; // default cyan
}

// Phase 2 Step 9: Check if event is A2A coordination
function isA2AEvent(topic: string): boolean {
  return topic.startsWith('a2a.');
}

function formatBucketRange(startMs: number, endMs: number): string {
  const start = new Date(startMs).toISOString().slice(11, 19);
  const end = new Date(endMs).toISOString().slice(11, 19);
  return `${start}-${end}Z`;
}

export const BusPulseWidget = component$<BusPulseWidgetProps>(({
  events,
  title = 'BUS PULSE',
  heightClass = 'h-[520px]',
  maxEvents = 2500,
}) => {
  const filter = useSignal<PulseFilter>('all');
  const canvasRef = useSignal<HTMLCanvasElement>();
  const lastDrawn = useSignal(0);
  const lastFilterKey = useSignal('');
  // Phase 2 Step 10: Selected event for detail popover
  const selectedEvent = useSignal<BusEvent | null>(null);
  const actorFilter = useSignal<ActorFilter>(null);
  const bucketMeta = useSignal<BucketMeta>({ buckets: 0, bucketMs: 0, windowMs: 0, now: 0 });
  const selectedBucket = useSignal<BucketSelection | null>(null);

  const cycleFilter = $(() => {
    const next: Record<PulseFilter, PulseFilter> = {
      all: 'a2a',       // Phase 2 Step 8: A2A filter added
      a2a: 'error',
      error: 'warn',
      warn: 'metric',
      metric: 'artifact',
      artifact: 'all',
    };
    filter.value = next[filter.value];
  });

  const slice = useComputed$(() => events.slice(-maxEvents));

  const filtered = useComputed$(() => {
    let list = slice.value;
    switch (filter.value) {
      case 'a2a':         // Phase 2 Step 8: A2A filter
        list = list.filter(e => isA2AEvent(e.topic));
        break;
      case 'error':
        list = list.filter(e => e.level === 'error');
        break;
      case 'warn':
        list = list.filter(e => e.level === 'warn');
        break;
      case 'metric':
        list = list.filter(e => e.kind === 'metric');
        break;
      case 'artifact':
        list = list.filter(e => e.kind === 'artifact');
        break;
      case 'all':
      default:
        break;
    }

    if (actorFilter.value) {
      list = list.filter(e => e.actor === actorFilter.value);
    }

    return list;
  });

  const stats = useComputed$(() => {
    const list = slice.value;
    const now = Date.now();
    const last5m = now - 5 * 60 * 1000;
    const last60s = now - 60 * 1000;

    let total5m = 0;
    let total60s = 0;
    let errors5m = 0;
    let warns5m = 0;
    let a2a5m = 0;  // Phase 2 Step 9: Track A2A events

    const domains = new Map<string, number>();
    const actors = new Map<string, number>();

    for (const e of list) {
      const ms = safeIsoToMs(e.iso) ?? e.ts;
      if (!Number.isFinite(ms)) continue;
      if (ms >= last5m) {
        total5m++;
        if (e.level === 'error') errors5m++;
        if (e.level === 'warn') warns5m++;
        if (isA2AEvent(e.topic)) a2a5m++;  // Phase 2 Step 9
      }
      if (ms >= last60s) total60s++;
      domains.set(domainOf(e.topic), (domains.get(domainOf(e.topic)) || 0) + 1);
      actors.set(e.actor, (actors.get(e.actor) || 0) + 1);
    }

    const topDomains = Array.from(domains.entries()).sort((a, b) => b[1] - a[1]).slice(0, 10);
    const topActors = Array.from(actors.entries()).sort((a, b) => b[1] - a[1]).slice(0, 8);

    const last = list[list.length - 1];

    return {
      lastIso: last?.iso || null,
      total: list.length,
      eps60: total60s / 60,
      epm60: total60s,
      total5m,
      errors5m,
      warns5m,
      a2a5m,  // Phase 2 Step 9
      uniqueDomains: domains.size,
      uniqueActors: actors.size,
      topDomains,
      topActors,
    };
  });

  const actorChips = useComputed$(() => {
    const base = new Set(['claude', 'codex', 'gemini', 'qwen']);
    for (const [actor] of stats.value.topActors) {
      if (actor) base.add(actor);
    }
    return Array.from(base);
  });

  const actorCounts = useComputed$(() => Object.fromEntries(stats.value.topActors));

  const navigateToEvents = $((searchPattern: string, searchMode: 'glob' | 'regex' | 'ltl' | 'actor' | 'topic' = 'glob') => {
    try {
      window.dispatchEvent(new CustomEvent('pluribus:navigate', {
        detail: { view: 'events', searchPattern, searchMode },
      }));
    } catch {
      // ignore
    }
  });

  // Phase 2 Step 6: Navigate to FalkorDB graph view with event context
  const navigateToGraph = $((eventId?: string, topic?: string) => {
    try {
      window.dispatchEvent(new CustomEvent('pluribus:navigate', {
        detail: {
          view: 'persistence',
          focus: eventId ? `bus_event:${eventId}` : undefined,
          filter: topic ? { topic } : undefined,
        },
      }));
      // Also try direct navigation as fallback
      if (typeof window !== 'undefined') {
        const params = new URLSearchParams();
        if (eventId) params.set('focus', eventId);
        if (topic) params.set('topic', topic);
        const url = `/persistence${params.toString() ? '?' + params.toString() : ''}`;
        window.history.pushState({}, '', url);
        window.dispatchEvent(new PopStateEvent('popstate'));
      }
    } catch {
      // ignore
    }
  });

  const handleCanvasClick = $((ev: MouseEvent) => {
    const canvas = canvasRef.value;
    const meta = bucketMeta.value;
    if (!canvas || meta.buckets === 0) return;

    const rect = canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const idx = Math.min(meta.buckets - 1, Math.max(0, Math.floor((x / rect.width) * meta.buckets)));
    const startMs = meta.now - meta.windowMs + idx * meta.bucketMs;
    const endMs = startMs + meta.bucketMs;
    const bucketEvents = filtered.value.filter((e) => {
      const ms = safeIsoToMs(e.iso) ?? e.ts;
      return Number.isFinite(ms) && ms >= startMs && ms < endMs;
    });

    if (selectedBucket.value && selectedBucket.value.index === idx) {
      selectedBucket.value = null;
      return;
    }

    selectedBucket.value = {
      index: idx,
      startMs,
      endMs,
      events: bucketEvents.slice(-50),
    };
  });

  useVisibleTask$(({ track }) => {
    track(() => slice.value.length);
    track(() => filter.value);
    track(() => actorFilter.value);

    const canvas = canvasRef.value;
    if (!canvas) return;

    const dpr = Math.max(1, Math.floor(window.devicePixelRatio || 1));
    const width = canvas.clientWidth || 600;
    const height = canvas.clientHeight || 64;
    const drawKey = slice.value.length;
    const filterKey = `${filter.value}:${actorFilter.value || 'all'}`;

    if (lastDrawn.value === drawKey && lastFilterKey.value === filterKey) {
      // Avoid over-drawing on hot streams when we have no filter change.
      return;
    }

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    // Background
    const bg = ctx.createLinearGradient(0, 0, width, 0);
    bg.addColorStop(0, 'rgba(0,255,255,0.08)');
    bg.addColorStop(0.5, 'rgba(255,0,255,0.08)');
    bg.addColorStop(1, 'rgba(255,255,0,0.06)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    // Buckets: last 5 minutes
    const now = Date.now();
    const windowMs = 5 * 60 * 1000;
    const buckets = Math.max(60, Math.floor(width / 6));
    const bucketMs = windowMs / buckets;
    bucketMeta.value = { buckets, bucketMs, windowMs, now };
    const counts = new Array<number>(buckets).fill(0);
    const errs = new Array<number>(buckets).fill(0);
    const a2aCounts = new Array<number>(buckets).fill(0); // Phase 2 Step 9
    // Phase 2 Step 7: Track topic breakdown per bucket
    const topicCounts = new Array<Map<string, number>>(buckets);
    for (let i = 0; i < buckets; i++) topicCounts[i] = new Map();

    const list = filtered.value;
    for (const e of list) {
      const ms = safeIsoToMs(e.iso) ?? e.ts;
      if (!Number.isFinite(ms)) continue; // Skip invalid timestamps
      const age = now - ms;
      if (age < 0 || age > windowMs) continue;
      const idx = Math.min(buckets - 1, Math.max(0, Math.floor((windowMs - age) / bucketMs)));
      counts[idx] += 1;
      if (e.level === 'error') errs[idx] += 1;
      if (isA2AEvent(e.topic)) a2aCounts[idx] += 1; // Phase 2 Step 9
      // Phase 2 Step 7: Track topic
      const domain = domainOf(e.topic);
      topicCounts[idx].set(domain, (topicCounts[idx].get(domain) || 0) + 1);
    }

    const max = Math.max(1, ...counts);
    const barW = width / buckets;

    for (let i = 0; i < buckets; i++) {
      const h = (counts[i] / max) * (height - 8);
      const x = i * barW;

      // Phase 2 Step 7: Draw stacked bars by topic
      const tc = topicCounts[i];
      if (tc.size > 0 && counts[i] > 0) {
        // Sort topics by count descending
        const sortedTopics = Array.from(tc.entries()).sort((a, b) => b[1] - a[1]);
        let yOffset = height;
        for (const [domain, count] of sortedTopics) {
          const segmentH = (count / counts[i]) * h;
          ctx.fillStyle = topicToColor(domain);
          ctx.fillRect(x, yOffset - segmentH, Math.max(1, barW - 1), segmentH);
          yOffset -= segmentH;
        }
      } else {
        // Fallback: single color
        ctx.fillStyle = 'rgba(34,211,238,0.55)';
        ctx.fillRect(x, height - h, Math.max(1, barW - 1), h);
      }

      // Phase 2 Step 9: A2A highlight border
      if (a2aCounts[i] > 0) {
        const ah = (a2aCounts[i] / max) * (height - 8);
        ctx.strokeStyle = 'rgba(249, 115, 22, 0.9)'; // orange
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.5, height - ah - 0.5, Math.max(1, barW - 2), ah);
      }

      // Error overlay
      if (errs[i] > 0) {
        const eh = Math.min(h, (errs[i] / max) * (height - 8));
        ctx.fillStyle = 'rgba(239,68,68,0.75)'; // red
        ctx.fillRect(x, height - eh, Math.max(1, barW - 1), eh);
      }
    }

    // Scanline
    ctx.strokeStyle = 'rgba(255,255,255,0.15)';
    ctx.beginPath();
    ctx.moveTo(0, height - 0.5);
    ctx.lineTo(width, height - 0.5);
    ctx.stroke();

    lastDrawn.value = drawKey;
    lastFilterKey.value = filterKey;
  });

  const f = filter.value;
  const filterBadge = f === 'all'
    ? 'glass-chip'
    : f === 'a2a'  // Phase 2 Step 8: A2A badge style
      ? 'glass-chip text-orange-400'
      : f === 'error'
        ? 'glass-status-critical'
        : f === 'warn'
          ? 'glass-status-warning'
          : f === 'metric'
            ? 'glass-status-healthy'
            : 'glass-chip text-purple-400';

  return (
    <div class={`glass-surface-elevated glass-animate-enter flex flex-col ${heightClass} overflow-hidden`}>
      <div class="px-4 py-3 border-b border-glass-border-subtle flex items-center justify-between flex-shrink-0 glass-surface-subtle">
        <div class="flex items-center gap-2 min-w-0">
          <h3 class="text-sm font-semibold text-glass-text-muted">{title}</h3>
          <span class="text-[10px] px-2 py-0.5 rounded glass-chip text-green-400 shadow-[0_0_6px_rgba(74,222,128,0.3)]">
            {stats.value.total}
          </span>
          <FreshnessBadge
            timestamp={stats.value.lastIso}
            source="bus"
            ttlFresh={10}
            ttlRecent={60}
            ttlStale={300}
          />
        </div>
        <Button
          onClick$={cycleFilter}
          variant="tonal"
          class={`text-[11px] h-7 px-2.5 py-1 rounded border ${filterBadge}`}
          title="Cycle filter (all → a2a → error → warn → metric → artifact)"
        >
          {filter.value}
        </Button>
      </div>

      <div class="px-4 py-3 border-b border-[var(--glass-border-subtle)] flex-shrink-0">
        <canvas
          ref={canvasRef}
          class="w-full h-[80px] rounded-md bg-black/30 border border-[var(--glass-border)] cursor-pointer"
          title="Click a bucket to inspect events"
          onClick$={handleCanvasClick}
        />
        <div class="mt-3 grid grid-cols-2 md:grid-cols-6 gap-2 text-[11px] text-muted-foreground">
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">epm60</span> <span class="text-cyan-300">{Math.round(stats.value.epm60)}</span>
          </div>
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">eps</span> <span class="text-cyan-300">{stats.value.eps60.toFixed(2)}</span>
          </div>
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">5m</span> <span class="text-cyan-300">{stats.value.total5m}</span>
          </div>
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">err5m</span> <span class="text-red-300">{stats.value.errors5m}</span>
          </div>
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">a2a</span> <span class="text-orange-300">{stats.value.a2a5m}</span>
          </div>
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">dom</span> <span class="text-purple-300">{stats.value.uniqueDomains}</span>
          </div>
          <div class="rounded-md border border-[var(--glass-border)] bg-muted/20 px-2.5 py-1.5">
            <span class="mono">actors</span> <span class="text-purple-300">{stats.value.uniqueActors}</span>
          </div>
        </div>
        <div class="mt-3 flex flex-wrap gap-1 text-[11px]">
          <Button
            variant={!actorFilter.value ? 'primary' : 'text'}
            class="px-2.5 py-1 h-7"
            onClick$={() => { actorFilter.value = null; }}
            title="Clear actor filter"
          >
            all
          </Button>
          {actorChips.value.map((actor) => {
            const count = (actorCounts.value as Record<string, number>)[actor];
            const active = actorFilter.value === actor;
            return (
              <Button
                key={actor}
                variant={active ? 'primary' : 'text'}
                class="px-2.5 py-1 h-7"
                onClick$={() => { actorFilter.value = active ? null : actor; }}
                title={count ? `Events: ${count}` : 'Filter by actor'}
              >
                @{actor}{count ? `:${count}` : ''}
              </Button>
            );
          })}
        </div>
        {selectedBucket.value && (
          <div class="mt-3 rounded-md border border-[var(--glass-border)] bg-muted/30 p-3 text-[11px]">
            <div class="flex items-center justify-between">
              <span class="text-muted-foreground">
                bucket {selectedBucket.value.index + 1} - {formatBucketRange(selectedBucket.value.startMs, selectedBucket.value.endMs)} - {selectedBucket.value.events.length} events
              </span>
              <Button
                variant="tonal"
                class="text-[10px] h-6 px-2.5 py-1"
                onClick$={() => { selectedBucket.value = null; }}
              >
                clear
              </Button>
            </div>
            <div class="mt-2 space-y-1">
              {selectedBucket.value.events.length === 0 ? (
                <div class="text-muted-foreground/70">No events in bucket</div>
              ) : (
                selectedBucket.value.events.slice(-8).reverse().map((event) => (
                  <div key={`${event.id || event.ts}-${event.topic}`} class="flex items-center justify-between gap-2">
                    <div class="truncate text-muted-foreground">
                      {event.topic} <span class="text-muted-foreground/60">(@{event.actor})</span>
                    </div>
                                      <Button
                                        variant="tonal"
                                        class="text-[9px] h-6 px-2.5 py-1 bg-orange-500/20 text-orange-300 hover:bg-orange-500/30"
                                        onClick$={async (ev: Event) => { ev.stopPropagation(); await navigateToGraph(event.id, event.topic); }}
                                      >                      graph
                    </Button>
                  </div>
                ))
              )}
            </div>
          </div>
        )}
      </div>

      <div class="px-4 py-3 border-b border-[var(--glass-border-subtle)] flex-shrink-0">
        <div class="text-[11px] text-muted-foreground mb-2">TOP DOMAINS</div>
        <div class="flex flex-wrap gap-1">
          {stats.value.topDomains.map(([d, n]) => (
            <Button
              key={d}
              variant="tonal"
              class="text-[10px] h-6 px-2 py-1 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 hover:bg-cyan-500/20"
              onClick$={() => navigateToEvents(`${d}.`, 'topic')}
              title="Jump to Events view"
            >
              {d}:{n}
            </Button>
          ))}
        </div>
        <div class="text-[11px] text-muted-foreground mt-3 mb-2">TOP ACTORS</div>
        <div class="flex flex-wrap gap-1">
          {stats.value.topActors.map(([a, n]) => (
            <Button
              key={a}
              variant="tonal"
              class="text-[10px] h-6 px-2 py-1 rounded bg-purple-500/10 text-purple-300 border border-purple-500/20 hover:bg-purple-500/20"
              onClick$={() => navigateToEvents(a, 'actor')}
              title="Filter by actor in Events view"
            >
              @{a}:{n}
            </Button>
          ))}
        </div>
      </div>

      <div class="flex-1 min-h-0 overflow-auto p-3 font-mono text-[11px] leading-snug space-y-1.5">
        {filtered.value.slice(-60).reverse().map((e, i) => {
          const parts = e.topic.split('.');
          const d = parts[0];
          const rest = parts.slice(1).join('.');
          // Unified 3-color severity system (cyan/magenta/purple + error)
          const sev =
            e.level === 'error' ? 'glass-status-error border-l-2' :
            e.level === 'warn' ? 'glass-status-warn border-l-2' :
            e.kind === 'artifact' ? 'glass-status-ok border-l-2 opacity-70' :
            'bg-muted/20 text-muted-foreground';
          return (
            <div
              key={i}
              class={`p-2 rounded-md transition-colors hover:bg-muted/30 ${sev} group cursor-pointer relative overflow-hidden`}
              onClick$={() => { selectedEvent.value = e; }}
            >
              <md-ripple></md-ripple>
              <div class="flex items-center gap-1">
                <span class="text-muted-foreground/60 flex-shrink-0">{e.iso?.slice(11, 19)}</span>
                <span class="text-cyan-300 font-semibold">{d}</span>
                {rest && <span class="text-primary/70">.{rest}</span>}
                {/* Phase 2 Step 6: View in Graph button */}
                <Button
                  variant="text"
                  class="ml-auto text-[9px] h-5 px-1.5 py-0.5 rounded bg-orange-500/0 text-orange-400/0 group-hover:bg-orange-500/20 group-hover:text-orange-300 transition-all"
                  onClick$={async (ev: Event) => { ev.stopPropagation(); await navigateToGraph(e.id, e.topic); }}
                  title="View in FalkorDB Graph"
                >
                  →Graph
                </Button>
              </div>
              <div class="flex items-center justify-between mt-0.5">
                <span class="text-muted-foreground/60">@{e.actor}</span>
                <span class="text-muted-foreground/50">{e.kind}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Phase 2 Step 10: Event Detail Popover */}
      {selectedEvent.value && (
        <div
          class="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick$={() => { selectedEvent.value = null; }}
        >
          <div
            class="glass-surface-overlay bg-card/95 border border-[var(--glass-border)] rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-hidden flex flex-col"
            onClick$={(ev) => ev.stopPropagation()}
          >
            <div class="p-4 border-b border-[var(--glass-border)] flex items-center justify-between">
              <div>
                <h3 class="text-sm font-semibold text-foreground">Event Detail</h3>
                <p class="text-xs text-muted-foreground mt-0.5">
                  {selectedEvent.value.topic} @ {selectedEvent.value.iso}
                </p>
              </div>
              <div class="flex items-center gap-2">
                <Button
                  variant="tonal"
                  class="text-[11px] h-7 px-2.5 py-1 rounded bg-orange-500/20 text-orange-300 hover:bg-orange-500/30"
                  onClick$={() => {
                    navigateToGraph(selectedEvent.value?.id, selectedEvent.value?.topic);
                    selectedEvent.value = null;
                  }}
                >
                  View in Graph
                </Button>
                <Button
                  variant="icon"
                  icon="close"
                  class="text-muted-foreground hover:text-foreground"
                  onClick$={() => { selectedEvent.value = null; }}
                />
              </div>
            </div>
            <div class="p-4 overflow-auto flex-1">
              <div class="grid grid-cols-2 gap-3 text-xs mb-4">
                <div class="p-2 rounded bg-muted/30">
                  <span class="text-muted-foreground">ID:</span>
                  <span class="ml-2 font-mono text-cyan-400">{selectedEvent.value.id}</span>
                </div>
                <div class="p-2 rounded bg-muted/30">
                  <span class="text-muted-foreground">Actor:</span>
                  <span class="ml-2 font-mono text-purple-400">@{selectedEvent.value.actor}</span>
                </div>
                <div class="p-2 rounded bg-muted/30">
                  <span class="text-muted-foreground">Kind:</span>
                  <span class="ml-2 font-mono">{selectedEvent.value.kind}</span>
                </div>
                <div class="p-2 rounded bg-muted/30">
                  <span class="text-muted-foreground">Level:</span>
                  <span class={`ml-2 font-mono ${
                    selectedEvent.value.level === 'error' ? 'text-red-400' :
                    selectedEvent.value.level === 'warn' ? 'text-yellow-400' : 'text-green-400'
                  }`}>{selectedEvent.value.level}</span>
                </div>
              </div>
              <div class="text-[10px] text-muted-foreground mb-2">FULL JSON</div>
              <pre class="p-3 rounded bg-black/30 text-[10px] font-mono text-foreground/80 overflow-x-auto whitespace-pre-wrap break-all">
                {JSON.stringify(selectedEvent.value, null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default BusPulseWidget;
