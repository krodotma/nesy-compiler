import { component$, useComputed$, type Signal } from '@builder.io/qwik';
import { useTracking } from '../../lib/telemetry/use-tracking';
import type { BusEvent } from '../../lib/state/types';

type MetricTone = 'ok' | 'warn' | 'info';
type LayerState = 'active' | 'partial' | 'idle';
type HealthState = 'ok' | 'warn' | 'hold';

interface RegistryAtlasViewProps {
  events?: Signal<BusEvent[]>;
}

const metricToneClass: Record<MetricTone, string> = {
  ok: 'text-emerald-400',
  warn: 'text-amber-400',
  info: 'text-cyan-400',
};

const layerStateClass: Record<LayerState, string> = {
  active: 'glass-status-dot-success',
  partial: 'glass-status-dot-warning',
  idle: 'glass-status-dot-info',
};

const healthStateClass: Record<HealthState, string> = {
  ok: 'glass-chip-accent-emerald',
  warn: 'glass-chip-accent-amber',
  hold: 'glass-chip-accent-rose',
};

export const RegistryAtlasView = component$<RegistryAtlasViewProps>((props) => {
  useTracking('comp:registry-atlas');

  // Computed: filter registry events from bus
  const registryEvents = useComputed$(() => {
    if (!props.events?.value) return [];
    return props.events.value
      .filter((e) => String(e.topic || '').startsWith('registry.'))
      .slice(-500);
  });

  // Computed: aggregate registry stats
  const registryStats = useComputed$(() => {
    const evts = registryEvents.value;
    const registries = new Set<string>();
    const entries = new Map<string, Set<string>>();
    let snapshotCount = 0;
    let entryCount = 0;

    for (const e of evts) {
      const topic = String(e.topic || '');
      const parts = topic.split('.');
      if (parts.length >= 2) {
        const regId = parts[1];
        registries.add(regId);
        if (!entries.has(regId)) entries.set(regId, new Set());

        if (topic.endsWith('.snapshot')) {
          snapshotCount++;
        } else if (topic.endsWith('.entry')) {
          entryCount++;
          const entryId = (e.data as Record<string, unknown>)?.entry?.id;
          if (entryId) entries.get(regId)?.add(String(entryId));
        }
      }
    }

    const totalEntries = Array.from(entries.values()).reduce((sum, s) => sum + s.size, 0);
    return {
      registryCount: registries.size,
      snapshotCount,
      entryCount,
      totalUniqueEntries: totalEntries,
      registries: Array.from(registries),
    };
  });

  // Computed: registry events for display
  const recentRegistryEvents = useComputed$(() => {
    return registryEvents.value.slice(-10).reverse().map((e, idx) => {
      const topic = String(e.topic || '');
      const action = topic.split('.').pop() || 'unknown';
      const regId = topic.split('.')[1] || 'unknown';
      const entryId = (e.data as Record<string, unknown>)?.entry?.id;
      const target = entryId ? `${regId}.${entryId}` : regId;
      const iso = String(e.iso || '');
      const time = iso.slice(11, 19) || '--:--:--';
      return {
        id: `evt-${idx}`,
        time,
        action: `registry.${action}`,
        target: String(target).slice(0, 30),
        status: 'ok' as const,
      };
    });
  });

  // Use live data if available, fallback to static
  const hasLiveData = useComputed$(() => registryEvents.value.length > 0);

  const metrics = useComputed$(() => {
    const stats = registryStats.value;
    if (hasLiveData.value) {
      return [
        { label: 'Registry Nodes', value: String(stats.totalUniqueEntries), delta: `+${stats.entryCount}`, tone: 'info' as MetricTone },
        { label: 'Registries', value: String(stats.registryCount), delta: '+0', tone: 'ok' as MetricTone },
        { label: 'Snapshots', value: String(stats.snapshotCount), delta: `+${stats.snapshotCount}`, tone: 'info' as MetricTone },
        { label: 'Events', value: String(registryEvents.value.length), delta: '+0', tone: 'ok' as MetricTone },
      ];
    }
    return [
      { label: 'Registry Nodes', value: '2,418', delta: '+82', tone: 'info' as MetricTone },
      { label: 'Domains', value: '14', delta: '+1', tone: 'ok' as MetricTone },
      { label: 'Live Links', value: '9,204', delta: '+317', tone: 'info' as MetricTone },
      { label: 'Drift Alerts', value: '3', delta: '+1', tone: 'warn' as MetricTone },
    ];
  });

  const layers = useComputed$(() => {
    const stats = registryStats.value;
    if (hasLiveData.value && stats.registries.length > 0) {
      return stats.registries.slice(0, 4).map((regId) => ({
        label: regId,
        count: '?',
        state: 'active' as LayerState,
      }));
    }
    return [
      { label: 'Agents', count: '42', state: 'active' as LayerState },
      { label: 'Services', count: '128', state: 'active' as LayerState },
      { label: 'Stores', count: '12', state: 'partial' as LayerState },
      { label: 'Pipelines', count: '9', state: 'idle' as LayerState },
    ];
  });

  const focusNodes = [
    { id: 'registry.core', type: 'core', region: 'global', status: 'stable' },
    { id: 'agents.codex', type: 'agent', region: 'edge', status: 'sync' },
    { id: 'services.signal-hub', type: 'service', region: 'zone-2', status: 'hot' },
    { id: 'stores.rhizome', type: 'store', region: 'zone-1', status: 'watch' },
  ];

  const inspectorFields = [
    { label: 'Node', value: 'registry.core' },
    { label: 'Class', value: 'primary' },
    { label: 'Owner', value: 'nucleus' },
    { label: 'Updated', value: '00:03:21' },
    { label: 'Region', value: 'global' },
    { label: 'Links', value: '84 inbound / 57 outbound' },
  ];

  const lineage = [
    { label: 'ingest', detail: 'registry.yaml', state: 'ok' as HealthState },
    { label: 'normalize', detail: 'schema v3', state: 'ok' as HealthState },
    { label: 'merge', detail: 'falkor', state: 'warn' as HealthState },
    { label: 'publish', detail: 'atlas feed', state: 'hold' as HealthState },
  ];

  const health = [
    { label: 'Consistency', value: '99.2%', state: 'ok' as HealthState },
    { label: 'Freshness', value: '8m lag', state: 'warn' as HealthState },
    { label: 'Coverage', value: '87%', state: 'ok' as HealthState },
    { label: 'Latency', value: '24ms', state: 'ok' as HealthState },
  ];

  const events = useComputed$(() => {
    if (hasLiveData.value && recentRegistryEvents.value.length > 0) {
      return recentRegistryEvents.value.slice(0, 5);
    }
    return [
      { id: 'ev-01', time: '12:03:14', action: 'node.register', target: 'agent.codex', status: 'ok' },
      { id: 'ev-02', time: '12:02:41', action: 'link.refresh', target: 'services.signal-hub', status: 'ok' },
      { id: 'ev-03', time: '12:02:09', action: 'policy.warn', target: 'stores.rhizome', status: 'warn' },
      { id: 'ev-04', time: '12:01:33', action: 'edge.drop', target: 'zone-3', status: 'hold' },
      { id: 'ev-05', time: '12:00:58', action: 'merge.queue', target: 'registry.atlas', status: 'ok' },
    ];
  });

  const signalBars = [12, 28, 18, 34, 22, 42, 30, 26, 18, 24, 36, 20];

  return (
    <div class="space-y-6">
      <div class="glass-surface p-5 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div class="space-y-2">
          <div class="text-[10px] uppercase tracking-[0.35em] text-muted-foreground">Registry</div>
          <h2 class="text-2xl font-semibold tracking-tight">Registry Atlas</h2>
          <p class="text-sm text-muted-foreground max-w-2xl">
            Live topology shell for registry domains, lineage, and active signal corridors.
          </p>
        </div>
        <div class="flex flex-wrap gap-2">
          <span class="glass-chip glass-chip-accent-cyan">atlas:active</span>
          <span class="glass-chip">scope:global</span>
          {hasLiveData.value ? (
            <span class="glass-chip glass-chip-accent-emerald">live data</span>
          ) : (
            <span class="glass-chip glass-chip-accent-amber">static mockup</span>
          )}
        </div>
      </div>

      <div class="grid gap-4 md:grid-cols-4">
        {metrics.value.map((metric) => (
          <div key={metric.label} class="glass-panel rounded-xl p-4">
            <div class="text-xs text-muted-foreground">{metric.label}</div>
            <div class={`text-2xl font-semibold ${metricToneClass[metric.tone]}`}>{metric.value}</div>
            <div class="text-[10px] text-muted-foreground">delta {metric.delta}</div>
          </div>
        ))}
      </div>

      <div class="grid grid-cols-12 gap-6">
        <div class="col-span-12 lg:col-span-3 space-y-4">
          <div class="glass-surface p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-sm font-semibold">Atlas Controls</h3>
              <span class="text-[10px] text-muted-foreground">scopes</span>
            </div>
            <div class="space-y-3">
              <input
                type="text"
                placeholder="Filter registry nodes..."
                class="w-full rounded-lg bg-muted/40 border border-border px-3 py-2 text-sm focus:outline-none focus:border-primary"
              />
              <div class="grid grid-cols-2 gap-2">
                {['All', 'Core', 'Edge', 'Drift'].map((filter) => (
                  <button
                    key={filter}
                    class="glass-chip w-full justify-center text-[10px] uppercase tracking-[0.2em]"
                  >
                    {filter}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div class="glass-surface p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-sm font-semibold">Layers</h3>
              <span class="text-[10px] text-muted-foreground">visibility</span>
            </div>
            <div class="space-y-3">
              {layers.value.map((layer) => (
                <div key={layer.label} class="flex items-center justify-between text-sm">
                  <div class="flex items-center gap-2">
                    <span class={`glass-status-dot ${layerStateClass[layer.state]}`} />
                    <span>{layer.label}</span>
                  </div>
                  <span class="text-xs text-muted-foreground">{layer.count}</span>
                </div>
              ))}
            </div>
          </div>

          <div class="glass-surface p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-sm font-semibold">Pinned Nodes</h3>
              <span class="text-[10px] text-muted-foreground">focus</span>
            </div>
            <div class="space-y-3">
              {focusNodes.map((node) => (
                <div key={node.id} class="rounded-lg border border-border/40 bg-muted/20 px-3 py-2">
                  <div class="text-xs text-muted-foreground">{node.type}</div>
                  <div class="text-sm font-medium">{node.id}</div>
                  <div class="text-[10px] text-muted-foreground">
                    {node.region} / {node.status}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div class="col-span-12 lg:col-span-6 space-y-4">
          <div class="glass-surface p-4">
            <div class="flex flex-wrap items-center justify-between gap-3 mb-3">
              <h3 class="text-sm font-semibold">Atlas Canvas</h3>
              <div class="flex flex-wrap gap-2">
                {['Orbit', 'Trace', 'Select'].map((mode) => (
                  <button
                    key={mode}
                    class="glass-chip text-[10px] uppercase tracking-[0.2em]"
                  >
                    {mode}
                  </button>
                ))}
              </div>
            </div>
            <div
              class="relative h-[360px] rounded-xl border border-border/60 overflow-hidden"
              style={{
                backgroundImage:
                  'linear-gradient(90deg, rgba(0, 255, 255, 0.08) 1px, transparent 1px), linear-gradient(0deg, rgba(0, 255, 255, 0.08) 1px, transparent 1px)',
                backgroundSize: '40px 40px',
              }}
            >
              <div class="absolute inset-0 bg-gradient-to-br from-cyan-500/5 via-transparent to-purple-500/10" />
              <div class="absolute left-4 top-4 text-xs text-muted-foreground">
                focus: registry.core
              </div>
              <div class="absolute right-4 top-4 text-xs text-muted-foreground">
                links: 124
              </div>
              <div class="absolute left-6 bottom-6 text-xs text-muted-foreground">
                latency: 24ms
              </div>
              <div class="absolute inset-0 flex items-center justify-center">
                <div class="glass-panel-subtle rounded-full px-4 py-2 text-[10px] uppercase tracking-[0.3em] text-muted-foreground">
                  atlas renderer pending
                </div>
              </div>
              <div class="absolute left-[18%] top-[30%] h-3 w-3 rounded-full bg-cyan-400/80 shadow-[0_0_10px_rgba(0,255,255,0.6)]" />
              <div class="absolute left-[45%] top-[55%] h-2.5 w-2.5 rounded-full bg-purple-400/70 shadow-[0_0_10px_rgba(188,19,254,0.5)]" />
              <div class="absolute left-[65%] top-[25%] h-2 w-2 rounded-full bg-emerald-400/70 shadow-[0_0_8px_rgba(16,185,129,0.6)]" />
              <div class="absolute left-[72%] top-[62%] h-2 w-2 rounded-full bg-amber-400/70 shadow-[0_0_8px_rgba(245,158,11,0.6)]" />
            </div>
          </div>

          <div class="grid gap-4 lg:grid-cols-2">
            <div class="glass-surface p-4">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-semibold">Query Builder</h3>
                <span class="text-[10px] text-muted-foreground">draft</span>
              </div>
              <div class="space-y-2 text-xs">
                {[
                  'select registry nodes where class = primary',
                  'expand links depth = 2',
                  'rank by freshness, coverage',
                ].map((step, index) => (
                  <div key={step} class="flex items-start gap-2 rounded-lg border border-border/40 bg-muted/20 px-3 py-2">
                    <span class="text-[10px] text-muted-foreground">{String(index + 1).padStart(2, '0')}</span>
                    <span class="leading-relaxed">{step}</span>
                  </div>
                ))}
              </div>
            </div>

            <div class="glass-surface p-4">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-sm font-semibold">Coverage Map</h3>
                <span class="text-[10px] text-muted-foreground">last 24h</span>
              </div>
              <div class="space-y-3">
                {[
                  { label: 'Global', value: 86 },
                  { label: 'Zone-1', value: 92 },
                  { label: 'Zone-2', value: 78 },
                  { label: 'Zone-3', value: 64 },
                ].map((row) => (
                  <div key={row.label} class="space-y-1">
                    <div class="flex items-center justify-between text-xs">
                      <span>{row.label}</span>
                      <span class="text-muted-foreground">{row.value}%</span>
                    </div>
                    <div class="h-2 rounded-full bg-muted/40 overflow-hidden">
                      <div
                        class="h-full bg-cyan-400/70"
                        style={{ width: `${row.value}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        <div class="col-span-12 lg:col-span-3 space-y-4">
          <div class="glass-surface p-4">
            <div class="flex items-center justify-between">
              <h3 class="text-sm font-semibold">Inspector</h3>
              <span class="glass-chip glass-chip-accent-emerald">active</span>
            </div>
            <div class="mt-3 space-y-2 text-xs">
              {inspectorFields.map((field) => (
                <div key={field.label} class="flex items-center justify-between border-b border-border/30 pb-2 last:border-b-0 last:pb-0">
                  <span class="text-muted-foreground">{field.label}</span>
                  <span class="text-right">{field.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div class="glass-surface p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-sm font-semibold">Lineage Trace</h3>
              <span class="text-[10px] text-muted-foreground">pipeline</span>
            </div>
            <div class="space-y-2">
              {lineage.map((step) => (
                <div key={step.label} class="flex items-center justify-between text-xs">
                  <div class="space-y-1">
                    <div class="font-medium">{step.label}</div>
                    <div class="text-[10px] text-muted-foreground">{step.detail}</div>
                  </div>
                  <span class={`glass-chip ${healthStateClass[step.state]}`}>{step.state}</span>
                </div>
              ))}
            </div>
          </div>

          <div class="glass-surface p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-sm font-semibold">Health Gates</h3>
              <span class="text-[10px] text-muted-foreground">live</span>
            </div>
            <div class="space-y-2">
              {health.map((item) => (
                <div key={item.label} class="flex items-center justify-between text-xs">
                  <span class="text-muted-foreground">{item.label}</span>
                  <span class={`glass-chip ${healthStateClass[item.state]}`}>{item.value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div class="grid gap-6 lg:grid-cols-2">
        <div class="glass-surface p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-semibold">Registry Events</h3>
            <span class="text-[10px] text-muted-foreground">last 60 min</span>
          </div>
          <div class="space-y-2">
            {events.value.map((event) => (
              <div key={event.id} class="flex items-center justify-between rounded-lg border border-border/40 bg-muted/20 px-3 py-2 text-xs">
                <div class="flex items-center gap-3">
                  <span class="text-muted-foreground">{event.time}</span>
                  <span class="font-medium">{event.action}</span>
                  <span class="text-muted-foreground">{event.target}</span>
                </div>
                <span class={`glass-chip ${healthStateClass[event.status === 'ok' ? 'ok' : event.status === 'warn' ? 'warn' : 'hold']}`}>
                  {event.status}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div class="glass-surface p-4">
          <div class="flex items-center justify-between mb-3">
            <h3 class="text-sm font-semibold">Signal Constellation</h3>
            <span class="text-[10px] text-muted-foreground">sparks</span>
          </div>
          <div class="grid grid-cols-2 gap-3 text-xs">
            {[
              { label: 'Throughput', value: '6.4k/min' },
              { label: 'Entropy', value: '0.14' },
              { label: 'Backlog', value: '31' },
              { label: 'Route Drift', value: '2.3%' },
            ].map((stat) => (
              <div key={stat.label} class="rounded-lg border border-border/40 bg-muted/20 px-3 py-2">
                <div class="text-[10px] text-muted-foreground">{stat.label}</div>
                <div class="text-sm font-semibold">{stat.value}</div>
              </div>
            ))}
          </div>
          <div class="mt-4 h-24 rounded-lg border border-border/40 bg-muted/30 px-3 py-2 flex items-end gap-2">
            {signalBars.map((bar, index) => (
              <div
                key={`bar-${index}`}
                class="flex-1 rounded-sm bg-cyan-400/70"
                style={{ height: `${bar}%` }}
              />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});
