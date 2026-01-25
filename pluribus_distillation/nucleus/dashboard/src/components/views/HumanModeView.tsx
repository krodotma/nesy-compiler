import { component$, useComputed$, useSignal, useVisibleTask$, type Signal, $ } from '@builder.io/qwik';
import type { AgentStatus, BusEvent, ServiceDef, STRpRequest, TaskLedgerEntry } from '../../lib/state/types';
import { buildHumanModeSnapshot, type HumanModeMode } from '../../lib/human-mode/humanModeData';
import { OrchestrationTimeline, type TimelineAgent, type TimelineEvent } from '../OrchestrationTimeline';
import { TimelineSparkline, EventFlowmap } from '../EventVisualization';
import { PBLanesWidget } from '../PBLanesWidget';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { NeonTitle } from '../ui/NeonTitle';

interface HumanModeViewProps {
  events: Signal<BusEvent[]>;
  agents: Signal<AgentStatus[]>;
  requests: Signal<STRpRequest[]>;
  services: Signal<ServiceDef[]>;
  connected: Signal<boolean>;
}

const PALETTE = ['#22c55e', '#38bdf8', '#f97316', '#a855f7', '#facc15', '#14b8a6'];

const formatAge = (ageMs: number | null): string => {
  if (!ageMs || ageMs <= 0) return 'unknown';
  const seconds = Math.floor(ageMs / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h`;
};

const statusChip = (status: string) => {
  if (status === 'completed') return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
  if (status === 'abandoned' || status === 'failed') return 'bg-zinc-500/20 text-zinc-300 border-zinc-500/30';
  if (status === 'blocked') return 'bg-red-500/20 text-red-400 border-red-500/30';
  if (status === 'in_progress' || status === 'working') return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  if (status === 'planned' || status === 'pending') return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
  return 'bg-muted/40 text-muted-foreground border-border/40';
};

const EVENT_PREFIXES = ['all', 'strp.', 'dialogos.', 'provider.', 'omega.', 'task_ledger.'] as const;
const TASK_FILTERS = ['all', 'planned', 'in_progress', 'blocked', 'completed', 'abandoned'] as const;
const TASK_SOURCES = ['all', 'task_ledger', 'strp'] as const;

const csvEscape = (value: unknown) => {
  const str = value === null || value === undefined ? '' : String(value);
  return `"${str.replace(/"/g, '""')}"`;
};

const evidenceToCsv = (events: BusEvent[]) => {
  const header = ['id', 'topic', 'kind', 'level', 'actor', 'iso', 'ts', 'data'];
  const rows = events.map((event) => ([
    event.id,
    event.topic,
    event.kind,
    event.level,
    event.actor,
    event.iso,
    typeof event.ts === 'number' ? event.ts : '',
    JSON.stringify(event.data ?? {}),
  ].map(csvEscape).join(',')));
  return [header.join(','), ...rows].join('\n');
};

const triggerDownload = (filename: string, content: string, mime: string) => {
  if (typeof document === 'undefined') return;
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

const buildCounts = (values: string[], limit: number) => {
  const counts = new Map<string, number>();
  values.forEach((value) => {
    if (!value) return;
    counts.set(value, (counts.get(value) || 0) + 1);
  });
  return Array.from(counts.entries())
    .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
    .slice(0, limit)
    .map(([key, count]) => ({ key, count }));
};

const topicBucket = (topic: string) => {
  const parts = topic.split('.');
  if (parts.length >= 2) return `${parts[0]}.${parts[1]}.*`;
  return topic || 'unknown';
};

export const HumanModeView = component$<HumanModeViewProps>((props) => {
  const mode = useSignal<HumanModeMode>('present');
  const windowMinutes = useSignal(60);
  const anchorMs = useSignal(Date.now());
  const followNow = useSignal(true);
  const eventSearch = useSignal('');
  const eventActor = useSignal('all');
  const eventLevel = useSignal('all');
  const eventPrefix = useSignal<(typeof EVENT_PREFIXES)[number]>('all');
  const eventErrorsOnly = useSignal(false);
  const taskStatus = useSignal<(typeof TASK_FILTERS)[number]>('all');
  const taskSource = useSignal<(typeof TASK_SOURCES)[number]>('all');
  const taskLedger = useSignal<TaskLedgerEntry[]>([]);
  const taskLedgerStatus = useSignal<'idle' | 'ok' | 'error'>('idle');
  const taskLedgerError = useSignal<string | null>(null);
  const taskLedgerPath = useSignal<string | null>(null);
  const taskLedgerUpdatedAt = useSignal<string | null>(null);

  useVisibleTask$(({ track, cleanup }) => {
    track(() => followNow.value);
    if (!followNow.value) return;
    const timer = setInterval(() => {
      anchorMs.value = Date.now();
    }, 60000);
    cleanup(() => clearInterval(timer));
  });

  useVisibleTask$(({ cleanup }) => {
    let cancelled = false;

    const fetchLedger = async () => {
      try {
        const res = await fetch('/api/git/task_ledger?limit=200');
        if (!res.ok) {
          throw new Error(`Task ledger fetch failed (${res.status})`);
        }
        const data = await res.json();
        if (cancelled) return;
        taskLedger.value = Array.isArray(data.entries) ? data.entries : [];
        taskLedgerStatus.value = data.error ? 'error' : 'ok';
        taskLedgerError.value = data.error || null;
        taskLedgerPath.value = typeof data.path === 'string' ? data.path : null;
        taskLedgerUpdatedAt.value = new Date().toISOString();
      } catch (err) {
        if (cancelled) return;
        taskLedgerStatus.value = 'error';
        taskLedgerError.value = String(err);
        taskLedgerUpdatedAt.value = new Date().toISOString();
      }
    };

    fetchLedger();
    const timer = setInterval(fetchLedger, 60000);
    cleanup(() => {
      cancelled = true;
      clearInterval(timer);
    });
  });

  const snapshot = useComputed$(() => buildHumanModeSnapshot({
    mode: mode.value,
    windowMinutes: windowMinutes.value,
    anchorMs: anchorMs.value,
    events: props.events.value,
    requests: props.requests.value,
    taskLedger: taskLedger.value,
    taskLedgerAvailable: taskLedgerStatus.value === 'ok' ? true : taskLedgerStatus.value === 'error' ? false : undefined,
    taskLedgerError: taskLedgerError.value,
    taskLedgerPath: taskLedgerPath.value,
    agents: props.agents.value,
    services: props.services.value,
    connected: props.connected.value,
  }));

  const actorOptions = useComputed$(() => {
    const actors = new Set<string>();
    snapshot.value.events.forEach((event) => {
      if (event.actor) actors.add(event.actor);
    });
    return ['all', ...Array.from(actors).sort()];
  });

  const levelOptions = ['all', 'debug', 'info', 'warn', 'error'];

  const filteredEvents = useComputed$(() => {
    const search = eventSearch.value.trim().toLowerCase();
    return snapshot.value.events.filter((event) => {
      if (eventErrorsOnly.value && event.level !== 'error') return false;
      if (eventActor.value !== 'all' && event.actor !== eventActor.value) return false;
      if (eventLevel.value !== 'all' && event.level !== eventLevel.value) return false;
      if (eventPrefix.value !== 'all' && !(event.topic || '').startsWith(eventPrefix.value)) return false;
      if (search) {
        const hay = `${event.topic || ''} ${event.actor || ''} ${event.kind || ''} ${event.level || ''}`.toLowerCase();
        if (!hay.includes(search)) return false;
      }
      return true;
    });
  });

  const filteredEvidence = useComputed$(() => {
    return filteredEvents.value.filter((event) => {
      const topic = event.topic || '';
      const kind = event.kind || '';
      if (kind === 'artifact' || kind === 'alert') return true;
      return ['evidence', 'beam', 'task_ledger', 'report'].some((token) => topic.includes(token));
    });
  });

  const filteredTasks = useComputed$(() => {
    let tasks = snapshot.value.tasks;
    if (taskSource.value !== 'all') {
      tasks = tasks.filter((task) => task.source === taskSource.value);
    }
    if (taskStatus.value !== 'all') {
      tasks = tasks.filter((task) => task.status === taskStatus.value);
    }
    return tasks;
  });

  const topTopics = useComputed$(() => {
    return buildCounts(
      filteredEvents.value.map((event) => topicBucket(event.topic || '')),
      6,
    );
  });

  const topActors = useComputed$(() => {
    return buildCounts(
      filteredEvents.value.map((event) => event.actor || 'unknown'),
      6,
    );
  });

  const taskStatusMix = useComputed$(() => {
    return buildCounts(
      filteredTasks.value.map((task) => task.status || 'unknown'),
      6,
    );
  });

  const timelineAgents = useComputed$(() => {
    const names = props.agents.value.length > 0
      ? props.agents.value.map((agent) => agent.actor)
      : Array.from(new Set(filteredEvents.value.map((event) => event.actor || 'unknown')));
    return names.map((name, idx) => ({
      id: name,
      name,
      color: PALETTE[idx % PALETTE.length],
    })) as TimelineAgent[];
  });

  const timelineEvents = useComputed$(() => {
    return filteredEvents.value.map((event) => {
      const topic = event.topic || 'event';
      const type = topic.includes('start')
        ? 'task_start'
        : topic.includes('complete')
        ? 'task_end'
        : event.level === 'error'
        ? 'error'
        : 'milestone';
      return {
        id: event.id || `${topic}-${event.ts}`,
        agentId: event.actor || 'unknown',
        agentName: event.actor || 'unknown',
        type,
        title: topic,
        timestamp: event.iso || new Date().toISOString(),
      } as TimelineEvent;
    });
  });

  const windowSummaries = useComputed$(() => {
    const base = {
      mode: mode.value,
      anchorMs: anchorMs.value,
      events: props.events.value,
      requests: props.requests.value,
      taskLedger: taskLedger.value,
      taskLedgerAvailable: taskLedgerStatus.value === 'ok' ? true : taskLedgerStatus.value === 'error' ? false : undefined,
      taskLedgerError: taskLedgerError.value,
      agents: props.agents.value,
      services: props.services.value,
      connected: props.connected.value,
    };
    return [15, 60, 360].map((minutes) => {
      const summary = buildHumanModeSnapshot({ ...base, windowMinutes: minutes }).summary;
      return { minutes, events: summary.eventCount, tasks: summary.taskCount };
    });
  });

  const downloadEvidence = $((format: 'json' | 'csv') => {
    const payload = {
      exported_at: new Date().toISOString(),
      mode: mode.value,
      window: snapshot.value.window,
      summary: snapshot.value.summary,
      filters: {
        actor: eventActor.value,
        level: eventLevel.value,
        prefix: eventPrefix.value,
        errors_only: eventErrorsOnly.value,
        task_status: taskStatus.value,
        task_source: taskSource.value,
      },
      evidence: filteredEvidence.value,
    };

    if (format === 'json') {
      triggerDownload('human-mode-evidence.json', JSON.stringify(payload, null, 2), 'application/json');
      return;
    }
    triggerDownload('human-mode-evidence.csv', evidenceToCsv(filteredEvidence.value), 'text/csv');
  });

  return (
    <div class="space-y-6 glass-panel p-6">
      <div class="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div class="space-y-1">
          <NeonTitle level="h2" color="cyan" size="sm">Human Mode</NeonTitle>
          <p class="text-sm text-muted-foreground">Temporal orchestration lens for tasks, agents, services, and evidence.</p>
        </div>
        <div class="flex flex-wrap gap-2">
          <Button
            variant="tonal"
            onClick$={() => { mode.value = 'historic'; followNow.value = false; }}
            class={`glass-transition-standard ${mode.value === 'historic' ? 'glass-chip-accent-amber' : 'glass-bg-card'}`}
          >
            Historic
          </Button>
          <Button
            variant="tonal"
            onClick$={() => { mode.value = 'present'; followNow.value = true; }}
            class={`glass-transition-standard ${mode.value === 'present' ? 'glass-chip-accent-emerald' : 'glass-bg-card'}`}
          >
            Present
          </Button>
          <Button
            variant="tonal"
            onClick$={() => { mode.value = 'future'; followNow.value = false; }}
            class={`glass-transition-standard ${mode.value === 'future' ? 'glass-chip-accent-cyan' : 'glass-bg-card'}`}
          >
            Future
          </Button>
        </div>
      </div>

      <div class="flex flex-wrap items-center gap-2 glass-surface-elevated p-3">
        <span class="text-xs uppercase tracking-[0.2em] text-muted-foreground">Window</span>
        {[15, 60, 360].map((minutes) => (
          <Button
            key={minutes}
            variant="secondary"
            onClick$={() => { windowMinutes.value = minutes; followNow.value = mode.value === 'present'; }}
            class={`text-xs ${windowMinutes.value === minutes ? 'glass-chip-accent-magenta' : ''}`}
          >
            {minutes >= 60 ? `${minutes / 60}h` : `${minutes}m`}
          </Button>
        ))}
        <div class="flex items-center gap-2 ml-auto">
          <Button
            variant="secondary"
            onClick$={() => { anchorMs.value -= 15 * 60 * 1000; followNow.value = false; }}
            class="text-xs"
          >
            -15m
          </Button>
          <Button
            variant="secondary"
            onClick$={() => { anchorMs.value += 15 * 60 * 1000; followNow.value = false; }}
            class="text-xs"
          >
            +15m
          </Button>
          <Button
            variant="secondary"
            onClick$={() => { anchorMs.value = Date.now(); followNow.value = mode.value === 'present'; }}
            class="text-xs"
          >
            Now
          </Button>
          <span class="text-xs text-muted-foreground">Anchor: {new Date(anchorMs.value).toLocaleString()}</span>
        </div>
      </div>

      <div class="grid gap-4 lg:grid-cols-[2fr,1fr]">
        <div class="glass-surface-elevated p-4 space-y-3">
          <div class="flex items-center justify-between">
            <NeonTitle level="span" color="cyan" size="xs">Filters</NeonTitle>
            <Button
              variant="text"
              onClick$={() => {
                eventSearch.value = '';
                eventActor.value = 'all';
                eventLevel.value = 'all';
                eventPrefix.value = 'all';
                eventErrorsOnly.value = false;
                taskStatus.value = 'all';
                taskSource.value = 'all';
              }}
              class="text-xs"
            >
              Clear
            </Button>
          </div>
          <div class="grid gap-2 md:grid-cols-[1.2fr,1fr,1fr]">
            <Input
              type="search"
              placeholder="Search topic / actor / level"
              value={eventSearch.value}
              onInput$={(event, el) => {
                eventSearch.value = el.value;
              }}
              class="w-full"
            />
            <select
              class="bg-muted/30 border border-border rounded-md px-2 py-2 text-xs"
              value={eventActor.value}
              onChange$={(event) => {
                eventActor.value = (event.target as HTMLSelectElement).value;
              }}
            >
              {actorOptions.value.map((actor) => (
                <option key={actor} value={actor}>{actor}</option>
              ))}
            </select>
            <select
              class="bg-muted/30 border border-border rounded-md px-2 py-2 text-xs"
              value={eventLevel.value}
              onChange$={(event) => {
                eventLevel.value = (event.target as HTMLSelectElement).value;
              }}
            >
              {levelOptions.map((level) => (
                <option key={level} value={level}>{level}</option>
              ))}
            </select>
          </div>
          <div class="flex flex-wrap gap-2">
            {EVENT_PREFIXES.map((prefix) => (
              <Button
                key={prefix}
                variant="secondary"
                onClick$={() => { eventPrefix.value = prefix; }}
                class={`text-xs ${eventPrefix.value === prefix ? 'glass-chip-accent-amber' : ''}`}
              >
                {prefix === 'all' ? 'All topics' : prefix}
              </Button>
            ))}
            <Button
              variant="secondary"
              onClick$={() => { eventErrorsOnly.value = !eventErrorsOnly.value; }}
              class={`text-xs ${eventErrorsOnly.value ? 'glass-chip-accent-magenta' : ''}`}
            >
              Errors only
            </Button>
          </div>
        </div>
        <div class="glass-surface-elevated p-4">
          <NeonTitle level="span" color="amber" size="xs">Sources</NeonTitle>
          <div class="flex flex-wrap gap-2 mt-3 text-[10px]">
            <span class={`px-2 py-1 rounded border ${props.connected.value ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30' : 'bg-red-500/20 text-red-400 border-red-500/30'}`}>
              Bus: {props.connected.value ? 'live' : 'stale'}
            </span>
            <span class={`px-2 py-1 rounded border ${
              taskLedgerStatus.value === 'ok'
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30'
                : taskLedgerStatus.value === 'error'
                ? 'bg-red-500/20 text-red-400 border-red-500/30'
                : 'bg-muted/30 text-muted-foreground border-border/50'
            }`}>
              Task ledger: {taskLedgerStatus.value}
            </span>
            <span class="px-2 py-1 rounded border bg-muted/30 text-muted-foreground border-border/50">
              Events: {snapshot.value.events.length}
            </span>
            <span class="px-2 py-1 rounded border bg-muted/30 text-muted-foreground border-border/50">
              Requests: {props.requests.value.length}
            </span>
            <span class="px-2 py-1 rounded border bg-muted/30 text-muted-foreground border-border/50">
              Ledger: {taskLedger.value.length}
            </span>
            <span class="px-2 py-1 rounded border bg-muted/30 text-muted-foreground border-border/50">
              Agents: {props.agents.value.length}
            </span>
            <span class="px-2 py-1 rounded border bg-muted/30 text-muted-foreground border-border/50">
              Services: {props.services.value.length}
            </span>
          </div>
        </div>
      </div>

      <div class="grid gap-4 lg:grid-cols-4">
        <div class="glass-surface-elevated p-4">
          <div class="text-xs uppercase tracking-[0.25em] text-muted-foreground">Events</div>
          <div class="text-2xl font-semibold text-primary">{snapshot.value.summary.eventCount}</div>
          <div class="text-xs text-muted-foreground">
            {snapshot.value.summary.eventRate.toFixed(1)} / min · filtered {filteredEvents.value.length}
          </div>
        </div>
        <div class="glass-surface-elevated p-4">
          <div class="text-xs uppercase tracking-[0.25em] text-muted-foreground">Tasks</div>
          <div class="text-2xl font-semibold text-primary">{snapshot.value.summary.taskCount}</div>
          <div class="text-xs text-muted-foreground">Mode: {mode.value}</div>
        </div>
        <div class="glass-surface-elevated p-4">
          <div class="text-xs uppercase tracking-[0.25em] text-muted-foreground">Agents</div>
          <div class="text-2xl font-semibold text-primary">{snapshot.value.summary.agentCount}</div>
          <div class="text-xs text-muted-foreground">Services: {snapshot.value.summary.serviceCount}</div>
        </div>
        <div class="glass-surface-elevated p-4">
          <div class="text-xs uppercase tracking-[0.25em] text-muted-foreground">Freshness</div>
          <div class="text-2xl font-semibold text-primary">{formatAge(snapshot.value.summary.lastEventAgeMs)}</div>
          <div class="text-xs text-muted-foreground">Last event age</div>
        </div>
      </div>

      <div class="glass-surface-elevated p-4">
        <div class="flex items-center justify-between">
          <NeonTitle level="span" color="magenta" size="xs">Temporal Spread</NeonTitle>
          <span class="text-xs text-muted-foreground">Windowed counts from anchor</span>
        </div>
        <div class="grid gap-3 mt-3 md:grid-cols-3">
          {windowSummaries.value.map((entry) => (
            <div key={entry.minutes} class="rounded border border-border/50 bg-muted/20 p-3">
              <div class="text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
                {entry.minutes >= 60 ? `${entry.minutes / 60}h` : `${entry.minutes}m`}
              </div>
              <div class="text-lg font-semibold text-primary mt-1">{entry.events}</div>
              <div class="text-[10px] text-muted-foreground">{entry.tasks} tasks</div>
            </div>
          ))}
        </div>
      </div>

      <div class="grid gap-6 lg:grid-cols-[2fr,1fr]">
        <div class="space-y-6">
          <OrchestrationTimeline
            events={timelineEvents.value}
            agents={timelineAgents.value}
            startTime={snapshot.value.window.startIso}
            endTime={snapshot.value.window.endIso}
            rangeMinutes={snapshot.value.window.windowMinutes}
          />
          <div class="grid gap-4 lg:grid-cols-2">
            <TimelineSparkline events={filteredEvents.value} />
            <EventFlowmap events={filteredEvents.value} maxNodes={18} />
          </div>
          <PBLanesWidget events={props.events} />
        </div>

        <div class="space-y-6">
          <div class="glass-surface-elevated p-4">
            <div class="flex items-center justify-between mb-3">
              <NeonTitle level="span" color="amber" size="xs">Tasks</NeonTitle>
              <span class="text-xs text-muted-foreground">{filteredTasks.value.length} filtered</span>
            </div>
            <div class="flex flex-wrap gap-2 mb-3">
              {TASK_FILTERS.map((status) => (
                <Button
                  key={status}
                  variant="secondary"
                  onClick$={() => { taskStatus.value = status; }}
                  class={`text-xs ${taskStatus.value === status ? 'glass-chip-accent-cyan' : ''}`}
                >
                  {status}
                </Button>
              ))}
            </div>
            <div class="flex flex-wrap gap-2 mb-3">
              {TASK_SOURCES.map((source) => (
                <Button
                  key={source}
                  variant="secondary"
                  onClick$={() => { taskSource.value = source; }}
                  class={`text-xs ${taskSource.value === source ? 'glass-chip-accent-amber' : ''}`}
                >
                  {source === 'all' ? 'All sources' : source}
                </Button>
              ))}
            </div>
            <div class="space-y-2">
              {filteredTasks.value.length === 0 && (
                <div class="text-xs text-muted-foreground">No tasks in this window.</div>
              )}
              {filteredTasks.value.slice(0, 8).map((task) => (
                <div key={task.id} class="p-2 rounded border border-border/50 bg-muted/20">
                  <div class="flex items-center justify-between gap-2">
                    <span class="text-xs font-mono">{task.id ? task.id.slice(0, 8) : 'unknown'}</span>
                    <span class={`text-[10px] px-2 py-0.5 rounded border ${statusChip(task.status)}`}>
                      {task.status}
                    </span>
                  </div>
                  <div class="text-xs text-muted-foreground mt-1 truncate">{task.label}</div>
                  <div class="text-[10px] text-muted-foreground mt-1">
                    {task.actor} · {task.source}
                    {task.rawStatus && task.rawStatus !== task.status ? ` (${task.rawStatus})` : ''}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div class="glass-surface-elevated p-4">
            <div class="flex items-center justify-between mb-3">
              <NeonTitle level="span" color="magenta" size="xs">Gaps</NeonTitle>
              <span class="text-xs text-muted-foreground">{snapshot.value.gaps.length} flagged</span>
            </div>
            <div class="space-y-2">
              {snapshot.value.gaps.length === 0 && (
                <div class="text-xs text-muted-foreground">No gaps detected.</div>
              )}
              {snapshot.value.gaps.map((gap) => (
                <div key={gap.id} class="rounded border border-border/50 bg-muted/20 p-2">
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-semibold">{gap.summary}</span>
                    <span class="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">{gap.kind}</span>
                  </div>
                  {gap.evidence && (
                    <div class="text-[10px] text-muted-foreground mt-1">{gap.evidence}</div>
                  )}
                </div>
              ))}
            </div>
          </div>

          <div class="glass-surface-elevated p-4">
            <div class="flex items-center justify-between mb-3">
              <NeonTitle level="span" color="emerald" size="xs">Signal Mix</NeonTitle>
              <span class="text-xs text-muted-foreground">Top segments</span>
            </div>
            <div class="grid gap-3 md:grid-cols-2">
              <div>
                <div class="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Topics</div>
                <div class="mt-2 space-y-1">
                  {topTopics.value.length === 0 && (
                    <div class="text-xs text-muted-foreground">No topic data.</div>
                  )}
                  {topTopics.value.map((entry) => (
                    <div key={entry.key} class="flex items-center justify-between text-xs">
                      <span class="text-muted-foreground">{entry.key}</span>
                      <span class="text-[10px] text-primary">{entry.count}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <div class="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Actors</div>
                <div class="mt-2 space-y-1">
                  {topActors.value.length === 0 && (
                    <div class="text-xs text-muted-foreground">No actor data.</div>
                  )}
                  {topActors.value.map((entry) => (
                    <div key={entry.key} class="flex items-center justify-between text-xs">
                      <span class="text-muted-foreground">{entry.key}</span>
                      <span class="text-[10px] text-primary">{entry.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div class="mt-3">
              <div class="text-[10px] uppercase tracking-[0.2em] text-muted-foreground">Task Status</div>
              <div class="mt-2 flex flex-wrap gap-2">
                {taskStatusMix.value.length === 0 && (
                  <span class="text-xs text-muted-foreground">No task data.</span>
                )}
                {taskStatusMix.value.map((entry) => (
                  <span key={entry.key} class="text-[10px] px-2 py-1 rounded border border-border/50 bg-muted/30 text-muted-foreground">
                    {entry.key}: {entry.count}
                  </span>
                ))}
              </div>
            </div>
          </div>

          <div class="glass-surface-elevated p-4">
            <div class="flex items-center justify-between mb-3">
              <NeonTitle level="span" color="cyan" size="xs">Evidence</NeonTitle>
              <div class="flex items-center gap-2">
                <span class="text-xs text-muted-foreground">{filteredEvidence.value.length} items</span>
                <Button
                  variant="text"
                  onClick$={() => downloadEvidence('json')}
                  class="text-xs"
                >
                  JSON
                </Button>
                <Button
                  variant="text"
                  onClick$={() => downloadEvidence('csv')}
                  class="text-xs"
                >
                  CSV
                </Button>
              </div>
            </div>
            <div class="space-y-2">
              {filteredEvidence.value.length === 0 && (
                <div class="text-xs text-muted-foreground">No evidence events in window.</div>
              )}
              {filteredEvidence.value.slice(0, 6).map((event) => (
                <div key={event.id} class="text-xs text-muted-foreground">
                  {event.topic} · {event.actor}
                </div>
              ))}
            </div>
          </div>

          <div class="glass-surface-elevated p-4">
            <div class="flex items-center justify-between mb-3">
              <NeonTitle level="span" color="amber" size="xs">Provenance</NeonTitle>
              <span class="text-xs text-muted-foreground">snapshot meta</span>
            </div>
            <div class="space-y-2 text-xs text-muted-foreground">
              <div>Mode: {mode.value}</div>
              <div>Window: {snapshot.value.window.startIso} → {snapshot.value.window.endIso}</div>
              <div>Anchor: {new Date(anchorMs.value).toISOString()}</div>
              <div>Event filter: actor={eventActor.value}, level={eventLevel.value}, prefix={eventPrefix.value}</div>
              <div>Errors only: {eventErrorsOnly.value ? 'yes' : 'no'} | Task status: {taskStatus.value}</div>
              <div>Task source: {taskSource.value}</div>
              <div>Task ledger: {taskLedgerStatus.value}{taskLedgerPath.value ? ` (${taskLedgerPath.value})` : ''}</div>
              <div>Task ledger refresh: {taskLedgerUpdatedAt.value || 'pending'}</div>
              <div>Counts: total events {snapshot.value.events.length}, filtered {filteredEvents.value.length}</div>
              <div>Evidence exported: {filteredEvidence.value.length} items</div>
            </div>
          </div>
        </div>
      </div>

      <div class="grid gap-6 lg:grid-cols-2">
        <div class="glass-surface-elevated p-4">
          <NeonTitle level="span" color="emerald" size="xs">Agents</NeonTitle>
          <div class="divide-y divide-border/40 mt-3">
            {props.agents.value.length === 0 && (
              <div class="text-xs text-muted-foreground py-4">No agent telemetry available.</div>
            )}
            {props.agents.value.slice(0, 8).map((agent) => (
              <div key={agent.actor} class="py-2 flex items-center justify-between">
                <div class="text-xs font-mono">{agent.actor}</div>
                <div class="text-[10px] text-muted-foreground">{agent.status}</div>
                <div class="text-[10px] text-muted-foreground">queue {agent.queue_depth}</div>
              </div>
            ))}
          </div>
        </div>
        <div class="glass-surface-elevated p-4">
          <NeonTitle level="span" color="amber" size="xs">Services</NeonTitle>
          <div class="divide-y divide-border/40 mt-3">
            {props.services.value.length === 0 && (
              <div class="text-xs text-muted-foreground py-4">No service registry data.</div>
            )}
            {props.services.value.slice(0, 8).map((service) => (
              <div key={service.id} class="py-2 flex items-center justify-between">
                <div class="text-xs">{service.name}</div>
                <div class="text-[10px] text-muted-foreground">{service.status || 'unknown'}</div>
                <div class="text-[10px] text-muted-foreground">{service.kind}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});
