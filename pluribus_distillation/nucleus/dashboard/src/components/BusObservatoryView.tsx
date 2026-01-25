import { component$, type QRL, type Signal, useComputed$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { useTracking } from '../lib/telemetry/use-tracking';

import type { AgentStatus, BusEvent, STRpRequest, VPSSession } from '../lib/state/types';
import { AgentIOBuffer } from './AgentIOBuffer';
import { countCheckboxes, parseTaskStatus } from '../lib/task-utils';

type EmitBus = QRL<(topic: string, kind: string, data: Record<string, unknown>) => Promise<void>>;

interface TaskFileSummary {
  name: string;
  path: string;
  size: number;
  status: string;
  pending: number;
  completed: number;
}

interface BusObservatoryViewProps {
  connected: boolean;
  events: Signal<BusEvent[]>;
  agents: Signal<AgentStatus[]>;
  requests: Signal<STRpRequest[]>;
  session: Signal<VPSSession>;
  emitBus$?: EmitBus;
}

export const BusObservatoryView = component$<BusObservatoryViewProps>((props) => {
  const tasks = useSignal<TaskFileSummary[]>([]);
  useTracking("comp:bus-observatory");
  const tasksErr = useSignal<string | null>(null);
  const tasksLoading = useSignal<boolean>(true);
  const repoInfo = useSignal<{ main: string | null; dirtyFiles: number; nested: string[] }>({
    main: null,
    dirtyFiles: 0,
    nested: [],
  });
  const repoErr = useSignal<string | null>(null);
  const repoLoading = useSignal<boolean>(true);

  const domainStats = useComputed$(() => {
    const window = props.events.value.slice(-1500);
    const counts = new Map<string, number>();
    for (const e of window) {
      const domain = String(e.topic || '').split('.')[0] || '?';
      counts.set(domain, (counts.get(domain) || 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 12);
  });

  const actorStats = useComputed$(() => {
    const window = props.events.value.slice(-1500);
    const counts = new Map<string, number>();
    for (const e of window) {
      const actor = String(e.actor || 'unknown');
      counts.set(actor, (counts.get(actor) || 0) + 1);
    }
    return Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 12);
  });

  const latestSignals = useComputed$(() => {
    const latest: Record<string, BusEvent | null> = {
      omega: null,
      providers: null,
      hexis: null,
      strp: null,
      dialogos: null,
      a2a: null,
      mcp: null,
      git: null,
      rhizome: null,
    };

    for (let i = props.events.value.length - 1; i >= 0; i -= 1) {
      const e = props.events.value[i];
      const t = String(e.topic || '');
      if (!latest.omega && t.startsWith('omega.')) latest.omega = e;
      if (!latest.providers && t.startsWith('provider.')) latest.providers = e;
      if (!latest.hexis && t.startsWith('hexis.')) latest.hexis = e;
      if (!latest.strp && t.startsWith('strp.')) latest.strp = e;
      if (!latest.dialogos && t.startsWith('dialogos.')) latest.dialogos = e;
      if (!latest.a2a && t.startsWith('a2a.')) latest.a2a = e;
      if (!latest.mcp && t.startsWith('mcp.')) latest.mcp = e;
      if (!latest.git && t.startsWith('git.')) latest.git = e;
      if (!latest.rhizome && t.startsWith('rhizome.')) latest.rhizome = e;
      const done = Object.values(latest).every((v) => v !== null);
      if (done) break;
    }
    return latest;
  });

  const navigateToEvents = $((searchPattern: string, searchMode: string = 'glob', eventFilter: string | null = null) => {
    try {
      const detail = { view: 'events', searchPattern, searchMode, eventFilter };
      window.dispatchEvent(new CustomEvent('pluribus:navigate', { detail }));
    } catch {
      // no-op
    }
  });

  const requestCollab = $(async () => {
    if (!props.emitBus$) return;
    const reqId = typeof crypto !== 'undefined' && 'randomUUID' in crypto ? crypto.randomUUID() : `req-${Date.now()}`;
    await props.emitBus$('dashboard.bus_observatory.iterate.request', 'request', {
      req_id: reqId,
      ask: 'Propose improvements to Bus Observatory: densest layout, automata/flow views, and the top 10 operator metrics to show.',
      targets: ['dashboard'],
      constraints: { append_only: true, non_blocking: true, tests_first: true },
    });
  });

  const fetchTasks = $(async () => {
    tasksLoading.value = true;
    tasksErr.value = null;
    try {
      const listingRes = await fetch('/api/fs/tasks');
      if (!listingRes.ok) throw new Error(`tasks listing HTTP ${listingRes.status}`);
      const listing = await listingRes.json();
      const entries = Array.isArray(listing?.entries) ? listing.entries : [];
      const taskFiles = entries
        .filter((e: any) => e?.type === 'file' && typeof e?.name === 'string' && e.name.endsWith('.tasks.md'))
        .map((e: any) => ({ name: String(e.name), path: String(e.path || `tasks/${e.name}`), size: Number(e.size || 0) }));

      const summaries: TaskFileSummary[] = [];
      for (const f of taskFiles) {
        try {
          const res = await fetch(`/api/fs/${encodeURIComponent(f.path)}`.replaceAll('%2F', '/'));
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const text = await res.text();
          const { pending, completed } = countCheckboxes(text);
          summaries.push({
            name: f.name,
            path: f.path,
            size: f.size,
            status: parseTaskStatus(text),
            pending,
            completed,
          });
        } catch {
          summaries.push({ name: f.name, path: f.path, size: f.size, status: 'unreadable', pending: 0, completed: 0 });
        }
      }
      tasks.value = summaries.sort((a, b) => {
        if (a.status === b.status) return a.name.localeCompare(b.name);
        if (a.status.toLowerCase().includes('active')) return -1;
        if (b.status.toLowerCase().includes('active')) return 1;
        return a.status.localeCompare(b.status);
      });
    } catch (e) {
      tasksErr.value = String(e);
    } finally {
      tasksLoading.value = false;
    }
  });

  const fetchRepoInfo = $(async () => {
    repoLoading.value = true;
    repoErr.value = null;
    try {
      const res = await fetch('/api/git/status');
      if (!res.ok) throw new Error(`git status HTTP ${res.status}`);
      const data = await res.json();
      const main = data?.root || null;
      const dirty = Array.isArray(data?.dirty) ? data.dirty.length : 0;
      const nested = Array.isArray(data?.nested_repos) ? data.nested_repos : [];
      repoInfo.value = { main, dirtyFiles: dirty, nested };
    } catch (e) {
      repoErr.value = String(e);
    } finally {
      repoLoading.value = false;
    }
  });

  const emitSnapshot = $(async (source: string) => {
    if (!props.emitBus$) return;
    await props.emitBus$('dashboard.bus_observatory.update', 'metric', {
      source,
      iso: new Date().toISOString(),
      tasks_total: tasks.value.length,
      tasks_error: tasksErr.value,
      requests: props.requests.value.length,
      agents: props.agents.value.length,
      repo_dirty: repoInfo.value.dirtyFiles,
    });
  });

  useVisibleTask$(({ cleanup }) => {
    const refresh = async (source: string) => {
      await fetchTasks();
      await fetchRepoInfo();
      await emitSnapshot(source);
    };
    void refresh('init');
    const interval = setInterval(() => {
      void refresh('interval');
    }, 15000);
    cleanup(() => clearInterval(interval));
  });

  return (
    <div class="space-y-4">
      <div class="flex items-center justify-between gap-3">
        <div>
          <h2 class="text-lg font-semibold">Bus Observatory</h2>
          <p class="text-sm text-muted-foreground">
            Evidence • Work • Control plane • Memory • Evolution
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button
            onClick$={requestCollab}
            class="text-xs px-3 py-2 rounded-lg border border-primary/30 bg-primary/15 hover:bg-primary/25 text-primary"
            title="Emit a non-blocking request for other agents to propose Bus Observatory improvements"
          >
            Request Collab
          </button>
          <button
            onClick$={fetchTasks}
            class="text-xs px-3 py-2 rounded-lg border border-border bg-muted/30 hover:bg-muted/50 text-muted-foreground"
          >
            Refresh Tasks
          </button>
        </div>
      </div>

      {/* Dense “Now” signal row */}
      <div class="grid grid-cols-2 md:grid-cols-5 gap-2">
        <div class="rounded-lg border border-border bg-card p-3">
          <div class="text-xs text-muted-foreground">Bus</div>
          <div class="text-lg font-semibold">{props.connected ? 'LIVE' : 'OFFLINE'}</div>
          <div class="text-[11px] text-muted-foreground">{props.events.value.length} events buffered</div>
        </div>
        <div class="rounded-lg border border-border bg-card p-3">
          <div class="text-xs text-muted-foreground">Agents</div>
          <div class="text-lg font-semibold">{props.agents.value.length}</div>
          <div class="text-[11px] text-muted-foreground">pluribus.check.report</div>
        </div>
        <div class="rounded-lg border border-border bg-card p-3">
          <div class="text-xs text-muted-foreground">Requests</div>
          <div class="text-lg font-semibold">{props.requests.value.length}</div>
          <div class="text-[11px] text-muted-foreground">STRp queue (last 100)</div>
        </div>
        <div class="rounded-lg border border-border bg-card p-3">
          <div class="text-xs text-muted-foreground">Providers</div>
          <div class="text-lg font-semibold">
            {Object.values(props.session.value.providers || {}).filter((p) => p.available).length}/{Object.keys(props.session.value.providers || {}).length}
          </div>
          <div class="text-[11px] text-muted-foreground">{props.session.value.activeFallback || 'no active fallback'}</div>
        </div>
        <div class="rounded-lg border border-border bg-card p-3">
          <div class="text-xs text-muted-foreground">Ω</div>
          <div class="text-lg font-semibold">{latestSignals.value.omega ? 'SEEN' : '—'}</div>
          <div class="text-[11px] text-muted-foreground">
            {latestSignals.value.omega?.iso?.slice(11, 19) || 'no omega.* yet'}
          </div>
        </div>
      </div>

      <div class="grid grid-cols-12 gap-4">
        {/* Left: pipelines + domain/actor lenses */}
        <div class="col-span-12 lg:col-span-7 space-y-4">
          <AgentIOBuffer />

          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div class="rounded-lg border border-border bg-card overflow-hidden">
              <div class="p-3 border-b border-border flex items-center justify-between">
                <div class="font-semibold text-sm">Domain Lens</div>
                <button
                  class="text-xs text-muted-foreground hover:text-foreground"
                  onClick$={() => navigateToEvents('**', 'glob', null)}
                >
                  Open Events →
                </button>
              </div>
              <div class="p-3 space-y-2">
                {domainStats.value.map(([domain, count]) => (
                  <button
                    key={domain}
                    onClick$={() => navigateToEvents(`${domain}.*`, 'glob', null)}
                    class="w-full text-left group"
                    title={`Filter events to ${domain}.*`}
                  >
                    <div class="flex items-center justify-between text-xs">
                      <span class="font-mono text-cyan-300">{domain}</span>
                      <span class="text-muted-foreground">{count}</span>
                    </div>
                    <div class="h-1 mt-1 rounded bg-muted/40 overflow-hidden">
                      <div
                        class="h-1 bg-cyan-500/60"
                        style={{ width: `${Math.min(100, Math.round((count / (domainStats.value[0]?.[1] || 1)) * 100))}%` }}
                      />
                    </div>
                  </button>
                ))}
              </div>
            </div>

            <div class="rounded-lg border border-border bg-card overflow-hidden">
              <div class="p-3 border-b border-border flex items-center justify-between">
                <div class="font-semibold text-sm">Actor Lens</div>
                <div class="text-xs text-muted-foreground">click → filter</div>
              </div>
              <div class="p-3 space-y-2">
                {actorStats.value.map(([actor, count]) => (
                  <button
                    key={actor}
                    onClick$={() => navigateToEvents(`@${actor}`, 'actor', null)}
                    class="w-full text-left group"
                    title={`Filter events to actor=${actor}`}
                  >
                    <div class="flex items-center justify-between text-xs">
                      <span class="font-mono text-purple-300 truncate">{actor}</span>
                      <span class="text-muted-foreground">{count}</span>
                    </div>
                    <div class="h-1 mt-1 rounded bg-muted/40 overflow-hidden">
                      <div
                        class="h-1 bg-purple-500/60"
                        style={{ width: `${Math.min(100, Math.round((count / (actorStats.value[0]?.[1] || 1)) * 100))}%` }}
                      />
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div class="rounded-lg border border-border bg-card overflow-hidden">
            <div class="p-3 border-b border-border flex items-center justify-between">
              <div class="font-semibold text-sm">Latest Cross-Lane Signals</div>
              <div class="text-xs text-muted-foreground">ω / providers / hexis / strp / dialogos / a2a / mcp / git</div>
            </div>
            <div class="p-3 grid grid-cols-1 md:grid-cols-2 gap-2">
              {Object.entries(latestSignals.value).map(([k, e]) => (
                <div key={k} class="rounded border border-border/60 bg-muted/20 p-2">
                  <div class="flex items-center justify-between text-xs">
                    <span class="font-mono text-muted-foreground">{k}</span>
                    <span class="font-mono text-muted-foreground/70">{e?.iso?.slice(11, 19) || '—'}</span>
                  </div>
                  <div class="text-[11px] font-mono text-cyan-200 truncate">{e?.topic || 'none'}</div>
                  <div class="text-[11px] text-muted-foreground truncate">
                    @{e?.actor || '—'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: work + tasks + providers */}
        <div class="col-span-12 lg:col-span-5 space-y-4">
          <div class="rounded-lg border border-border bg-card overflow-hidden">
            <div class="p-3 border-b border-border flex items-center justify-between">
              <div class="font-semibold text-sm">Repo Topology</div>
              <div class="text-xs text-muted-foreground">monorepo + nested repos</div>
            </div>
            <div class="p-3 space-y-2">
              {repoLoading.value ? (
                <div class="text-sm text-muted-foreground">Loading…</div>
              ) : repoErr.value ? (
                <div class="text-sm text-red-400">{repoErr.value}</div>
              ) : (
                <>
                  <div class="text-xs text-muted-foreground">Root: {repoInfo.value.main || 'unknown'}</div>
                  <div class="text-xs">
                    <span class="px-2 py-0.5 rounded bg-yellow-500/15 text-yellow-300">dirty: {repoInfo.value.dirtyFiles}</span>
                  </div>
                  <div class="text-xs text-muted-foreground">Nested repos:</div>
                  <div class="text-[11px] space-y-1">
                    {repoInfo.value.nested.length === 0 && <div class="text-muted-foreground">none</div>}
                    {repoInfo.value.nested.map((n) => (
                      <div key={n} class="font-mono text-cyan-200 truncate">{n}</div>
                    ))}
                  </div>
                </>
              )}
            </div>
          </div>

          <div class="rounded-lg border border-border bg-card overflow-hidden">
            <div class="p-3 border-b border-border flex items-center justify-between">
              <div class="font-semibold text-sm">Work Snapshot</div>
              <div class="text-xs text-muted-foreground">present (STRp) + agent health</div>
            </div>
            <div class="p-3 space-y-3">
              <div>
                <div class="text-xs text-muted-foreground mb-2">Latest STRp requests</div>
                <div class="space-y-1">
                  {props.requests.value.slice(0, 8).map((r) => (
                    <button
                      key={r.id}
                      class="w-full text-left rounded border border-border/50 bg-muted/10 hover:bg-muted/30 px-2 py-1"
                      onClick$={() => navigateToEvents(`/req_id:${r.id}`, 'regex', null)}
                      title="Open correlated events (best-effort)"
                    >
                      <div class="flex items-center justify-between text-xs">
                        <span class="font-mono text-cyan-300">{(r.id || '').slice(0, 8)}</span>
                        <span class={`text-[10px] px-1.5 py-0.5 rounded ${
                          r.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                          r.status === 'pending' ? 'bg-yellow-500/20 text-yellow-300' :
                          r.status === 'failed' ? 'bg-red-500/20 text-red-300' :
                          'bg-muted text-muted-foreground'
                        }`}>
                          {r.status}
                        </span>
                      </div>
                      <div class="text-[11px] text-muted-foreground truncate">{r.goal || '-'}</div>
                    </button>
                  ))}
                  {props.requests.value.length === 0 && (
                    <div class="text-xs text-muted-foreground">No STRp requests seen yet.</div>
                  )}
                </div>
              </div>

              <div>
                <div class="text-xs text-muted-foreground mb-2">Agent status</div>
                <div class="space-y-1">
                  {props.agents.value.slice(0, 8).map((a) => (
                    <button
                      key={a.actor}
                      class="w-full text-left rounded border border-border/50 bg-muted/10 hover:bg-muted/30 px-2 py-1"
                      onClick$={() => navigateToEvents(`@${a.actor}`, 'actor', null)}
                      title="Filter events by actor"
                    >
                      <div class="flex items-center justify-between text-xs">
                        <span class="font-mono text-purple-300 truncate">{a.actor}</span>
                        <span class="text-[10px] text-muted-foreground">{a.last_seen_iso?.slice(11, 19) || '—'}</span>
                      </div>
                      <div class="flex items-center justify-between text-[11px] text-muted-foreground">
                        <span class="truncate">{a.current_task || '—'}</span>
                        <span class="font-mono">q:{a.queue_depth}</span>
                      </div>
                    </button>
                  ))}
                  {props.agents.value.length === 0 && (
                    <div class="text-xs text-muted-foreground">No agent reports received yet.</div>
                  )}
                </div>
              </div>
            </div>
          </div>

          <div class="rounded-lg border border-border bg-card overflow-hidden">
            <div class="p-3 border-b border-border flex items-center justify-between">
              <div class="font-semibold text-sm">Tasks (Future Work)</div>
              <div class="text-xs text-muted-foreground">/tasks/*.tasks.md</div>
            </div>
            <div class="p-3">
              {tasksLoading.value ? (
                <div class="text-sm text-muted-foreground">Loading tasks…</div>
              ) : tasksErr.value ? (
                <div class="text-sm text-red-400">{tasksErr.value}</div>
              ) : (
                <div class="space-y-2">
                  {tasks.value.map((t) => (
                    <div key={t.path} class="rounded border border-border/50 bg-muted/10 p-2">
                      <div class="flex items-center justify-between text-xs">
                        <span class="font-mono text-cyan-200 truncate">{t.name}</span>
                        <span class="text-muted-foreground">{t.status}</span>
                      </div>
                      <div class="flex items-center gap-2 text-[11px] mt-1">
                        <span class="px-2 py-0.5 rounded bg-yellow-500/15 text-yellow-300">todo:{t.pending}</span>
                        <span class="px-2 py-0.5 rounded bg-green-500/15 text-green-300">done:{t.completed}</span>
                        <a
                          class="ml-auto text-muted-foreground hover:text-foreground underline decoration-white/10"
                          href={`/api/fs/${t.path}`}
                          target="_blank"
                          rel="noreferrer"
                        >
                          open
                        </a>
                      </div>
                    </div>
                  ))}
                  {tasks.value.length === 0 && (
                    <div class="text-sm text-muted-foreground">No task files found.</div>
                  )}
                </div>
              )}
            </div>
          </div>

          <div class="rounded-lg border border-border bg-card overflow-hidden">
            <div class="p-3 border-b border-border">
              <div class="font-semibold text-sm">Providers (Control Plane)</div>
              <div class="text-xs text-muted-foreground">availability + block reasons</div>
            </div>
            <div class="p-3 space-y-2">
              {Object.entries(props.session.value.providers || {}).map(([name, p]) => (
                <div key={name} class="rounded border border-border/50 bg-muted/10 p-2">
                  <div class="flex items-center justify-between text-xs">
                    <span class="font-mono">{name}</span>
                    <span class={`text-[10px] px-1.5 py-0.5 rounded ${
                      p.available ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-300'
                    }`}>
                      {p.available ? 'available' : 'blocked'}
                    </span>
                  </div>
                  <div class="text-[11px] text-muted-foreground truncate">{p.model || '—'}</div>
                  {p.error && <div class="text-[11px] text-red-300/80 truncate">{p.error}</div>}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});
