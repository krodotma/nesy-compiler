/**
 * AgentIOBuffer - Live visualization of agent I/O with MCP/A2A/Dialogos integration
 *
 * Layout:
 * - Top row: Input (green) | Processing (amber) | Output (blue)
 * - Bottom row: A2A Negotiations | Dialogos Cells | MCP Activity
 *
 * Integrates with omega_pairs.json protocol for request/response correlation
 */

import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

interface IOItem {
  id: string;
  ts: number;
  topic: string;
  actor: string;
  kind?: string;
  summary: string;
  pairId?: string;
  reqId?: string;
}

interface A2AEvent {
  id: string;
  ts: number;
  topic: string;
  from: string;
  to: string;
  state: string;
}

interface DialogosCell {
  id: string;
  cellId: string;
  state: string;
  actor: string;
  ts: number;
}

interface MCPEvent {
  id: string;
  ts: number;
  server: string;
  method: string;
  status: string;
}

interface IOBufferData {
  input: IOItem[];
  processing: IOItem[];
  output: IOItem[];
  a2a: A2AEvent[];
  dialogos: DialogosCell[];
  mcp: MCPEvent[];
  metrics: {
    queueDepth: number;
    pendingPairs: number;
    pendingA2a: number;
    inputCount: number;
    outputCount: number;
  };
}

export const AgentIOBuffer = component$(() => {
  const data = useSignal<IOBufferData | null>(null);
  const loading = useSignal(true);
  const error = useSignal<string | null>(null);
  const autoRefresh = useSignal(true);
  const showProtocols = useSignal(true);

  const fetchData = $(async () => {
    try {
      // Use proxy path for HTTPS compatibility
      const res = await fetch('/api/io-buffer');
      if (res.ok) {
        data.value = await res.json();
        error.value = null;
      } else {
        error.value = 'Failed to fetch IO buffer';
      }
    } catch (e) {
      error.value = String(e);
    } finally {
      loading.value = false;
    }
  });

  useVisibleTask$(({ cleanup }) => {
    fetchData();
    const interval = setInterval(() => {
      if (autoRefresh.value) fetchData();
    }, 2500);
    cleanup(() => clearInterval(interval));
  });

  const formatTime = (ts: number) => {
    if (!ts) return '--:--';
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString('en-US', { hour12: false }).slice(0, 8);
  };

  const topicColor = (topic: string) => {
    if (topic.includes('request') || topic.includes('submit') || topic.includes('trigger')) return 'text-green-400';
    if (topic.includes('tick') || topic.includes('probe')) return 'text-cyan-400';
    if (topic.includes('response') || topic.includes('complete') || topic.includes('ack')) return 'text-blue-400';
    if (topic.includes('heartbeat') || topic.includes('state')) return 'text-purple-400';
    if (topic.includes('a2a') || topic.includes('negotiate')) return 'text-pink-400';
    if (topic.includes('dialogos') || topic.includes('cell')) return 'text-orange-400';
    if (topic.includes('mcp')) return 'text-teal-400';
    return 'text-muted-foreground';
  };

  const pairColors: Record<string, string> = {
    infer_sync: 'bg-blue-500/20 text-blue-300',
    dialogos: 'bg-orange-500/20 text-orange-300',
    a2a_negotiate: 'bg-pink-500/20 text-pink-300',
    pluribuscheck: 'bg-green-500/20 text-green-300',
    pbflush: 'bg-purple-500/20 text-purple-300',
  };

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="p-3 border-b border-border flex items-center justify-between bg-muted/30">
        <div class="flex items-center gap-3">
          <span class="text-sm font-semibold">IO BUFFER</span>
          <span class="text-xs text-muted-foreground">MCP/A2A/Dialogos integrated</span>
        </div>
        <div class="flex items-center gap-2">
          {data.value?.metrics && (
            <div class="flex items-center gap-2 text-xs">
              <span class="px-2 py-0.5 rounded bg-green-500/20 text-green-400" title="Queue Depth">
                Q:{data.value.metrics.queueDepth}
              </span>
              <span class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-400" title="Pending Pairs">
                Ω:{data.value.metrics.pendingPairs}
              </span>
              <span class="px-2 py-0.5 rounded bg-pink-500/20 text-pink-400" title="A2A Pending">
                A2A:{data.value.metrics.pendingA2a}
              </span>
            </div>
          )}
          <button
            onClick$={() => { showProtocols.value = !showProtocols.value; }}
            class={`text-xs px-2 py-1 rounded ${showProtocols.value ? 'bg-primary/20 text-primary' : 'bg-muted text-muted-foreground'}`}
          >
            {showProtocols.value ? 'Protocols' : 'Basic'}
          </button>
          <button
            onClick$={() => { autoRefresh.value = !autoRefresh.value; }}
            class={`text-xs px-2 py-1 rounded ${autoRefresh.value ? 'bg-green-500/20 text-green-400' : 'bg-muted text-muted-foreground'}`}
          >
            {autoRefresh.value ? 'Live' : 'Paused'}
          </button>
        </div>
      </div>

      {loading.value ? (
        <div class="p-8 text-center text-muted-foreground">Loading IO buffer...</div>
      ) : error.value ? (
        <div class="p-4 text-red-400 text-sm">{error.value}</div>
      ) : (
        <div class="flex flex-col">
          {/* Main IO Grid */}
          <div class="grid grid-cols-3 gap-0 min-h-[220px]">
            {/* INPUT - Green Section */}
            <div class="border-r border-border">
              <div class="p-2 bg-green-500/10 border-b border-green-500/20 flex items-center justify-between">
                <span class="text-xs font-semibold text-green-400 flex items-center gap-1">
                  <span class="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                  INPUT QUEUE
                </span>
                <span class="text-xs text-green-400/60">{data.value?.input.length || 0}</span>
              </div>
              <div class="max-h-[180px] overflow-auto">
                {data.value?.input.length === 0 ? (
                  <div class="p-3 text-xs text-muted-foreground text-center">No pending input</div>
                ) : (
                  data.value?.input.map((item) => (
                    <div key={item.id} class="p-1.5 border-b border-border/30 hover:bg-green-500/5 transition-colors">
                      <div class="flex items-center justify-between gap-1">
                        <span class={`text-[11px] font-mono truncate ${topicColor(item.topic)}`}>
                          {item.topic.split('.').slice(-2).join('.')}
                        </span>
                        <div class="flex items-center gap-1">
                          {item.pairId && <span class={`text-[10px] px-1 rounded ${pairColors[item.pairId] || 'bg-muted text-muted-foreground'}`}>{item.pairId}</span>}
                          <span class="text-[10px] text-muted-foreground">{formatTime(item.ts)}</span>
                        </div>
                      </div>
                      <div class="text-[10px] text-muted-foreground truncate">{item.summary}</div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* PROCESSING - Amber Section */}
            <div class="border-r border-border">
              <div class="p-2 bg-amber-500/10 border-b border-amber-500/20 flex items-center justify-between">
                <span class="text-xs font-semibold text-amber-400 flex items-center gap-1">
                  <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
                  PROCESSING
                </span>
                <span class="text-xs text-amber-400/60">{data.value?.processing.length || 0}</span>
              </div>
              <div class="max-h-[180px] overflow-auto">
                {data.value?.processing.length === 0 ? (
                  <div class="p-3 text-xs text-muted-foreground text-center">No active processing</div>
                ) : (
                  data.value?.processing.map((item) => (
                    <div key={item.id} class="p-1.5 border-b border-border/30 hover:bg-amber-500/5 transition-colors">
                      <div class="flex items-center justify-between gap-1">
                        <span class={`text-[11px] font-mono truncate ${topicColor(item.topic)}`}>
                          {item.topic.split('.').slice(-2).join('.')}
                        </span>
                        <span class="text-[10px] text-muted-foreground">{formatTime(item.ts)}</span>
                      </div>
                      <div class="flex items-center gap-1 mt-0.5">
                        <span class="text-[10px] px-1 rounded bg-amber-500/20 text-amber-300">{item.actor}</span>
                        <span class="text-[10px] text-muted-foreground truncate flex-1">{item.summary}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* OUTPUT - Blue Section */}
            <div>
              <div class="p-2 bg-blue-500/10 border-b border-blue-500/20 flex items-center justify-between">
                <span class="text-xs font-semibold text-blue-400 flex items-center gap-1">
                  <span class="w-2 h-2 rounded-full bg-blue-400" />
                  OUTPUT STREAM
                </span>
                <span class="text-xs text-blue-400/60">{data.value?.output.length || 0}</span>
              </div>
              <div class="max-h-[180px] overflow-auto">
                {data.value?.output.length === 0 ? (
                  <div class="p-3 text-xs text-muted-foreground text-center">No output yet</div>
                ) : (
                  data.value?.output.map((item) => (
                    <div key={item.id} class="p-1.5 border-b border-border/30 hover:bg-blue-500/5 transition-colors">
                      <div class="flex items-center justify-between gap-1">
                        <span class={`text-[11px] font-mono truncate ${topicColor(item.topic)}`}>
                          {item.topic.split('.').slice(-2).join('.')}
                        </span>
                        <div class="flex items-center gap-1">
                          {item.pairId && <span class={`text-[10px] px-1 rounded ${pairColors[item.pairId] || 'bg-muted text-muted-foreground'}`}>{item.pairId}</span>}
                          <span class="text-[10px] text-muted-foreground">{formatTime(item.ts)}</span>
                        </div>
                      </div>
                      <div class="flex items-center gap-1 mt-0.5">
                        <span class={`text-[10px] px-1 rounded ${
                          item.kind === 'metric' ? 'bg-purple-500/20 text-purple-300' :
                          item.kind === 'artifact' ? 'bg-cyan-500/20 text-cyan-300' :
                          'bg-muted text-muted-foreground'
                        }`}>{item.kind}</span>
                        <span class="text-[10px] text-muted-foreground truncate flex-1">{item.summary}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* Protocol Details Row (toggleable) */}
          {showProtocols.value && (
            <div class="grid grid-cols-3 gap-0 border-t border-border min-h-[140px]">
              {/* A2A Negotiations */}
              <div class="border-r border-border">
                <div class="p-2 bg-pink-500/10 border-b border-pink-500/20 flex items-center justify-between">
                  <span class="text-xs font-semibold text-pink-400">A2A NEGOTIATIONS</span>
                  <span class="text-xs text-pink-400/60">{data.value?.a2a?.length || 0}</span>
                </div>
                <div class="max-h-[100px] overflow-auto">
                  {!data.value?.a2a?.length ? (
                    <div class="p-2 text-[10px] text-muted-foreground text-center">No A2A activity</div>
                  ) : (
                    data.value.a2a.map((evt) => (
                      <div key={evt.id} class="p-1.5 border-b border-border/30 hover:bg-pink-500/5">
                        <div class="flex items-center justify-between">
                          <span class="text-[10px] font-mono text-pink-300">{evt.from} → {evt.to}</span>
                          <span class={`text-[10px] px-1 rounded ${
                            evt.state === 'pending' ? 'bg-yellow-500/20 text-yellow-300' :
                            evt.state === 'accepted' ? 'bg-green-500/20 text-green-300' :
                            'bg-red-500/20 text-red-300'
                          }`}>{evt.state}</span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Dialogos Cells */}
              <div class="border-r border-border">
                <div class="p-2 bg-orange-500/10 border-b border-orange-500/20 flex items-center justify-between">
                  <span class="text-xs font-semibold text-orange-400">DIALOGOS CELLS</span>
                  <span class="text-xs text-orange-400/60">{data.value?.dialogos?.length || 0}</span>
                </div>
                <div class="max-h-[100px] overflow-auto">
                  {!data.value?.dialogos?.length ? (
                    <div class="p-2 text-[10px] text-muted-foreground text-center">No active cells</div>
                  ) : (
                    data.value.dialogos.map((cell) => (
                      <div key={cell.id} class="p-1.5 border-b border-border/30 hover:bg-orange-500/5">
                        <div class="flex items-center justify-between">
                          <span class="text-[10px] font-mono text-orange-300">{cell.cellId}</span>
                          <span class={`text-[10px] px-1 rounded ${
                            cell.state === 'active' ? 'bg-green-500/20 text-green-300 animate-pulse' :
                            cell.state === 'streaming' ? 'bg-blue-500/20 text-blue-300 animate-pulse' :
                            cell.state === 'complete' ? 'bg-gray-500/20 text-gray-300' :
                            'bg-muted text-muted-foreground'
                          }`}>{cell.state}</span>
                        </div>
                        <div class="text-[10px] text-muted-foreground">{cell.actor}</div>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* MCP Activity */}
              <div>
                <div class="p-2 bg-teal-500/10 border-b border-teal-500/20 flex items-center justify-between">
                  <span class="text-xs font-semibold text-teal-400">MCP SERVERS</span>
                  <span class="text-xs text-teal-400/60">{data.value?.mcp?.length || 0}</span>
                </div>
                <div class="max-h-[100px] overflow-auto">
                  {!data.value?.mcp?.length ? (
                    <div class="p-2 text-[10px] text-muted-foreground text-center">No MCP activity</div>
                  ) : (
                    data.value.mcp.map((evt) => (
                      <div key={evt.id} class="p-1.5 border-b border-border/30 hover:bg-teal-500/5">
                        <div class="flex items-center justify-between">
                          <span class="text-[10px] font-mono text-teal-300">{evt.server}</span>
                          <span class={`text-[10px] px-1 rounded ${
                            evt.status === 'ok' ? 'bg-green-500/20 text-green-300' :
                            'bg-red-500/20 text-red-300'
                          }`}>{evt.status}</span>
                        </div>
                        <div class="text-[10px] text-muted-foreground">{evt.method}</div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
});
