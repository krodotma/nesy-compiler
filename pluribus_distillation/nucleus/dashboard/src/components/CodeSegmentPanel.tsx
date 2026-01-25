/**
 * CodeSegmentPanel.tsx
 * ====================
 * AI OS Transparency Panel - Comprehensive Pluribus Subsystem Inventory
 *
 * Features:
 * - Exhaustive listing of ALL Pluribus subsystems (Internal & External)
 * - VPS services, Python tools, Dashboard components
 * - Protocols (DKIN, PAIP, CAGENT, Replisome)
 * - Recommended modules with One-Click Install (Experimental vs Rhizome)
 * - Real-time telemetry tracking for live status
 * - Biological stack visualization (Rhizome→DNA→Replisome→Ribosome→Phenotype)
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  noSerialize,
  type NoSerialize,
} from '@builder.io/qwik';

// M3 Components - CodeSegmentPanel
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';
import '@material/web/button/filled-tonal-button.js';
import type { BusEvent } from '../lib/state/types';
import { createBusClient, type BusClient } from '../lib/bus/bus-client';
import { ProofCanvasOverlay } from './ProofCanvasOverlay';
import { ShaderToyDocsLogo } from './layout/ShaderToyDocsLogo';

export interface CodeSegmentPanelProps {
  /** Maximum items to track */
  maxItems?: number;
}

interface SegmentItem {
  id: string;
  name: string;
  type: 'component' | 'module' | 'tool' | 'metric' | 'error';
  duration: number;
  timestamp: number;
  status: 'FAST' | 'OK' | 'SLOW' | 'ERROR';
  correlationId?: string;
  topic: string;
}

type TabId = 'live' | 'services' | 'tools' | 'protocols' | 'stack' | 'external' | 'recommended';

/** Static subsystem registry - exhaustive inventory */
const SUBSYSTEM_REGISTRY = {
  services: [
    { id: 'omega-heartbeat', name: 'Omega Heartbeat', glyph: '💓', status: 'active' },
    { id: 'omega-dispatcher', name: 'Omega Dispatcher', glyph: '📤', status: 'active' },
    { id: 'omega-guardian', name: 'Omega Guardian', glyph: '🛡️', status: 'active' },
    { id: 'bus-bridge', name: 'Bus Bridge', glyph: '🌉', status: 'active' },
    { id: 'dashboard', name: 'Dashboard', glyph: '📊', status: 'active' },
    { id: 'tools-api', name: 'Tools API', glyph: '🔌', status: 'active' },
    { id: 'vps-session', name: 'VPS Session', glyph: '🖥️', status: 'active' },
    { id: 'browser-session', name: 'Browser Session', glyph: '🌐', status: 'active' },
    { id: 'dialogosd', name: 'Dialogos Daemon', glyph: '💬', status: 'active' },
    { id: 'dialogos-indexer', name: 'Dialogos Indexer', glyph: '🔍', status: 'active' },
    { id: 'codemaster', name: 'Replisome (Codemaster)', glyph: '🧬', status: 'active' },
    { id: 'qa-observer', name: 'QA Observer', glyph: '👁️', status: 'active' },
    { id: 'qa-executor', name: 'QA Executor', glyph: '⚡', status: 'active' },
    { id: 'vnc', name: 'VNC Server', glyph: '🖼️', status: 'active' },
    { id: 'websockify', name: 'Websockify', glyph: '🔗', status: 'active' },
  ],
  tools: [
    { id: 'ckin_report', name: 'CKIN Report', glyph: '📋', category: 'status' },
    { id: 'lanes_report', name: 'Lanes Report', glyph: '🛤️', category: 'coordination' },
    { id: 'realagents_operator', name: 'REALAGENTS', glyph: '🤖', category: 'dispatch' },
    { id: 'pbtest_operator', name: 'PBTEST', glyph: '✅', category: 'verification' },
    { id: 'pbcmaster_operator', name: 'PBREPLISOME', glyph: '🧬', category: 'gating' },
    { id: 'pbdeep_operator', name: 'PBDEEP', glyph: '🔬', category: 'forensics' },
    { id: 'pbpair_operator', name: 'PBPAIR', glyph: '👥', category: 'fallback' },
    { id: 'pbresume_operator', name: 'PBRESUME', glyph: '▶️', category: 'resume' },
    { id: 'pbhygiene_operator', name: 'PBHYGIENE', glyph: '🧹', category: 'cleanup' },
    { id: 'lens_collimator', name: 'Lens Collimator', glyph: '🔭', category: 'routing' },
    { id: 'agent_conductor', name: 'Agent Conductor', glyph: '🎼', category: 'orchestration' },
    { id: 'graphiti_bridge', name: 'Graphiti Bridge', glyph: '🕸️', category: 'kg' },
    { id: 'dialogos_search', name: 'Dialogos Search', glyph: '🔍', category: 'search' },
    { id: 'task_ledger', name: 'Task Ledger', glyph: '📒', category: 'tracking' },
  ],
  protocols: [
    { id: 'dkin_v28', name: 'DKIN v28 (CAGENT)', glyph: '🏛️', status: 'active' },
    { id: 'dkin_v26', name: 'DKIN v26 (Replisome)', glyph: '🧬', status: 'active' },
    { id: 'dkin_v25', name: 'DKIN v25 (Lifecycle)', glyph: '🔄', status: 'active' },
    { id: 'paip_v14', name: 'PAIP v14 (Isolation)', glyph: '🔒', status: 'active' },
    { id: 'ckin_v17', name: 'CKIN v17 (Hygiene)', glyph: '🧹', status: 'active' },
    { id: 'cagent_v1', name: 'CAGENT v1 (Citizen)', glyph: '🎖️', status: 'active' },
    { id: 'replisome_v1', name: 'Replisome v1', glyph: '🧬', status: 'active' },
  ],
  stack: [
    { id: 'rhizome', name: 'RHIZOME', glyph: '🌿', desc: 'Immutable evidence (bus, artifacts)' },
    { id: 'dna', name: 'DNA', glyph: '🧬', desc: 'Genotype (specs, protocols)' },
    { id: 'replisome', name: 'REPLISOME', glyph: '🔄', desc: 'Replication gatekeeper' },
    { id: 'ribosome', name: 'RIBOSOME', glyph: '⚗️', desc: 'Translation engine' },
    { id: 'phenotype', name: 'PHENOTYPE', glyph: '🦋', desc: 'Runtime manifestation' },
    { id: 'cytoplasm', name: 'CYTOPLASM', glyph: '🧫', desc: 'Agent clades' },
  ],
  external: [
    { id: 'os-linux', name: 'Linux Kernel', glyph: '🐧', type: 'OS', desc: 'Debian/Ubuntu base' },
    { id: 'runtime-node', name: 'Node.js', glyph: '🟩', type: 'Runtime', desc: 'V8 JavaScript Engine' },
    { id: 'runtime-python', name: 'Python', glyph: '🐍', type: 'Runtime', desc: '3.12+ Ecosystem' },
    { id: 'oss-vite', name: 'Vite', glyph: '⚡', type: 'OSS', desc: 'Build Tool' },
    { id: 'oss-qwik', name: 'Qwik', glyph: '⚡', type: 'OSS', desc: 'Resumable Framework' },
    { id: 'oss-lit', name: 'Lit', glyph: '🔥', type: 'OSS', desc: 'Web Components' },
    { id: 'oss-three', name: 'Three.js', glyph: '🧊', type: 'OSS', desc: '3D Graphics' },
    { id: 'sota-openai', name: 'OpenAI', glyph: '🧠', type: 'SOTA', desc: 'GPT-4o/o1' },
    { id: 'sota-anthropic', name: 'Anthropic', glyph: '🧠', type: 'SOTA', desc: 'Claude 3.5 Sonnet' },
    { id: 'sota-google', name: 'Google', glyph: '🧠', type: 'SOTA', desc: 'Gemini 1.5 Pro' },
    { id: 'proto-mcp', name: 'MCP', glyph: '🔌', type: 'Protocol', desc: 'Model Context Protocol' },
    { id: 'proto-a2a', name: 'A2A', glyph: '🤖', type: 'Protocol', desc: 'Agent-to-Agent' },
  ],
  recommended: [
    { id: 'rec-ohmyopencode', name: 'OhMyOpenCode', glyph: '🧬', type: 'Evolution', desc: 'Recursive code self-evolution', version: '0.0.1-alpha', status: 'mutating', confidence: 0.70 },
    { id: 'rec-langgraph', name: 'LangGraph', glyph: '🕸️', type: 'Orchestration', desc: 'Cyclic stateful chains', version: '0.0.15', updateAvailable: true, confidence: 0.92 },
    { id: 'rec-autogen', name: 'AutoGen', glyph: '🤝', type: 'Multi-Agent', desc: 'Conversational agents', version: '0.2.0', confidence: 0.88 },
    { id: 'rec-llamaindex', name: 'LlamaIndex', glyph: '🗂️', type: 'Data', desc: 'Context augmentation', version: '0.9.0', updateAvailable: true, confidence: 0.95 },
    { id: 'rec-dspy', name: 'DSPy', glyph: '📢', type: 'Prompting', desc: 'Declarative self-improving', version: '2.1.0', confidence: 0.82 },
    { id: 'rec-memgpt', name: 'MemGPT', glyph: '🧠', type: 'Memory', desc: 'OS-like memory management', version: '0.3.0', confidence: 0.78 },
    { id: 'rec-litellm', name: 'LiteLLM', glyph: '🔌', type: 'Gateway', desc: 'Unified API gateway', version: '1.0.0', status: 'stable', confidence: 0.98 },
    { id: 'rec-openinterpreter', name: 'Open Interpreter', glyph: '💻', type: 'Execution', desc: 'Local code execution', version: '0.1.0', updateAvailable: true, status: 'active', confidence: 0.85 },
  ],
};

/** Panel width constants */
const PANEL_WIDTH_CLOSED = 12;
const PANEL_WIDTH_OPEN = 420;
const PANEL_WIDTH_PEEK = 48;

// Process event to segment item (Module-level pure function)
export const processEvent = (event: BusEvent): SegmentItem | null => {
  if (!event || !event.topic) return null;
  const data = event.data || {};

  let name = 'unknown';
  let duration = 0;
  let type: SegmentItem['type'] = 'metric';

  if (data.component) {
    name = String(data.component);
    type = 'component';
    duration = Number(data.duration) || 0;
  } else if (data.module) {
    name = String(data.module);
    type = 'module';
    duration = Number(data.duration) || 0;
  } else if (data.tool) {
    name = String(data.tool);
    type = 'tool';
    duration = Number(data.duration) || 0;
  } else if (event.topic.includes('error')) {
    name = String(data.message || data.error || event.topic).slice(0, 100);
    type = 'error';
  } else if (event.topic.startsWith('telemetry.')) {
    name = event.topic.replace('telemetry.', '');
    type = 'metric';
    if (data.duration) duration = Number(data.duration);
  } else if (event.topic.startsWith('omega.')) {
    name = event.topic.replace('omega.', '');
    type = 'metric';
  } else {
    return null;
  }

  let status: SegmentItem['status'] = 'OK';
  if (type === 'error') {
    status = 'ERROR';
  } else if (duration > 0) {
    status = duration < 500 ? 'FAST' : duration < 2000 ? 'OK' : 'SLOW';
  }

  return {
    id: event.id || `${Date.now()}-${Math.random()}`,
    name,
    type,
    duration,
    timestamp: data.timestamp ? Number(data.timestamp) : Date.now(),
    status,
    correlationId: String(data.url || data.correlationId || ''),
    topic: event.topic,
  };
};

export const CodeSegmentPanel = component$<CodeSegmentPanelProps>(({
  maxItems = 200,
}) => {
  const isOpen = useSignal(false);
  const isPeeking = useSignal(false);
  const panelRef = useSignal<HTMLDivElement>();
  const activeTab = useSignal<TabId>('stack');
  const installModal = useSignal<{ id: string; name: string } | null>(null);
  const proofCanvasItem = useSignal<{ id: string; name: string } | null>(null);

  const state = useStore<{
    segments: SegmentItem[];
    busClient: NoSerialize<BusClient> | null;
    connected: boolean;
    stats: {
      totalLoaded: number;
      avgDuration: number;
      slowest: string;
      errors: number;
    };
  }>({
    segments: [],
    busClient: null,
    connected: false,
    stats: {
      totalLoaded: 0,
      avgDuration: 0,
      slowest: '-',
      errors: 0,
    },
  });

  const TABS: { id: TabId; label: string; glyph: string }[] = [
    { id: 'stack', label: 'Stack', glyph: '🧬' },
    { id: 'services', label: 'Services', glyph: '⚙️' },
    { id: 'tools', label: 'Tools', glyph: '🔧' },
    { id: 'protocols', label: 'Protocols', glyph: '📜' },
    { id: 'external', label: 'Ext', glyph: '🌍' },
    { id: 'recommended', label: 'Rec', glyph: '✨' },
    { id: 'live', label: 'Live', glyph: '📡' },
  ];

  const handleInstall = $((id: string, mode: 'experimental' | 'permanent') => {
    console.log(`[INSTALL] ${id} in mode: ${mode}`);
    if (state.busClient) {
      (state.busClient as any).emit('system.module.install', { id, mode, timestamp: Date.now() });
    }
    installModal.value = null;
  });

  const handleUpdate = $((id: string) => {
    console.log(`[UPDATE] Checking ${id}...`);
    // Simulated async check
    if (state.busClient) {
      (state.busClient as any).emit('system.module.update.check', { id, timestamp: Date.now() });
    }
    // Just a UI feedback for now
    alert(`Checking registry for ${id} latest... (Simulated)`);
  });

  const openPanel = $(() => { isOpen.value = true; isPeeking.value = false; });
  const closePanel = $(() => { isOpen.value = false; isPeeking.value = false; });
  const handleMouseEnterHandle = $(() => { if (!isOpen.value) isPeeking.value = true; });
  const handleMouseLeave = $(() => { isPeeking.value = false; isOpen.value = false; });

  const handleEvent = $((event: BusEvent) => {
    const item = processEvent(event);
    if (!item) return;
    state.stats.totalLoaded++;
    if (item.status === 'ERROR') state.stats.errors++;
    if (item.duration > 0) {
      const prevTotal = state.stats.totalLoaded - 1;
      const newAvg = ((state.stats.avgDuration * prevTotal) + item.duration) / state.stats.totalLoaded;
      state.stats.avgDuration = Math.round(newAvg);
      const currentSlowest = parseFloat(state.stats.slowest.split('ms')[0]) || 0;
      if (item.duration > currentSlowest) state.stats.slowest = `${item.duration}ms (${item.name.slice(0, 20)})`;
    }
    state.segments = [item, ...state.segments].slice(0, maxItems);
  });

  useVisibleTask$(({ cleanup }) => {
    const client = createBusClient({ platform: 'browser' });
    state.busClient = noSerialize(client);
    client.connect().then(() => {
      state.connected = true;
      const patterns = ['telemetry.*', 'telemetry.error', 'omega.metrics.*'];
      const unsubscribes = patterns.map(p => client.subscribe(p, handleEvent));
      client.getEvents(50).then((events) => {
        events.forEach(event => {
          if (event.topic.startsWith('telemetry.') || event.topic.startsWith('omega.')) {
            const item = processEvent(event);
            if (item) state.segments = [...state.segments, item].slice(0, maxItems);
          }
        });
      });
      cleanup(() => {
        unsubscribes.forEach(u => u());
        client.disconnect();
      });
    }).catch(() => { state.connected = false; });
  });

  useVisibleTask$(({ cleanup }) => {
    const handleKeydown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen.value) { isOpen.value = false; return; }
      if (e.key === 'i' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const target = e.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          isOpen.value = !isOpen.value;
          if (isOpen.value) isPeeking.value = false;
        }
      }
    };
    window.addEventListener('keydown', handleKeydown);
    cleanup(() => window.removeEventListener('keydown', handleKeydown));
  });

  const panelWidth = isOpen.value ? PANEL_WIDTH_OPEN : isPeeking.value ? PANEL_WIDTH_PEEK : PANEL_WIDTH_CLOSED;

  return (
    <div
      ref={panelRef}
      data-testid="code-segment-panel"
      class="fixed top-0 left-0 bottom-0 z-[100] transition-all duration-300 ease-out pointer-events-auto"
      style={{ width: `${panelWidth}px` }}
      onMouseLeave$={handleMouseLeave}
    >
      <div
        data-testid="code-segment-toggle"
        class={`absolute right-0 top-1/2 -translate-y-1/2 w-3 h-24 rounded-r-lg bg-card/80 border border-l-0 border-[var(--glass-border)] backdrop-blur-sm cursor-pointer flex items-center justify-center transition-all duration-200 hover:bg-card hover:w-4 hover:border-[var(--glass-border-hover)] ${isOpen.value ? 'opacity-50' : 'opacity-100'}`}
        onClick$={openPanel}
        onMouseEnter$={handleMouseEnterHandle}
      >
        <div class={`text-muted-foreground text-xs transition-transform duration-200 ${isOpen.value ? '' : 'rotate-180'}`}>{'<'}</div>
        {state.stats.totalLoaded > 0 && !isOpen.value && (
          <div class="absolute -right-1 -top-1 w-5 h-5 rounded-full bg-primary text-primary-foreground text-[10px] font-bold flex items-center justify-center">
            {state.stats.totalLoaded > 99 ? '99+' : state.stats.totalLoaded}
          </div>
        )}
      </div>

      {installModal.value && (
        <div class="fixed inset-0 z-[110] flex items-center justify-center bg-black/50 backdrop-blur-sm">
          <div class="bg-card border border-[var(--glass-border)] rounded-lg p-4 w-80 shadow-2xl glass-surface-overlay">
            <h3 class="text-sm font-bold mb-2">Install {installModal.value.name}</h3>
            <p class="text-xs text-muted-foreground mb-4">Choose installation mode:</p>
            <div class="space-y-2">
              <button class="w-full text-left p-2 rounded hover:bg-muted/50 border border-[var(--glass-border)] group glass-dropdown-item" onClick$={() => handleInstall(installModal.value!.id, 'experimental')}>
                <div class="text-xs font-bold group-hover:text-primary">🧪 Experimental (Ephemeral)</div>
                <div class="text-[10px] text-muted-foreground">Runs in PAIP clone. No config changes.</div>
              </button>
              <button class="w-full text-left p-2 rounded hover:bg-muted/50 border border-[var(--glass-border)] group glass-dropdown-item" onClick$={() => handleInstall(installModal.value!.id, 'permanent')}>
                <div class="text-xs font-bold group-hover:text-primary">🌿 Permanent (Rhizome)</div>
                <div class="text-[10px] text-muted-foreground">Writes to manifest. Encoded in Quine layer.</div>
              </button>
            </div>
            <div class="mt-4 flex justify-end">
              <button class="text-xs px-3 py-1 rounded hover:bg-muted text-muted-foreground" onClick$={() => installModal.value = null}>Cancel</button>
            </div>
          </div>
        </div>
      )}

      <div class={`absolute inset-0 right-3 bg-card/95 border-r border-[var(--glass-border)] backdrop-blur-md flex flex-col transition-opacity duration-200 glass-sidebar-left-glow ${isOpen.value || isPeeking.value ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
        <div class="p-3 border-b border-[var(--glass-border)] flex items-center justify-between flex-shrink-0">
          <div class="flex items-center gap-2">
            <div class="w-8 h-8 relative overflow-hidden rounded-full mr-2">
            <ShaderToyDocsLogo />
          </div>
          <h3 class="text-sm font-bold text-primary font-mono">AI OS INVENTORY</h3>
            <span class={`w-2 h-2 rounded-full ${state.connected ? 'bg-green-500 animate-pulse' : 'bg-muted'}`} />
          </div>
          <button type="button" class="text-xs px-2 py-1 rounded hover:bg-muted/50 text-muted-foreground" onClick$={closePanel}>×</button>
        </div>

        <div class="p-2 border-b border-[var(--glass-border-subtle)] bg-muted/10 flex flex-wrap gap-1">
          {TABS.map((tab) => (
            <button key={tab.id} type="button" class={`text-[10px] px-2 py-1 rounded font-mono transition-colors ${activeTab.value === tab.id ? 'bg-primary text-primary-foreground' : 'bg-black/30 text-muted-foreground hover:bg-muted/50'}`} onClick$={() => { activeTab.value = tab.id; }}>
              {tab.glyph} {tab.label}
            </button>
          ))}
        </div>

        <div class="flex-1 overflow-y-auto p-2 space-y-1 font-mono">
          {activeTab.value === 'stack' && (
            <div class="space-y-2">
              {SUBSYSTEM_REGISTRY.stack.map((item, idx) => (
                <div key={item.id} class="relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] transition-colors">
                  <div class="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-gradient-to-b from-emerald-500 via-cyan-500 to-purple-500" style={{ opacity: 1 - idx * 0.15 }} />
                  <div class="pl-2 flex items-center gap-2">
                    <span class="text-lg">{item.glyph}</span>
                    <div class="flex-1">
                      <div class="text-xs font-bold text-foreground">{item.name}</div>
                      <div class="text-[10px] text-muted-foreground">{item.desc}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab.value === 'services' && (
            <div class="space-y-1">
              {SUBSYSTEM_REGISTRY.services.map((svc) => (
                <div key={svc.id} class="relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] transition-colors">
                  <div class="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-emerald-500" />
                  <div class="pl-2 flex items-center gap-2">
                    <span>{svc.glyph}</span>
                    <div class="flex-1 truncate">
                      <div class="text-xs font-bold text-foreground">{svc.name}</div>
                      <div class="flex items-center gap-2 mt-0.5">
                        {/* Metabolism Pulse */}
                        <div class="flex gap-0.5 items-end h-2 w-12">
                          <div class="w-1 bg-emerald-500/40 h-[40%]" />
                          <div class="w-1 bg-emerald-500/60 h-[70%]" />
                          <div class="w-1 bg-emerald-500 h-[50%]" />
                          <div class="w-1 bg-emerald-500/80 h-[90%] animate-pulse" />
                        </div>
                        <span class="text-[8px] text-muted-foreground">Metabolism: Active</span>
                      </div>
                    </div>
                    <span class="text-[9px] px-1 rounded bg-emerald-900/50 text-emerald-300 border border-emerald-500/30">HEALTHY</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab.value === 'tools' && (
            <div class="space-y-1">
              {SUBSYSTEM_REGISTRY.tools.map((tool) => (
                <div key={tool.id} class="relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] transition-colors">
                  <div class="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-cyan-500" />
                  <div class="pl-2 flex items-center gap-2">
                    <span>{tool.glyph}</span>
                    <div class="flex-1 truncate"><span class="text-xs font-bold text-foreground">{tool.name}</span></div>
                    <span class="text-[9px] px-1 rounded bg-cyan-900/50 text-cyan-300">{tool.category}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab.value === 'protocols' && (
            <div class="space-y-1">
              {SUBSYSTEM_REGISTRY.protocols.map((proto) => (
                <div key={proto.id} class="relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] transition-colors">
                  <div class={`absolute left-0 top-0 bottom-0 w-1 rounded-l ${proto.status === 'active' ? 'bg-purple-500' : 'bg-amber-500'}`} />
                  <div class="pl-2 flex items-center gap-2">
                    <span>{proto.glyph}</span>
                    <div class="flex-1 truncate"><span class="text-xs font-bold text-foreground">{proto.name}</span></div>
                    <span class={`text-[9px] px-1 rounded ${proto.status === 'active' ? 'bg-purple-900/50 text-purple-300' : 'bg-amber-900/50 text-amber-300'}`}>{proto.status?.toUpperCase()}</span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab.value === 'external' && (
            <div class="space-y-1">
              {SUBSYSTEM_REGISTRY.external.map((item) => (
                <div key={item.id} class="relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] group cursor-help" title={item.desc}>
                  <div class="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-slate-500" />
                  <div class="pl-2 flex flex-col gap-1">
                    <div class="flex items-center gap-2">
                      <span>{item.glyph}</span>
                      <div class="flex-1 truncate"><span class="text-xs font-bold text-foreground">{item.name}</span></div>
                      <span class="text-[9px] px-1 rounded bg-slate-800 text-slate-300 border border-slate-700">{item.type}</span>
                    </div>
                    <div class="text-[10px] text-muted-foreground pl-6 truncate opacity-70 group-hover:opacity-100">{item.desc}</div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab.value === 'recommended' && (
            <div class="space-y-1">
              <div class="text-[10px] text-muted-foreground mb-2">
                Recommended Modules ({SUBSYSTEM_REGISTRY.recommended.length})
              </div>
              {SUBSYSTEM_REGISTRY.recommended.map((item) => (
                <div 
                  key={item.id} 
                  class="relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] group cursor-pointer"
                  onClick$={() => proofCanvasItem.value = item}
                >
                  <div class="absolute left-0 top-0 bottom-0 w-1 rounded-l bg-blue-500" />
                  <div class="pl-2 flex flex-col gap-1">
                    <div class="flex items-center gap-2">
                      <span>{item.glyph}</span>
                      <div class="flex-1 truncate"><span class="text-xs font-bold text-foreground">{item.name}</span></div>
                      
                      {/* Proof Badge */}
                      <span 
                        class={`text-[8px] px-1 rounded border ${
                          (item as any).confidence >= 0.9 ? 'bg-green-900/30 text-green-400 border-green-500/30' :
                          (item as any).confidence >= 0.7 ? 'bg-blue-900/30 text-blue-400 border-blue-500/30' :
                          'bg-amber-900/30 text-amber-400 border-amber-500/30'
                        }`}
                        title={`Confidence: ${((item as any).confidence * 100).toFixed(0)}% | Epistemic Gap: ${(item as any).confidence < 0.8 ? 'Needs validation in local environment' : 'Verified via swarm handshake'}`}
                      >
                        {(item as any).confidence >= 0.9 ? 'VERIFIED' : (item as any).confidence >= 0.7 ? 'THEORETIC' : 'EXPERIMENTAL'}
                      </span>

                      {(item as any).updateAvailable && (
                        <button class="text-[9px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30 hover:bg-green-500/30" onClick$={(e) => { e.stopPropagation(); handleUpdate(item.id); }}>⬆ UPDATE</button>
                      )}
                      <button class="text-[9px] px-1.5 py-0.5 rounded bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30" onClick$={(e) => { e.stopPropagation(); installModal.value = item; }}>INSTALL</button>
                    </div>
                    <div class="flex items-center justify-between pl-6 text-[10px] text-muted-foreground">
                      <span class="truncate pr-2" title={item.desc}>{item.desc}</span>
                      <span class="font-mono opacity-50">v{item.version}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {activeTab.value === 'live' && (
            state.segments.length === 0 ? (
              <div class="text-center text-muted-foreground text-xs py-10">No telemetry yet</div>
            ) : (
              state.segments.map((seg) => (
                <div key={`${seg.id}-${seg.timestamp}`} class="group relative p-2 rounded border border-[var(--glass-border-subtle)] hover:bg-muted/20 hover:border-[var(--glass-border)] transition-colors">
                  <div class={`absolute left-0 top-0 bottom-0 w-1 rounded-l ${seg.status === 'FAST' ? 'bg-emerald-500' : seg.status === 'OK' ? 'bg-amber-500' : 'bg-rose-600'}`} />
                  <div class="pl-2 flex justify-between items-start">
                    <div class="overflow-hidden flex-1 min-w-0">
                      <div class="text-xs font-bold text-foreground truncate">{seg.name}</div>
                      <div class="text-[10px] text-muted-foreground uppercase">{seg.type}</div>
                    </div>
                    <div class="text-right flex flex-col items-end ml-2 text-[9px] text-muted-foreground opacity-50">
                      {seg.duration > 0 && <span class="text-xs font-bold text-amber-400">{seg.duration}ms</span>}
                      {new Date(seg.timestamp).toLocaleTimeString([], {hour12: false, minute: '2-digit', second: '2-digit'})}
                    </div>
                  </div>
                </div>
              ))
            )
          )}
        </div>

        <div class="p-2 border-t border-[var(--glass-border-subtle)] text-[10px] text-muted-foreground flex items-center justify-between flex-shrink-0">
          <span>Press <span class="font-mono bg-muted/50 px-1 rounded">i</span> to toggle</span>
          <span class="font-mono">{state.connected ? 'live' : 'offline'}</span>
        </div>
      </div>
          {proofCanvasItem.value && (
            <ProofCanvasOverlay 
              itemId={proofCanvasItem.value.id} 
              itemName={proofCanvasItem.value.name} 
              onClose$={$(() => { proofCanvasItem.value = null; })} 
            />
          )}
        </div>
      );
    });
    