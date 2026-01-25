/**
 * Pluribus Dashboard - Isomorphic Control Center
 *
 * Mirrors TUI functionality with rich interactive panels:
 * - Flow Mode + Provider Status
 * - Event Log with filtering
 * - Agent Status (VOR metrics)
 * - Pending Requests
 * - SOTA Catalog (curated tools/sources)
 * - Command Center
 */

import { component$, useSignal, useStore, useVisibleTask$, useTask$, useComputed$, useContext, $, noSerialize, type NoSerialize } from '@builder.io/qwik';
import { type DocumentHead, useLocation } from '@builder.io/qwik-city';
import type { VPSSession, ServiceDef, BusEvent, AgentStatus, STRpRequest } from '../lib/state/types';
import type { ActionCell, ActionRequest, ActionOutput, ActionResult, InferCellActionPayload } from '../lib/actions/types';
import { createServiceActionHandlers, type BusEmitter } from '../lib/actions/serviceActions';
import { createBusClient } from '../lib/bus/bus-client';
import { OutputCell } from '../lib/actions/OutputCell';
// Phase 2 Optimization: Heavy components lazy-loaded to improve initial page load
// Terminal (xterm.js 278KB), GitView (41KB), GenerativeCanvas (three.js 673KB)
import { LazyTerminal, LazyPluriChatTerminal } from '../components/LazyTerminal';
import { LazyGitView } from '../components/LazyGitView';
import { LazyGenerativeCanvas } from '../components/LazyGenerativeCanvas';
// WebLLMWidget loaded lazily via LazyWebLLM to defer 5.3MB bundle
import { LazyWebLLM } from '../components/LazyWebLLM';
import { LazyAuraluxConsole } from '../components/LazyAuraluxConsole';
import { LazyVoiceOverlay } from '../components/LazyVoiceOverlay';

import { InferCellCard, InferCellGrid, type ModuleInfo, type InferCellSession, type InferCellStatus, type LiveModuleData } from '../components/InferCellCard';
import { SuperMotd } from '../components/supermotd';
import { CodeViewer } from '../components/ui/CodeViewer';
import { Button } from '../components/ui/Button';
import { Input } from '../components/ui/Input';
import { LocalLLMStatusWidget } from '../components/LocalLLMStatusWidget';
import { EdgeInferenceStatusWidget } from '../components/EdgeInferenceStatusWidget';
import { EdgeInferenceCatalog } from '../components/EdgeInferenceCatalog';
import { MarimoWidget } from '../components/MarimoWidget';
import { DiagnosticsPanel } from '../components/DiagnosticsPanel';
import { EventSearchBox, TimelineSparkline, EventFlowmap, EnrichedEventCard, EventStatsBadges, type SearchMode } from '../components/EventVisualization';
import { AgentTelemetryPanel } from '../components/AgentTelemetryPanel';
import { VNCAuthPanel } from '../components/VNCAuthPanel';
import { SemopsEditor } from '../components/SemopsEditor';
import { BusObservatoryView } from '../components/BusObservatoryView';
import { BusPulseWidget } from '../components/BusPulseWidget';
import { DKINFlowMonitor } from '../components/dkin/DKINFlowMonitor';
import { StudioView } from '../components/views/StudioView';
import { VoiceSpeechView } from '../components/views/VoiceSpeechView';
import { TypesTreeOverlay } from '../components/types/TypesTreeOverlay';
import { BicameralNav } from '../components/nav/BicameralNav';
import { MemoryIngestBar } from '../components/ingest/MemoryIngestBar';
import { NotificationSidepanel } from '../components/NotificationSidepanel';
import { CodeSegmentPanel } from '../components/CodeSegmentPanel';
import { LoadingStage } from '../components/LoadingStage';
import { DashboardLayoutContext } from '../lib/state/dashboard_layout_context';
import { PBLanesWidget } from '../components/PBLanesWidget';
import { MetaTestWidget } from '../components/MetaTestWidget';
import { STRpLeadsView } from '../components/views/STRpLeadsView';
import { RegistryAtlasView } from '../components/views/RegistryAtlasView';
import type { STRpLead } from '../lib/state/leads_types';

// Material Web Components (M3)
import '@material/web/ripple/ripple.js';
import '@material/web/elevation/elevation.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/progress/circular-progress.js';

// Extended types for SOTA items
interface SOTAItem {
  id: string;
  url: string;
  title: string;
  org: string;
  region: string;
  type: string;
  priority: number;
  cadence_days: number;
  tags: string[];
  notes: string;
  distill_status?: 'idle' | 'queued' | 'running' | 'completed' | 'failed';
  distill_last_iso?: string;
  distill_req_id?: string;
  distill_artifact_path?: string;
  distill_snippet?: string;
}

// Default VPS session state
const defaultSession: VPSSession = {
  flowMode: 'm',
  providers: {
    'chatgpt-web': { available: false, lastCheck: '' },
    'claude-web': { available: false, lastCheck: '' },
    'gemini-web': { available: false, lastCheck: '' },
  },
  fallbackOrder: [
    'chatgpt-web',
    'claude-web',
    'gemini-web',
  ],
  activeFallback: null,
  pbpair: { activeRequests: [], pendingProposals: [] },
  auth: { claudeLoggedIn: false, geminiCliLoggedIn: false },
};

// ID generator for actions
let actionIdCounter = 0;
const generateActionId = () => `action-${++actionIdCounter}-${Date.now().toString(36)}`;

const SOTA_PROVIDER_OPTIONS = ['auto', 'chatgpt-web', 'claude-web', 'gemini-web'] as const;
const SOTA_CACHE_KEY = 'pluribus.sota.items.v1';

const loadSotaCache = () => {
  if (typeof localStorage === 'undefined') return null;
  try {
    const raw = localStorage.getItem(SOTA_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as { items?: SOTAItem[]; lastFetch?: string | null };
    if (!parsed || !Array.isArray(parsed.items)) return null;
    return parsed;
  } catch {
    return null;
  }
};

const saveSotaCache = (items: SOTAItem[], lastFetch: string | null) => {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(
      SOTA_CACHE_KEY,
      JSON.stringify({ items, lastFetch }),
    );
  } catch {
    // ignore cache write failures
  }
};

export default component$(() => {
  const location = useLocation();
  const layoutCtx = useContext(DashboardLayoutContext);
  // Core state
  const session = useSignal<VPSSession>(defaultSession);
  const events = useSignal<BusEvent[]>([]);
  const services = useSignal<ServiceDef[]>([]);
  const serviceInstances = useSignal<any[]>([]); // Track instances separately
  const agents = useSignal<AgentStatus[]>([]);
  const requests = useSignal<STRpRequest[]>([]);
  const sotaItems = useSignal<SOTAItem[]>([]);
  const sotaError = useSignal<string | null>(null);
  const sotaLastFetch = useSignal<string | null>(null);
  const sotaRebuildNote = useSignal<string | null>(null);
  const sotaStale = useSignal(false);
  const sotaAutoRebuild = useSignal(false);
  const leads = useSignal<STRpLead[]>([]);
  const gitLog = useSignal<any[]>([]);
  const gitStatus = useSignal<any[]>([]);
  const gitBranches = useSignal<any[]>([]);
  const gitCurrentBranch = useSignal<string | null>(null);
  const selectedCommit = useSignal<string | null>(null);

  // Track bootstrap emissions to prevent duplicates (per source, per session)
  const bootstrapEmitted = useSignal<Record<string, boolean>>({});

  const bootstrapBrowserAuth = $(async (source: string) => {
    // Dedup: only emit once per source per page session
    if (bootstrapEmitted.value[source]) {
      return;
    }
    bootstrapEmitted.value = { ...bootstrapEmitted.value, [source]: true };

    // Non-blocking bootstrap: start daemon + enqueue check_login so providers recover autonomously.
    const timeout = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

    try {
      fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: 'dashboard.browser.bootstrap.requested',
          kind: 'metric',
          level: 'info',
          actor: 'dashboard',
          data: { source, at: new Date().toISOString() },
        }),
        keepalive: true,
      }).catch(() => {});
    } catch { /* ignore */ }
    try {
      await Promise.race([
        fetch('/api/browser/bootstrap', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ actor: 'dashboard', wait: false }),
          keepalive: true,
        }).catch(() => undefined),
        timeout(3000),
      ]);
    } catch { /* ignore - timeout or network error is fine */ }
    (window as any).__loadingRegistry?.exit("hook:auth");
  });

  const openAuthOverlay = $(() => {
    layoutCtx.authOverlayOpen.value = true;
    bootstrapBrowserAuth('open-overlay');
  });

  // Action cells state (notebook-style outputs)
  const actionCells = useStore<ActionCell[]>([]);
  const showActionPanel = useSignal(false);

  // UI state
  const connected = useSignal(false);

  useVisibleTask$(() => {
    (window as any).__loadingRegistry?.entry("hook:auth");
    if (__E2E__) { (window as any).__loadingRegistry?.exit("hook:auth"); return; }
    // Each page-load: try to restore provider liveness (daemon autostart + cache/cookie health checks).
    (window as any).__verboseTracker?.entry('sys:auth'); bootstrapBrowserAuth('page-load'); (window as any).__verboseTracker?.exit('sys:auth');
  });

  useVisibleTask$(() => {
    (window as any).__loadingRegistry?.entry("hook:omega-channel");
    const channel = new BroadcastChannel('pluribus-omega');
    channel.onmessage = (ev) => {
      if (ev.data.type === 'OMEGA_RELOAD_REQUIRED') {
        console.warn('[Omega] Critical System Update Detected. Reloading View in 2s...', ev.data.reason);
        setTimeout(() => {
          window.location.reload();
        }, 2000);
      }
    };
    (window as any).__loadingRegistry?.exit("hook:omega-channel");
    return () => channel.close();
  });

  useVisibleTask$(() => {
    (window as any).__loadingRegistry?.entry("hook:shadow-channel");
    if (__E2E__) { (window as any).__loadingRegistry?.exit("hook:shadow-channel"); return; }
    const channel = new BroadcastChannel('pluribus-shadow');
    channel.onmessage = (ev) => {
      const payload = ev.data || {};
      if (payload.type !== 'DATA_UPDATE') return;
      switch (payload.key) {
        case 'sota':
          {
            const items = Array.isArray(payload.data) ? payload.data : [];
            sotaItems.value = items;
            sotaError.value = null;
            sotaRebuildNote.value = null;
            sotaStale.value = false;
            if (items.length > 0) {
              const now = new Date().toISOString();
              sotaLastFetch.value = now;
              saveSotaCache(items, now);
            }
          }
          break;
        case 'gitLog':
          gitLog.value = Array.isArray(payload.data) ? payload.data : [];
          break;
        case 'gitStatus':
          gitStatus.value = Array.isArray(payload.data) ? payload.data : [];
          break;
        case 'gitBranches':
          gitBranches.value = Array.isArray(payload.data) ? payload.data : [];
          break;
        case 'gitCurrentBranch':
          gitCurrentBranch.value = typeof payload.data === 'string' ? payload.data : payload.data ? String(payload.data) : null;
          break;
        case 'leads':
          leads.value = Array.isArray(payload.data) ? payload.data : [];
          break;
        default:
          break;
      }
    };
    (window as any).__loadingRegistry?.exit("hook:shadow-channel");
    return () => channel.close();
  });

  // Real-time Bus Subscription (Entelexis Realization)
  useVisibleTask$(({ cleanup }) => {
    if (typeof __E2E__ !== 'undefined' && __E2E__) return;
    
    const client = createBusClient({ platform: 'browser' });
    client.connect().catch(console.error);

    const unsub = client.subscribe('*', (event) => {
      // 1. Feed Event Log (Raw stream)
      // Limit frequency to avoid UI thrashing? Qwik signals are batched usually.
      events.value = [event, ...events.value].slice(0, 100);

      // 2. Feed Telemetry Snapshot (SOTA, Agents, Services)
      if (event.topic === 'dashboard.telemetry.snapshot' && event.data) {
        const data = event.data as any;
        
        // Agents Map -> List
        if (data.agents) {
           const list: AgentStatus[] = [];
           for (const [actor, info] of Object.entries(data.agents)) {
               const i = info as any;
               list.push({ 
                   actor, 
                   status: i.status || 'unknown',
                   health: i.health || 'unknown',
                   current_task: i.current_task || '',
                   last_seen_iso: i.last_seen || '',
                   queue_depth: 0,
                   blockers: []
               });
           }
           agents.value = list;
           layoutCtx.agents.value = list;
        }

        // Services List
        if (Array.isArray(data.services)) {
            services.value = data.services;
        }

        // SOTA List (if present and non-empty)
        if (Array.isArray(data.sota) && data.sota.length > 0) {
            sotaItems.value = data.sota;
        }
      }
    });

    cleanup(() => {
      unsub();
      client.disconnect();
    });
  });

  const fetchSota = $(async function fetchSota(rebuild = false) {
    try {
      (window as any).__loadingRegistry?.entry("hook:sota-data");
      if (!rebuild && sotaItems.value.length === 0) {
        const cached = loadSotaCache();
        if (cached?.items?.length) {
          sotaItems.value = cached.items;
          if (!sotaLastFetch.value && cached.lastFetch) {
            sotaLastFetch.value = cached.lastFetch;
          }
          sotaStale.value = true;
        }
      }
      const baseUrl = rebuild ? '/api/sota?rebuild=1' : '/api/sota';
      const url = `${baseUrl}${baseUrl.includes('?') ? '&' : '?'}ts=${Date.now()}`;
      const res = await fetch(url, { cache: 'no-store' });
      const contentType = res.headers.get('content-type') || '';
      const raw = await res.text();
      let data: any = null;
      try {
        data = JSON.parse(raw);
      } catch {
        if (!rebuild) {
          const fallbackUrl = `/api/falkordb/dashboard/sota?ts=${Date.now()}`;
          const fallbackRes = await fetch(fallbackUrl, { cache: 'no-store' });
          const fallbackRaw = await fallbackRes.text();
          try {
            data = JSON.parse(fallbackRaw);
          } catch {
            const prefix = raw.trim().slice(0, 120);
            throw new Error(`Invalid JSON response (${contentType || 'unknown'}): ${prefix}`);
          }
          if (!fallbackRes.ok) {
            throw new Error(`HTTP ${fallbackRes.status}`);
          }
        } else {
          const prefix = raw.trim().slice(0, 120);
          throw new Error(`Invalid JSON response (${contentType || 'unknown'}): ${prefix}`);
        }
      }
      if (!res.ok && !data) {
        throw new Error(`HTTP ${res.status}`);
      }
      if (data.error) {
        throw new Error(String(data.error));
      }
      sotaItems.value = data.items || [];
      sotaLastFetch.value = new Date().toISOString();
      sotaError.value = null;
      sotaRebuildNote.value = data.rebuilt ? (data.rebuild_reason || 'rebuilt') : null;
      sotaStale.value = false;
      if (!rebuild && sotaItems.value.length === 0 && !sotaAutoRebuild.value) {
        sotaAutoRebuild.value = true;
        sotaRebuildNote.value = 'empty catalog; refreshing';
        setTimeout(() => {
          fetchSota(true);
        }, 600);
      }
      if (sotaItems.value.length > 0) {
        saveSotaCache(sotaItems.value, sotaLastFetch.value);
      }
      (window as any).__loadingRegistry?.exit("hook:sota-data");
    } catch (e) {
      (window as any).__loadingRegistry?.exit("hook:sota-data", String(e));
      sotaError.value = e instanceof Error ? e.message : String(e);
      if (sotaItems.value.length > 0) {
        sotaStale.value = true;
      } else {
        const cached = loadSotaCache();
        if (cached?.items?.length) {
          sotaItems.value = cached.items;
          if (!sotaLastFetch.value && cached.lastFetch) {
            sotaLastFetch.value = cached.lastFetch;
          }
          sotaStale.value = true;
        }
      }
      console.error('SOTA fetch error', e);
    }
  });

  const initialView = (() => {
    const raw = String(location.url.searchParams.get('view') || '').trim();
    const allowed = [
      'home',
      'studio',
      'bus',
      'events',
      'agents',
      'requests',
      'sota',
      'semops',
      'services',
      'rhizome',
      'git',
      'types',
      'terminal',
      'plurichat',
      'webllm',
      'voice',
      'distill',
      'diagnostics',
      'generative',
      'dkin',
      'metatest',
      'leads',
    ] as const;
    if (raw === 'browser-auth') return 'home';
    return (allowed as readonly string[]).includes(raw) ? raw : 'home';
  })();

  const activeView = useSignal<
    | 'home'
    | 'studio'
    | 'bus'
    | 'events'
    | 'agents'
    | 'requests'
    | 'sota'
    | 'semops'
    | 'services'
    | 'rhizome'
    | 'git'
    | 'types'
    | 'terminal'
    | 'plurichat'
    | 'webllm'
    | 'voice'
    | 'distill'
    | 'diagnostics'
    | 'browser-auth'
    | 'generative'
    | 'dkin'
    | 'metatest'
    | 'leads'
  >(initialView as any);
  const eventFilter = useSignal<string | null>(null);
  const eventSearchPattern = useSignal<string>('');
  const eventSearchMode = useSignal<SearchMode>('glob');
  const showEventFlowmap = useSignal(true);
  const showEventTimeline = useSignal(true);
  const showNdjsonView = useSignal(false);
  const commandInput = useSignal('');
  const workerCount = useSignal(0);
  const updateTrigger = useSignal(0);
  const selectedProviders = useSignal<string[]>([]);
  const providerOptions = useComputed$(() => {
    const uniq = Array.from(new Set(session.value.fallbackOrder || []));
    const nonMock = uniq.filter((p) => p !== 'mock');
    return nonMock.length > 0 ? nonMock : uniq;
  });
  const sotaProvider = useSignal<string>('auto');
  const sotaProviderOptions = SOTA_PROVIDER_OPTIONS;
  const vncProviderStatus = useComputed$(() => {
    const out: Record<string, { available: boolean; error?: string }> = {};
    const providers = (session.value as any)?.providers || {};
    for (const pid of Object.keys(providers)) {
      const st = providers[pid] || {};
      out[pid] = { available: !!st.available, error: typeof st.error === 'string' ? st.error : undefined };
    }
    return out;
  });

  useVisibleTask$(({ track }) => {
    track(() => vncProviderStatus.value);
    layoutCtx.providerStatus.value = vncProviderStatus.value;
  });

  // Allow deep-linking to a view via `/?view=<id>` (used by e2e + mobile workflows).
  useVisibleTask$(() => {
    try {
      const params = new URLSearchParams(window.location.search);
      const view = params.get('view');
      if (view === 'browser-auth') {
        layoutCtx.authOverlayOpen.value = true;
        bootstrapBrowserAuth('deep-link');
        activeView.value = 'home';
      } else if (view) {
        activeView.value = view as any;
      }
    } catch {
      // ignore
    }
  });

  // Cross-panel navigation bus (SemOps, etc.)
  useVisibleTask$(() => {
    (window as any).__loadingRegistry?.entry("hook:menu-handlers");
    (window as any).__loadingRegistry?.entry("hook:theme");

    // Theme is CSS-based, mark immediately complete
    (window as any).__loadingRegistry?.exit("hook:theme");

    const handler = (ev: Event) => {
        const detail = (ev as any)?.detail || {};
      const view = typeof detail.view === 'string' ? detail.view : null;
      if (view) {
        if (view === 'browser-auth') {
          layoutCtx.authOverlayOpen.value = true;
          bootstrapBrowserAuth('navigate-event');
          activeView.value = 'home';
          return;
        }
        activeView.value = view as any;
      }
      if (view === 'events') {
        if (typeof detail.searchPattern === 'string') eventSearchPattern.value = detail.searchPattern;
        if (typeof detail.searchMode === 'string') eventSearchMode.value = detail.searchMode as any;
        if (typeof detail.eventFilter === 'string' || detail.eventFilter === null) eventFilter.value = detail.eventFilter;
      }
    };
    window.addEventListener('pluribus:navigate', handler as any);
    (window as any).__loadingRegistry?.exit("hook:menu-handlers");
    return () => window.removeEventListener('pluribus:navigate', handler as any);
  });

  // InferCell State for distill view
  const inferCellSessions = useStore<Record<string, InferCellSession>>({});
  const inferCellStatuses = useStore<Record<string, InferCellStatus>>({});
  const inferCellLiveData = useStore<Record<string, LiveModuleData>>({});
  const expandedInferCell = useSignal<string | null>(null);

  // GOLDEN modules (G1-G10)
  const goldenModules: ModuleInfo[] = [
    { name: 'The Lens', file: 'lens_collimator.py', icon: '🔍', goldenId: 'G1', description: 'Neurosymbolic query routing - classifies depth, selects lane, determines context mode and topology' },
    { name: 'The Membrane', file: 'agent_bus.py', icon: '🧫', goldenId: 'G2', description: 'Bus-first IPC architecture - append-only event bus for all inter-agent communication' },
    { name: 'The Cells', file: 'infercell_manager.py', icon: '🧬', goldenId: 'G3', description: 'InferCell fork/merge lattice - context cells with trace correlation and divergence tracking' },
    { name: 'The Fallback', file: 'vps_session.py', icon: '🔄', goldenId: 'G4', description: 'Provider cascade chains - graceful fallback from primary to secondary providers' },
    { name: 'The Evidence', file: 'beam_append.py', icon: '📜', goldenId: 'G5', description: 'Append-only audit trails - immutable BEAM discourse log for provenance' },
    { name: 'The Topology', file: 'strp_topology.py', icon: '🌐', goldenId: 'G6', description: 'STRp execution modes - single, star, and peer_debate multi-agent patterns' },
    { name: 'The Vector', file: 'rag_vector.py', icon: '📊', goldenId: 'G7', description: 'RAG hybrid search - BM25 + semantic vector retrieval with sqlite-vec' },
    { name: 'The Guard', file: 'iso_git.mjs', icon: '🛡️', goldenId: 'G8', description: 'HGT validation ladder - 6-check guard for safe horizontal gene transfer' },
    { name: 'The Quine', file: 'iso_pqc.mjs', icon: '♾️', goldenId: 'G9', description: 'Self-referential constitution - PQC quine architecture with bounded polymorphism' },
    { name: 'The Soul', file: 'pluribus_check.py', icon: '✨', goldenId: 'G10', description: 'Unified agentic substrate - the integrated Pluribus organism heartbeat' },
  ];

  // Subsystem modules
  const subsystemModules: ModuleInfo[] = [
    { name: 'Lens/Collimator', file: 'lens_collimator.py' },
    { name: 'PluriChat', file: 'plurichat.py' },
    { name: 'InferCell', file: 'infercell_manager.py' },
    { name: 'RAG Vector', file: 'rag_vector.py' },
    { name: 'Bus Events', file: 'agent_bus.py' },
    { name: 'STRp Topology', file: 'strp_topology.py' },
    { name: 'HGT Guard', file: 'iso_git.mjs' },
    { name: 'Provider Fallback', file: 'vps_session.py' },
    { name: 'BEAM Append', file: 'beam_append.py' },
    { name: 'OITERATE', file: 'oiterate_operator.py' },
  ];

  // InferCell action handler
  const handleInferCellAction = $(async (action: string, module: ModuleInfo) => {
    // Update status to show action in progress
    inferCellStatuses[module.name] = 'checking';

    try {
      // Emit bus event for the action
      const eventPayload = {
        topic: `infercell.${action}`,
        kind: 'request',
        level: 'info',
        actor: 'dashboard',
        data: {
          moduleName: module.name,
          moduleFile: module.file,
          goldenId: module.goldenId,
          action,
          timestamp: new Date().toISOString(),
        },
      };

      // Try to emit via bus bridge (using proxy path for HTTPS compatibility)
      try {
        await fetch('/api/emit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(eventPayload),
        });
      } catch { /* Bus bridge may not be available */ }

      // Handle specific actions
      if (action === 'inspect') {
        // Fetch live module data (using proxy path for HTTPS compatibility)
        const res = await fetch(`/api/module/inspect?file=${encodeURIComponent(module.file)}`);
        if (res.ok) {
          const data = await res.json();
          inferCellLiveData[module.name] = {
            loaded: true,
            lastCheck: new Date().toISOString(),
            lineCount: data.lineCount,
            exports: data.exports,
            imports: data.imports,
            testStatus: data.testStatus,
            metrics: data.metrics,
          };
        }
      } else if (action === 'verify') {
        // Run module verification (using proxy path for HTTPS compatibility)
        const res = await fetch(`/api/module/verify?file=${encodeURIComponent(module.file)}`);
        if (res.ok) {
          const data = await res.json();
          inferCellStatuses[module.name] = data.success ? 'ok' : 'error';
          inferCellLiveData[module.name] = {
            loaded: data.loaded,
            lastCheck: new Date().toISOString(),
            testStatus: data.testStatus,
            errors: data.errors,
          };
        }
      } else if (action === 'fork') {
        // Create InferCell fork
        const traceId = crypto.randomUUID();
        const cellId = crypto.randomUUID();
        inferCellSessions[module.name] = {
          traceId,
          cellId,
          state: 'active',
          forkReason: `Manual fork from dashboard for ${module.name}`,
          children: [],
          workingMemory: { module: module.file, action },
          lastEvent: {
            topic: 'infercell.fork',
            timestamp: new Date().toISOString(),
            summary: `Forked context for ${module.name}`,
          },
        };
        inferCellStatuses[module.name] = 'forked';
      } else if (action === 'test') {
        // Run module tests (using proxy path for HTTPS compatibility)
        const res = await fetch(`/api/module/test?file=${encodeURIComponent(module.file)}`);
        if (res.ok) {
          const data = await res.json();
          inferCellLiveData[module.name] = {
            ...inferCellLiveData[module.name],
            loaded: true,
            lastCheck: new Date().toISOString(),
            testStatus: data.success ? 'pass' : 'fail',
            errors: data.errors,
          };
          inferCellStatuses[module.name] = data.success ? 'ok' : 'error';
        }
      }

      // Default to OK status after action completes
      if (!['forked', 'error'].includes(inferCellStatuses[module.name] || '')) {
        inferCellStatuses[module.name] = 'ok';
      }
    } catch (err) {
      console.error('InferCell action error:', err);
      inferCellStatuses[module.name] = 'warn';
    }
  });

  // Fetch Git Data
  useVisibleTask$(({ track }) => {
    track(() => activeView.value);
    if (activeView.value === 'git') {
        const fetchGit = async () => {
            try {
                // Fetch all git data in parallel (using proxy paths for HTTPS compatibility)
                const [logRes, statusRes, branchRes] = await Promise.all([
                    fetch('/api/git/log'),
                    fetch('/api/git/status'),
                    fetch('/api/git/branches'),
                ]);
                if (logRes.ok) {
                    const data = await logRes.json();
                    gitLog.value = data.commits || [];
                }
                if (statusRes.ok) {
                    const data = await statusRes.json();
                    gitStatus.value = data.status || [];
                }
                if (branchRes.ok) {
                    const data = await branchRes.json();
                    gitBranches.value = data.branches || [];
                    gitCurrentBranch.value = data.current || null;
                }
            } catch (e) {
                console.error('Git fetch error', e);
            }
        };
        fetchGit();
    }
  });

  // Fetch SOTA Data
  useVisibleTask$(({ track }) => {
    track(() => activeView.value);
    if (activeView.value === 'sota') {
        fetchSota(false);
    }
  });

  // Rhizome State
  const currentPath = useSignal('/');
  const fileTree = useSignal<any[]>([]);
  const selectedFile = useSignal<string | null>(null);
  const fileContent = useSignal<string>('');

  // Fetch Rhizome Tree
  useVisibleTask$(({ track }) => {
    track(() => activeView.value);
    track(() => currentPath.value);
    if (__E2E__) return;
    if (activeView.value !== 'rhizome') return;

    const fetchTree = async () => {
        try {
            // Use proxy path for HTTPS compatibility
            const res = await fetch(`/api/fs${currentPath.value}`);
            if (res.ok) {
                const data = await res.json();
                if (data.entries) {
                    fileTree.value = data.entries;
                }
            }
        } catch (e) {
            console.error('Rhizome fetch error', e);
        }
    };
    fetchTree();
  });

  // Fetch File Content
  const loadFile = $(async (path: string) => {
      selectedFile.value = path;
      try {
          // Using proxy path for HTTPS compatibility
          const res = await fetch(`/api/fs${currentPath.value === '/' ? '' : currentPath.value}/${path}`);
          if (res.ok) {
              fileContent.value = await res.text();
          }
      } catch (e) {
          fileContent.value = `Error loading file: ${e}`;
      }
  });

  const navigateUp = $(() => {
      if (currentPath.value === '/') return;
      const parts = currentPath.value.split('/');
      parts.pop();
      currentPath.value = parts.join('/') || '/';
  });

  const enterDir = $((name: string) => {
      currentPath.value = currentPath.value === '/' ? `/${name}` : `${currentPath.value}/${name}`;
  });

  // ... computed ...
  const providersList = useComputed$(() => {
    const _ = updateTrigger.value;
    const allow = new Set(['chatgpt-web', 'claude-web', 'gemini-web']);
    const entries = Object.entries(session.value.providers).filter(([name]) => allow.has(name));
    return entries;
  });

  // Filtered events
  const filteredEvents = useComputed$(() => {
    let filtered = events.value;

    // Apply category filter
    if (eventFilter.value) {
      if (eventFilter.value === 'error') {
        filtered = filtered.filter(e => e.level === 'error');
      } else if (eventFilter.value === 'high-impact') {
        filtered = filtered.filter(e => {
          const semantic = (e as any).semantic;
          return semantic?.impact === 'high' || semantic?.impact === 'critical';
        });
      } else {
        filtered = filtered.filter(e => e.kind === eventFilter.value);
      }
    }

    // Apply search pattern
    if (eventSearchPattern.value.trim()) {
      const pattern = eventSearchPattern.value.trim();
      const mode = eventSearchMode.value;

      filtered = filtered.filter(e => {
        switch (mode) {
          case 'glob': {
            // Convert glob to regex: * → .*, ? → .
            const regexPattern = pattern
              .replace(/[.+^${}()|[\]\\]/g, '\\$&')
              .replace(/\*/g, '.*')
              .replace(/\?/g, '.');
            try {
              return new RegExp(regexPattern, 'i').test(e.topic);
            } catch {
              return e.topic.includes(pattern);
            }
          }
          case 'regex': {
            // Strip /.../ wrapper if present
            const match = pattern.match(/^\/(.+)\/([gimsuy]*)$/);
            try {
              const regex = match
                ? new RegExp(match[1], match[2] || 'i')
                : new RegExp(pattern, 'i');
              return regex.test(e.topic) || regex.test(e.actor || '') || regex.test(JSON.stringify(e.data || {}));
            } catch {
              return e.topic.includes(pattern);
            }
          }
          case 'ltl': {
            // LTL mode: look for temporal patterns (simplified)
            // ◇ = eventually (exists), □ = always, ○ = next
            if (pattern.startsWith('◇') || pattern.startsWith('eventually')) {
              const target = pattern.replace(/^(◇|eventually)\s*/, '');
              return e.topic.includes(target);
            }
            if (pattern.startsWith('□') || pattern.startsWith('always')) {
              const target = pattern.replace(/^(□|always)\s*/, '');
              return e.topic.includes(target);
            }
            return e.topic.includes(pattern);
          }
          case 'actor':
            return (e.actor || '').toLowerCase().includes(pattern.toLowerCase());
          case 'topic':
            return e.topic.toLowerCase().startsWith(pattern.toLowerCase());
          default:
            return e.topic.includes(pattern);
        }
      });
    }

    return filtered;
  });

  const qaLaneStats = useComputed$(() => {
    const list = events.value.filter((event) => event.topic.startsWith('qa.stack.lane.'));
    return {
      total: list.length,
      recent: list.slice(-8).reverse(),
    };
  });

  // SOTA items grouped by type
  const sotaByType = useComputed$(() => {
    const groups: Record<string, SOTAItem[]> = {};
    sotaItems.value.forEach(item => {
      const type = item.type || 'other';
      if (!groups[type]) groups[type] = [];
      groups[type].push(item);
    });
    return groups;
  });

  // Merge service definitions with running instances
  const mergedServices = useComputed$(() => {
    const map = new Map<string, ServiceDef & { status?: string, instanceId?: string, pid?: number }>();
    
    // Base definitions
    services.value.forEach(s => {
      map.set(s.id, { ...s, status: 'stopped' });
    });

    // Overlay running instances
    serviceInstances.value.forEach(inst => {
      const s = map.get(inst.service_id);
      if (s) {
        s.status = inst.status;
        s.instanceId = inst.instance_id;
        s.pid = inst.pid;
      }
    });

    return Array.from(map.values());
  });

  // WebSocket connection & Real Bus
  const wsRef = useStore<{ socket: NoSerialize<WebSocket> | null }>({ socket: null });

  // Real Bus Emitter Adapter - wrapped with noSerialize to avoid Qwik serialization errors
  const busEmitterRef = useSignal<NoSerialize<BusEmitter> | null>(null);

  const emitBus = $(async (topic: string, kind: string, data: Record<string, unknown>) => {
    const emitter = busEmitterRef.value;
    if (!emitter) return;
    await emitter.emit(topic, kind, data);
  });

  useVisibleTask$(({ cleanup }) => {
    if (__E2E__) {
      (window as any).__loadingRegistry?.entry("hook:websocket");
      (window as any).__loadingRegistry?.exit("hook:websocket");
      return;
    }
    const wsHost = typeof window !== 'undefined' ? window.location.host : 'localhost';
    const apiHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost';
    const isSecure = typeof window !== 'undefined' && window.location.protocol === 'https:';
    const apiBaseUrl = isSecure 
        ? `https://${wsHost}/api/git` 
        : `http://${apiHost}:9300`;
    
    // kroma.live/Caddy (and local Vite dev proxy) route the bus at `/ws/bus`.
    const wsUrl = `${isSecure ? 'wss' : 'ws'}://${wsHost}/ws/bus`;
        
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let liveFlushTimer: ReturnType<typeof setTimeout> | null = null;
    let syncProcessTimer: ReturnType<typeof setTimeout> | null = null;

    // PERF: Reduced limits to prevent main thread blocking
    const EVENTS_MAX = 200;  // Reduced from 500 for performance  // Was 2000 - too many DOM updates
    const LIVE_MAX_PER_FLUSH = 25;  // Reduced from 100 for performance  // Was 800 - floods main thread
    const LIVE_FLUSH_INTERVAL_MS = 250;  // Increased from 100 for performance  // Was 60 - too aggressive
    const SYNC_PROCESS_MAX = 500;  // Was 5000 - way too many on initial load
    const SYNC_CHUNK_SIZE = 50;  // Was 200 - smaller chunks for responsiveness

    let liveQueue: BusEvent[] = [];
    let syncQueue: BusEvent[] = [];

    let eventsFetchStarted = false;
    let eventsFetchCompleted = false;
    let eventLoopStarted = false;
    let eventLoopCompleted = false;

    const markEventsFetchStart = () => {
      if (eventsFetchStarted) return;
      eventsFetchStarted = true;
      (window as any).__loadingRegistry?.entry("hook:events-fetch");
    };

    const markEventsFetchComplete = () => {
      if (eventsFetchCompleted) return;
      eventsFetchCompleted = true;
      (window as any).__loadingRegistry?.exit("hook:events-fetch");
    };

    const markEventLoopStart = () => {
      if (eventLoopStarted) return;
      eventLoopStarted = true;
      (window as any).__loadingRegistry?.entry("hook:event-loop");
    };

    const markEventLoopComplete = () => {
      if (eventLoopCompleted) return;
      eventLoopCompleted = true;
      (window as any).__loadingRegistry?.exit("hook:event-loop");
    };

    // PERF: Debounced state updates - accumulate changes, flush together
    type PendingUpdates = {
      services: ServiceDef[] | null;
      serviceInstances: any[] | null;
      agents: Map<string, AgentStatus>;  // Keyed by actor for dedup
      requests: Map<string, STRpRequest>; // Keyed by id for dedup
      sotaItems: SOTAItem[] | null;
      session: Partial<VPSSession> | null;
      workerDelta: number;
    };
    const pendingUpdates: PendingUpdates = {
      services: null,
      serviceInstances: null,
      agents: new Map(),
      requests: new Map(),
      sotaItems: null,
      session: null,
      workerDelta: 0,
    };
    let stateFlushTimer: ReturnType<typeof setTimeout> | null = null;
    const STATE_FLUSH_INTERVAL_MS = 50; // Batch state updates every 50ms

    const flushStateUpdates = () => {
      stateFlushTimer = null;
      // Flush all pending state updates in one go
      if (pendingUpdates.services) {
        services.value = pendingUpdates.services;
        pendingUpdates.services = null;
      }
      if (pendingUpdates.serviceInstances) {
        serviceInstances.value = pendingUpdates.serviceInstances;
        pendingUpdates.serviceInstances = null;
      }
      if (pendingUpdates.agents.size > 0) {
        const currentAgents = [...agents.value];
        for (const [actor, status] of pendingUpdates.agents) {
          const idx = currentAgents.findIndex(a => a.actor === actor);
          if (idx >= 0) {
            currentAgents[idx] = status;
          } else {
            currentAgents.push(status);
          }
        }
        agents.value = currentAgents;
        pendingUpdates.agents.clear();
      }
      if (pendingUpdates.requests.size > 0) {
        const currentRequests = [...requests.value];
        for (const [id, req] of pendingUpdates.requests) {
          const idx = currentRequests.findIndex(r => r.id === id);
          if (idx >= 0) {
            currentRequests[idx] = req;
          } else {
            currentRequests.unshift(req);
          }
        }
        requests.value = currentRequests.slice(0, 100);
        pendingUpdates.requests.clear();
      }
      if (pendingUpdates.sotaItems) {
        sotaItems.value = pendingUpdates.sotaItems;
        pendingUpdates.sotaItems = null;
      }
      if (pendingUpdates.session) {
        session.value = { ...session.value, ...pendingUpdates.session };
        pendingUpdates.session = null;
        updateTrigger.value++;
      }
      if (pendingUpdates.workerDelta !== 0) {
        workerCount.value = Math.max(0, workerCount.value + pendingUpdates.workerDelta);
        pendingUpdates.workerDelta = 0;
      }
    };

    const scheduleStateFlush = () => {
      if (stateFlushTimer) return;
      // Use requestIdleCallback if available for non-blocking updates
      if (typeof requestIdleCallback === 'function') {
        stateFlushTimer = setTimeout(() => {
          requestIdleCallback(() => flushStateUpdates(), { timeout: 100 });
        }, STATE_FLUSH_INTERVAL_MS);
      } else {
        stateFlushTimer = setTimeout(flushStateUpdates, STATE_FLUSH_INTERVAL_MS);
      }
    };

    const flushLive = () => {
      liveFlushTimer = null;
      const batch = liveQueue.splice(0, LIVE_MAX_PER_FLUSH);
      if (batch.length === 0) return;

      events.value = [...events.value, ...batch].slice(-EVENTS_MAX);
      for (const ev of batch) handleBusEvent(ev);
      markEventLoopComplete();

      if (liveQueue.length > 0) scheduleLiveFlush();
    };

    const scheduleLiveFlush = () => {
      if (liveFlushTimer) return;
      liveFlushTimer = setTimeout(flushLive, LIVE_FLUSH_INTERVAL_MS);
    };

    const processSyncChunk = () => {
      syncProcessTimer = null;
      const chunk = syncQueue.splice(0, SYNC_CHUNK_SIZE);
      if (chunk.length === 0) return;
      for (const ev of chunk) handleBusEvent(ev);
      markEventLoopComplete();
      if (syncQueue.length > 0) scheduleSyncProcess();
    };

    const scheduleSyncProcess = () => {
      if (syncProcessTimer) return;
      syncProcessTimer = setTimeout(processSyncChunk, 0);
    };

    function connect() {
      (window as any).__loadingRegistry?.entry("hook:websocket");
      const ws = new WebSocket(wsUrl);
      wsRef.socket = noSerialize(ws);

      ws.onopen = () => {
        connected.value = true; (window as any).__verboseTracker?.exit('sys:websocket');
        (window as any).__loadingRegistry?.exit("hook:websocket");
        // Create bus emitter with active socket
        busEmitterRef.value = noSerialize({
          emit: async (topic: string, kind: string, data: Record<string, unknown>) => {
            if (ws.readyState === WebSocket.OPEN) {
              const event: BusEvent = {
                id: (data.req_id as string) || (data.request_id as string) || generateActionId(),
                topic,
                kind,
                level: 'info',
                actor: 'dashboard-user',
                ts: Date.now(),
                iso: new Date().toISOString(),
                data,
              };
              ws.send(JSON.stringify({ type: 'publish', event }));
            }
          }
        });
        markEventsFetchStart();
        ws.send(JSON.stringify({ type: 'sync' }));

        // Fetch session and agents from REST API in parallel with 5s timeout
        // (using proxy paths for HTTPS compatibility)
        const sessionCtrl = new AbortController();
        const agentsCtrl = new AbortController();
        setTimeout(() => sessionCtrl.abort(), 5000);
        setTimeout(() => agentsCtrl.abort(), 5000);

        fetch('/api/session', { signal: sessionCtrl.signal }).then(function(res) {
          if (!res.ok) return null;
          return res.json();
        }).then(function(data) {
          if (!data) return;
          const newProviders: VPSSession['providers'] = { ...session.value.providers };
          Object.keys(data.providers || {}).forEach(function(name) {
            const p: any = data.providers[name];
            newProviders[name] = {
              available: Boolean(p.available),
              lastCheck: p.last_check || new Date().toISOString(),
              error: p.error || undefined,
              model: p.model || undefined,
            };
          });
          session.value = {
            ...session.value,
            flowMode: data.flow_mode || session.value.flowMode,
            providers: newProviders,
            fallbackOrder: (() => {
              const allow = new Set(['chatgpt-web', 'claude-web', 'gemini-web']);
              const fb = Array.isArray(data.fallback_order) ? data.fallback_order : session.value.fallbackOrder;
              const filtered = (fb || []).filter((x: string) => allow.has(String(x)));
              return filtered.length > 0 ? filtered : ['chatgpt-web', 'claude-web', 'gemini-web'];
            })(),
            activeFallback: (() => {
              const allow = new Set(['chatgpt-web', 'claude-web', 'gemini-web']);
              const af = data.active_fallback as string | undefined;
              return af && allow.has(af) ? af : null;
            })(),
          };
          // Never default-select the internal mock provider in the UI.
          const fb = session.value.fallbackOrder;
          const nextDefault = (session.value.activeFallback || fb[0] || 'chatgpt-web');
          if (selectedProviders.value.length === 0 || selectedProviders.value.includes('mock')) {
            selectedProviders.value = [nextDefault];
          }
        }).catch(function() { /* timeout or network error is fine */ });

        fetch('/api/agents', { signal: agentsCtrl.signal }).then(function(res) {
          if (!res.ok) return null;
          return res.json();
        }).then(function(data) {
          if (!data || !data.agents) return;
          agents.value = data.agents.map(function(a: any) {
            return {
              actor: a.actor || 'unknown',
              status: a.status || 'unknown',
              health: a.health || 'unknown',
              queue_depth: a.queue_depth || 0,
              current_task: a.current_task || '',
              blockers: a.blockers || [],
              vor_cdi: undefined,
              last_seen_iso: new Date().toISOString(),
            };
          });
        }).catch(function() { /* timeout or network error is fine */ });
      };
      ws.onclose = () => {
        connected.value = false;
        wsRef.socket = null;
        busEmitterRef.value = null;
        liveQueue = [];
        syncQueue = [];
        if (liveFlushTimer) {
          clearTimeout(liveFlushTimer);
          liveFlushTimer = null;
        }
        if (syncProcessTimer) {
          clearTimeout(syncProcessTimer);
          syncProcessTimer = null;
        }
        reconnectTimer = setTimeout(connect, 5000);
      };
      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if (data.type === 'sync' && data.events) {
            const incoming = Array.isArray(data.events) ? (data.events as BusEvent[]) : [];
            const recent = incoming.slice(-SYNC_PROCESS_MAX);
            events.value = recent.slice(-EVENTS_MAX);
            syncQueue = recent;
            markEventsFetchComplete();
            markEventLoopStart();
            scheduleSyncProcess();
          } else if (data.type === 'event') {
            const ev = data.event as BusEvent | undefined;
            if (ev) {
              markEventsFetchComplete();
              markEventLoopStart();
              liveQueue.push(ev);
              scheduleLiveFlush();
            }
          }
        } catch { /* ignore */ }
      };
    }

    function handleBusEvent(event: BusEvent) {
      const data = (event.data || {}) as Record<string, unknown>;

      // Link responses to Dialogos Cells
      const reqId = (data.req_id as string) || (data.request_id as string);
      if (reqId) {
        const cellIdx = actionCells.findIndex(c => c.request.id === reqId);
        if (cellIdx >= 0) {
           const cell = actionCells[cellIdx];
           // Always retain correlated raw bus events for trace viewing.
           if (cell.result) {
             const last = cell.result.events[cell.result.events.length - 1];
             if (!last || last.id !== event.id || last.topic !== event.topic || last.ts !== event.ts) {
               cell.result.events.push(event);
               if (cell.result.events.length > 500) {
                 cell.result.events.splice(0, cell.result.events.length - 500);
               }
             }
           }
           // Dialogos cell protocol (preferred)
           if (event.topic.startsWith('dialogos.cell.')) {
               if (event.topic === 'dialogos.cell.start') {
                   cell.result!.status = 'streaming';
                   cell.result!.outputs.push({
                       type: 'progress',
                       content: `Started (${String(data.mode || 'llm')})`,
                       timestamp: Date.now(),
                   });
               } else if (event.topic === 'dialogos.cell.output') {
                   const t = String(data.type || 'text');
                   const outType = (t === 'error' ? 'error' : t === 'json' ? 'json' : 'text') as ActionOutput['type'];
                   cell.result!.status = 'streaming';
                   cell.result!.outputs.push({
                       type: outType,
                       content: (data.content as any) || '',
                       timestamp: Date.now(),
                       metadata: data.provider ? { title: String(data.provider) } : undefined,
                   });
               } else if (event.topic === 'dialogos.cell.end') {
                   const ok = Boolean(data.ok);
                   cell.result!.status = ok ? 'success' : 'error';
                   cell.result!.completedAt = Date.now();
                   if (!ok) {
                       const errs = (data.errors as unknown as string[]) || [];
                       cell.result!.error = errs.join(',') || 'dialogos failed';
                   }
               }
               return;
           }
           // If it's a completion or specific output type
           if (event.topic.includes('response') || event.topic.includes('completed') || event.topic.includes('log')) {
               // Streaming update
               if (event.topic.includes('response')) {
                   cell.result!.status = 'success';
                   cell.result!.completedAt = Date.now();
               }
               
               let outputContent = data;
               // Try to extract useful content if it's an strp response
               if (data.output && typeof data.output === 'string') {
                   outputContent = data.output as any;
               }

               cell.result!.outputs.push({
                   type: typeof outputContent === 'string' ? 'text' : 'json',
                   content: outputContent,
                   timestamp: Date.now(),
                   metadata: { title: `Response (${event.actor})` }
               });
           } else if (event.level === 'error') {
               cell.result!.status = 'error';
               cell.result!.error = (data.error as string) || 'Unknown error';
           }
        }
      }

      // Telemetry Snapshot (Phase 1 Unification) - DEBOUNCED
      if (event.topic === 'dashboard.telemetry.snapshot') {
        const snap = data as any;
        if (snap) {
           if (snap.services && Array.isArray(snap.services)) {
             pendingUpdates.services = snap.services;
           }
           if (snap.sota && Array.isArray(snap.sota)) {
             pendingUpdates.sotaItems = snap.sota;
           }
           if (snap.agents && typeof snap.agents === 'object') {
             Object.entries(snap.agents).forEach(([actor, st]: any) => {
               const agent: AgentStatus = {
                 actor: actor,
                 status: st.status || 'unknown',
                 health: st.health || 'unknown',
                 last_seen_iso: st.last_seen || event.iso,
                 queue_depth: 0,
                 current_task: '',
                 blockers: [],
                 vor_cdi: undefined
               };
               pendingUpdates.agents.set(actor, agent);
             });
           }
           scheduleStateFlush();
        }
      }

      // Services list - DEBOUNCED
      if (event.topic === 'services.list') {
        const svcList = data.services as ServiceDef[] | undefined;
        const instList = data.instances as any[] | undefined;
        if (svcList && Array.isArray(svcList)) {
          pendingUpdates.services = svcList;
          scheduleStateFlush();
        }
        if (instList && Array.isArray(instList)) {
          pendingUpdates.serviceInstances = instList;
          scheduleStateFlush();
        }
      }
      // SOTA items - DEBOUNCED
      else if (event.topic === 'sota.list') {
        const items = data.items as SOTAItem[] | undefined;
        if (items && Array.isArray(items)) {
          pendingUpdates.sotaItems = items;
          scheduleStateFlush();
        }
      }
      // Provider status - DEBOUNCED
      else if (event.topic.includes('provider')) {
        const provider = data.provider as string | undefined;
        if (provider && provider in session.value.providers) {
          const currentSession = pendingUpdates.session || {};
          const currentProviders = (currentSession.providers as VPSSession['providers']) || { ...session.value.providers };
          pendingUpdates.session = {
            ...currentSession,
            providers: {
              ...currentProviders,
              [provider]: {
                available: Boolean(data.available),
                lastCheck: event.iso || new Date().toISOString(),
                error: data.error as string | undefined,
                model: data.model as string | undefined,
                note: data.note as string | undefined,
              },
            },
          };
          scheduleStateFlush();
        }
      }
      // Flow mode - DEBOUNCED
      else if (event.topic.includes('flow_mode')) {
        pendingUpdates.session = {
          ...(pendingUpdates.session || {}),
          flowMode: (data.mode as 'm' | 'A') ?? session.value.flowMode,
        };
        scheduleStateFlush();
      }
      // Agent status - DEBOUNCED
      else if (event.topic === 'pluribus.check.report') {
        const actor = event.actor;
        const newStatus: AgentStatus = {
          actor,
          status: data.status as string || 'unknown',
          health: data.health as string || 'unknown',
          queue_depth: (data.queue_depth as number) || 0,
          current_task: ((data.current_task as Record<string, unknown>)?.goal as string) || '',
          blockers: (data.blockers as string[]) || [],
          vor_cdi: (data.vor_metrics as Record<string, unknown>)?.cdi as number | undefined,
          last_seen_iso: event.iso,
        };
        pendingUpdates.agents.set(actor, newStatus);
        scheduleStateFlush();
      }
      // Worker events - DEBOUNCED
      else if (event.topic.startsWith('strp.worker.')) {
        pendingUpdates.workerDelta += event.topic.includes('start') ? 1 : event.topic.includes('end') ? -1 : 0;
        scheduleStateFlush();
      }
      // Requests - DEBOUNCED
      else if (event.topic.startsWith('strp.request')) {
        const isEnd = event.topic.includes('end') || event.topic.includes('complete') || event.topic.includes('fail');
        const reqId = (data.req_id as string) || (data.id as string) || event.id || '';
        const req: STRpRequest = {
          id: reqId,
          kind: event.kind,
          actor: event.actor,
          goal: data.goal as string || '',
          status: isEnd ? 'completed' : 'pending',
          created_iso: event.iso,
        };
        // Check both pending and current for status regression
        const existingPending = pendingUpdates.requests.get(reqId);
        const existingCurrent = requests.value.find(r => r.id === reqId);
        const existing = existingPending || existingCurrent;
        if (existing?.status === 'completed' && req.status === 'pending') {
          // Ignore regression
        } else {
          pendingUpdates.requests.set(reqId, req);
          scheduleStateFlush();
        }
      }

      // SOTA distillation status - DEBOUNCED
      else if (event.topic === 'sota.distill.status') {
        const itemId = (data.item_id as string | undefined) || (data.itemId as string | undefined);
        const status = data.status as SOTAItem['distill_status'] | undefined;
        if (itemId && status) {
          const current = pendingUpdates.sotaItems || [...sotaItems.value];
          const idx = current.findIndex(i => i.id === itemId);
          if (idx >= 0) {
            current[idx] = {
              ...current[idx],
              distill_status: status,
              distill_last_iso: event.iso,
              distill_req_id: (data.req_id as string | undefined) || current[idx].distill_req_id,
              distill_artifact_path: (data.path as string | undefined) || current[idx].distill_artifact_path,
            };
            pendingUpdates.sotaItems = current;
            scheduleStateFlush();
          }
        }
      }
      else if (event.topic === 'sota.distill.artifact') {
        const itemId = (data.item_id as string | undefined) || (data.itemId as string | undefined);
        const path = data.path as string | undefined;
        if (itemId && path) {
          const current = pendingUpdates.sotaItems || [...sotaItems.value];
          const idx = current.findIndex(i => i.id === itemId);
          if (idx >= 0) {
            current[idx] = {
              ...current[idx],
              distill_status: current[idx].distill_status || 'completed',
              distill_last_iso: event.iso,
              distill_req_id: data.req_id as string | undefined,
              distill_artifact_path: path,
              distill_snippet: data.snippet as string | undefined,
            };
            pendingUpdates.sotaItems = current;
            scheduleStateFlush();
          }
        }
      }
      else if (event.topic === 'sota.kg.status') {
        const itemId = (data.item_id as string | undefined) || (data.itemId as string | undefined);
        const status = data.status as string | undefined;
        if (itemId && status) {
          const current = pendingUpdates.sotaItems || [...sotaItems.value];
          const idx = current.findIndex(i => i.id === itemId);
          if (idx >= 0) {
            current[idx] = { ...current[idx], distill_last_iso: event.iso };
            pendingUpdates.sotaItems = current;
            scheduleStateFlush();
          }
        }
      }
    }

    connect();
    cleanup(() => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (liveFlushTimer) clearTimeout(liveFlushTimer);
      if (syncProcessTimer) clearTimeout(syncProcessTimer);
      if (stateFlushTimer) clearTimeout(stateFlushTimer);
      wsRef.socket?.close();
    });
  });

  // Sync Flow Mode from Context (Header Control)
  useTask$(({ track }) => {
    const mode = track(() => layoutCtx.flowMode.value);
    if (session.value.flowMode !== mode) {
      session.value = { ...session.value, flowMode: mode };
    }
  });

  // Actions
  const setFlowMode = $((mode: 'm' | 'A') => {
    session.value = { ...session.value, flowMode: mode };
    layoutCtx.setFlowMode$(mode);
    
    // Emit to bus
    if (busEmitterRef.value) {
      busEmitterRef.value.emit('dashboard.vps.flow_mode_changed', 'command', { mode });
    }
  });

  const cycleFilter = $(() => {
    const filters = [null, 'strp.', 'pluribus.', 'omega.', 'provider.', 'agent.', 'infer_sync.'];
    const idx = filters.indexOf(eventFilter.value);
    eventFilter.value = filters[(idx + 1) % filters.length];
  });

  // Action dispatch for notebook-style outputs
  const dispatchAction = $((actionType: string, payload: Record<string, unknown>) => {
    const requestId = generateActionId();
    const request: ActionRequest = {
      id: requestId,
      type: actionType,
      payload,
      timestamp: Date.now(),
    };

    const cell: ActionCell = {
      id: generateActionId(),
      request,
      result: {
        id: generateActionId(),
        requestId,
        status: 'pending',
        outputs: [],
        events: [],
        startedAt: Date.now(),
      },
      collapsed: false,
      activeTab: 'outputs',
    };

    // Add to beginning for most recent first
    actionCells.unshift(cell);
    if (actionCells.length > 50) actionCells.pop();
    showActionPanel.value = true;

    // Require a live bus connection for E2E correctness (no implicit mock fallback).
    const emitter = busEmitterRef.value;
    if (!emitter) {
      const idx = actionCells.findIndex(c => c.request.id === requestId);
      if (idx >= 0 && actionCells[idx].result) {
        actionCells[idx].result!.status = 'error';
        actionCells[idx].result!.error = 'bus_not_connected';
        actionCells[idx].result!.outputs.push({
          type: 'error',
          content: 'Bus not connected: ensure bus-bridge is running and the dashboard is connected before dispatching actions.',
          timestamp: Date.now(),
        });
        actionCells[idx].result!.completedAt = Date.now();
      }
      return requestId;
    }
    const handlers = createServiceActionHandlers(emitter);

    const handler = handlers[actionType];
    if (handler) {
      handler(request, {
        onOutput: (output: ActionOutput) => {
          const idx = actionCells.findIndex(c => c.request.id === requestId);
          if (idx >= 0 && actionCells[idx].result) {
            actionCells[idx].result!.outputs.push(output);
          }
        },
        onComplete: (error?: string) => {
          const idx = actionCells.findIndex(c => c.request.id === requestId);
          if (idx >= 0 && actionCells[idx].result) {
            if (error) {
                actionCells[idx].result!.status = 'error';
                actionCells[idx].result!.error = error;
            } 
            // If no error, wait for bus response to complete
            if (!error && actionCells[idx].result!.status !== 'success') {
                 // Keep pending until bus event arrives
            } else {
                 actionCells[idx].result!.completedAt = Date.now();
            }
          }
        },
        onStatusChange: (status: ActionResult['status']) => {
          const idx = actionCells.findIndex(c => c.request.id === requestId);
          if (idx >= 0 && actionCells[idx].result) {
            actionCells[idx].result!.status = status;
          }
        },
      });
    }

    return requestId;
  });

  // Toggle cell collapse
  const toggleCellCollapse = $((cellId: string) => {
    const idx = actionCells.findIndex(c => c.id === cellId);
    if (idx >= 0) {
      actionCells[idx].collapsed = !actionCells[idx].collapsed;
    }
  });

  // Clear all cells
  const clearActionCells = $(() => {
    actionCells.length = 0;
  });

  // Toggle provider selection
  const toggleProvider = $((provider: string) => {
    const p = provider.trim();
    if (!p) return;
    if (selectedProviders.value.includes(p)) {
      selectedProviders.value = selectedProviders.value.filter(x => x !== p);
    } else {
      selectedProviders.value = [...selectedProviders.value, p];
    }
    if (selectedProviders.value.length === 0) {
      const fallback = session.value.activeFallback || session.value.fallbackOrder.find((x) => x !== 'mock') || 'chatgpt-web';
      selectedProviders.value = [fallback];
    }
  });

  const toggleNdjsonView = $(() => {
    showNdjsonView.value = !showNdjsonView.value;
  });
  return (
    <div class="flex flex-col h-full">
      {/* Navigation */}
      <LoadingStage id="comp:bicameral-nav">
        <BicameralNav
          activeView={activeView.value}
          onSelect$={$((view) => {
            activeView.value = view;
          })}
        />
      </LoadingStage>

      <LoadingStage id="comp:memory-ingest">
        <MemoryIngestBar />
      </LoadingStage>

      {/* Main Content */}
      <main
        class={`flex-1 min-h-0 relative ${
          activeView.value === 'terminal' || activeView.value === 'plurichat' || activeView.value === 'webllm'
            ? 'overflow-hidden p-0'
            : 'overflow-auto p-6'
        }`}
      >
          {/* RHIZOME VIEW */}
          {activeView.value === 'rhizome' && (
            <div class="grid grid-cols-12 gap-6 h-full">
                {/* File Tree */}
                <div class="col-span-3 rounded-lg border border-border bg-card flex flex-col h-[calc(100vh-200px)]">
                    <div class="p-3 border-b border-border flex items-center justify-between">
                        <h3 class="font-semibold text-sm">Explorer</h3>
                        <div class="text-xs text-muted-foreground mono truncate max-w-[150px]">{currentPath.value}</div>
                    </div>
                    <div class="p-2 border-b border-border">
                        <Button 
                            variant="secondary"
                            onClick$={navigateUp}
                            disabled={currentPath.value === '/'}
                            class="w-full justify-start text-xs h-8"
                        >
                            .. (Up)
                        </Button>
                    </div>
                    <div class="flex-1 overflow-auto p-2 space-y-1">
                        {fileTree.value.map((entry) => (
                            <Button
                                key={entry.name}
                                variant={selectedFile.value === entry.name ? 'tonal' : 'text'}
                                onClick$={() => entry.type === 'dir' ? enterDir(entry.name) : loadFile(entry.name)}
                                class="w-full justify-start text-xs font-mono h-8"
                            >
                                <span class="mr-2">{entry.type === 'dir' ? '📁' : '📄'}</span>
                                <span class="truncate">{entry.name}</span>
                            </Button>
                        ))}
                    </div>
                </div>

                {/* Code Viewer */}
                <div class="col-span-9 rounded-lg border border-border bg-card flex flex-col h-[calc(100vh-200px)]">
                    <div class="p-3 border-b border-border">
                        <h3 class="font-semibold text-sm">{selectedFile.value || 'No file selected'}</h3>
                    </div>
                    <div class="flex-1 overflow-hidden bg-[#08080a]">
                        {selectedFile.value ? (
                            <LoadingStage id="comp:code-viewer">
                              <CodeViewer code={fileContent.value} />
                            </LoadingStage>
                        ) : (
                            <div class="flex items-center justify-center h-full text-muted-foreground">
                                Select a file to view content
                            </div>
                        )}
                    </div>
                </div>
            </div>
        )}

        {/* GIT VIEW */}
        {activeView.value === 'git' && (
            <div class="absolute inset-0 flex flex-col" style={{ height: '100%', width: '100%' }}>
                <LazyGitView />
            </div>
        )}

        {/* TERMINAL VIEW */}
        {activeView.value === 'terminal' && (
            <div class="absolute inset-0 flex flex-col min-h-0">
                <LazyTerminal />
            </div>
        )}

        {/* PLURICHAT VIEW */}
        {activeView.value === 'plurichat' && (
            <div class="absolute inset-0 flex flex-col min-h-0">
                <LazyPluriChatTerminal />
            </div>
        )}

        {/* GENERATIVE VIEW */}
	        {activeView.value === 'generative' && (
	            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 p-6 h-full overflow-auto">
	                <div class="flex flex-col gap-4">
	                    <h3 class="text-sm font-semibold text-muted-foreground">Generative Surface</h3>
	                    <LazyGenerativeCanvas />
	                </div>
	                <div class="flex flex-col gap-4">
	                    <h3 class="text-sm font-semibold text-muted-foreground">Interactive Notebooks</h3>
	                    <LoadingStage id="comp:marimo">
	                      <MarimoWidget />
	                    </LoadingStage>
	                </div>
	            </div>
	        )}

	        {/* STUDIO VIEW */}
	        {activeView.value === 'studio' && (
	          <LoadingStage id="comp:studio">
	            <StudioView session={session} events={events} agents={agents} requests={requests} onOpenAuth$={openAuthOverlay} />
	          </LoadingStage>
	        )}

	        {/* VOICE / SPEECH VIEW */}
	        {activeView.value === 'voice' && (
            <div class="h-full flex flex-col gap-6">
	            <LoadingStage id="comp:auralux">
	              <LazyAuraluxConsole session={session} emitBus$={emitBus} />
	            </LoadingStage>
              <div class="border-t border-border pt-6">
                <h3 class="text-sm font-semibold text-muted-foreground mb-4 px-2">LEGACY FALLBACK</h3>
	              <LoadingStage id="comp:voice-speech">
	                <VoiceSpeechView session={session} emitBus$={emitBus} />
	              </LoadingStage>
              </div>
              <LazyVoiceOverlay />
            </div>
	        )}

	        {/* WEBLLM VIEW (FULLSCREEN WRAPPER) */}
            {activeView.value === 'webllm' && (
              <div class="absolute inset-0 flex flex-col min-h-0 p-4 gap-4">
              <LoadingStage id="comp:edge-inference">
                <EdgeInferenceStatusWidget />
              </LoadingStage>
              <LoadingStage id="comp:edge-catalog">
                <EdgeInferenceCatalog />
              </LoadingStage>
              <LoadingStage id="comp:local-llm">
                <LocalLLMStatusWidget />
              </LoadingStage>
              <div class="flex-1 min-h-0">
	              <LazyWebLLM fullScreen={true} />
              </div>
	          </div>
	        )}

	        {/* HOME VIEW */}
        {activeView.value === 'home' && (
          <div class="space-y-6 glass-panel p-6">
	          <div class="grid gap-6 lg:grid-cols-3">
            {/* Left Column - Controls */}
            <div class="space-y-6">
              {/* Flow Mode */}
              <div class="glass-surface glass-surface-1 p-4 glass-hover-lift" style={{ '--stagger': 0 }}>
                <md-elevation></md-elevation>
                <h3 class="glass-section-header -mx-4 -mt-4 mb-3">FLOW MODE</h3>
                <div class="flex gap-2">
                  <Button
                    variant={session.value.flowMode === 'm' ? 'tonal' : 'secondary'}
                    onClick$={() => setFlowMode('m')}
                    class={`flex-1 py-3 rounded-lg font-medium transition-all ${
                      session.value.flowMode === 'm'
                        ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50'
                        : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                    }`}
                  >
                    <md-ripple></md-ripple>
                    <div class="text-lg">[m]</div>
                    <div class="text-xs">Monitor</div>
                  </Button>
                  <Button
                    variant={session.value.flowMode === 'A' ? 'tonal' : 'secondary'}
                    onClick$={() => setFlowMode('A')}
                    class={`flex-1 py-3 rounded-lg font-medium transition-all ${
                      session.value.flowMode === 'A'
                        ? 'bg-green-500/20 text-green-400 border border-green-500/50'
                        : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                    }`}
                  >
                    <md-ripple></md-ripple>
                    <div class="text-lg">[A]</div>
                    <div class="text-xs">Auto</div>
                  </Button>
                </div>
              </div>

              {/* Quick Actions */}
              <div class="glass-surface glass-surface-1 p-4">
                <h3 class="glass-section-header -mx-4 -mt-4 mb-3">ACTIONS</h3>
                <div class="grid grid-cols-2 gap-2">
                  <Button
                    variant="tonal"
                    onClick$={() => dispatchAction('curation.trigger', { source: 'dashboard' })}
                    class="p-3 rounded-lg glass-status-ok hover:opacity-90 transition-all text-sm font-medium"
                  >
                    <div>🔄</div>
                    <div>Curate</div>
                  </Button>
	                  <Button
                    variant="tonal"
	                    onClick$={() => dispatchAction('worker.spawn', { provider: session.value.activeFallback || session.value.fallbackOrder.find((x) => x !== 'mock') || 'chatgpt-web' })}
	                    class="p-3 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-all text-sm font-medium"
	                  >
                    <div>👷</div>
                    <div>Worker</div>
                  </Button>
                  <Button
                    variant="tonal"
                    onClick$={() => dispatchAction('verify.run', {})}
                    class="p-3 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-all text-sm font-medium"
                  >
                    <div>✓</div>
                    <div>Verify</div>
                  </Button>
                  <Button
                    variant="tonal"
                    onClick$={() => dispatchAction('command.send', { topic: 'pluribus.status', kind: 'request', data: {} })}
                    class="p-3 rounded-lg bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 transition-all text-sm font-medium"
                  >
                    <div>📊</div>
                    <div>Status</div>
                  </Button>
                </div>
              </div>

              {/* Dialogos Stream (Action Results) */}
              {actionCells.length > 0 && (
                <div class="glass-surface glass-surface-1 p-4">
                  <div class="flex items-center justify-between mb-3">
                    <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">DIALOGOS STREAM</h3>
                    <button
                      onClick$={() => { activeView.value = 'services'; }} // Or separate view
                      class="text-xs text-primary hover:underline"
                    >
                      Expand
                    </button>
                  </div>
                  <div class="space-y-2">
                    {actionCells.slice(0, 3).map((cell) => (
                      <div key={cell.id} class="p-2 rounded bg-muted/30 flex items-center gap-2">
                        <div class={`w-2 h-2 rounded-full ${
                          cell.result?.status === 'success' ? 'bg-green-500' :
                          cell.result?.status === 'error' ? 'bg-red-500' :
                          'bg-yellow-500 animate-pulse'
                        }`} />
                        <span class="text-xs font-mono">{cell.request.type}</span>
                        <span class="text-xs text-muted-foreground ml-auto">
                          {cell.result?.outputs.length || 0} outputs
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* InferCell (Command Input) */}
              <div class="glass-surface glass-surface-1 p-4">
                <div class="flex items-center justify-between mb-3">
                  <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">INFERCELL</h3>
                  <span class="text-xs px-2 py-0.5 rounded glass-status-ok">
                    Context: [New]
                  </span>
                </div>
                
                {/* Provider Selection */}
                <div class="flex flex-wrap gap-2 mb-3">
                  {providerOptions.value.map(p => (
                    <label key={p} class="flex items-center gap-1 text-xs text-muted-foreground bg-muted/30 px-2 py-1 rounded border border-border/50">
                      <input
                        type="checkbox"
                        checked={selectedProviders.value.includes(p)}
                        onChange$={() => toggleProvider(p)}
                      />
                      <span class="mono">{p}</span>
                    </label>
                  ))}
                </div>

                <div class="flex gap-2">
                  <Input
                    label="Command"
                    value={commandInput.value}
                    placeholder="Interject command to selected providers..."
                    class="flex-1"
                    onInput$={(_, el) => { commandInput.value = el.value; }}
                    onKeyDown$={(e) => {
                      if (e.key === 'Enter') {
                         const providers = selectedProviders.value.length > 0
                           ? selectedProviders.value
                           : [session.value.activeFallback || session.value.fallbackOrder.find(x => x !== 'mock') || 'chatgpt-web'];
                         dispatchAction('command.send', {
                             topic: 'dialogos.submit',
                             kind: 'request',
                             data: { mode: 'llm', providers, prompt: commandInput.value }
                         });
                         commandInput.value = '';
                      }
                    }}
                  />
                  <Button 
                    onClick$={() => {
                         const providers = selectedProviders.value.length > 0
                           ? selectedProviders.value
                           : [session.value.activeFallback || session.value.fallbackOrder.find(x => x !== 'mock') || 'chatgpt-web'];
                         dispatchAction('command.send', {
                             topic: 'dialogos.submit',
                             kind: 'request',
                             data: { mode: 'llm', providers, prompt: commandInput.value }
                         });
                         commandInput.value = '';
                    }}
                  >
                    Send
                  </Button>
                </div>
                <div class="text-xs text-muted-foreground mt-2">
                  Interjects prompt to selected inference engines (via `dialogosd`).
                </div>
              </div>
            </div>

            {/* Center Column - Providers (click to select for inference) */}
            <div class="space-y-6">
	              <div class="glass-surface glass-surface-1 p-4">
	                <div class="flex items-center justify-between mb-3">
	                  <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">PROVIDERS</h3>
	                  <Button
	                    variant="tonal"
	                    onClick$={openAuthOverlay}
	                    class="text-[10px] px-2 py-1 flex items-center gap-1"
	                    title="Open the Auth overlay (cloud icon)"
	                  >
	                    <span class="w-1.5 h-1.5 rounded-full bg-cyan-400/70" />
	                    Auth
	                  </Button>
	                </div>
                <div class="space-y-2">
                  {providersList.value.map(([name, status]) => {
                    // Detect web providers and their auth status
                    const isWebProvider = ['chatgpt-web', 'claude-web', 'gemini-web'].includes(name);
                    const needsLogin = status.error?.toLowerCase().includes('login') || status.error?.toLowerCase().includes('needs_login');
                    const blockedBot = status.error?.toLowerCase().includes('bot') || status.error?.toLowerCase().includes('challenge');
                    const needsOnboarding = status.error?.toLowerCase().includes('onboarding');
                    const webAuthIssue = isWebProvider && !status.available && (needsLogin || blockedBot || needsOnboarding);

                    return (
                    <div
                      key={name}
                      class={`p-3 rounded-lg border transition-all ${
                        status.available
                          ? 'border-green-500/30 bg-green-500/10'
                          : webAuthIssue
                          ? 'border-amber-500/30 bg-amber-500/10'
                          : 'border-border bg-muted/20'
                      }`}
                    >
                      <div class="flex items-center justify-between">
                        <div class="flex items-center gap-2">
                          <span class={`status-dot ${status.available ? 'available' : webAuthIssue ? 'warning' : 'unavailable'}`} />
                          <span class="font-medium mono text-sm">{name}</span>
                          {isWebProvider && (
                            <span class="text-[9px] px-1 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                              WEB
                            </span>
                          )}
                        </div>
                        <div class="flex items-center gap-1">
                          {webAuthIssue && (
                            <span class={`text-[9px] px-1.5 py-0.5 rounded ${
                              blockedBot
                                ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                                : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            }`}>
                              {blockedBot ? '🤖 BOT BLOCKED' : needsOnboarding ? '🎯 ONBOARD' : '🔐 AUTH REQUIRED'}
                            </span>
                          )}
                          {status.model && (
                            <span class="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
                              {status.model}
                            </span>
                          )}
                        </div>
                      </div>
                      {(status.note || status.error) && (
                        <div class={`text-xs mt-1 ml-5 ${webAuthIssue ? 'text-amber-400/80' : 'text-muted-foreground'}`}>
                          {status.available ? status.note : status.error}
                        </div>
                      )}
		                      {webAuthIssue && (
		                        <div class="mt-1 ml-5 space-y-2">
		                          <div class="text-[10px] text-muted-foreground italic">
		                            {blockedBot
		                              ? 'Bot challenge detected — open Auth overlay to complete it'
		                              : needsLogin
		                              ? 'OAuth login required — open Auth overlay'
		                              : 'Onboarding required — open Auth overlay'}
		                          </div>
		                          <div class="flex flex-wrap gap-2">
		                            <Button
		                              variant="tonal"
		                              onClick$={openAuthOverlay}
		                              class="text-[10px] px-2 py-1 text-amber-300 border-amber-500/30 hover:bg-amber-500/30"
		                              title="Open the Auth overlay (VNC + noVNC + controls)"
		                            >
		                              Open Auth
		                            </Button>
		                          </div>
		                        </div>
		                      )}
                    </div>
                  )})}
                </div>
              </div>

              {/* Fallback Chain */}
              <div class="glass-surface glass-surface-1 p-4">
                <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-3">FALLBACK CHAIN</h3>
                <div class="flex flex-wrap gap-1 items-center">
                  {session.value.fallbackOrder.filter((p) => p !== 'mock').map((provider, i) => {
                    const status = session.value.providers[provider as keyof typeof session.value.providers];
                    const isWebProvider = ['chatgpt-web', 'claude-web', 'gemini-web'].includes(provider);
                    const needsAuth = isWebProvider && !status?.available && (
                      status?.error?.toLowerCase().includes('login') ||
                      status?.error?.toLowerCase().includes('bot') ||
                      status?.error?.toLowerCase().includes('challenge')
                    );
                    const isAvailable = status?.available;
                    const isActive = provider === session.value.activeFallback;

                    return (
                    <span key={provider} class="flex items-center gap-1">
                      {i > 0 && <span class="text-muted-foreground text-xs">→</span>}
                      <span
                        class={`text-xs px-2 py-1 rounded relative ${
                          isActive
                            ? 'bg-primary text-primary-foreground ring-2 ring-primary/50'
                            : needsAuth
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 opacity-60'
                            : isAvailable
                            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                            : 'bg-muted/50 text-muted-foreground opacity-60'
                        }`}
                        title={needsAuth ? 'Auth required - OAuth login needed' : status?.error || ''}
                      >
                        {needsAuth && <span class="mr-0.5">🔐</span>}
                        {provider}
                      </span>
                    </span>
                  )})}
                </div>
                {/* Active fallback indicator */}
                {session.value.activeFallback && (
                  <div class="mt-2 text-xs text-muted-foreground">
                    Active: <span class="text-primary font-medium">{session.value.activeFallback}</span>
                    {['chatgpt-web', 'claude-web', 'gemini-web'].some(p =>
                      session.value.fallbackOrder.indexOf(p) < session.value.fallbackOrder.indexOf(session.value.activeFallback || '')
                    ) && (
                      <span class="ml-2 text-amber-400/80">(web providers bypassed - need auth)</span>
                    )}
                  </div>
                )}
              </div>

              {/* WebLLM Edge Inference (fullscreen) */}
              <div class="glass-surface glass-surface-1 p-4">
                <div class="flex items-center justify-between gap-3">
                  <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">EDGE INFERENCE</h3>
                  <Button
                    variant="text"
                    onClick$={() => (activeView.value = 'webllm')}
                    class="text-xs text-primary hover:underline p-0 h-auto min-h-0"
                  >
                    Open WebLLM
                  </Button>
                </div>
                <div class="text-xs text-muted-foreground mt-2">Server inference is always on; browser WebLLM is always-on unless disabled.</div>
                <div class="mt-3">
                  <LoadingStage id="comp:local-llm">
                    <LocalLLMStatusWidget />
                  </LoadingStage>
                </div>
                <div class="mt-3 flex flex-wrap gap-2">
                  <Button
                    variant="tonal"
                    onClick$={() => (activeView.value = 'studio')}
                    class="text-[10px] px-2 py-1"
                  >
                    🧪 Studio
                  </Button>
                  <Button
                    variant="tonal"
                    onClick$={() => (activeView.value = 'voice')}
                    class="text-[10px] px-2 py-1"
                  >
                    🎙️ Voice
                  </Button>
                  <Button
                    variant="tonal"
                    onClick$={openAuthOverlay}
                    class="text-[10px] px-2 py-1"
                  >
                    🌐 Browser Auth
                  </Button>
                </div>
              </div>
            </div>

            {/* Right Column - Live Pulse (dense abstraction + drill-down) */}
            <div class="space-y-6">
              <LoadingStage id="comp:bus-pulse">
                <BusPulseWidget events={events.value} title="LIVE PULSE" heightClass="h-[500px]" />
              </LoadingStage>

              {/* PBLANES Widget - Lanes progress with WIP meters */}
              <LoadingStage id="comp:pb-lanes">
                <PBLanesWidget refreshIntervalMs={30000} maxLanes={8} events={events} />
              </LoadingStage>

              <div class="glass-surface glass-surface-1 p-4">
                <div class="flex items-center justify-between mb-3">
                  <div class="flex items-center gap-2">
                    <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">QA LANES</h3>
                    <span class="text-[10px] px-2 py-0.5 rounded glass-status-ok">
                      {qaLaneStats.value.total}
                    </span>
                  </div>
                  <button
                    onClick$={() => {
                      try {
                        window.dispatchEvent(new CustomEvent('pluribus:navigate', {
                          detail: {
                            view: 'events',
                            searchPattern: 'qa.stack.lane.message',
                            searchMode: 'topic',
                            eventFilter: null,
                          },
                        }));
                      } catch {
                        // no-op
                      }
                    }}
                    class="text-[10px] px-2 py-1 rounded bg-muted/30 hover:bg-muted/50"
                  >
                    View All
                  </button>
                </div>

                {qaLaneStats.value.recent.length === 0 ? (
                  <div class="text-xs text-muted-foreground">No QA lane messages yet.</div>
                ) : (
                  <div class="space-y-2">
                    {qaLaneStats.value.recent.map((event, i) => {
                      const data = (event.data || {}) as Record<string, unknown>;
                      const laneLabel =
                        (typeof data.lane === 'string' && data.lane) ||
                        (typeof data.lane_id === 'string' && data.lane_id) ||
                        (typeof data.laneId === 'string' && data.laneId) ||
                        (typeof data.lane_name === 'string' && data.lane_name) ||
                        (typeof data.laneName === 'string' && data.laneName) ||
                        'lane';
                      const stageLabel =
                        (typeof data.stage === 'string' && data.stage) ||
                        (typeof data.phase === 'string' && data.phase) ||
                        (typeof data.step === 'string' && data.step) ||
                        (typeof data.status === 'string' && data.status) ||
                        '';
                      const summary =
                        (typeof data.summary === 'string' && data.summary) ||
                        (typeof data.message === 'string' && data.message) ||
                        (typeof data.msg === 'string' && data.msg) ||
                        (typeof data.text === 'string' && data.text) ||
                        (typeof data.note === 'string' && data.note) ||
                        (typeof data.body === 'string' && data.body) ||
                        (Object.keys(data).length > 0 ? JSON.stringify(data).slice(0, 160) : 'No payload');
                      const laneKey = laneLabel.trim().toUpperCase().slice(0, 1);
                      const laneClass = {
                        S: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
                        O: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
                        Q: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
                        D: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
                        T: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
                      }[laneKey] || 'bg-muted/40 text-muted-foreground border-border/50';

                      return (
                        <div key={`${event.id || event.ts}-${i}`} class="rounded-md border border-border/60 bg-muted/20 p-2">
                          <div class="flex items-center gap-2">
                            <span class={`text-[10px] px-1.5 py-0.5 rounded border ${laneClass}`}>
                              {laneLabel}
                            </span>
                            {stageLabel && (
                              <span class="text-[10px] px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground border border-border/50">
                                {stageLabel}
                              </span>
                            )}
                            <span class="ml-auto text-[10px] text-muted-foreground">
                              {event.iso?.slice(11, 19) || ''}
                            </span>
                          </div>
                          <div class="mt-1 text-xs text-muted-foreground line-clamp-2">{summary}</div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>

          <LoadingStage id="comp:supermotd">
            <SuperMotd
              connected={connected.value}
              events={events.value}
              session={session.value}
              emitBus$={emitBus}
            />
          </LoadingStage>

          {/* Agent Telemetry Panel - Real-time debugging feedback */}
          <LoadingStage id="comp:agent-telemetry">
            <AgentTelemetryPanel
              events={events.value}
              maxHeight="320px"
            />
          </LoadingStage>
          </div>
        )}

        {/* BUS OBSERVATORY (consolidated) */}
        {activeView.value === 'bus' && (
          <LoadingStage id="comp:bus-observatory">
            <BusObservatoryView
              connected={connected.value}
              events={events}
              agents={agents}
              requests={requests}
              session={session}
              emitBus$={emitBus}
            />
          </LoadingStage>
        )}

        {/* EVENTS VIEW - Enhanced with Symbolic/LTL/Vector/KG Visualizations */}
        {activeView.value === 'events' && (
          <LoadingStage id="comp:event-viz">
          <div class="space-y-4">
            {/* Search & Visualization Controls */}
            <div class="flex flex-col lg:flex-row gap-4">
              {/* Search Box */}
              <div class="flex-1">
                <EventSearchBox
                  onSearch$={$((pattern: string, mode: SearchMode) => {
                    eventSearchPattern.value = pattern;
                    eventSearchMode.value = mode;
                  })}
                  placeholder="Search: strp.*, /error|warn/i, ◇response, @claude..."
                />
              </div>

              {/* Visualization Toggles */}
              <div class="flex items-center gap-2">
                <button
                  onClick$={() => showEventTimeline.value = !showEventTimeline.value}
                  class={`text-xs px-3 py-2 rounded-lg border transition-colors ${
                    showEventTimeline.value
                      ? 'bg-primary/20 text-primary border-primary/30'
                      : 'bg-muted/30 text-muted-foreground border-border hover:bg-muted/50'
                  }`}
                >
                  📈 Timeline
                </button>
                <button
                  onClick$={() => showEventFlowmap.value = !showEventFlowmap.value}
                  class={`text-xs px-3 py-2 rounded-lg border transition-colors ${
                    showEventFlowmap.value
                      ? 'bg-purple-500/20 text-purple-400 border-purple-500/30'
                      : 'bg-muted/30 text-muted-foreground border-border hover:bg-muted/50'
                  }`}
                >
                  🔗 Flowmap
                </button>
                <button
                  onClick$={toggleNdjsonView}
                  class={`text-xs px-3 py-2 rounded-lg border transition-colors ${
                    showNdjsonView.value
                      ? 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30'
                      : 'bg-muted/30 text-muted-foreground border-border hover:bg-muted/50'
                  }`}
                >
                  📋 NDJSON
                </button>
              </div>
            </div>

            <div class="flex flex-wrap items-center gap-2 text-xs">
              <span class="text-muted-foreground">Quick:</span>
              <button
                onClick$={() => {
                  eventSearchPattern.value = 'qa.';
                  eventSearchMode.value = 'topic';
                  eventFilter.value = null;
                }}
                class={`px-2 py-1 rounded border transition-colors ${
                  eventSearchMode.value === 'topic' && eventSearchPattern.value.toLowerCase().startsWith('qa.')
                    ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30'
                    : 'bg-muted/30 text-muted-foreground border-border hover:bg-muted/50'
                }`}
              >
                qa.
              </button>
            </div>

            {/* Dimensional Event Stats Header (Clickable Filters) */}
            <div class="grid grid-cols-2 md:grid-cols-6 gap-2">
              <div
                class={`rounded-lg border border-border bg-card p-3 text-center cursor-pointer hover:bg-muted/50 transition-colors ${!eventFilter.value ? 'ring-2 ring-primary' : ''}`}
                onClick$={() => eventFilter.value = null}
              >
                <div class="text-2xl font-bold text-primary">{events.value.length}</div>
                <div class="text-xs text-muted-foreground">Total Events</div>
              </div>
              <div
                class={`rounded-lg border border-green-500/30 bg-green-500/10 p-3 text-center cursor-pointer hover:bg-green-500/20 transition-colors ${eventFilter.value === 'metric' ? 'ring-2 ring-green-500' : ''}`}
                onClick$={() => eventFilter.value = 'metric'}
              >
                <div class="text-2xl font-bold text-green-400">
                  {events.value.filter(e => e.kind === 'metric').length}
                </div>
                <div class="text-xs text-muted-foreground">Metrics</div>
              </div>
              <div
                class={`rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 text-center cursor-pointer hover:bg-blue-500/20 transition-colors ${eventFilter.value === 'request' ? 'ring-2 ring-blue-500' : ''}`}
                onClick$={() => eventFilter.value = 'request'}
              >
                <div class="text-2xl font-bold text-blue-400">
                  {events.value.filter(e => e.kind === 'request').length}
                </div>
                <div class="text-xs text-muted-foreground">Requests</div>
              </div>
              <div
                class={`rounded-lg border border-purple-500/30 bg-purple-500/10 p-3 text-center cursor-pointer hover:bg-purple-500/20 transition-colors ${eventFilter.value === 'artifact' ? 'ring-2 ring-purple-500' : ''}`}
                onClick$={() => eventFilter.value = 'artifact'}
              >
                <div class="text-2xl font-bold text-purple-400">
                  {events.value.filter(e => e.kind === 'artifact').length}
                </div>
                <div class="text-xs text-muted-foreground">Artifacts</div>
              </div>
              <div
                class={`rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-center cursor-pointer hover:bg-red-500/20 transition-colors ${eventFilter.value === 'error' ? 'ring-2 ring-red-500' : ''}`}
                onClick$={() => eventFilter.value = 'error'}
              >
                <div class="text-2xl font-bold text-red-400">
                  {events.value.filter(e => e.level === 'error').length}
                </div>
                <div class="text-xs text-muted-foreground">Errors</div>
              </div>
              <div
                class={`rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-3 text-center cursor-pointer hover:bg-yellow-500/20 transition-colors ${eventFilter.value === 'high-impact' ? 'ring-2 ring-yellow-500' : ''}`}
                onClick$={() => eventFilter.value = 'high-impact'}
              >
                <div class="text-2xl font-bold text-yellow-400">
                  {events.value.filter(e => (e as any).semantic?.impact === 'high' || (e as any).semantic?.impact === 'critical').length}
                </div>
                <div class="text-xs text-muted-foreground">High Impact</div>
              </div>
            </div>

            {/* Semantic Stats Badges */}
            <EventStatsBadges events={filteredEvents.value} />

            {/* Timeline & Flowmap Visualizations */}
            {(showEventTimeline.value || showEventFlowmap.value) && (
              <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                {showEventTimeline.value && (
                  <TimelineSparkline events={filteredEvents.value} buckets={60} height={60} width={400} />
                )}
                {showEventFlowmap.value && (
                  <EventFlowmap events={filteredEvents.value} maxNodes={16} height={180} />
                )}
              </div>
            )}

            {/* Event Card Profiler Grid OR NDJSON Raw View */}
            <div class="rounded-lg border border-border bg-card">
              <div class="p-3 border-b border-border flex items-center justify-between">
                <div class="flex items-center gap-3">
                  <h2 class="font-semibold">{showNdjsonView.value ? 'NDJSON Raw Feed' : 'Event Profiler'}</h2>
                  <span class="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">Live Feed</span>
                  {eventSearchPattern.value && (
                    <span class="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 font-mono">
                      {eventSearchMode.value}: {eventSearchPattern.value}
                    </span>
                  )}
                </div>
                <span class="text-sm text-muted-foreground">{filteredEvents.value.length} items</span>
              </div>

              {/* NDJSON Raw View */}
              {showNdjsonView.value ? (
                <div class="p-2 overflow-auto h-[calc(100vh-520px)] bg-black/50 font-mono text-[10px]">
                  {filteredEvents.value.slice().reverse().slice(0, 100).map((event, i) => (
                    <div
                      key={i}
                      class={`p-1.5 border-b border-white/5 hover:bg-white/5 ${
                        event.level === 'error' ? 'text-red-400' :
                        event.level === 'warn' ? 'text-yellow-400' :
                        event.kind === 'artifact' ? 'text-purple-400' :
                        'text-gray-400'
                      }`}
                    >
                      {JSON.stringify(event)}
                    </div>
                  ))}
                  {filteredEvents.value.length === 0 && (
                    <div class="p-8 text-center text-muted-foreground">
                      No events to display
                    </div>
                  )}
                </div>
              ) : (
                /* Event Cards View */
                <div class="p-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 overflow-auto h-[calc(100vh-520px)] content-start">
                  {filteredEvents.value.slice().reverse().map((event, i) => (
                    <EnrichedEventCard
                      key={i}
                      event={event}
                      index={i}
                      showLTL={true}
                      showVectors={true}
                      showKG={true}
                    />
                  ))}

                  {filteredEvents.value.length === 0 && (
                    <div class="col-span-full p-12 text-center text-muted-foreground border-2 border-dashed border-white/5 rounded-xl">
                      <div class="text-4xl mb-4">📭</div>
                      No events match the active filter.
                      {eventSearchPattern.value && (
                        <div class="mt-2 text-xs">
                          Search: <code class="bg-muted/30 px-1 rounded">{eventSearchPattern.value}</code>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
          </LoadingStage>
        )}

        {/* AGENTS VIEW */}
        {activeView.value === 'agents' && (
          <div class="rounded-lg border border-border bg-card">
            <div class="p-4 border-b border-border">
              <h2 class="font-semibold">Agent Status</h2>
              <p class="text-sm text-muted-foreground">VOR metrics and health from pluribus.check.report</p>
            </div>
            <div class="overflow-auto">
              <table class="w-full text-sm">
                <thead class="bg-muted/30">
                  <tr>
                    <th class="text-left p-3 font-medium">Actor</th>
                    <th class="text-left p-3 font-medium">Status</th>
                    <th class="text-left p-3 font-medium">Health</th>
                    <th class="text-left p-3 font-medium">Queue</th>
                    <th class="text-left p-3 font-medium">Task</th>
                    <th class="text-left p-3 font-medium">VOR CDI</th>
                    <th class="text-left p-3 font-medium">Last Seen</th>
                  </tr>
                </thead>
                <tbody>
                  {agents.value.length === 0 ? (
                    <tr>
                      <td colSpan={7} class="p-8 text-center text-muted-foreground">
                        No agent reports received yet
                      </td>
                    </tr>
                  ) : (
                    agents.value.map((agent) => (
                      <tr key={agent.actor} class="border-b border-border/30 hover:bg-muted/20">
                        <td class="p-3 font-mono">{agent.actor}</td>
                        <td class="p-3">
                          <span class={`px-2 py-0.5 rounded text-xs ${
                            agent.status === 'idle' ? 'bg-green-500/20 text-green-400' :
                            agent.status === 'working' ? 'bg-blue-500/20 text-blue-400' :
                            'bg-muted text-muted-foreground'
                          }`}>
                            {agent.status}
                          </span>
                        </td>
                        <td class="p-3">
                          <span class={agent.health === 'ok' ? 'text-green-400' : 'text-yellow-400'}>
                            {agent.health}
                          </span>
                        </td>
                        <td class="p-3">{agent.queue_depth}</td>
                        <td class="p-3 text-muted-foreground max-w-[200px] truncate">{agent.current_task || '-'}</td>
                        <td class="p-3 font-mono">{agent.vor_cdi?.toFixed(2) || '-'}</td>
                        <td class="p-3 text-xs text-muted-foreground">{agent.last_seen_iso?.slice(11, 19)}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* REQUESTS VIEW */}
        {activeView.value === 'requests' && (
          <div class="rounded-lg border border-border bg-card">
            <div class="p-4 border-b border-border">
              <h2 class="font-semibold">Pending Requests</h2>
              <p class="text-sm text-muted-foreground">STRp pipeline requests</p>
            </div>
            <div class="overflow-auto">
              <table class="w-full text-sm">
                <thead class="bg-muted/30">
                  <tr>
                    <th class="text-left p-3 font-medium">ID</th>
                    <th class="text-left p-3 font-medium">Time</th>
                    <th class="text-left p-3 font-medium">Kind</th>
                    <th class="text-left p-3 font-medium">Actor</th>
                    <th class="text-left p-3 font-medium">Goal</th>
                    <th class="text-left p-3 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody>
                  {requests.value.length === 0 ? (
                    <tr>
                      <td colSpan={6} class="p-8 text-center text-muted-foreground">
                        No pending requests
                      </td>
                    </tr>
                  ) : (
                    requests.value.slice().reverse().map((req) => (
                      <tr key={req.id} class="border-b border-border/30 hover:bg-muted/20">
                        <td class="p-3 font-mono text-xs">{req.id?.slice(0, 8)}</td>
                        <td class="p-3 text-xs text-muted-foreground">{req.created_iso?.slice(11, 19)}</td>
                        <td class="p-3">{req.kind}</td>
                        <td class="p-3 font-mono">{req.actor}</td>
                        <td class="p-3 max-w-[300px] truncate">{req.goal}</td>
                        <td class="p-3">
                          <span class={`px-2 py-0.5 rounded text-xs ${
                            req.status === 'completed' ? 'bg-green-500/20 text-green-400' :
                            req.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                            req.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                            'bg-blue-500/20 text-blue-400'
                          }`}>
                            {req.status}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* SOTA VIEW */}
        {activeView.value === 'sota' && (
          <div class="space-y-6">
            <div class="flex items-center justify-between">
              <div>
                <h2 class="text-lg font-semibold">SOTA Catalog</h2>
                <p class="text-sm text-muted-foreground">Curated tools, sources, and research feeds</p>
                {sotaError.value && (
                  <p class="text-xs text-red-400 mt-1">SOTA fetch failed: {sotaError.value}</p>
                )}
              </div>
              <div class="flex items-center gap-3">
                <select
                  class="px-3 py-2 rounded bg-muted/50 border border-border text-sm"
                  value={sotaProvider.value}
                  onChange$={(e) => { sotaProvider.value = (e.target as HTMLSelectElement).value; }}
                >
                  {sotaProviderOptions.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
                <button
                  class="px-3 py-2 rounded bg-muted/50 border border-border text-sm text-muted-foreground hover:bg-muted/70"
                  onClick$={() => fetchSota(false)}
                  title="Refresh SOTA catalog"
                >
                  Refresh
                </button>
                <button
                  class="px-3 py-2 rounded bg-muted/50 border border-border text-sm text-muted-foreground hover:bg-muted/70"
                  onClick$={() => fetchSota(true)}
                  title="Rebuild SOTA index"
                >
                  Rebuild
                </button>
                <div class="text-sm text-muted-foreground">
                  {sotaItems.value.length} items
                </div>
                {sotaLastFetch.value && (
                  <div class="text-xs text-muted-foreground">
                    {sotaLastFetch.value.slice(11, 19)}
                  </div>
                )}
                {sotaStale.value && (
                  <div class="text-xs text-amber-400">
                    cached
                  </div>
                )}
                {sotaRebuildNote.value && (
                  <div class="text-xs text-amber-400">
                    rebuilt: {sotaRebuildNote.value}
                  </div>
                )}
              </div>
            </div>

            {Object.entries(sotaByType.value).map(([type, items]) => (
              <div key={type} class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-primary">
                    {type === 'rss' ? '📰' : type === 'repo' ? '📦' : type === 'blog' ? '📝' : type === 'site' ? '🌐' : '📄'}
                  </span>
                  <h3 class="font-semibold capitalize">{type}</h3>
                  <span class="text-xs text-muted-foreground">({items.length})</span>
                </div>
                <div class="divide-y divide-border/30">
                  {items.map((item) => (
                    <div key={item.id} class="p-4 hover:bg-muted/20 transition-colors">
                      <div class="flex items-start justify-between gap-4">
                        <div class="flex-1">
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noopener"
                            class="font-medium text-primary hover:underline"
                          >
                            {item.title}
                          </a>
                          <div class="text-sm text-muted-foreground mt-1">
                            {item.org} • {item.region}
                          </div>
                          {item.notes && (
                            <div class="text-xs text-muted-foreground mt-2">{item.notes}</div>
                          )}
                        </div>
                        <div class="flex flex-col items-end gap-2">
                          <span class={`text-xs px-2 py-0.5 rounded ${
                            item.priority === 1 ? 'bg-red-500/20 text-red-400' :
                            item.priority === 2 ? 'bg-yellow-500/20 text-yellow-400' :
                            'bg-muted text-muted-foreground'
                          }`}>
                            P{item.priority}
                          </span>
                          {item.distill_status && (
                            <span class={`text-xs px-2 py-0.5 rounded ${
                              item.distill_status === 'completed' ? 'bg-green-500/20 text-green-400' :
                              item.distill_status === 'failed' ? 'bg-red-500/20 text-red-400' :
                              item.distill_status === 'running' ? 'bg-blue-500/20 text-blue-400' :
                              item.distill_status === 'queued' ? 'bg-yellow-500/20 text-yellow-400' :
                              'bg-muted text-muted-foreground'
                            }`}>
                              {item.distill_status}
                            </span>
                          )}
                          <div class="flex gap-1 flex-wrap justify-end">
                            {item.tags.slice(0, 3).map((tag) => (
                              <span key={tag} class="text-xs px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground">
                                {tag}
                              </span>
                            ))}
                          </div>
                          <button
                             onClick$={() => dispatchAction('sota.distill', { itemId: item.id, provider: sotaProvider.value })}
                             class="text-xs text-primary hover:underline mt-1"
                          >
                            Distill
                          </button>
                          <button
                             onClick$={() => dispatchAction('sota.kg.add', { itemId: item.id, ref: item.distill_artifact_path || item.url })}
                             class="text-xs text-primary hover:underline"
                          >
                            KG
                          </button>
                        </div>
                      </div>
                      {item.distill_artifact_path && (
                        <div class="mt-3 rounded bg-muted/30 border border-border/50 p-3">
                          <div class="text-xs text-muted-foreground">
                            Distillation artifact: <span class="mono text-primary">{item.distill_artifact_path}</span>
                          </div>
                          {item.distill_snippet && (
                            <pre class="mono text-xs whitespace-pre-wrap mt-2 max-h-40 overflow-auto">{item.distill_snippet}</pre>
                          )}
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {sotaItems.value.length === 0 && (
              <div class="rounded-lg border border-border bg-card p-8 text-center text-muted-foreground">
                No SOTA items loaded. Check /api/sota or rebuild the index.
              </div>
            )}
          </div>
        )}

        {/* SERVICES VIEW */}
        {activeView.value === 'services' && (
          <div class="grid gap-6 lg:grid-cols-2">
            {/* Left: Service Lists */}
            <div class="space-y-6">
              {/* Port Services */}
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-primary">⚡</span>
                  <h3 class="font-semibold">Port Services</h3>
                  <span class="text-xs text-muted-foreground">(HTTP/WS, MCP, APIs)</span>
                </div>
                <div class="divide-y divide-border/30">
                  {mergedServices.value.filter(s => s.kind === 'port').map((svc) => (
                    <div key={svc.id} class="p-4 flex items-center justify-between hover:bg-muted/20">
                      <div class="flex items-center gap-3">
                        <span class={`status-dot ${svc.status === 'running' ? 'available pulse' : 'unavailable'}`} />
                        <div>
                          <div class="font-medium">{svc.name}</div>
                          <div class="text-xs text-muted-foreground">{svc.description}</div>
                        </div>
                      </div>
                      <div class="flex items-center gap-3">
                        {svc.port && <span class="mono text-sm text-primary">:{svc.port}</span>}
                        <div class="flex gap-1">
                          <button
                            onClick$={() => dispatchAction(svc.status === 'running' ? 'service.stop' : 'service.start', { serviceId: svc.id, instanceId: svc.instanceId })}
                            class={`px-3 py-1 rounded text-xs font-medium ${
                              svc.status === 'running'
                                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                            }`}
                          >
                            {svc.status === 'running' ? 'Stop' : 'Start'}
                          </button>
                          <button
                            onClick$={() => dispatchAction('service.logs', { serviceId: svc.id, lines: 50 })}
                            class="px-2 py-1 rounded text-xs font-medium bg-muted/50 text-muted-foreground hover:bg-muted"
                          >
                            Logs
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Process Services */}
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-primary">🔄</span>
                  <h3 class="font-semibold">Process Services</h3>
                  <span class="text-xs text-muted-foreground">(Workers, Daemons, TUIs)</span>
                </div>
                <div class="divide-y divide-border/30">
                  {mergedServices.value.filter(s => s.kind === 'process').map((svc) => (
                    <div key={svc.id} class="p-4 flex items-center justify-between hover:bg-muted/20">
                      <div class="flex items-center gap-3">
                        <span class={`status-dot ${svc.status === 'running' ? 'available pulse' : 'unavailable'}`} />
                        <div>
                          <div class="font-medium">{svc.name}</div>
                          <div class="text-xs text-muted-foreground">{svc.description}</div>
                        </div>
                      </div>
                      <div class="flex items-center gap-3">
                        <div class="flex gap-1">
                          {(svc.tags || []).slice(0, 2).map((tag) => (
                            <span key={tag} class="text-xs px-2 py-0.5 rounded bg-muted/50">{tag}</span>
                          ))}
                        </div>
                        <div class="flex gap-1">
                          <button
                            onClick$={() => dispatchAction(svc.status === 'running' ? 'service.stop' : 'service.start', { serviceId: svc.id, instanceId: svc.instanceId })}
                            class={`px-3 py-1 rounded text-xs font-medium ${
                              svc.status === 'running'
                                ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                                : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                            }`}
                          >
                            {svc.status === 'running' ? 'Stop' : 'Start'}
                          </button>
                          <button
                            onClick$={() => dispatchAction('service.restart', { serviceId: svc.id, instanceId: svc.instanceId })}
                            class="px-2 py-1 rounded text-xs font-medium bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30"
                          >
                            Restart
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Compositions */}
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-primary">🔗</span>
                  <h3 class="font-semibold">Compositions</h3>
                  <span class="text-xs text-muted-foreground">(Pipelines, Workflows)</span>
                </div>
                <div class="divide-y divide-border/30">
                  {mergedServices.value.filter(s => s.kind === 'composition').map((svc) => (
                    <div key={svc.id} class="p-4 flex items-center justify-between hover:bg-muted/20">
                      <div class="flex items-center gap-3">
                        <span class={`status-dot ${svc.status === 'running' ? 'available pulse' : 'unavailable'}`} />
                        <div>
                          <div class="font-medium">{svc.name}</div>
                          <div class="text-xs text-muted-foreground">{svc.description}</div>
                        </div>
                      </div>
                      <button
                        onClick$={() => dispatchAction('composition.run', { compositionId: svc.id })}
                        class="px-3 py-1 rounded text-xs font-medium bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
                      >
                        Run
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Right: Dialogos Stream (Action Results Panel) */}
            <div class="rounded-lg border border-border bg-card flex flex-col h-[calc(100vh-200px)]">
              <div class="p-4 border-b border-border flex items-center justify-between flex-shrink-0">
                <h3 class="font-semibold">Dialogos</h3>
                <div class="flex items-center gap-2">
                  <span class="text-xs text-muted-foreground">{actionCells.length} actions</span>
                  {actionCells.length > 0 && (
                    <button
                      onClick$={clearActionCells}
                      class="text-xs px-2 py-1 rounded bg-muted/50 hover:bg-muted text-muted-foreground"
                    >
                      Clear
                    </button>
                  )}
                </div>
              </div>

              <div class="flex-1 overflow-auto p-4 space-y-4">
                {actionCells.length === 0 ? (
                  <div class="text-center py-8 text-muted-foreground">
                    <div class="text-4xl mb-3">💬</div>
                    <div>Dialogos Stream Empty</div>
                    <div class="text-xs mt-1">Interject commands to begin the dialogue.</div>
                  </div>
                ) : (
                  actionCells.map((cell) => (
                    <OutputCell
                      key={cell.id}
                      cell={cell}
                      onToggleCollapse$={$(() => toggleCellCollapse(cell.id))}
                    />
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* 10x10 DISTILLATION VIEW */}
        {activeView.value === 'distill' && (
          <div class="space-y-6">
            {/* Header Stats */}
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div class="rounded-lg border border-green-500/30 bg-green-500/10 p-4">
                <div class="text-3xl font-bold text-green-400">10/10</div>
                <div class="text-sm text-muted-foreground">Generations Complete</div>
              </div>
              <div class="rounded-lg border border-blue-500/30 bg-blue-500/10 p-4">
                <div class="text-3xl font-bold text-blue-400">102</div>
                <div class="text-sm text-muted-foreground">BEAM Entries</div>
              </div>
              <div class="rounded-lg border border-purple-500/30 bg-purple-500/10 p-4">
                <div class="text-3xl font-bold text-purple-400">651</div>
                <div class="text-sm text-muted-foreground">GOLDEN Lines</div>
              </div>
              <div class="rounded-lg border border-yellow-500/30 bg-yellow-500/10 p-4">
                <div class="text-3xl font-bold text-yellow-400">92%</div>
                <div class="text-sm text-muted-foreground">Verified (93/101)</div>
              </div>
            </div>

            {/* GOLDEN Synthesis Grid - Interactive InferCell Cards */}
            <LoadingStage id="comp:infer-cell">
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-2xl">🏆</span>
                  <h3 class="font-semibold">GOLDEN Synthesis (G1-G10)</h3>
                  <span class="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">INTERACTIVE</span>
                  <span class="text-xs text-muted-foreground ml-auto">Click to expand • Trigger actions</span>
                </div>
                <div class="p-4 space-y-2">
                  {goldenModules.map((mod) => (
                    <InferCellCard
                      key={mod.goldenId}
                      module={mod}
                      status={inferCellStatuses[mod.name] || 'idle'}
                      session={inferCellSessions[mod.name]}
                      liveData={inferCellLiveData[mod.name]}
                      onTrigger$={handleInferCellAction}
                      compact={false}
                    />
                  ))}
                </div>
              </div>
            </LoadingStage>

            {/* Subsystem Verification Matrix - Interactive Grid */}
            <LoadingStage id="comp:infer-cell">
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-2xl">✅</span>
                  <h3 class="font-semibold">Subsystem Verification Matrix</h3>
                  <span class="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">
                    {Object.values(inferCellStatuses).filter(s => s === 'ok').length}/{subsystemModules.length} VERIFIED
                  </span>
                  <button
                    class="text-xs px-3 py-1 rounded border border-cyan-500/30 bg-cyan-500/10 hover:bg-cyan-500/20 ml-auto"
                    onClick$={async () => {
                      // Verify all subsystems
                      for (const mod of subsystemModules) {
                        await handleInferCellAction('verify', mod);
                      }
                    }}
                  >
                    🔄 Verify All
                  </button>
                </div>
                <div class="p-4">
                  <InferCellGrid
                    modules={subsystemModules}
                    statuses={inferCellStatuses}
                    sessions={inferCellSessions}
                    liveData={inferCellLiveData}
                    onTrigger$={handleInferCellAction}
                    compact={true}
                    columns={5}
                  />
                </div>
              </div>
            </LoadingStage>

            {/* Lens/Collimator Status */}
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-2xl">🔍</span>
                  <h3 class="font-semibold">Lens/Collimator Routing</h3>
                </div>
                <div class="p-4 space-y-3">
                  <div class="flex items-center justify-between p-2 rounded bg-muted/20">
                    <span class="text-sm">Depth Classification</span>
                    <span class="text-xs px-2 py-0.5 rounded bg-blue-500/20 text-blue-400">narrow | deep</span>
                  </div>
                  <div class="flex items-center justify-between p-2 rounded bg-muted/20">
                    <span class="text-sm">Lane Selection</span>
                    <span class="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">dialogos | pbpair</span>
                  </div>
                  <div class="flex items-center justify-between p-2 rounded bg-muted/20">
                    <span class="text-sm">Context Mode</span>
                    <span class="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">min | lite | full</span>
                  </div>
                  <div class="flex items-center justify-between p-2 rounded bg-muted/20">
                    <span class="text-sm">Topology</span>
                    <span class="text-xs px-2 py-0.5 rounded bg-orange-500/20 text-orange-400">single | star | peer_debate</span>
                  </div>
                </div>
              </div>

              {/* Protocol Evolution */}
              <div class="rounded-lg border border-border bg-card">
                <div class="p-4 border-b border-border flex items-center gap-3">
                  <span class="text-2xl">📋</span>
                  <h3 class="font-semibold">DKIN Protocol v12</h3>
                </div>
                <div class="p-4 space-y-2 text-sm">
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v1</span><span>Minimal check-in</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v2</span><span>Drift guards</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v3</span><span>Enhanced dashboard</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v4</span><span>ITERATE operator</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v5</span><span>Silent monitoring</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v6</span><span>Shared-ledger sync</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v7</span><span>Gap detection</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v8</span><span>OITERATE omega-loop</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v9</span><span>DKIN alias established</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v10</span><span>Intelligent membrane</span></div>
                  <div class="flex gap-2"><span class="text-muted-foreground w-8">v11</span><span>MCP interop observability</span></div>
                  <div class="flex gap-2 text-primary font-medium"><span class="w-8">v12</span><span>Parallel agent isolation (PAIP)</span></div>
                </div>
              </div>
            </div>

            {/* BEAM Discourse Summary */}
            <div class="rounded-lg border border-border bg-card">
              <div class="p-4 border-b border-border flex items-center gap-3">
                <span class="text-2xl">📝</span>
                <h3 class="font-semibold">BEAM Discourse Summary</h3>
                <span class="text-xs text-muted-foreground">102 entries across 10 iterations</span>
              </div>
              <div class="p-4">
                <div class="grid grid-cols-5 md:grid-cols-10 gap-2">
                  {[1,2,3,4,5,6,7,8,9,10].map((i) => (
                    <div key={i} class="rounded border border-border bg-muted/20 p-2 text-center">
                      <div class="text-xs text-muted-foreground">Iter</div>
                      <div class="text-lg font-bold text-primary">{i}</div>
                      <div class="text-xs text-green-400">~10</div>
                    </div>
                  ))}
                </div>
                <div class="mt-4 flex gap-4 text-sm">
                  <div class="flex items-center gap-2">
                    <span class="w-3 h-3 rounded bg-green-500"></span>
                    <span>V - Verified</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="w-3 h-3 rounded bg-blue-500"></span>
                    <span>R - Reported</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="w-3 h-3 rounded bg-purple-500"></span>
                    <span>I - Intent</span>
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="w-3 h-3 rounded bg-yellow-500"></span>
                    <span>G - Gap</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

	        {/* DIAGNOSTICS VIEW - Deep Dive Control Panel */}
	        {activeView.value === 'diagnostics' && (
	          <LoadingStage id="comp:diagnostics">
	            <DiagnosticsPanel />
	          </LoadingStage>
	        )}

	        {/* SEMOPS VIEW - Semantic Operators CRUD */}
	        {activeView.value === 'semops' && (
	          <LoadingStage id="comp:semops-editor">
	            <SemopsEditor />
	          </LoadingStage>
	        )}

	        {/* DKIN FLOW MONITOR VIEW - Protocol v1-v19 Synthesis */}
        {activeView.value === 'dkin' && (
          <LoadingStage id="comp:dkin-monitor">
            <div class="h-full flex flex-col">
              <DKINFlowMonitor events={events.value} />
            </div>
          </LoadingStage>
        )}

        {/* METATEST VIEW - Comprehensive Test Dashboard */}
        {activeView.value === 'metatest' && (
          <LoadingStage id="comp:metatest">
            <div class="h-full flex flex-col p-4">
              <div class="mb-4">
                <h1 class="text-xl font-bold text-foreground">METATEST Dashboard</h1>
                <p class="text-sm text-muted-foreground">Comprehensive Test Inventory & Coverage Analysis</p>
              </div>
              <div class="flex-1 overflow-auto">
                <MetaTestWidget refreshIntervalMs={30000} maxItems={60} />
              </div>
            </div>
          </LoadingStage>
        )}

        {/* BROWSER AUTH VIEW - VNC Access for Manual OAuth Login */}
	        {activeView.value === 'browser-auth' && (
	          <LoadingStage id="comp:vnc-auth">
	            <div class="h-full flex flex-col p-4">
                <VNCAuthPanel fullScreen={true} />
              </div>
	          </LoadingStage>
        )}

        {/* TYPES ATLAS VIEW - Sextet Ontology Tree */}
        {activeView.value === 'types' && (
          <LoadingStage id="comp:types-tree">
            <div class="h-full overflow-y-auto p-4">
              <TypesTreeOverlay
                onClose$={$(() => {
                  activeView.value = 'home';
                })}
              />
            </div>
          </LoadingStage>
        )}

        {/* LEADS VIEW - STRp Content Curation Leads */}
        {activeView.value === 'leads' && (
          <LoadingStage id="comp:leads">
            <div class="h-full overflow-y-auto p-4">
              <STRpLeadsView
                leads={leads}
                dispatchAction={dispatchAction}
                connected={connected.value}
              />
            </div>
          </LoadingStage>
        )}

        {/* REGISTRY ATLAS VIEW - Registry topology visualization */}
        {activeView.value === 'registry-atlas' && (
          <LoadingStage id="comp:registry-atlas">
            <div class="h-full overflow-y-auto p-6">
              <RegistryAtlasView events={events} />
            </div>
          </LoadingStage>
        )}

        <LoadingStage id="comp:code-segment">
          <CodeSegmentPanel />
        </LoadingStage>
        <LoadingStage id="comp:notifications">
          <NotificationSidepanel />
        </LoadingStage>
      </main>
    </div>
  );
});

export const head: DocumentHead = {
  title: 'Pluribus Control Center',
  meta: [
    { name: 'description', content: 'Unified isomorphic control center for Pluribus multi-agent orchestration' },
  ],
};
