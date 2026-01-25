/**
 * MEMORY INGEST BAR (The "Magic" Bar) - v2 Non-Blocking Edition
 * ==============================================================
 *
 * A sophisticated visualization of the Pluribus "Percolation Loop".
 *
 * v2 Changes:
 * - NON-BLOCKING: All data fetching is async with immediate fallbacks
 * - THREE.js is deferred and optional - bar renders instantly without it
 * - Real metrics from bus/API with graceful degradation
 * - Recent records display with truncated content
 * - Key metadata for AI/ML and human utility
 *
 * Features:
 * - Ingest: Drag-and-drop zone for multimodal blobs
 * - Flow: Optional Three.js visualization (loads in background)
 * - Memory: Real-time capacity and state of Vector/KG/Dialogos stores
 * - Teleology: Display of current evolutionary intent (CMP)
 * - Recent: Latest records with truncated content snapshots
 *
 * Style: VMware Clarity UX (Clean, Enterprise, Functional, Dark Mode).
 */

import { component$, useSignal, useVisibleTask$, useStore } from '@builder.io/qwik';
import { classifyTopicTypes, type TopicTypesTag } from '../../lib/types-classify';
import { LoadingOrbShader } from '../LoadingOrbShader';

// ============================================================================
// Types
// ============================================================================

interface IngestMetric {
  type: 'blob' | 'stream' | 'inference' | 'dialogos' | 'kg';
  id: string;
  status: 'pending' | 'processing' | 'assimilated' | 'error';
  progress: number;
  label?: string;
}

interface RecentRecord {
  id: string;
  ts: number;
  iso: string;
  type: string;
  actor: string;
  topic: string;
  content: string;
  contentType: string;
  typesTag: TopicTypesTag;
  metadata: Record<string, string | number | boolean>;
}

interface LabelStat {
  label: string;
  count: number;
  pct: number;
}

interface MemoryState {
  vectorCount: number;
  kgNodes: number;
  kgFacts: number;
  dialogosRecords: number;
  cmpScore: number;
  entropy: number;
  indexerHealth: string;
  vectorCapacityPct: number;
  kgCapacityPct: number;
  contextWindowPct: number;
  ingestVelocity: number;
  lastIngestTs: number;
  loaded: boolean;
  error: string | null;
  // FalkorDB Integration (Phase 1)
  falkordbConnected: boolean;
  falkordbHealth: 'healthy' | 'degraded' | 'disconnected' | 'unknown';
  falkordbNodes: number;
  falkordbEdges: number;
  falkordbSyncPct: number;
  falkordbLatencyMs: number;
  falkordbLabelStats: LabelStat[];
  falkordbLabelTotal: number;
  // Phase 1 Step 5: Sparkline history
  velocityHistory: number[];
}

interface BusEvent {
  id: string;
  ts: number;
  iso: string;
  topic: string;
  actor: string;
  kind: string;
  level: string;
  data?: Record<string, unknown>;
}

// ============================================================================
// Utilities
// ============================================================================

function truncate(text: string, maxLen: number): string {
  if (!text) return '';
  text = String(text).trim();
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 3) + '...';
}

function formatAge(ts: number): string {
  const now = Date.now() / 1000;
  const age = now - ts;
  if (age < 60) return `${Math.floor(age)}s`;
  if (age < 3600) return `${Math.floor(age / 60)}m`;
  if (age < 86400) return `${Math.floor(age / 3600)}h`;
  return `${Math.floor(age / 86400)}d`;
}

function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function extractContentPreview(event: BusEvent): { content: string; contentType: string } {
  const data = event.data || {};
  const contentFields = [
    ['prompt', 'prompt'],
    ['response', 'response'],
    ['content', 'content'],
    ['message', 'message'],
    ['text', 'text'],
    ['summary', 'summary'],
    ['subject', 'fact'],
    ['tool', 'tool_call'],
    ['error', 'error'],
  ];

  for (const [field, type] of contentFields) {
    const val = data[field];
    if (val && typeof val === 'string' && val.length > 5) {
      return { content: truncate(val, 80), contentType: type };
    }
  }

  if (event.topic.includes('dialogos')) {
    return { content: truncate(JSON.stringify(data).slice(0, 60), 60), contentType: 'dialogos' };
  }
  if (event.topic.includes('graphiti') || event.topic.includes('kg')) {
    return { content: truncate(JSON.stringify(data).slice(0, 60), 60), contentType: 'fact' };
  }

  return { content: truncate(event.topic, 40), contentType: 'event' };
}

function semanticBadgeClass(semantic: TopicTypesTag['semantics']): string {
  switch (semantic) {
    case 'Object':
      return 'bg-emerald-500/20 text-emerald-300';
    case 'Process':
      return 'bg-cyan-500/20 text-cyan-300';
    case 'Type':
      return 'bg-indigo-500/20 text-indigo-300';
    case 'Shape':
      return 'bg-purple-500/20 text-purple-300';
    case 'Symbol':
      return 'bg-orange-500/20 text-orange-300';
    case 'Observer':
      return 'bg-yellow-500/20 text-yellow-300';
    default:
      return 'bg-gray-500/20 text-gray-400';
  }
}

// Phase 1 Step 5: Sparkline renderer
function renderSparklinePath(values: number[], width: number, height: number): string {
  if (values.length < 2) return '';
  const max = Math.max(...values, 1);
  const min = Math.min(...values, 0);
  const range = max - min || 1;
  const stepX = width / (values.length - 1);
  const points = values.map((v, i) => {
    const x = i * stepX;
    const y = height - ((v - min) / range) * height;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  return `M${points.join(' L')}`;
}
// ============================================================================
// Non-blocking data fetchers
// ============================================================================

async function fetchMemoryMetrics(signal?: AbortSignal): Promise<Partial<MemoryState>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch('/api/metrics/snapshot?window=60', {
      signal: signal || controller.signal,
    });

    if (response.ok) {
      const data = await response.json();
      return {
        vectorCount: data.agents?.count || 0,
        kgNodes: data.topics?.count || 0,
        kgFacts: data.kpis?.total_events || 0,
        dialogosRecords: data.queue?.completed || 0,
        cmpScore: 1.0 - (data.kpis?.error_rate || 0),
        entropy: data.entropy?.topic_entropy || 0,
        indexerHealth: data.kpis?.error_rate > 0.1 ? 'degraded' : 'ok',
        ingestVelocity: data.kpis?.velocity || 0,
        lastIngestTs: data.ts || Date.now() / 1000,
        loaded: true,
        error: null,
      };
    }
  } catch {
    // Silent fallback
  } finally {
    clearTimeout(timeoutId);
  }
  return { loaded: true, error: null };
}

async function fetchRecentRecords(limit: number = 5): Promise<RecentRecord[]> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 5000);
  try {
    const response = await fetch('/api/bus/events?limit=' + limit, {
      signal: controller.signal,
    });

    if (response.ok) {
      const payload = await response.json();
      const events: BusEvent[] = Array.isArray(payload) ? payload : (payload?.events || []);
      return events
        .filter((e) => e.topic.startsWith('dialogos') || e.topic.includes('graphiti'))
        .slice(0, limit)
        .map((e) => {
          const { content, contentType } = extractContentPreview(e);
          const typesTag = classifyTopicTypes(e.topic);
          return {
            id: e.id,
            ts: e.ts,
            iso: e.iso,
            type: e.kind,
            actor: e.actor,
            topic: e.topic,
            content,
            contentType,
            typesTag,
            metadata: { level: e.level, age: formatAge(e.ts) },
          };
        });
    }
  } catch {
    // Silent fallback
  } finally {
    clearTimeout(timeoutId);
  }
  return [];
}

async function fetchIngestStats(): Promise<Partial<MemoryState>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch('/api/dialogos/health', {
      signal: controller.signal,
    });

    if (response.ok) {
      const health = await response.json();
      return {
        dialogosRecords: health.records_indexed_total || 0,
        indexerHealth: health.status || 'unknown',
        ingestVelocity: health.records_indexed_total / Math.max(health.uptime_seconds || 1, 1) * 60,
      };
    }
  } catch {
    // Silent fallback
  } finally {
    clearTimeout(timeoutId);
  }
  return {};
}

async function fetchFalkorDBHealth(): Promise<Partial<MemoryState>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch('/api/falkordb/health', {
      signal: controller.signal,
    });

    if (response.ok) {
      const health = await response.json();
      return {
        falkordbConnected: health.connected ?? false,
        falkordbHealth: health.status || 'unknown',
        falkordbNodes: health.nodes || 0,
        falkordbEdges: health.edges || 0,
        falkordbSyncPct: health.sync_pct || 0,
        falkordbLatencyMs: health.latency_ms || 0,
      };
    }
  } catch {
    // Silent fallback - FalkorDB might not be running
  } finally {
    clearTimeout(timeoutId);
  }
  return {
    falkordbConnected: false,
    falkordbHealth: 'disconnected',
    falkordbNodes: 0,
    falkordbEdges: 0,
    falkordbSyncPct: 0,
    falkordbLatencyMs: 0,
  };
}

async function fetchFalkorDBStats(): Promise<Partial<MemoryState>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch('/api/falkordb/stats', {
      signal: controller.signal,
    });

    if (response.ok) {
      const stats = await response.json();
      const rawLabels = Array.isArray(stats?.node_labels) ? stats.node_labels : [];
      const totalGraphNodes = Number(stats?.total_graph_nodes || 0);
      const labelsTotal = rawLabels.reduce((sum: number, item: any) => sum + (Number(item?.count) || 0), 0);
      const total = totalGraphNodes > 0 ? totalGraphNodes : labelsTotal;
      const topLabels = rawLabels.slice(0, 5).map((item: any) => {
        const count = Number(item?.count) || 0;
        return {
          label: String(item?.label || 'unknown'),
          count,
          pct: total > 0 ? (count / total) * 100 : 0,
        };
      });

      return {
        falkordbLabelStats: topLabels,
        falkordbLabelTotal: total,
      };
    }
  } catch {
    // Silent fallback
  } finally {
    clearTimeout(timeoutId);
  }
  return {};
}

// ============================================================================
// Component
// ============================================================================

export const MemoryIngestBar = component$(() => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const containerRef = useSignal<HTMLDivElement>();
  const isExpanded = useSignal(false);
  const threeLoaded = useSignal(false);
  const threeError = useSignal(false);

  const ingestQueue = useStore<IngestMetric[]>([]);
  const recentRecords = useStore<RecentRecord[]>([]);

  const memoryState = useStore<MemoryState>({
    vectorCount: 0,
    kgNodes: 0,
    kgFacts: 0,
    dialogosRecords: 0,
    cmpScore: 0,
    entropy: 0,
    indexerHealth: 'unknown',
    vectorCapacityPct: 0,
    kgCapacityPct: 0,
    contextWindowPct: 0,
    ingestVelocity: 0,
    lastIngestTs: 0,
    loaded: false,
    error: null,
    // FalkorDB Integration
    falkordbConnected: false,
    falkordbHealth: 'unknown',
    falkordbNodes: 0,
    falkordbEdges: 0,
    falkordbSyncPct: 0,
    falkordbLatencyMs: 0,
    falkordbLabelStats: [],
    falkordbLabelTotal: 0,
    // Sparkline history
    velocityHistory: [],
  });

  // NON-BLOCKING DATA LOADING
  useVisibleTask$(async ({ cleanup }) => {
    const abortController = new AbortController();
    cleanup(() => abortController.abort());

    const loadData = async () => {
      try {
        const falkordbStatsPromise = isExpanded.value
          ? fetchFalkorDBStats()
          : Promise.resolve<Partial<MemoryState>>({});

        const [metrics, ingestStats, records, falkordbHealth, falkordbStats] = await Promise.allSettled([
          fetchMemoryMetrics(abortController.signal),
          fetchIngestStats(),
          fetchRecentRecords(5),
          fetchFalkorDBHealth(),
          falkordbStatsPromise,
        ]);

        if (metrics.status === 'fulfilled') {
          Object.assign(memoryState, metrics.value);
        }
        if (ingestStats.status === 'fulfilled') {
          Object.assign(memoryState, ingestStats.value);
        }
        if (records.status === 'fulfilled' && records.value.length > 0) {
          recentRecords.splice(0, recentRecords.length, ...records.value);
        }
        if (falkordbHealth.status === 'fulfilled') {
          Object.assign(memoryState, falkordbHealth.value);
        }
        if (falkordbStats.status === 'fulfilled') {
          Object.assign(memoryState, falkordbStats.value);
        }

        // Track velocity history for sparkline (keep last 20 samples)
        if (memoryState.ingestVelocity > 0 || memoryState.velocityHistory.length > 0) {
          const newHistory = [...memoryState.velocityHistory, memoryState.ingestVelocity];
          if (newHistory.length > 20) newHistory.shift();
          memoryState.velocityHistory = newHistory;
        }

        memoryState.loaded = true;
      } catch (e) {
        memoryState.error = e instanceof Error ? e.message : 'Unknown error';
        memoryState.loaded = true;
      }
    };

    loadData();
    const interval = setInterval(loadData, 10_000);
    cleanup(() => clearInterval(interval));
  });

  // DEFERRED THREE.JS VISUALIZATION
  useVisibleTask$(async ({ track, cleanup }) => {
    track(() => isExpanded.value);

    if (!isExpanded.value || !canvasRef.value || threeLoaded.value || threeError.value) {
      return;
    }

    const loadThree = async () => {
      try {
        const threePromise = import('three');
        const timeoutPromise = new Promise<never>((_, reject) =>
          setTimeout(() => reject(new Error('THREE.js load timeout')), 5000)
        );

        const THREE = await Promise.race([threePromise, timeoutPromise]);

        if (!canvasRef.value) return;

        threeLoaded.value = true;

        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, 300 / 100, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({
          canvas: canvasRef.value,
          alpha: true,
          antialias: false,
          powerPreference: 'low-power',
        });

        const width = containerRef.value?.clientWidth || 300;
        const height = 36;
        renderer.setSize(width, height);
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
        camera.position.z = 5;

        const geometry = new THREE.BufferGeometry();
        const count = 200;
        const positions = new Float32Array(count * 3);
        const colors = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
          positions[i * 3] = (Math.random() - 0.5) * 10;
          positions[i * 3 + 1] = (Math.random() - 0.5) * 2;
          positions[i * 3 + 2] = (Math.random() - 0.5) * 2;
          colors[i * 3] = 0.0;
          colors[i * 3 + 1] = 0.8;
          colors[i * 3 + 2] = 0.9;
        }

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
          size: 0.05,
          vertexColors: true,
          transparent: true,
          opacity: 0.8
        });

        const points = new THREE.Points(geometry, material);
        scene.add(points);

        let frameId: number;
        const animate = () => {
          frameId = requestAnimationFrame(animate);
          const positions = points.geometry.attributes.position.array as Float32Array;
          for (let i = 0; i < count; i++) {
            positions[i * 3] += 0.02 * (1 + Math.random());
            positions[i * 3 + 1] += Math.sin(Date.now() * 0.001 + positions[i * 3]) * 0.01;
            if (positions[i * 3] > 5) positions[i * 3] = -5;
          }
          points.geometry.attributes.position.needsUpdate = true;
          renderer.render(scene, camera);
        };

        animate();

        cleanup(() => {
          cancelAnimationFrame(frameId);
          geometry.dispose();
          material.dispose();
          renderer.dispose();
        });
      } catch (e) {
        console.warn('[MemoryIngestBar] THREE.js failed to load:', e);
        threeError.value = true;
      }
    };

    if ('requestIdleCallback' in window) {
      const idleId = (window as Window & { requestIdleCallback: (cb: () => void, opts?: { timeout: number }) => number }).requestIdleCallback(loadThree, { timeout: 2000 });
      cleanup(() => (window as Window & { cancelIdleCallback: (id: number) => void }).cancelIdleCallback(idleId));
    } else {
      const timeoutId = setTimeout(loadThree, 100);
      cleanup(() => clearTimeout(timeoutId));
    }
  });

  const healthColor = memoryState.indexerHealth === 'ok'
    ? 'bg-green-400'
    : memoryState.indexerHealth === 'degraded'
      ? 'bg-yellow-400'
      : 'bg-red-400';

  return (
    <section
      ref={containerRef}
      class={`w-full bg-background/80 border-b border-border/50 backdrop-blur-sm relative z-10 transition-colors duration-300 ${isExpanded.value ? 'bg-background/95' : ''}`}
    >
      {/* Compact Bar */}
      <div class="flex items-center h-10 px-4 gap-4">
        {/* INGEST Section */}
        <div class="flex items-center gap-4">
          <div class="flex flex-col items-start justify-center">
            <span class="text-[10px] font-bold tracking-wider text-primary uppercase">INGEST</span>
            <div class="flex items-center gap-2">
              <span class={`w-1.5 h-1.5 rounded-full ${healthColor} ${memoryState.indexerHealth === 'ok' ? 'animate-pulse' : ''}`} />
              <span class="font-mono text-xs text-foreground/90">
                {memoryState.loaded ? `${formatNumber(memoryState.dialogosRecords)} REC` : '...'}
              </span>
            </div>
          </div>
          <div
            class="hidden md:flex items-center px-3 py-1 rounded bg-muted/50 border border-border text-[10px] font-medium text-muted-foreground hover:bg-muted hover:text-foreground hover:border-primary/20 transition-all cursor-pointer"
            title="Drop Multimodal Blobs Here"
          >
            <span>DROP INPUT +</span>
          </div>
        </div>

        {/* Visual Flow Section */}
        <div
          class="flex-1 relative h-full flex items-center justify-center cursor-pointer group overflow-hidden"
          onClick$={() => (isExpanded.value = !isExpanded.value)}
        >
          <div class="absolute inset-0 flex items-center justify-center text-[10px] font-bold tracking-widest text-cyan-500/0 group-hover:text-cyan-500/100 transition-all bg-black/0 group-hover:bg-black/60 backdrop-blur-[0px] group-hover:backdrop-blur-sm z-10 pointer-events-none">
            {isExpanded.value ? 'COLLAPSE' : 'EXPAND'}
          </div>
          {isExpanded.value && threeLoaded.value ? (
            <canvas ref={canvasRef} class="w-full h-full opacity-60" />
          ) : (
            <div class="w-full h-full bg-gradient-to-r from-cyan-900/20 via-purple-900/20 to-pink-900/20 opacity-60 animate-pulse" />
          )}
        </div>

        {/* Metrics Section */}
        <div class="flex items-center gap-4 md:gap-6 ml-auto">
          {/* FalkorDB Health Indicator (Phase 1 Step 1) */}
          <div class="flex flex-col items-start justify-center" title={`FalkorDB: ${memoryState.falkordbHealth} | Latency: ${memoryState.falkordbLatencyMs}ms`}>
            <span class="text-[10px] font-bold tracking-wider text-orange-400 uppercase flex items-center gap-1">
              <span class={`w-1.5 h-1.5 rounded-full ${
                memoryState.falkordbHealth === 'healthy' ? 'bg-green-400 animate-pulse' :
                memoryState.falkordbHealth === 'degraded' ? 'bg-yellow-400' :
                memoryState.falkordbHealth === 'disconnected' ? 'bg-red-400' : 'bg-gray-400'
              }`} />
              FDB
            </span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? formatNumber(memoryState.falkordbNodes) : '...'}
            </span>
          </div>
          {/* FalkorDB Edges (Phase 1 Step 2) */}
          <div class="hidden lg:flex flex-col items-start justify-center" title="FalkorDB edge count">
            <span class="text-[10px] font-bold tracking-wider text-amber-400 uppercase">EDGES</span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? formatNumber(memoryState.falkordbEdges) : '...'}
            </span>
          </div>
          {/* FalkorDB Sync (Phase 1 Step 3) */}
          <div class="hidden xl:flex flex-col items-start justify-center" title="Bus-to-FalkorDB sync percentage">
            <span class="text-[10px] font-bold tracking-wider text-teal-400 uppercase">SYNC</span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? `${memoryState.falkordbSyncPct.toFixed(1)}%` : '...'}
            </span>
          </div>
          <div class="hidden sm:flex flex-col items-start justify-center">
            <span class="text-[10px] font-bold tracking-wider text-green-400 uppercase">VEL</span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? `${memoryState.ingestVelocity.toFixed(1)}/m` : '...'}
            </span>
          </div>
          <div class="flex flex-col items-start justify-center">
            <span class="text-[10px] font-bold tracking-wider text-purple-400 uppercase">VEC</span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? formatNumber(memoryState.vectorCount) : '...'}
            </span>
          </div>
          <div class="hidden md:flex flex-col items-start justify-center">
            <span class="text-[10px] font-bold tracking-wider text-blue-400 uppercase">KG</span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? formatNumber(memoryState.kgFacts) : '...'}
            </span>
          </div>
          <div class="flex flex-col items-start justify-center">
            <span class="text-[10px] font-bold tracking-wider text-pink-400 uppercase">CMP</span>
            <span class="font-mono text-xs text-foreground/90">
              {memoryState.loaded ? memoryState.cmpScore.toFixed(3) : '...'}
            </span>
          </div>
        </div>
      </div>

      {/* Expanded Panel */}
      {isExpanded.value && (
        <div class="border-t border-border/50 bg-card/90 backdrop-blur-md p-4 relative overflow-hidden animate-in slide-in-from-top-2 duration-200">
          {/* Sphere Visualization Background */}
          <div class="absolute inset-0 flex items-center justify-center pointer-events-none opacity-15">
            <div class="w-64 h-64">
              <LoadingOrbShader />
            </div>
          </div>
          {/* Content Grid */}
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6 relative z-10">
            {/* Column 1: Active Flows */}
            <div class="space-y-3">
              <h4 class="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-2">ACTIVE FLOWS</h4>
              {ingestQueue.length === 0 ? (
                <div class="text-xs text-muted-foreground/60 italic">No active ingest streams</div>
              ) : (
                ingestQueue.map((item) => (
                  <div key={item.id} class="flex items-center justify-between text-xs font-mono p-2 rounded bg-muted/30">
                    <div class="flex items-center gap-2">
                      <span class={`w-1.5 h-1.5 rounded-full ${item.status === 'processing' ? 'bg-cyan-400 animate-pulse' :
                        item.status === 'assimilated' ? 'bg-green-400' :
                          item.status === 'error' ? 'bg-red-400' : 'bg-gray-400'
                        }`} />
                      <span>{item.label || item.id}</span>
                    </div>
                    <div class="w-20 h-1 bg-white/10 rounded-full overflow-hidden ml-3">
                      <div class="h-full bg-cyan-500 transition-all" style={{ width: `${item.progress * 100}%` }} />
                    </div>
                  </div>
                ))
              )}
              <div class="mt-4 p-2 rounded bg-muted/30 border border-border">
                <div class="flex items-center justify-between text-[10px]">
                  <span class="text-muted-foreground">Indexer Status</span>
                  <span class={`font-bold ${memoryState.indexerHealth === 'ok' ? 'text-green-400' :
                    memoryState.indexerHealth === 'degraded' ? 'text-yellow-400' : 'text-red-400'
                    }`}>{memoryState.indexerHealth.toUpperCase()}</span>
                </div>
                <div class="flex items-center justify-between text-[10px] mt-1">
                  <span class="text-muted-foreground">Entropy</span>
                  <span class="font-mono text-foreground/80">{memoryState.entropy.toFixed(4)}</span>
                </div>
              </div>
              {/* Phase 1 Step 5: Ingest Rate Sparkline */}
              <div class="mt-4 p-2 rounded bg-muted/30 border border-border">
                <div class="flex items-center justify-between text-[10px] mb-2">
                  <span class="text-muted-foreground">Ingest Velocity</span>
                  <span class="font-mono text-cyan-400">{memoryState.ingestVelocity.toFixed(1)}/min</span>
                </div>
                <div class="h-8 w-full">
                  {memoryState.velocityHistory.length >= 2 ? (
                    <svg width="100%" height="100%" viewBox="0 0 100 32" preserveAspectRatio="none" class="overflow-visible">
                      <defs>
                        <linearGradient id="sparklineGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                          <stop offset="0%" stop-color="rgb(34, 211, 238)" stop-opacity="0.3" />
                          <stop offset="100%" stop-color="rgb(34, 211, 238)" stop-opacity="0.05" />
                        </linearGradient>
                      </defs>
                      {/* Fill area under sparkline */}
                      <path
                        d={`${renderSparklinePath(memoryState.velocityHistory, 100, 28)} L100,28 L0,28 Z`}
                        fill="url(#sparklineGradient)"
                      />
                      {/* Sparkline stroke */}
                      <path
                        d={renderSparklinePath(memoryState.velocityHistory, 100, 28)}
                        fill="none"
                        stroke="rgb(34, 211, 238)"
                        stroke-width="1.5"
                        stroke-linecap="round"
                        stroke-linejoin="round"
                      />
                      {/* Current value dot */}
                      {memoryState.velocityHistory.length > 0 && (
                        <circle
                          cx="100"
                          cy={28 - ((memoryState.velocityHistory[memoryState.velocityHistory.length - 1] - Math.min(...memoryState.velocityHistory)) / (Math.max(...memoryState.velocityHistory, 1) - Math.min(...memoryState.velocityHistory, 0) || 1)) * 28}
                          r="2"
                          fill="rgb(34, 211, 238)"
                        />
                      )}
                    </svg>
                  ) : (
                    <div class="h-full flex items-center justify-center text-[9px] text-muted-foreground/50">
                      Collecting data...
                    </div>
                  )}
                </div>
                <div class="flex justify-between text-[8px] text-muted-foreground/50 mt-1">
                  <span>-{memoryState.velocityHistory.length * 10}s</span>
                  <span>now</span>
                </div>
              </div>
            </div>

            {/* Column 2: Recent Records */}
            <div class="space-y-3">
              <h4 class="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-2">RECENT RECORDS</h4>
              {recentRecords.length === 0 ? (
                <div class="text-xs text-muted-foreground/60 italic">Loading records...</div>
              ) : (
                <div class="space-y-2 max-h-48 overflow-y-auto">
                  {recentRecords.map((record) => (
                    <div key={record.id} class="p-2 rounded bg-muted/30 border border-border hover:bg-muted/50 transition-colors">
                      <div class="flex items-center justify-between text-[10px] mb-1">
                        <div class="flex items-center gap-1.5">
                          <span class={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${record.contentType === 'prompt' ? 'bg-cyan-500/20 text-cyan-400' :
                            record.contentType === 'response' ? 'bg-green-500/20 text-green-400' :
                              record.contentType === 'tool_call' ? 'bg-orange-500/20 text-orange-400' :
                                record.contentType === 'fact' ? 'bg-blue-500/20 text-blue-400' :
                                  record.contentType === 'error' ? 'bg-red-500/20 text-red-400' :
                                    'bg-gray-500/20 text-gray-400'
                            }`}>{record.contentType}</span>
                          <span class={`px-1.5 py-0.5 rounded text-[9px] font-semibold uppercase ${semanticBadgeClass(record.typesTag.semantics)}`}>
                            {record.typesTag.semantics}
                          </span>
                        </div>
                        <span class="text-muted-foreground">{record.metadata.age}</span>
                      </div>
                      <div class="text-xs font-mono text-foreground/80 break-words">{record.content}</div>
                      <div class="flex items-center gap-2 mt-1 text-[9px] text-muted-foreground">
                        <span>{record.actor}</span>
                        <span>•</span>
                        <span class="truncate">{record.topic}</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Column 3: Store Capacity + Metadata */}
            <div class="space-y-3">
              <h4 class="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-2">STORE CAPACITY</h4>
              <div class="space-y-2">
                {/* FalkorDB Graph (Primary) */}
                <div class="flex items-center gap-3 text-[10px] font-medium text-muted-foreground">
                  <span class="w-20 flex items-center gap-1">
                    <span class={`w-1.5 h-1.5 rounded-full ${
                      memoryState.falkordbHealth === 'healthy' ? 'bg-green-400' :
                      memoryState.falkordbHealth === 'degraded' ? 'bg-yellow-400' : 'bg-red-400'
                    }`} />
                    FalkorDB
                  </span>
                  <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div class="h-full bg-orange-500 rounded-full transition-all" style={{ width: `${memoryState.falkordbSyncPct || 0}%` }} />
                  </div>
                  <span class="w-12 text-right font-mono">{memoryState.falkordbSyncPct.toFixed(0)}%</span>
                </div>
                <div class="flex items-center gap-3 text-[10px] font-medium text-muted-foreground">
                  <span class="w-20">Vector DB</span>
                  <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div class="h-full bg-purple-500 rounded-full transition-all" style={{ width: `${memoryState.vectorCapacityPct || 64}%` }} />
                  </div>
                  <span class="w-12 text-right font-mono">{memoryState.vectorCapacityPct || 64}%</span>
                </div>
                <div class="flex items-center gap-3 text-[10px] font-medium text-muted-foreground">
                  <span class="w-20">KG Nodes</span>
                  <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div class="h-full bg-cyan-500 rounded-full transition-all" style={{ width: `${memoryState.kgCapacityPct || 21}%` }} />
                  </div>
                  <span class="w-12 text-right font-mono">{memoryState.kgCapacityPct || 21}%</span>
                </div>
                <div class="flex items-center gap-3 text-[10px] font-medium text-muted-foreground">
                  <span class="w-20">Context</span>
                  <div class="flex-1 h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <div class="h-full bg-amber-500 rounded-full transition-all" style={{ width: `${memoryState.contextWindowPct || 45}%` }} />
                  </div>
                  <span class="w-12 text-right font-mono">{memoryState.contextWindowPct || 45}%</span>
                </div>
              </div>
              <div class="mt-4">
                <h4 class="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-2">FALKORDB GRAPH</h4>
                <div class="grid grid-cols-2 gap-2 text-[10px]">
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Nodes</span>
                    <span class="font-mono text-orange-400">{formatNumber(memoryState.falkordbNodes)}</span>
                  </div>
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Edges</span>
                    <span class="font-mono text-amber-400">{formatNumber(memoryState.falkordbEdges)}</span>
                  </div>
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Latency</span>
                    <span class="font-mono text-teal-400">{memoryState.falkordbLatencyMs.toFixed(0)}ms</span>
                  </div>
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Sync</span>
                    <span class={`font-mono ${memoryState.falkordbSyncPct >= 99 ? 'text-green-400' : memoryState.falkordbSyncPct >= 90 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {memoryState.falkordbSyncPct.toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
              <div class="mt-4">
                <h4 class="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-2">TOP LABELS</h4>
                {memoryState.falkordbLabelStats.length === 0 ? (
                  <div class="text-xs text-muted-foreground/60 italic">No label stats yet</div>
                ) : (
                  <div class="space-y-1">
                    {memoryState.falkordbLabelStats.map((label) => (
                      <div key={label.label} class="flex items-center gap-2 text-[10px]">
                        <span class="w-16 truncate text-muted-foreground" title={label.label}>{label.label}</span>
                        <div class="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
                          <div
                            class="h-full bg-cyan-500/70 rounded-full transition-all"
                            style={{ width: `${Math.max(2, label.pct)}%` }}
                          />
                        </div>
                        <span class="w-12 text-right font-mono text-cyan-300">{formatNumber(label.count)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
              <div class="mt-4">
                <h4 class="text-[10px] font-bold tracking-widest text-muted-foreground uppercase mb-2">METADATA</h4>
                <div class="grid grid-cols-2 gap-2 text-[10px]">
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Records/min</span>
                    <span class="font-mono text-cyan-400">{memoryState.ingestVelocity.toFixed(1)}</span>
                  </div>
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">CMP Score</span>
                    <span class="font-mono text-pink-400">{memoryState.cmpScore.toFixed(3)}</span>
                  </div>
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Entropy</span>
                    <span class="font-mono text-green-400">{memoryState.entropy.toFixed(4)}</span>
                  </div>
                  <div class="flex justify-between p-1.5 rounded bg-muted/30">
                    <span class="text-muted-foreground">Total Facts</span>
                    <span class="font-mono text-blue-400">{formatNumber(memoryState.kgFacts)}</span>
                  </div>
                </div>
              </div>
              {memoryState.lastIngestTs > 0 && (
                <div class="text-[9px] text-muted-foreground/60 text-right mt-2">
                  Last update: {formatAge(memoryState.lastIngestTs)} ago
                </div>
              )}
            </div>
          </div> {/* Close grid wrapper */}
        </div>
      )}
    </section>
  );
});
