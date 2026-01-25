/**
 * Metrics Client: TypeScript client for fetching compiled metrics
 * ================================================================
 *
 * Provides functions to fetch metric summaries, KPIs, and trends
 * from the Pluribus metrics library API endpoints.
 *
 * Usage:
 *   import { getMetricsSnapshot, detectAnomalies, getTrends } from './metrics-client';
 *
 *   const snapshot = await getMetricsSnapshot({ windowSeconds: 60 });
 *   const anomalies = await detectAnomalies(snapshot);
 *   const trends = await getTrends(['velocity', 'error_rate']);
 */

import type { BusEvent } from './state/types';

// ============================================================================
// Types
// ============================================================================

export type AnomalyType =
  | 'velocity_drop'
  | 'error_spike'
  | 'latency_violation'
  | 'agent_silent'
  | 'queue_backlog'
  | 'topology_imbalance'
  | 'entropy_drift';

export type SeverityLevel = 'info' | 'warn' | 'error' | 'critical';

export type TrendDirection = 'increasing' | 'decreasing' | 'stable' | 'volatile';

export interface AgentMetrics {
  actor: string;
  event_count: number;
  error_count: number;
  last_seen_ts: number;
  avg_latency_ms: number;
  topics: string[];
  health: string;
  queue_depth: number;
}

export interface TopicMetrics {
  topic: string;
  event_count: number;
  error_count: number;
  avg_latency_ms: number;
  actors: string[];
  first_seen_ts: number;
  last_seen_ts: number;
}

export interface TopologyMetrics {
  single_count: number;
  star_count: number;
  peer_debate_count: number;
  avg_fanout: number;
  coordination_budget_total: number;
}

export interface EvolutionaryMetrics {
  vgt_transfers: number;
  hgt_transfers: number;
  total_generations: number;
  avg_speciation_potential: number;
  lineage_health_distribution: Record<string, number>;
}

export interface EntropyMetrics {
  topic_entropy: number;
  actor_entropy: number;
  level_entropy: number;
  causal_depth_avg: number;
  reversibility_avg: number;
}

export interface LatencyMetrics {
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
}

export interface KPIMetrics {
  total_events: number;
  total_errors: number;
  velocity: number;
  error_rate: number;
  latency: LatencyMetrics;
}

export interface MetricsSnapshot {
  snapshot_id: string;
  ts: number;
  iso: string;
  window_seconds: number;
  kpis: KPIMetrics;
  agents: {
    count: number;
    active: number;
    silent: number;
    details: Record<string, AgentMetrics>;
  };
  topics: {
    count: number;
    details: Record<string, TopicMetrics>;
  };
  topology: TopologyMetrics;
  evolutionary: EvolutionaryMetrics;
  entropy: EntropyMetrics;
  queue: {
    depth: number;
    pending: number;
    completed: number;
  };
}

export interface Anomaly {
  anomaly_id: string;
  ts: number;
  iso: string;
  anomaly_type: AnomalyType;
  severity: SeverityLevel;
  metric_name: string;
  current_value: number;
  threshold: number;
  deviation_sigma: number;
  description: string;
  suggested_actions: string[];
  context: Record<string, unknown>;
}

export interface CompressedSummary {
  summary_id: string;
  ts: number;
  iso: string;
  headline: string;
  body: string;
  severity: SeverityLevel;
  actor: string;
  topic: string;
  impact: string;
  actionable: string[];
  key_metrics: Record<string, number>;
  original_size_bytes: number;
  compressed_size_bytes: number;
  compression_ratio: number;
}

export interface TrendAnalysis {
  metric_name: string;
  direction: TrendDirection;
  slope: number;
  volatility: number;
  samples: number;
  prediction_next: number;
  confidence: number;
}

export interface CompressionResult {
  original: string;
  compressed: string;
  model: string;
  method: 'ollama' | 'heuristic' | 'cache';
  compression_ratio: number;
  input_tokens_estimate: number;
  output_tokens_estimate: number;
  latency_ms: number;
  cache_key: string;
  success: boolean;
  error?: string;
}

// ============================================================================
// Configuration
// ============================================================================

export interface MetricsClientOptions {
  baseUrl?: string;
  timeout?: number;
  headers?: Record<string, string>;
}

const DEFAULT_OPTIONS: Required<MetricsClientOptions> = {
  baseUrl: '/api/metrics',
  timeout: 10000,
  headers: {},
};

// ============================================================================
// API Client
// ============================================================================

class MetricsClient {
  private options: Required<MetricsClientOptions>;

  constructor(options: MetricsClientOptions = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  private async fetch<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.options.timeout);

    try {
      const response = await fetch(`${this.options.baseUrl}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...this.options.headers,
          ...options.headers,
        },
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      return await response.json();
    } finally {
      clearTimeout(timeoutId);
    }
  }

  /**
   * Get current metrics snapshot
   */
  async getSnapshot(params: { windowSeconds?: number } = {}): Promise<MetricsSnapshot> {
    const query = params.windowSeconds ? `?window=${params.windowSeconds}` : '';
    return this.fetch<MetricsSnapshot>(`/snapshot${query}`);
  }

  /**
   * Detect anomalies in current or provided snapshot
   */
  async detectAnomalies(
    snapshot?: MetricsSnapshot,
    thresholds?: Record<string, number>
  ): Promise<Anomaly[]> {
    return this.fetch<Anomaly[]>('/anomalies', {
      method: 'POST',
      body: JSON.stringify({ snapshot, thresholds }),
    });
  }

  /**
   * Get trend analysis for specified metrics
   */
  async getTrends(metricPaths: string[]): Promise<TrendAnalysis[]> {
    return this.fetch<TrendAnalysis[]>('/trends', {
      method: 'POST',
      body: JSON.stringify({ metrics: metricPaths }),
    });
  }

  /**
   * Compress a message for notification
   */
  async compressMessage(
    message: string,
    options: { model?: string; maxLength?: number } = {}
  ): Promise<CompressionResult> {
    return this.fetch<CompressionResult>('/compress', {
      method: 'POST',
      body: JSON.stringify({
        message,
        model: options.model,
        max_length: options.maxLength,
      }),
    });
  }

  /**
   * Get compression statistics
   */
  async getCompressionStats(): Promise<{
    total_compressions: number;
    cache_hits: number;
    avg_compression_ratio: number;
  }> {
    return this.fetch('/compression/stats');
  }

  /**
   * Summarize a bus event for notification
   */
  async summarizeEvent(event: BusEvent): Promise<CompressedSummary> {
    return this.fetch<CompressedSummary>('/summarize', {
      method: 'POST',
      body: JSON.stringify({ event }),
    });
  }
}

// Default client instance
let defaultClient: MetricsClient | null = null;

function getClient(options?: MetricsClientOptions): MetricsClient {
  if (!defaultClient) {
    defaultClient = new MetricsClient(options);
  }
  return defaultClient;
}

// ============================================================================
// Local Computation (When API unavailable)
// ============================================================================

/**
 * Compile KPIs from bus events locally (client-side computation)
 */
export function compileKPIsLocal(
  events: BusEvent[],
  windowSeconds: number = 60
): Partial<MetricsSnapshot> {
  const now = Date.now() / 1000;
  const cutoff = now - windowSeconds;

  // Filter events in window
  const windowEvents = events.filter((e) => e.ts >= cutoff);

  if (windowEvents.length === 0) {
    return {
      ts: now,
      iso: new Date().toISOString(),
      window_seconds: windowSeconds,
      kpis: {
        total_events: 0,
        total_errors: 0,
        velocity: 0,
        error_rate: 0,
        latency: { avg_ms: 0, p50_ms: 0, p95_ms: 0, p99_ms: 0 },
      },
    };
  }

  // Count events and errors
  const totalEvents = windowEvents.length;
  const totalErrors = windowEvents.filter((e) => e.level === 'error').length;

  // Extract latencies
  const latencies: number[] = [];
  for (const event of windowEvents) {
    const data = event.data as Record<string, unknown> | undefined;
    if (data && typeof data.latency_ms === 'number') {
      latencies.push(data.latency_ms);
    }
  }

  // Calculate percentiles
  const sorted = [...latencies].sort((a, b) => a - b);
  const p50 = percentile(sorted, 50);
  const p95 = percentile(sorted, 95);
  const p99 = percentile(sorted, 99);
  const avg = latencies.length > 0 ? latencies.reduce((a, b) => a + b, 0) / latencies.length : 0;

  // Count agents and topics
  const agents = new Set(windowEvents.map((e) => e.actor));
  const topics = new Set(windowEvents.map((e) => e.topic));

  // Topology counts
  let starCount = 0;
  let peerDebateCount = 0;
  const fanouts: number[] = [];

  for (const event of windowEvents) {
    const topo = (event as unknown as { topology?: TopologyMetrics }).topology;
    if (topo) {
      if ((topo as unknown as { topology?: string }).topology === 'star') starCount++;
      if ((topo as unknown as { topology?: string }).topology === 'peer_debate') peerDebateCount++;
      if (typeof topo.avg_fanout === 'number') fanouts.push(topo.avg_fanout);
    }
  }

  return {
    snapshot_id: crypto.randomUUID?.() || `local-${Date.now()}`,
    ts: now,
    iso: new Date().toISOString(),
    window_seconds: windowSeconds,
    kpis: {
      total_events: totalEvents,
      total_errors: totalErrors,
      velocity: totalEvents / windowSeconds,
      error_rate: totalErrors / totalEvents,
      latency: {
        avg_ms: avg,
        p50_ms: p50,
        p95_ms: p95,
        p99_ms: p99,
      },
    },
    agents: {
      count: agents.size,
      active: agents.size,
      silent: 0,
      details: {},
    },
    topics: {
      count: topics.size,
      details: {},
    },
    topology: {
      single_count: windowEvents.length - starCount - peerDebateCount,
      star_count: starCount,
      peer_debate_count: peerDebateCount,
      avg_fanout: fanouts.length > 0 ? fanouts.reduce((a, b) => a + b, 0) / fanouts.length : 1,
      coordination_budget_total: 0,
    },
  };
}

/**
 * Detect anomalies locally (client-side)
 */
export function detectAnomaliesLocal(
  snapshot: Partial<MetricsSnapshot>,
  thresholds: Record<string, number> = {}
): Anomaly[] {
  const anomalies: Anomaly[] = [];
  const now = Date.now() / 1000;

  const defaults = {
    min_velocity: 0.1,
    max_error_rate: 0.1,
    max_latency_p95_ms: 2000,
    max_queue_depth: 50,
  };

  const t = { ...defaults, ...thresholds };
  const kpis = snapshot.kpis;

  if (!kpis) return anomalies;

  // Velocity drop
  if (kpis.velocity < t.min_velocity) {
    anomalies.push({
      anomaly_id: `v-${Date.now()}`,
      ts: now,
      iso: new Date().toISOString(),
      anomaly_type: 'velocity_drop',
      severity: 'warn',
      metric_name: 'velocity',
      current_value: kpis.velocity,
      threshold: t.min_velocity,
      deviation_sigma: 0,
      description: `Velocity ${kpis.velocity.toFixed(3)} MPS below threshold ${t.min_velocity}`,
      suggested_actions: ['Check worker health', 'Review queue processing'],
      context: {},
    });
  }

  // Error spike
  if (kpis.error_rate > t.max_error_rate) {
    anomalies.push({
      anomaly_id: `e-${Date.now()}`,
      ts: now,
      iso: new Date().toISOString(),
      anomaly_type: 'error_spike',
      severity: kpis.error_rate > 0.3 ? 'critical' : 'error',
      metric_name: 'error_rate',
      current_value: kpis.error_rate,
      threshold: t.max_error_rate,
      deviation_sigma: 0,
      description: `Error rate ${(kpis.error_rate * 100).toFixed(1)}% exceeds threshold`,
      suggested_actions: ['Review error logs', 'Check provider health'],
      context: { total_errors: kpis.total_errors },
    });
  }

  // Latency violation
  if (kpis.latency.p95_ms > t.max_latency_p95_ms) {
    anomalies.push({
      anomaly_id: `l-${Date.now()}`,
      ts: now,
      iso: new Date().toISOString(),
      anomaly_type: 'latency_violation',
      severity: 'warn',
      metric_name: 'p95_latency_ms',
      current_value: kpis.latency.p95_ms,
      threshold: t.max_latency_p95_ms,
      deviation_sigma: 0,
      description: `P95 latency ${kpis.latency.p95_ms.toFixed(0)}ms exceeds SLA`,
      suggested_actions: ['Check provider response times', 'Consider backoff'],
      context: {},
    });
  }

  return anomalies;
}

/**
 * Summarize event locally (heuristic compression)
 */
export function summarizeEventLocal(event: BusEvent): CompressedSummary {
  const now = Date.now() / 1000;
  const data = event.data as Record<string, unknown> | undefined;

  // Generate headline from semantic or topic
  let headline = '';
  if (event.semantic) {
    headline = event.semantic.slice(0, 100);
  } else {
    const parts = event.topic.split('.');
    headline = parts.length >= 2
      ? `${parts[0].toUpperCase()}: ${parts.slice(1).join('.')}`
      : event.topic;
  }

  // Generate body from key fields
  const bodyParts: string[] = [];
  if (event.actor) bodyParts.push(`Actor: ${event.actor}`);
  if (event.kind) bodyParts.push(`Kind: ${event.kind}`);

  // Extract key metrics
  const keyMetrics: Record<string, number> = {};
  const metricKeys = ['latency_ms', 'velocity', 'error_rate', 'queue_depth'];
  if (data) {
    for (const key of metricKeys) {
      if (typeof data[key] === 'number') {
        keyMetrics[key] = data[key] as number;
        bodyParts.push(`${key}: ${data[key]}`);
      }
    }
  }

  const body = bodyParts.slice(0, 5).join(' | ');

  // Determine severity
  let severity: SeverityLevel = 'info';
  if (event.impact === 'critical') severity = 'critical';
  else if (event.impact === 'high') severity = 'error';
  else if (event.level === 'error') severity = 'error';
  else if (event.level === 'warn') severity = 'warn';

  const originalSize = JSON.stringify(event).length;
  const compressedSize = (headline + body).length;

  return {
    summary_id: `s-${Date.now()}`,
    ts: now,
    iso: new Date().toISOString(),
    headline,
    body,
    severity,
    actor: event.actor,
    topic: event.topic,
    impact: event.impact || 'low',
    actionable: event.actionable || [],
    key_metrics: keyMetrics,
    original_size_bytes: originalSize,
    compressed_size_bytes: compressedSize,
    compression_ratio: originalSize / Math.max(compressedSize, 1),
  };
}

/**
 * Calculate Shannon entropy of a distribution
 */
export function shannonEntropy(counts: Record<string, number>): number {
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  if (total <= 0) return 0;

  let entropy = 0;
  for (const count of Object.values(counts)) {
    if (count > 0) {
      const p = count / total;
      entropy -= p * Math.log2(p);
    }
  }
  return entropy;
}

// ============================================================================
// Helper Functions
// ============================================================================

function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  const idx = ((sorted.length - 1) * p) / 100;
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  if (upper >= sorted.length) return sorted[sorted.length - 1];
  const frac = idx - lower;
  return sorted[lower] * (1 - frac) + sorted[upper] * frac;
}

// ============================================================================
// Exported Functions (Use API when available, fall back to local)
// ============================================================================

/**
 * Get metrics snapshot (API with local fallback)
 */
export async function getMetricsSnapshot(
  options: { windowSeconds?: number; events?: BusEvent[] } = {}
): Promise<Partial<MetricsSnapshot>> {
  // If events provided, compute locally
  if (options.events) {
    return compileKPIsLocal(options.events, options.windowSeconds);
  }

  // Try API
  try {
    const client = getClient();
    return await client.getSnapshot({ windowSeconds: options.windowSeconds });
  } catch (error) {
    console.warn('Metrics API unavailable, returning empty snapshot:', error);
    return compileKPIsLocal([], options.windowSeconds);
  }
}

/**
 * Detect anomalies (API with local fallback)
 */
export async function detectAnomalies(
  snapshot: Partial<MetricsSnapshot>,
  thresholds?: Record<string, number>
): Promise<Anomaly[]> {
  try {
    const client = getClient();
    return await client.detectAnomalies(snapshot as MetricsSnapshot, thresholds);
  } catch (error) {
    console.warn('Anomaly detection API unavailable, using local:', error);
    return detectAnomaliesLocal(snapshot, thresholds);
  }
}

/**
 * Get trend analysis (API only)
 */
export async function getTrends(metricPaths: string[]): Promise<TrendAnalysis[]> {
  const client = getClient();
  return client.getTrends(metricPaths);
}

/**
 * Compress message for notification (API with local fallback)
 */
export async function compressMessage(
  message: string,
  options: { model?: string; maxLength?: number } = {}
): Promise<string> {
  try {
    const client = getClient();
    const result = await client.compressMessage(message, options);
    return result.compressed;
  } catch (error) {
    console.warn('Compression API unavailable, using heuristic:', error);
    // Basic heuristic: first sentence, truncated
    const firstSentence = message.split(/[.!?]/)[0] || message;
    const maxLen = options.maxLength || 280;
    return firstSentence.length > maxLen
      ? firstSentence.slice(0, maxLen - 3) + '...'
      : firstSentence;
  }
}

/**
 * Summarize event for notification (local computation)
 */
export function summarizeEvent(event: BusEvent): CompressedSummary {
  return summarizeEventLocal(event);
}

/**
 * Create metrics client with custom options
 */
export function createMetricsClient(options: MetricsClientOptions): MetricsClient {
  return new MetricsClient(options);
}

// ============================================================================
// Real-time Metrics Hooks (for React/Qwik)
// ============================================================================

export interface MetricsState {
  snapshot: Partial<MetricsSnapshot> | null;
  anomalies: Anomaly[];
  loading: boolean;
  error: Error | null;
  lastUpdated: number;
}

/**
 * Create a metrics polling function
 */
export function createMetricsPoller(
  onUpdate: (state: MetricsState) => void,
  options: { interval?: number; windowSeconds?: number } = {}
): { start: () => void; stop: () => void } {
  const interval = options.interval || 5000;
  const windowSeconds = options.windowSeconds || 60;
  let timer: ReturnType<typeof setInterval> | null = null;

  const poll = async () => {
    try {
      const snapshot = await getMetricsSnapshot({ windowSeconds });
      const anomalies = await detectAnomalies(snapshot);
      onUpdate({
        snapshot,
        anomalies,
        loading: false,
        error: null,
        lastUpdated: Date.now(),
      });
    } catch (error) {
      onUpdate({
        snapshot: null,
        anomalies: [],
        loading: false,
        error: error instanceof Error ? error : new Error(String(error)),
        lastUpdated: Date.now(),
      });
    }
  };

  return {
    start: () => {
      if (timer) return;
      onUpdate({ snapshot: null, anomalies: [], loading: true, error: null, lastUpdated: 0 });
      poll();
      timer = setInterval(poll, interval);
    },
    stop: () => {
      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    },
  };
}

// Default export for convenience
export default MetricsClient;
