/**
 * BusSubscription - WebSocket Bus Client for Chromatic Visualization
 *
 * Steps 16-17: Dashboard Component Integration & Performance Optimization
 *
 * Features:
 * - Connects to ws://localhost:9200 (bus bridge)
 * - Subscribes to VIZ_TOPICS from types.ts
 * - Parses NDJSON events
 * - Emits to mutation queue
 * - Automatic reconnect on disconnect with exponential backoff
 */

import type { AgentVisualEvent, CodeGraph, VizTopic } from './types';
import { VIZ_TOPICS, AgentVisualState } from './types';
import type { BusEvent } from '../../lib/state/types';

// =============================================================================
// Types
// =============================================================================

export interface BusSubscriptionOptions {
  /** WebSocket URL (default: ws://localhost:9200/ws/bus) */
  wsUrl?: string;
  /** Topics to subscribe to (default: VIZ_TOPICS) */
  topics?: readonly string[];
  /** Reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Max reconnect attempts (default: 10) */
  maxReconnectAttempts?: number;
  /** Initial reconnect delay ms (default: 1000) */
  initialReconnectDelay?: number;
  /** Max reconnect delay ms (default: 30000) */
  maxReconnectDelay?: number;
  /** Ping interval ms (default: 30000) */
  pingInterval?: number;
  /** Connection timeout ms (default: 10000) */
  connectionTimeout?: number;
}

export interface BusSubscriptionState {
  connected: boolean;
  reconnecting: boolean;
  reconnectAttempts: number;
  eventsPerMinute: number;
  lastEventTime: number;
  error: string | null;
}

export type BusEventHandler = (event: BusEvent) => void;
export type VisualizationEventHandler = (event: AgentVisualEvent) => void;
export type StateChangeHandler = (state: BusSubscriptionState) => void;

// =============================================================================
// BusSubscription Class
// =============================================================================

export class BusSubscription {
  private ws: WebSocket | null = null;
  private options: Required<BusSubscriptionOptions>;
  private state: BusSubscriptionState;
  private handlers: Set<BusEventHandler> = new Set();
  private vizHandlers: Set<VisualizationEventHandler> = new Set();
  private stateHandlers: Set<StateChangeHandler> = new Set();
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private pingTimer: ReturnType<typeof setInterval> | null = null;
  private connectionTimeout: ReturnType<typeof setTimeout> | null = null;
  private eventCountWindow: number[] = [];
  private disposed = false;

  constructor(options: BusSubscriptionOptions = {}) {
    this.options = {
      wsUrl: options.wsUrl ?? this.detectWsUrl(),
      topics: options.topics ?? VIZ_TOPICS,
      autoReconnect: options.autoReconnect ?? true,
      maxReconnectAttempts: options.maxReconnectAttempts ?? 10,
      initialReconnectDelay: options.initialReconnectDelay ?? 1000,
      maxReconnectDelay: options.maxReconnectDelay ?? 30000,
      pingInterval: options.pingInterval ?? 30000,
      connectionTimeout: options.connectionTimeout ?? 10000,
    };

    this.state = {
      connected: false,
      reconnecting: false,
      reconnectAttempts: 0,
      eventsPerMinute: 0,
      lastEventTime: 0,
      error: null,
    };
  }

  // ===========================================================================
  // Public API
  // ===========================================================================

  /**
   * Connect to the bus bridge WebSocket
   */
  async connect(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    if (this.disposed) {
      throw new Error('BusSubscription has been disposed');
    }

    return new Promise((resolve, reject) => {
      this.clearTimers();

      try {
        this.ws = new WebSocket(this.options.wsUrl);
      } catch (err) {
        this.updateState({ error: `Failed to create WebSocket: ${err}` });
        reject(err);
        return;
      }

      // Connection timeout
      this.connectionTimeout = setTimeout(() => {
        if (this.ws?.readyState !== WebSocket.OPEN) {
          this.ws?.close();
          const error = 'Connection timeout';
          this.updateState({ error });
          reject(new Error(error));
        }
      }, this.options.connectionTimeout);

      this.ws.onopen = () => {
        this.clearConnectionTimeout();
        this.updateState({
          connected: true,
          reconnecting: false,
          reconnectAttempts: 0,
          error: null,
        });

        // Subscribe to topics
        this.subscribeToTopics();

        // Start ping timer to keep connection alive
        this.startPingTimer();

        resolve();
      };

      this.ws.onmessage = (event) => {
        this.handleMessage(event.data);
      };

      this.ws.onerror = (event) => {
        console.error('[BusSubscription] WebSocket error:', event);
        this.updateState({ error: 'WebSocket error' });
      };

      this.ws.onclose = (event) => {
        this.clearTimers();
        this.updateState({ connected: false });

        if (event.wasClean) {
          console.log('[BusSubscription] Connection closed cleanly');
        } else {
          console.warn('[BusSubscription] Connection lost, code:', event.code);
        }

        // Attempt reconnect if enabled
        if (this.options.autoReconnect && !this.disposed) {
          this.scheduleReconnect();
        }
      };
    });
  }

  /**
   * Disconnect from the bus bridge
   */
  disconnect(): void {
    this.disposed = true;
    this.clearTimers();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.updateState({
      connected: false,
      reconnecting: false,
      error: null,
    });
  }

  /**
   * Subscribe to raw bus events
   */
  onEvent(handler: BusEventHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  /**
   * Subscribe to parsed visualization events
   */
  onVisualizationEvent(handler: VisualizationEventHandler): () => void {
    this.vizHandlers.add(handler);
    return () => this.vizHandlers.delete(handler);
  }

  /**
   * Subscribe to state changes
   */
  onStateChange(handler: StateChangeHandler): () => void {
    this.stateHandlers.add(handler);
    // Immediately emit current state
    handler(this.state);
    return () => this.stateHandlers.delete(handler);
  }

  /**
   * Get current connection state
   */
  getState(): Readonly<BusSubscriptionState> {
    return { ...this.state };
  }

  /**
   * Publish an event to the bus
   */
  publish(event: Partial<BusEvent> & { topic: string; kind: string; actor: string; data: unknown }): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      console.warn('[BusSubscription] Cannot publish - not connected');
      return;
    }

    const fullEvent: BusEvent = {
      ...event,
      ts: Date.now() / 1000,
      iso: new Date().toISOString(),
      level: event.level ?? 'info',
    };

    this.ws.send(JSON.stringify({ type: 'publish', event: fullEvent }));
  }

  // ===========================================================================
  // Private Methods
  // ===========================================================================

  private detectWsUrl(): string {
    if (typeof window !== 'undefined') {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      return `${protocol}//${host}/ws/bus`;
    }
    return 'ws://localhost:9200/ws/bus';
  }

  private subscribeToTopics(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    // Request initial sync
    this.ws.send(JSON.stringify({ type: 'sync' }));

    // Subscribe to each visualization topic
    for (const topic of this.options.topics) {
      this.ws.send(JSON.stringify({ type: 'subscribe', topic }));
    }

    // Also subscribe to wildcards for paip and agent events
    this.ws.send(JSON.stringify({ type: 'subscribe', topic: 'paip.*' }));
    this.ws.send(JSON.stringify({ type: 'subscribe', topic: 'agent.*' }));
    this.ws.send(JSON.stringify({ type: 'subscribe', topic: 'viz.*' }));
    this.ws.send(JSON.stringify({ type: 'subscribe', topic: 'git.*' }));
  }

  private handleMessage(data: string): void {
    try {
      const parsed = JSON.parse(data);

      if (parsed.type === 'event' || parsed.type === 'sync_topic') {
        const event = parsed.event as BusEvent;
        if (event && typeof event.topic === 'string') {
          this.processEvent(event);
        }
      } else if (parsed.type === 'sync' && Array.isArray(parsed.events)) {
        // Process initial sync events
        for (const event of parsed.events) {
          if (this.isRelevantTopic(event.topic)) {
            this.processEvent(event);
          }
        }
      }
    } catch (err) {
      // Ignore malformed messages
      console.debug('[BusSubscription] Failed to parse message:', err);
    }
  }

  private processEvent(event: BusEvent): void {
    // Track event rate
    const now = Date.now();
    this.eventCountWindow.push(now);
    // Keep only last 60 seconds
    this.eventCountWindow = this.eventCountWindow.filter((t) => now - t < 60000);
    this.updateState({
      lastEventTime: now,
      eventsPerMinute: this.eventCountWindow.length,
    });

    // Emit to raw handlers
    for (const handler of this.handlers) {
      try {
        handler(event);
      } catch (err) {
        console.error('[BusSubscription] Handler error:', err);
      }
    }

    // Parse and emit visualization events
    const vizEvent = this.parseVisualizationEvent(event);
    if (vizEvent) {
      for (const handler of this.vizHandlers) {
        try {
          handler(vizEvent);
        } catch (err) {
          console.error('[BusSubscription] Viz handler error:', err);
        }
      }
    }
  }

  private isRelevantTopic(topic: string): boolean {
    for (const pattern of this.options.topics) {
      if (pattern === topic) return true;
      if (pattern.endsWith('*') && topic.startsWith(pattern.slice(0, -1))) return true;
    }
    // Also check common patterns
    if (topic.startsWith('paip.') || topic.startsWith('agent.') ||
        topic.startsWith('viz.') || topic.startsWith('git.')) {
      return true;
    }
    return false;
  }

  private parseVisualizationEvent(event: BusEvent): AgentVisualEvent | null {
    const data = event.data as Record<string, unknown> | undefined;
    if (!data) return null;

    // Extract agent ID from topic or data
    let agentId = data.agent_id as string | undefined;
    if (!agentId) {
      // Try to infer from topic (e.g., "paip.clone.created" with data.actor = "claude")
      agentId = data.actor as string | undefined;
      if (!agentId) {
        agentId = event.actor;
      }
    }

    // Normalize agent ID
    const normalizedAgentId = this.normalizeAgentId(agentId);
    if (!normalizedAgentId) return null;

    // Determine state from topic/data
    const state = this.inferState(event.topic, data);

    // Extract or construct code graph
    const codeGraph = data.code_graph as CodeGraph | null | undefined;

    return {
      agent_id: normalizedAgentId,
      branch: (data.branch as string) || `${normalizedAgentId.toUpperCase()}_WORKING`,
      clone_path: (data.clone_path as string) || `/tmp/pluribus_${normalizedAgentId}_${Date.now()}`,
      state,
      color_hue: this.getAgentHue(normalizedAgentId),
      code_graph: codeGraph ?? null,
      activity_intensity: this.inferIntensity(state, data),
      timestamp_iso: event.iso,
      trace_id: event.trace_id,
    };
  }

  private normalizeAgentId(id: string | undefined): 'claude' | 'qwen' | 'gemini' | 'codex' | 'main' | null {
    if (!id) return null;
    const lower = id.toLowerCase();
    if (lower.includes('claude')) return 'claude';
    if (lower.includes('qwen')) return 'qwen';
    if (lower.includes('gemini')) return 'gemini';
    if (lower.includes('codex') || lower.includes('openai')) return 'codex';
    if (lower === 'main' || lower === 'operator') return 'main';
    return null;
  }

  private inferState(topic: string, data: Record<string, unknown>): AgentVisualState {
    // Check explicit state in data
    if (data.state && typeof data.state === 'string') {
      const state = data.state.toLowerCase();
      if (state in AgentVisualState) {
        return state as AgentVisualState;
      }
    }

    // Infer from topic
    if (topic.includes('clone.created') || topic.includes('cloning')) {
      return AgentVisualState.CLONING;
    }
    if (topic.includes('clone.deleted') || topic.includes('cleanup')) {
      return AgentVisualState.CLEANUP;
    }
    if (topic.includes('commit')) {
      return AgentVisualState.COMMITTING;
    }
    if (topic.includes('push')) {
      return AgentVisualState.PUSHING;
    }
    if (topic.includes('merge')) {
      return AgentVisualState.MERGED;
    }
    if (topic.includes('codegraph') || topic.includes('working')) {
      return AgentVisualState.WORKING;
    }

    // Default to working if activity detected
    return AgentVisualState.WORKING;
  }

  private inferIntensity(state: AgentVisualState, data: Record<string, unknown>): number {
    // Check explicit intensity
    if (typeof data.activity_intensity === 'number') {
      return Math.max(0, Math.min(1, data.activity_intensity));
    }
    if (typeof data.intensity === 'number') {
      return Math.max(0, Math.min(1, data.intensity));
    }

    // Infer from state
    switch (state) {
      case AgentVisualState.IDLE:
        return 0;
      case AgentVisualState.CLEANUP:
        return 0.1;
      case AgentVisualState.CLONING:
        return 0.3;
      case AgentVisualState.WORKING:
        return 0.6;
      case AgentVisualState.COMMITTING:
        return 0.9;
      case AgentVisualState.PUSHING:
        return 0.8;
      case AgentVisualState.MERGED:
        return 1.0;
      default:
        return 0.5;
    }
  }

  private getAgentHue(agentId: string): number {
    const hues: Record<string, number> = {
      claude: 300,  // Magenta
      qwen: 180,    // Cyan
      gemini: 60,   // Yellow
      codex: 120,   // Green
      main: 0,      // White (achromatic)
    };
    return hues[agentId] ?? 0;
  }

  private scheduleReconnect(): void {
    if (this.disposed) return;
    if (this.state.reconnectAttempts >= this.options.maxReconnectAttempts) {
      console.error('[BusSubscription] Max reconnect attempts reached');
      this.updateState({ error: 'Max reconnect attempts reached' });
      return;
    }

    // Exponential backoff with jitter
    const attempt = this.state.reconnectAttempts;
    const delay = Math.min(
      this.options.maxReconnectDelay,
      this.options.initialReconnectDelay * Math.pow(2, attempt)
    );
    const jitter = delay * 0.2 * Math.random();
    const totalDelay = delay + jitter;

    this.updateState({
      reconnecting: true,
      reconnectAttempts: attempt + 1,
    });

    console.log(`[BusSubscription] Reconnecting in ${Math.round(totalDelay)}ms (attempt ${attempt + 1})`);

    this.reconnectTimer = setTimeout(() => {
      this.connect().catch((err) => {
        console.error('[BusSubscription] Reconnect failed:', err);
      });
    }, totalDelay);
  }

  private startPingTimer(): void {
    this.pingTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        try {
          this.ws.send(JSON.stringify({ type: 'ping', ts: Date.now() }));
        } catch {
          // Ignore ping failures
        }
      }
    }, this.options.pingInterval);
  }

  private clearTimers(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.pingTimer) {
      clearInterval(this.pingTimer);
      this.pingTimer = null;
    }
    this.clearConnectionTimeout();
  }

  private clearConnectionTimeout(): void {
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }
  }

  private updateState(partial: Partial<BusSubscriptionState>): void {
    this.state = { ...this.state, ...partial };
    for (const handler of this.stateHandlers) {
      try {
        handler(this.state);
      } catch (err) {
        console.error('[BusSubscription] State handler error:', err);
      }
    }
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let defaultInstance: BusSubscription | null = null;

/**
 * Get the default BusSubscription instance (singleton)
 */
export function getDefaultBusSubscription(): BusSubscription {
  if (!defaultInstance) {
    defaultInstance = new BusSubscription();
  }
  return defaultInstance;
}

/**
 * Create a new BusSubscription instance with custom options
 */
export function createBusSubscription(options?: BusSubscriptionOptions): BusSubscription {
  return new BusSubscription(options);
}

export default BusSubscription;
