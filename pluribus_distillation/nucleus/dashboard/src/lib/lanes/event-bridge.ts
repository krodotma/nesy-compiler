/**
 * Event Bridge - Store-Bus Synchronization
 *
 * Phase 7, Iteration 53 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Bidirectional sync between lanes store and Pluribus bus
 * - Event coalescing to reduce bus traffic
 * - Optimistic updates with server confirmation
 * - Conflict detection and resolution
 * - Offline queue for pending changes
 */

import type { Lane, LanesState, LaneAction } from './store';
import type { BusEvent } from '../state/types';

// ============================================================================
// Types
// ============================================================================

export interface BridgeEvent {
  id: string;
  type: 'lane.update' | 'lane.create' | 'lane.delete' | 'lanes.sync' | 'lane.conflict';
  timestamp: number;
  payload: unknown;
  source: 'local' | 'remote';
  confirmed: boolean;
}

export interface ConflictInfo {
  laneId: string;
  localVersion: Partial<Lane>;
  remoteVersion: Partial<Lane>;
  field: string;
  resolvedBy?: 'local' | 'remote' | 'merge';
}

export interface BridgeConfig {
  /** Coalesce window in milliseconds */
  coalesceWindow: number;
  /** Maximum queued events before flush */
  maxQueueSize: number;
  /** Enable conflict detection */
  detectConflicts: boolean;
  /** Auto-resolve strategy */
  conflictStrategy: 'local-wins' | 'remote-wins' | 'latest-wins' | 'manual';
  /** Bus endpoint */
  busEndpoint: string;
}

export type BridgeListener = (event: BridgeEvent) => void;
export type ConflictHandler = (conflict: ConflictInfo) => Promise<'local' | 'remote' | 'merge'>;

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: BridgeConfig = {
  coalesceWindow: 500,
  maxQueueSize: 50,
  detectConflicts: true,
  conflictStrategy: 'latest-wins',
  busEndpoint: '/api/bus/emit',
};

// ============================================================================
// Event Bridge Class
// ============================================================================

export class EventBridge {
  private config: BridgeConfig;
  private listeners: Set<BridgeListener> = new Set();
  private conflictHandlers: Set<ConflictHandler> = new Set();
  private pendingEvents: Map<string, BridgeEvent> = new Map();
  private coalesceTimer: ReturnType<typeof setTimeout> | null = null;
  private versionMap: Map<string, number> = new Map();
  private offlineQueue: BridgeEvent[] = [];
  private isOnline: boolean = true;

  constructor(config: Partial<BridgeConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ============================================================================
  // Public Methods
  // ============================================================================

  /**
   * Emit a lane action to the bus
   */
  emitAction(action: LaneAction): void {
    const event = this.actionToEvent(action);
    if (!event) return;

    this.queueEvent(event);
  }

  /**
   * Handle incoming bus event
   */
  handleBusEvent(busEvent: BusEvent): void {
    if (!this.isLanesEvent(busEvent)) return;

    const event = this.busEventToBridgeEvent(busEvent);
    if (!event) return;

    // Check for conflicts
    if (this.config.detectConflicts && event.type === 'lane.update') {
      const conflict = this.detectConflict(event);
      if (conflict) {
        this.handleConflict(conflict);
        return;
      }
    }

    // Update version tracking
    if (event.type === 'lane.update' || event.type === 'lane.create') {
      const payload = event.payload as { laneId: string };
      this.versionMap.set(payload.laneId, event.timestamp);
    }

    // Notify listeners
    this.notifyListeners(event);
  }

  /**
   * Subscribe to bridge events
   */
  subscribe(listener: BridgeListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Register conflict handler
   */
  onConflict(handler: ConflictHandler): () => void {
    this.conflictHandlers.add(handler);
    return () => this.conflictHandlers.delete(handler);
  }

  /**
   * Set online status
   */
  setOnline(online: boolean): void {
    const wasOffline = !this.isOnline;
    this.isOnline = online;

    // Flush offline queue when coming back online
    if (online && wasOffline && this.offlineQueue.length > 0) {
      this.flushOfflineQueue();
    }
  }

  /**
   * Force flush pending events
   */
  flush(): void {
    if (this.coalesceTimer) {
      clearTimeout(this.coalesceTimer);
      this.coalesceTimer = null;
    }
    this.processPendingEvents();
  }

  /**
   * Get pending event count
   */
  getPendingCount(): number {
    return this.pendingEvents.size + this.offlineQueue.length;
  }

  /**
   * Clear all pending events
   */
  clearPending(): void {
    this.pendingEvents.clear();
    this.offlineQueue = [];
    if (this.coalesceTimer) {
      clearTimeout(this.coalesceTimer);
      this.coalesceTimer = null;
    }
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private queueEvent(event: BridgeEvent): void {
    // Coalesce events for the same lane
    const key = this.getCoalesceKey(event);
    const existing = this.pendingEvents.get(key);

    if (existing && event.type === 'lane.update') {
      // Merge updates
      const merged = this.mergeEvents(existing, event);
      this.pendingEvents.set(key, merged);
    } else {
      this.pendingEvents.set(key, event);
    }

    // Start or reset coalesce timer
    if (this.coalesceTimer) {
      clearTimeout(this.coalesceTimer);
    }

    // Flush immediately if queue is full
    if (this.pendingEvents.size >= this.config.maxQueueSize) {
      this.processPendingEvents();
    } else {
      this.coalesceTimer = setTimeout(() => {
        this.processPendingEvents();
      }, this.config.coalesceWindow);
    }
  }

  private getCoalesceKey(event: BridgeEvent): string {
    switch (event.type) {
      case 'lane.update':
      case 'lane.delete':
        return `${event.type}:${(event.payload as any).laneId}`;
      case 'lane.create':
        return `create:${(event.payload as any).lane?.id || event.id}`;
      default:
        return event.id;
    }
  }

  private mergeEvents(existing: BridgeEvent, incoming: BridgeEvent): BridgeEvent {
    if (existing.type !== 'lane.update' || incoming.type !== 'lane.update') {
      return incoming;
    }

    const existingPayload = existing.payload as { laneId: string; changes: Partial<Lane> };
    const incomingPayload = incoming.payload as { laneId: string; changes: Partial<Lane> };

    return {
      ...incoming,
      payload: {
        laneId: incomingPayload.laneId,
        changes: {
          ...existingPayload.changes,
          ...incomingPayload.changes,
        },
      },
    };
  }

  private async processPendingEvents(): Promise<void> {
    if (this.pendingEvents.size === 0) return;

    const events = Array.from(this.pendingEvents.values());
    this.pendingEvents.clear();

    if (!this.isOnline) {
      // Queue for later
      this.offlineQueue.push(...events);
      return;
    }

    // Send to bus
    for (const event of events) {
      await this.sendToBus(event);
    }
  }

  private async sendToBus(event: BridgeEvent): Promise<void> {
    try {
      const busEvent = this.bridgeEventToBusEvent(event);
      await fetch(this.config.busEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(busEvent),
      });

      // Mark as confirmed
      event.confirmed = true;
      this.notifyListeners({ ...event, confirmed: true });
    } catch (err) {
      console.warn('Failed to send event to bus:', err);
      // Re-queue for retry
      this.offlineQueue.push(event);
    }
  }

  private async flushOfflineQueue(): Promise<void> {
    const queue = [...this.offlineQueue];
    this.offlineQueue = [];

    for (const event of queue) {
      await this.sendToBus(event);
    }
  }

  private detectConflict(event: BridgeEvent): ConflictInfo | null {
    if (event.type !== 'lane.update') return null;

    const payload = event.payload as { laneId: string; changes: Partial<Lane> };
    const localVersion = this.versionMap.get(payload.laneId);

    if (!localVersion) return null;

    // Check if there's a pending local change that conflicts
    const key = `lane.update:${payload.laneId}`;
    const pending = this.pendingEvents.get(key);

    if (pending && pending.source === 'local') {
      const pendingPayload = pending.payload as { laneId: string; changes: Partial<Lane> };

      // Find conflicting fields
      const conflictingField = Object.keys(payload.changes).find(
        field => field in pendingPayload.changes
      );

      if (conflictingField) {
        return {
          laneId: payload.laneId,
          localVersion: pendingPayload.changes,
          remoteVersion: payload.changes,
          field: conflictingField,
        };
      }
    }

    return null;
  }

  private async handleConflict(conflict: ConflictInfo): Promise<void> {
    let resolution: 'local' | 'remote' | 'merge';

    // Check for custom handlers
    if (this.conflictHandlers.size > 0) {
      const handler = Array.from(this.conflictHandlers)[0];
      resolution = await handler(conflict);
    } else {
      // Use configured strategy
      switch (this.config.conflictStrategy) {
        case 'local-wins':
          resolution = 'local';
          break;
        case 'remote-wins':
          resolution = 'remote';
          break;
        case 'latest-wins':
          resolution = 'remote'; // Remote is always "latest" in this context
          break;
        case 'manual':
          resolution = 'remote'; // Default to remote if no handler
          break;
        default:
          resolution = 'remote';
      }
    }

    conflict.resolvedBy = resolution;

    // Emit conflict event
    this.notifyListeners({
      id: `conflict-${Date.now()}`,
      type: 'lane.conflict',
      timestamp: Date.now(),
      payload: conflict,
      source: 'local',
      confirmed: true,
    });
  }

  private actionToEvent(action: LaneAction): BridgeEvent | null {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const timestamp = Date.now();

    switch (action.type) {
      case 'UPDATE_LANE':
        return {
          id,
          type: 'lane.update',
          timestamp,
          payload: {
            laneId: action.payload.id,
            changes: action.payload.changes,
          },
          source: 'local',
          confirmed: false,
        };

      case 'ADD_LANE':
        return {
          id,
          type: 'lane.create',
          timestamp,
          payload: { lane: action.payload },
          source: 'local',
          confirmed: false,
        };

      case 'REMOVE_LANE':
        return {
          id,
          type: 'lane.delete',
          timestamp,
          payload: { laneId: action.payload },
          source: 'local',
          confirmed: false,
        };

      default:
        return null;
    }
  }

  private busEventToBridgeEvent(busEvent: BusEvent): BridgeEvent | null {
    if (!busEvent.topic.startsWith('operator.lanes.')) return null;

    const id = busEvent.id || `${busEvent.ts}-${Math.random().toString(36).slice(2)}`;

    if (busEvent.topic === 'operator.lanes.update') {
      return {
        id,
        type: 'lane.update',
        timestamp: busEvent.ts,
        payload: busEvent.data,
        source: 'remote',
        confirmed: true,
      };
    }

    if (busEvent.topic === 'operator.lanes.state') {
      return {
        id,
        type: 'lanes.sync',
        timestamp: busEvent.ts,
        payload: busEvent.data,
        source: 'remote',
        confirmed: true,
      };
    }

    return null;
  }

  private bridgeEventToBusEvent(event: BridgeEvent): BusEvent {
    let topic = 'operator.lanes.update';
    if (event.type === 'lane.create') topic = 'operator.lanes.create';
    if (event.type === 'lane.delete') topic = 'operator.lanes.delete';

    return {
      topic,
      kind: event.type,
      level: 'info',
      actor: 'dashboard',
      ts: event.timestamp,
      iso: new Date(event.timestamp).toISOString(),
      data: event.payload,
    };
  }

  private isLanesEvent(event: BusEvent): boolean {
    return event.topic.startsWith('operator.lanes.');
  }

  private notifyListeners(event: BridgeEvent): void {
    this.listeners.forEach(listener => listener(event));
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalBridge: EventBridge | null = null;

export function getGlobalEventBridge(config?: Partial<BridgeConfig>): EventBridge {
  if (!globalBridge) {
    globalBridge = new EventBridge(config);
  }
  return globalBridge;
}

export function resetGlobalEventBridge(): void {
  if (globalBridge) {
    globalBridge.clearPending();
  }
  globalBridge = null;
}

export default EventBridge;
