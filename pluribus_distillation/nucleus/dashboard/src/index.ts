/**
 * Pluribus Dashboard - Main Entry Point
 *
 * Unified isomorphic dashboard for TUI/Web/Native/WASM.
 */

// Re-export all modules
export * from './lib/state';
export * from './lib/bus';
export * from './components';

// Version
export const VERSION = '0.1.0';

// Platform detection
export type Platform = 'node' | 'browser' | 'native' | 'wasm';

export function detectPlatform(): Platform {
  if (typeof globalThis.process !== 'undefined' && globalThis.process.versions?.node) {
    return 'node';
  }
  if (typeof globalThis.window !== 'undefined') {
    return 'browser';
  }
  if (typeof globalThis.navigator !== 'undefined' && globalThis.navigator.product === 'ReactNative') {
    return 'native';
  }
  return 'wasm';
}

// Dashboard configuration
export interface DashboardConfig {
  busPath?: string;
  wsUrl?: string;
  apiUrl?: string;
  theme?: 'light' | 'dark' | 'system' | 'chroma';
  pollInterval?: number;
}

// Create dashboard instance
import { createBusClient, type BusClient } from './lib/bus';
import { createDefaultState, type DashboardState, type DashboardAction } from './lib/state';
import { dashboardReducer, busEventToActions } from './lib/state/reducer';

export class Dashboard {
  private state: DashboardState;
  private bus: BusClient;
  private listeners: Set<(state: DashboardState) => void> = new Set();

  constructor(config: DashboardConfig = {}) {
    this.state = createDefaultState();
    this.bus = createBusClient({
      platform: detectPlatform(),
      busPath: config.busPath,
      wsUrl: config.wsUrl,
      pollInterval: config.pollInterval,
    });

    if (config.theme) {
      this.state.ui.theme = config.theme;
    }
  }

  async connect(): Promise<void> {
    await this.bus.connect();
    this.state.connected = true;
    this.notify();

    // Subscribe to all events
    this.bus.subscribe('*', (event) => {
      const actions = busEventToActions(event);
      for (const action of actions) {
        this.dispatch(action);
      }
    });

    // Load initial events
    const events = await this.bus.getEvents(100);
    this.dispatch({ type: 'SET_EVENTS', events });
  }

  disconnect(): void {
    this.bus.disconnect();
    this.state.connected = false;
    this.notify();
  }

  dispatch(action: DashboardAction): void {
    this.state = dashboardReducer(this.state, action);
    this.notify();
  }

  getState(): DashboardState {
    return this.state;
  }

  subscribe(listener: (state: DashboardState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notify(): void {
    for (const listener of this.listeners) {
      listener(this.state);
    }
  }

  // Service actions
  async startService(serviceId: string): Promise<void> {
    await this.bus.publish({
      topic: 'dashboard.command.start_service',
      kind: 'command',
      level: 'info',
      actor: 'dashboard',
      data: { service_id: serviceId },
    });
  }

  async stopService(serviceId: string): Promise<void> {
    await this.bus.publish({
      topic: 'dashboard.command.stop_service',
      kind: 'command',
      level: 'info',
      actor: 'dashboard',
      data: { service_id: serviceId },
    });
  }

  // VPS session actions
  async setFlowMode(mode: 'm' | 'A'): Promise<void> {
    await this.bus.publish({
      topic: 'dashboard.vps.set_flow_mode',
      kind: 'command',
      level: 'info',
      actor: 'dashboard',
      data: { mode },
    });
    this.dispatch({ type: 'SET_FLOW_MODE', mode });
  }

  async refreshProviders(): Promise<void> {
    await this.bus.publish({
      topic: 'dashboard.vps.refresh_providers',
      kind: 'command',
      level: 'info',
      actor: 'dashboard',
      data: {},
    });
  }
}

// Default export
export default Dashboard;
