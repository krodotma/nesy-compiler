/**
 * Dashboard State Reducer
 *
 * Pure function that handles all state transitions.
 * Isomorphic - runs identically in all environments.
 */

import type {
  DashboardState,
  DashboardAction,
  BusEvent,
  Notification,
} from './types';

export function dashboardReducer(
  state: DashboardState,
  action: DashboardAction
): DashboardState {
  switch (action.type) {
    case 'SET_MODE':
      return {
        ...state,
        mode: action.mode,
        ui: { ...state.ui, activePanel: action.mode },
      };

    case 'SET_SERVICES':
      return { ...state, services: action.services };

    case 'SET_INSTANCES':
      return { ...state, instances: action.instances };

    case 'SELECT_SERVICE':
      return { ...state, selectedService: action.id };

    case 'ADD_EVENT': {
      const events = [action.event, ...state.events].slice(0, state.maxEvents);
      return { ...state, events };
    }

    case 'SET_EVENTS':
      return { ...state, events: action.events.slice(0, state.maxEvents) };

    case 'SET_EVENT_FILTER':
      return { ...state, eventFilter: action.filter };

    case 'SET_REQUESTS':
      return { ...state, requests: action.requests };

    case 'SET_AGENTS':
      return { ...state, agents: action.agents };

    case 'SET_WORKERS':
      return { ...state, workers: action.workers };

    case 'SELECT_AGENT':
      return { ...state, selectedAgent: action.actor };

    case 'SET_SESSION':
      return { ...state, session: action.session };

    case 'UPDATE_PROVIDER':
      return {
        ...state,
        session: {
          ...state.session,
          providers: {
            ...state.session.providers,
            [action.provider]: action.status,
          },
        },
      };

    case 'SET_FLOW_MODE':
      return {
        ...state,
        session: { ...state.session, flowMode: action.mode },
      };

    case 'SET_WORKFLOW':
      return { ...state, workflow: action.workflow };

    case 'SET_THEME':
      return {
        ...state,
        ui: { ...state.ui, theme: action.theme },
      };

    case 'TOGGLE_SIDEBAR':
      return {
        ...state,
        ui: { ...state.ui, sidebarOpen: !state.ui.sidebarOpen },
      };

    case 'ADD_NOTIFICATION': {
      const notifications = [action.notification, ...state.ui.notifications];
      return {
        ...state,
        ui: { ...state.ui, notifications },
      };
    }

    case 'DISMISS_NOTIFICATION': {
      const notifications = state.ui.notifications.filter(
        (n) => n.id !== action.id
      );
      return {
        ...state,
        ui: { ...state.ui, notifications },
      };
    }

    case 'SET_CONNECTED':
      return { ...state, connected: action.connected };

    case 'SYNC_STATE':
      return {
        ...state,
        ...action.state,
        lastSync: new Date().toISOString(),
      };

    default:
      return state;
  }
}

/**
 * Parse a bus event and convert to dashboard action(s)
 */
export function busEventToActions(event: BusEvent): DashboardAction[] {
  const actions: DashboardAction[] = [];

  // Always add the event to the log
  actions.push({ type: 'ADD_EVENT', event });

  // Handle specific topics
  const { topic, data } = event;

  // Service events - populate services from bus events
  if (topic === 'service.register' && typeof data === 'object') {
    const d = data as Record<string, unknown>;
    // Service registration - update services list via sync
    actions.push({
      type: 'SYNC_STATE',
      state: {
        services: [{
          id: String(d['id'] ?? ''),
          name: String(d['name'] ?? ''),
          kind: (d['kind'] as 'port' | 'composition' | 'process') ?? 'process',
          entry_point: String(d['entry_point'] ?? ''),
          description: String(d['description'] ?? ''),
          port: typeof d['port'] === 'number' ? d['port'] : undefined,
          depends_on: Array.isArray(d['depends_on']) ? d['depends_on'] : [],
          env: (typeof d['env'] === 'object' && d['env'] !== null ? d['env'] : {}) as Record<string, string>,
          args: Array.isArray(d['args']) ? d['args'] : [],
          tags: Array.isArray(d['tags']) ? d['tags'] : [],
          auto_start: Boolean(d['auto_start']),
          restart_policy: (d['restart_policy'] as 'never' | 'on_failure' | 'always') ?? 'never',
          health_check: typeof d['health_check'] === 'string' ? d['health_check'] : undefined,
          created_iso: String(d['created_iso'] ?? event.iso),
          provenance: (typeof d['provenance'] === 'object' ? d['provenance'] : {}) as Record<string, unknown>,
        }],
      },
    });
  }

  if (topic === 'service.control/start' || topic === 'service.control/stop' || topic === 'service.instance.status') {
    // Service instance status update
    const d = data as Record<string, unknown>;
    if (d['service_id']) {
      // This would require special handling to merge instances
      // For now, just log it
    }
  }

  if (topic === 'service.registry.sync' && typeof data === 'object') {
    // Full registry sync from service_registry.py
    const d = data as Record<string, unknown>;
    if (Array.isArray(d['services'])) {
      actions.push({
        type: 'SET_SERVICES',
        services: d['services'] as any[],
      });
    }
    if (Array.isArray(d['instances'])) {
      actions.push({
        type: 'SET_INSTANCES',
        instances: d['instances'] as any[],
      });
    }
  }

  if (topic === 'dashboard.state.sync' && typeof data === 'object') {
    actions.push({ type: 'SYNC_STATE', state: data as Partial<DashboardState> });
  }

  if (topic === 'pluribus.check.report' && typeof data === 'object') {
    // Agent status update - handled by polling, but could update here
  }

  if (topic.startsWith('providers.') && typeof data === 'object') {
    const d = data as Record<string, unknown>;
    if (topic === 'providers.smoke.result') {
      const provider = d['provider'] as string;
      const success = d['success'] as boolean;
      actions.push({
        type: 'UPDATE_PROVIDER',
        provider,
        status: {
          available: success,
          lastCheck: event.iso,
          error: success ? undefined : (d['error'] as string),
        },
      });
    }
  }

  if (topic === 'dashboard.vps.provider_status' && typeof data === 'object') {
    const d = data as Record<string, unknown>;
    const provider = String(d['provider'] ?? '');
    if (provider) {
      actions.push({
        type: 'UPDATE_PROVIDER',
        provider,
        status: {
          available: Boolean(d['available']),
          lastCheck: event.iso,
          error: typeof d['error'] === 'string' ? d['error'] : undefined,
          model: typeof d['model'] === 'string' ? d['model'] : undefined,
        },
      });
    }
  }

  if (topic === 'dashboard.vps.flow_mode_changed' && typeof data === 'object') {
    const d = data as Record<string, unknown>;
    const mode = d['mode'] === 'A' ? 'A' : 'm';
    actions.push({ type: 'SET_FLOW_MODE', mode });
  }

  if (topic === 'dashboard.vps.fallback_activated' && typeof data === 'object') {
    const d = data as Record<string, unknown>;
    actions.push({
      type: 'SET_SESSION',
      session: {
        ...createDefaultSession(),
        activeFallback: d['fallback'] as string,
      },
    });
  }

  return actions;
}

function createDefaultSession() {
  return {
    flowMode: 'm' as const,
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
    pbpair: {
      activeRequests: [],
      pendingProposals: [],
    },
    auth: {
      claudeLoggedIn: false,
      geminiCliLoggedIn: false,
    },
  };
}

/**
 * Create a notification from an event
 */
export function createNotification(
  event: BusEvent
): Notification | null {
  // Only create notifications for important events
  if (event.level !== 'error' && event.level !== 'warn') {
    return null;
  }

  return {
    id: `${event.ts}-${Math.random().toString(36).slice(2)}`,
    type: event.level === 'error' ? 'error' : 'warning',
    message: `[${event.topic}] ${JSON.stringify(event.data)}`,
    timestamp: event.ts,
    read: false,
  };
}
