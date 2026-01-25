/**
 * Unified Lanes Store - Single Source of Truth
 *
 * Phase 7, Iteration 51 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Qwik-compatible reactive store
 * - Single source of truth for all lanes state
 * - Derived computations (stats, filters, sorting)
 * - Action dispatching with type safety
 * - Subscription-based updates
 * - Optimistic update support
 */

// ============================================================================
// Types
// ============================================================================

export interface LaneHistory {
  ts: string;
  wip_pct: number;
  note: string;
}

export interface Lane {
  id: string;
  name: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  owner: string;
  description: string;
  commits: string[];
  blockers: string[];
  next_actions: string[];
  history: LaneHistory[];
  tier?: string;
  host?: string;
  slot?: number;
  tags?: string[];
  priority?: number;
  targetDate?: string;
  dependencies?: string[];
}

export interface Agent {
  id: string;
  status: 'active' | 'idle' | 'offline';
  lane: string | null;
  last_seen: string;
}

export interface LanesState {
  version: string;
  generated: string;
  updated: string;
  lanes: Lane[];
  agents: Agent[];
}

export interface LanesFilter {
  status?: ('green' | 'yellow' | 'red')[];
  owner?: string;
  search?: string;
  tags?: string[];
  minWip?: number;
  maxWip?: number;
  hasBlockers?: boolean;
}

export type LanesSortKey = 'name' | 'wip_pct' | 'status' | 'owner' | 'updated' | 'priority';
export type LanesSortDirection = 'asc' | 'desc';

export interface LanesSort {
  key: LanesSortKey;
  direction: LanesSortDirection;
}

export interface LanesUIState {
  selectedLaneId: string | null;
  expandedLaneIds: Set<string>;
  collapsedSections: Record<string, boolean>;
  filter: LanesFilter;
  sort: LanesSort;
  viewMode: 'list' | 'grid' | 'timeline' | 'kanban';
  showDetails: boolean;
}

export interface LanesStoreState {
  // Core data
  data: LanesState | null;

  // Loading state
  loading: boolean;
  error: string | null;
  lastSync: string | null;

  // UI state
  ui: LanesUIState;

  // Sync state
  syncStatus: 'synced' | 'syncing' | 'pending' | 'error' | 'offline';
  pendingChanges: LaneAction[];

  // Undo/redo
  undoStack: LaneAction[];
  redoStack: LaneAction[];
}

// ============================================================================
// Actions
// ============================================================================

export type LaneAction =
  | { type: 'SET_DATA'; payload: LanesState }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SET_SYNC_STATUS'; payload: LanesStoreState['syncStatus'] }
  | { type: 'UPDATE_LANE'; payload: { id: string; changes: Partial<Lane> } }
  | { type: 'ADD_LANE'; payload: Lane }
  | { type: 'REMOVE_LANE'; payload: string }
  | { type: 'SELECT_LANE'; payload: string | null }
  | { type: 'TOGGLE_LANE_EXPAND'; payload: string }
  | { type: 'TOGGLE_SECTION'; payload: string }
  | { type: 'SET_FILTER'; payload: Partial<LanesFilter> }
  | { type: 'CLEAR_FILTER' }
  | { type: 'SET_SORT'; payload: LanesSort }
  | { type: 'SET_VIEW_MODE'; payload: LanesUIState['viewMode'] }
  | { type: 'UPDATE_AGENT'; payload: { id: string; changes: Partial<Agent> } }
  | { type: 'PUSH_UNDO'; payload: LaneAction }
  | { type: 'POP_UNDO' }
  | { type: 'POP_REDO' }
  | { type: 'CLEAR_HISTORY' }
  | { type: 'ADD_PENDING_CHANGE'; payload: LaneAction }
  | { type: 'CLEAR_PENDING_CHANGES' }
  | { type: 'BATCH_UPDATE'; payload: LaneAction[] };

// ============================================================================
// Default State
// ============================================================================

export function createDefaultUIState(): LanesUIState {
  return {
    selectedLaneId: null,
    expandedLaneIds: new Set(),
    collapsedSections: {},
    filter: {},
    sort: { key: 'wip_pct', direction: 'asc' },
    viewMode: 'list',
    showDetails: true,
  };
}

export function createDefaultStoreState(): LanesStoreState {
  return {
    data: null,
    loading: true,
    error: null,
    lastSync: null,
    ui: createDefaultUIState(),
    syncStatus: 'pending',
    pendingChanges: [],
    undoStack: [],
    redoStack: [],
  };
}

// ============================================================================
// Reducer
// ============================================================================

export function lanesReducer(state: LanesStoreState, action: LaneAction): LanesStoreState {
  switch (action.type) {
    case 'SET_DATA':
      return {
        ...state,
        data: action.payload,
        loading: false,
        error: null,
        lastSync: new Date().toISOString(),
        syncStatus: 'synced',
      };

    case 'SET_LOADING':
      return { ...state, loading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false, syncStatus: 'error' };

    case 'SET_SYNC_STATUS':
      return { ...state, syncStatus: action.payload };

    case 'UPDATE_LANE': {
      if (!state.data) return state;
      const lanes = state.data.lanes.map(lane =>
        lane.id === action.payload.id
          ? { ...lane, ...action.payload.changes }
          : lane
      );
      return {
        ...state,
        data: {
          ...state.data,
          lanes,
          updated: new Date().toISOString(),
        },
      };
    }

    case 'ADD_LANE': {
      if (!state.data) return state;
      return {
        ...state,
        data: {
          ...state.data,
          lanes: [...state.data.lanes, action.payload],
          updated: new Date().toISOString(),
        },
      };
    }

    case 'REMOVE_LANE': {
      if (!state.data) return state;
      return {
        ...state,
        data: {
          ...state.data,
          lanes: state.data.lanes.filter(l => l.id !== action.payload),
          updated: new Date().toISOString(),
        },
        ui: {
          ...state.ui,
          selectedLaneId: state.ui.selectedLaneId === action.payload ? null : state.ui.selectedLaneId,
        },
      };
    }

    case 'SELECT_LANE':
      return {
        ...state,
        ui: { ...state.ui, selectedLaneId: action.payload },
      };

    case 'TOGGLE_LANE_EXPAND': {
      const newExpanded = new Set(state.ui.expandedLaneIds);
      if (newExpanded.has(action.payload)) {
        newExpanded.delete(action.payload);
      } else {
        newExpanded.add(action.payload);
      }
      return {
        ...state,
        ui: { ...state.ui, expandedLaneIds: newExpanded },
      };
    }

    case 'TOGGLE_SECTION':
      return {
        ...state,
        ui: {
          ...state.ui,
          collapsedSections: {
            ...state.ui.collapsedSections,
            [action.payload]: !state.ui.collapsedSections[action.payload],
          },
        },
      };

    case 'SET_FILTER':
      return {
        ...state,
        ui: {
          ...state.ui,
          filter: { ...state.ui.filter, ...action.payload },
        },
      };

    case 'CLEAR_FILTER':
      return {
        ...state,
        ui: { ...state.ui, filter: {} },
      };

    case 'SET_SORT':
      return {
        ...state,
        ui: { ...state.ui, sort: action.payload },
      };

    case 'SET_VIEW_MODE':
      return {
        ...state,
        ui: { ...state.ui, viewMode: action.payload },
      };

    case 'UPDATE_AGENT': {
      if (!state.data) return state;
      const agents = state.data.agents.map(agent =>
        agent.id === action.payload.id
          ? { ...agent, ...action.payload.changes }
          : agent
      );
      return {
        ...state,
        data: { ...state.data, agents },
      };
    }

    case 'PUSH_UNDO':
      return {
        ...state,
        undoStack: [...state.undoStack.slice(-99), action.payload],
        redoStack: [], // Clear redo on new action
      };

    case 'POP_UNDO': {
      if (state.undoStack.length === 0) return state;
      const lastAction = state.undoStack[state.undoStack.length - 1];
      return {
        ...state,
        undoStack: state.undoStack.slice(0, -1),
        redoStack: [...state.redoStack, lastAction],
      };
    }

    case 'POP_REDO': {
      if (state.redoStack.length === 0) return state;
      const lastAction = state.redoStack[state.redoStack.length - 1];
      return {
        ...state,
        redoStack: state.redoStack.slice(0, -1),
        undoStack: [...state.undoStack, lastAction],
      };
    }

    case 'CLEAR_HISTORY':
      return {
        ...state,
        undoStack: [],
        redoStack: [],
      };

    case 'ADD_PENDING_CHANGE':
      return {
        ...state,
        pendingChanges: [...state.pendingChanges, action.payload],
        syncStatus: 'pending',
      };

    case 'CLEAR_PENDING_CHANGES':
      return {
        ...state,
        pendingChanges: [],
        syncStatus: 'synced',
      };

    case 'BATCH_UPDATE': {
      let newState = state;
      for (const subAction of action.payload) {
        newState = lanesReducer(newState, subAction);
      }
      return newState;
    }

    default:
      return state;
  }
}

// ============================================================================
// Selectors (Derived State)
// ============================================================================

export interface LanesStats {
  totalLanes: number;
  greenCount: number;
  yellowCount: number;
  redCount: number;
  avgWip: number;
  activeAgents: number;
  totalBlockers: number;
  completedLanes: number;
}

export function selectStats(state: LanesStoreState): LanesStats {
  if (!state.data) {
    return {
      totalLanes: 0,
      greenCount: 0,
      yellowCount: 0,
      redCount: 0,
      avgWip: 0,
      activeAgents: 0,
      totalBlockers: 0,
      completedLanes: 0,
    };
  }

  const lanes = state.data.lanes;
  const agents = state.data.agents;

  return {
    totalLanes: lanes.length,
    greenCount: lanes.filter(l => l.status === 'green').length,
    yellowCount: lanes.filter(l => l.status === 'yellow').length,
    redCount: lanes.filter(l => l.status === 'red').length,
    avgWip: lanes.length > 0
      ? Math.round(lanes.reduce((sum, l) => sum + l.wip_pct, 0) / lanes.length)
      : 0,
    activeAgents: agents.filter(a => a.status === 'active').length,
    totalBlockers: lanes.reduce((sum, l) => sum + l.blockers.length, 0),
    completedLanes: lanes.filter(l => l.wip_pct >= 100).length,
  };
}

export function selectFilteredLanes(state: LanesStoreState): Lane[] {
  if (!state.data) return [];

  let lanes = [...state.data.lanes];
  const filter = state.ui.filter;

  // Apply filters
  if (filter.status && filter.status.length > 0) {
    lanes = lanes.filter(l => filter.status!.includes(l.status));
  }

  if (filter.owner) {
    lanes = lanes.filter(l => l.owner.toLowerCase().includes(filter.owner!.toLowerCase()));
  }

  if (filter.search) {
    const search = filter.search.toLowerCase();
    lanes = lanes.filter(l =>
      l.name.toLowerCase().includes(search) ||
      l.description.toLowerCase().includes(search) ||
      l.id.toLowerCase().includes(search)
    );
  }

  if (filter.tags && filter.tags.length > 0) {
    lanes = lanes.filter(l =>
      l.tags && filter.tags!.some(tag => l.tags!.includes(tag))
    );
  }

  if (typeof filter.minWip === 'number') {
    lanes = lanes.filter(l => l.wip_pct >= filter.minWip!);
  }

  if (typeof filter.maxWip === 'number') {
    lanes = lanes.filter(l => l.wip_pct <= filter.maxWip!);
  }

  if (filter.hasBlockers !== undefined) {
    lanes = lanes.filter(l =>
      filter.hasBlockers ? l.blockers.length > 0 : l.blockers.length === 0
    );
  }

  return lanes;
}

export function selectSortedLanes(state: LanesStoreState): Lane[] {
  const lanes = selectFilteredLanes(state);
  const { key, direction } = state.ui.sort;

  return lanes.sort((a, b) => {
    let cmp = 0;

    switch (key) {
      case 'name':
        cmp = a.name.localeCompare(b.name);
        break;
      case 'wip_pct':
        cmp = a.wip_pct - b.wip_pct;
        break;
      case 'status': {
        const statusOrder = { red: 0, yellow: 1, green: 2 };
        cmp = statusOrder[a.status] - statusOrder[b.status];
        break;
      }
      case 'owner':
        cmp = a.owner.localeCompare(b.owner);
        break;
      case 'priority':
        cmp = (a.priority || 0) - (b.priority || 0);
        break;
      default:
        cmp = 0;
    }

    return direction === 'asc' ? cmp : -cmp;
  });
}

export function selectGroupedLanes(state: LanesStoreState): Record<string, Lane[]> {
  const sorted = selectSortedLanes(state);
  return {
    red: sorted.filter(l => l.status === 'red'),
    yellow: sorted.filter(l => l.status === 'yellow'),
    green: sorted.filter(l => l.status === 'green'),
  };
}

export function selectLaneById(state: LanesStoreState, id: string): Lane | undefined {
  return state.data?.lanes.find(l => l.id === id);
}

export function selectSelectedLane(state: LanesStoreState): Lane | undefined {
  if (!state.ui.selectedLaneId) return undefined;
  return selectLaneById(state, state.ui.selectedLaneId);
}

export function selectAgentsByLane(state: LanesStoreState, laneId: string): Agent[] {
  if (!state.data) return [];
  return state.data.agents.filter(a => a.lane === laneId);
}

export function selectActiveAgents(state: LanesStoreState): Agent[] {
  if (!state.data) return [];
  return state.data.agents.filter(a => a.status === 'active');
}

export function selectLanesWithBlockers(state: LanesStoreState): Lane[] {
  return selectFilteredLanes(state).filter(l => l.blockers.length > 0);
}

export function selectCanUndo(state: LanesStoreState): boolean {
  return state.undoStack.length > 0;
}

export function selectCanRedo(state: LanesStoreState): boolean {
  return state.redoStack.length > 0;
}

// ============================================================================
// Store Class
// ============================================================================

export type Subscriber = (state: LanesStoreState) => void;

export class LanesStore {
  private state: LanesStoreState;
  private subscribers: Set<Subscriber> = new Set();

  constructor(initialState?: Partial<LanesStoreState>) {
    this.state = { ...createDefaultStoreState(), ...initialState };
  }

  getState(): LanesStoreState {
    return this.state;
  }

  dispatch(action: LaneAction): void {
    const prevState = this.state;
    this.state = lanesReducer(this.state, action);

    // Only notify if state actually changed
    if (this.state !== prevState) {
      this.notifySubscribers();
    }
  }

  subscribe(fn: Subscriber): () => void {
    this.subscribers.add(fn);
    return () => {
      this.subscribers.delete(fn);
    };
  }

  private notifySubscribers(): void {
    this.subscribers.forEach(fn => fn(this.state));
  }

  // Convenience methods
  setData(data: LanesState): void {
    this.dispatch({ type: 'SET_DATA', payload: data });
  }

  updateLane(id: string, changes: Partial<Lane>): void {
    this.dispatch({ type: 'UPDATE_LANE', payload: { id, changes } });
  }

  selectLane(id: string | null): void {
    this.dispatch({ type: 'SELECT_LANE', payload: id });
  }

  setFilter(filter: Partial<LanesFilter>): void {
    this.dispatch({ type: 'SET_FILTER', payload: filter });
  }

  setSort(sort: LanesSort): void {
    this.dispatch({ type: 'SET_SORT', payload: sort });
  }

  // Selectors as methods
  getStats(): LanesStats {
    return selectStats(this.state);
  }

  getFilteredLanes(): Lane[] {
    return selectFilteredLanes(this.state);
  }

  getSortedLanes(): Lane[] {
    return selectSortedLanes(this.state);
  }

  getGroupedLanes(): Record<string, Lane[]> {
    return selectGroupedLanes(this.state);
  }
}

// ============================================================================
// Singleton Instance
// ============================================================================

let globalStore: LanesStore | null = null;

export function getGlobalStore(): LanesStore {
  if (!globalStore) {
    globalStore = new LanesStore();
  }
  return globalStore;
}

export function resetGlobalStore(): void {
  globalStore = null;
}

export default LanesStore;
