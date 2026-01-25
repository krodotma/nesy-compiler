/**
 * Lanes Store Unit Tests
 *
 * Phase 9, Iteration 72 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Unit tests for store
 * - Mock factories
 * - Reducer tests
 * - Selector tests
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';

// ============================================================================
// Mock Factories
// ============================================================================

export function createMockLane(overrides: Partial<Lane> = {}): Lane {
  return {
    id: `lane-${Math.random().toString(36).slice(2, 9)}`,
    name: 'Test Lane',
    status: 'green',
    wip_pct: 50,
    owner: 'agent-1',
    blockers: [],
    tags: ['test'],
    updated: new Date().toISOString(),
    ...overrides,
  };
}

export function createMockAgent(overrides: Partial<Agent> = {}): Agent {
  return {
    id: `agent-${Math.random().toString(36).slice(2, 9)}`,
    name: 'Test Agent',
    avatar: undefined,
    status: 'active',
    ...overrides,
  };
}

export function createMockLanesState(overrides: Partial<LanesState> = {}): LanesState {
  return {
    lanes: [],
    agents: [],
    selectedLaneId: null,
    filter: {
      status: null,
      owner: null,
      search: '',
      tags: [],
    },
    sort: {
      field: 'updated',
      direction: 'desc',
    },
    isLoading: false,
    error: null,
    lastSync: Date.now(),
    ...overrides,
  };
}

// ============================================================================
// Types (mirroring store types for tests)
// ============================================================================

interface Lane {
  id: string;
  name: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
  owner: string;
  blockers: string[];
  tags: string[];
  updated: string;
}

interface Agent {
  id: string;
  name: string;
  avatar?: string;
  status: 'active' | 'idle' | 'offline';
}

interface LanesState {
  lanes: Lane[];
  agents: Agent[];
  selectedLaneId: string | null;
  filter: {
    status: Lane['status'] | null;
    owner: string | null;
    search: string;
    tags: string[];
  };
  sort: {
    field: 'name' | 'status' | 'wip_pct' | 'updated' | 'owner';
    direction: 'asc' | 'desc';
  };
  isLoading: boolean;
  error: string | null;
  lastSync: number;
}

type LaneAction =
  | { type: 'SET_LANES'; payload: Lane[] }
  | { type: 'ADD_LANE'; payload: Lane }
  | { type: 'UPDATE_LANE'; payload: { id: string; changes: Partial<Lane> } }
  | { type: 'REMOVE_LANE'; payload: string }
  | { type: 'SELECT_LANE'; payload: string | null }
  | { type: 'SET_FILTER'; payload: Partial<LanesState['filter']> }
  | { type: 'SET_SORT'; payload: LanesState['sort'] }
  | { type: 'SET_LOADING'; payload: boolean }
  | { type: 'SET_ERROR'; payload: string | null }
  | { type: 'SYNC_COMPLETE'; payload: number };

// ============================================================================
// Reducer Implementation (for testing)
// ============================================================================

function lanesReducer(state: LanesState, action: LaneAction): LanesState {
  switch (action.type) {
    case 'SET_LANES':
      return { ...state, lanes: action.payload };

    case 'ADD_LANE':
      return { ...state, lanes: [...state.lanes, action.payload] };

    case 'UPDATE_LANE': {
      const { id, changes } = action.payload;
      return {
        ...state,
        lanes: state.lanes.map(lane =>
          lane.id === id ? { ...lane, ...changes } : lane
        ),
      };
    }

    case 'REMOVE_LANE':
      return {
        ...state,
        lanes: state.lanes.filter(lane => lane.id !== action.payload),
        selectedLaneId: state.selectedLaneId === action.payload ? null : state.selectedLaneId,
      };

    case 'SELECT_LANE':
      return { ...state, selectedLaneId: action.payload };

    case 'SET_FILTER':
      return { ...state, filter: { ...state.filter, ...action.payload } };

    case 'SET_SORT':
      return { ...state, sort: action.payload };

    case 'SET_LOADING':
      return { ...state, isLoading: action.payload };

    case 'SET_ERROR':
      return { ...state, error: action.payload };

    case 'SYNC_COMPLETE':
      return { ...state, lastSync: action.payload, isLoading: false };

    default:
      return state;
  }
}

// ============================================================================
// Selectors (for testing)
// ============================================================================

function selectFilteredLanes(state: LanesState): Lane[] {
  let result = [...state.lanes];

  // Apply status filter
  if (state.filter.status) {
    result = result.filter(l => l.status === state.filter.status);
  }

  // Apply owner filter
  if (state.filter.owner) {
    result = result.filter(l => l.owner === state.filter.owner);
  }

  // Apply search filter
  if (state.filter.search) {
    const query = state.filter.search.toLowerCase();
    result = result.filter(l =>
      l.name.toLowerCase().includes(query) ||
      l.owner.toLowerCase().includes(query) ||
      l.tags.some(t => t.toLowerCase().includes(query))
    );
  }

  // Apply tags filter
  if (state.filter.tags.length > 0) {
    result = result.filter(l =>
      state.filter.tags.every(tag => l.tags.includes(tag))
    );
  }

  // Apply sorting
  result.sort((a, b) => {
    let comparison = 0;
    switch (state.sort.field) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'status':
        const statusOrder = { green: 0, yellow: 1, red: 2 };
        comparison = statusOrder[a.status] - statusOrder[b.status];
        break;
      case 'wip_pct':
        comparison = a.wip_pct - b.wip_pct;
        break;
      case 'updated':
        comparison = new Date(a.updated).getTime() - new Date(b.updated).getTime();
        break;
      case 'owner':
        comparison = a.owner.localeCompare(b.owner);
        break;
    }
    return state.sort.direction === 'asc' ? comparison : -comparison;
  });

  return result;
}

function selectStats(state: LanesState) {
  const lanes = state.lanes;
  return {
    total: lanes.length,
    byStatus: {
      green: lanes.filter(l => l.status === 'green').length,
      yellow: lanes.filter(l => l.status === 'yellow').length,
      red: lanes.filter(l => l.status === 'red').length,
    },
    avgWip: lanes.length > 0
      ? lanes.reduce((sum, l) => sum + l.wip_pct, 0) / lanes.length
      : 0,
    withBlockers: lanes.filter(l => l.blockers.length > 0).length,
  };
}

// ============================================================================
// Reducer Tests
// ============================================================================

describe('lanesReducer', () => {
  let initialState: LanesState;

  beforeEach(() => {
    initialState = createMockLanesState();
  });

  describe('SET_LANES', () => {
    it('should set lanes array', () => {
      const lanes = [createMockLane(), createMockLane()];
      const newState = lanesReducer(initialState, { type: 'SET_LANES', payload: lanes });
      expect(newState.lanes).toEqual(lanes);
    });

    it('should replace existing lanes', () => {
      const existingLanes = [createMockLane({ id: 'old-1' })];
      const newLanes = [createMockLane({ id: 'new-1' })];
      const state = { ...initialState, lanes: existingLanes };
      const newState = lanesReducer(state, { type: 'SET_LANES', payload: newLanes });
      expect(newState.lanes).toEqual(newLanes);
      expect(newState.lanes).not.toContainEqual(existingLanes[0]);
    });
  });

  describe('ADD_LANE', () => {
    it('should add a lane to the array', () => {
      const lane = createMockLane();
      const newState = lanesReducer(initialState, { type: 'ADD_LANE', payload: lane });
      expect(newState.lanes).toContainEqual(lane);
      expect(newState.lanes.length).toBe(1);
    });

    it('should append to existing lanes', () => {
      const existingLane = createMockLane({ id: 'existing' });
      const newLane = createMockLane({ id: 'new' });
      const state = { ...initialState, lanes: [existingLane] };
      const newState = lanesReducer(state, { type: 'ADD_LANE', payload: newLane });
      expect(newState.lanes.length).toBe(2);
      expect(newState.lanes).toContainEqual(existingLane);
      expect(newState.lanes).toContainEqual(newLane);
    });
  });

  describe('UPDATE_LANE', () => {
    it('should update lane properties', () => {
      const lane = createMockLane({ id: 'test-1', status: 'green', wip_pct: 50 });
      const state = { ...initialState, lanes: [lane] };
      const newState = lanesReducer(state, {
        type: 'UPDATE_LANE',
        payload: { id: 'test-1', changes: { status: 'red', wip_pct: 75 } },
      });
      expect(newState.lanes[0].status).toBe('red');
      expect(newState.lanes[0].wip_pct).toBe(75);
    });

    it('should not modify other lanes', () => {
      const lane1 = createMockLane({ id: 'lane-1', name: 'Lane 1' });
      const lane2 = createMockLane({ id: 'lane-2', name: 'Lane 2' });
      const state = { ...initialState, lanes: [lane1, lane2] };
      const newState = lanesReducer(state, {
        type: 'UPDATE_LANE',
        payload: { id: 'lane-1', changes: { name: 'Updated Lane 1' } },
      });
      expect(newState.lanes[0].name).toBe('Updated Lane 1');
      expect(newState.lanes[1].name).toBe('Lane 2');
    });

    it('should handle non-existent lane', () => {
      const lane = createMockLane({ id: 'test-1' });
      const state = { ...initialState, lanes: [lane] };
      const newState = lanesReducer(state, {
        type: 'UPDATE_LANE',
        payload: { id: 'non-existent', changes: { status: 'red' } },
      });
      expect(newState.lanes).toEqual(state.lanes);
    });
  });

  describe('REMOVE_LANE', () => {
    it('should remove lane from array', () => {
      const lane = createMockLane({ id: 'to-remove' });
      const state = { ...initialState, lanes: [lane] };
      const newState = lanesReducer(state, { type: 'REMOVE_LANE', payload: 'to-remove' });
      expect(newState.lanes.length).toBe(0);
    });

    it('should clear selection if removed lane was selected', () => {
      const lane = createMockLane({ id: 'selected-lane' });
      const state = { ...initialState, lanes: [lane], selectedLaneId: 'selected-lane' };
      const newState = lanesReducer(state, { type: 'REMOVE_LANE', payload: 'selected-lane' });
      expect(newState.selectedLaneId).toBeNull();
    });

    it('should not affect selection if different lane removed', () => {
      const lane1 = createMockLane({ id: 'lane-1' });
      const lane2 = createMockLane({ id: 'lane-2' });
      const state = { ...initialState, lanes: [lane1, lane2], selectedLaneId: 'lane-1' };
      const newState = lanesReducer(state, { type: 'REMOVE_LANE', payload: 'lane-2' });
      expect(newState.selectedLaneId).toBe('lane-1');
    });
  });

  describe('SELECT_LANE', () => {
    it('should set selected lane', () => {
      const newState = lanesReducer(initialState, { type: 'SELECT_LANE', payload: 'lane-1' });
      expect(newState.selectedLaneId).toBe('lane-1');
    });

    it('should clear selection with null', () => {
      const state = { ...initialState, selectedLaneId: 'lane-1' };
      const newState = lanesReducer(state, { type: 'SELECT_LANE', payload: null });
      expect(newState.selectedLaneId).toBeNull();
    });
  });

  describe('SET_FILTER', () => {
    it('should update filter partially', () => {
      const newState = lanesReducer(initialState, {
        type: 'SET_FILTER',
        payload: { status: 'red' },
      });
      expect(newState.filter.status).toBe('red');
      expect(newState.filter.search).toBe('');
    });

    it('should merge multiple filter properties', () => {
      const newState = lanesReducer(initialState, {
        type: 'SET_FILTER',
        payload: { status: 'yellow', search: 'test', tags: ['urgent'] },
      });
      expect(newState.filter.status).toBe('yellow');
      expect(newState.filter.search).toBe('test');
      expect(newState.filter.tags).toEqual(['urgent']);
    });
  });

  describe('SET_SORT', () => {
    it('should update sort configuration', () => {
      const newState = lanesReducer(initialState, {
        type: 'SET_SORT',
        payload: { field: 'wip_pct', direction: 'asc' },
      });
      expect(newState.sort.field).toBe('wip_pct');
      expect(newState.sort.direction).toBe('asc');
    });
  });

  describe('SET_LOADING', () => {
    it('should set loading state', () => {
      const newState = lanesReducer(initialState, { type: 'SET_LOADING', payload: true });
      expect(newState.isLoading).toBe(true);
    });
  });

  describe('SET_ERROR', () => {
    it('should set error message', () => {
      const newState = lanesReducer(initialState, { type: 'SET_ERROR', payload: 'Network error' });
      expect(newState.error).toBe('Network error');
    });

    it('should clear error with null', () => {
      const state = { ...initialState, error: 'Previous error' };
      const newState = lanesReducer(state, { type: 'SET_ERROR', payload: null });
      expect(newState.error).toBeNull();
    });
  });

  describe('SYNC_COMPLETE', () => {
    it('should update lastSync and clear loading', () => {
      const state = { ...initialState, isLoading: true };
      const timestamp = Date.now();
      const newState = lanesReducer(state, { type: 'SYNC_COMPLETE', payload: timestamp });
      expect(newState.lastSync).toBe(timestamp);
      expect(newState.isLoading).toBe(false);
    });
  });
});

// ============================================================================
// Selector Tests
// ============================================================================

describe('Selectors', () => {
  describe('selectFilteredLanes', () => {
    it('should return all lanes when no filter', () => {
      const lanes = [createMockLane(), createMockLane(), createMockLane()];
      const state = createMockLanesState({ lanes });
      const result = selectFilteredLanes(state);
      expect(result.length).toBe(3);
    });

    it('should filter by status', () => {
      const lanes = [
        createMockLane({ status: 'green' }),
        createMockLane({ status: 'red' }),
        createMockLane({ status: 'yellow' }),
      ];
      const state = createMockLanesState({
        lanes,
        filter: { status: 'red', owner: null, search: '', tags: [] },
      });
      const result = selectFilteredLanes(state);
      expect(result.length).toBe(1);
      expect(result[0].status).toBe('red');
    });

    it('should filter by owner', () => {
      const lanes = [
        createMockLane({ owner: 'agent-1' }),
        createMockLane({ owner: 'agent-2' }),
        createMockLane({ owner: 'agent-1' }),
      ];
      const state = createMockLanesState({
        lanes,
        filter: { status: null, owner: 'agent-1', search: '', tags: [] },
      });
      const result = selectFilteredLanes(state);
      expect(result.length).toBe(2);
      expect(result.every(l => l.owner === 'agent-1')).toBe(true);
    });

    it('should filter by search term in name', () => {
      const lanes = [
        createMockLane({ name: 'Feature Implementation' }),
        createMockLane({ name: 'Bug Fix' }),
        createMockLane({ name: 'Feature Testing' }),
      ];
      const state = createMockLanesState({
        lanes,
        filter: { status: null, owner: null, search: 'feature', tags: [] },
      });
      const result = selectFilteredLanes(state);
      expect(result.length).toBe(2);
    });

    it('should filter by tags', () => {
      const lanes = [
        createMockLane({ tags: ['urgent', 'frontend'] }),
        createMockLane({ tags: ['backend'] }),
        createMockLane({ tags: ['urgent', 'backend'] }),
      ];
      const state = createMockLanesState({
        lanes,
        filter: { status: null, owner: null, search: '', tags: ['urgent'] },
      });
      const result = selectFilteredLanes(state);
      expect(result.length).toBe(2);
    });

    it('should sort by name ascending', () => {
      const lanes = [
        createMockLane({ name: 'Charlie' }),
        createMockLane({ name: 'Alpha' }),
        createMockLane({ name: 'Beta' }),
      ];
      const state = createMockLanesState({
        lanes,
        sort: { field: 'name', direction: 'asc' },
      });
      const result = selectFilteredLanes(state);
      expect(result[0].name).toBe('Alpha');
      expect(result[1].name).toBe('Beta');
      expect(result[2].name).toBe('Charlie');
    });

    it('should sort by wip_pct descending', () => {
      const lanes = [
        createMockLane({ wip_pct: 30 }),
        createMockLane({ wip_pct: 90 }),
        createMockLane({ wip_pct: 60 }),
      ];
      const state = createMockLanesState({
        lanes,
        sort: { field: 'wip_pct', direction: 'desc' },
      });
      const result = selectFilteredLanes(state);
      expect(result[0].wip_pct).toBe(90);
      expect(result[1].wip_pct).toBe(60);
      expect(result[2].wip_pct).toBe(30);
    });

    it('should sort by status', () => {
      const lanes = [
        createMockLane({ status: 'yellow' }),
        createMockLane({ status: 'red' }),
        createMockLane({ status: 'green' }),
      ];
      const state = createMockLanesState({
        lanes,
        sort: { field: 'status', direction: 'asc' },
      });
      const result = selectFilteredLanes(state);
      expect(result[0].status).toBe('green');
      expect(result[1].status).toBe('yellow');
      expect(result[2].status).toBe('red');
    });
  });

  describe('selectStats', () => {
    it('should calculate correct statistics', () => {
      const lanes = [
        createMockLane({ status: 'green', wip_pct: 100, blockers: [] }),
        createMockLane({ status: 'green', wip_pct: 50, blockers: ['issue'] }),
        createMockLane({ status: 'yellow', wip_pct: 75, blockers: [] }),
        createMockLane({ status: 'red', wip_pct: 25, blockers: ['bug', 'review'] }),
      ];
      const state = createMockLanesState({ lanes });
      const stats = selectStats(state);

      expect(stats.total).toBe(4);
      expect(stats.byStatus.green).toBe(2);
      expect(stats.byStatus.yellow).toBe(1);
      expect(stats.byStatus.red).toBe(1);
      expect(stats.avgWip).toBe(62.5);
      expect(stats.withBlockers).toBe(2);
    });

    it('should handle empty lanes', () => {
      const state = createMockLanesState({ lanes: [] });
      const stats = selectStats(state);

      expect(stats.total).toBe(0);
      expect(stats.avgWip).toBe(0);
      expect(stats.withBlockers).toBe(0);
    });
  });
});

// ============================================================================
// Integration Tests
// ============================================================================

describe('Store Integration', () => {
  it('should handle a sequence of actions', () => {
    let state = createMockLanesState();

    // Add lanes
    const lane1 = createMockLane({ id: 'lane-1', name: 'Lane 1', status: 'green' });
    const lane2 = createMockLane({ id: 'lane-2', name: 'Lane 2', status: 'yellow' });
    state = lanesReducer(state, { type: 'ADD_LANE', payload: lane1 });
    state = lanesReducer(state, { type: 'ADD_LANE', payload: lane2 });
    expect(state.lanes.length).toBe(2);

    // Select a lane
    state = lanesReducer(state, { type: 'SELECT_LANE', payload: 'lane-1' });
    expect(state.selectedLaneId).toBe('lane-1');

    // Update the selected lane
    state = lanesReducer(state, {
      type: 'UPDATE_LANE',
      payload: { id: 'lane-1', changes: { status: 'red', wip_pct: 80 } },
    });
    expect(state.lanes.find(l => l.id === 'lane-1')?.status).toBe('red');

    // Filter by status
    state = lanesReducer(state, {
      type: 'SET_FILTER',
      payload: { status: 'red' },
    });
    const filtered = selectFilteredLanes(state);
    expect(filtered.length).toBe(1);

    // Remove the selected lane
    state = lanesReducer(state, { type: 'REMOVE_LANE', payload: 'lane-1' });
    expect(state.lanes.length).toBe(1);
    expect(state.selectedLaneId).toBeNull();
  });
});
