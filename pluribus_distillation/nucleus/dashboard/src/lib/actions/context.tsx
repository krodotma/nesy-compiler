/**
 * Action Context - Provides action dispatch and result management
 *
 * Creates a notebook-style execution environment where actions
 * produce streaming outputs displayed in cells.
 */

import {
  component$,
  createContextId,
  useContextProvider,
  useContext,
  useSignal,
  useStore,
  $,
  type QRL,
  Slot,
} from '@builder.io/qwik';
import type {
  ActionCell,
  ActionRequest,
  ActionResult,
  ActionOutput,
  ActionStatus,
} from './types';

// Generate unique IDs
function generateId(): string {
  return Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
}

// Context interface
export interface ActionContextValue {
  cells: ActionCell[];
  dispatch: QRL<(type: string, payload: Record<string, unknown>) => string>;
  updateResult: QRL<(requestId: string, update: Partial<ActionResult>) => void>;
  appendOutput: QRL<(requestId: string, output: ActionOutput) => void>;
  clearCells: QRL<() => void>;
  toggleCollapse: QRL<(cellId: string) => void>;
  wsConnected: boolean;
}

export const ActionContext = createContextId<ActionContextValue>('action-context');

// Provider component
export const ActionProvider = component$(() => {
  const cells = useStore<ActionCell[]>([]);
  const wsConnected = useSignal(false);
  const wsRef = useSignal<WebSocket | null>(null);

  // Dispatch an action
  const dispatch = $((type: string, payload: Record<string, unknown>): string => {
    const requestId = generateId();
    const request: ActionRequest = {
      id: requestId,
      type,
      payload,
      timestamp: Date.now(),
    };

    const cell: ActionCell = {
      id: generateId(),
      request,
      result: {
        id: generateId(),
        requestId,
        status: 'pending',
        outputs: [],
        events: [],
        startedAt: Date.now(),
      },
      collapsed: false,
      activeTab: 'outputs',
    };

    cells.push(cell);

    // Send via WebSocket if connected
    if (wsRef.value?.readyState === WebSocket.OPEN) {
      wsRef.value.send(JSON.stringify({
        type: 'action',
        request,
      }));
    }

    return requestId;
  });

  // Update result for a request
  const updateResult = $((requestId: string, update: Partial<ActionResult>) => {
    const cellIndex = cells.findIndex(c => c.request.id === requestId);
    if (cellIndex >= 0 && cells[cellIndex].result) {
      Object.assign(cells[cellIndex].result!, update);
    }
  });

  // Append output to a result
  const appendOutput = $((requestId: string, output: ActionOutput) => {
    const cellIndex = cells.findIndex(c => c.request.id === requestId);
    if (cellIndex >= 0 && cells[cellIndex].result) {
      cells[cellIndex].result!.outputs.push(output);
    }
  });

  // Clear all cells
  const clearCells = $(() => {
    cells.length = 0;
  });

  // Toggle cell collapse
  const toggleCollapse = $((cellId: string) => {
    const cell = cells.find(c => c.id === cellId);
    if (cell) {
      cell.collapsed = !cell.collapsed;
    }
  });

  const contextValue: ActionContextValue = {
    cells,
    dispatch,
    updateResult,
    appendOutput,
    clearCells,
    toggleCollapse,
    wsConnected: wsConnected.value,
  };

  useContextProvider(ActionContext, contextValue);

  return <Slot />;
});

// Hook to use action context
export function useActionContext(): ActionContextValue {
  return useContext(ActionContext);
}
