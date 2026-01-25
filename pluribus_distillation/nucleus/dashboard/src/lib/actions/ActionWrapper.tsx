/**
 * ActionWrapper - Uniform wrapper for all server action interactions
 *
 * Provides a consistent UI pattern for triggering server actions and
 * displaying their results in notebook-style output cells.
 */

import {
  component$,
  useStore,
  useSignal,
  $,
  type QRL,
  Slot,
} from '@builder.io/qwik';
import type { ActionCell, ActionRequest, ActionResult, ActionOutput } from './types';
import { OutputCells } from './OutputCell';

// Generate unique IDs
function generateId(): string {
  return Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
}

export interface ActionWrapperState {
  cells: ActionCell[];
  wsConnected: boolean;
  lastError: string | null;
}

export interface ActionWrapperAPI {
  dispatch: (type: string, payload: Record<string, unknown>) => string;
  updateResult: (requestId: string, update: Partial<ActionResult>) => void;
  appendOutput: (requestId: string, output: ActionOutput) => void;
  clearCells: () => void;
  toggleCollapse: (cellId: string) => void;
}

interface ActionWrapperProps {
  title?: string;
  showHeader?: boolean;
  maxCells?: number;
  wsUrl?: string;
  onAction?: QRL<(request: ActionRequest) => void>;
}

export const ActionWrapper = component$<ActionWrapperProps>(({
  title = 'Actions',
  showHeader = true,
  maxCells = 50,
  wsUrl,
  onAction,
}) => {
  const state = useStore<ActionWrapperState>({
    cells: [],
    wsConnected: false,
    lastError: null,
  });

  const wsRef = useSignal<WebSocket | null>(null);

  // Connect to WebSocket if URL provided
  const connectWs = $(() => {
    if (!wsUrl || wsRef.value) return;

    try {
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        state.wsConnected = true;
        state.lastError = null;
      };

      ws.onclose = () => {
        state.wsConnected = false;
        wsRef.value = null;
      };

      ws.onerror = () => {
        state.lastError = 'WebSocket connection error';
        state.wsConnected = false;
      };

      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          handleWsMessage(msg);
        } catch {
          console.error('Invalid WebSocket message');
        }
      };

      wsRef.value = ws;
    } catch (err) {
      state.lastError = `Failed to connect: ${err}`;
    }
  });

  // Handle incoming WebSocket messages
  const handleWsMessage = (msg: {
    type: string;
    requestId?: string;
    status?: ActionResult['status'];
    output?: ActionOutput;
    error?: string;
  }) => {
    if (msg.type === 'action.status' && msg.requestId) {
      updateResult(msg.requestId, { status: msg.status });
    } else if (msg.type === 'action.output' && msg.requestId && msg.output) {
      appendOutput(msg.requestId, msg.output);
    } else if (msg.type === 'action.complete' && msg.requestId) {
      updateResult(msg.requestId, {
        status: msg.error ? 'error' : 'success',
        completedAt: Date.now(),
        error: msg.error,
      });
    }
  };

  // Dispatch an action
  const dispatch = (type: string, payload: Record<string, unknown>): string => {
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

    // Add cell, trim if exceeds max
    state.cells.unshift(cell);
    if (state.cells.length > maxCells) {
      state.cells.pop();
    }

    // Send via WebSocket if connected
    if (wsRef.value?.readyState === WebSocket.OPEN) {
      wsRef.value.send(JSON.stringify({
        type: 'action',
        request,
      }));
    }

    // Call custom handler if provided
    onAction?.(request);

    return requestId;
  };

  // Update result for a request
  const updateResult = (requestId: string, update: Partial<ActionResult>) => {
    const cell = state.cells.find(c => c.request.id === requestId);
    if (cell?.result) {
      Object.assign(cell.result, update);
    }
  };

  // Append output to a result
  const appendOutput = (requestId: string, output: ActionOutput) => {
    const cell = state.cells.find(c => c.request.id === requestId);
    if (cell?.result) {
      cell.result.outputs.push(output);
    }
  };

  // Clear all cells
  const clearCells = $(() => {
    state.cells.length = 0;
  });

  // Toggle cell collapse
  const toggleCollapse = $((cellId: string) => {
    const cell = state.cells.find(c => c.id === cellId);
    if (cell) {
      cell.collapsed = !cell.collapsed;
    }
  });

  // Expose API via data attributes for external use
  const handleActionClick = $((event: Event) => {
    const target = event.target as HTMLElement;
    const actionBtn = target.closest('[data-action-type]') as HTMLElement;
    if (actionBtn) {
      const actionType = actionBtn.dataset.actionType;
      const payloadStr = actionBtn.dataset.actionPayload;
      if (actionType) {
        const payload = payloadStr ? JSON.parse(payloadStr) : {};
        dispatch(actionType, payload);
      }
    }
  });

  return (
    <div class="action-wrapper" onClick$={handleActionClick}>
      {/* Header */}
      {showHeader && (
        <div class="action-header flex items-center justify-between mb-4">
          <h2 class="text-lg font-semibold text-white/90">{title}</h2>
          <div class="flex items-center gap-3">
            {/* Connection status */}
            {wsUrl && (
              <div class="flex items-center gap-2">
                <div
                  class={`w-2 h-2 rounded-full ${
                    state.wsConnected ? 'bg-green-500' : 'bg-red-500'
                  }`}
                />
                <span class="text-xs text-white/40">
                  {state.wsConnected ? 'Connected' : 'Disconnected'}
                </span>
                {!state.wsConnected && (
                  <button
                    class="text-xs text-cyan-400 hover:text-cyan-300"
                    onClick$={connectWs}
                  >
                    Reconnect
                  </button>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error display */}
      {state.lastError && (
        <div class="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded text-red-400 text-sm">
          {state.lastError}
        </div>
      )}

      {/* Slotted content (buttons, forms, etc.) */}
      <div class="action-controls mb-6">
        <Slot />
      </div>

      {/* Output cells */}
      <OutputCells
        cells={state.cells}
        onToggleCollapse={toggleCollapse}
        onClear={clearCells}
      />
    </div>
  );
});

export default ActionWrapper;
