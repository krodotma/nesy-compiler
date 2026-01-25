/**
 * LaneUndoRedoPanel - Visual History Browser
 *
 * Phase 7, Iteration 56 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Visual undo/redo history browser
 * - One-click undo/redo buttons
 * - History timeline view
 * - Keyboard shortcut hints
 * - Stack depth indicator
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  $,
} from '@builder.io/qwik';

// M3 Components - LaneUndoRedoPanel
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/button/text-button.js';

import {
  getGlobalUndoRedoManager,
  type UndoRedoManager,
} from '../lib/lanes/undo-redo';

// ============================================================================
// Types
// ============================================================================

export interface LaneUndoRedoPanelProps {
  /** Show expanded history view */
  showHistory?: boolean;
  /** Maximum history items to display */
  maxHistoryItems?: number;
  /** Compact mode (icons only) */
  compact?: boolean;
  /** Custom undo/redo manager */
  manager?: UndoRedoManager;
}

// ============================================================================
// Helpers
// ============================================================================

function formatTimestamp(ts: number): string {
  const now = Date.now();
  const diff = now - ts;

  if (diff < 60000) {
    return 'just now';
  } else if (diff < 3600000) {
    const mins = Math.floor(diff / 60000);
    return `${mins}m ago`;
  } else if (diff < 86400000) {
    const hours = Math.floor(diff / 3600000);
    return `${hours}h ago`;
  } else {
    return new Date(ts).toLocaleDateString();
  }
}

function getActionIcon(description: string): string {
  if (description.toLowerCase().includes('update')) return '\u270F';
  if (description.toLowerCase().includes('add')) return '+';
  if (description.toLowerCase().includes('remove')) return '-';
  if (description.toLowerCase().includes('change')) return '\u21C4';
  return '\u2022';
}

// ============================================================================
// Component
// ============================================================================

export const LaneUndoRedoPanel = component$<LaneUndoRedoPanelProps>(({
  showHistory = false,
  maxHistoryItems = 10,
  compact = false,
  manager: externalManager,
}) => {
  // State
  const canUndo = useSignal(false);
  const canRedo = useSignal(false);
  const undoStack = useSignal<Array<{ id: string; description: string; timestamp: number }>>([]);
  const redoStack = useSignal<Array<{ id: string; description: string; timestamp: number }>>([]);
  const historyExpanded = useSignal(showHistory);

  // Get manager
  const manager = externalManager || getGlobalUndoRedoManager();

  // Subscribe to undo/redo state changes
  useVisibleTask$(({ cleanup }) => {
    const unsubscribe = manager.subscribe((undo, redo) => {
      canUndo.value = undo;
      canRedo.value = redo;
      undoStack.value = manager.getUndoStack().slice(-maxHistoryItems).reverse();
      redoStack.value = manager.getRedoStack().slice(-maxHistoryItems);
    });

    cleanup(unsubscribe);
  });

  // Actions
  const handleUndo = $(() => {
    manager.undo();
  });

  const handleRedo = $(() => {
    manager.redo();
  });

  const handleClear = $(() => {
    manager.clear();
  });

  // Compact mode
  if (compact) {
    return (
      <div class="flex items-center gap-1">
        <button
          onClick$={handleUndo}
          disabled={!canUndo.value}
          class={`p-1.5 rounded text-xs transition-colors ${
            canUndo.value
              ? 'bg-muted/30 hover:bg-muted/50 text-foreground'
              : 'bg-muted/10 text-muted-foreground/50 cursor-not-allowed'
          }`}
          title="Undo (Ctrl+Z)"
        >
          \u21A9
        </button>
        <button
          onClick$={handleRedo}
          disabled={!canRedo.value}
          class={`p-1.5 rounded text-xs transition-colors ${
            canRedo.value
              ? 'bg-muted/30 hover:bg-muted/50 text-foreground'
              : 'bg-muted/10 text-muted-foreground/50 cursor-not-allowed'
          }`}
          title="Redo (Ctrl+Shift+Z)"
        >
          \u21AA
        </button>
        {(undoStack.value.length > 0 || redoStack.value.length > 0) && (
          <span class="text-[9px] text-muted-foreground ml-1">
            {undoStack.value.length}/{redoStack.value.length}
          </span>
        )}
      </div>
    );
  }

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-2 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-[10px] font-semibold text-muted-foreground">UNDO/REDO</span>
          <span class="text-[9px] px-1.5 py-0.5 rounded bg-muted/20 text-muted-foreground">
            {undoStack.value.length} / {redoStack.value.length}
          </span>
        </div>
        <div class="flex items-center gap-1">
          <button
            onClick$={() => { historyExpanded.value = !historyExpanded.value; }}
            class="text-[9px] px-2 py-0.5 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
          >
            {historyExpanded.value ? 'Hide History' : 'Show History'}
          </button>
          {(undoStack.value.length > 0 || redoStack.value.length > 0) && (
            <button
              onClick$={handleClear}
              class="text-[9px] px-2 py-0.5 rounded bg-red-500/10 hover:bg-red-500/20 text-red-400"
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {/* Action buttons */}
      <div class="p-2 flex items-center gap-2">
        <button
          onClick$={handleUndo}
          disabled={!canUndo.value}
          class={`flex-1 flex items-center justify-center gap-2 p-2 rounded transition-colors ${
            canUndo.value
              ? 'bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30'
              : 'bg-muted/10 text-muted-foreground/50 border border-border/30 cursor-not-allowed'
          }`}
        >
          <span class="text-sm">\u21A9</span>
          <span class="text-xs">Undo</span>
          <span class="text-[8px] px-1 py-0.5 rounded bg-muted/30 text-muted-foreground">
            Ctrl+Z
          </span>
        </button>
        <button
          onClick$={handleRedo}
          disabled={!canRedo.value}
          class={`flex-1 flex items-center justify-center gap-2 p-2 rounded transition-colors ${
            canRedo.value
              ? 'bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30'
              : 'bg-muted/10 text-muted-foreground/50 border border-border/30 cursor-not-allowed'
          }`}
        >
          <span class="text-sm">\u21AA</span>
          <span class="text-xs">Redo</span>
          <span class="text-[8px] px-1 py-0.5 rounded bg-muted/30 text-muted-foreground">
            Ctrl+Y
          </span>
        </button>
      </div>

      {/* History timeline */}
      {historyExpanded.value && (
        <div class="border-t border-border/50 max-h-[300px] overflow-y-auto">
          {/* Redo stack (future) */}
          {redoStack.value.length > 0 && (
            <div class="p-2 border-b border-border/30 bg-muted/5">
              <div class="text-[9px] font-semibold text-muted-foreground mb-2">
                REDO ({redoStack.value.length})
              </div>
              <div class="space-y-1">
                {redoStack.value.map((item, index) => (
                  <div
                    key={item.id}
                    class="flex items-center gap-2 p-1.5 rounded bg-purple-500/5 border border-purple-500/20 text-[10px]"
                  >
                    <span class="text-purple-400 w-4 text-center">
                      {getActionIcon(item.description)}
                    </span>
                    <span class="flex-1 text-foreground/70 truncate">
                      {item.description}
                    </span>
                    <span class="text-muted-foreground/50">
                      {formatTimestamp(item.timestamp)}
                    </span>
                    <span class="text-[8px] text-purple-400/50">
                      +{index + 1}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Current position indicator */}
          <div class="flex items-center gap-2 px-2 py-1 bg-primary/5">
            <div class="flex-1 h-px bg-primary/30" />
            <span class="text-[9px] text-primary">Current State</span>
            <div class="flex-1 h-px bg-primary/30" />
          </div>

          {/* Undo stack (past) */}
          {undoStack.value.length > 0 ? (
            <div class="p-2">
              <div class="text-[9px] font-semibold text-muted-foreground mb-2">
                HISTORY ({undoStack.value.length})
              </div>
              <div class="space-y-1">
                {undoStack.value.map((item, index) => (
                  <div
                    key={item.id}
                    class="flex items-center gap-2 p-1.5 rounded bg-muted/10 border border-border/30 text-[10px] hover:bg-muted/20 transition-colors"
                  >
                    <span class="text-cyan-400 w-4 text-center">
                      {getActionIcon(item.description)}
                    </span>
                    <span class="flex-1 text-foreground truncate">
                      {item.description}
                    </span>
                    <span class="text-muted-foreground/50">
                      {formatTimestamp(item.timestamp)}
                    </span>
                    <span class="text-[8px] text-muted-foreground/50">
                      -{index + 1}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div class="p-4 text-center text-[10px] text-muted-foreground">
              No history yet. Make some changes to start tracking.
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        Keyboard: Ctrl+Z (undo) | Ctrl+Shift+Z or Ctrl+Y (redo)
      </div>
    </div>
  );
});

export default LaneUndoRedoPanel;
