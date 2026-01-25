/**
 * LaneBulkActions - Bulk Action Bar for Multiple Lanes
 *
 * Phase 8, Iteration 62 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Floating action bar when items selected
 * - Bulk status changes
 * - Bulk owner assignment
 * - Bulk tag operations
 * - Progress tracking for batch operations
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LaneBulkActions
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/progress/linear-progress.js';

import type { Lane } from '../lib/lanes/store';

// ============================================================================
// Types
// ============================================================================

export type BulkActionType =
  | 'set-status'
  | 'set-owner'
  | 'add-tag'
  | 'remove-tag'
  | 'archive'
  | 'delete'
  | 'export';

export interface BulkActionResult {
  success: boolean;
  affectedCount: number;
  errors: Array<{ laneId: string; error: string }>;
}

export interface LaneBulkActionsProps {
  /** Selected lane IDs */
  selectedIds: string[];
  /** All available lanes (for preview) */
  lanes: Lane[];
  /** Available owners for assignment */
  availableOwners?: string[];
  /** Available tags */
  availableTags?: string[];
  /** Callback when bulk action is executed */
  onAction$?: QRL<(action: BulkActionType, params: unknown) => Promise<BulkActionResult>>;
  /** Callback to clear selection */
  onClearSelection$?: QRL<() => void>;
  /** Position */
  position?: 'top' | 'bottom' | 'floating';
}

// ============================================================================
// Component
// ============================================================================

export const LaneBulkActions = component$<LaneBulkActionsProps>(({
  selectedIds,
  lanes,
  availableOwners = ['claude', 'codex', 'gemini', 'qwen'],
  availableTags = ['priority', 'blocked', 'review', 'testing', 'deployed'],
  onAction$,
  onClearSelection$,
  position = 'floating',
}) => {
  // State
  const activeAction = useSignal<BulkActionType | null>(null);
  const isProcessing = useSignal(false);
  const progress = useSignal({ current: 0, total: 0 });
  const result = useSignal<BulkActionResult | null>(null);

  // Selected lanes
  const selectedLanes = useComputed$(() =>
    lanes.filter(l => selectedIds.includes(l.id))
  );

  // Execute bulk action
  const executeAction = $(async (action: BulkActionType, params: unknown) => {
    if (!onAction$) return;

    isProcessing.value = true;
    progress.value = { current: 0, total: selectedIds.length };
    result.value = null;

    try {
      const actionResult = await onAction$(action, params);
      result.value = actionResult;

      // Auto-clear selection on success
      if (actionResult.success && onClearSelection$) {
        await onClearSelection$();
      }
    } catch (err: any) {
      result.value = {
        success: false,
        affectedCount: 0,
        errors: [{ laneId: 'all', error: err?.message || 'Unknown error' }],
      };
    } finally {
      isProcessing.value = false;
      activeAction.value = null;
    }
  });

  // Action handlers
  const handleSetStatus = $(async (status: 'green' | 'yellow' | 'red') => {
    await executeAction('set-status', { status, laneIds: selectedIds });
  });

  const handleSetOwner = $(async (owner: string) => {
    await executeAction('set-owner', { owner, laneIds: selectedIds });
  });

  const handleAddTag = $(async (tag: string) => {
    await executeAction('add-tag', { tag, laneIds: selectedIds });
  });

  const handleRemoveTag = $(async (tag: string) => {
    await executeAction('remove-tag', { tag, laneIds: selectedIds });
  });

  const handleArchive = $(async () => {
    await executeAction('archive', { laneIds: selectedIds });
  });

  const handleExport = $(async () => {
    await executeAction('export', { laneIds: selectedIds });
  });

  // Don't render if nothing selected
  if (selectedIds.length === 0) {
    return null;
  }

  const positionClasses = {
    top: 'top-4 left-1/2 -translate-x-1/2',
    bottom: 'bottom-4 left-1/2 -translate-x-1/2',
    floating: 'bottom-4 left-1/2 -translate-x-1/2 animate-in fade-in slide-in-from-bottom-4',
  };

  return (
    <div
      class={`fixed z-50 ${positionClasses[position]} rounded-lg border border-[var(--glass-border)] bg-card/95 backdrop-blur-md shadow-xl p-3 min-w-[400px] glass-surface-overlay`}
    >
      {/* Header */}
      <div class="flex items-center justify-between mb-3">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-primary">
            {selectedIds.length} lane{selectedIds.length > 1 ? 's' : ''} selected
          </span>
          {isProcessing.value && (
            <span class="text-[9px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 animate-pulse">
              Processing...
            </span>
          )}
        </div>
        <button
          onClick$={onClearSelection$}
          disabled={isProcessing.value}
          class="text-muted-foreground hover:text-foreground text-xs"
        >
          \u2715
        </button>
      </div>

      {/* Result message */}
      {result.value && (
        <div
          class={`mb-3 p-2 rounded text-[10px] ${
            result.value.success
              ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30'
              : 'bg-red-500/10 text-red-400 border border-red-500/30'
          }`}
        >
          {result.value.success
            ? `Successfully updated ${result.value.affectedCount} lane${result.value.affectedCount > 1 ? 's' : ''}`
            : `Failed: ${result.value.errors[0]?.error || 'Unknown error'}`}
        </div>
      )}

      {/* Action buttons */}
      <div class="grid grid-cols-4 gap-2 mb-3">
        {/* Set Status */}
        <div class="relative">
          <button
            onClick$={() => { activeAction.value = activeAction.value === 'set-status' ? null : 'set-status'; }}
            disabled={isProcessing.value}
            class="w-full text-[10px] px-2 py-1.5 rounded bg-muted/20 hover:bg-muted/40 text-foreground transition-colors disabled:opacity-50"
          >
            Set Status
          </button>
          {activeAction.value === 'set-status' && (
            <div class="absolute top-full left-0 mt-1 p-2 rounded glass-dropdown border border-[var(--glass-border)] shadow-lg z-10 min-w-[100px]">
              <button
                onClick$={() => handleSetStatus('green')}
                class="w-full text-[10px] px-2 py-1 rounded hover:bg-emerald-500/20 text-emerald-400 text-left"
              >
                \u2713 Green
              </button>
              <button
                onClick$={() => handleSetStatus('yellow')}
                class="w-full text-[10px] px-2 py-1 rounded hover:bg-amber-500/20 text-amber-400 text-left"
              >
                \u25CF Yellow
              </button>
              <button
                onClick$={() => handleSetStatus('red')}
                class="w-full text-[10px] px-2 py-1 rounded hover:bg-red-500/20 text-red-400 text-left"
              >
                ! Red
              </button>
            </div>
          )}
        </div>

        {/* Set Owner */}
        <div class="relative">
          <button
            onClick$={() => { activeAction.value = activeAction.value === 'set-owner' ? null : 'set-owner'; }}
            disabled={isProcessing.value}
            class="w-full text-[10px] px-2 py-1.5 rounded bg-muted/20 hover:bg-muted/40 text-foreground transition-colors disabled:opacity-50"
          >
            Set Owner
          </button>
          {activeAction.value === 'set-owner' && (
            <div class="absolute top-full left-0 mt-1 p-2 rounded glass-dropdown border border-[var(--glass-border)] shadow-lg z-10 min-w-[100px]">
              {availableOwners.map(owner => (
                <button
                  key={owner}
                  onClick$={() => handleSetOwner(owner)}
                  class="w-full text-[10px] px-2 py-1 rounded hover:bg-cyan-500/20 text-cyan-400 text-left"
                >
                  @{owner}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Add Tag */}
        <div class="relative">
          <button
            onClick$={() => { activeAction.value = activeAction.value === 'add-tag' ? null : 'add-tag'; }}
            disabled={isProcessing.value}
            class="w-full text-[10px] px-2 py-1.5 rounded bg-muted/20 hover:bg-muted/40 text-foreground transition-colors disabled:opacity-50"
          >
            Add Tag
          </button>
          {activeAction.value === 'add-tag' && (
            <div class="absolute top-full left-0 mt-1 p-2 rounded glass-dropdown border border-[var(--glass-border)] shadow-lg z-10 min-w-[100px]">
              {availableTags.map(tag => (
                <button
                  key={tag}
                  onClick$={() => handleAddTag(tag)}
                  class="w-full text-[10px] px-2 py-1 rounded hover:bg-purple-500/20 text-purple-400 text-left"
                >
                  #{tag}
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Export */}
        <button
          onClick$={handleExport}
          disabled={isProcessing.value}
          class="text-[10px] px-2 py-1.5 rounded bg-muted/20 hover:bg-muted/40 text-foreground transition-colors disabled:opacity-50"
        >
          Export
        </button>
      </div>

      {/* Danger zone */}
      <div class="flex items-center gap-2 pt-3 border-t border-[var(--glass-border-subtle)]">
        <button
          onClick$={handleArchive}
          disabled={isProcessing.value}
          class="text-[10px] px-3 py-1.5 rounded bg-amber-500/10 hover:bg-amber-500/20 text-amber-400 border border-amber-500/30 transition-colors disabled:opacity-50"
        >
          Archive Selected
        </button>
        <span class="flex-1" />
        <span class="text-[9px] text-muted-foreground">
          {selectedLanes.value.filter(l => l.status === 'red').length} blocked,
          {selectedLanes.value.filter(l => l.status === 'green').length} complete
        </span>
      </div>

      {/* Progress bar */}
      {isProcessing.value && progress.value.total > 0 && (
        <div class="mt-3">
          <div class="h-1 rounded-full bg-muted/20 overflow-hidden">
            <div
              class="h-full bg-primary transition-all"
              style={{ width: `${(progress.value.current / progress.value.total) * 100}%` }}
            />
          </div>
          <div class="text-[8px] text-muted-foreground text-center mt-1">
            {progress.value.current} / {progress.value.total}
          </div>
        </div>
      )}
    </div>
  );
});

export default LaneBulkActions;
