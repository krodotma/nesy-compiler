/**
 * LaneMultiSelect - Multi-Lane Selection Component
 *
 * Phase 8, Iteration 61 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Checkbox selection for multiple lanes
 * - Shift+click range selection
 * - Ctrl+A select all
 * - Selection state management
 * - Visual selection indicators
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LaneMultiSelect
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/checkbox/checkbox.js';
import '@material/web/button/text-button.js';

import type { Lane } from '../lib/lanes/store';

// ============================================================================
// Types
// ============================================================================

export interface LaneMultiSelectProps {
  /** Available lanes */
  lanes: Lane[];
  /** Currently selected lane IDs */
  selectedIds?: string[];
  /** Callback when selection changes */
  onSelectionChange$?: QRL<(selectedIds: string[]) => void>;
  /** Enable Shift+click range selection */
  enableRangeSelect?: boolean;
  /** Enable Ctrl+A select all */
  enableSelectAll?: boolean;
  /** Compact mode (smaller checkboxes) */
  compact?: boolean;
  /** Disable selection */
  disabled?: boolean;
}

// ============================================================================
// Component
// ============================================================================

export const LaneMultiSelect = component$<LaneMultiSelectProps>(({
  lanes,
  selectedIds: propSelectedIds,
  onSelectionChange$,
  enableRangeSelect = true,
  enableSelectAll = true,
  compact = false,
  disabled = false,
}) => {
  // Selection state
  const selectedIds = useSignal<Set<string>>(new Set(propSelectedIds || []));
  const lastSelectedIndex = useSignal<number>(-1);

  // Sync with prop changes
  useVisibleTask$(({ track }) => {
    track(() => propSelectedIds);
    if (propSelectedIds) {
      selectedIds.value = new Set(propSelectedIds);
    }
  });

  // Computed values
  const allSelected = useComputed$(() =>
    lanes.length > 0 && lanes.every(l => selectedIds.value.has(l.id))
  );

  const someSelected = useComputed$(() =>
    lanes.some(l => selectedIds.value.has(l.id)) && !allSelected.value
  );

  const selectionCount = useComputed$(() => selectedIds.value.size);

  // Notify parent of selection changes
  const notifyChange = $(async () => {
    if (onSelectionChange$) {
      await onSelectionChange$(Array.from(selectedIds.value));
    }
  });

  // Toggle single item
  const toggleItem = $((id: string, index: number, shiftKey: boolean) => {
    if (disabled) return;

    const newSelected = new Set(selectedIds.value);

    // Shift+click range selection
    if (shiftKey && enableRangeSelect && lastSelectedIndex.value >= 0) {
      const start = Math.min(lastSelectedIndex.value, index);
      const end = Math.max(lastSelectedIndex.value, index);

      for (let i = start; i <= end; i++) {
        newSelected.add(lanes[i].id);
      }
    } else {
      // Toggle single item
      if (newSelected.has(id)) {
        newSelected.delete(id);
      } else {
        newSelected.add(id);
      }
    }

    selectedIds.value = newSelected;
    lastSelectedIndex.value = index;
    notifyChange();
  });

  // Select all / deselect all
  const toggleSelectAll = $(() => {
    if (disabled) return;

    if (allSelected.value) {
      selectedIds.value = new Set();
    } else {
      selectedIds.value = new Set(lanes.map(l => l.id));
    }
    notifyChange();
  });

  // Clear selection
  const clearSelection = $(() => {
    if (disabled) return;
    selectedIds.value = new Set();
    lastSelectedIndex.value = -1;
    notifyChange();
  });

  // Keyboard shortcuts
  useVisibleTask$(({ cleanup }) => {
    if (!enableSelectAll || typeof window === 'undefined') return;

    const handleKeyDown = (e: KeyboardEvent) => {
      if (disabled) return;

      // Ignore if in input/textarea
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

      // Ctrl+A / Cmd+A for select all
      if ((e.ctrlKey || e.metaKey) && e.key === 'a') {
        e.preventDefault();
        toggleSelectAll();
      }

      // Escape to clear selection
      if (e.key === 'Escape' && selectedIds.value.size > 0) {
        e.preventDefault();
        clearSelection();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    cleanup(() => window.removeEventListener('keydown', handleKeyDown));
  });

  // Status color helper
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'green': return 'bg-emerald-500';
      case 'yellow': return 'bg-amber-500';
      case 'red': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const checkboxSize = compact ? 'w-3 h-3' : 'w-4 h-4';
  const itemPadding = compact ? 'p-1.5' : 'p-2';

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header with select all */}
      <div class="flex items-center justify-between p-2 border-b border-border/50 bg-muted/5">
        <div class="flex items-center gap-2">
          <button
            onClick$={toggleSelectAll}
            disabled={disabled || lanes.length === 0}
            class={`${checkboxSize} rounded border transition-colors ${
              disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer'
            } ${
              allSelected.value
                ? 'bg-primary border-primary'
                : someSelected.value
                ? 'bg-primary/50 border-primary'
                : 'bg-card border-border/50 hover:border-border'
            }`}
          >
            {(allSelected.value || someSelected.value) && (
              <span class="text-primary-foreground text-[10px] flex items-center justify-center">
                {allSelected.value ? '\u2713' : '-'}
              </span>
            )}
          </button>
          <span class="text-[10px] text-muted-foreground">
            {selectionCount.value > 0
              ? `${selectionCount.value} of ${lanes.length} selected`
              : `${lanes.length} lanes`}
          </span>
        </div>

        {selectionCount.value > 0 && (
          <button
            onClick$={clearSelection}
            class="text-[9px] px-2 py-0.5 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
          >
            Clear
          </button>
        )}
      </div>

      {/* Lane list */}
      <div class="max-h-[300px] overflow-y-auto">
        {lanes.length > 0 ? (
          <div class="divide-y divide-border/30">
            {lanes.map((lane, index) => {
              const isSelected = selectedIds.value.has(lane.id);
              return (
                <div
                  key={lane.id}
                  onClick$={(e) => toggleItem(lane.id, index, e.shiftKey)}
                  class={`flex items-center gap-2 ${itemPadding} cursor-pointer transition-colors ${
                    disabled ? 'opacity-50 cursor-not-allowed' : ''
                  } ${
                    isSelected
                      ? 'bg-primary/10'
                      : 'hover:bg-muted/10'
                  }`}
                >
                  {/* Checkbox */}
                  <div
                    class={`${checkboxSize} rounded border transition-colors flex-shrink-0 ${
                      isSelected
                        ? 'bg-primary border-primary'
                        : 'bg-card border-border/50'
                    }`}
                  >
                    {isSelected && (
                      <span class="text-primary-foreground text-[10px] flex items-center justify-center h-full">
                        \u2713
                      </span>
                    )}
                  </div>

                  {/* Status indicator */}
                  <div class={`w-2 h-2 rounded-full flex-shrink-0 ${getStatusColor(lane.status)}`} />

                  {/* Lane info */}
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-2">
                      <span class={`${compact ? 'text-[10px]' : 'text-xs'} font-medium text-foreground truncate`}>
                        {lane.name}
                      </span>
                      <span class={`${compact ? 'text-[8px]' : 'text-[9px]'} text-muted-foreground`}>
                        {lane.wip_pct}%
                      </span>
                    </div>
                    {!compact && (
                      <div class="text-[9px] text-muted-foreground truncate">
                        @{lane.owner}
                        {lane.blockers.length > 0 && (
                          <span class="text-red-400 ml-2">
                            {lane.blockers.length} blocker{lane.blockers.length > 1 ? 's' : ''}
                          </span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* WIP indicator */}
                  <div class={`${compact ? 'w-8' : 'w-12'} h-1.5 rounded-full bg-muted/20 overflow-hidden flex-shrink-0`}>
                    <div
                      class={`h-full rounded-full ${
                        lane.wip_pct >= 90 ? 'bg-emerald-500' :
                        lane.wip_pct >= 60 ? 'bg-cyan-500' :
                        lane.wip_pct >= 30 ? 'bg-amber-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${lane.wip_pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div class="p-4 text-center text-[10px] text-muted-foreground">
            No lanes available
          </div>
        )}
      </div>

      {/* Footer with keyboard hints */}
      {enableRangeSelect && (
        <div class="p-2 border-t border-border/50 text-[8px] text-muted-foreground text-center">
          Shift+Click for range | {enableSelectAll && 'Ctrl+A for all | '}Esc to clear
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Hook for external usage
// ============================================================================

export interface UseMultiSelectReturn {
  selectedIds: Set<string>;
  isSelected: (id: string) => boolean;
  toggle: (id: string) => void;
  selectAll: (ids: string[]) => void;
  deselectAll: () => void;
  toggleAll: (ids: string[]) => void;
}

export default LaneMultiSelect;
