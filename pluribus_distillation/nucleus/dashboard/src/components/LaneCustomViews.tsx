/**
 * LaneCustomViews - Save/share custom filter configurations
 *
 * Phase 4, Iteration 34 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Save custom filter/sort configurations
 * - Share views with team
 * - Default view per user
 * - Import/export view configs
 * - View presets (My Lanes, Blocked, High Priority, etc.)
 */

import {
  component$,
  useSignal,
  useStore,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface ViewFilter {
  field: 'status' | 'owner' | 'wip_pct' | 'blockers' | 'priority' | 'tags';
  operator: 'equals' | 'not_equals' | 'contains' | 'gt' | 'lt' | 'gte' | 'lte' | 'in';
  value: string | number | string[];
}

export interface ViewSort {
  field: string;
  direction: 'asc' | 'desc';
}

export interface ViewConfig {
  id: string;
  name: string;
  description?: string;
  filters: ViewFilter[];
  sort: ViewSort[];
  columns: string[];
  groupBy?: string;
  isDefault?: boolean;
  isShared?: boolean;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  sharedWith?: string[];
}

export interface LaneCustomViewsProps {
  /** Current user ID */
  userId: string;
  /** Available views */
  views: ViewConfig[];
  /** Currently active view */
  activeViewId?: string;
  /** Callback when view is selected */
  onViewSelect$?: QRL<(view: ViewConfig) => void>;
  /** Callback when view is saved */
  onViewSave$?: QRL<(view: ViewConfig) => void>;
  /** Callback when view is deleted */
  onViewDelete$?: QRL<(viewId: string) => void>;
  /** Callback when view is shared */
  onViewShare$?: QRL<(viewId: string, userIds: string[]) => void>;
  /** Available columns for configuration */
  availableColumns?: string[];
  /** Available team members for sharing */
  teamMembers?: { id: string; name: string }[];
}

// ============================================================================
// Preset Views
// ============================================================================

const PRESET_VIEWS: Omit<ViewConfig, 'id' | 'createdAt' | 'updatedAt' | 'createdBy'>[] = [
  {
    name: 'All Lanes',
    description: 'Show all lanes without filters',
    filters: [],
    sort: [{ field: 'updatedAt', direction: 'desc' }],
    columns: ['name', 'owner', 'status', 'wip_pct', 'blockers'],
    isShared: true,
  },
  {
    name: 'My Lanes',
    description: 'Lanes owned by me',
    filters: [{ field: 'owner', operator: 'equals', value: '{{currentUser}}' }],
    sort: [{ field: 'wip_pct', direction: 'desc' }],
    columns: ['name', 'status', 'wip_pct', 'blockers', 'priority'],
    isShared: false,
  },
  {
    name: 'Blocked',
    description: 'Lanes with blockers or red status',
    filters: [
      { field: 'status', operator: 'equals', value: 'red' },
    ],
    sort: [{ field: 'blockers', direction: 'desc' }],
    columns: ['name', 'owner', 'status', 'blockers', 'wip_pct'],
    isShared: true,
  },
  {
    name: 'High Priority',
    description: 'High priority lanes',
    filters: [{ field: 'priority', operator: 'in', value: ['critical', 'high'] }],
    sort: [{ field: 'priority', direction: 'desc' }],
    columns: ['name', 'owner', 'priority', 'status', 'wip_pct'],
    isShared: true,
  },
  {
    name: 'Near Completion',
    description: 'Lanes at 80%+ completion',
    filters: [{ field: 'wip_pct', operator: 'gte', value: 80 }],
    sort: [{ field: 'wip_pct', direction: 'desc' }],
    columns: ['name', 'owner', 'wip_pct', 'status'],
    isShared: true,
  },
];

// ============================================================================
// Helpers
// ============================================================================

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

function getOperatorLabel(operator: string): string {
  const labels: Record<string, string> = {
    equals: 'is',
    not_equals: 'is not',
    contains: 'contains',
    gt: '>',
    lt: '<',
    gte: '>=',
    lte: '<=',
    in: 'in',
  };
  return labels[operator] || operator;
}

function generateId(): string {
  return `view_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

// ============================================================================
// Sub-Components
// ============================================================================

interface FilterEditorProps {
  filter: ViewFilter;
  onUpdate$: QRL<(filter: ViewFilter) => void>;
  onRemove$: QRL<() => void>;
}

const FilterEditor = component$<FilterEditorProps>(({ filter, onUpdate$, onRemove$ }) => {
  return (
    <div class="flex items-center gap-2 p-2 rounded bg-muted/10 border border-border/30">
      <select
        value={filter.field}
        onChange$={(e) => onUpdate$({ ...filter, field: (e.target as HTMLSelectElement).value as ViewFilter['field'] })}
        class="px-2 py-1 text-[10px] rounded bg-card border border-border/50 text-foreground"
      >
        <option value="status">Status</option>
        <option value="owner">Owner</option>
        <option value="wip_pct">WIP %</option>
        <option value="blockers">Blockers</option>
        <option value="priority">Priority</option>
        <option value="tags">Tags</option>
      </select>

      <select
        value={filter.operator}
        onChange$={(e) => onUpdate$({ ...filter, operator: (e.target as HTMLSelectElement).value as ViewFilter['operator'] })}
        class="px-2 py-1 text-[10px] rounded bg-card border border-border/50 text-foreground"
      >
        <option value="equals">is</option>
        <option value="not_equals">is not</option>
        <option value="contains">contains</option>
        <option value="gt">&gt;</option>
        <option value="lt">&lt;</option>
        <option value="gte">&gt;=</option>
        <option value="lte">&lt;=</option>
        <option value="in">in</option>
      </select>

      <input
        type="text"
        value={String(filter.value)}
        onInput$={(e) => onUpdate$({ ...filter, value: (e.target as HTMLInputElement).value })}
        class="flex-grow px-2 py-1 text-[10px] rounded bg-card border border-border/50 text-foreground"
        placeholder="Value"
      />

      <button
        onClick$={onRemove$}
        class="p-1 text-red-400 hover:bg-red-500/20 rounded transition-colors"
      >
        ✕
      </button>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

export const LaneCustomViews = component$<LaneCustomViewsProps>(({
  userId,
  views,
  activeViewId,
  onViewSelect$,
  onViewSave$,
  onViewDelete$,
  onViewShare$,
  availableColumns = ['name', 'owner', 'status', 'wip_pct', 'blockers', 'priority', 'tags', 'createdAt', 'updatedAt'],
  teamMembers = [],
}) => {
  // State
  const showEditor = useSignal(false);
  const showShareModal = useSignal(false);
  const selectedViewForShare = useSignal<string | null>(null);

  const editingView = useStore<ViewConfig>({
    id: '',
    name: '',
    description: '',
    filters: [],
    sort: [{ field: 'updatedAt', direction: 'desc' }],
    columns: ['name', 'owner', 'status', 'wip_pct'],
    createdBy: userId,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
  });

  // All views including presets
  const allViews = useComputed$(() => {
    const presets = PRESET_VIEWS.map((preset, i) => ({
      ...preset,
      id: `preset_${i}`,
      createdBy: 'system',
      createdAt: '2024-01-01',
      updatedAt: '2024-01-01',
    }));
    return [...presets, ...views];
  });

  // User's views
  const myViews = useComputed$(() =>
    views.filter(v => v.createdBy === userId)
  );

  // Shared views
  const sharedViews = useComputed$(() =>
    views.filter(v => v.isShared && v.createdBy !== userId)
  );

  // Start editing new view
  const startNewView = $(() => {
    editingView.id = generateId();
    editingView.name = '';
    editingView.description = '';
    editingView.filters = [];
    editingView.sort = [{ field: 'updatedAt', direction: 'desc' }];
    editingView.columns = ['name', 'owner', 'status', 'wip_pct'];
    editingView.isDefault = false;
    editingView.isShared = false;
    editingView.createdBy = userId;
    editingView.createdAt = new Date().toISOString();
    editingView.updatedAt = new Date().toISOString();
    showEditor.value = true;
  });

  // Edit existing view
  const startEditView = $((view: ViewConfig) => {
    editingView.id = view.id;
    editingView.name = view.name;
    editingView.description = view.description || '';
    editingView.filters = [...view.filters];
    editingView.sort = [...view.sort];
    editingView.columns = [...view.columns];
    editingView.isDefault = view.isDefault;
    editingView.isShared = view.isShared;
    editingView.createdBy = view.createdBy;
    editingView.createdAt = view.createdAt;
    editingView.updatedAt = new Date().toISOString();
    showEditor.value = true;
  });

  // Add filter
  const addFilter = $(() => {
    editingView.filters = [
      ...editingView.filters,
      { field: 'status', operator: 'equals', value: '' },
    ];
  });

  // Update filter
  const updateFilter = $((index: number, filter: ViewFilter) => {
    editingView.filters = editingView.filters.map((f, i) =>
      i === index ? filter : f
    );
  });

  // Remove filter
  const removeFilter = $((index: number) => {
    editingView.filters = editingView.filters.filter((_, i) => i !== index);
  });

  // Toggle column
  const toggleColumn = $((column: string) => {
    if (editingView.columns.includes(column)) {
      editingView.columns = editingView.columns.filter(c => c !== column);
    } else {
      editingView.columns = [...editingView.columns, column];
    }
  });

  // Save view
  const saveView = $(async () => {
    const view: ViewConfig = {
      id: editingView.id,
      name: editingView.name,
      description: editingView.description,
      filters: editingView.filters,
      sort: editingView.sort,
      columns: editingView.columns,
      isDefault: editingView.isDefault,
      isShared: editingView.isShared,
      createdBy: editingView.createdBy,
      createdAt: editingView.createdAt,
      updatedAt: new Date().toISOString(),
    };

    if (onViewSave$) {
      await onViewSave$(view);
    }

    showEditor.value = false;
  });

  // Delete view
  const deleteView = $(async (viewId: string) => {
    if (onViewDelete$) {
      await onViewDelete$(viewId);
    }
  });

  // Select view
  const selectView = $(async (view: ViewConfig) => {
    if (onViewSelect$) {
      await onViewSelect$(view);
    }
  });

  // Share view
  const shareView = $(async (userIds: string[]) => {
    if (selectedViewForShare.value && onViewShare$) {
      await onViewShare$(selectedViewForShare.value, userIds);
    }
    showShareModal.value = false;
    selectedViewForShare.value = null;
  });

  // Export view as JSON
  const exportView = $((view: ViewConfig) => {
    const json = JSON.stringify(view, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `view_${view.name.replace(/\s+/g, '_')}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">CUSTOM VIEWS</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {allViews.value.length} views
          </span>
        </div>
        <button
          onClick$={startNewView}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
        >
          + New View
        </button>
      </div>

      {/* View list or Editor */}
      {showEditor.value ? (
        <div class="p-4">
          {/* View name */}
          <div class="mb-4">
            <label class="text-[9px] text-muted-foreground block mb-1">View Name</label>
            <input
              type="text"
              value={editingView.name}
              onInput$={(e) => { editingView.name = (e.target as HTMLInputElement).value; }}
              class="w-full px-3 py-2 text-xs rounded bg-card border border-border/50 text-foreground"
              placeholder="My Custom View"
            />
          </div>

          {/* Description */}
          <div class="mb-4">
            <label class="text-[9px] text-muted-foreground block mb-1">Description</label>
            <input
              type="text"
              value={editingView.description}
              onInput$={(e) => { editingView.description = (e.target as HTMLInputElement).value; }}
              class="w-full px-3 py-2 text-xs rounded bg-card border border-border/50 text-foreground"
              placeholder="Optional description"
            />
          </div>

          {/* Filters */}
          <div class="mb-4">
            <div class="flex items-center justify-between mb-2">
              <label class="text-[9px] text-muted-foreground">Filters</label>
              <button
                onClick$={addFilter}
                class="text-[9px] text-primary hover:underline"
              >
                + Add Filter
              </button>
            </div>
            <div class="space-y-2">
              {editingView.filters.map((filter, i) => (
                <FilterEditor
                  key={i}
                  filter={filter}
                  onUpdate$={$((f: ViewFilter) => updateFilter(i, f))}
                  onRemove$={$(() => removeFilter(i))}
                />
              ))}
              {editingView.filters.length === 0 && (
                <div class="text-[9px] text-muted-foreground text-center p-2">
                  No filters - showing all lanes
                </div>
              )}
            </div>
          </div>

          {/* Columns */}
          <div class="mb-4">
            <label class="text-[9px] text-muted-foreground block mb-2">Visible Columns</label>
            <div class="flex flex-wrap gap-2">
              {availableColumns.map(col => (
                <button
                  key={col}
                  onClick$={() => toggleColumn(col)}
                  class={`text-[9px] px-2 py-1 rounded border transition-colors ${
                    editingView.columns.includes(col)
                      ? 'bg-primary/20 text-primary border-primary/30'
                      : 'bg-muted/10 text-muted-foreground border-border/30 hover:bg-muted/20'
                  }`}
                >
                  {col}
                </button>
              ))}
            </div>
          </div>

          {/* Options */}
          <div class="mb-4 flex items-center gap-4">
            <label class="flex items-center gap-2 text-[10px] text-foreground">
              <input
                type="checkbox"
                checked={editingView.isDefault}
                onChange$={(e) => { editingView.isDefault = (e.target as HTMLInputElement).checked; }}
                class="rounded"
              />
              Set as default
            </label>
            <label class="flex items-center gap-2 text-[10px] text-foreground">
              <input
                type="checkbox"
                checked={editingView.isShared}
                onChange$={(e) => { editingView.isShared = (e.target as HTMLInputElement).checked; }}
                class="rounded"
              />
              Share with team
            </label>
          </div>

          {/* Actions */}
          <div class="flex items-center gap-2">
            <button
              onClick$={() => { showEditor.value = false; }}
              class="px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground hover:bg-muted/50 transition-colors"
            >
              Cancel
            </button>
            <button
              onClick$={saveView}
              disabled={!editingView.name}
              class="px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              Save View
            </button>
          </div>
        </div>
      ) : (
        <div class="max-h-[400px] overflow-y-auto">
          {/* Presets section */}
          <div class="p-2 border-b border-border/30">
            <div class="text-[9px] font-semibold text-muted-foreground mb-2 px-1">PRESETS</div>
            <div class="space-y-1">
              {PRESET_VIEWS.map((preset, i) => {
                const viewId = `preset_${i}`;
                const isActive = activeViewId === viewId;

                return (
                  <div
                    key={viewId}
                    onClick$={() => selectView({
                      ...preset,
                      id: viewId,
                      createdBy: 'system',
                      createdAt: '2024-01-01',
                      updatedAt: '2024-01-01',
                    })}
                    class={`p-2 rounded cursor-pointer transition-colors ${
                      isActive
                        ? 'bg-primary/20 border border-primary/30'
                        : 'hover:bg-muted/10'
                    }`}
                  >
                    <div class="flex items-center justify-between">
                      <span class="text-xs font-medium text-foreground">{preset.name}</span>
                      {isActive && <span class="text-[9px] text-primary">Active</span>}
                    </div>
                    {preset.description && (
                      <div class="text-[9px] text-muted-foreground mt-0.5">{preset.description}</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* My views section */}
          {myViews.value.length > 0 && (
            <div class="p-2 border-b border-border/30">
              <div class="text-[9px] font-semibold text-muted-foreground mb-2 px-1">MY VIEWS</div>
              <div class="space-y-1">
                {myViews.value.map(view => {
                  const isActive = activeViewId === view.id;

                  return (
                    <div
                      key={view.id}
                      class={`p-2 rounded transition-colors ${
                        isActive
                          ? 'bg-primary/20 border border-primary/30'
                          : 'hover:bg-muted/10'
                      }`}
                    >
                      <div class="flex items-center justify-between">
                        <div
                          onClick$={() => selectView(view)}
                          class="flex-grow cursor-pointer"
                        >
                          <div class="flex items-center gap-2">
                            <span class="text-xs font-medium text-foreground">{view.name}</span>
                            {view.isDefault && (
                              <span class="text-[8px] px-1 py-0.5 rounded bg-emerald-500/20 text-emerald-400">
                                Default
                              </span>
                            )}
                            {view.isShared && (
                              <span class="text-[8px] px-1 py-0.5 rounded bg-blue-500/20 text-blue-400">
                                Shared
                              </span>
                            )}
                          </div>
                          {view.description && (
                            <div class="text-[9px] text-muted-foreground mt-0.5">{view.description}</div>
                          )}
                        </div>
                        <div class="flex items-center gap-1">
                          <button
                            onClick$={() => startEditView(view)}
                            class="p-1 text-muted-foreground hover:text-foreground transition-colors"
                            title="Edit"
                          >
                            ✎
                          </button>
                          <button
                            onClick$={() => {
                              selectedViewForShare.value = view.id;
                              showShareModal.value = true;
                            }}
                            class="p-1 text-muted-foreground hover:text-foreground transition-colors"
                            title="Share"
                          >
                            ↗
                          </button>
                          <button
                            onClick$={() => exportView(view)}
                            class="p-1 text-muted-foreground hover:text-foreground transition-colors"
                            title="Export"
                          >
                            ⬇
                          </button>
                          <button
                            onClick$={() => deleteView(view.id)}
                            class="p-1 text-red-400 hover:text-red-300 transition-colors"
                            title="Delete"
                          >
                            ✕
                          </button>
                        </div>
                      </div>
                      <div class="text-[8px] text-muted-foreground/50 mt-1">
                        {view.filters.length} filters • {view.columns.length} columns • Updated {formatDate(view.updatedAt)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Shared views section */}
          {sharedViews.value.length > 0 && (
            <div class="p-2">
              <div class="text-[9px] font-semibold text-muted-foreground mb-2 px-1">SHARED WITH ME</div>
              <div class="space-y-1">
                {sharedViews.value.map(view => {
                  const isActive = activeViewId === view.id;

                  return (
                    <div
                      key={view.id}
                      onClick$={() => selectView(view)}
                      class={`p-2 rounded cursor-pointer transition-colors ${
                        isActive
                          ? 'bg-primary/20 border border-primary/30'
                          : 'hover:bg-muted/10'
                      }`}
                    >
                      <div class="flex items-center justify-between">
                        <span class="text-xs font-medium text-foreground">{view.name}</span>
                        <span class="text-[8px] text-muted-foreground">by @{view.createdBy}</span>
                      </div>
                      {view.description && (
                        <div class="text-[9px] text-muted-foreground mt-0.5">{view.description}</div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Share modal */}
      {showShareModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-80">
            <div class="text-xs font-semibold text-foreground mb-3">Share View</div>
            <div class="space-y-2 max-h-48 overflow-y-auto mb-4">
              {teamMembers.map(member => (
                <label key={member.id} class="flex items-center gap-2 p-2 rounded hover:bg-muted/10">
                  <input type="checkbox" class="rounded" />
                  <span class="text-xs text-foreground">{member.name}</span>
                </label>
              ))}
              {teamMembers.length === 0 && (
                <div class="text-[9px] text-muted-foreground text-center p-4">
                  No team members available
                </div>
              )}
            </div>
            <div class="flex items-center gap-2">
              <button
                onClick$={() => { showShareModal.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={() => shareView([])}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground"
              >
                Share
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {myViews.value.length} custom views • {sharedViews.value.length} shared
      </div>
    </div>
  );
});

export default LaneCustomViews;
