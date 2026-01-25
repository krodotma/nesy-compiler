/**
 * LaneDependencyManager - Enhanced Dependency Management
 *
 * Phase 8, Iteration 64 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Dependency graph visualization
 * - Cascade configuration
 * - Dependency templates
 * - Critical path analysis
 * - Blocking chain detection
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LaneDependencyManager
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';

import type { Lane } from '../lib/lanes/store';

// ============================================================================
// Types
// ============================================================================

export interface Dependency {
  id: string;
  sourceId: string;
  targetId: string;
  type: 'blocks' | 'depends_on' | 'related' | 'child_of' | 'follows';
  cascadeConfig?: CascadeConfig;
  createdAt: string;
  metadata?: Record<string, unknown>;
}

export interface CascadeConfig {
  propagateStatus: boolean;
  propagateBlockers: boolean;
  autoProgress: boolean;
  delayDays?: number;
}

export interface DependencyTemplate {
  id: string;
  name: string;
  description: string;
  dependencies: Array<{
    fromRole: string;
    toRole: string;
    type: Dependency['type'];
    cascadeConfig?: CascadeConfig;
  }>;
}

export interface CriticalPath {
  lanes: string[];
  totalDuration: number;
  bottlenecks: string[];
}

export interface LaneDependencyManagerProps {
  /** Available lanes */
  lanes: Lane[];
  /** Existing dependencies */
  dependencies: Dependency[];
  /** Dependency templates */
  templates?: DependencyTemplate[];
  /** Callback when dependency is added */
  onAddDependency$?: QRL<(dep: Omit<Dependency, 'id' | 'createdAt'>) => void>;
  /** Callback when dependency is updated */
  onUpdateDependency$?: QRL<(id: string, updates: Partial<Dependency>) => void>;
  /** Callback when dependency is removed */
  onRemoveDependency$?: QRL<(id: string) => void>;
  /** Callback when template is applied */
  onApplyTemplate$?: QRL<(templateId: string, roleMapping: Record<string, string>) => void>;
}

// ============================================================================
// Built-in Templates
// ============================================================================

const DEFAULT_TEMPLATES: DependencyTemplate[] = [
  {
    id: 'sequential',
    name: 'Sequential Workflow',
    description: 'Each lane starts after the previous completes',
    dependencies: [
      { fromRole: 'step1', toRole: 'step2', type: 'follows', cascadeConfig: { propagateStatus: true, propagateBlockers: true, autoProgress: false } },
      { fromRole: 'step2', toRole: 'step3', type: 'follows', cascadeConfig: { propagateStatus: true, propagateBlockers: true, autoProgress: false } },
    ],
  },
  {
    id: 'parallel-merge',
    name: 'Parallel then Merge',
    description: 'Multiple lanes run in parallel, then merge',
    dependencies: [
      { fromRole: 'parallel1', toRole: 'merge', type: 'blocks', cascadeConfig: { propagateStatus: false, propagateBlockers: true, autoProgress: false } },
      { fromRole: 'parallel2', toRole: 'merge', type: 'blocks', cascadeConfig: { propagateStatus: false, propagateBlockers: true, autoProgress: false } },
    ],
  },
  {
    id: 'parent-child',
    name: 'Parent-Child',
    description: 'Child lanes contribute to parent completion',
    dependencies: [
      { fromRole: 'child1', toRole: 'parent', type: 'child_of', cascadeConfig: { propagateStatus: true, propagateBlockers: false, autoProgress: true } },
      { fromRole: 'child2', toRole: 'parent', type: 'child_of', cascadeConfig: { propagateStatus: true, propagateBlockers: false, autoProgress: true } },
    ],
  },
];

// ============================================================================
// Helpers
// ============================================================================

function getTypeColor(type: Dependency['type']): string {
  switch (type) {
    case 'blocks': return 'text-red-400 bg-red-500/20 border-red-500/30';
    case 'depends_on': return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
    case 'follows': return 'text-cyan-400 bg-cyan-500/20 border-cyan-500/30';
    case 'related': return 'text-purple-400 bg-purple-500/20 border-purple-500/30';
    case 'child_of': return 'text-amber-400 bg-amber-500/20 border-amber-500/30';
    default: return 'text-muted-foreground bg-muted/20 border-border/30';
  }
}

function findCriticalPath(lanes: Lane[], dependencies: Dependency[]): CriticalPath {
  // Simple critical path: find longest chain of blocking dependencies
  const graph = new Map<string, string[]>();

  for (const dep of dependencies) {
    if (dep.type === 'blocks' || dep.type === 'depends_on') {
      const targets = graph.get(dep.sourceId) || [];
      targets.push(dep.targetId);
      graph.set(dep.sourceId, targets);
    }
  }

  // Find longest path using DFS
  let longestPath: string[] = [];
  const visited = new Set<string>();

  function dfs(laneId: string, path: string[]): void {
    if (visited.has(laneId)) return;
    visited.add(laneId);

    const newPath = [...path, laneId];
    if (newPath.length > longestPath.length) {
      longestPath = newPath;
    }

    const targets = graph.get(laneId) || [];
    for (const target of targets) {
      dfs(target, newPath);
    }

    visited.delete(laneId);
  }

  for (const lane of lanes) {
    dfs(lane.id, []);
  }

  // Find bottlenecks (lanes with most dependencies)
  const depCounts = new Map<string, number>();
  for (const dep of dependencies) {
    depCounts.set(dep.targetId, (depCounts.get(dep.targetId) || 0) + 1);
  }

  const bottlenecks = Array.from(depCounts.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3)
    .map(([id]) => id);

  return {
    lanes: longestPath,
    totalDuration: longestPath.length,
    bottlenecks,
  };
}

// ============================================================================
// Component
// ============================================================================

export const LaneDependencyManager = component$<LaneDependencyManagerProps>(({
  lanes,
  dependencies,
  templates = DEFAULT_TEMPLATES,
  onAddDependency$,
  onUpdateDependency$,
  onRemoveDependency$,
  onApplyTemplate$,
}) => {
  // State
  const activeTab = useSignal<'graph' | 'list' | 'templates' | 'analysis'>('list');
  const showAddModal = useSignal(false);
  const showTemplateModal = useSignal(false);
  const selectedTemplate = useSignal<string | null>(null);
  const roleMapping = useSignal<Record<string, string>>({});

  const newDep = useSignal<Omit<Dependency, 'id' | 'createdAt'>>({
    sourceId: '',
    targetId: '',
    type: 'depends_on',
    cascadeConfig: {
      propagateStatus: false,
      propagateBlockers: true,
      autoProgress: false,
    },
  });

  // Computed
  const criticalPath = useComputed$(() => findCriticalPath(lanes, dependencies));

  const dependencyStats = useComputed$(() => ({
    total: dependencies.length,
    blocking: dependencies.filter(d => d.type === 'blocks').length,
    dependsOn: dependencies.filter(d => d.type === 'depends_on').length,
    follows: dependencies.filter(d => d.type === 'follows').length,
    childOf: dependencies.filter(d => d.type === 'child_of').length,
  }));

  // Actions
  const addDependency = $(async () => {
    if (!newDep.value.sourceId || !newDep.value.targetId) return;
    if (newDep.value.sourceId === newDep.value.targetId) return;
    if (onAddDependency$) {
      await onAddDependency$(newDep.value);
    }
    showAddModal.value = false;
    newDep.value = {
      sourceId: '',
      targetId: '',
      type: 'depends_on',
      cascadeConfig: { propagateStatus: false, propagateBlockers: true, autoProgress: false },
    };
  });

  const removeDependency = $(async (id: string) => {
    if (onRemoveDependency$) {
      await onRemoveDependency$(id);
    }
  });

  const applyTemplate = $(async () => {
    if (!selectedTemplate.value || !onApplyTemplate$) return;
    await onApplyTemplate$(selectedTemplate.value, roleMapping.value);
    showTemplateModal.value = false;
    selectedTemplate.value = null;
    roleMapping.value = {};
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">DEPENDENCY MANAGER</span>
          <span class="text-[9px] px-2 py-0.5 rounded bg-muted/20 text-muted-foreground">
            {dependencies.length} dependencies
          </span>
        </div>
        <div class="flex items-center gap-1">
          <button
            onClick$={() => { showTemplateModal.value = true; }}
            class="text-[9px] px-2 py-1 rounded bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
          >
            Templates
          </button>
          <button
            onClick$={() => { showAddModal.value = true; }}
            class="text-[9px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
          >
            + Add
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div class="flex border-b border-border/50">
        {(['list', 'graph', 'templates', 'analysis'] as const).map(tab => (
          <button
            key={tab}
            onClick$={() => { activeTab.value = tab; }}
            class={`flex-1 p-2 text-[10px] transition-colors ${
              activeTab.value === tab
                ? 'bg-primary/10 text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:bg-muted/10'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      <div class="max-h-[400px] overflow-y-auto">
        {/* List Tab */}
        {activeTab.value === 'list' && (
          <div class="p-2 space-y-2">
            {dependencies.length > 0 ? (
              dependencies.map(dep => {
                const source = lanes.find(l => l.id === dep.sourceId);
                const target = lanes.find(l => l.id === dep.targetId);
                return (
                  <div
                    key={dep.id}
                    class="p-2 rounded bg-muted/10 border border-border/30"
                  >
                    <div class="flex items-center justify-between mb-1">
                      <div class="flex items-center gap-2 text-[10px]">
                        <span class="text-foreground font-medium">{source?.name || dep.sourceId}</span>
                        <span class={`px-1.5 py-0.5 rounded border ${getTypeColor(dep.type)}`}>
                          {dep.type.replace('_', ' ')}
                        </span>
                        <span class="text-foreground font-medium">{target?.name || dep.targetId}</span>
                      </div>
                      <button
                        onClick$={() => removeDependency(dep.id)}
                        class="text-red-400 hover:text-red-300 text-xs"
                      >
                        \u2715
                      </button>
                    </div>
                    {dep.cascadeConfig && (
                      <div class="flex items-center gap-2 text-[8px] text-muted-foreground mt-1">
                        {dep.cascadeConfig.propagateStatus && <span class="px-1 py-0.5 rounded bg-muted/20">Status</span>}
                        {dep.cascadeConfig.propagateBlockers && <span class="px-1 py-0.5 rounded bg-muted/20">Blockers</span>}
                        {dep.cascadeConfig.autoProgress && <span class="px-1 py-0.5 rounded bg-muted/20">Auto</span>}
                      </div>
                    )}
                  </div>
                );
              })
            ) : (
              <div class="text-[10px] text-muted-foreground text-center py-4">
                No dependencies defined
              </div>
            )}
          </div>
        )}

        {/* Graph Tab (simplified) */}
        {activeTab.value === 'graph' && (
          <div class="p-4">
            <div class="text-[10px] text-muted-foreground text-center mb-4">
              Dependency Graph (simplified view)
            </div>
            <div class="grid grid-cols-3 gap-2">
              {lanes.slice(0, 9).map(lane => {
                const inCount = dependencies.filter(d => d.targetId === lane.id).length;
                const outCount = dependencies.filter(d => d.sourceId === lane.id).length;
                return (
                  <div
                    key={lane.id}
                    class="p-2 rounded border border-border/30 bg-muted/5 text-center"
                  >
                    <div class="text-[9px] font-medium text-foreground truncate">{lane.name}</div>
                    <div class="text-[8px] text-muted-foreground mt-1">
                      <span class="text-blue-400">{inCount}\u2193</span>
                      {' '}
                      <span class="text-purple-400">{outCount}\u2191</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Templates Tab */}
        {activeTab.value === 'templates' && (
          <div class="p-2 space-y-2">
            {templates.map(template => (
              <div
                key={template.id}
                class="p-2 rounded bg-muted/10 border border-border/30"
              >
                <div class="flex items-center justify-between">
                  <div>
                    <div class="text-xs font-medium text-foreground">{template.name}</div>
                    <div class="text-[9px] text-muted-foreground">{template.description}</div>
                  </div>
                  <button
                    onClick$={() => {
                      selectedTemplate.value = template.id;
                      showTemplateModal.value = true;
                    }}
                    class="text-[9px] px-2 py-1 rounded bg-purple-500/20 text-purple-400"
                  >
                    Apply
                  </button>
                </div>
                <div class="mt-2 text-[8px] text-muted-foreground">
                  {template.dependencies.length} dependency rules
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Analysis Tab */}
        {activeTab.value === 'analysis' && (
          <div class="p-3 space-y-4">
            {/* Stats */}
            <div class="grid grid-cols-5 gap-2">
              <div class="text-center p-2 rounded bg-muted/10">
                <div class="text-sm font-bold text-foreground">{dependencyStats.value.total}</div>
                <div class="text-[8px] text-muted-foreground">Total</div>
              </div>
              <div class="text-center p-2 rounded bg-red-500/10">
                <div class="text-sm font-bold text-red-400">{dependencyStats.value.blocking}</div>
                <div class="text-[8px] text-muted-foreground">Blocking</div>
              </div>
              <div class="text-center p-2 rounded bg-blue-500/10">
                <div class="text-sm font-bold text-blue-400">{dependencyStats.value.dependsOn}</div>
                <div class="text-[8px] text-muted-foreground">Depends</div>
              </div>
              <div class="text-center p-2 rounded bg-cyan-500/10">
                <div class="text-sm font-bold text-cyan-400">{dependencyStats.value.follows}</div>
                <div class="text-[8px] text-muted-foreground">Follows</div>
              </div>
              <div class="text-center p-2 rounded bg-amber-500/10">
                <div class="text-sm font-bold text-amber-400">{dependencyStats.value.childOf}</div>
                <div class="text-[8px] text-muted-foreground">Child</div>
              </div>
            </div>

            {/* Critical Path */}
            <div class="p-2 rounded bg-muted/10 border border-border/30">
              <div class="text-[10px] font-semibold text-muted-foreground mb-2">Critical Path</div>
              {criticalPath.value.lanes.length > 0 ? (
                <div class="flex items-center gap-1 flex-wrap">
                  {criticalPath.value.lanes.map((id, i) => {
                    const lane = lanes.find(l => l.id === id);
                    return (
                      <span key={id} class="flex items-center gap-1">
                        <span class="text-[9px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                          {lane?.name || id}
                        </span>
                        {i < criticalPath.value.lanes.length - 1 && (
                          <span class="text-muted-foreground">\u2192</span>
                        )}
                      </span>
                    );
                  })}
                </div>
              ) : (
                <div class="text-[9px] text-muted-foreground">No critical path detected</div>
              )}
            </div>

            {/* Bottlenecks */}
            {criticalPath.value.bottlenecks.length > 0 && (
              <div class="p-2 rounded bg-amber-500/10 border border-amber-500/30">
                <div class="text-[10px] font-semibold text-amber-400 mb-2">Bottlenecks</div>
                <div class="flex items-center gap-1 flex-wrap">
                  {criticalPath.value.bottlenecks.map(id => {
                    const lane = lanes.find(l => l.id === id);
                    return (
                      <span key={id} class="text-[9px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300">
                        {lane?.name || id}
                      </span>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Add Dependency Modal */}
      {showAddModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-80">
            <div class="text-xs font-semibold text-foreground mb-4">Add Dependency</div>
            <div class="space-y-3">
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Source Lane</label>
                <select
                  value={newDep.value.sourceId}
                  onChange$={(e) => { newDep.value = { ...newDep.value, sourceId: (e.target as HTMLSelectElement).value }; }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="">Select...</option>
                  {lanes.map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
                </select>
              </div>
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Type</label>
                <select
                  value={newDep.value.type}
                  onChange$={(e) => { newDep.value = { ...newDep.value, type: (e.target as HTMLSelectElement).value as Dependency['type'] }; }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="depends_on">Depends On</option>
                  <option value="blocks">Blocks</option>
                  <option value="follows">Follows</option>
                  <option value="related">Related To</option>
                  <option value="child_of">Child Of</option>
                </select>
              </div>
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Target Lane</label>
                <select
                  value={newDep.value.targetId}
                  onChange$={(e) => { newDep.value = { ...newDep.value, targetId: (e.target as HTMLSelectElement).value }; }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="">Select...</option>
                  {lanes.filter(l => l.id !== newDep.value.sourceId).map(l => <option key={l.id} value={l.id}>{l.name}</option>)}
                </select>
              </div>
              <div class="text-[9px] text-muted-foreground">Cascade Options</div>
              <div class="space-y-1">
                <label class="flex items-center gap-2 text-[10px]">
                  <input
                    type="checkbox"
                    checked={newDep.value.cascadeConfig?.propagateBlockers}
                    onChange$={(e) => {
                      newDep.value = {
                        ...newDep.value,
                        cascadeConfig: { ...newDep.value.cascadeConfig!, propagateBlockers: (e.target as HTMLInputElement).checked },
                      };
                    }}
                  />
                  Propagate blockers
                </label>
                <label class="flex items-center gap-2 text-[10px]">
                  <input
                    type="checkbox"
                    checked={newDep.value.cascadeConfig?.propagateStatus}
                    onChange$={(e) => {
                      newDep.value = {
                        ...newDep.value,
                        cascadeConfig: { ...newDep.value.cascadeConfig!, propagateStatus: (e.target as HTMLInputElement).checked },
                      };
                    }}
                  />
                  Propagate status
                </label>
              </div>
            </div>
            <div class="flex items-center gap-2 mt-4">
              <button
                onClick$={() => { showAddModal.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={addDependency}
                disabled={!newDep.value.sourceId || !newDep.value.targetId}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
              >
                Add
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default LaneDependencyManager;
