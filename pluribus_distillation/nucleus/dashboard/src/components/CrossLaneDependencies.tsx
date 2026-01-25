/**
 * CrossLaneDependencies - Dependency management between lanes
 *
 * Phase 5, Iteration 37 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Visualize dependencies between lanes
 * - Add/remove dependencies
 * - Impact propagation analysis
 * - Circular dependency detection
 * - Dependency health status
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - CrossLaneDependencies
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';

// ============================================================================
// Types
// ============================================================================

export type DependencyType = 'blocks' | 'depends_on' | 'related' | 'child_of';

export interface Dependency {
  id: string;
  sourceId: string;
  targetId: string;
  type: DependencyType;
  description?: string;
  isBlocking: boolean;
  createdAt: string;
}

export interface DependencyLane {
  id: string;
  name: string;
  owner: string;
  status: 'green' | 'yellow' | 'red';
  wip_pct: number;
}

export interface ImpactAnalysis {
  laneId: string;
  impactType: 'direct' | 'indirect';
  impactLevel: 'high' | 'medium' | 'low';
  path: string[];
}

export interface CrossLaneDependenciesProps {
  /** Available lanes */
  lanes: DependencyLane[];
  /** Existing dependencies */
  dependencies: Dependency[];
  /** Selected lane for focus */
  focusLaneId?: string;
  /** Callback when dependency is added */
  onAddDependency$?: QRL<(dep: Omit<Dependency, 'id' | 'createdAt'>) => void>;
  /** Callback when dependency is removed */
  onRemoveDependency$?: QRL<(depId: string) => void>;
  /** Callback when lane is selected */
  onLaneSelect$?: QRL<(laneId: string) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getTypeColor(type: DependencyType): string {
  switch (type) {
    case 'blocks': return 'text-red-400 bg-red-500/20 border-red-500/30';
    case 'depends_on': return 'text-blue-400 bg-blue-500/20 border-blue-500/30';
    case 'related': return 'text-purple-400 bg-purple-500/20 border-purple-500/30';
    case 'child_of': return 'text-cyan-400 bg-cyan-500/20 border-cyan-500/30';
    default: return 'text-muted-foreground bg-muted/20 border-border/30';
  }
}

function getTypeLabel(type: DependencyType): string {
  switch (type) {
    case 'blocks': return 'Blocks';
    case 'depends_on': return 'Depends On';
    case 'related': return 'Related To';
    case 'child_of': return 'Child Of';
    default: return type;
  }
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'green': return 'bg-emerald-500';
    case 'yellow': return 'bg-amber-500';
    case 'red': return 'bg-red-500';
    default: return 'bg-gray-500';
  }
}

function detectCircularDependencies(
  dependencies: Dependency[],
  laneId: string,
  visited: Set<string> = new Set(),
  path: string[] = []
): string[] | null {
  if (visited.has(laneId)) {
    const cycleStart = path.indexOf(laneId);
    return path.slice(cycleStart).concat(laneId);
  }

  visited.add(laneId);
  path.push(laneId);

  const outgoing = dependencies.filter(d => d.sourceId === laneId);
  for (const dep of outgoing) {
    const cycle = detectCircularDependencies(dependencies, dep.targetId, new Set(visited), [...path]);
    if (cycle) return cycle;
  }

  return null;
}

function analyzeImpact(
  dependencies: Dependency[],
  laneId: string,
  direction: 'upstream' | 'downstream'
): ImpactAnalysis[] {
  const impacts: ImpactAnalysis[] = [];
  const visited = new Set<string>();

  function traverse(currentId: string, path: string[], level: number) {
    if (visited.has(currentId)) return;
    visited.add(currentId);

    const related = direction === 'downstream'
      ? dependencies.filter(d => d.sourceId === currentId)
      : dependencies.filter(d => d.targetId === currentId);

    for (const dep of related) {
      const nextId = direction === 'downstream' ? dep.targetId : dep.sourceId;
      const newPath = [...path, nextId];

      impacts.push({
        laneId: nextId,
        impactType: level === 0 ? 'direct' : 'indirect',
        impactLevel: level === 0 ? 'high' : level === 1 ? 'medium' : 'low',
        path: newPath,
      });

      if (level < 3) {
        traverse(nextId, newPath, level + 1);
      }
    }
  }

  traverse(laneId, [laneId], 0);
  return impacts;
}

// ============================================================================
// Component
// ============================================================================

export const CrossLaneDependencies = component$<CrossLaneDependenciesProps>(({
  lanes,
  dependencies,
  focusLaneId,
  onAddDependency$,
  onRemoveDependency$,
  onLaneSelect$,
}) => {
  // State
  const selectedLaneId = useSignal<string | null>(focusLaneId || null);
  const showAddModal = useSignal(false);
  const showImpactAnalysis = useSignal(false);
  const impactDirection = useSignal<'upstream' | 'downstream'>('downstream');

  const newDep = useSignal<Omit<Dependency, 'id' | 'createdAt'>>({
    sourceId: '',
    targetId: '',
    type: 'depends_on',
    isBlocking: false,
  });

  // Computed
  const selectedLane = useComputed$(() =>
    lanes.find(l => l.id === selectedLaneId.value)
  );

  const incomingDeps = useComputed$(() =>
    dependencies.filter(d => d.targetId === selectedLaneId.value)
  );

  const outgoingDeps = useComputed$(() =>
    dependencies.filter(d => d.sourceId === selectedLaneId.value)
  );

  const circularDeps = useComputed$(() => {
    if (!selectedLaneId.value) return null;
    return detectCircularDependencies(dependencies, selectedLaneId.value);
  });

  const impactAnalysis = useComputed$(() => {
    if (!selectedLaneId.value) return [];
    return analyzeImpact(dependencies, selectedLaneId.value, impactDirection.value);
  });

  const dependencyHealth = useComputed$(() => {
    const blocking = dependencies.filter(d => d.isBlocking);
    const blockedLanes = new Set(blocking.map(d => d.targetId));
    const redLaneDeps = dependencies.filter(d => {
      const source = lanes.find(l => l.id === d.sourceId);
      return source?.status === 'red';
    });

    return {
      total: dependencies.length,
      blocking: blocking.length,
      blockedLanes: blockedLanes.size,
      atRisk: redLaneDeps.length,
      hasCircular: circularDeps.value !== null,
    };
  });

  // Actions
  const selectLane = $(async (laneId: string) => {
    selectedLaneId.value = laneId;
    if (onLaneSelect$) {
      await onLaneSelect$(laneId);
    }
  });

  const addDependency = $(async () => {
    if (!newDep.value.sourceId || !newDep.value.targetId) return;
    if (newDep.value.sourceId === newDep.value.targetId) return;

    if (onAddDependency$) {
      await onAddDependency$(newDep.value);
    }

    showAddModal.value = false;
    newDep.value = {
      sourceId: selectedLaneId.value || '',
      targetId: '',
      type: 'depends_on',
      isBlocking: false,
    };
  });

  const removeDependency = $(async (depId: string) => {
    if (onRemoveDependency$) {
      await onRemoveDependency$(depId);
    }
  });

  // SVG dimensions for mini graph
  const graphWidth = 300;
  const graphHeight = 200;

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">CROSS-LANE DEPENDENCIES</span>
          {circularDeps.value && (
            <span class="text-[9px] px-2 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 animate-pulse">
              Circular Dependency Detected
            </span>
          )}
        </div>
        <button
          onClick$={() => {
            newDep.value.sourceId = selectedLaneId.value || '';
            showAddModal.value = true;
          }}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
        >
          + Add Dependency
        </button>
      </div>

      {/* Health summary */}
      <div class="grid grid-cols-5 gap-2 p-3 border-b border-border/30 bg-muted/5">
        <div class="text-center">
          <div class="text-lg font-bold text-foreground">{dependencyHealth.value.total}</div>
          <div class="text-[9px] text-muted-foreground">Total</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-red-400">{dependencyHealth.value.blocking}</div>
          <div class="text-[9px] text-muted-foreground">Blocking</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-amber-400">{dependencyHealth.value.blockedLanes}</div>
          <div class="text-[9px] text-muted-foreground">Blocked Lanes</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-orange-400">{dependencyHealth.value.atRisk}</div>
          <div class="text-[9px] text-muted-foreground">At Risk</div>
        </div>
        <div class="text-center">
          <div class={`text-lg font-bold ${dependencyHealth.value.hasCircular ? 'text-red-400' : 'text-emerald-400'}`}>
            {dependencyHealth.value.hasCircular ? '!' : '✓'}
          </div>
          <div class="text-[9px] text-muted-foreground">Circular</div>
        </div>
      </div>

      {/* Main content */}
      <div class="grid grid-cols-2 gap-0">
        {/* Lane selector */}
        <div class="border-r border-border/30 max-h-[300px] overflow-y-auto">
          <div class="p-2 text-[9px] font-semibold text-muted-foreground border-b border-border/30">
            SELECT LANE
          </div>
          <div class="p-2 space-y-1">
            {lanes.map(lane => {
              const inCount = dependencies.filter(d => d.targetId === lane.id).length;
              const outCount = dependencies.filter(d => d.sourceId === lane.id).length;
              const isSelected = selectedLaneId.value === lane.id;

              return (
                <div
                  key={lane.id}
                  onClick$={() => selectLane(lane.id)}
                  class={`p-2 rounded cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-primary/10 border border-primary/30'
                      : 'hover:bg-muted/10'
                  }`}
                >
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                      <div class={`w-2 h-2 rounded-full ${getStatusColor(lane.status)}`} />
                      <span class="text-xs font-medium text-foreground">{lane.name}</span>
                    </div>
                    <div class="flex items-center gap-1 text-[8px]">
                      <span class="text-blue-400">{inCount}↓</span>
                      <span class="text-purple-400">{outCount}↑</span>
                    </div>
                  </div>
                  <div class="mt-1 text-[9px] text-muted-foreground">
                    @{lane.owner} • {lane.wip_pct}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Dependency details */}
        <div class="max-h-[300px] overflow-y-auto">
          {selectedLane.value ? (
            <div>
              {/* Selected lane header */}
              <div class="p-3 border-b border-border/30 bg-muted/5">
                <div class="flex items-center gap-2">
                  <div class={`w-3 h-3 rounded-full ${getStatusColor(selectedLane.value.status)}`} />
                  <span class="text-sm font-medium text-foreground">{selectedLane.value.name}</span>
                </div>
                <div class="mt-1 text-[9px] text-muted-foreground">
                  {incomingDeps.value.length} incoming • {outgoingDeps.value.length} outgoing
                </div>
              </div>

              {/* Circular dependency warning */}
              {circularDeps.value && (
                <div class="p-2 bg-red-500/10 border-b border-red-500/30">
                  <div class="text-[10px] font-medium text-red-400 mb-1">Circular Dependency Chain:</div>
                  <div class="text-[9px] text-red-300">
                    {circularDeps.value.map((id, i) => {
                      const lane = lanes.find(l => l.id === id);
                      return (
                        <span key={i}>
                          {lane?.name || id}
                          {i < circularDeps.value!.length - 1 && ' → '}
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Incoming dependencies */}
              <div class="p-2 border-b border-border/30">
                <div class="text-[9px] font-semibold text-muted-foreground mb-2">
                  INCOMING ({incomingDeps.value.length})
                </div>
                {incomingDeps.value.length > 0 ? (
                  <div class="space-y-1">
                    {incomingDeps.value.map(dep => {
                      const source = lanes.find(l => l.id === dep.sourceId);
                      return (
                        <div
                          key={dep.id}
                          class="flex items-center justify-between p-1.5 rounded bg-muted/10"
                        >
                          <div class="flex items-center gap-2">
                            <div class={`w-2 h-2 rounded-full ${getStatusColor(source?.status || 'gray')}`} />
                            <span class="text-[10px] text-foreground">{source?.name || dep.sourceId}</span>
                            <span class={`text-[8px] px-1 py-0.5 rounded border ${getTypeColor(dep.type)}`}>
                              {getTypeLabel(dep.type)}
                            </span>
                            {dep.isBlocking && (
                              <span class="text-[8px] px-1 py-0.5 rounded bg-red-500/20 text-red-400">
                                Blocking
                              </span>
                            )}
                          </div>
                          <button
                            onClick$={() => removeDependency(dep.id)}
                            class="text-red-400 hover:text-red-300 text-xs"
                          >
                            ✕
                          </button>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div class="text-[9px] text-muted-foreground text-center py-2">
                    No incoming dependencies
                  </div>
                )}
              </div>

              {/* Outgoing dependencies */}
              <div class="p-2 border-b border-border/30">
                <div class="text-[9px] font-semibold text-muted-foreground mb-2">
                  OUTGOING ({outgoingDeps.value.length})
                </div>
                {outgoingDeps.value.length > 0 ? (
                  <div class="space-y-1">
                    {outgoingDeps.value.map(dep => {
                      const target = lanes.find(l => l.id === dep.targetId);
                      return (
                        <div
                          key={dep.id}
                          class="flex items-center justify-between p-1.5 rounded bg-muted/10"
                        >
                          <div class="flex items-center gap-2">
                            <div class={`w-2 h-2 rounded-full ${getStatusColor(target?.status || 'gray')}`} />
                            <span class="text-[10px] text-foreground">{target?.name || dep.targetId}</span>
                            <span class={`text-[8px] px-1 py-0.5 rounded border ${getTypeColor(dep.type)}`}>
                              {getTypeLabel(dep.type)}
                            </span>
                            {dep.isBlocking && (
                              <span class="text-[8px] px-1 py-0.5 rounded bg-red-500/20 text-red-400">
                                Blocking
                              </span>
                            )}
                          </div>
                          <button
                            onClick$={() => removeDependency(dep.id)}
                            class="text-red-400 hover:text-red-300 text-xs"
                          >
                            ✕
                          </button>
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <div class="text-[9px] text-muted-foreground text-center py-2">
                    No outgoing dependencies
                  </div>
                )}
              </div>

              {/* Impact analysis toggle */}
              <div class="p-2">
                <button
                  onClick$={() => { showImpactAnalysis.value = !showImpactAnalysis.value; }}
                  class="w-full p-2 text-[9px] text-muted-foreground hover:bg-muted/10 rounded transition-colors flex items-center justify-center gap-1"
                >
                  <span>{showImpactAnalysis.value ? '▼' : '▶'}</span>
                  <span>Impact Analysis</span>
                </button>

                {showImpactAnalysis.value && (
                  <div class="mt-2 p-2 rounded bg-muted/5 border border-border/30">
                    <div class="flex items-center gap-2 mb-2">
                      <button
                        onClick$={() => { impactDirection.value = 'downstream'; }}
                        class={`text-[9px] px-2 py-1 rounded ${
                          impactDirection.value === 'downstream'
                            ? 'bg-primary/20 text-primary'
                            : 'bg-muted/20 text-muted-foreground'
                        }`}
                      >
                        Downstream ↓
                      </button>
                      <button
                        onClick$={() => { impactDirection.value = 'upstream'; }}
                        class={`text-[9px] px-2 py-1 rounded ${
                          impactDirection.value === 'upstream'
                            ? 'bg-primary/20 text-primary'
                            : 'bg-muted/20 text-muted-foreground'
                        }`}
                      >
                        Upstream ↑
                      </button>
                    </div>

                    {impactAnalysis.value.length > 0 ? (
                      <div class="space-y-1">
                        {impactAnalysis.value.map((impact, i) => {
                          const lane = lanes.find(l => l.id === impact.laneId);
                          return (
                            <div key={i} class="flex items-center gap-2 text-[9px]">
                              <span class={`px-1 py-0.5 rounded ${
                                impact.impactLevel === 'high' ? 'bg-red-500/20 text-red-400' :
                                impact.impactLevel === 'medium' ? 'bg-amber-500/20 text-amber-400' :
                                'bg-blue-500/20 text-blue-400'
                              }`}>
                                {impact.impactType}
                              </span>
                              <span class="text-foreground">{lane?.name || impact.laneId}</span>
                              <span class="text-muted-foreground/50 text-[8px]">
                                ({impact.path.length - 1} hops)
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <div class="text-[9px] text-muted-foreground text-center">
                        No {impactDirection.value} impact
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground p-8">
              Select a lane to view dependencies
            </div>
          )}
        </div>
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
                  onChange$={(e) => {
                    newDep.value = { ...newDep.value, sourceId: (e.target as HTMLSelectElement).value };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="">Select source...</option>
                  {lanes.map(l => (
                    <option key={l.id} value={l.id}>{l.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Type</label>
                <select
                  value={newDep.value.type}
                  onChange$={(e) => {
                    newDep.value = { ...newDep.value, type: (e.target as HTMLSelectElement).value as DependencyType };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="depends_on">Depends On</option>
                  <option value="blocks">Blocks</option>
                  <option value="related">Related To</option>
                  <option value="child_of">Child Of</option>
                </select>
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Target Lane</label>
                <select
                  value={newDep.value.targetId}
                  onChange$={(e) => {
                    newDep.value = { ...newDep.value, targetId: (e.target as HTMLSelectElement).value };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                >
                  <option value="">Select target...</option>
                  {lanes.filter(l => l.id !== newDep.value.sourceId).map(l => (
                    <option key={l.id} value={l.id}>{l.name}</option>
                  ))}
                </select>
              </div>

              <label class="flex items-center gap-2 text-[10px]">
                <input
                  type="checkbox"
                  checked={newDep.value.isBlocking}
                  onChange$={(e) => {
                    newDep.value = { ...newDep.value, isBlocking: (e.target as HTMLInputElement).checked };
                  }}
                />
                Blocking dependency
              </label>
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

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {lanes.length} lanes • {dependencies.length} dependencies
      </div>
    </div>
  );
});

export default CrossLaneDependencies;
