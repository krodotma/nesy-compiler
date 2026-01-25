/**
 * LanePipeline - Pipeline Builder and Visualizer
 *
 * Phase 8, Iteration 67 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Visual pipeline builder
 * - Stage transitions
 * - Pipeline execution
 * - Rollback support
 * - Pipeline history
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

import type { Lane } from '../lib/lanes/store';

// ============================================================================
// Types
// ============================================================================

export interface PipelineStage {
  id: string;
  name: string;
  laneIds: string[];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  order: number;
  autoTransition: boolean;
  rollbackOnFailure: boolean;
  completionCriteria?: 'all' | 'any' | 'percentage';
  completionThreshold?: number;
}

export interface Pipeline {
  id: string;
  name: string;
  description?: string;
  stages: PipelineStage[];
  currentStageIndex: number;
  status: 'idle' | 'running' | 'paused' | 'completed' | 'failed';
  startedAt?: string;
  completedAt?: string;
  history: PipelineHistoryEntry[];
}

export interface PipelineHistoryEntry {
  timestamp: string;
  action: 'start' | 'advance' | 'rollback' | 'pause' | 'complete' | 'fail';
  stageIndex: number;
  details?: string;
}

export interface LanePipelineProps {
  /** Available lanes */
  lanes: Lane[];
  /** Active pipeline */
  pipeline?: Pipeline;
  /** Callback when pipeline is created */
  onCreatePipeline$?: QRL<(pipeline: Pipeline) => void>;
  /** Callback when stage transitions */
  onStageTransition$?: QRL<(pipelineId: string, fromStage: number, toStage: number) => void>;
  /** Callback when pipeline is started */
  onStartPipeline$?: QRL<(pipelineId: string) => void>;
  /** Callback when pipeline is paused */
  onPausePipeline$?: QRL<(pipelineId: string) => void>;
  /** Callback when rollback is triggered */
  onRollback$?: QRL<(pipelineId: string, toStage: number) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getStageStatusColor(status: PipelineStage['status']): string {
  switch (status) {
    case 'pending': return 'bg-muted/20 text-muted-foreground border-border/30';
    case 'running': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'completed': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'skipped': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getPipelineProgress(pipeline: Pipeline): number {
  if (pipeline.stages.length === 0) return 0;
  const completed = pipeline.stages.filter(s => s.status === 'completed').length;
  return Math.round((completed / pipeline.stages.length) * 100);
}

// ============================================================================
// Component
// ============================================================================

export const LanePipeline = component$<LanePipelineProps>(({
  lanes,
  pipeline,
  onCreatePipeline$,
  onStageTransition$,
  onStartPipeline$,
  onPausePipeline$,
  onRollback$,
}) => {
  // State
  const showBuilder = useSignal(false);
  const showHistory = useSignal(false);

  // New pipeline builder state
  const newPipeline = useSignal<Partial<Pipeline>>({
    name: '',
    description: '',
    stages: [],
    status: 'idle',
  });

  const newStage = useSignal<Partial<PipelineStage>>({
    name: '',
    laneIds: [],
    autoTransition: true,
    rollbackOnFailure: true,
    completionCriteria: 'all',
  });

  // Computed
  const progress = useComputed$(() => pipeline ? getPipelineProgress(pipeline) : 0);

  const currentStage = useComputed$(() =>
    pipeline?.stages[pipeline.currentStageIndex]
  );

  const canAdvance = useComputed$(() => {
    if (!pipeline || pipeline.status !== 'running') return false;
    const current = pipeline.stages[pipeline.currentStageIndex];
    return current?.status === 'completed';
  });

  const canRollback = useComputed$(() => {
    if (!pipeline || pipeline.status === 'idle') return false;
    return pipeline.currentStageIndex > 0;
  });

  // Actions
  const startPipeline = $(async () => {
    if (!pipeline || !onStartPipeline$) return;
    await onStartPipeline$(pipeline.id);
  });

  const pausePipeline = $(async () => {
    if (!pipeline || !onPausePipeline$) return;
    await onPausePipeline$(pipeline.id);
  });

  const advanceStage = $(async () => {
    if (!pipeline || !canAdvance.value || !onStageTransition$) return;
    await onStageTransition$(
      pipeline.id,
      pipeline.currentStageIndex,
      pipeline.currentStageIndex + 1
    );
  });

  const rollbackStage = $(async () => {
    if (!pipeline || !canRollback.value || !onRollback$) return;
    await onRollback$(pipeline.id, pipeline.currentStageIndex - 1);
  });

  const addStage = $(() => {
    if (!newStage.value.name) return;

    const stage: PipelineStage = {
      id: `stage-${Date.now()}`,
      name: newStage.value.name!,
      laneIds: newStage.value.laneIds || [],
      status: 'pending',
      order: (newPipeline.value.stages?.length || 0) + 1,
      autoTransition: newStage.value.autoTransition ?? true,
      rollbackOnFailure: newStage.value.rollbackOnFailure ?? true,
      completionCriteria: newStage.value.completionCriteria,
    };

    newPipeline.value = {
      ...newPipeline.value,
      stages: [...(newPipeline.value.stages || []), stage],
    };

    newStage.value = {
      name: '',
      laneIds: [],
      autoTransition: true,
      rollbackOnFailure: true,
      completionCriteria: 'all',
    };
  });

  const removeStage = $((index: number) => {
    const stages = [...(newPipeline.value.stages || [])];
    stages.splice(index, 1);
    newPipeline.value = { ...newPipeline.value, stages };
  });

  const createPipeline = $(async () => {
    if (!newPipeline.value.name || !newPipeline.value.stages?.length || !onCreatePipeline$) return;

    const pipeline: Pipeline = {
      id: `pipeline-${Date.now()}`,
      name: newPipeline.value.name,
      description: newPipeline.value.description,
      stages: newPipeline.value.stages,
      currentStageIndex: 0,
      status: 'idle',
      history: [],
    };

    await onCreatePipeline$(pipeline);
    showBuilder.value = false;
    newPipeline.value = { name: '', description: '', stages: [], status: 'idle' };
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">PIPELINE</span>
          {pipeline && (
            <span class={`text-[9px] px-2 py-0.5 rounded ${
              pipeline.status === 'running' ? 'bg-blue-500/20 text-blue-400' :
              pipeline.status === 'completed' ? 'bg-emerald-500/20 text-emerald-400' :
              pipeline.status === 'failed' ? 'bg-red-500/20 text-red-400' :
              'bg-muted/20 text-muted-foreground'
            }`}>
              {pipeline.status}
            </span>
          )}
        </div>
        <div class="flex items-center gap-1">
          {pipeline && (
            <button
              onClick$={() => { showHistory.value = !showHistory.value; }}
              class="text-[9px] px-2 py-1 rounded bg-muted/20 hover:bg-muted/40 text-muted-foreground"
            >
              History
            </button>
          )}
          <button
            onClick$={() => { showBuilder.value = true; }}
            class="text-[9px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
          >
            + New Pipeline
          </button>
        </div>
      </div>

      {/* Active pipeline visualization */}
      {pipeline ? (
        <div class="p-3">
          {/* Progress bar */}
          <div class="mb-4">
            <div class="flex items-center justify-between mb-1">
              <span class="text-xs font-medium text-foreground">{pipeline.name}</span>
              <span class="text-[10px] text-muted-foreground">{progress.value}%</span>
            </div>
            <div class="h-2 rounded-full bg-muted/20 overflow-hidden">
              <div
                class={`h-full transition-all ${
                  pipeline.status === 'failed' ? 'bg-red-500' :
                  pipeline.status === 'completed' ? 'bg-emerald-500' :
                  'bg-primary'
                }`}
                style={{ width: `${progress.value}%` }}
              />
            </div>
          </div>

          {/* Stages */}
          <div class="relative">
            {/* Connector line */}
            <div class="absolute left-4 top-4 bottom-4 w-0.5 bg-border/50" />

            <div class="space-y-3">
              {pipeline.stages.map((stage, index) => {
                const isCurrentStage = index === pipeline.currentStageIndex;
                const stageLanes = lanes.filter(l => stage.laneIds.includes(l.id));

                return (
                  <div key={stage.id} class="relative pl-10">
                    {/* Stage dot */}
                    <div class={`absolute left-2 top-2 w-4 h-4 rounded-full border-2 ${
                      isCurrentStage ? 'border-primary bg-primary/20' :
                      stage.status === 'completed' ? 'border-emerald-500 bg-emerald-500' :
                      stage.status === 'failed' ? 'border-red-500 bg-red-500' :
                      'border-border bg-card'
                    }`} />

                    {/* Stage content */}
                    <div class={`p-2 rounded border ${getStageStatusColor(stage.status)} ${
                      isCurrentStage ? 'ring-1 ring-primary/50' : ''
                    }`}>
                      <div class="flex items-center justify-between mb-1">
                        <span class="text-xs font-medium text-foreground">{stage.name}</span>
                        <span class={`text-[8px] px-1.5 py-0.5 rounded ${getStageStatusColor(stage.status)}`}>
                          {stage.status}
                        </span>
                      </div>

                      {/* Stage lanes */}
                      {stageLanes.length > 0 && (
                        <div class="flex flex-wrap gap-1 mt-1">
                          {stageLanes.map(lane => (
                            <span
                              key={lane.id}
                              class="text-[8px] px-1.5 py-0.5 rounded bg-muted/20 text-muted-foreground"
                            >
                              {lane.name} ({lane.wip_pct}%)
                            </span>
                          ))}
                        </div>
                      )}

                      {/* Stage options */}
                      <div class="flex items-center gap-2 mt-1 text-[8px] text-muted-foreground">
                        {stage.autoTransition && <span>\u21AA Auto</span>}
                        {stage.rollbackOnFailure && <span>\u21A9 Rollback</span>}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Controls */}
          <div class="flex items-center gap-2 mt-4 pt-3 border-t border-border/50">
            {pipeline.status === 'idle' && (
              <button
                onClick$={startPipeline}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground"
              >
                Start Pipeline
              </button>
            )}

            {pipeline.status === 'running' && (
              <>
                <button
                  onClick$={pausePipeline}
                  class="flex-1 px-3 py-1.5 text-xs rounded bg-amber-500/20 text-amber-400"
                >
                  Pause
                </button>
                <button
                  onClick$={advanceStage}
                  disabled={!canAdvance.value}
                  class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
                >
                  Advance \u2192
                </button>
              </>
            )}

            {pipeline.status === 'paused' && (
              <button
                onClick$={startPipeline}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground"
              >
                Resume
              </button>
            )}

            {canRollback.value && (
              <button
                onClick$={rollbackStage}
                class="px-3 py-1.5 text-xs rounded bg-red-500/20 text-red-400"
              >
                \u21A9 Rollback
              </button>
            )}
          </div>

          {/* History */}
          {showHistory.value && pipeline.history.length > 0 && (
            <div class="mt-3 pt-3 border-t border-border/30">
              <div class="text-[9px] font-semibold text-muted-foreground mb-2">HISTORY</div>
              <div class="space-y-1 max-h-[100px] overflow-y-auto">
                {pipeline.history.slice().reverse().map((entry, i) => (
                  <div key={i} class="flex items-center gap-2 text-[9px]">
                    <span class="text-muted-foreground/50">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                    <span class="text-foreground">{entry.action}</span>
                    <span class="text-muted-foreground">Stage {entry.stageIndex + 1}</span>
                    {entry.details && (
                      <span class="text-muted-foreground/50">{entry.details}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      ) : (
        <div class="p-8 text-center text-[10px] text-muted-foreground">
          No active pipeline. Create one to get started.
        </div>
      )}

      {/* Pipeline Builder Modal */}
      {showBuilder.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-[500px] max-h-[80vh] overflow-y-auto">
            <div class="text-xs font-semibold text-foreground mb-4">Create Pipeline</div>

            <div class="space-y-3">
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Pipeline Name</label>
                <input
                  type="text"
                  value={newPipeline.value.name}
                  onInput$={(e) => {
                    newPipeline.value = { ...newPipeline.value, name: (e.target as HTMLInputElement).value };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                />
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Description</label>
                <textarea
                  value={newPipeline.value.description}
                  onInput$={(e) => {
                    newPipeline.value = { ...newPipeline.value, description: (e.target as HTMLTextAreaElement).value };
                  }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50 h-16"
                />
              </div>

              {/* Stages */}
              <div>
                <div class="text-[9px] text-muted-foreground mb-2">
                  Stages ({newPipeline.value.stages?.length || 0})
                </div>

                {newPipeline.value.stages?.map((stage, i) => (
                  <div key={stage.id} class="flex items-center gap-2 mb-1 p-2 rounded bg-muted/10">
                    <span class="text-[10px] text-muted-foreground w-4">{i + 1}.</span>
                    <span class="flex-1 text-xs text-foreground">{stage.name}</span>
                    <span class="text-[9px] text-muted-foreground">{stage.laneIds.length} lanes</span>
                    <button
                      onClick$={() => removeStage(i)}
                      class="text-red-400 hover:text-red-300 text-xs"
                    >
                      \u2715
                    </button>
                  </div>
                ))}

                {/* Add stage form */}
                <div class="p-2 rounded border border-dashed border-border/50 mt-2">
                  <input
                    type="text"
                    value={newStage.value.name}
                    onInput$={(e) => {
                      newStage.value = { ...newStage.value, name: (e.target as HTMLInputElement).value };
                    }}
                    placeholder="Stage name..."
                    class="w-full px-2 py-1 text-xs rounded bg-card border border-border/50 mb-2"
                  />

                  <div class="flex flex-wrap gap-1 mb-2">
                    {lanes.slice(0, 10).map(lane => (
                      <label key={lane.id} class="flex items-center gap-1 text-[9px]">
                        <input
                          type="checkbox"
                          checked={newStage.value.laneIds?.includes(lane.id)}
                          onChange$={(e) => {
                            const ids = new Set(newStage.value.laneIds || []);
                            if ((e.target as HTMLInputElement).checked) {
                              ids.add(lane.id);
                            } else {
                              ids.delete(lane.id);
                            }
                            newStage.value = { ...newStage.value, laneIds: Array.from(ids) };
                          }}
                        />
                        {lane.name}
                      </label>
                    ))}
                  </div>

                  <button
                    onClick$={addStage}
                    disabled={!newStage.value.name}
                    class="w-full px-2 py-1 text-xs rounded bg-primary/20 text-primary disabled:opacity-50"
                  >
                    Add Stage
                  </button>
                </div>
              </div>
            </div>

            <div class="flex items-center gap-2 mt-4">
              <button
                onClick$={() => { showBuilder.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={createPipeline}
                disabled={!newPipeline.value.name || !newPipeline.value.stages?.length}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
              >
                Create Pipeline
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

export default LanePipeline;
