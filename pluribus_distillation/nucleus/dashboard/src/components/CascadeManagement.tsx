/**
 * CascadeManagement - Task cascade orchestration UI
 *
 * Phase 5, Iteration 38 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Visual cascade flow editor
 * - Step-by-step execution tracking
 * - Pause/resume/cancel controls
 * - Error handling and recovery
 * - Cascade templates
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - CascadeManagement
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/progress/circular-progress.js';

// ============================================================================
// Types
// ============================================================================

export type CascadeStatus = 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'cancelled';
export type StepStatus = 'pending' | 'running' | 'completed' | 'failed' | 'skipped';

export interface CascadeStep {
  id: string;
  name: string;
  description?: string;
  type: 'task' | 'gate' | 'parallel' | 'condition';
  status: StepStatus;
  assignedAgent?: string;
  laneId?: string;
  startedAt?: string;
  completedAt?: string;
  duration?: number;
  error?: string;
  output?: string;
  retryCount?: number;
  children?: CascadeStep[]; // For parallel/condition types
}

export interface Cascade {
  id: string;
  name: string;
  description?: string;
  status: CascadeStatus;
  steps: CascadeStep[];
  currentStepIndex: number;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  createdBy: string;
  totalDuration?: number;
  progress: number; // 0-100
}

export interface CascadeTemplate {
  id: string;
  name: string;
  description: string;
  steps: Omit<CascadeStep, 'status' | 'startedAt' | 'completedAt' | 'duration' | 'error' | 'output'>[];
}

export interface CascadeManagementProps {
  /** Active cascades */
  cascades: Cascade[];
  /** Available templates */
  templates?: CascadeTemplate[];
  /** Callback when cascade action is triggered */
  onCascadeAction$?: QRL<(cascadeId: string, action: 'start' | 'pause' | 'resume' | 'cancel' | 'retry') => void>;
  /** Callback when new cascade is created */
  onCreateCascade$?: QRL<(templateId: string, name: string) => void>;
  /** Available agents for assignment */
  agents?: { id: string; name: string }[];
}

// ============================================================================
// Default Templates
// ============================================================================

const DEFAULT_TEMPLATES: CascadeTemplate[] = [
  {
    id: 'deploy',
    name: 'Deployment Pipeline',
    description: 'Build, test, and deploy a feature',
    steps: [
      { id: 's1', name: 'Build', type: 'task' },
      { id: 's2', name: 'Run Tests', type: 'task' },
      { id: 's3', name: 'Quality Gate', type: 'gate' },
      { id: 's4', name: 'Deploy to Staging', type: 'task' },
      { id: 's5', name: 'Integration Tests', type: 'task' },
      { id: 's6', name: 'Approval Gate', type: 'gate' },
      { id: 's7', name: 'Deploy to Production', type: 'task' },
    ],
  },
  {
    id: 'review',
    name: 'Code Review Flow',
    description: 'Automated code review process',
    steps: [
      { id: 's1', name: 'Lint Check', type: 'task' },
      { id: 's2', name: 'Security Scan', type: 'task' },
      { id: 's3', name: 'Parallel Reviews', type: 'parallel', children: [
        { id: 's3a', name: 'AI Review', type: 'task' },
        { id: 's3b', name: 'Style Review', type: 'task' },
      ]},
      { id: 's4', name: 'Human Review Gate', type: 'gate' },
      { id: 's5', name: 'Merge', type: 'task' },
    ],
  },
];

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: CascadeStatus | StepStatus): string {
  switch (status) {
    case 'completed': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'running': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    case 'paused': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'failed': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'cancelled': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    case 'skipped': return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    case 'pending': return 'bg-muted/20 text-muted-foreground border-border/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getStatusIcon(status: StepStatus): string {
  switch (status) {
    case 'completed': return '✓';
    case 'running': return '●';
    case 'failed': return '✕';
    case 'skipped': return '−';
    case 'pending': return '○';
    default: return '○';
  }
}

function getTypeIcon(type: string): string {
  switch (type) {
    case 'task': return '▶';
    case 'gate': return '◆';
    case 'parallel': return '⫸';
    case 'condition': return '◇';
    default: return '●';
  }
}

function formatDuration(ms?: number): string {
  if (!ms) return '-';
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${Math.round(ms / 1000)}s`;
  return `${Math.round(ms / 60000)}m`;
}

function formatTime(dateStr?: string): string {
  if (!dateStr) return '-';
  try {
    return new Date(dateStr).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  } catch {
    return dateStr;
  }
}

// ============================================================================
// Component
// ============================================================================

export const CascadeManagement = component$<CascadeManagementProps>(({
  cascades,
  templates = DEFAULT_TEMPLATES,
  onCascadeAction$,
  onCreateCascade$,
  agents = [],
}) => {
  // State
  const selectedCascadeId = useSignal<string | null>(cascades[0]?.id || null);
  const showCreateModal = useSignal(false);
  const selectedTemplateId = useSignal<string | null>(null);
  const newCascadeName = useSignal('');
  const expandedStepId = useSignal<string | null>(null);

  // Computed
  const selectedCascade = useComputed$(() =>
    cascades.find(c => c.id === selectedCascadeId.value)
  );

  const stats = useComputed$(() => ({
    total: cascades.length,
    running: cascades.filter(c => c.status === 'running').length,
    completed: cascades.filter(c => c.status === 'completed').length,
    failed: cascades.filter(c => c.status === 'failed').length,
  }));

  // Actions
  const handleAction = $(async (action: 'start' | 'pause' | 'resume' | 'cancel' | 'retry') => {
    if (!selectedCascadeId.value) return;
    if (onCascadeAction$) {
      await onCascadeAction$(selectedCascadeId.value, action);
    }
  });

  const createCascade = $(async () => {
    if (!selectedTemplateId.value || !newCascadeName.value) return;
    if (onCreateCascade$) {
      await onCreateCascade$(selectedTemplateId.value, newCascadeName.value);
    }
    showCreateModal.value = false;
    selectedTemplateId.value = null;
    newCascadeName.value = '';
  });

  // Render step recursively
  const renderStep = (step: CascadeStep, depth: number = 0) => {
    const isExpanded = expandedStepId.value === step.id;
    const hasChildren = step.children && step.children.length > 0;

    return (
      <div key={step.id} class={depth > 0 ? 'ml-4' : ''}>
        <div
          class={`flex items-center gap-2 p-2 rounded transition-colors ${
            step.status === 'running' ? 'bg-blue-500/10 border border-blue-500/30' :
            isExpanded ? 'bg-muted/10' : 'hover:bg-muted/5'
          }`}
        >
          {/* Expand toggle for parallel/condition */}
          {hasChildren ? (
            <button
              onClick$={() => { expandedStepId.value = isExpanded ? null : step.id; }}
              class="text-[10px] text-muted-foreground"
            >
              {isExpanded ? '▼' : '▶'}
            </button>
          ) : (
            <span class="w-3" />
          )}

          {/* Status indicator */}
          <span class={`text-xs ${
            step.status === 'completed' ? 'text-emerald-400' :
            step.status === 'running' ? 'text-blue-400 animate-pulse' :
            step.status === 'failed' ? 'text-red-400' :
            'text-muted-foreground'
          }`}>
            {getStatusIcon(step.status)}
          </span>

          {/* Type icon */}
          <span class="text-[9px] text-muted-foreground">{getTypeIcon(step.type)}</span>

          {/* Name */}
          <span class="text-[10px] font-medium text-foreground flex-grow">{step.name}</span>

          {/* Assigned agent */}
          {step.assignedAgent && (
            <span class="text-[8px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400">
              @{step.assignedAgent}
            </span>
          )}

          {/* Duration */}
          {step.duration && (
            <span class="text-[8px] text-muted-foreground">
              {formatDuration(step.duration)}
            </span>
          )}

          {/* Retry badge */}
          {step.retryCount && step.retryCount > 0 && (
            <span class="text-[8px] px-1 py-0.5 rounded bg-amber-500/20 text-amber-400">
              retry {step.retryCount}
            </span>
          )}
        </div>

        {/* Error display */}
        {step.error && (
          <div class="ml-6 mt-1 p-2 rounded bg-red-500/10 border border-red-500/30">
            <div class="text-[9px] text-red-400">{step.error}</div>
          </div>
        )}

        {/* Output display */}
        {step.output && isExpanded && (
          <div class="ml-6 mt-1 p-2 rounded bg-muted/10 border border-border/30">
            <div class="text-[9px] text-muted-foreground font-mono whitespace-pre-wrap">
              {step.output}
            </div>
          </div>
        )}

        {/* Children (for parallel/condition) */}
        {hasChildren && isExpanded && (
          <div class="mt-1 border-l-2 border-border/30">
            {step.children!.map(child => renderStep(child, depth + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">CASCADE MANAGEMENT</span>
          <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/20 text-blue-400 border border-blue-500/30">
            {stats.value.running} running
          </span>
        </div>
        <button
          onClick$={() => { showCreateModal.value = true; }}
          class="text-[10px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
        >
          + New Cascade
        </button>
      </div>

      {/* Summary stats */}
      <div class="grid grid-cols-4 gap-2 p-3 border-b border-border/30 bg-muted/5">
        <div class="text-center">
          <div class="text-lg font-bold text-foreground">{stats.value.total}</div>
          <div class="text-[9px] text-muted-foreground">Total</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-blue-400">{stats.value.running}</div>
          <div class="text-[9px] text-muted-foreground">Running</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-emerald-400">{stats.value.completed}</div>
          <div class="text-[9px] text-muted-foreground">Completed</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-red-400">{stats.value.failed}</div>
          <div class="text-[9px] text-muted-foreground">Failed</div>
        </div>
      </div>

      {/* Main content */}
      <div class="grid grid-cols-3 gap-0 min-h-[300px]">
        {/* Cascade list */}
        <div class="border-r border-border/30 max-h-[400px] overflow-y-auto">
          <div class="p-2 space-y-1">
            {cascades.map(cascade => (
              <div
                key={cascade.id}
                onClick$={() => { selectedCascadeId.value = cascade.id; }}
                class={`p-2 rounded cursor-pointer transition-colors ${
                  selectedCascadeId.value === cascade.id
                    ? 'bg-primary/10 border border-primary/30'
                    : 'hover:bg-muted/10'
                }`}
              >
                <div class="flex items-center justify-between">
                  <span class="text-xs font-medium text-foreground">{cascade.name}</span>
                  <span class={`text-[8px] px-1.5 py-0.5 rounded border ${getStatusColor(cascade.status)}`}>
                    {cascade.status}
                  </span>
                </div>

                {/* Progress bar */}
                <div class="mt-2">
                  <div class="h-1.5 rounded-full bg-muted/30 overflow-hidden">
                    <div
                      class={`h-full rounded-full transition-all ${
                        cascade.status === 'completed' ? 'bg-emerald-500' :
                        cascade.status === 'failed' ? 'bg-red-500' :
                        cascade.status === 'running' ? 'bg-blue-500' :
                        'bg-muted'
                      }`}
                      style={{ width: `${cascade.progress}%` }}
                    />
                  </div>
                </div>

                <div class="mt-1 flex items-center justify-between text-[8px] text-muted-foreground">
                  <span>{cascade.steps.length} steps</span>
                  <span>{cascade.progress}%</span>
                </div>
              </div>
            ))}

            {cascades.length === 0 && (
              <div class="text-center py-8 text-[10px] text-muted-foreground">
                No cascades yet
              </div>
            )}
          </div>
        </div>

        {/* Selected cascade details */}
        <div class="col-span-2 max-h-[400px] overflow-y-auto">
          {selectedCascade.value ? (
            <div>
              {/* Cascade header */}
              <div class="p-3 border-b border-border/30 bg-muted/5">
                <div class="flex items-center justify-between">
                  <div>
                    <div class="text-sm font-medium text-foreground">{selectedCascade.value.name}</div>
                    {selectedCascade.value.description && (
                      <div class="text-[9px] text-muted-foreground mt-0.5">
                        {selectedCascade.value.description}
                      </div>
                    )}
                  </div>
                  <span class={`text-[10px] px-2 py-1 rounded border ${getStatusColor(selectedCascade.value.status)}`}>
                    {selectedCascade.value.status.toUpperCase()}
                  </span>
                </div>

                {/* Controls */}
                <div class="flex items-center gap-2 mt-3">
                  {selectedCascade.value.status === 'pending' && (
                    <button
                      onClick$={() => handleAction('start')}
                      class="px-3 py-1 text-[10px] rounded bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                    >
                      ▶ Start
                    </button>
                  )}
                  {selectedCascade.value.status === 'running' && (
                    <button
                      onClick$={() => handleAction('pause')}
                      class="px-3 py-1 text-[10px] rounded bg-amber-500/20 text-amber-400 hover:bg-amber-500/30"
                    >
                      ⏸ Pause
                    </button>
                  )}
                  {selectedCascade.value.status === 'paused' && (
                    <button
                      onClick$={() => handleAction('resume')}
                      class="px-3 py-1 text-[10px] rounded bg-blue-500/20 text-blue-400 hover:bg-blue-500/30"
                    >
                      ▶ Resume
                    </button>
                  )}
                  {(selectedCascade.value.status === 'running' || selectedCascade.value.status === 'paused') && (
                    <button
                      onClick$={() => handleAction('cancel')}
                      class="px-3 py-1 text-[10px] rounded bg-red-500/20 text-red-400 hover:bg-red-500/30"
                    >
                      ✕ Cancel
                    </button>
                  )}
                  {selectedCascade.value.status === 'failed' && (
                    <button
                      onClick$={() => handleAction('retry')}
                      class="px-3 py-1 text-[10px] rounded bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
                    >
                      ↺ Retry
                    </button>
                  )}
                </div>

                {/* Timing info */}
                <div class="grid grid-cols-3 gap-2 mt-3 text-[9px]">
                  <div>
                    <span class="text-muted-foreground">Started:</span>
                    <span class="ml-1 text-foreground">{formatTime(selectedCascade.value.startedAt)}</span>
                  </div>
                  <div>
                    <span class="text-muted-foreground">Duration:</span>
                    <span class="ml-1 text-foreground">{formatDuration(selectedCascade.value.totalDuration)}</span>
                  </div>
                  <div>
                    <span class="text-muted-foreground">By:</span>
                    <span class="ml-1 text-foreground">@{selectedCascade.value.createdBy}</span>
                  </div>
                </div>
              </div>

              {/* Steps */}
              <div class="p-3">
                <div class="text-[9px] font-semibold text-muted-foreground mb-2">
                  STEPS ({selectedCascade.value.steps.length})
                </div>
                <div class="space-y-1">
                  {selectedCascade.value.steps.map(step => renderStep(step))}
                </div>
              </div>
            </div>
          ) : (
            <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground">
              Select a cascade to view details
            </div>
          )}
        </div>
      </div>

      {/* Create Modal */}
      {showCreateModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-96">
            <div class="text-xs font-semibold text-foreground mb-4">Create New Cascade</div>

            <div class="space-y-4">
              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Template</label>
                <div class="space-y-1">
                  {templates.map(template => (
                    <div
                      key={template.id}
                      onClick$={() => { selectedTemplateId.value = template.id; }}
                      class={`p-2 rounded cursor-pointer transition-colors ${
                        selectedTemplateId.value === template.id
                          ? 'bg-primary/10 border border-primary/30'
                          : 'hover:bg-muted/10 border border-transparent'
                      }`}
                    >
                      <div class="text-xs font-medium text-foreground">{template.name}</div>
                      <div class="text-[9px] text-muted-foreground">{template.description}</div>
                      <div class="text-[8px] text-muted-foreground/50 mt-1">
                        {template.steps.length} steps
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <label class="text-[9px] text-muted-foreground block mb-1">Cascade Name</label>
                <input
                  type="text"
                  value={newCascadeName.value}
                  onInput$={(e) => { newCascadeName.value = (e.target as HTMLInputElement).value; }}
                  class="w-full px-2 py-1.5 text-xs rounded bg-card border border-border/50"
                  placeholder="My Deployment"
                />
              </div>
            </div>

            <div class="flex items-center gap-2 mt-4">
              <button
                onClick$={() => { showCreateModal.value = false; }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={createCascade}
                disabled={!selectedTemplateId.value || !newCascadeName.value}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {cascades.length} cascades • {templates.length} templates
      </div>
    </div>
  );
});

export default CascadeManagement;
