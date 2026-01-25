/**
 * LaneCICDIntegration - CI/CD Integration for Lanes
 *
 * Phase 8, Iteration 70 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - CI/CD status display
 * - Build metrics
 * - Pipeline visualization
 * - Deployment tracking
 * - Build history
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - LaneCICDIntegration
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-tonal-button.js';
import '@material/web/progress/circular-progress.js';
import '@material/web/progress/linear-progress.js';

// ============================================================================
// Types
// ============================================================================

export interface BuildJob {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'success' | 'failure' | 'cancelled' | 'skipped';
  startedAt?: string;
  finishedAt?: string;
  duration?: number;
  logs?: string;
  artifacts?: string[];
}

export interface PipelineRun {
  id: string;
  pipelineId: string;
  pipelineName: string;
  status: 'pending' | 'running' | 'success' | 'failure' | 'cancelled';
  trigger: 'push' | 'pr' | 'schedule' | 'manual' | 'api';
  branch: string;
  commit: {
    sha: string;
    message: string;
    author: string;
  };
  jobs: BuildJob[];
  startedAt: string;
  finishedAt?: string;
  duration?: number;
}

export interface DeploymentRecord {
  id: string;
  environment: 'development' | 'staging' | 'production';
  status: 'pending' | 'in_progress' | 'success' | 'failure' | 'rolled_back';
  version: string;
  commit: string;
  deployedAt: string;
  deployedBy: string;
  duration?: number;
  rollbackAvailable: boolean;
}

export interface BuildMetrics {
  totalBuilds: number;
  successRate: number;
  avgDuration: number;
  failureRate: number;
  trend: 'up' | 'down' | 'stable';
}

export interface LaneCICDLink {
  laneId: string;
  pipelineId?: string;
  pipelineName?: string;
  lastRun?: PipelineRun;
  deployments?: DeploymentRecord[];
  metrics?: BuildMetrics;
}

export interface LaneCICDIntegrationProps {
  /** Lane ID */
  laneId?: string;
  /** Lane name */
  laneName?: string;
  /** CI/CD links */
  cicdLinks?: LaneCICDLink[];
  /** Available pipelines */
  pipelines?: Array<{ id: string; name: string; provider: string }>;
  /** Recent pipeline runs */
  recentRuns?: PipelineRun[];
  /** Callback when pipeline is linked */
  onLinkPipeline$?: QRL<(laneId: string, pipelineId: string) => void>;
  /** Callback when link is removed */
  onUnlink$?: QRL<(laneId: string) => void>;
  /** Callback to trigger build */
  onTriggerBuild$?: QRL<(pipelineId: string, branch?: string) => void>;
  /** Callback to rollback deployment */
  onRollback$?: QRL<(deploymentId: string) => void>;
  /** Refresh callback */
  onRefresh$?: QRL<() => void>;
  /** Compact mode */
  compact?: boolean;
}

// ============================================================================
// Helper Functions
// ============================================================================

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const seconds = Math.floor(ms / 1000);
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes < 60) return `${minutes}m ${remainingSeconds}s`;
  const hours = Math.floor(minutes / 60);
  const remainingMinutes = minutes % 60;
  return `${hours}h ${remainingMinutes}m`;
}

function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffHours < 1) return 'just now';
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;
  return date.toLocaleDateString();
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'success': return 'text-emerald-400';
    case 'failure': return 'text-red-400';
    case 'running': return 'text-blue-400';
    case 'pending': return 'text-yellow-400';
    case 'cancelled': return 'text-muted-foreground';
    case 'skipped': return 'text-muted-foreground/50';
    case 'in_progress': return 'text-blue-400';
    case 'rolled_back': return 'text-orange-400';
    default: return 'text-muted-foreground';
  }
}

function getStatusIcon(status: string): string {
  switch (status) {
    case 'success': return '✓';
    case 'failure': return '✗';
    case 'running': return '◐';
    case 'pending': return '○';
    case 'cancelled': return '⊘';
    case 'skipped': return '⊝';
    case 'in_progress': return '◐';
    case 'rolled_back': return '↺';
    default: return '?';
  }
}

function getEnvironmentColor(env: string): string {
  switch (env) {
    case 'production': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'staging': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    case 'development': return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
    default: return 'bg-muted/20 text-muted-foreground border-border/30';
  }
}

function getTriggerIcon(trigger: string): string {
  switch (trigger) {
    case 'push': return '⬆';
    case 'pr': return '⊕';
    case 'schedule': return '⏰';
    case 'manual': return '▶';
    case 'api': return '⚡';
    default: return '○';
  }
}

// ============================================================================
// Component
// ============================================================================

export const LaneCICDIntegration = component$<LaneCICDIntegrationProps>(({
  laneId,
  laneName,
  cicdLinks = [],
  pipelines = [],
  recentRuns = [],
  onLinkPipeline$,
  onUnlink$,
  onTriggerBuild$,
  onRollback$,
  onRefresh$,
  compact = false,
}) => {
  // State
  const showLinkModal = useSignal(false);
  const showDetailsModal = useSignal(false);
  const selectedPipelineId = useSignal<string | null>(null);
  const selectedRunId = useSignal<string | null>(null);
  const activeTab = useSignal<'builds' | 'deployments' | 'metrics'>('builds');
  const isRefreshing = useSignal(false);

  // Get current lane's CI/CD link
  const currentLink = cicdLinks.find(l => l.laneId === laneId);

  // Get runs for current lane's pipeline
  const pipelineRuns = useComputed$(() => {
    if (!currentLink?.pipelineId) return [];
    return recentRuns.filter(r => r.pipelineId === currentLink.pipelineId);
  });

  // Auto-refresh for running builds
  useVisibleTask$(({ cleanup }) => {
    const interval = setInterval(async () => {
      const hasRunning = pipelineRuns.value.some(r => r.status === 'running');
      if (hasRunning && onRefresh$) {
        isRefreshing.value = true;
        await onRefresh$();
        isRefreshing.value = false;
      }
    }, 5000); // Check every 5 seconds for running builds

    cleanup(() => clearInterval(interval));
  });

  // Handlers
  const handleLinkPipeline = $(async () => {
    if (!laneId || !selectedPipelineId.value || !onLinkPipeline$) return;
    await onLinkPipeline$(laneId, selectedPipelineId.value);
    showLinkModal.value = false;
    selectedPipelineId.value = null;
  });

  const handleTriggerBuild = $(async () => {
    if (!currentLink?.pipelineId || !onTriggerBuild$) return;
    await onTriggerBuild$(currentLink.pipelineId);
  });

  // Compact view
  if (compact) {
    const lastRun = currentLink?.lastRun;
    return (
      <div class="flex items-center gap-2 text-[10px]">
        {lastRun ? (
          <>
            <span class={`flex items-center gap-1 ${getStatusColor(lastRun.status)}`}>
              <span>{getStatusIcon(lastRun.status)}</span>
              <span>{lastRun.pipelineName}</span>
            </span>
            {lastRun.duration && (
              <span class="text-muted-foreground">
                {formatDuration(lastRun.duration)}
              </span>
            )}
          </>
        ) : (
          <button
            onClick$={() => { showLinkModal.value = true; }}
            class="text-muted-foreground hover:text-foreground"
          >
            + Link CI/CD
          </button>
        )}
      </div>
    );
  }

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">CI/CD INTEGRATION</span>
          {currentLink?.pipelineName && (
            <span class="text-[9px] px-2 py-0.5 rounded bg-muted/20 text-foreground">
              {currentLink.pipelineName}
            </span>
          )}
          {isRefreshing.value && (
            <span class="text-[8px] text-muted-foreground animate-pulse">Syncing...</span>
          )}
        </div>
        <div class="flex items-center gap-1">
          {currentLink?.pipelineId && onTriggerBuild$ && (
            <button
              onClick$={handleTriggerBuild}
              class="text-[9px] px-2 py-1 rounded bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
            >
              ▶ Run
            </button>
          )}
          <button
            onClick$={() => { showLinkModal.value = true; }}
            class="text-[9px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
          >
            {currentLink ? '⚙ Change' : '+ Link'}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div class="flex border-b border-border/30">
        {(['builds', 'deployments', 'metrics'] as const).map(tab => (
          <button
            key={tab}
            onClick$={() => { activeTab.value = tab; }}
            class={`flex-1 px-3 py-2 text-[10px] font-medium transition-colors ${
              activeTab.value === tab
                ? 'text-primary border-b-2 border-primary'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Content */}
      <div class="p-3">
        {/* Builds Tab */}
        {activeTab.value === 'builds' && (
          <div class="space-y-2">
            {pipelineRuns.value.length > 0 ? (
              pipelineRuns.value.slice(0, 5).map(run => (
                <div
                  key={run.id}
                  onClick$={() => {
                    selectedRunId.value = run.id;
                    showDetailsModal.value = true;
                  }}
                  class="p-2 rounded bg-muted/10 border border-border/30 cursor-pointer hover:bg-muted/20"
                >
                  <div class="flex items-center justify-between mb-1">
                    <div class="flex items-center gap-2">
                      <span class={`${getStatusColor(run.status)}`}>
                        {getStatusIcon(run.status)}
                      </span>
                      <span class="text-xs font-medium text-foreground">
                        #{run.id.slice(0, 8)}
                      </span>
                      <span class="text-[9px] text-muted-foreground">
                        {getTriggerIcon(run.trigger)} {run.trigger}
                      </span>
                    </div>
                    <span class="text-[9px] text-muted-foreground">
                      {formatDate(run.startedAt)}
                    </span>
                  </div>

                  <div class="text-[9px] text-muted-foreground mb-2 truncate">
                    <span class="font-mono">{run.commit.sha.slice(0, 7)}</span>
                    {' '}{run.commit.message}
                  </div>

                  {/* Jobs summary */}
                  <div class="flex items-center gap-1">
                    {run.jobs.map(job => (
                      <div
                        key={job.id}
                        title={`${job.name}: ${job.status}`}
                        class={`w-6 h-1.5 rounded-full ${
                          job.status === 'success' ? 'bg-emerald-500' :
                          job.status === 'failure' ? 'bg-red-500' :
                          job.status === 'running' ? 'bg-blue-500 animate-pulse' :
                          job.status === 'pending' ? 'bg-yellow-500' :
                          'bg-muted/30'
                        }`}
                      />
                    ))}
                  </div>

                  {run.duration && (
                    <div class="text-[8px] text-muted-foreground mt-1">
                      Duration: {formatDuration(run.duration)}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <div class="text-center py-4 text-[10px] text-muted-foreground">
                {currentLink?.pipelineId
                  ? 'No recent builds'
                  : 'Link a pipeline to view builds'}
              </div>
            )}
          </div>
        )}

        {/* Deployments Tab */}
        {activeTab.value === 'deployments' && (
          <div class="space-y-2">
            {currentLink?.deployments && currentLink.deployments.length > 0 ? (
              currentLink.deployments.map(deployment => (
                <div
                  key={deployment.id}
                  class="p-2 rounded bg-muted/10 border border-border/30"
                >
                  <div class="flex items-center justify-between mb-1">
                    <div class="flex items-center gap-2">
                      <span class={`text-[8px] px-1.5 py-0.5 rounded border ${getEnvironmentColor(deployment.environment)}`}>
                        {deployment.environment}
                      </span>
                      <span class={`text-xs ${getStatusColor(deployment.status)}`}>
                        {getStatusIcon(deployment.status)} {deployment.status}
                      </span>
                    </div>
                    {deployment.rollbackAvailable && onRollback$ && (
                      <button
                        onClick$={() => onRollback$(deployment.id)}
                        class="text-[8px] px-1.5 py-0.5 rounded bg-orange-500/20 text-orange-400"
                      >
                        Rollback
                      </button>
                    )}
                  </div>

                  <div class="flex items-center justify-between text-[9px]">
                    <div>
                      <span class="text-muted-foreground">Version: </span>
                      <span class="text-foreground font-mono">{deployment.version}</span>
                    </div>
                    <span class="text-muted-foreground">
                      {formatDate(deployment.deployedAt)}
                    </span>
                  </div>

                  <div class="text-[9px] text-muted-foreground mt-1">
                    by {deployment.deployedBy}
                    {deployment.duration && ` · ${formatDuration(deployment.duration)}`}
                  </div>
                </div>
              ))
            ) : (
              <div class="text-center py-4 text-[10px] text-muted-foreground">
                No deployments recorded
              </div>
            )}
          </div>
        )}

        {/* Metrics Tab */}
        {activeTab.value === 'metrics' && (
          <div class="space-y-3">
            {currentLink?.metrics ? (
              <>
                <div class="grid grid-cols-2 gap-2">
                  <div class="p-2 rounded bg-muted/10 border border-border/30">
                    <div class="text-[9px] text-muted-foreground">Total Builds</div>
                    <div class="text-lg font-semibold text-foreground">
                      {currentLink.metrics.totalBuilds}
                    </div>
                  </div>
                  <div class="p-2 rounded bg-muted/10 border border-border/30">
                    <div class="text-[9px] text-muted-foreground">Success Rate</div>
                    <div class={`text-lg font-semibold ${
                      currentLink.metrics.successRate >= 90 ? 'text-emerald-400' :
                      currentLink.metrics.successRate >= 70 ? 'text-yellow-400' :
                      'text-red-400'
                    }`}>
                      {currentLink.metrics.successRate.toFixed(1)}%
                    </div>
                  </div>
                  <div class="p-2 rounded bg-muted/10 border border-border/30">
                    <div class="text-[9px] text-muted-foreground">Avg Duration</div>
                    <div class="text-lg font-semibold text-foreground">
                      {formatDuration(currentLink.metrics.avgDuration)}
                    </div>
                  </div>
                  <div class="p-2 rounded bg-muted/10 border border-border/30">
                    <div class="text-[9px] text-muted-foreground">Trend</div>
                    <div class={`text-lg font-semibold ${
                      currentLink.metrics.trend === 'up' ? 'text-emerald-400' :
                      currentLink.metrics.trend === 'down' ? 'text-red-400' :
                      'text-muted-foreground'
                    }`}>
                      {currentLink.metrics.trend === 'up' ? '↑' :
                       currentLink.metrics.trend === 'down' ? '↓' : '→'}
                    </div>
                  </div>
                </div>

                {/* Success Rate Bar */}
                <div>
                  <div class="flex justify-between text-[9px] text-muted-foreground mb-1">
                    <span>Build Health</span>
                    <span>{currentLink.metrics.successRate.toFixed(1)}%</span>
                  </div>
                  <div class="h-2 bg-muted/20 rounded-full overflow-hidden">
                    <div
                      class={`h-full rounded-full transition-all ${
                        currentLink.metrics.successRate >= 90 ? 'bg-emerald-500' :
                        currentLink.metrics.successRate >= 70 ? 'bg-yellow-500' :
                        'bg-red-500'
                      }`}
                      style={{ width: `${currentLink.metrics.successRate}%` }}
                    />
                  </div>
                </div>
              </>
            ) : (
              <div class="text-center py-4 text-[10px] text-muted-foreground">
                No metrics available
              </div>
            )}
          </div>
        )}
      </div>

      {/* Link Modal */}
      {showLinkModal.value && (
        <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div class="bg-card rounded-lg border border-border p-4 w-96">
            <div class="text-xs font-semibold text-foreground mb-4">
              Link Pipeline to {laneName || 'Lane'}
            </div>

            <div class="space-y-2 max-h-[300px] overflow-y-auto">
              {pipelines.map(pipeline => (
                <div
                  key={pipeline.id}
                  onClick$={() => { selectedPipelineId.value = pipeline.id; }}
                  class={`p-3 rounded cursor-pointer ${
                    selectedPipelineId.value === pipeline.id
                      ? 'bg-primary/20 border border-primary/30'
                      : 'bg-muted/10 border border-transparent hover:bg-muted/20'
                  }`}
                >
                  <div class="flex items-center justify-between">
                    <span class="text-xs font-medium text-foreground">
                      {pipeline.name}
                    </span>
                    <span class="text-[8px] px-1.5 py-0.5 rounded bg-muted/20 text-muted-foreground">
                      {pipeline.provider}
                    </span>
                  </div>
                </div>
              ))}

              {pipelines.length === 0 && (
                <div class="text-center py-4 text-[10px] text-muted-foreground">
                  No pipelines available
                </div>
              )}
            </div>

            <div class="flex items-center gap-2 mt-4">
              <button
                onClick$={() => {
                  showLinkModal.value = false;
                  selectedPipelineId.value = null;
                }}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-muted/30 text-muted-foreground"
              >
                Cancel
              </button>
              <button
                onClick$={handleLinkPipeline}
                disabled={!selectedPipelineId.value}
                class="flex-1 px-3 py-1.5 text-xs rounded bg-primary text-primary-foreground disabled:opacity-50"
              >
                Link
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Run Details Modal */}
      {showDetailsModal.value && selectedRunId.value && (
        <BuildRunDetails
          run={pipelineRuns.value.find(r => r.id === selectedRunId.value)!}
          onClose$={() => { showDetailsModal.value = false; }}
        />
      )}
    </div>
  );
});

// ============================================================================
// Build Run Details Component
// ============================================================================

interface BuildRunDetailsProps {
  run: PipelineRun;
  onClose$: QRL<() => void>;
}

const BuildRunDetails = component$<BuildRunDetailsProps>(({ run, onClose$ }) => {
  const expandedJobs = useSignal<Set<string>>(new Set());

  const toggleJob = $((jobId: string) => {
    const newExpanded = new Set(expandedJobs.value);
    if (newExpanded.has(jobId)) {
      newExpanded.delete(jobId);
    } else {
      newExpanded.add(jobId);
    }
    expandedJobs.value = newExpanded;
  });

  return (
    <div class="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div class="bg-card rounded-lg border border-border p-4 w-[500px] max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-2">
            <span class={`${getStatusColor(run.status)}`}>
              {getStatusIcon(run.status)}
            </span>
            <span class="text-xs font-semibold text-foreground">
              Build #{run.id.slice(0, 8)}
            </span>
            <span class="text-[9px] px-2 py-0.5 rounded bg-muted/20 text-muted-foreground">
              {run.status}
            </span>
          </div>
          <button
            onClick$={onClose$}
            class="text-muted-foreground hover:text-foreground"
          >
            ✕
          </button>
        </div>

        {/* Commit info */}
        <div class="p-2 rounded bg-muted/10 mb-3">
          <div class="text-[10px] text-muted-foreground mb-1">Commit</div>
          <div class="text-xs text-foreground">
            <span class="font-mono text-cyan-400">{run.commit.sha.slice(0, 7)}</span>
            {' '}{run.commit.message}
          </div>
          <div class="text-[9px] text-muted-foreground mt-1">
            by {run.commit.author} · {getTriggerIcon(run.trigger)} {run.trigger}
          </div>
        </div>

        {/* Jobs */}
        <div class="flex-1 overflow-y-auto space-y-2">
          <div class="text-[10px] text-muted-foreground">Jobs ({run.jobs.length})</div>
          {run.jobs.map(job => (
            <div key={job.id} class="rounded border border-border/30 overflow-hidden">
              <div
                onClick$={() => toggleJob(job.id)}
                class="flex items-center justify-between p-2 bg-muted/10 cursor-pointer hover:bg-muted/20"
              >
                <div class="flex items-center gap-2">
                  <span class={`${getStatusColor(job.status)}`}>
                    {getStatusIcon(job.status)}
                  </span>
                  <span class="text-xs font-medium text-foreground">{job.name}</span>
                </div>
                <div class="flex items-center gap-2 text-[9px] text-muted-foreground">
                  {job.duration && <span>{formatDuration(job.duration)}</span>}
                  <span>{expandedJobs.value.has(job.id) ? '▼' : '▶'}</span>
                </div>
              </div>

              {expandedJobs.value.has(job.id) && (
                <div class="p-2 bg-black/20">
                  {job.logs ? (
                    <pre class="text-[9px] font-mono text-muted-foreground whitespace-pre-wrap max-h-[150px] overflow-y-auto">
                      {job.logs}
                    </pre>
                  ) : (
                    <div class="text-[9px] text-muted-foreground">No logs available</div>
                  )}
                  {job.artifacts && job.artifacts.length > 0 && (
                    <div class="mt-2 pt-2 border-t border-border/30">
                      <div class="text-[9px] text-muted-foreground mb-1">Artifacts</div>
                      {job.artifacts.map((artifact, i) => (
                        <div key={i} class="text-[9px] text-cyan-400">{artifact}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Footer */}
        <div class="flex items-center justify-between mt-4 pt-3 border-t border-border/50 text-[9px] text-muted-foreground">
          <div>
            Started: {formatDate(run.startedAt)}
            {run.finishedAt && ` · Finished: ${formatDate(run.finishedAt)}`}
          </div>
          {run.duration && <div>Total: {formatDuration(run.duration)}</div>}
        </div>
      </div>
    </div>
  );
});

export default LaneCICDIntegration;
