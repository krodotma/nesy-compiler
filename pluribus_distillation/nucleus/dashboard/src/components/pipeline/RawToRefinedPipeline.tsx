/**
 * RawToRefinedPipeline.tsx - HITL Pipeline Visualization
 *
 * A comprehensive visualization component for Human-In-The-Loop (HITL)
 * processing pipelines. Shows the flow from raw input through processing
 * stages to refined output, with human checkpoints at critical gates.
 *
 * Features:
 * - Pipeline flow diagram with animated transitions
 * - Stage status indicators (pending, processing, awaiting_review, approved, rejected)
 * - Inline preview panels for each stage
 * - Action buttons at HITL gates
 * - Progress tracking with metrics
 *
 * Reference: PORTAL metabolic flow, strp_leads ingest process
 */

import {
  component$,
  useSignal,
  useStore,
  useComputed$,
  useVisibleTask$,
  type QRL,
  $,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

/** Status of a pipeline stage */
export type StageStatus =
  | 'pending'
  | 'processing'
  | 'awaiting_review'
  | 'approved'
  | 'rejected'
  | 'skipped'
  | 'error';

/** Type of processing stage */
export type StageType =
  | 'raw_input'
  | 'extraction'
  | 'cleaning'
  | 'enrichment'
  | 'hitl_gate'
  | 'refined_output';

/** Preview data for a stage */
export interface StagePreview {
  type: 'text' | 'json' | 'image' | 'diff' | 'table';
  content: string;
  metadata?: Record<string, string | number | boolean>;
}

/** A single stage in the pipeline */
export interface PipelineStage {
  id: string;
  name: string;
  type: StageType;
  status: StageStatus;
  progress: number; // 0-100
  preview?: StagePreview;
  metrics?: {
    duration_ms?: number;
    items_processed?: number;
    items_filtered?: number;
    confidence?: number;
  };
  hitl?: {
    reviewer?: string;
    reviewed_at?: string;
    notes?: string;
    actions: HITLAction[];
  };
  error?: string;
}

/** HITL action button configuration */
export interface HITLAction {
  id: string;
  label: string;
  variant: 'primary' | 'secondary' | 'danger' | 'success';
  icon?: string;
}

/** Pipeline configuration */
export interface PipelineConfig {
  id: string;
  name: string;
  description?: string;
  stages: PipelineStage[];
  created_at: string;
  updated_at: string;
  source?: {
    type: string;
    id: string;
    label: string;
  };
}

/** Props for the component */
export interface RawToRefinedPipelineProps {
  /** Pipeline configuration and state */
  pipeline: PipelineConfig;
  /** Callback when HITL action is triggered */
  onHITLAction$?: QRL<(stageId: string, actionId: string, notes?: string) => void>;
  /** Callback when stage is expanded */
  onStageExpand$?: QRL<(stageId: string) => void>;
  /** Compact mode for embedding */
  compact?: boolean;
  /** Enable live updates */
  liveUpdates?: boolean;
}

// ============================================================================
// Utility Functions
// ============================================================================

function getStatusColor(status: StageStatus): string {
  switch (status) {
    case 'pending':
      return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
    case 'processing':
      return 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30';
    case 'awaiting_review':
      return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'approved':
      return 'bg-green-500/20 text-green-400 border-green-500/30';
    case 'rejected':
      return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'skipped':
      return 'bg-purple-500/20 text-purple-400 border-purple-500/30';
    case 'error':
      return 'bg-red-600/30 text-red-300 border-red-500/50';
  }
}

function getStatusIcon(status: StageStatus): string {
  switch (status) {
    case 'pending':
      return '...';
    case 'processing':
      return '*';
    case 'awaiting_review':
      return '?';
    case 'approved':
      return '+';
    case 'rejected':
      return 'x';
    case 'skipped':
      return '-';
    case 'error':
      return '!';
  }
}

function getStageTypeIcon(type: StageType): string {
  switch (type) {
    case 'raw_input':
      return '[RAW]';
    case 'extraction':
      return '[EXT]';
    case 'cleaning':
      return '[CLN]';
    case 'enrichment':
      return '[ENR]';
    case 'hitl_gate':
      return '[HITL]';
    case 'refined_output':
      return '[OUT]';
  }
}

function getActionButtonClass(variant: HITLAction['variant']): string {
  switch (variant) {
    case 'primary':
      return 'bg-cyan-600 hover:bg-cyan-500 text-white';
    case 'secondary':
      return 'bg-gray-600 hover:bg-gray-500 text-white';
    case 'danger':
      return 'bg-red-600 hover:bg-red-500 text-white';
    case 'success':
      return 'bg-green-600 hover:bg-green-500 text-white';
  }
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// ============================================================================
// Sub-components
// ============================================================================

/** Stage connection line with animation */
const StageConnector = component$<{
  status: StageStatus;
  isLast: boolean;
}>(({ status, isLast }) => {
  if (isLast) return null;

  const lineColor = (() => {
    switch (status) {
      case 'approved':
        return 'from-green-500 to-green-500/30';
      case 'processing':
        return 'from-cyan-500 to-cyan-500/30 animate-pulse';
      case 'awaiting_review':
        return 'from-amber-500 to-amber-500/30';
      case 'rejected':
        return 'from-red-500 to-red-500/30';
      default:
        return 'from-gray-600 to-gray-600/30';
    }
  })();

  return (
    <div class="flex-shrink-0 w-12 md:w-20 flex items-center justify-center">
      <div class={`h-0.5 w-full bg-gradient-to-r ${lineColor}`} />
      {status === 'processing' && (
        <div class="absolute w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
      )}
    </div>
  );
});

/** Preview panel for stage content */
const PreviewPanel = component$<{
  preview: StagePreview;
  compact?: boolean;
}>(({ preview, compact }) => {
  const maxHeight = compact ? 'max-h-24' : 'max-h-48';

  return (
    <div class={`mt-2 rounded border border-border/50 bg-black/40 overflow-hidden ${maxHeight}`}>
      <div class="flex items-center justify-between px-2 py-1 border-b border-border/30 bg-muted/30">
        <span class="text-[9px] font-mono uppercase text-muted-foreground">
          {preview.type}
        </span>
        {preview.metadata && (
          <div class="flex gap-2">
            {Object.entries(preview.metadata).slice(0, 3).map(([k, v]) => (
              <span key={k} class="text-[8px] text-muted-foreground">
                {k}:{String(v)}
              </span>
            ))}
          </div>
        )}
      </div>
      <div class={`p-2 overflow-auto ${maxHeight}`}>
        {preview.type === 'text' && (
          <pre class="text-[10px] text-foreground/80 whitespace-pre-wrap font-mono">
            {preview.content}
          </pre>
        )}
        {preview.type === 'json' && (
          <pre class="text-[10px] text-cyan-300/80 whitespace-pre font-mono">
            {preview.content}
          </pre>
        )}
        {preview.type === 'image' && (
          <div class="flex items-center justify-center">
            <img
              src={preview.content}
              alt="Stage preview"
              class="max-h-32 object-contain rounded"
            />
          </div>
        )}
        {preview.type === 'diff' && (
          <pre class="text-[10px] font-mono">
            {preview.content.split('\n').map((line, i) => (
              <div
                key={i}
                class={
                  line.startsWith('+')
                    ? 'text-green-400 bg-green-500/10'
                    : line.startsWith('-')
                    ? 'text-red-400 bg-red-500/10'
                    : 'text-foreground/60'
                }
              >
                {line}
              </div>
            ))}
          </pre>
        )}
        {preview.type === 'table' && (
          <div class="text-[10px] font-mono text-foreground/80">
            {preview.content}
          </div>
        )}
      </div>
    </div>
  );
});

/** HITL Gate action panel */
const HITLGatePanel = component$<{
  stage: PipelineStage;
  onAction$: QRL<(actionId: string, notes?: string) => void>;
}>(({ stage, onAction$ }) => {
  const notes = useSignal('');
  const isSubmitting = useSignal(false);

  if (!stage.hitl || stage.status !== 'awaiting_review') return null;

  return (
    <div class="mt-3 p-3 rounded-lg border-2 border-amber-500/30 bg-amber-500/5 animate-pulse-slow">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-amber-400 text-sm font-bold">[HITL GATE]</span>
        <span class="text-[10px] text-muted-foreground">Human review required</span>
      </div>

      {/* Notes input */}
      <textarea
        value={notes.value}
        onInput$={(e) => {
          notes.value = (e.target as HTMLTextAreaElement).value;
        }}
        placeholder="Add review notes (optional)..."
        class="w-full px-2 py-1.5 text-xs rounded border border-border bg-background/50 placeholder:text-muted-foreground/50 focus:border-amber-500/50 focus:outline-none resize-none"
        rows={2}
      />

      {/* Action buttons */}
      <div class="flex items-center gap-2 mt-2">
        {stage.hitl.actions.map((action) => (
          <button
            key={action.id}
            onClick$={async () => {
              isSubmitting.value = true;
              await onAction$(action.id, notes.value || undefined);
              isSubmitting.value = false;
              notes.value = '';
            }}
            disabled={isSubmitting.value}
            class={`px-3 py-1.5 rounded text-xs font-medium transition-all ${getActionButtonClass(action.variant)} ${
              isSubmitting.value ? 'opacity-50 cursor-not-allowed' : ''
            }`}
          >
            {action.icon && <span class="mr-1">{action.icon}</span>}
            {action.label}
          </button>
        ))}
      </div>

      {/* Previous review info */}
      {stage.hitl.reviewer && (
        <div class="mt-2 pt-2 border-t border-border/30 text-[9px] text-muted-foreground">
          Last reviewed by <span class="text-foreground/80">{stage.hitl.reviewer}</span>
          {stage.hitl.reviewed_at && (
            <span> at {new Date(stage.hitl.reviewed_at).toLocaleString()}</span>
          )}
          {stage.hitl.notes && (
            <div class="mt-1 italic">"{stage.hitl.notes}"</div>
          )}
        </div>
      )}
    </div>
  );
});

/** Individual stage card */
const StageCard = component$<{
  stage: PipelineStage;
  index: number;
  total: number;
  onAction$?: QRL<(stageId: string, actionId: string, notes?: string) => void>;
  onExpand$?: QRL<(stageId: string) => void>;
  compact?: boolean;
}>(({ stage, index, total, onAction$, onExpand$, compact }) => {
  const isExpanded = useSignal(stage.status === 'awaiting_review');
  const statusColor = getStatusColor(stage.status);
  const statusIcon = getStatusIcon(stage.status);
  const typeIcon = getStageTypeIcon(stage.type);

  return (
    <div class="flex items-start flex-shrink-0">
      {/* Stage node */}
      <div
        class={`relative flex flex-col min-w-[180px] max-w-[280px] rounded-lg border bg-card/80 backdrop-blur-sm transition-all hover:bg-card ${
          stage.status === 'awaiting_review'
            ? 'border-amber-500/50 shadow-lg shadow-amber-500/10'
            : 'border-border/50 hover:border-border'
        } ${compact ? 'p-2' : 'p-3'}`}
        onClick$={() => {
          isExpanded.value = !isExpanded.value;
          if (onExpand$) onExpand$(stage.id);
        }}
      >
        {/* Stage number indicator */}
        <div class="absolute -top-2 -left-2 w-5 h-5 rounded-full bg-background border border-border flex items-center justify-center text-[10px] font-mono text-muted-foreground">
          {index + 1}
        </div>

        {/* Header */}
        <div class="flex items-center justify-between mb-2">
          <div class="flex items-center gap-2">
            <span class="text-[10px] font-mono text-primary/70">{typeIcon}</span>
            <span class="text-xs font-medium text-foreground truncate">{stage.name}</span>
          </div>
          <span
            class={`text-[10px] px-1.5 py-0.5 rounded border font-mono ${statusColor}`}
          >
            {statusIcon}
          </span>
        </div>

        {/* Progress bar */}
        {stage.status === 'processing' && (
          <div class="w-full h-1 bg-muted/30 rounded-full overflow-hidden mb-2">
            <div
              class="h-full bg-cyan-500 transition-all duration-300"
              style={{ width: `${stage.progress}%` }}
            />
          </div>
        )}

        {/* Metrics row */}
        {stage.metrics && !compact && (
          <div class="flex flex-wrap gap-2 text-[9px] text-muted-foreground mb-2">
            {stage.metrics.duration_ms !== undefined && (
              <span class="px-1.5 py-0.5 rounded bg-muted/30">
                T:{formatDuration(stage.metrics.duration_ms)}
              </span>
            )}
            {stage.metrics.items_processed !== undefined && (
              <span class="px-1.5 py-0.5 rounded bg-muted/30">
                IN:{stage.metrics.items_processed}
              </span>
            )}
            {stage.metrics.items_filtered !== undefined && (
              <span class="px-1.5 py-0.5 rounded bg-muted/30">
                OUT:{stage.metrics.items_filtered}
              </span>
            )}
            {stage.metrics.confidence !== undefined && (
              <span class="px-1.5 py-0.5 rounded bg-muted/30">
                C:{(stage.metrics.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
        )}

        {/* Error display */}
        {stage.error && (
          <div class="text-[10px] text-red-400 bg-red-500/10 rounded px-2 py-1 mb-2 font-mono">
            ERR: {stage.error}
          </div>
        )}

        {/* Preview panel (expanded) */}
        {isExpanded.value && stage.preview && (
          <PreviewPanel preview={stage.preview} compact={compact} />
        )}

        {/* HITL Gate panel */}
        {isExpanded.value && stage.hitl && onAction$ && (
          <HITLGatePanel
            stage={stage}
            onAction$={$((actionId, notes) => onAction$(stage.id, actionId, notes))}
          />
        )}

        {/* Expand indicator */}
        {(stage.preview || stage.hitl) && (
          <div class="mt-2 text-center text-[9px] text-muted-foreground/50">
            {isExpanded.value ? '[collapse]' : '[expand]'}
          </div>
        )}
      </div>

      {/* Connector to next stage */}
      <StageConnector status={stage.status} isLast={index === total - 1} />
    </div>
  );
});

/** Progress summary bar */
const PipelineProgressBar = component$<{
  stages: PipelineStage[];
}>(({ stages }) => {
  const stats = useComputed$(() => {
    const total = stages.length;
    const completed = stages.filter(
      (s) => s.status === 'approved' || s.status === 'rejected' || s.status === 'skipped'
    ).length;
    const inProgress = stages.filter((s) => s.status === 'processing').length;
    const awaiting = stages.filter((s) => s.status === 'awaiting_review').length;
    const pending = stages.filter((s) => s.status === 'pending').length;
    const errors = stages.filter((s) => s.status === 'error').length;

    return { total, completed, inProgress, awaiting, pending, errors };
  });

  return (
    <div class="flex flex-col gap-2">
      {/* Progress bar */}
      <div class="flex h-2 rounded-full overflow-hidden bg-muted/30">
        <div
          class="bg-green-500 transition-all duration-500"
          style={{ width: `${(stats.value.completed / stats.value.total) * 100}%` }}
        />
        <div
          class="bg-cyan-500 animate-pulse transition-all duration-500"
          style={{ width: `${(stats.value.inProgress / stats.value.total) * 100}%` }}
        />
        <div
          class="bg-amber-500 transition-all duration-500"
          style={{ width: `${(stats.value.awaiting / stats.value.total) * 100}%` }}
        />
        <div
          class="bg-red-500 transition-all duration-500"
          style={{ width: `${(stats.value.errors / stats.value.total) * 100}%` }}
        />
      </div>

      {/* Stats row */}
      <div class="flex items-center gap-4 text-[10px]">
        <span class="text-green-400">{stats.value.completed} completed</span>
        {stats.value.inProgress > 0 && (
          <span class="text-cyan-400">{stats.value.inProgress} processing</span>
        )}
        {stats.value.awaiting > 0 && (
          <span class="text-amber-400 font-bold animate-pulse">
            {stats.value.awaiting} awaiting review
          </span>
        )}
        {stats.value.pending > 0 && (
          <span class="text-gray-400">{stats.value.pending} pending</span>
        )}
        {stats.value.errors > 0 && (
          <span class="text-red-400">{stats.value.errors} errors</span>
        )}
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

export const RawToRefinedPipeline = component$<RawToRefinedPipelineProps>((props) => {
  const { pipeline, onHITLAction$, onStageExpand$, compact = false, liveUpdates = true } = props;

  const containerRef = useSignal<HTMLDivElement>();
  const isAutoScrolling = useSignal(true);
  const lastUpdateTime = useSignal<number>(Date.now());

  // Auto-scroll to awaiting_review stage
  useVisibleTask$(({ track }) => {
    track(() => pipeline.stages);

    if (!isAutoScrolling.value || !containerRef.value) return;

    const awaitingIndex = pipeline.stages.findIndex((s) => s.status === 'awaiting_review');
    if (awaitingIndex >= 0) {
      const stageWidth = compact ? 200 : 240;
      const connectorWidth = compact ? 48 : 80;
      const scrollTarget = awaitingIndex * (stageWidth + connectorWidth);
      containerRef.value.scrollTo({ left: scrollTarget, behavior: 'smooth' });
    }
  });

  // Live update pulse
  useVisibleTask$(({ cleanup }) => {
    if (!liveUpdates) return;

    const interval = setInterval(() => {
      lastUpdateTime.value = Date.now();
    }, 5000);

    cleanup(() => clearInterval(interval));
  });

  // Calculate overall progress
  const overallProgress = useComputed$(() => {
    const weights: Record<StageStatus, number> = {
      pending: 0,
      processing: 0.5,
      awaiting_review: 0.75,
      approved: 1,
      rejected: 1,
      skipped: 1,
      error: 0,
    };
    const total = pipeline.stages.length;
    if (total === 0) return 0;
    const sum = pipeline.stages.reduce((acc, s) => acc + weights[s.status], 0);
    return (sum / total) * 100;
  });

  return (
    <div class={`space-y-4 ${compact ? 'text-sm' : ''}`}>
      {/* Header */}
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div class="flex items-center gap-3">
            <h2 class={`font-semibold text-foreground ${compact ? 'text-base' : 'text-lg'}`}>
              {pipeline.name}
            </h2>
            <span class="text-[10px] font-mono text-muted-foreground px-2 py-0.5 rounded bg-muted/30">
              ID:{pipeline.id.slice(0, 8)}
            </span>
            {liveUpdates && (
              <span class="flex items-center gap-1 text-[10px] text-green-400">
                <span class="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
                LIVE
              </span>
            )}
          </div>
          {pipeline.description && !compact && (
            <p class="text-sm text-muted-foreground mt-1">{pipeline.description}</p>
          )}
        </div>

        <div class="flex items-center gap-4">
          {/* Source badge */}
          {pipeline.source && (
            <div class="text-[10px] px-2 py-1 rounded border border-border bg-muted/30">
              <span class="text-muted-foreground">Source:</span>{' '}
              <span class="text-primary">{pipeline.source.label}</span>
            </div>
          )}

          {/* Overall progress */}
          <div class="flex items-center gap-2">
            <span class="text-[10px] text-muted-foreground">Progress:</span>
            <span class="text-sm font-mono text-primary font-bold">
              {overallProgress.value.toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* Progress summary */}
      <PipelineProgressBar stages={pipeline.stages} />

      {/* Pipeline flow */}
      <div class="relative">
        {/* Scroll controls */}
        <button
          onClick$={() => {
            if (containerRef.value) {
              containerRef.value.scrollBy({ left: -200, behavior: 'smooth' });
            }
          }}
          class="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-8 h-8 rounded-full bg-background/90 border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-primary/50 transition-all"
        >
          {'<'}
        </button>
        <button
          onClick$={() => {
            if (containerRef.value) {
              containerRef.value.scrollBy({ left: 200, behavior: 'smooth' });
            }
          }}
          class="absolute right-0 top-1/2 -translate-y-1/2 z-10 w-8 h-8 rounded-full bg-background/90 border border-border flex items-center justify-center text-muted-foreground hover:text-foreground hover:border-primary/50 transition-all"
        >
          {'>'}
        </button>

        {/* Stage flow container */}
        <div
          ref={containerRef}
          class="flex items-start gap-0 overflow-x-auto py-4 px-10 scroll-smooth"
          onScroll$={() => {
            isAutoScrolling.value = false;
          }}
        >
          {pipeline.stages.map((stage, index) => (
            <StageCard
              key={stage.id}
              stage={stage}
              index={index}
              total={pipeline.stages.length}
              onAction$={onHITLAction$}
              onExpand$={onStageExpand$}
              compact={compact}
            />
          ))}
        </div>
      </div>

      {/* Footer metadata */}
      <div class="flex items-center justify-between text-[9px] text-muted-foreground border-t border-border/30 pt-2">
        <div class="flex items-center gap-4">
          <span>Created: {new Date(pipeline.created_at).toLocaleString()}</span>
          <span>Updated: {new Date(pipeline.updated_at).toLocaleString()}</span>
        </div>
        <div class="flex items-center gap-2">
          <button
            onClick$={() => {
              isAutoScrolling.value = true;
            }}
            class="px-2 py-0.5 rounded border border-border hover:border-primary/30 hover:text-foreground transition-all"
          >
            Auto-focus
          </button>
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Demo / Test Data Generator
// ============================================================================

/** Generate a demo pipeline for testing */
export function createDemoPipeline(): PipelineConfig {
  return {
    id: 'pipe-demo-001',
    name: 'Content Ingestion Pipeline',
    description: 'Processes raw content through extraction, cleaning, and enrichment with human review gates.',
    created_at: new Date(Date.now() - 3600000).toISOString(),
    updated_at: new Date().toISOString(),
    source: {
      type: 'file',
      id: 'upload-123',
      label: 'document.pdf',
    },
    stages: [
      {
        id: 'stage-1',
        name: 'Raw Input',
        type: 'raw_input',
        status: 'approved',
        progress: 100,
        preview: {
          type: 'text',
          content: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.',
          metadata: { size: '2.4KB', format: 'PDF' },
        },
        metrics: {
          duration_ms: 234,
          items_processed: 1,
        },
      },
      {
        id: 'stage-2',
        name: 'Text Extraction',
        type: 'extraction',
        status: 'approved',
        progress: 100,
        preview: {
          type: 'json',
          content: JSON.stringify({ pages: 3, paragraphs: 12, words: 847 }, null, 2),
        },
        metrics: {
          duration_ms: 1872,
          items_processed: 3,
          confidence: 0.94,
        },
      },
      {
        id: 'stage-3',
        name: 'Data Cleaning',
        type: 'cleaning',
        status: 'approved',
        progress: 100,
        preview: {
          type: 'diff',
          content: '- Lorem ipsum dolor sit amet...\n+ Lorem ipsum dolor sit amet, cleaned...\n  No changes to paragraph 2\n- Duplicate content removed\n+ [MERGED]',
        },
        metrics: {
          duration_ms: 542,
          items_processed: 847,
          items_filtered: 823,
        },
      },
      {
        id: 'stage-4',
        name: 'Entity Enrichment',
        type: 'enrichment',
        status: 'processing',
        progress: 67,
        preview: {
          type: 'json',
          content: JSON.stringify({
            entities: ['Person:John Doe', 'Org:Acme Corp', 'Location:New York'],
            relations: 2,
          }, null, 2),
        },
        metrics: {
          items_processed: 67,
        },
      },
      {
        id: 'stage-5',
        name: 'Quality Review',
        type: 'hitl_gate',
        status: 'pending',
        progress: 0,
        hitl: {
          actions: [
            { id: 'approve', label: 'Approve', variant: 'success', icon: '+' },
            { id: 'reject', label: 'Reject', variant: 'danger', icon: 'x' },
            { id: 'request_changes', label: 'Request Changes', variant: 'secondary', icon: '?' },
          ],
        },
      },
      {
        id: 'stage-6',
        name: 'Refined Output',
        type: 'refined_output',
        status: 'pending',
        progress: 0,
      },
    ],
  };
}

export default RawToRefinedPipeline;
