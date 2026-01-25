/**
 * Pipeline Types - HITL Pipeline Schema Definitions
 *
 * Types for the Raw-to-Refined HITL Pipeline visualization.
 * Supports multi-stage processing with human checkpoints.
 */

// ============================================================================
// Core Types (re-exported from component for convenience)
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

/** Stage metrics */
export interface StageMetrics {
  duration_ms?: number;
  items_processed?: number;
  items_filtered?: number;
  confidence?: number;
}

/** HITL action button configuration */
export interface HITLAction {
  id: string;
  label: string;
  variant: 'primary' | 'secondary' | 'danger' | 'success';
  icon?: string;
}

/** HITL gate configuration */
export interface HITLGate {
  reviewer?: string;
  reviewed_at?: string;
  notes?: string;
  actions: HITLAction[];
}

/** A single stage in the pipeline */
export interface PipelineStage {
  id: string;
  name: string;
  type: StageType;
  status: StageStatus;
  progress: number;
  preview?: StagePreview;
  metrics?: StageMetrics;
  hitl?: HITLGate;
  error?: string;
}

/** Pipeline source configuration */
export interface PipelineSource {
  type: string;
  id: string;
  label: string;
}

/** Pipeline configuration */
export interface PipelineConfig {
  id: string;
  name: string;
  description?: string;
  stages: PipelineStage[];
  created_at: string;
  updated_at: string;
  source?: PipelineSource;
}

// ============================================================================
// Bus Event Types
// ============================================================================

/** Bus topics for pipeline integration */
export const PIPELINE_BUS_TOPICS = {
  // Incoming
  PIPELINE_SYNC: 'pipeline.sync',
  PIPELINE_CREATED: 'pipeline.created',
  PIPELINE_UPDATED: 'pipeline.updated',
  PIPELINE_DELETED: 'pipeline.deleted',
  STAGE_PROGRESS: 'pipeline.stage.progress',
  STAGE_STATUS: 'pipeline.stage.status',

  // Outgoing (actions)
  HITL_APPROVE: 'pipeline.hitl.approve',
  HITL_REJECT: 'pipeline.hitl.reject',
  HITL_REQUEST_CHANGES: 'pipeline.hitl.request_changes',
  PIPELINE_CANCEL: 'pipeline.cancel',
  PIPELINE_RETRY: 'pipeline.retry',
} as const;

/** HITL action request payload */
export interface HITLActionRequest {
  pipeline_id: string;
  stage_id: string;
  action: string;
  notes?: string;
  reviewer?: string;
}

/** Pipeline update event payload */
export interface PipelineUpdateEvent {
  pipeline_id: string;
  stage_id?: string;
  status?: StageStatus;
  progress?: number;
  preview?: StagePreview;
  metrics?: StageMetrics;
  error?: string;
}

// ============================================================================
// Helper Functions
// ============================================================================

/** Get overall pipeline status */
export function getPipelineStatus(stages: PipelineStage[]): StageStatus {
  if (stages.some((s) => s.status === 'error')) return 'error';
  if (stages.some((s) => s.status === 'awaiting_review')) return 'awaiting_review';
  if (stages.some((s) => s.status === 'processing')) return 'processing';
  if (stages.every((s) => s.status === 'approved' || s.status === 'skipped')) return 'approved';
  if (stages.some((s) => s.status === 'rejected')) return 'rejected';
  return 'pending';
}

/** Calculate overall pipeline progress (0-100) */
export function getPipelineProgress(stages: PipelineStage[]): number {
  if (stages.length === 0) return 0;

  const weights: Record<StageStatus, number> = {
    pending: 0,
    processing: 0.5,
    awaiting_review: 0.75,
    approved: 1,
    rejected: 1,
    skipped: 1,
    error: 0,
  };

  const sum = stages.reduce((acc, s) => {
    const baseWeight = weights[s.status];
    // For processing stages, factor in the progress percentage
    if (s.status === 'processing') {
      return acc + (baseWeight * s.progress) / 100;
    }
    return acc + baseWeight;
  }, 0);

  return (sum / stages.length) * 100;
}

/** Get the current active stage (first non-completed) */
export function getActiveStage(stages: PipelineStage[]): PipelineStage | undefined {
  return stages.find(
    (s) => s.status === 'processing' || s.status === 'awaiting_review' || s.status === 'pending'
  );
}

/** Get count of stages requiring human review */
export function getHITLGateCount(stages: PipelineStage[]): number {
  return stages.filter((s) => s.status === 'awaiting_review').length;
}

/** Check if pipeline has any errors */
export function hasErrors(stages: PipelineStage[]): boolean {
  return stages.some((s) => s.status === 'error');
}

/** Get estimated time remaining based on stage metrics */
export function getEstimatedTimeRemaining(stages: PipelineStage[]): number | undefined {
  const completedStages = stages.filter(
    (s) => s.status === 'approved' || s.status === 'rejected' || s.status === 'skipped'
  );
  const remainingStages = stages.filter(
    (s) => s.status === 'pending' || s.status === 'processing' || s.status === 'awaiting_review'
  );

  if (completedStages.length === 0 || remainingStages.length === 0) {
    return undefined;
  }

  const avgDuration =
    completedStages.reduce((acc, s) => acc + (s.metrics?.duration_ms || 0), 0) /
    completedStages.length;

  return avgDuration * remainingStages.length;
}

// ============================================================================
// Pipeline Factory Functions
// ============================================================================

/** Create a new empty pipeline configuration */
export function createEmptyPipeline(name: string, source?: PipelineSource): PipelineConfig {
  const now = new Date().toISOString();
  return {
    id: `pipe-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name,
    stages: [],
    created_at: now,
    updated_at: now,
    source,
  };
}

/** Create a standard HITL pipeline with common stages */
export function createStandardPipeline(
  name: string,
  source?: PipelineSource,
  options?: {
    includeExtraction?: boolean;
    includeCleaning?: boolean;
    includeEnrichment?: boolean;
    hitlBeforeOutput?: boolean;
  }
): PipelineConfig {
  const pipeline = createEmptyPipeline(name, source);
  const opts = {
    includeExtraction: true,
    includeCleaning: true,
    includeEnrichment: true,
    hitlBeforeOutput: true,
    ...options,
  };

  let stageIndex = 1;

  // Raw input stage
  pipeline.stages.push({
    id: `stage-${stageIndex++}`,
    name: 'Raw Input',
    type: 'raw_input',
    status: 'pending',
    progress: 0,
  });

  // Extraction stage
  if (opts.includeExtraction) {
    pipeline.stages.push({
      id: `stage-${stageIndex++}`,
      name: 'Text Extraction',
      type: 'extraction',
      status: 'pending',
      progress: 0,
    });
  }

  // Cleaning stage
  if (opts.includeCleaning) {
    pipeline.stages.push({
      id: `stage-${stageIndex++}`,
      name: 'Data Cleaning',
      type: 'cleaning',
      status: 'pending',
      progress: 0,
    });
  }

  // Enrichment stage
  if (opts.includeEnrichment) {
    pipeline.stages.push({
      id: `stage-${stageIndex++}`,
      name: 'Entity Enrichment',
      type: 'enrichment',
      status: 'pending',
      progress: 0,
    });
  }

  // HITL gate before output
  if (opts.hitlBeforeOutput) {
    pipeline.stages.push({
      id: `stage-${stageIndex++}`,
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
    });
  }

  // Refined output stage
  pipeline.stages.push({
    id: `stage-${stageIndex}`,
    name: 'Refined Output',
    type: 'refined_output',
    status: 'pending',
    progress: 0,
  });

  return pipeline;
}

/** Clone a pipeline with new IDs */
export function clonePipeline(pipeline: PipelineConfig): PipelineConfig {
  const now = new Date().toISOString();
  return {
    ...pipeline,
    id: `pipe-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    stages: pipeline.stages.map((stage, index) => ({
      ...stage,
      id: `stage-${index + 1}`,
      status: 'pending' as StageStatus,
      progress: 0,
      metrics: undefined,
      error: undefined,
      hitl: stage.hitl
        ? {
            ...stage.hitl,
            reviewer: undefined,
            reviewed_at: undefined,
            notes: undefined,
          }
        : undefined,
    })),
    created_at: now,
    updated_at: now,
  };
}
