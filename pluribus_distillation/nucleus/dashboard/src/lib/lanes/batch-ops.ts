/**
 * Batch Operations for Lanes
 *
 * Phase 8, Iteration 63 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Atomic batch updates
 * - Rollback on failure
 * - Progress tracking
 * - Operation queuing
 * - Transaction support
 */

import type { Lane, LaneAction } from './store';

// ============================================================================
// Types
// ============================================================================

export interface BatchOperation {
  id: string;
  type: 'update' | 'create' | 'delete';
  laneId: string;
  changes?: Partial<Lane>;
  lane?: Lane;
}

export interface BatchResult {
  success: boolean;
  totalOperations: number;
  successCount: number;
  failedCount: number;
  errors: Array<{ operationId: string; laneId: string; error: string }>;
  rollbackPerformed: boolean;
  duration: number;
}

export interface BatchProgress {
  current: number;
  total: number;
  currentOperation?: string;
  startTime: number;
  estimatedTimeRemaining?: number;
}

export type BatchProgressCallback = (progress: BatchProgress) => void;

export interface BatchConfig {
  /** Stop on first error */
  stopOnError: boolean;
  /** Auto-rollback on any error */
  rollbackOnError: boolean;
  /** Maximum concurrent operations */
  concurrency: number;
  /** Operation timeout in ms */
  operationTimeout: number;
  /** Progress callback */
  onProgress?: BatchProgressCallback;
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: BatchConfig = {
  stopOnError: false,
  rollbackOnError: true,
  concurrency: 1, // Sequential by default for safety
  operationTimeout: 5000,
};

// ============================================================================
// Batch Executor
// ============================================================================

export class BatchExecutor {
  private config: BatchConfig;
  private operations: BatchOperation[] = [];
  private undoStack: BatchOperation[] = [];
  private aborted: boolean = false;

  constructor(config: Partial<BatchConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Add an update operation to the batch
   */
  addUpdate(laneId: string, changes: Partial<Lane>, previousState?: Partial<Lane>): this {
    this.operations.push({
      id: `update-${laneId}-${Date.now()}`,
      type: 'update',
      laneId,
      changes,
      lane: previousState as Lane,
    });
    return this;
  }

  /**
   * Add a create operation to the batch
   */
  addCreate(lane: Lane): this {
    this.operations.push({
      id: `create-${lane.id}-${Date.now()}`,
      type: 'create',
      laneId: lane.id,
      lane,
    });
    return this;
  }

  /**
   * Add a delete operation to the batch
   */
  addDelete(lane: Lane): this {
    this.operations.push({
      id: `delete-${lane.id}-${Date.now()}`,
      type: 'delete',
      laneId: lane.id,
      lane, // Store for rollback
    });
    return this;
  }

  /**
   * Add multiple update operations
   */
  addBulkUpdate(
    laneIds: string[],
    changes: Partial<Lane>,
    getLane?: (id: string) => Lane | undefined
  ): this {
    for (const laneId of laneIds) {
      const previousState = getLane ? getLane(laneId) : undefined;
      this.addUpdate(laneId, changes, previousState);
    }
    return this;
  }

  /**
   * Get operation count
   */
  getOperationCount(): number {
    return this.operations.length;
  }

  /**
   * Preview operations (for dry run)
   */
  preview(): BatchOperation[] {
    return [...this.operations];
  }

  /**
   * Abort execution
   */
  abort(): void {
    this.aborted = true;
  }

  /**
   * Execute all operations
   */
  async execute(
    applyOperation: (op: BatchOperation) => Promise<void>
  ): Promise<BatchResult> {
    const startTime = Date.now();
    const errors: Array<{ operationId: string; laneId: string; error: string }> = [];
    let successCount = 0;

    this.aborted = false;
    this.undoStack = [];

    const progress: BatchProgress = {
      current: 0,
      total: this.operations.length,
      startTime,
    };

    for (const operation of this.operations) {
      if (this.aborted) {
        break;
      }

      progress.current++;
      progress.currentOperation = `${operation.type} ${operation.laneId}`;

      // Estimate remaining time
      if (progress.current > 1) {
        const elapsed = Date.now() - startTime;
        const avgTimePerOp = elapsed / progress.current;
        progress.estimatedTimeRemaining = avgTimePerOp * (progress.total - progress.current);
      }

      // Notify progress
      if (this.config.onProgress) {
        this.config.onProgress(progress);
      }

      try {
        // Apply operation with timeout
        await Promise.race([
          applyOperation(operation),
          new Promise<never>((_, reject) =>
            setTimeout(() => reject(new Error('Operation timeout')), this.config.operationTimeout)
          ),
        ]);

        // Store for potential rollback
        this.undoStack.push(this.createUndoOperation(operation));
        successCount++;
      } catch (err: any) {
        errors.push({
          operationId: operation.id,
          laneId: operation.laneId,
          error: err?.message || 'Unknown error',
        });

        if (this.config.stopOnError) {
          break;
        }
      }
    }

    // Rollback on error if configured
    let rollbackPerformed = false;
    if (errors.length > 0 && this.config.rollbackOnError && this.undoStack.length > 0) {
      await this.rollback(applyOperation);
      rollbackPerformed = true;
    }

    const duration = Date.now() - startTime;

    return {
      success: errors.length === 0,
      totalOperations: this.operations.length,
      successCount: rollbackPerformed ? 0 : successCount,
      failedCount: errors.length,
      errors,
      rollbackPerformed,
      duration,
    };
  }

  /**
   * Rollback all executed operations
   */
  async rollback(applyOperation: (op: BatchOperation) => Promise<void>): Promise<void> {
    // Apply undo operations in reverse order
    while (this.undoStack.length > 0) {
      const undoOp = this.undoStack.pop()!;
      try {
        await applyOperation(undoOp);
      } catch (err) {
        console.error(`Failed to rollback operation ${undoOp.id}:`, err);
        // Continue with remaining rollbacks
      }
    }
  }

  /**
   * Create undo operation for a given operation
   */
  private createUndoOperation(op: BatchOperation): BatchOperation {
    switch (op.type) {
      case 'update':
        // Undo update by restoring previous state
        return {
          id: `undo-${op.id}`,
          type: 'update',
          laneId: op.laneId,
          changes: op.lane as Partial<Lane>, // Previous state
        };

      case 'create':
        // Undo create by deleting
        return {
          id: `undo-${op.id}`,
          type: 'delete',
          laneId: op.laneId,
        };

      case 'delete':
        // Undo delete by recreating
        return {
          id: `undo-${op.id}`,
          type: 'create',
          laneId: op.laneId,
          lane: op.lane,
        };

      default:
        return op;
    }
  }

  /**
   * Clear all operations
   */
  clear(): void {
    this.operations = [];
    this.undoStack = [];
    this.aborted = false;
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Create a batch update for setting status on multiple lanes
 */
export function createStatusUpdateBatch(
  laneIds: string[],
  status: 'green' | 'yellow' | 'red',
  getLane?: (id: string) => Lane | undefined
): BatchExecutor {
  const batch = new BatchExecutor();
  batch.addBulkUpdate(laneIds, { status }, getLane);
  return batch;
}

/**
 * Create a batch update for setting owner on multiple lanes
 */
export function createOwnerUpdateBatch(
  laneIds: string[],
  owner: string,
  getLane?: (id: string) => Lane | undefined
): BatchExecutor {
  const batch = new BatchExecutor();
  batch.addBulkUpdate(laneIds, { owner }, getLane);
  return batch;
}

/**
 * Create a batch update for adding tags to multiple lanes
 */
export function createAddTagBatch(
  laneIds: string[],
  tag: string,
  getLane: (id: string) => Lane | undefined
): BatchExecutor {
  const batch = new BatchExecutor();

  for (const laneId of laneIds) {
    const lane = getLane(laneId);
    if (lane) {
      const currentTags = lane.tags || [];
      if (!currentTags.includes(tag)) {
        batch.addUpdate(laneId, { tags: [...currentTags, tag] }, lane);
      }
    }
  }

  return batch;
}

/**
 * Create a batch update for removing tags from multiple lanes
 */
export function createRemoveTagBatch(
  laneIds: string[],
  tag: string,
  getLane: (id: string) => Lane | undefined
): BatchExecutor {
  const batch = new BatchExecutor();

  for (const laneId of laneIds) {
    const lane = getLane(laneId);
    if (lane && lane.tags) {
      const newTags = lane.tags.filter(t => t !== tag);
      if (newTags.length !== lane.tags.length) {
        batch.addUpdate(laneId, { tags: newTags }, lane);
      }
    }
  }

  return batch;
}

/**
 * Create a batch for archiving lanes (set status to green, add 'archived' tag)
 */
export function createArchiveBatch(
  laneIds: string[],
  getLane: (id: string) => Lane | undefined
): BatchExecutor {
  const batch = new BatchExecutor();

  for (const laneId of laneIds) {
    const lane = getLane(laneId);
    if (lane) {
      const currentTags = lane.tags || [];
      batch.addUpdate(
        laneId,
        {
          status: 'green',
          wip_pct: 100,
          tags: currentTags.includes('archived') ? currentTags : [...currentTags, 'archived'],
        },
        lane
      );
    }
  }

  return batch;
}

/**
 * Convert batch operations to lane actions
 */
export function batchOperationsToActions(operations: BatchOperation[]): LaneAction[] {
  return operations.map(op => {
    switch (op.type) {
      case 'update':
        return {
          type: 'UPDATE_LANE' as const,
          payload: { id: op.laneId, changes: op.changes || {} },
        };
      case 'create':
        return {
          type: 'ADD_LANE' as const,
          payload: op.lane!,
        };
      case 'delete':
        return {
          type: 'REMOVE_LANE' as const,
          payload: op.laneId,
        };
      default:
        throw new Error(`Unknown operation type: ${(op as any).type}`);
    }
  });
}

export default BatchExecutor;
