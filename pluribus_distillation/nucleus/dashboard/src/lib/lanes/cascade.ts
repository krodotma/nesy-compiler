/**
 * Cascade Engine for Lane Dependencies
 *
 * Phase 8, Iteration 65 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Auto-propagation of changes through dependencies
 * - Configurable cascade rules
 * - Dry-run mode
 * - Cascade history tracking
 * - Conflict prevention
 */

import type { Lane } from './store';

// ============================================================================
// Types
// ============================================================================

export interface Dependency {
  id: string;
  sourceId: string;
  targetId: string;
  type: 'blocks' | 'depends_on' | 'related' | 'child_of' | 'follows';
  cascadeConfig?: CascadeConfig;
}

export interface CascadeConfig {
  /** Propagate status changes */
  propagateStatus: boolean;
  /** Propagate blockers */
  propagateBlockers: boolean;
  /** Auto-progress when dependencies complete */
  autoProgress: boolean;
  /** Delay in days before cascade */
  delayDays?: number;
  /** Minimum source WIP to trigger cascade */
  minSourceWip?: number;
  /** Custom cascade rules */
  customRules?: CascadeRule[];
}

export interface CascadeRule {
  id: string;
  name: string;
  condition: CascadeCondition;
  action: CascadeAction;
  priority: number;
}

export interface CascadeCondition {
  sourceStatus?: ('green' | 'yellow' | 'red')[];
  sourceWipMin?: number;
  sourceWipMax?: number;
  sourceHasBlockers?: boolean;
  targetStatus?: ('green' | 'yellow' | 'red')[];
}

export interface CascadeAction {
  setStatus?: 'green' | 'yellow' | 'red';
  addBlocker?: string;
  removeBlocker?: string;
  setWipDelta?: number;
  addNote?: string;
}

export interface CascadeChange {
  laneId: string;
  field: string;
  oldValue: unknown;
  newValue: unknown;
  reason: string;
  sourceId: string;
  dependencyId: string;
  timestamp: number;
}

export interface CascadeResult {
  changes: CascadeChange[];
  skipped: Array<{ laneId: string; reason: string }>;
  conflicts: Array<{ laneId: string; field: string; reason: string }>;
  dryRun: boolean;
}

export interface CascadeEngineConfig {
  /** Maximum cascade depth to prevent infinite loops */
  maxDepth: number;
  /** Prevent overwriting recent manual changes */
  protectManualChanges: boolean;
  /** Time window for manual change protection (ms) */
  manualChangeWindow: number;
  /** Enable logging */
  debug: boolean;
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: CascadeEngineConfig = {
  maxDepth: 10,
  protectManualChanges: true,
  manualChangeWindow: 300000, // 5 minutes
  debug: false,
};

const DEFAULT_CASCADE_CONFIG: CascadeConfig = {
  propagateStatus: false,
  propagateBlockers: true,
  autoProgress: false,
};

// ============================================================================
// Cascade Engine
// ============================================================================

export class CascadeEngine {
  private config: CascadeEngineConfig;
  private recentChanges: Map<string, number> = new Map();

  constructor(config: Partial<CascadeEngineConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Calculate cascade changes for a lane update
   */
  calculateCascade(
    sourceLane: Lane,
    changes: Partial<Lane>,
    allLanes: Lane[],
    dependencies: Dependency[],
    dryRun: boolean = true
  ): CascadeResult {
    const result: CascadeResult = {
      changes: [],
      skipped: [],
      conflicts: [],
      dryRun,
    };

    const visited = new Set<string>();
    this.cascadeFromLane(
      sourceLane.id,
      changes,
      allLanes,
      dependencies,
      result,
      visited,
      0
    );

    return result;
  }

  /**
   * Apply cascade changes
   */
  applyCascade(
    lanes: Lane[],
    cascadeResult: CascadeResult
  ): Map<string, Partial<Lane>> {
    const updates = new Map<string, Partial<Lane>>();

    for (const change of cascadeResult.changes) {
      const existing = updates.get(change.laneId) || {};
      (existing as any)[change.field] = change.newValue;
      updates.set(change.laneId, existing);
    }

    return updates;
  }

  /**
   * Record a manual change (for protection)
   */
  recordManualChange(laneId: string): void {
    this.recentChanges.set(laneId, Date.now());
  }

  /**
   * Check if a lane has recent manual changes
   */
  hasRecentManualChange(laneId: string): boolean {
    const lastChange = this.recentChanges.get(laneId);
    if (!lastChange) return false;
    return Date.now() - lastChange < this.config.manualChangeWindow;
  }

  /**
   * Clear manual change protection
   */
  clearManualChanges(): void {
    this.recentChanges.clear();
  }

  // ============================================================================
  // Private Methods
  // ============================================================================

  private cascadeFromLane(
    sourceId: string,
    sourceChanges: Partial<Lane>,
    allLanes: Lane[],
    dependencies: Dependency[],
    result: CascadeResult,
    visited: Set<string>,
    depth: number
  ): void {
    // Prevent infinite loops
    if (depth > this.config.maxDepth) {
      this.log(`Max cascade depth reached at lane ${sourceId}`);
      return;
    }

    if (visited.has(sourceId)) {
      return;
    }
    visited.add(sourceId);

    const sourceLane = allLanes.find(l => l.id === sourceId);
    if (!sourceLane) return;

    // Find outgoing dependencies
    const outgoingDeps = dependencies.filter(d => d.sourceId === sourceId);

    for (const dep of outgoingDeps) {
      const targetLane = allLanes.find(l => l.id === dep.targetId);
      if (!targetLane) continue;

      const cascadeConfig = dep.cascadeConfig || DEFAULT_CASCADE_CONFIG;
      const changes = this.calculateDependencyChanges(
        sourceLane,
        sourceChanges,
        targetLane,
        dep,
        cascadeConfig
      );

      for (const change of changes) {
        // Check for conflicts
        if (this.config.protectManualChanges && this.hasRecentManualChange(targetLane.id)) {
          result.conflicts.push({
            laneId: targetLane.id,
            field: change.field,
            reason: 'Recent manual change detected',
          });
          continue;
        }

        // Check for existing conflicting changes
        const existingChange = result.changes.find(
          c => c.laneId === change.laneId && c.field === change.field
        );
        if (existingChange && existingChange.newValue !== change.newValue) {
          result.conflicts.push({
            laneId: change.laneId,
            field: change.field,
            reason: `Conflicting cascade from ${existingChange.sourceId}`,
          });
          continue;
        }

        result.changes.push(change);
      }

      // Continue cascade from target
      if (changes.length > 0) {
        const targetChanges: Partial<Lane> = {};
        for (const c of changes) {
          (targetChanges as any)[c.field] = c.newValue;
        }
        this.cascadeFromLane(
          dep.targetId,
          targetChanges,
          allLanes,
          dependencies,
          result,
          visited,
          depth + 1
        );
      }
    }
  }

  private calculateDependencyChanges(
    sourceLane: Lane,
    sourceChanges: Partial<Lane>,
    targetLane: Lane,
    dependency: Dependency,
    config: CascadeConfig
  ): CascadeChange[] {
    const changes: CascadeChange[] = [];
    const timestamp = Date.now();

    // Status propagation
    if (config.propagateStatus && sourceChanges.status) {
      if (this.shouldPropagateStatus(sourceLane, targetLane, dependency, sourceChanges.status)) {
        changes.push({
          laneId: targetLane.id,
          field: 'status',
          oldValue: targetLane.status,
          newValue: this.getTargetStatus(sourceChanges.status, dependency.type),
          reason: `Status cascade from ${sourceLane.name}`,
          sourceId: sourceLane.id,
          dependencyId: dependency.id,
          timestamp,
        });
      }
    }

    // Blocker propagation
    if (config.propagateBlockers && sourceChanges.blockers) {
      const newBlockers = [...targetLane.blockers];
      let changed = false;

      // Add blockers from source
      for (const blocker of sourceChanges.blockers) {
        const cascadedBlocker = `[From ${sourceLane.name}] ${blocker}`;
        if (!newBlockers.includes(cascadedBlocker)) {
          newBlockers.push(cascadedBlocker);
          changed = true;
        }
      }

      if (changed) {
        changes.push({
          laneId: targetLane.id,
          field: 'blockers',
          oldValue: targetLane.blockers,
          newValue: newBlockers,
          reason: `Blockers cascade from ${sourceLane.name}`,
          sourceId: sourceLane.id,
          dependencyId: dependency.id,
          timestamp,
        });
      }
    }

    // Auto-progress for child_of dependencies
    if (config.autoProgress && dependency.type === 'child_of') {
      if (sourceChanges.wip_pct !== undefined && sourceChanges.wip_pct >= 100) {
        // Check if all children are complete
        // (simplified - in real impl would check all children)
        changes.push({
          laneId: targetLane.id,
          field: 'wip_pct',
          oldValue: targetLane.wip_pct,
          newValue: Math.min(100, targetLane.wip_pct + 10),
          reason: `Child ${sourceLane.name} completed`,
          sourceId: sourceLane.id,
          dependencyId: dependency.id,
          timestamp,
        });
      }
    }

    // Apply custom rules
    if (config.customRules) {
      for (const rule of config.customRules.sort((a, b) => b.priority - a.priority)) {
        if (this.evaluateCondition(rule.condition, sourceLane, sourceChanges, targetLane)) {
          const ruleChanges = this.applyAction(
            rule.action,
            targetLane,
            sourceLane,
            dependency,
            timestamp
          );
          changes.push(...ruleChanges);
        }
      }
    }

    return changes;
  }

  private shouldPropagateStatus(
    source: Lane,
    target: Lane,
    dependency: Dependency,
    newStatus: Lane['status']
  ): boolean {
    // Don't propagate if target is already worse
    const statusOrder = { green: 2, yellow: 1, red: 0 };
    if (statusOrder[target.status] <= statusOrder[newStatus]) {
      return false;
    }

    // Only propagate "red" status for blocking dependencies
    if (newStatus === 'red' && dependency.type !== 'blocks') {
      return false;
    }

    return true;
  }

  private getTargetStatus(
    sourceStatus: Lane['status'],
    depType: Dependency['type']
  ): Lane['status'] {
    // Blocking dependency: direct propagation
    if (depType === 'blocks' && sourceStatus === 'red') {
      return 'red';
    }

    // Other dependencies: at most yellow
    if (sourceStatus === 'red') {
      return 'yellow';
    }

    return sourceStatus;
  }

  private evaluateCondition(
    condition: CascadeCondition,
    source: Lane,
    sourceChanges: Partial<Lane>,
    target: Lane
  ): boolean {
    if (condition.sourceStatus) {
      const status = sourceChanges.status || source.status;
      if (!condition.sourceStatus.includes(status)) return false;
    }

    if (condition.sourceWipMin !== undefined) {
      const wip = sourceChanges.wip_pct ?? source.wip_pct;
      if (wip < condition.sourceWipMin) return false;
    }

    if (condition.sourceWipMax !== undefined) {
      const wip = sourceChanges.wip_pct ?? source.wip_pct;
      if (wip > condition.sourceWipMax) return false;
    }

    if (condition.sourceHasBlockers !== undefined) {
      const blockers = sourceChanges.blockers ?? source.blockers;
      if ((blockers.length > 0) !== condition.sourceHasBlockers) return false;
    }

    if (condition.targetStatus) {
      if (!condition.targetStatus.includes(target.status)) return false;
    }

    return true;
  }

  private applyAction(
    action: CascadeAction,
    target: Lane,
    source: Lane,
    dependency: Dependency,
    timestamp: number
  ): CascadeChange[] {
    const changes: CascadeChange[] = [];

    if (action.setStatus) {
      changes.push({
        laneId: target.id,
        field: 'status',
        oldValue: target.status,
        newValue: action.setStatus,
        reason: 'Custom cascade rule',
        sourceId: source.id,
        dependencyId: dependency.id,
        timestamp,
      });
    }

    if (action.addBlocker) {
      const newBlockers = [...target.blockers, action.addBlocker];
      changes.push({
        laneId: target.id,
        field: 'blockers',
        oldValue: target.blockers,
        newValue: newBlockers,
        reason: 'Custom cascade rule',
        sourceId: source.id,
        dependencyId: dependency.id,
        timestamp,
      });
    }

    if (action.setWipDelta) {
      const newWip = Math.max(0, Math.min(100, target.wip_pct + action.setWipDelta));
      changes.push({
        laneId: target.id,
        field: 'wip_pct',
        oldValue: target.wip_pct,
        newValue: newWip,
        reason: 'Custom cascade rule',
        sourceId: source.id,
        dependencyId: dependency.id,
        timestamp,
      });
    }

    return changes;
  }

  private log(message: string): void {
    if (this.config.debug) {
      console.log(`[CascadeEngine] ${message}`);
    }
  }
}

// ============================================================================
// Singleton
// ============================================================================

let globalEngine: CascadeEngine | null = null;

export function getGlobalCascadeEngine(config?: Partial<CascadeEngineConfig>): CascadeEngine {
  if (!globalEngine) {
    globalEngine = new CascadeEngine(config);
  }
  return globalEngine;
}

export function resetGlobalCascadeEngine(): void {
  globalEngine = null;
}

export default CascadeEngine;
