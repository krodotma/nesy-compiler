/**
 * BranchReconciler: Git Branch Analysis and Merge Conflict Prediction
 *
 * Step 22 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Analyzes git branches for:
 * - Divergence patterns
 * - Merge conflict likelihood
 * - Branch health metrics
 * - Reconciliation strategies
 *
 * Core insight: Branches that diverge on the same files
 * will conflict; branches that diverge on different files
 * can be merged cleanly.
 */

import type { GitCommit, GitFileChange } from './git-walker';

export interface BranchInfo {
  name: string;
  head: string;
  baseCommit: string;
  divergencePoint: string;
  commitCount: number;
  filesChanged: Set<string>;
  authors: Set<string>;
  lastActivity: Date;
  aheadOf: number;
  behindOf: number;
}

export interface BranchPair {
  source: BranchInfo;
  target: BranchInfo;
  commonAncestor: string;
  conflictingFiles: string[];
  conflictProbability: number;
  mergeComplexity: MergeComplexity;
  recommendation: ReconciliationStrategy;
}

export type MergeComplexity = 'trivial' | 'simple' | 'moderate' | 'complex' | 'dangerous';

export type ReconciliationStrategy =
  | 'fast-forward'
  | 'clean-merge'
  | 'rebase'
  | 'squash-merge'
  | 'manual-review'
  | 'split-pr';

export interface ConflictPrediction {
  file: string;
  probability: number;
  sourceChanges: number;
  targetChanges: number;
  overlappingLines: boolean;
  conflictType: 'content' | 'rename' | 'delete' | 'mode';
  severity: 'low' | 'medium' | 'high';
}

export interface ReconciliationPlan {
  pair: BranchPair;
  strategy: ReconciliationStrategy;
  steps: ReconciliationStep[];
  estimatedEffort: number; // 1-10 scale
  risks: string[];
  prerequisites: string[];
}

export interface ReconciliationStep {
  action: string;
  command?: string;
  description: string;
  automated: boolean;
}

export interface BranchReconcilerConfig {
  /** Threshold for conflict probability to flag as risky */
  conflictThreshold: number;
  /** Threshold for considering branch stale (days) */
  staleThreshold: number;
  /** Max commits before recommending squash */
  squashThreshold: number;
  /** Max files before recommending split */
  splitThreshold: number;
}

const DEFAULT_CONFIG: BranchReconcilerConfig = {
  conflictThreshold: 0.3,
  staleThreshold: 30,
  squashThreshold: 20,
  splitThreshold: 50,
};

/**
 * BranchReconciler: Analyze and reconcile git branches.
 */
export class BranchReconciler {
  private config: BranchReconcilerConfig;

  constructor(config?: Partial<BranchReconcilerConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze a pair of branches for merge potential.
   */
  analyzePair(
    sourceCommits: GitCommit[],
    targetCommits: GitCommit[],
    sourceBranch: string,
    targetBranch: string,
    commonAncestor: string
  ): BranchPair {
    const source = this.buildBranchInfo(sourceCommits, sourceBranch, commonAncestor);
    const target = this.buildBranchInfo(targetCommits, targetBranch, commonAncestor);

    // Find conflicting files
    const conflictingFiles = this.findConflictingFiles(source, target);

    // Calculate conflict probability
    const conflictProbability = this.calculateConflictProbability(
      source,
      target,
      conflictingFiles
    );

    // Determine merge complexity
    const mergeComplexity = this.assessMergeComplexity(
      source,
      target,
      conflictingFiles,
      conflictProbability
    );

    // Generate recommendation
    const recommendation = this.recommendStrategy(
      source,
      target,
      mergeComplexity,
      conflictProbability
    );

    return {
      source,
      target,
      commonAncestor,
      conflictingFiles,
      conflictProbability,
      mergeComplexity,
      recommendation,
    };
  }

  /**
   * Build branch info from commits.
   */
  private buildBranchInfo(
    commits: GitCommit[],
    branchName: string,
    divergencePoint: string
  ): BranchInfo {
    const filesChanged = new Set<string>();
    const authors = new Set<string>();

    for (const commit of commits) {
      authors.add(commit.author);
      for (const file of commit.files) {
        filesChanged.add(file.path);
        if (file.oldPath) {
          filesChanged.add(file.oldPath);
        }
      }
    }

    const sortedCommits = [...commits].sort(
      (a, b) => b.date.getTime() - a.date.getTime()
    );

    return {
      name: branchName,
      head: sortedCommits[0]?.hash || divergencePoint,
      baseCommit: sortedCommits[sortedCommits.length - 1]?.hash || divergencePoint,
      divergencePoint,
      commitCount: commits.length,
      filesChanged,
      authors,
      lastActivity: sortedCommits[0]?.date || new Date(),
      aheadOf: commits.length,
      behindOf: 0, // Would need target commits to calculate
    };
  }

  /**
   * Find files that both branches modified.
   */
  private findConflictingFiles(source: BranchInfo, target: BranchInfo): string[] {
    const conflicts: string[] = [];

    for (const file of source.filesChanged) {
      if (target.filesChanged.has(file)) {
        conflicts.push(file);
      }
    }

    return conflicts;
  }

  /**
   * Calculate probability of merge conflicts.
   */
  private calculateConflictProbability(
    source: BranchInfo,
    target: BranchInfo,
    conflictingFiles: string[]
  ): number {
    if (conflictingFiles.length === 0) {
      return 0;
    }

    // Base probability from file overlap
    const overlapRatio = conflictingFiles.length /
      Math.max(source.filesChanged.size, target.filesChanged.size);

    // Adjust for number of commits (more commits = more changes = higher conflict chance)
    const commitFactor = Math.min(1, (source.commitCount + target.commitCount) / 50);

    // Adjust for author overlap (same authors = less conflict)
    const authorOverlap = this.calculateSetOverlap(source.authors, target.authors);
    const authorFactor = 1 - (authorOverlap * 0.3);

    // Combine factors
    const probability = Math.min(1, overlapRatio * commitFactor * authorFactor);

    return probability;
  }

  /**
   * Calculate overlap between two sets.
   */
  private calculateSetOverlap<T>(a: Set<T>, b: Set<T>): number {
    let overlap = 0;
    for (const item of a) {
      if (b.has(item)) {
        overlap++;
      }
    }
    return overlap / Math.max(a.size, b.size);
  }

  /**
   * Assess the complexity of merging.
   */
  private assessMergeComplexity(
    source: BranchInfo,
    target: BranchInfo,
    conflictingFiles: string[],
    conflictProbability: number
  ): MergeComplexity {
    // Fast-forward possible
    if (target.commitCount === 0) {
      return 'trivial';
    }

    // No conflicts
    if (conflictingFiles.length === 0) {
      return 'simple';
    }

    // Low conflict probability
    if (conflictProbability < 0.2) {
      return 'moderate';
    }

    // High conflict probability but manageable
    if (conflictProbability < 0.5 && conflictingFiles.length < 10) {
      return 'complex';
    }

    // Dangerous merge
    return 'dangerous';
  }

  /**
   * Recommend a reconciliation strategy.
   */
  private recommendStrategy(
    source: BranchInfo,
    target: BranchInfo,
    complexity: MergeComplexity,
    conflictProbability: number
  ): ReconciliationStrategy {
    // Check for stale branch
    const daysSinceActivity = (Date.now() - source.lastActivity.getTime()) / (1000 * 60 * 60 * 24);
    const isStale = daysSinceActivity > this.config.staleThreshold;

    // Fast-forward if possible
    if (complexity === 'trivial') {
      return 'fast-forward';
    }

    // Clean merge for simple cases
    if (complexity === 'simple') {
      return 'clean-merge';
    }

    // Rebase for moderate complexity with few commits
    if (complexity === 'moderate' && source.commitCount < 10) {
      return 'rebase';
    }

    // Squash for many commits
    if (source.commitCount > this.config.squashThreshold) {
      return 'squash-merge';
    }

    // Split PR for too many files
    if (source.filesChanged.size > this.config.splitThreshold) {
      return 'split-pr';
    }

    // Manual review for dangerous merges
    return 'manual-review';
  }

  /**
   * Predict conflicts for each file.
   */
  predictConflicts(
    pair: BranchPair,
    sourceCommits: GitCommit[],
    targetCommits: GitCommit[]
  ): ConflictPrediction[] {
    const predictions: ConflictPrediction[] = [];

    for (const file of pair.conflictingFiles) {
      const sourceChanges = this.countFileChanges(file, sourceCommits);
      const targetChanges = this.countFileChanges(file, targetCommits);

      // Check for rename/delete conflicts
      const sourceStatus = this.getFileStatus(file, sourceCommits);
      const targetStatus = this.getFileStatus(file, targetCommits);

      let conflictType: ConflictPrediction['conflictType'] = 'content';
      if (sourceStatus === 'D' || targetStatus === 'D') {
        conflictType = 'delete';
      } else if (sourceStatus === 'R' || targetStatus === 'R') {
        conflictType = 'rename';
      }

      // Estimate probability based on change volume
      const changeFactor = Math.min(1, (sourceChanges + targetChanges) / 100);
      const probability = pair.conflictProbability * changeFactor;

      // Determine severity
      let severity: ConflictPrediction['severity'] = 'low';
      if (probability > 0.7) {
        severity = 'high';
      } else if (probability > 0.4) {
        severity = 'medium';
      }

      predictions.push({
        file,
        probability,
        sourceChanges,
        targetChanges,
        overlappingLines: sourceChanges > 0 && targetChanges > 0,
        conflictType,
        severity,
      });
    }

    return predictions.sort((a, b) => b.probability - a.probability);
  }

  /**
   * Count total changes to a file across commits.
   */
  private countFileChanges(file: string, commits: GitCommit[]): number {
    let total = 0;
    for (const commit of commits) {
      const fileChange = commit.files.find(
        f => f.path === file || f.oldPath === file
      );
      if (fileChange) {
        total += fileChange.additions + fileChange.deletions;
      }
    }
    return total;
  }

  /**
   * Get file status from commits.
   */
  private getFileStatus(file: string, commits: GitCommit[]): string {
    for (const commit of commits) {
      const fileChange = commit.files.find(
        f => f.path === file || f.oldPath === file
      );
      if (fileChange) {
        return fileChange.status;
      }
    }
    return 'M';
  }

  /**
   * Generate a reconciliation plan.
   */
  generatePlan(
    pair: BranchPair,
    predictions: ConflictPrediction[]
  ): ReconciliationPlan {
    const steps: ReconciliationStep[] = [];
    const risks: string[] = [];
    const prerequisites: string[] = [];

    // Common prerequisites
    prerequisites.push('Ensure local repository is up to date');
    prerequisites.push('Create backup branch');

    // Strategy-specific steps
    switch (pair.recommendation) {
      case 'fast-forward':
        steps.push({
          action: 'fast-forward',
          command: `git merge --ff-only ${pair.source.name}`,
          description: 'Fast-forward merge (no conflicts possible)',
          automated: true,
        });
        break;

      case 'clean-merge':
        steps.push({
          action: 'merge',
          command: `git merge ${pair.source.name}`,
          description: 'Clean merge with no expected conflicts',
          automated: true,
        });
        break;

      case 'rebase':
        steps.push({
          action: 'rebase',
          command: `git rebase ${pair.target.name}`,
          description: 'Rebase source branch onto target',
          automated: false,
        });
        risks.push('May require conflict resolution during rebase');
        break;

      case 'squash-merge':
        steps.push({
          action: 'squash',
          command: `git merge --squash ${pair.source.name}`,
          description: 'Squash all commits into one',
          automated: true,
        });
        steps.push({
          action: 'commit',
          command: 'git commit -m "Squash merge from branch"',
          description: 'Commit the squashed changes',
          automated: false,
        });
        break;

      case 'split-pr':
        steps.push({
          action: 'analyze',
          description: 'Identify logical groupings of changes',
          automated: false,
        });
        steps.push({
          action: 'cherry-pick',
          description: 'Cherry-pick commits into separate branches',
          automated: false,
        });
        risks.push('Requires manual splitting of changes');
        risks.push('May take multiple PRs to complete');
        break;

      case 'manual-review':
        steps.push({
          action: 'review',
          description: 'Carefully review all conflicting files',
          automated: false,
        });
        for (const pred of predictions.filter(p => p.severity === 'high')) {
          steps.push({
            action: 'resolve',
            description: `Resolve conflicts in ${pred.file}`,
            automated: false,
          });
        }
        risks.push('High probability of merge conflicts');
        risks.push('Requires careful manual resolution');
        break;
    }

    // Calculate effort
    let effort = 1;
    if (pair.recommendation === 'manual-review') {
      effort = Math.min(10, 3 + predictions.filter(p => p.severity === 'high').length);
    } else if (pair.recommendation === 'split-pr') {
      effort = Math.min(10, 5 + Math.floor(pair.source.filesChanged.size / 20));
    } else if (pair.recommendation === 'rebase') {
      effort = Math.min(10, 2 + Math.floor(pair.source.commitCount / 10));
    }

    return {
      pair,
      strategy: pair.recommendation,
      steps,
      estimatedEffort: effort,
      risks,
      prerequisites,
    };
  }

  /**
   * Find all branches that could conflict with a target.
   */
  findPotentialConflicts(
    branches: Map<string, { commits: GitCommit[]; ancestor: string }>,
    targetBranch: string
  ): BranchPair[] {
    const pairs: BranchPair[] = [];
    const target = branches.get(targetBranch);

    if (!target) {
      return pairs;
    }

    for (const [branchName, branchData] of branches) {
      if (branchName === targetBranch) {
        continue;
      }

      const pair = this.analyzePair(
        branchData.commits,
        target.commits,
        branchName,
        targetBranch,
        branchData.ancestor
      );

      if (pair.conflictProbability > this.config.conflictThreshold) {
        pairs.push(pair);
      }
    }

    return pairs.sort((a, b) => b.conflictProbability - a.conflictProbability);
  }
}

/**
 * Quick function to analyze branch pair.
 */
export function analyzeBranches(
  sourceCommits: GitCommit[],
  targetCommits: GitCommit[],
  sourceBranch: string,
  targetBranch: string,
  commonAncestor: string,
  config?: Partial<BranchReconcilerConfig>
): BranchPair {
  const reconciler = new BranchReconciler(config);
  return reconciler.analyzePair(
    sourceCommits,
    targetCommits,
    sourceBranch,
    targetBranch,
    commonAncestor
  );
}
