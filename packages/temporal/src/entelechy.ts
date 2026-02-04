/**
 * EntelechyExtraction: Signal vs Noise Identification
 *
 * Step 21 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Core Heuristic: "Code that survives 10 refactors is Signal;
 *                  code that vanishes is Noise."
 *
 * Entelechy (Aristotle): The actualization of potential.
 * Code that persists has actualized its potential - it serves a purpose.
 *
 * This algorithm identifies:
 * - Signal: Code that has survived multiple refactors
 * - Noise: Code that keeps changing or getting deleted
 * - Entelechy Score: Measure of code's "actualized potential"
 */

import type { GitCommit, GitFileChange } from './git-walker';
import type { TemporalSignal } from './temporal-signal';

export type EntelechyClass = 'signal' | 'emerging' | 'unstable' | 'noise';

export interface EntelechyResult {
  path: string;
  class: EntelechyClass;
  score: number;  // 0-100, higher = more "actualized"
  survivedRefactors: number;
  totalRefactors: number;
  survivalRate: number;
  lineageDepth: number;
  indicators: EntelechyIndicator[];
  recommendation: string;
}

export interface EntelechyIndicator {
  name: string;
  value: number;
  weight: number;
  contribution: number;  // Weighted contribution to score
}

export interface CodeLineage {
  /** Original creation commit */
  origin: string;
  /** All commits that touched this code */
  touchPoints: TouchPoint[];
  /** Current state (exists or deleted) */
  exists: boolean;
  /** How many refactors this code survived */
  refactorsSurvived: number;
  /** Depth of lineage (transformations) */
  depth: number;
}

export interface TouchPoint {
  commitHash: string;
  date: Date;
  author: string;
  changeType: 'create' | 'modify' | 'refactor' | 'move' | 'delete';
  linesAffected: number;
  survived: boolean;
}

export interface EntelechyConfig {
  /** Minimum refactors to be considered "signal" */
  signalThreshold: number;
  /** Minimum age in days for evaluation */
  minAgeForEval: number;
  /** Weight for survival rate */
  survivalWeight: number;
  /** Weight for age stability */
  ageWeight: number;
  /** Weight for author diversity */
  authorWeight: number;
  /** Weight for change frequency */
  frequencyWeight: number;
}

const DEFAULT_CONFIG: EntelechyConfig = {
  signalThreshold: 10,
  minAgeForEval: 30,
  survivalWeight: 0.4,
  ageWeight: 0.2,
  authorWeight: 0.2,
  frequencyWeight: 0.2,
};

/**
 * EntelechyExtractor: Identify signal vs noise in code.
 */
export class EntelechyExtractor {
  private config: EntelechyConfig;

  constructor(config?: Partial<EntelechyConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze a file for its entelechy (actualized potential).
   */
  analyze(
    signal: TemporalSignal,
    commits: GitCommit[]
  ): EntelechyResult {
    // Too new for meaningful analysis
    if (signal.ageInDays < this.config.minAgeForEval) {
      return this.createResult(
        signal.path,
        'emerging',
        50,
        0,
        0,
        0,
        0,
        [],
        'File too new - potential not yet actualized.'
      );
    }

    // Count refactors and survival
    const refactorAnalysis = this.analyzeRefactors(signal.path, commits);
    const { survivedRefactors, totalRefactors, survivalRate, lineageDepth } = refactorAnalysis;

    // Build indicators
    const indicators = this.buildIndicators(signal, refactorAnalysis);

    // Calculate entelechy score
    const score = this.calculateScore(indicators);

    // Classify
    const entelechyClass = this.classify(score, survivedRefactors, survivalRate);

    // Generate recommendation
    const recommendation = this.generateRecommendation(entelechyClass, indicators);

    return this.createResult(
      signal.path,
      entelechyClass,
      score,
      survivedRefactors,
      totalRefactors,
      survivalRate,
      lineageDepth,
      indicators,
      recommendation
    );
  }

  /**
   * Analyze refactor patterns in commit history.
   */
  private analyzeRefactors(
    path: string,
    commits: GitCommit[]
  ): {
    survivedRefactors: number;
    totalRefactors: number;
    survivalRate: number;
    lineageDepth: number;
    touchPoints: TouchPoint[];
  } {
    const touchPoints: TouchPoint[] = [];
    let totalRefactors = 0;
    let survivedRefactors = 0;

    // Keywords that indicate refactoring
    const refactorKeywords = [
      'refactor', 'restructure', 'reorganize', 'simplify',
      'extract', 'rename', 'move', 'split', 'merge', 'cleanup'
    ];

    for (const commit of commits) {
      const msg = commit.message.toLowerCase();
      const isRefactor = refactorKeywords.some(kw => msg.includes(kw));

      if (isRefactor) {
        totalRefactors++;
      }

      const fileChange = commit.files.find(f =>
        f.path === path || f.oldPath === path
      );

      if (fileChange) {
        const changeType = this.determineChangeType(fileChange, msg);

        touchPoints.push({
          commitHash: commit.hash,
          date: commit.date,
          author: commit.author,
          changeType,
          linesAffected: fileChange.additions + fileChange.deletions,
          survived: changeType !== 'delete',
        });

        // If file still exists after a refactor, it survived
        if (isRefactor && changeType !== 'delete') {
          survivedRefactors++;
        }
      }
    }

    const survivalRate = totalRefactors > 0
      ? survivedRefactors / totalRefactors
      : 1;  // No refactors = perfect survival

    // Lineage depth = number of significant transformations
    const lineageDepth = touchPoints.filter(tp =>
      tp.changeType === 'refactor' || tp.changeType === 'move'
    ).length;

    return {
      survivedRefactors,
      totalRefactors,
      survivalRate,
      lineageDepth,
      touchPoints,
    };
  }

  /**
   * Determine the type of change from file change and message.
   */
  private determineChangeType(
    change: GitFileChange,
    message: string
  ): TouchPoint['changeType'] {
    if (change.status === 'D') return 'delete';
    if (change.status === 'A') return 'create';
    if (change.status === 'R') return 'move';

    if (message.includes('refactor') || message.includes('restructure')) {
      return 'refactor';
    }

    return 'modify';
  }

  /**
   * Build entelechy indicators.
   */
  private buildIndicators(
    signal: TemporalSignal,
    refactorAnalysis: {
      survivedRefactors: number;
      totalRefactors: number;
      survivalRate: number;
      lineageDepth: number;
    }
  ): EntelechyIndicator[] {
    const indicators: EntelechyIndicator[] = [];

    // Survival Rate indicator
    const survivalScore = refactorAnalysis.survivalRate * 100;
    indicators.push({
      name: 'Refactor Survival',
      value: survivalScore,
      weight: this.config.survivalWeight,
      contribution: survivalScore * this.config.survivalWeight,
    });

    // Age Stability indicator (older stable code = more signal)
    const ageStability = Math.min(100, (signal.ageInDays / 365) * 100);
    const ageScore = ageStability * (1 - signal.churnRate / 10);  // Penalize high churn
    indicators.push({
      name: 'Age Stability',
      value: Math.max(0, ageScore),
      weight: this.config.ageWeight,
      contribution: Math.max(0, ageScore) * this.config.ageWeight,
    });

    // Author Diversity indicator (multiple authors = reviewed code)
    const authorScore = Math.min(100, signal.authorCount * 20);
    indicators.push({
      name: 'Author Diversity',
      value: authorScore,
      weight: this.config.authorWeight,
      contribution: authorScore * this.config.authorWeight,
    });

    // Change Frequency indicator (lower frequency = more stable)
    const freqScore = Math.max(0, 100 - signal.commitFreq * 10);
    indicators.push({
      name: 'Stability (Low Churn)',
      value: freqScore,
      weight: this.config.frequencyWeight,
      contribution: freqScore * this.config.frequencyWeight,
    });

    // Lineage Depth indicator (survived transformations)
    const lineageScore = Math.min(100, refactorAnalysis.lineageDepth * 15);
    indicators.push({
      name: 'Lineage Depth',
      value: lineageScore,
      weight: 0.1,
      contribution: lineageScore * 0.1,
    });

    return indicators;
  }

  /**
   * Calculate entelechy score from indicators.
   */
  private calculateScore(indicators: EntelechyIndicator[]): number {
    const totalContribution = indicators.reduce(
      (sum, ind) => sum + ind.contribution,
      0
    );
    const totalWeight = indicators.reduce(
      (sum, ind) => sum + ind.weight,
      0
    );
    return totalWeight > 0 ? totalContribution / totalWeight : 0;
  }

  /**
   * Classify based on score and survival metrics.
   */
  private classify(
    score: number,
    survivedRefactors: number,
    survivalRate: number
  ): EntelechyClass {
    // Strong signal: High score AND survived many refactors
    if (score >= 70 && survivedRefactors >= this.config.signalThreshold) {
      return 'signal';
    }

    // Clear noise: Low survival rate
    if (survivalRate < 0.3 && survivedRefactors > 3) {
      return 'noise';
    }

    // Unstable: Moderate score but high churn
    if (score < 40 || survivalRate < 0.5) {
      return 'unstable';
    }

    // Emerging: Shows potential but not proven
    return 'emerging';
  }

  /**
   * Generate recommendation based on classification.
   */
  private generateRecommendation(
    cls: EntelechyClass,
    indicators: EntelechyIndicator[]
  ): string {
    switch (cls) {
      case 'signal':
        return 'SIGNAL: This code has proven its value through multiple refactors. ' +
               'Treat as trusted kernel code (Ring 0-1).';

      case 'emerging':
        const weakest = [...indicators].sort((a, b) => a.value - b.value)[0];
        return `EMERGING: Code shows potential but needs more validation. ` +
               `Weakest indicator: ${weakest?.name}. Consider Ring 2.`;

      case 'unstable':
        return 'UNSTABLE: Code changes frequently without stabilizing. ' +
               'Review for architectural issues. Ring 3 recommended.';

      case 'noise':
        return 'NOISE: Code does not survive refactors. ' +
               'Consider removal or complete rewrite. Do not trust.';
    }
  }

  /**
   * Create result object.
   */
  private createResult(
    path: string,
    cls: EntelechyClass,
    score: number,
    survivedRefactors: number,
    totalRefactors: number,
    survivalRate: number,
    lineageDepth: number,
    indicators: EntelechyIndicator[],
    recommendation: string
  ): EntelechyResult {
    return {
      path,
      class: cls,
      score,
      survivedRefactors,
      totalRefactors,
      survivalRate,
      lineageDepth,
      indicators,
      recommendation,
    };
  }
}

/**
 * Quick function to extract entelechy for a file.
 */
export function extractEntelechy(
  signal: TemporalSignal,
  commits: GitCommit[],
  config?: Partial<EntelechyConfig>
): EntelechyResult {
  const extractor = new EntelechyExtractor(config);
  return extractor.analyze(signal, commits);
}

/**
 * Batch extract entelechy for multiple files.
 */
export function extractEntelechyBatch(
  files: Array<{ signal: TemporalSignal; commits: GitCommit[] }>,
  config?: Partial<EntelechyConfig>
): EntelechyResult[] {
  const extractor = new EntelechyExtractor(config);
  return files.map(f => extractor.analyze(f.signal, f.commits));
}

/**
 * Get class description for documentation.
 */
export function getEntelechyClassDescription(cls: EntelechyClass): string {
  const descriptions: Record<EntelechyClass, string> = {
    signal: 'Code that has actualized its potential. Survived multiple refactors.',
    emerging: 'Code showing potential but not yet proven through time.',
    unstable: 'Code that changes frequently without settling into a stable form.',
    noise: 'Code that does not survive refactors. Likely temporary or misguided.',
  };
  return descriptions[cls];
}
