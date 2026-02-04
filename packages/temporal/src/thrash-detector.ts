/**
 * ThrashDetector: Identify High-Churn Low-Value Code
 *
 * Step 19 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Detects files with high churn but low semantic change:
 * - Files that change constantly but don't improve
 * - "Spinning wheels" in the codebase
 * - Code that needs attention or deletion
 *
 * Uses combination of:
 * - Git history analysis (churn rate)
 * - LSA semantic distance (actual change)
 * - Linter status (quality improvement)
 */

import type { TemporalSignal } from './temporal-signal';
import type { GitCommit } from './git-walker';

export type ThrashLevel = 'none' | 'low' | 'medium' | 'high' | 'critical';

export interface ThrashReport {
  path: string;
  level: ThrashLevel;
  score: number;  // 0-100, higher = more thrash
  indicators: ThrashIndicator[];
  recommendation: string;
  details: ThrashDetails;
}

export interface ThrashIndicator {
  name: string;
  value: number;
  threshold: number;
  triggered: boolean;
  weight: number;
}

export interface ThrashDetails {
  /** Total commits to this file */
  totalCommits: number;
  /** Commits in last 30 days */
  recentCommits: number;
  /** Churn rate (changes / size) */
  churnRate: number;
  /** Semantic change ratio (actual change / total changes) */
  semanticChangeRatio?: number;
  /** Revert count (commits that were later reverted) */
  revertCount: number;
  /** Fix-fix count (fixes to previous fixes) */
  fixFixCount: number;
  /** Authors involved */
  authorCount: number;
  /** Days since creation */
  ageInDays: number;
  /** Average days between changes */
  avgDaysBetweenChanges: number;
}

export interface ThrashDetectorConfig {
  /** High churn threshold (commits per month) */
  highChurnThreshold: number;
  /** Critical churn threshold */
  criticalChurnThreshold: number;
  /** Minimum age to consider (days) */
  minAgeForAnalysis: number;
  /** Weight for churn indicator */
  churnWeight: number;
  /** Weight for revert indicator */
  revertWeight: number;
  /** Weight for fix-fix indicator */
  fixFixWeight: number;
  /** Weight for low semantic change */
  semanticWeight: number;
}

const DEFAULT_CONFIG: ThrashDetectorConfig = {
  highChurnThreshold: 10,     // 10+ commits/month = high
  criticalChurnThreshold: 20, // 20+ commits/month = critical
  minAgeForAnalysis: 14,      // At least 2 weeks old
  churnWeight: 0.4,
  revertWeight: 0.2,
  fixFixWeight: 0.2,
  semanticWeight: 0.2,
};

/**
 * ThrashDetector: Analyze files for thrash patterns.
 */
export class ThrashDetector {
  private config: ThrashDetectorConfig;

  constructor(config?: Partial<ThrashDetectorConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Analyze a file for thrash patterns.
   */
  analyze(
    signal: TemporalSignal,
    commits: GitCommit[],
    semanticChangeRatio?: number
  ): ThrashReport {
    // Too new for analysis
    if (signal.ageInDays < this.config.minAgeForAnalysis) {
      return this.createReport(signal.path, 'none', 0, [], 'File too new for thrash analysis');
    }

    const details = this.calculateDetails(signal, commits);
    const indicators = this.evaluateIndicators(details, semanticChangeRatio);
    const score = this.calculateScore(indicators);
    const level = this.scoreToLevel(score);
    const recommendation = this.generateRecommendation(level, indicators, details);

    return this.createReport(signal.path, level, score, indicators, recommendation, details);
  }

  /**
   * Batch analyze multiple files.
   */
  analyzeMultiple(
    files: Array<{
      signal: TemporalSignal;
      commits: GitCommit[];
      semanticChangeRatio?: number;
    }>
  ): ThrashReport[] {
    return files.map(f => this.analyze(f.signal, f.commits, f.semanticChangeRatio));
  }

  /**
   * Find the worst offenders in a codebase.
   */
  findHotspots(
    reports: ThrashReport[],
    limit: number = 10
  ): ThrashReport[] {
    return reports
      .filter(r => r.level !== 'none')
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  /**
   * Calculate detailed thrash metrics.
   */
  private calculateDetails(
    signal: TemporalSignal,
    commits: GitCommit[]
  ): ThrashDetails {
    const now = new Date();
    const thirtyDaysAgo = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000);

    // Count recent commits
    const recentCommits = commits.filter(c => c.date >= thirtyDaysAgo).length;

    // Detect reverts
    const revertCount = commits.filter(c =>
      c.message.toLowerCase().includes('revert') ||
      c.message.toLowerCase().includes('undo')
    ).length;

    // Detect fix-fix pattern (commits that fix previous fixes)
    const fixFixCount = this.detectFixFixPattern(commits);

    // Calculate average days between changes
    let avgDaysBetweenChanges = 0;
    if (commits.length > 1) {
      const sorted = [...commits].sort((a, b) => a.date.getTime() - b.date.getTime());
      let totalDays = 0;
      for (let i = 1; i < sorted.length; i++) {
        totalDays += (sorted[i].date.getTime() - sorted[i - 1].date.getTime()) / (1000 * 60 * 60 * 24);
      }
      avgDaysBetweenChanges = totalDays / (commits.length - 1);
    }

    return {
      totalCommits: commits.length,
      recentCommits,
      churnRate: signal.churnRate,
      revertCount,
      fixFixCount,
      authorCount: signal.authorCount,
      ageInDays: signal.ageInDays,
      avgDaysBetweenChanges,
    };
  }

  /**
   * Detect fix-fix pattern in commit history.
   */
  private detectFixFixPattern(commits: GitCommit[]): number {
    const fixKeywords = ['fix', 'bug', 'issue', 'patch', 'hotfix', 'correct'];
    let fixFixCount = 0;
    let lastWasFix = false;

    for (const commit of commits) {
      const msg = commit.message.toLowerCase();
      const isFix = fixKeywords.some(kw => msg.includes(kw));

      if (isFix && lastWasFix) {
        fixFixCount++;
      }
      lastWasFix = isFix;
    }

    return fixFixCount;
  }

  /**
   * Evaluate thrash indicators.
   */
  private evaluateIndicators(
    details: ThrashDetails,
    semanticChangeRatio?: number
  ): ThrashIndicator[] {
    const indicators: ThrashIndicator[] = [];
    const monthsActive = Math.max(1, details.ageInDays / 30);
    const commitsPerMonth = details.totalCommits / monthsActive;

    // High churn indicator
    indicators.push({
      name: 'High Churn',
      value: commitsPerMonth,
      threshold: this.config.highChurnThreshold,
      triggered: commitsPerMonth >= this.config.highChurnThreshold,
      weight: this.config.churnWeight,
    });

    // Revert indicator
    const revertRatio = details.revertCount / Math.max(1, details.totalCommits);
    indicators.push({
      name: 'Frequent Reverts',
      value: revertRatio * 100,
      threshold: 10,  // 10% revert rate
      triggered: revertRatio >= 0.1,
      weight: this.config.revertWeight,
    });

    // Fix-fix indicator
    const fixFixRatio = details.fixFixCount / Math.max(1, details.totalCommits);
    indicators.push({
      name: 'Fix-Fix Pattern',
      value: fixFixRatio * 100,
      threshold: 15,  // 15% fix-fix rate
      triggered: fixFixRatio >= 0.15,
      weight: this.config.fixFixWeight,
    });

    // Low semantic change indicator (if available)
    if (semanticChangeRatio !== undefined) {
      indicators.push({
        name: 'Low Semantic Change',
        value: (1 - semanticChangeRatio) * 100,
        threshold: 70,  // Less than 30% actual semantic change
        triggered: semanticChangeRatio < 0.3,
        weight: this.config.semanticWeight,
      });
    }

    // Rapid changes indicator
    indicators.push({
      name: 'Rapid Changes',
      value: details.avgDaysBetweenChanges,
      threshold: 2,  // Changes more than every 2 days
      triggered: details.avgDaysBetweenChanges < 2 && details.totalCommits > 5,
      weight: 0.15,
    });

    // Recent burst indicator
    const recentBurstRatio = details.recentCommits / Math.max(1, details.totalCommits);
    indicators.push({
      name: 'Recent Burst',
      value: recentBurstRatio * 100,
      threshold: 50,  // 50% of all commits in last 30 days
      triggered: recentBurstRatio >= 0.5 && details.recentCommits >= 5,
      weight: 0.1,
    });

    return indicators;
  }

  /**
   * Calculate thrash score from indicators.
   */
  private calculateScore(indicators: ThrashIndicator[]): number {
    let totalWeight = 0;
    let weightedScore = 0;

    for (const ind of indicators) {
      totalWeight += ind.weight;
      if (ind.triggered) {
        // Score based on how much the threshold is exceeded
        const excess = ind.value / ind.threshold;
        const score = Math.min(100, excess * 50);  // Cap at 100
        weightedScore += score * ind.weight;
      }
    }

    return totalWeight > 0 ? weightedScore / totalWeight : 0;
  }

  /**
   * Convert score to thrash level.
   */
  private scoreToLevel(score: number): ThrashLevel {
    if (score >= 80) return 'critical';
    if (score >= 60) return 'high';
    if (score >= 40) return 'medium';
    if (score >= 20) return 'low';
    return 'none';
  }

  /**
   * Generate recommendation based on analysis.
   */
  private generateRecommendation(
    level: ThrashLevel,
    indicators: ThrashIndicator[],
    details: ThrashDetails
  ): string {
    if (level === 'none') {
      return 'No thrash detected. File appears stable.';
    }

    const triggered = indicators.filter(i => i.triggered).map(i => i.name);

    if (level === 'critical') {
      if (triggered.includes('Frequent Reverts')) {
        return 'CRITICAL: File is unstable. Consider reverting to last known good state or complete rewrite.';
      }
      if (triggered.includes('Fix-Fix Pattern')) {
        return 'CRITICAL: File is in a fix loop. Root cause analysis needed before more changes.';
      }
      return 'CRITICAL: Excessive churn detected. Freeze changes and review architecture.';
    }

    if (level === 'high') {
      if (triggered.includes('Low Semantic Change')) {
        return 'HIGH: Many changes but little semantic improvement. Consider whether changes are necessary.';
      }
      return 'HIGH: Significant churn. Review change process and consider consolidating changes.';
    }

    if (level === 'medium') {
      if (details.authorCount === 1) {
        return 'MEDIUM: Single author with moderate churn. Consider code review requirement.';
      }
      return 'MEDIUM: Moderate churn detected. Monitor for improvement.';
    }

    return 'LOW: Minor thrash indicators. Continue monitoring.';
  }

  /**
   * Create a thrash report.
   */
  private createReport(
    path: string,
    level: ThrashLevel,
    score: number,
    indicators: ThrashIndicator[],
    recommendation: string,
    details?: ThrashDetails
  ): ThrashReport {
    return {
      path,
      level,
      score,
      indicators,
      recommendation,
      details: details || {
        totalCommits: 0,
        recentCommits: 0,
        churnRate: 0,
        revertCount: 0,
        fixFixCount: 0,
        authorCount: 0,
        ageInDays: 0,
        avgDaysBetweenChanges: 0,
      },
    };
  }
}

/**
 * Quick function to detect thrash in a file.
 */
export function detectThrash(
  signal: TemporalSignal,
  commits: GitCommit[],
  config?: Partial<ThrashDetectorConfig>
): ThrashReport {
  const detector = new ThrashDetector(config);
  return detector.analyze(signal, commits);
}
