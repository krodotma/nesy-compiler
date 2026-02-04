/**
 * AleatoricCheck: Identify Irreducible Uncertainty
 *
 * Step 23 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * Core principle from the Denoise Loop:
 * - Aleatoric uncertainty = irreducible randomness (world is stochastic)
 * - Epistemic uncertainty = reducible ignorance (we don't know enough)
 *
 * Aleatoric signals should be ACCEPTED (maintain diversity),
 * NOT "fixed" or "denoised". This module identifies which code
 * patterns represent healthy variance vs problematic instability.
 *
 * Key insight: Some thrash is actually GOOD - it represents
 * exploration, experimentation, and healthy iteration.
 */

import type { TemporalSignal } from './temporal-signal';
import type { ThrashReport } from './thrash-detector';
import type { EntelechyResult } from './entelechy';
import type { GitCommit } from './git-walker';

export type UncertaintyType = 'aleatoric' | 'epistemic' | 'mixed';

export interface AleatoricAssessment {
  path: string;
  uncertaintyType: UncertaintyType;
  aleatoricScore: number;  // 0-1, higher = more aleatoric (irreducible)
  epistemicScore: number;  // 0-1, higher = more epistemic (reducible)
  confidence: number;      // How confident we are in this assessment
  indicators: AleatoricIndicator[];
  recommendation: AleatoricRecommendation;
  details: AleatoricDetails;
}

export interface AleatoricIndicator {
  name: string;
  value: number;
  isAleatoric: boolean;  // true = suggests aleatoric, false = suggests epistemic
  weight: number;
  explanation: string;
}

export type AleatoricRecommendation =
  | 'accept-variance'      // Aleatoric: healthy variance, don't try to "fix"
  | 'investigate-root'     // Epistemic: find and fix root cause
  | 'monitor-pattern'      // Mixed: watch for trends before acting
  | 'stabilize-interface'  // High-churn boundary: stabilize the API
  | 'split-concerns';      // Mixed concerns: separate stable from unstable

export interface AleatoricDetails {
  /** Is this file at a system boundary? */
  isBoundaryFile: boolean;
  /** Do changes follow external triggers (API changes, dependencies)? */
  externallyDriven: boolean;
  /** Is the churn from different authors with different goals? */
  multiPurposeChurn: boolean;
  /** Does the file have stable core with volatile edges? */
  hasStableCore: boolean;
  /** Is this exploratory code (prototypes, experiments)? */
  isExploratory: boolean;
  /** Time pattern: regular vs burst changes */
  changePattern: 'regular' | 'burst' | 'declining' | 'growing';
}

export interface AleatoricCheckConfig {
  /** Threshold for classifying as aleatoric */
  aleatoricThreshold: number;
  /** Threshold for classifying as epistemic */
  epistemicThreshold: number;
  /** Minimum commits for meaningful analysis */
  minCommitsForAnalysis: number;
  /** Weight for author diversity indicator */
  authorDiversityWeight: number;
  /** Weight for external trigger indicator */
  externalTriggerWeight: number;
  /** Weight for time pattern indicator */
  timePatternWeight: number;
}

const DEFAULT_CONFIG: AleatoricCheckConfig = {
  aleatoricThreshold: 0.6,
  epistemicThreshold: 0.6,
  minCommitsForAnalysis: 5,
  authorDiversityWeight: 0.25,
  externalTriggerWeight: 0.25,
  timePatternWeight: 0.25,
};

/**
 * Keywords that suggest external drivers (aleatoric).
 */
const EXTERNAL_KEYWORDS = [
  'api', 'dependency', 'upgrade', 'migrate', 'external',
  'integration', 'protocol', 'schema', 'version', 'compat'
];

/**
 * Keywords that suggest internal issues (epistemic).
 */
const INTERNAL_KEYWORDS = [
  'fix', 'bug', 'broken', 'wrong', 'incorrect', 'mistake',
  'oops', 'revert', 'undo', 'typo', 'forgot'
];

/**
 * Keywords that suggest exploration (aleatoric).
 */
const EXPLORATION_KEYWORDS = [
  'try', 'experiment', 'test', 'prototype', 'poc', 'wip',
  'draft', 'explore', 'attempt', 'spike'
];

/**
 * AleatoricChecker: Distinguish irreducible from reducible uncertainty.
 */
export class AleatoricChecker {
  private config: AleatoricCheckConfig;

  constructor(config?: Partial<AleatoricCheckConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Assess uncertainty type for a file.
   */
  assess(
    signal: TemporalSignal,
    commits: GitCommit[],
    thrashReport?: ThrashReport,
    entelechyResult?: EntelechyResult
  ): AleatoricAssessment {
    // Not enough data
    if (commits.length < this.config.minCommitsForAnalysis) {
      return this.createAssessment(
        signal.path,
        'mixed',
        0.5,
        0.5,
        0.3,
        [],
        'monitor-pattern',
        this.defaultDetails()
      );
    }

    // Build indicators
    const indicators = this.buildIndicators(signal, commits, thrashReport, entelechyResult);

    // Calculate scores
    const { aleatoricScore, epistemicScore } = this.calculateScores(indicators);

    // Determine uncertainty type
    const uncertaintyType = this.classifyUncertainty(aleatoricScore, epistemicScore);

    // Calculate confidence
    const confidence = this.calculateConfidence(indicators, commits.length);

    // Build details
    const details = this.buildDetails(signal, commits, indicators);

    // Generate recommendation
    const recommendation = this.generateRecommendation(
      uncertaintyType,
      aleatoricScore,
      epistemicScore,
      details
    );

    return this.createAssessment(
      signal.path,
      uncertaintyType,
      aleatoricScore,
      epistemicScore,
      confidence,
      indicators,
      recommendation,
      details
    );
  }

  /**
   * Build assessment indicators.
   */
  private buildIndicators(
    signal: TemporalSignal,
    commits: GitCommit[],
    thrashReport?: ThrashReport,
    entelechyResult?: EntelechyResult
  ): AleatoricIndicator[] {
    const indicators: AleatoricIndicator[] = [];

    // 1. Author diversity (high diversity = aleatoric)
    const authorDiversity = Math.min(1, signal.authorCount / 5);
    indicators.push({
      name: 'Author Diversity',
      value: authorDiversity,
      isAleatoric: authorDiversity > 0.5,
      weight: this.config.authorDiversityWeight,
      explanation: authorDiversity > 0.5
        ? 'Multiple authors suggest different use cases (healthy variance)'
        : 'Single author suggests focused development (epistemic)',
    });

    // 2. External triggers (external = aleatoric)
    const externalRatio = this.calculateKeywordRatio(commits, EXTERNAL_KEYWORDS);
    indicators.push({
      name: 'External Triggers',
      value: externalRatio,
      isAleatoric: externalRatio > 0.3,
      weight: this.config.externalTriggerWeight,
      explanation: externalRatio > 0.3
        ? 'Changes driven by external factors (adapting to reality)'
        : 'Changes are internally motivated',
    });

    // 3. Fix patterns (fixes = epistemic)
    const fixRatio = this.calculateKeywordRatio(commits, INTERNAL_KEYWORDS);
    indicators.push({
      name: 'Fix Patterns',
      value: fixRatio,
      isAleatoric: fixRatio < 0.2,
      weight: 0.2,
      explanation: fixRatio > 0.3
        ? 'High fix rate suggests bugs (epistemic - learn from mistakes)'
        : 'Low fix rate suggests normal iteration',
    });

    // 4. Exploration patterns (exploration = aleatoric)
    const explorationRatio = this.calculateKeywordRatio(commits, EXPLORATION_KEYWORDS);
    indicators.push({
      name: 'Exploration',
      value: explorationRatio,
      isAleatoric: explorationRatio > 0.2,
      weight: 0.15,
      explanation: explorationRatio > 0.2
        ? 'Exploratory work (healthy experimentation)'
        : 'Production code (should be stable)',
    });

    // 5. Change pattern regularity (regular = aleatoric, burst = epistemic)
    const changePattern = this.analyzeChangePattern(commits);
    const patternScore = changePattern === 'regular' ? 0.8 :
                        changePattern === 'burst' ? 0.2 :
                        changePattern === 'growing' ? 0.5 : 0.6;
    indicators.push({
      name: 'Change Pattern',
      value: patternScore,
      isAleatoric: patternScore > 0.5,
      weight: this.config.timePatternWeight,
      explanation: `${changePattern} change pattern - ${
        patternScore > 0.5 ? 'normal variance' : 'may indicate instability'
      }`,
    });

    // 6. Entelechy signal (high entelechy = stable core exists = aleatoric ok)
    if (entelechyResult) {
      const entelechyScore = entelechyResult.score / 100;
      indicators.push({
        name: 'Code Maturity',
        value: entelechyScore,
        isAleatoric: entelechyScore > 0.5,
        weight: 0.15,
        explanation: entelechyScore > 0.5
          ? 'Mature code with stable core (variance is refinement)'
          : 'Immature code (churn may be finding direction)',
      });
    }

    // 7. Thrash severity (high thrash + no progress = epistemic)
    if (thrashReport && thrashReport.level !== 'none') {
      const thrashScore = 1 - (thrashReport.score / 100);
      indicators.push({
        name: 'Thrash Pattern',
        value: thrashScore,
        isAleatoric: thrashScore > 0.5,
        weight: 0.15,
        explanation: thrashScore < 0.5
          ? 'High thrash without progress (spinning wheels)'
          : 'Changes show forward motion',
      });
    }

    return indicators;
  }

  /**
   * Calculate keyword ratio in commit messages.
   */
  private calculateKeywordRatio(commits: GitCommit[], keywords: string[]): number {
    if (commits.length === 0) return 0;

    let matches = 0;
    for (const commit of commits) {
      const msg = commit.message.toLowerCase();
      if (keywords.some(kw => msg.includes(kw))) {
        matches++;
      }
    }

    return matches / commits.length;
  }

  /**
   * Analyze the temporal pattern of changes.
   */
  private analyzeChangePattern(commits: GitCommit[]): AleatoricDetails['changePattern'] {
    if (commits.length < 3) return 'regular';

    const sorted = [...commits].sort((a, b) => a.date.getTime() - b.date.getTime());
    const gaps: number[] = [];

    for (let i = 1; i < sorted.length; i++) {
      gaps.push(sorted[i].date.getTime() - sorted[i - 1].date.getTime());
    }

    const avgGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    const variance = gaps.reduce((sum, gap) => sum + Math.pow(gap - avgGap, 2), 0) / gaps.length;
    const stdDev = Math.sqrt(variance);
    const cv = stdDev / avgGap; // Coefficient of variation

    // Check for trend
    const recentGaps = gaps.slice(-3);
    const earlyGaps = gaps.slice(0, 3);
    const recentAvg = recentGaps.reduce((a, b) => a + b, 0) / recentGaps.length;
    const earlyAvg = earlyGaps.reduce((a, b) => a + b, 0) / earlyGaps.length;

    if (recentAvg < earlyAvg * 0.5) {
      return 'growing'; // Changes accelerating
    }
    if (recentAvg > earlyAvg * 2) {
      return 'declining'; // Changes slowing
    }
    if (cv > 1.5) {
      return 'burst'; // Irregular bursts
    }

    return 'regular';
  }

  /**
   * Calculate aleatoric and epistemic scores.
   */
  private calculateScores(indicators: AleatoricIndicator[]): {
    aleatoricScore: number;
    epistemicScore: number;
  } {
    let aleatoricSum = 0;
    let epistemicSum = 0;
    let totalWeight = 0;

    for (const ind of indicators) {
      totalWeight += ind.weight;
      if (ind.isAleatoric) {
        aleatoricSum += ind.value * ind.weight;
      } else {
        epistemicSum += ind.value * ind.weight;
      }
    }

    return {
      aleatoricScore: totalWeight > 0 ? aleatoricSum / totalWeight : 0.5,
      epistemicScore: totalWeight > 0 ? epistemicSum / totalWeight : 0.5,
    };
  }

  /**
   * Classify uncertainty type.
   */
  private classifyUncertainty(
    aleatoricScore: number,
    epistemicScore: number
  ): UncertaintyType {
    if (aleatoricScore > this.config.aleatoricThreshold &&
        aleatoricScore > epistemicScore) {
      return 'aleatoric';
    }
    if (epistemicScore > this.config.epistemicThreshold &&
        epistemicScore > aleatoricScore) {
      return 'epistemic';
    }
    return 'mixed';
  }

  /**
   * Calculate confidence in assessment.
   */
  private calculateConfidence(indicators: AleatoricIndicator[], commitCount: number): number {
    // More commits = more confidence
    const commitConfidence = Math.min(1, commitCount / 20);

    // More agreeing indicators = more confidence
    const aleatoricCount = indicators.filter(i => i.isAleatoric).length;
    const epistemicCount = indicators.length - aleatoricCount;
    const agreement = Math.abs(aleatoricCount - epistemicCount) / indicators.length;

    return (commitConfidence * 0.4 + agreement * 0.6);
  }

  /**
   * Build detailed analysis.
   */
  private buildDetails(
    signal: TemporalSignal,
    commits: GitCommit[],
    indicators: AleatoricIndicator[]
  ): AleatoricDetails {
    return {
      isBoundaryFile: this.isBoundaryFile(signal.path),
      externallyDriven: this.calculateKeywordRatio(commits, EXTERNAL_KEYWORDS) > 0.3,
      multiPurposeChurn: signal.authorCount > 2,
      hasStableCore: indicators.some(i => i.name === 'Code Maturity' && i.value > 0.5),
      isExploratory: this.calculateKeywordRatio(commits, EXPLORATION_KEYWORDS) > 0.2,
      changePattern: this.analyzeChangePattern(commits),
    };
  }

  /**
   * Check if file is at a system boundary.
   */
  private isBoundaryFile(path: string): boolean {
    const boundaryPatterns = [
      /api/i, /interface/i, /types/i, /schema/i,
      /client/i, /server/i, /handler/i, /adapter/i,
      /bridge/i, /gateway/i, /protocol/i
    ];
    return boundaryPatterns.some(p => p.test(path));
  }

  /**
   * Generate recommendation.
   */
  private generateRecommendation(
    type: UncertaintyType,
    aleatoricScore: number,
    epistemicScore: number,
    details: AleatoricDetails
  ): AleatoricRecommendation {
    // Pure aleatoric: accept the variance
    if (type === 'aleatoric') {
      if (details.isBoundaryFile) {
        return 'stabilize-interface';
      }
      return 'accept-variance';
    }

    // Pure epistemic: investigate and fix
    if (type === 'epistemic') {
      return 'investigate-root';
    }

    // Mixed: depends on details
    if (details.multiPurposeChurn) {
      return 'split-concerns';
    }
    if (details.hasStableCore) {
      return 'accept-variance';
    }

    return 'monitor-pattern';
  }

  /**
   * Create assessment object.
   */
  private createAssessment(
    path: string,
    type: UncertaintyType,
    aleatoricScore: number,
    epistemicScore: number,
    confidence: number,
    indicators: AleatoricIndicator[],
    recommendation: AleatoricRecommendation,
    details: AleatoricDetails
  ): AleatoricAssessment {
    return {
      path,
      uncertaintyType: type,
      aleatoricScore,
      epistemicScore,
      confidence,
      indicators,
      recommendation,
      details,
    };
  }

  /**
   * Default details for insufficient data.
   */
  private defaultDetails(): AleatoricDetails {
    return {
      isBoundaryFile: false,
      externallyDriven: false,
      multiPurposeChurn: false,
      hasStableCore: false,
      isExploratory: false,
      changePattern: 'regular',
    };
  }
}

/**
 * Quick function to assess uncertainty type.
 */
export function assessUncertainty(
  signal: TemporalSignal,
  commits: GitCommit[],
  thrashReport?: ThrashReport,
  entelechyResult?: EntelechyResult,
  config?: Partial<AleatoricCheckConfig>
): AleatoricAssessment {
  const checker = new AleatoricChecker(config);
  return checker.assess(signal, commits, thrashReport, entelechyResult);
}

/**
 * Get recommendation description.
 */
export function getRecommendationDescription(rec: AleatoricRecommendation): string {
  const descriptions: Record<AleatoricRecommendation, string> = {
    'accept-variance': 'This is healthy variance. Accept it and maintain diversity.',
    'investigate-root': 'This is reducible uncertainty. Find and fix the root cause.',
    'monitor-pattern': 'Mixed signals. Monitor for trends before taking action.',
    'stabilize-interface': 'Boundary file with high churn. Consider stabilizing the interface.',
    'split-concerns': 'Multiple purposes causing churn. Consider splitting into separate modules.',
  };
  return descriptions[rec];
}
