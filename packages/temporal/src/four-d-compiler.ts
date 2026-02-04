/**
 * FourDCompiler: 4D Code Compilation (Spatial + Temporal)
 *
 * Steps 24-25 of NeSy Evolution Phase 2 (Temporal Archeology)
 *
 * The "4D" refers to:
 * - X: Code structure (AST, imports, dependencies)
 * - Y: Semantic meaning (embeddings, types, contracts)
 * - Z: Trust hierarchy (ring levels, proof status)
 * - T: Temporal dimension (git history, evolution)
 *
 * This module integrates all temporal analyses with spatial
 * code understanding to produce a complete 4D compilation view.
 *
 * Key insight: Code that is spatially well-structured but
 * temporally unstable is different from code that is
 * spatially chaotic but temporally stable.
 */

import type { TemporalSignal } from './temporal-signal';
import type { ThrashReport, ThrashLevel } from './thrash-detector';
import type { EntelechyResult, EntelechyClass } from './entelechy';
import type { FileBlameMap, AuthorArchetype } from './blame-map';
import type { AleatoricAssessment, UncertaintyType } from './aleatoric-check';
import type { BranchPair, MergeComplexity } from './branch-reconciler';

/**
 * Complete 4D analysis for a code unit.
 */
export interface FourDAnalysis {
  /** File path */
  path: string;

  /** Spatial dimensions (X, Y, Z) */
  spatial: SpatialAnalysis;

  /** Temporal dimension (T) */
  temporal: TemporalAnalysis;

  /** Combined 4D metrics */
  combined: CombinedMetrics;

  /** Ring classification based on 4D analysis */
  ringClassification: RingLevel;

  /** Actionable recommendations */
  recommendations: FourDRecommendation[];

  /** Risk assessment */
  risk: RiskAssessment;
}

export interface SpatialAnalysis {
  /** Code structure metrics */
  structure: {
    linesOfCode: number;
    cyclomaticComplexity: number;
    dependencyCount: number;
    exportCount: number;
    isLeafModule: boolean;
    isHubModule: boolean;
  };

  /** Semantic metrics */
  semantics: {
    typeCompleteness: number;  // 0-1, how much is typed
    testCoverage: number;      // 0-1
    documentationRatio: number; // comments/code ratio
  };

  /** Trust metrics */
  trust: {
    proofStatus: 'proven' | 'tested' | 'untested' | 'unknown';
    linterScore: number;       // 0-100
    antipatternCount: number;
  };
}

export interface TemporalAnalysis {
  /** Core temporal signal */
  signal: TemporalSignal;

  /** Thrash analysis */
  thrash: {
    level: ThrashLevel;
    score: number;
  };

  /** Entelechy (code maturity) */
  entelechy: {
    class: EntelechyClass;
    score: number;
    survivedRefactors: number;
  };

  /** Uncertainty decomposition */
  uncertainty: {
    type: UncertaintyType;
    aleatoricScore: number;
    epistemicScore: number;
  };

  /** Dominant author archetype */
  dominantArchetype: AuthorArchetype;

  /** Branch status (if applicable) */
  branchStatus?: {
    conflictProbability: number;
    mergeComplexity: MergeComplexity;
  };
}

export interface CombinedMetrics {
  /** Overall health score (0-100) */
  healthScore: number;

  /** Stability score (temporal) */
  stabilityScore: number;

  /** Quality score (spatial) */
  qualityScore: number;

  /** Evolution velocity (changes over time) */
  velocity: number;

  /** Technical debt indicator */
  debtIndicator: number;

  /** Refactor priority (0-10) */
  refactorPriority: number;
}

export type RingLevel = 0 | 1 | 2 | 3;

export interface RingClassificationReason {
  factor: string;
  impact: 'positive' | 'negative';
  description: string;
}

export interface FourDRecommendation {
  priority: 'critical' | 'high' | 'medium' | 'low';
  action: string;
  reason: string;
  effort: number; // 1-10
}

export interface RiskAssessment {
  overall: 'low' | 'medium' | 'high' | 'critical';
  factors: RiskFactor[];
  mitigations: string[];
}

export interface RiskFactor {
  name: string;
  severity: number; // 0-1
  description: string;
}

export interface FourDCompilerConfig {
  /** Weight for spatial factors in health score */
  spatialWeight: number;
  /** Weight for temporal factors in health score */
  temporalWeight: number;
  /** Threshold for Ring 0 classification */
  ring0Threshold: number;
  /** Threshold for Ring 1 classification */
  ring1Threshold: number;
  /** Threshold for Ring 2 classification */
  ring2Threshold: number;
}

const DEFAULT_CONFIG: FourDCompilerConfig = {
  spatialWeight: 0.4,
  temporalWeight: 0.6,
  ring0Threshold: 85,
  ring1Threshold: 70,
  ring2Threshold: 50,
};

/**
 * FourDCompiler: Integrate spatial and temporal analysis.
 */
export class FourDCompiler {
  private config: FourDCompilerConfig;

  constructor(config?: Partial<FourDCompilerConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Perform complete 4D analysis.
   */
  analyze(
    path: string,
    spatialInput: Partial<SpatialAnalysis>,
    temporalInput: {
      signal: TemporalSignal;
      thrashReport?: ThrashReport;
      entelechyResult?: EntelechyResult;
      aleatoricAssessment?: AleatoricAssessment;
      blameMap?: FileBlameMap;
      branchPair?: BranchPair;
    }
  ): FourDAnalysis {
    // Build complete spatial analysis
    const spatial = this.buildSpatialAnalysis(spatialInput);

    // Build complete temporal analysis
    const temporal = this.buildTemporalAnalysis(temporalInput);

    // Calculate combined metrics
    const combined = this.calculateCombinedMetrics(spatial, temporal);

    // Classify ring level
    const ringClassification = this.classifyRing(combined, spatial, temporal);

    // Generate recommendations
    const recommendations = this.generateRecommendations(
      spatial,
      temporal,
      combined,
      ringClassification
    );

    // Assess risk
    const risk = this.assessRisk(spatial, temporal, combined);

    return {
      path,
      spatial,
      temporal,
      combined,
      ringClassification,
      recommendations,
      risk,
    };
  }

  /**
   * Build complete spatial analysis.
   */
  private buildSpatialAnalysis(input: Partial<SpatialAnalysis>): SpatialAnalysis {
    return {
      structure: {
        linesOfCode: input.structure?.linesOfCode || 0,
        cyclomaticComplexity: input.structure?.cyclomaticComplexity || 1,
        dependencyCount: input.structure?.dependencyCount || 0,
        exportCount: input.structure?.exportCount || 0,
        isLeafModule: input.structure?.isLeafModule ?? true,
        isHubModule: input.structure?.isHubModule ?? false,
      },
      semantics: {
        typeCompleteness: input.semantics?.typeCompleteness || 0,
        testCoverage: input.semantics?.testCoverage || 0,
        documentationRatio: input.semantics?.documentationRatio || 0,
      },
      trust: {
        proofStatus: input.trust?.proofStatus || 'unknown',
        linterScore: input.trust?.linterScore || 50,
        antipatternCount: input.trust?.antipatternCount || 0,
      },
    };
  }

  /**
   * Build complete temporal analysis.
   */
  private buildTemporalAnalysis(input: {
    signal: TemporalSignal;
    thrashReport?: ThrashReport;
    entelechyResult?: EntelechyResult;
    aleatoricAssessment?: AleatoricAssessment;
    blameMap?: FileBlameMap;
    branchPair?: BranchPair;
  }): TemporalAnalysis {
    return {
      signal: input.signal,
      thrash: {
        level: input.thrashReport?.level || 'none',
        score: input.thrashReport?.score || 0,
      },
      entelechy: {
        class: input.entelechyResult?.class || 'emerging',
        score: input.entelechyResult?.score || 50,
        survivedRefactors: input.entelechyResult?.survivedRefactors || 0,
      },
      uncertainty: {
        type: input.aleatoricAssessment?.uncertaintyType || 'mixed',
        aleatoricScore: input.aleatoricAssessment?.aleatoricScore || 0.5,
        epistemicScore: input.aleatoricAssessment?.epistemicScore || 0.5,
      },
      dominantArchetype: input.blameMap?.dominantArchetype || 'unknown',
      branchStatus: input.branchPair ? {
        conflictProbability: input.branchPair.conflictProbability,
        mergeComplexity: input.branchPair.mergeComplexity,
      } : undefined,
    };
  }

  /**
   * Calculate combined 4D metrics.
   */
  private calculateCombinedMetrics(
    spatial: SpatialAnalysis,
    temporal: TemporalAnalysis
  ): CombinedMetrics {
    // Calculate spatial quality score
    const qualityScore = this.calculateQualityScore(spatial);

    // Calculate temporal stability score
    const stabilityScore = this.calculateStabilityScore(temporal);

    // Combined health score
    const healthScore =
      qualityScore * this.config.spatialWeight +
      stabilityScore * this.config.temporalWeight;

    // Evolution velocity
    const velocity = temporal.signal.commitFreq * (1 / Math.max(1, temporal.signal.ageInDays / 30));

    // Technical debt indicator
    const debtIndicator = this.calculateDebtIndicator(spatial, temporal);

    // Refactor priority
    const refactorPriority = this.calculateRefactorPriority(
      healthScore,
      debtIndicator,
      temporal.thrash.level
    );

    return {
      healthScore,
      stabilityScore,
      qualityScore,
      velocity,
      debtIndicator,
      refactorPriority,
    };
  }

  /**
   * Calculate spatial quality score.
   */
  private calculateQualityScore(spatial: SpatialAnalysis): number {
    let score = 50; // Base score

    // Type completeness bonus
    score += spatial.semantics.typeCompleteness * 15;

    // Test coverage bonus
    score += spatial.semantics.testCoverage * 15;

    // Linter score contribution
    score += (spatial.trust.linterScore / 100) * 10;

    // Proof status bonus
    if (spatial.trust.proofStatus === 'proven') score += 10;
    else if (spatial.trust.proofStatus === 'tested') score += 5;

    // Antipattern penalty
    score -= Math.min(20, spatial.trust.antipatternCount * 2);

    // Complexity penalty
    if (spatial.structure.cyclomaticComplexity > 10) {
      score -= Math.min(15, (spatial.structure.cyclomaticComplexity - 10) * 1.5);
    }

    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate temporal stability score.
   */
  private calculateStabilityScore(temporal: TemporalAnalysis): number {
    let score = 50; // Base score

    // Entelechy bonus (mature code)
    if (temporal.entelechy.class === 'signal') score += 25;
    else if (temporal.entelechy.class === 'emerging') score += 10;
    else if (temporal.entelechy.class === 'unstable') score -= 10;
    else if (temporal.entelechy.class === 'noise') score -= 20;

    // Thrash penalty
    if (temporal.thrash.level === 'critical') score -= 30;
    else if (temporal.thrash.level === 'high') score -= 20;
    else if (temporal.thrash.level === 'medium') score -= 10;
    else if (temporal.thrash.level === 'low') score -= 5;

    // Aleatoric bonus (healthy variance is ok)
    if (temporal.uncertainty.type === 'aleatoric') {
      score += 10;
    } else if (temporal.uncertainty.type === 'epistemic') {
      score -= 10;
    }

    // Churn penalty (high churn = less stable)
    score -= Math.min(15, temporal.signal.churnRate * 3);

    // Age bonus (older stable code)
    if (temporal.signal.ageInDays > 365 && temporal.thrash.level === 'none') {
      score += 10;
    }

    return Math.max(0, Math.min(100, score));
  }

  /**
   * Calculate technical debt indicator.
   */
  private calculateDebtIndicator(
    spatial: SpatialAnalysis,
    temporal: TemporalAnalysis
  ): number {
    let debt = 0;

    // Missing types
    debt += (1 - spatial.semantics.typeCompleteness) * 20;

    // Missing tests
    debt += (1 - spatial.semantics.testCoverage) * 20;

    // Antipatterns
    debt += Math.min(20, spatial.trust.antipatternCount * 5);

    // High thrash
    if (temporal.thrash.level === 'critical') debt += 20;
    else if (temporal.thrash.level === 'high') debt += 15;

    // Epistemic uncertainty (bugs we don't understand)
    if (temporal.uncertainty.type === 'epistemic') {
      debt += temporal.uncertainty.epistemicScore * 20;
    }

    return Math.min(100, debt);
  }

  /**
   * Calculate refactor priority.
   */
  private calculateRefactorPriority(
    healthScore: number,
    debtIndicator: number,
    thrashLevel: ThrashLevel
  ): number {
    let priority = 5; // Default medium

    // Low health = high priority
    if (healthScore < 30) priority += 3;
    else if (healthScore < 50) priority += 2;
    else if (healthScore > 80) priority -= 2;

    // High debt = higher priority
    if (debtIndicator > 70) priority += 2;
    else if (debtIndicator > 50) priority += 1;

    // Thrash = higher priority
    if (thrashLevel === 'critical') priority += 2;
    else if (thrashLevel === 'high') priority += 1;

    return Math.max(1, Math.min(10, priority));
  }

  /**
   * Classify ring level based on 4D analysis.
   */
  private classifyRing(
    combined: CombinedMetrics,
    spatial: SpatialAnalysis,
    temporal: TemporalAnalysis
  ): RingLevel {
    // Ring 0: Kernel (highest trust)
    if (combined.healthScore >= this.config.ring0Threshold &&
        temporal.entelechy.class === 'signal' &&
        spatial.trust.proofStatus === 'proven') {
      return 0;
    }

    // Ring 1: Core (high trust)
    if (combined.healthScore >= this.config.ring1Threshold &&
        (temporal.entelechy.class === 'signal' || temporal.entelechy.class === 'emerging') &&
        temporal.thrash.level === 'none') {
      return 1;
    }

    // Ring 2: Standard (medium trust)
    if (combined.healthScore >= this.config.ring2Threshold &&
        temporal.thrash.level !== 'critical') {
      return 2;
    }

    // Ring 3: Untrusted
    return 3;
  }

  /**
   * Generate actionable recommendations.
   */
  private generateRecommendations(
    spatial: SpatialAnalysis,
    temporal: TemporalAnalysis,
    combined: CombinedMetrics,
    ring: RingLevel
  ): FourDRecommendation[] {
    const recommendations: FourDRecommendation[] = [];

    // Critical: High thrash needs immediate attention
    if (temporal.thrash.level === 'critical') {
      recommendations.push({
        priority: 'critical',
        action: 'Investigate and stabilize thrashing code',
        reason: 'Code is changing frequently without progress',
        effort: 8,
      });
    }

    // High: Missing types in important code
    if (spatial.semantics.typeCompleteness < 0.5 && ring <= 1) {
      recommendations.push({
        priority: 'high',
        action: 'Add type annotations',
        reason: 'Core code lacks type safety',
        effort: 5,
      });
    }

    // High: Epistemic uncertainty needs investigation
    if (temporal.uncertainty.type === 'epistemic' &&
        temporal.uncertainty.epistemicScore > 0.7) {
      recommendations.push({
        priority: 'high',
        action: 'Investigate root cause of instability',
        reason: 'High reducible uncertainty indicates bugs or misunderstanding',
        effort: 6,
      });
    }

    // Medium: Missing tests
    if (spatial.semantics.testCoverage < 0.5 && ring <= 2) {
      recommendations.push({
        priority: 'medium',
        action: 'Increase test coverage',
        reason: 'Important code lacks test coverage',
        effort: 6,
      });
    }

    // Medium: High complexity
    if (spatial.structure.cyclomaticComplexity > 15) {
      recommendations.push({
        priority: 'medium',
        action: 'Reduce cyclomatic complexity',
        reason: 'High complexity increases bug risk',
        effort: 7,
      });
    }

    // Low: Documentation
    if (spatial.semantics.documentationRatio < 0.1 && ring === 0) {
      recommendations.push({
        priority: 'low',
        action: 'Add documentation',
        reason: 'Kernel code should be well documented',
        effort: 3,
      });
    }

    return recommendations.sort((a, b) => {
      const priorityOrder = { critical: 0, high: 1, medium: 2, low: 3 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  }

  /**
   * Assess overall risk.
   */
  private assessRisk(
    spatial: SpatialAnalysis,
    temporal: TemporalAnalysis,
    combined: CombinedMetrics
  ): RiskAssessment {
    const factors: RiskFactor[] = [];
    const mitigations: string[] = [];

    // Thrash risk
    if (temporal.thrash.level !== 'none') {
      factors.push({
        name: 'Code Thrash',
        severity: temporal.thrash.score / 100,
        description: `${temporal.thrash.level} level of churn detected`,
      });
      mitigations.push('Review recent changes and stabilize before proceeding');
    }

    // Type safety risk
    if (spatial.semantics.typeCompleteness < 0.5) {
      factors.push({
        name: 'Type Safety',
        severity: 1 - spatial.semantics.typeCompleteness,
        description: 'Missing type annotations increase bug risk',
      });
      mitigations.push('Add strict TypeScript types');
    }

    // Test coverage risk
    if (spatial.semantics.testCoverage < 0.3) {
      factors.push({
        name: 'Test Coverage',
        severity: 1 - spatial.semantics.testCoverage,
        description: 'Low test coverage means bugs may go undetected',
      });
      mitigations.push('Write unit and integration tests');
    }

    // Branch conflict risk
    const conflictProb = temporal.branchStatus?.conflictProbability;
    if (conflictProb !== undefined && conflictProb > 0.5) {
      factors.push({
        name: 'Merge Conflict',
        severity: conflictProb,
        description: 'High probability of merge conflicts',
      });
      mitigations.push('Coordinate with other branches before merging');
    }

    // Calculate overall risk
    const avgSeverity = factors.length > 0
      ? factors.reduce((sum, f) => sum + f.severity, 0) / factors.length
      : 0;

    let overall: RiskAssessment['overall'] = 'low';
    if (avgSeverity > 0.7) overall = 'critical';
    else if (avgSeverity > 0.5) overall = 'high';
    else if (avgSeverity > 0.3) overall = 'medium';

    return { overall, factors, mitigations };
  }

  /**
   * Batch analyze multiple files.
   */
  analyzeMultiple(
    analyses: Array<{
      path: string;
      spatial: Partial<SpatialAnalysis>;
      temporal: Parameters<FourDCompiler['analyze']>[2];
    }>
  ): FourDAnalysis[] {
    return analyses.map(a => this.analyze(a.path, a.spatial, a.temporal));
  }

  /**
   * Get summary statistics for a codebase.
   */
  summarize(analyses: FourDAnalysis[]): {
    totalFiles: number;
    avgHealthScore: number;
    ringDistribution: Record<RingLevel, number>;
    topRisks: RiskFactor[];
    criticalRecommendations: FourDRecommendation[];
  } {
    const ringDist: Record<RingLevel, number> = { 0: 0, 1: 0, 2: 0, 3: 0 };
    let totalHealth = 0;
    const allRisks: RiskFactor[] = [];
    const allRecs: FourDRecommendation[] = [];

    for (const analysis of analyses) {
      ringDist[analysis.ringClassification]++;
      totalHealth += analysis.combined.healthScore;
      allRisks.push(...analysis.risk.factors);
      allRecs.push(...analysis.recommendations);
    }

    // Top risks by severity
    const topRisks = allRisks
      .sort((a, b) => b.severity - a.severity)
      .slice(0, 5);

    // Critical recommendations
    const criticalRecs = allRecs
      .filter(r => r.priority === 'critical')
      .slice(0, 5);

    return {
      totalFiles: analyses.length,
      avgHealthScore: analyses.length > 0 ? totalHealth / analyses.length : 0,
      ringDistribution: ringDist,
      topRisks,
      criticalRecommendations: criticalRecs,
    };
  }
}

/**
 * Create a 4D compiler instance.
 */
export function createFourDCompiler(config?: Partial<FourDCompilerConfig>): FourDCompiler {
  return new FourDCompiler(config);
}

/**
 * Quick 4D analysis for a single file.
 */
export function analyzeFourD(
  path: string,
  spatialInput: Partial<SpatialAnalysis>,
  temporalInput: Parameters<FourDCompiler['analyze']>[2],
  config?: Partial<FourDCompilerConfig>
): FourDAnalysis {
  const compiler = new FourDCompiler(config);
  return compiler.analyze(path, spatialInput, temporalInput);
}
