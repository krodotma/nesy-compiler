/**
 * SignalBooster: Symbolic > Neural Trust Arbitration
 *
 * Step 14 of NeSy Evolution Phase 1.5 (Sensor Fusion)
 *
 * Core Principle: When Linter (Symbolic) says "Error" and GNN (Neural) says "Okay",
 * TRUST THE LINTER. Compilers don't hallucinate.
 *
 * This algorithm combines signals from:
 * - Static analysis (linters, type checker) → Ground Truth
 * - LSA/Neural analysis → Soft Signals
 * - Historical data (git, stability) → Context Signals
 *
 * Output: Boosted confidence score that respects symbolic constraints.
 */

import type { LintResult, Severity } from './linter-bridge';
import type { AntipatternMatch } from './antipattern-mapper';
import { getAntipatternSummary } from './antipattern-mapper';

export type SignalSource =
  | 'linter'      // ESLint, Ruff, TSC, Biome
  | 'type-system' // TypeScript type analysis
  | 'lsa'         // Latent Semantic Analysis
  | 'gnn'         // Graph Neural Network
  | 'history'     // Git history analysis
  | 'human'       // Human annotation
  | 'test';       // Test coverage/results

export interface Signal {
  source: SignalSource;
  type: 'error' | 'warning' | 'info' | 'ok';
  confidence: number;  // 0-1
  message?: string;
  evidence?: string[];
}

export interface BoostedResult {
  /** Final confidence score (0-1) */
  confidence: number;
  /** Trust level derived from confidence */
  trustLevel: 0 | 1 | 2 | 3;
  /** Whether symbolic signals override neural */
  symbolicOverride: boolean;
  /** Aggregated signals by source */
  signals: Signal[];
  /** Reasoning for the decision */
  reasoning: string[];
  /** Recommended ring level for this code */
  recommendedRing: 0 | 1 | 2 | 3;
}

export interface SignalBoosterConfig {
  /** Weight for symbolic signals (linter, type-system) */
  symbolicWeight: number;
  /** Weight for neural signals (LSA, GNN) */
  neuralWeight: number;
  /** Weight for historical signals */
  historyWeight: number;
  /** If true, any symbolic error forces low confidence */
  strictSymbolicMode: boolean;
  /** Minimum confidence to pass verification */
  minConfidence: number;
}

const DEFAULT_CONFIG: SignalBoosterConfig = {
  symbolicWeight: 2.0,    // Symbolic signals count double
  neuralWeight: 1.0,
  historyWeight: 0.5,
  strictSymbolicMode: true,
  minConfidence: 0.6,
};

/**
 * SignalBooster: Combines multiple signals into a final confidence score.
 */
export class SignalBooster {
  private config: SignalBoosterConfig;

  constructor(config?: Partial<SignalBoosterConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Boost confidence using all available signals.
   */
  boost(signals: Signal[]): BoostedResult {
    const reasoning: string[] = [];

    // Separate signals by category
    const symbolic = signals.filter(s =>
      s.source === 'linter' || s.source === 'type-system'
    );
    const neural = signals.filter(s =>
      s.source === 'lsa' || s.source === 'gnn'
    );
    const historical = signals.filter(s =>
      s.source === 'history' || s.source === 'test'
    );

    // Check for symbolic errors
    const symbolicErrors = symbolic.filter(s => s.type === 'error');
    const symbolicWarnings = symbolic.filter(s => s.type === 'warning');
    const symbolicOk = symbolic.filter(s => s.type === 'ok');

    // Neural assessment
    const neuralErrors = neural.filter(s => s.type === 'error');
    const neuralOk = neural.filter(s => s.type === 'ok' || s.type === 'info');

    // RULE 1: Symbolic errors always reduce confidence
    let symbolicOverride = false;
    if (symbolicErrors.length > 0 && this.config.strictSymbolicMode) {
      symbolicOverride = true;
      reasoning.push(
        `SYMBOLIC OVERRIDE: ${symbolicErrors.length} symbolic error(s) detected. ` +
        `Linter/TypeSystem errors take precedence over neural assessment.`
      );

      // Even if neural says OK, symbolic errors win
      if (neuralOk.length > 0) {
        reasoning.push(
          `Neural signals (${neuralOk.length}) indicated OK, but symbolic errors override.`
        );
      }
    }

    // RULE 2: Calculate weighted confidence
    let confidence = this.calculateWeightedConfidence(
      symbolic,
      neural,
      historical
    );

    // RULE 3: Apply symbolic override penalty
    if (symbolicOverride) {
      const errorPenalty = symbolicErrors.length * 0.15;
      const warningPenalty = symbolicWarnings.length * 0.05;
      const totalPenalty = Math.min(0.7, errorPenalty + warningPenalty);
      confidence = Math.max(0, confidence - totalPenalty);
      reasoning.push(
        `Applied penalty: -${(totalPenalty * 100).toFixed(0)}% ` +
        `(${symbolicErrors.length} errors, ${symbolicWarnings.length} warnings)`
      );
    }

    // RULE 4: Boost for clean symbolic signals
    if (symbolicErrors.length === 0 && symbolicOk.length > 0) {
      const boost = Math.min(0.2, symbolicOk.length * 0.05);
      confidence = Math.min(1, confidence + boost);
      reasoning.push(
        `Clean symbolic check: +${(boost * 100).toFixed(0)}% boost`
      );
    }

    // RULE 5: Neural signals provide soft adjustment
    if (neuralErrors.length > 0 && !symbolicOverride) {
      // Neural concerns without symbolic validation
      const softPenalty = neuralErrors.length * 0.05;
      confidence = Math.max(0.3, confidence - softPenalty);
      reasoning.push(
        `Neural concerns (${neuralErrors.length}): -${(softPenalty * 100).toFixed(0)}% (soft adjustment)`
      );
    }

    // Derive trust level from confidence
    const trustLevel = this.confidenceToTrustLevel(confidence);

    // Recommend ring based on trust + signals
    const recommendedRing = this.recommendRing(
      trustLevel,
      symbolicErrors.length,
      symbolicWarnings.length
    );

    return {
      confidence,
      trustLevel,
      symbolicOverride,
      signals,
      reasoning,
      recommendedRing,
    };
  }

  /**
   * Create signals from lint results.
   */
  signalsFromLintResults(results: LintResult[]): Signal[] {
    const signals: Signal[] = [];

    for (const result of results) {
      if (result.errorCount > 0) {
        signals.push({
          source: 'linter',
          type: 'error',
          confidence: 0.95,  // High confidence - compilers don't lie
          message: `${result.errorCount} error(s) in ${result.filePath}`,
          evidence: result.violations
            .filter(v => v.severity === 'error')
            .map(v => `${v.ruleId}: ${v.message}`),
        });
      }

      if (result.warningCount > 0) {
        signals.push({
          source: 'linter',
          type: 'warning',
          confidence: 0.8,
          message: `${result.warningCount} warning(s) in ${result.filePath}`,
          evidence: result.violations
            .filter(v => v.severity === 'warning')
            .map(v => `${v.ruleId}: ${v.message}`),
        });
      }

      if (result.errorCount === 0 && result.warningCount === 0) {
        signals.push({
          source: 'linter',
          type: 'ok',
          confidence: 0.9,
          message: `Clean: ${result.filePath}`,
        });
      }
    }

    return signals;
  }

  /**
   * Create signals from LSA analysis.
   */
  signalsFromLSA(
    similarity: number,
    isRedundant: boolean,
    conceptCluster?: string
  ): Signal[] {
    const signals: Signal[] = [];

    if (isRedundant) {
      signals.push({
        source: 'lsa',
        type: 'warning',
        confidence: similarity,
        message: `Semantically similar to existing code (${(similarity * 100).toFixed(0)}%)`,
        evidence: conceptCluster ? [`Cluster: ${conceptCluster}`] : undefined,
      });
    } else if (similarity < 0.3) {
      signals.push({
        source: 'lsa',
        type: 'ok',
        confidence: 1 - similarity,
        message: `Semantically distinct (${((1 - similarity) * 100).toFixed(0)}% uniqueness)`,
      });
    } else {
      signals.push({
        source: 'lsa',
        type: 'info',
        confidence: 0.6,
        message: `Moderate semantic similarity (${(similarity * 100).toFixed(0)}%)`,
      });
    }

    return signals;
  }

  /**
   * Create signals from historical analysis.
   */
  signalsFromHistory(
    stabilityScore: number,
    churnRate: number,
    authorCount: number
  ): Signal[] {
    const signals: Signal[] = [];

    // High stability = good signal
    if (stabilityScore > 0.8) {
      signals.push({
        source: 'history',
        type: 'ok',
        confidence: stabilityScore,
        message: `Stable code (${(stabilityScore * 100).toFixed(0)}% stability)`,
      });
    } else if (stabilityScore < 0.3) {
      signals.push({
        source: 'history',
        type: 'warning',
        confidence: 1 - stabilityScore,
        message: `Unstable code (${((1 - stabilityScore) * 100).toFixed(0)}% churn)`,
      });
    }

    // High churn = concerning
    if (churnRate > 0.5) {
      signals.push({
        source: 'history',
        type: 'warning',
        confidence: churnRate,
        message: `High churn rate (${(churnRate * 100).toFixed(0)}%)`,
      });
    }

    // Single author = potential bus factor
    if (authorCount === 1) {
      signals.push({
        source: 'history',
        type: 'info',
        confidence: 0.5,
        message: 'Single author (bus factor concern)',
      });
    }

    return signals;
  }

  /**
   * Calculate weighted confidence from signal groups.
   */
  private calculateWeightedConfidence(
    symbolic: Signal[],
    neural: Signal[],
    historical: Signal[]
  ): number {
    const { symbolicWeight, neuralWeight, historyWeight } = this.config;

    let totalWeight = 0;
    let weightedSum = 0;

    // Symbolic signals
    for (const s of symbolic) {
      const signalValue = s.type === 'error' ? 0 :
                         s.type === 'warning' ? 0.5 :
                         s.type === 'ok' ? 1 : 0.7;
      weightedSum += signalValue * s.confidence * symbolicWeight;
      totalWeight += symbolicWeight;
    }

    // Neural signals
    for (const s of neural) {
      const signalValue = s.type === 'error' ? 0.2 :
                         s.type === 'warning' ? 0.5 :
                         s.type === 'ok' ? 0.9 : 0.7;
      weightedSum += signalValue * s.confidence * neuralWeight;
      totalWeight += neuralWeight;
    }

    // Historical signals
    for (const s of historical) {
      const signalValue = s.type === 'error' ? 0.3 :
                         s.type === 'warning' ? 0.5 :
                         s.type === 'ok' ? 0.9 : 0.7;
      weightedSum += signalValue * s.confidence * historyWeight;
      totalWeight += historyWeight;
    }

    if (totalWeight === 0) return 0.5;  // No signals = neutral
    return weightedSum / totalWeight;
  }

  /**
   * Convert confidence to trust level.
   */
  private confidenceToTrustLevel(confidence: number): 0 | 1 | 2 | 3 {
    if (confidence >= 0.9) return 0;  // KERNEL
    if (confidence >= 0.7) return 1;  // PRIVILEGED
    if (confidence >= 0.5) return 2;  // STANDARD
    return 3;                          // UNTRUSTED
  }

  /**
   * Recommend ring level based on trust and signals.
   */
  private recommendRing(
    trustLevel: 0 | 1 | 2 | 3,
    errorCount: number,
    warningCount: number
  ): 0 | 1 | 2 | 3 {
    // Start with trust-based ring
    let ring = trustLevel;

    // Errors push to higher ring (less trust)
    if (errorCount > 0) {
      ring = Math.min(3, ring + 1) as 0 | 1 | 2 | 3;
    }

    // Many warnings also push to higher ring
    if (warningCount >= 5) {
      ring = Math.min(3, ring + 1) as 0 | 1 | 2 | 3;
    }

    return ring;
  }
}

/**
 * Quick function to boost a file's confidence from lint results.
 */
export function boostFromLintResults(
  results: LintResult[],
  config?: Partial<SignalBoosterConfig>
): BoostedResult {
  const booster = new SignalBooster(config);
  const signals = booster.signalsFromLintResults(results);
  return booster.boost(signals);
}

/**
 * Combine multiple signal sources for comprehensive boost.
 */
export function boostComprehensive(
  lintResults: LintResult[],
  lsaData?: { similarity: number; isRedundant: boolean; cluster?: string },
  historyData?: { stability: number; churn: number; authors: number },
  config?: Partial<SignalBoosterConfig>
): BoostedResult {
  const booster = new SignalBooster(config);
  const signals: Signal[] = [];

  // Add lint signals
  signals.push(...booster.signalsFromLintResults(lintResults));

  // Add LSA signals
  if (lsaData) {
    signals.push(...booster.signalsFromLSA(
      lsaData.similarity,
      lsaData.isRedundant,
      lsaData.cluster
    ));
  }

  // Add history signals
  if (historyData) {
    signals.push(...booster.signalsFromHistory(
      historyData.stability,
      historyData.churn,
      historyData.authors
    ));
  }

  return booster.boost(signals);
}
