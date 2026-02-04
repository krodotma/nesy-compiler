/**
 * SemanticCollapser: Thrash Filter for Graph Node Creation
 *
 * Purpose: Only create new Graph Nodes if the code is semantically distinct.
 * Uses LSA distance to filter out redundant/thrash code.
 *
 * Algorithm:
 * 1. Project new code into existing LSA space
 * 2. Find nearest neighbors in the semantic space
 * 3. If similarity > threshold, COLLAPSE (merge with existing node)
 * 4. If similarity < threshold, CREATE (new distinct node)
 *
 * This prevents the Knowledge Graph from being polluted with:
 * - Duplicate implementations
 * - Minor variations (whitespace, naming)
 * - Copy-paste code (thrash)
 */

import { tokenize, TokenizerConfig } from './tokenizer';
import { LSAModel, projectDocument } from './lsa';

export type CollapseDecision = 'create' | 'collapse' | 'review';

export interface CollapseResult {
  decision: CollapseDecision;
  /** ID of similar node if collapsing */
  collapseTarget?: string;
  /** Similarity score with nearest neighbor */
  similarity: number;
  /** Semantic vector for the code */
  vector: number[];
  /** Confidence in the decision (0-1) */
  confidence: number;
  /** Reasoning for the decision */
  reason: string;
}

export interface CollapserConfig {
  /** Similarity threshold for collapsing (default: 0.85) */
  collapseThreshold: number;
  /** Threshold below which we definitely create (default: 0.5) */
  createThreshold: number;
  /** Minimum document length to consider (tokens) */
  minTokens: number;
  /** Tokenizer configuration */
  tokenizer?: Partial<TokenizerConfig>;
}

const DEFAULT_COLLAPSER_CONFIG: CollapserConfig = {
  collapseThreshold: 0.85,
  createThreshold: 0.5,
  minTokens: 10
};

export class SemanticCollapser {
  private model: LSAModel;
  private config: CollapserConfig;

  constructor(model: LSAModel, config?: Partial<CollapserConfig>) {
    this.model = model;
    this.config = { ...DEFAULT_COLLAPSER_CONFIG, ...config };
  }

  /**
   * Evaluate whether new code should create a new node or collapse into existing.
   *
   * @param code The code content to evaluate
   * @param candidateId Optional ID for the candidate (for logging)
   * @returns CollapseResult with decision and metadata
   */
  evaluate(code: string, candidateId?: string): CollapseResult {
    // Tokenize the code
    const tokens = tokenize(code, this.config.tokenizer);

    // Handle empty or trivial code
    if (tokens.length < this.config.minTokens) {
      return {
        decision: 'collapse',
        similarity: 1.0,
        vector: [],
        confidence: 1.0,
        reason: `Code too short (${tokens.length} tokens < ${this.config.minTokens} minimum)`
      };
    }

    // Handle empty model
    if (this.model.dimensions === 0) {
      return {
        decision: 'create',
        similarity: 0,
        vector: [],
        confidence: 1.0,
        reason: 'Empty LSA model - first document'
      };
    }

    // Project into LSA space
    const vector = projectDocument(this.model, tokens);

    // Find nearest neighbor
    const maxSimilarity = this.findMaxSimilarity(vector);
    const nearestDoc = this.findNearestDocument(vector);

    // Make decision based on thresholds
    if (maxSimilarity >= this.config.collapseThreshold) {
      return {
        decision: 'collapse',
        collapseTarget: nearestDoc?.docId,
        similarity: maxSimilarity,
        vector,
        confidence: this.calculateConfidence(maxSimilarity, 'collapse'),
        reason: `High similarity (${(maxSimilarity * 100).toFixed(1)}%) to ${nearestDoc?.docId || 'existing node'}`
      };
    }

    if (maxSimilarity <= this.config.createThreshold) {
      return {
        decision: 'create',
        similarity: maxSimilarity,
        vector,
        confidence: this.calculateConfidence(maxSimilarity, 'create'),
        reason: `Semantically distinct (${(maxSimilarity * 100).toFixed(1)}% max similarity)`
      };
    }

    // In the gray zone - flag for human review
    return {
      decision: 'review',
      collapseTarget: nearestDoc?.docId,
      similarity: maxSimilarity,
      vector,
      confidence: this.calculateConfidence(maxSimilarity, 'review'),
      reason: `Ambiguous similarity (${(maxSimilarity * 100).toFixed(1)}%) - requires review`
    };
  }

  /**
   * Batch evaluate multiple code snippets.
   */
  evaluateBatch(items: Array<{ id: string; code: string }>): Map<string, CollapseResult> {
    const results = new Map<string, CollapseResult>();
    for (const item of items) {
      results.set(item.id, this.evaluate(item.code, item.id));
    }
    return results;
  }

  /**
   * Get statistics about the current model's semantic clustering.
   */
  getClusterStats(): {
    totalDocuments: number;
    estimatedClusters: number;
    avgSimilarity: number;
    potentialThrash: number;
  } {
    const n = this.model.tfidf.documents.length;
    if (n < 2) {
      return {
        totalDocuments: n,
        estimatedClusters: n,
        avgSimilarity: 0,
        potentialThrash: 0
      };
    }

    // Sample pairwise similarities
    let totalSim = 0;
    let count = 0;
    let highSimPairs = 0;

    const sampleSize = Math.min(n, 100);
    for (let i = 0; i < sampleSize; i++) {
      for (let j = i + 1; j < sampleSize; j++) {
        const sim = this.cosineSimilarity(
          this.model.documentVectors[i],
          this.model.documentVectors[j]
        );
        totalSim += sim;
        count++;
        if (sim >= this.config.collapseThreshold) {
          highSimPairs++;
        }
      }
    }

    const avgSimilarity = count > 0 ? totalSim / count : 0;

    // Rough cluster estimate based on similarity distribution
    const estimatedClusters = Math.max(1, Math.round(n * (1 - avgSimilarity)));

    return {
      totalDocuments: n,
      estimatedClusters,
      avgSimilarity,
      potentialThrash: highSimPairs
    };
  }

  /**
   * Find maximum similarity between a vector and all document vectors.
   */
  private findMaxSimilarity(vector: number[]): number {
    let maxSim = 0;
    for (const docVector of this.model.documentVectors) {
      const sim = this.cosineSimilarity(vector, docVector);
      if (sim > maxSim) {
        maxSim = sim;
      }
    }
    return maxSim;
  }

  /**
   * Find the nearest document to a vector.
   */
  private findNearestDocument(vector: number[]): { docId: string; similarity: number } | null {
    let maxSim = -1;
    let nearestIdx = -1;

    for (let i = 0; i < this.model.documentVectors.length; i++) {
      const sim = this.cosineSimilarity(vector, this.model.documentVectors[i]);
      if (sim > maxSim) {
        maxSim = sim;
        nearestIdx = i;
      }
    }

    if (nearestIdx < 0) return null;

    return {
      docId: this.model.tfidf.documents[nearestIdx],
      similarity: maxSim
    };
  }

  /**
   * Cosine similarity between two vectors.
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length || a.length === 0) return 0;

    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denom = Math.sqrt(normA) * Math.sqrt(normB);
    return denom > 0 ? dot / denom : 0;
  }

  /**
   * Calculate confidence in the decision.
   */
  private calculateConfidence(similarity: number, decision: CollapseDecision): number {
    const { collapseThreshold, createThreshold } = this.config;
    const midpoint = (collapseThreshold + createThreshold) / 2;

    switch (decision) {
      case 'collapse':
        // Higher similarity = higher confidence
        return Math.min(1, (similarity - collapseThreshold) / (1 - collapseThreshold) * 0.5 + 0.5);

      case 'create':
        // Lower similarity = higher confidence
        return Math.min(1, (createThreshold - similarity) / createThreshold * 0.5 + 0.5);

      case 'review':
        // Distance from midpoint determines uncertainty
        const distFromMid = Math.abs(similarity - midpoint);
        const maxDist = (collapseThreshold - createThreshold) / 2;
        return Math.max(0.1, 1 - (distFromMid / maxDist) * 0.5);
    }
  }
}

/**
 * Create a SemanticCollapser from a corpus.
 * Convenience function that builds the LSA model internally.
 */
export async function createCollapser(
  corpus: Map<string, string>,
  config?: Partial<CollapserConfig & { lsaDimensions?: number }>
): Promise<SemanticCollapser> {
  // Dynamic import to avoid circular dependency
  const { buildLSAModel } = await import('./lsa.js');

  const model = buildLSAModel(corpus, {
    dimensions: config?.lsaDimensions ?? 100
  });

  return new SemanticCollapser(model, config);
}
