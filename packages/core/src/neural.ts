/**
 * Neural layer primitives
 *
 * - Embedding: discrete tokens → continuous vectors
 * - Attention: relevance-weighted aggregation
 * - Adapter: efficient fine-tuning (LoRA, prefix)
 */

import type { Embedding, AttentionWeights, NeuralFeatures } from './types';

// =============================================================================
// Embedding
// =============================================================================

export interface EmbeddingProvider {
  embed(text: string): Promise<Embedding>;
  batchEmbed(texts: string[]): Promise<Embedding[]>;
  dimension: number;
  model: string;
}

export function cosineSimilarity(a: Embedding, b: Embedding): number {
  if (a.dimension !== b.dimension) {
    throw new Error(`Dimension mismatch: ${a.dimension} vs ${b.dimension}`);
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.dimension; i++) {
    dot += a.vector[i] * b.vector[i];
    normA += a.vector[i] * a.vector[i];
    normB += b.vector[i] * b.vector[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

export function averageEmbedding(embeddings: Embedding[]): Embedding {
  if (embeddings.length === 0) {
    throw new Error('Cannot average empty embedding list');
  }

  const dim = embeddings[0].dimension;
  const result = new Float32Array(dim);

  for (const emb of embeddings) {
    for (let i = 0; i < dim; i++) {
      result[i] += emb.vector[i];
    }
  }

  for (let i = 0; i < dim; i++) {
    result[i] /= embeddings.length;
  }

  return {
    vector: result,
    dimension: dim,
    model: embeddings[0].model,
    timestamp: Date.now(),
  };
}

// =============================================================================
// Attention
// =============================================================================

export function extractTopAttention(
  weights: AttentionWeights,
  k: number = 10
): Array<{ index: number; weight: number }> {
  // Average across heads
  const avgWeights = new Float32Array(weights.heads[0].length);

  for (const head of weights.heads) {
    for (let i = 0; i < head.length; i++) {
      avgWeights[i] += head[i] / weights.heads.length;
    }
  }

  // Get top-k
  const indexed = Array.from(avgWeights).map((w, i) => ({ index: i, weight: w }));
  indexed.sort((a, b) => b.weight - a.weight);

  return indexed.slice(0, k);
}

export function attentionEntropy(weights: AttentionWeights): number {
  // Compute entropy of attention distribution (measure of focus)
  let totalEntropy = 0;

  for (const head of weights.heads) {
    let entropy = 0;
    for (const w of head) {
      if (w > 0) {
        entropy -= w * Math.log2(w);
      }
    }
    totalEntropy += entropy / weights.heads.length;
  }

  return totalEntropy;
}

// =============================================================================
// Feature Extraction
// =============================================================================

export function computeConfidence(features: NeuralFeatures): number {
  // Heuristic: lower attention entropy = more confident/focused
  const maxEntropy = Math.log2(features.embedding.dimension);
  const entropy = attentionEntropy(features.attention);

  return 1 - (entropy / maxEntropy);
}

export function extractFeatures(
  embedding: Embedding,
  attention: AttentionWeights
): NeuralFeatures {
  const features: NeuralFeatures = {
    embedding,
    attention,
    confidence: 0,
  };

  features.confidence = computeConfidence(features);

  return features;
}
