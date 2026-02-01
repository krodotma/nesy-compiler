import { describe, it, expect } from 'vitest';
import { cosineSimilarity, averageEmbedding, extractTopAttention, attentionEntropy } from '../neural.js';
import type { Embedding, AttentionWeights } from '../types.js';

function emb(values: number[]): Embedding {
  return { vector: new Float32Array(values), dimension: values.length, model: 'test', timestamp: Date.now() };
}

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const e = emb([1, 0, 0]);
    expect(cosineSimilarity(e, e)).toBeCloseTo(1);
  });

  it('returns 0 for orthogonal vectors', () => {
    expect(cosineSimilarity(emb([1, 0]), emb([0, 1]))).toBeCloseTo(0);
  });

  it('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity(emb([1, 0]), emb([1, 0, 0]))).toThrow();
  });
});

describe('averageEmbedding', () => {
  it('averages embeddings', () => {
    const avg = averageEmbedding([emb([2, 4]), emb([4, 6])]);
    expect(avg.vector[0]).toBeCloseTo(3);
    expect(avg.vector[1]).toBeCloseTo(5);
  });

  it('throws on empty list', () => {
    expect(() => averageEmbedding([])).toThrow();
  });
});

describe('extractTopAttention', () => {
  it('returns top-k attention indices', () => {
    const weights: AttentionWeights = { heads: [new Float32Array([0.1, 0.5, 0.3, 0.8])], layers: 1, normalized: true };
    const top = extractTopAttention(weights, 2);
    expect(top[0].index).toBe(3);
    expect(top[1].index).toBe(1);
  });
});

describe('attentionEntropy', () => {
  it('computes entropy of attention', () => {
    const weights: AttentionWeights = { heads: [new Float32Array([0.25, 0.25, 0.25, 0.25])], layers: 1, normalized: true };
    expect(attentionEntropy(weights)).toBeCloseTo(2.0);
  });
});
