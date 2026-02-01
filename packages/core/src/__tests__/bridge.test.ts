import { describe, it, expect } from 'vitest';
import { ThresholdGrounding, SoftGrounding, discretize, anneal } from '../bridge.js';
import type { NeuralFeatures } from '../types.js';

function mockFeatures(confidence: number): NeuralFeatures {
  return {
    embedding: { vector: new Float32Array([0.1, 0.2, 0.3]), dimension: 3, model: 'test', timestamp: Date.now() },
    attention: { heads: [new Float32Array([0.5, 0.3, 0.2])], layers: 1, normalized: true },
    confidence,
  };
}

describe('ThresholdGrounding', () => {
  it('grounds high-confidence features as constants', async () => {
    const g = new ThresholdGrounding(0.7);
    const result = await g.ground(mockFeatures(0.9));
    expect(result.target.terms[0].type).toBe('constant');
    expect(result.ambiguities).toHaveLength(0);
  });

  it('grounds low-confidence features as variables', async () => {
    const g = new ThresholdGrounding(0.7);
    const result = await g.ground(mockFeatures(0.3));
    expect(result.target.terms[0].type).toBe('variable');
    expect(result.ambiguities.length).toBeGreaterThan(0);
  });
});

describe('SoftGrounding', () => {
  it('applies temperature scaling', async () => {
    const g = new SoftGrounding({ temperature: 1.0, annealing: 'linear', threshold: 0.5 });
    const result = await g.ground(mockFeatures(0.8));
    expect(result.confidence).toBeGreaterThan(0);
  });
});

describe('discretize', () => {
  it('converts continuous to discrete', () => {
    const continuous = new Float32Array([0.1, 0.9, -0.5, 2.0]);
    const result = discretize(continuous, { temperature: 1.0, annealing: 'linear', threshold: 0.5 });
    expect(result).toBeInstanceOf(Int32Array);
    expect(result.length).toBe(4);
  });
});

describe('anneal', () => {
  it('reduces temperature over steps', () => {
    const config = { temperature: 1.0, annealing: 'linear' as const, threshold: 0.5 };
    const annealed = anneal(config, 50, 100);
    expect(annealed.temperature).toBeLessThan(1.0);
  });

  it('respects minimum temperature', () => {
    const config = { temperature: 1.0, annealing: 'exponential' as const, threshold: 0.5 };
    const annealed = anneal(config, 99, 100);
    expect(annealed.temperature).toBeGreaterThanOrEqual(0.01);
  });
});
