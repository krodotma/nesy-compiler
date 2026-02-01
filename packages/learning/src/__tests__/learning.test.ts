import { describe, it, expect } from 'vitest';
import { FeedbackCollector } from '../feedback.js';
import { ExperienceBuffer } from '../experience.js';
import { AdaptationStrategy } from '../strategy.js';
import { MetricsTracker } from '../metrics.js';

describe('FeedbackCollector', () => {
  it('records and retrieves feedback', () => {
    const collector = new FeedbackCollector();
    const mockResult = { ir: { id: 'test' } } as any;
    collector.record(mockResult, 0.8, true, 'good');
    expect(collector.getAll()).toHaveLength(1);
    expect(collector.averageQuality()).toBeCloseTo(0.8);
    expect(collector.getPositive()).toHaveLength(1);
    expect(collector.getNegative()).toHaveLength(0);
  });
});

describe('ExperienceBuffer', () => {
  it('adds and samples experiences', () => {
    const buffer = new ExperienceBuffer(10);
    const mockResult = { ir: { id: 'test' }, stages: { verify: { passed: true } } } as any;
    buffer.add(mockResult);
    expect(buffer.size()).toBe(1);
    expect(buffer.sample(5)).toHaveLength(1);
    expect(buffer.getSuccessful()).toHaveLength(1);
  });

  it('respects max size', () => {
    const buffer = new ExperienceBuffer(2);
    for (let i = 0; i < 5; i++) {
      buffer.add({ ir: { id: String(i) }, stages: { verify: { passed: true } } } as any);
    }
    expect(buffer.size()).toBe(2);
  });
});

describe('AdaptationStrategy', () => {
  it('has default config', () => {
    const strategy = new AdaptationStrategy();
    expect(strategy.getConfig().learningRate).toBe(0.01);
  });
});

describe('MetricsTracker', () => {
  it('tracks metrics history', () => {
    const tracker = new MetricsTracker();
    tracker.record({ epoch: 1, successRate: 0.5, averageQuality: 0.6, explorationRate: 0.1, bufferSize: 10 });
    expect(tracker.getLatest()?.epoch).toBe(1);
    expect(tracker.trend('successRate')).toEqual([0.5]);
  });
});
