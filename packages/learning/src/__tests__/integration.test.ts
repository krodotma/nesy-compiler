import { describe, it, expect } from 'vitest';
import { FeedbackCollector } from '../feedback.js';
import { ExperienceBuffer } from '../experience.js';
import { AdaptationStrategy } from '../strategy.js';
import { MetricsTracker } from '../metrics.js';
import { NeSyCompiler } from '@nesy/pipeline';

describe('Learning loop', () => {
  it('feedback->experience->strategy->metrics', async () => {
    const result = await new NeSyCompiler().compile({ mode: 'atom', intent: 'learn' });
    const collector = new FeedbackCollector();
    const buffer = new ExperienceBuffer(100);
    const fb = collector.record(result, 0.85, true);
    buffer.add(result, fb);
    const update = new AdaptationStrategy().computeUpdate(buffer);
    const tracker = new MetricsTracker();
    tracker.record({ epoch: 1, successRate: update.successRate, averageQuality: 0.85, explorationRate: 0.1, bufferSize: 1 });
    expect(tracker.getLatest()?.averageQuality).toBeCloseTo(0.85);
  });
});
