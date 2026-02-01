import type { ExperienceBuffer } from './experience.js';

export interface StrategyConfig {
  learningRate: number;
  explorationRate: number;
  batchSize: number;
}

/**
 * Adaptation strategy that learns from experience
 */
export class AdaptationStrategy {
  private config: StrategyConfig;

  constructor(config: Partial<StrategyConfig> = {}) {
    this.config = {
      learningRate: 0.01,
      explorationRate: 0.1,
      batchSize: 32,
      ...config,
    };
  }

  shouldExplore(): boolean {
    return Math.random() < this.config.explorationRate;
  }

  computeUpdate(buffer: ExperienceBuffer): Record<string, number> {
    const batch = buffer.sample(this.config.batchSize);
    const successful = batch.filter(e => e.result.stages.verify.passed);

    return {
      successRate: batch.length > 0 ? successful.length / batch.length : 0,
      averageGatesPassed: batch.length > 0
        ? batch.reduce((s, e) => s + e.result.metrics.verificationGatesPassed, 0) / batch.length
        : 0,
    };
  }

  getConfig(): StrategyConfig {
    return { ...this.config };
  }
}
