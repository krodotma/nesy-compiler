export interface LearningMetrics {
  epoch: number;
  successRate: number;
  averageQuality: number;
  explorationRate: number;
  bufferSize: number;
}

/**
 * Tracks learning metrics over time
 */
export class MetricsTracker {
  private history: LearningMetrics[] = [];

  record(metrics: LearningMetrics): void {
    this.history.push(metrics);
  }

  getLatest(): LearningMetrics | undefined {
    return this.history[this.history.length - 1];
  }

  getHistory(): LearningMetrics[] {
    return [...this.history];
  }

  trend(field: keyof LearningMetrics, window: number = 10): number[] {
    return this.history.slice(-window).map(m => m[field] as number);
  }
}
