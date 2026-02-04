/**
 * Compilation Analytics - Performance Tracking and Optimization
 *
 * Phase 4 Step 33: Analytics and Optimization
 *
 * Tracks compilation performance and provides optimization insights:
 * - Phase timing statistics
 * - Success/failure rates
 * - Resource usage patterns
 * - Bottleneck identification
 * - Optimization recommendations
 */

import type { TelemetryData, PhaseExecution, PhaseId } from './orchestration-engine.js';

/** Time series data point */
export interface DataPoint {
  timestamp: number;
  value: number;
  labels?: Record<string, string>;
}

/** Statistical summary */
export interface Statistics {
  count: number;
  sum: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  stdDev: number;
  p50: number;
  p90: number;
  p95: number;
  p99: number;
}

/** Phase analytics */
export interface PhaseAnalytics {
  phase: PhaseId;
  executions: number;
  successes: number;
  failures: number;
  successRate: number;
  timing: Statistics;
  lastExecution?: PhaseExecution;
}

/** Compilation analytics summary */
export interface AnalyticsSummary {
  totalCompilations: number;
  successfulCompilations: number;
  failedCompilations: number;
  successRate: number;
  averageDuration: number;
  phases: Map<PhaseId, PhaseAnalytics>;
  bottlenecks: Bottleneck[];
  recommendations: Recommendation[];
  timeSeries: {
    compilationsPerHour: DataPoint[];
    averageDurationPerHour: DataPoint[];
    errorRatePerHour: DataPoint[];
  };
}

/** Identified bottleneck */
export interface Bottleneck {
  phase: PhaseId;
  type: 'slow' | 'flaky' | 'memory' | 'timeout';
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  impact: number; // Percentage of total time
  occurrences: number;
}

/** Optimization recommendation */
export interface Recommendation {
  id: string;
  category: 'performance' | 'reliability' | 'resource' | 'strategy';
  priority: 'low' | 'medium' | 'high';
  title: string;
  description: string;
  estimatedImpact: string;
  implementation?: string;
}

/** Analytics configuration */
export interface AnalyticsConfig {
  /** Maximum data points to retain */
  maxDataPoints: number;
  /** Data retention period (ms) */
  retentionPeriod: number;
  /** Slow phase threshold (ms) */
  slowThreshold: number;
  /** High error rate threshold */
  errorRateThreshold: number;
  /** Enable anomaly detection */
  anomalyDetection: boolean;
}

const DEFAULT_CONFIG: AnalyticsConfig = {
  maxDataPoints: 10000,
  retentionPeriod: 7 * 24 * 60 * 60 * 1000, // 7 days
  slowThreshold: 5000,
  errorRateThreshold: 0.1,
  anomalyDetection: true,
};

/**
 * CompilationAnalytics: Track and analyze compilation performance.
 */
export class CompilationAnalytics {
  private config: AnalyticsConfig;
  private compilationData: TelemetryData[] = [];
  private phaseExecutions: Map<PhaseId, PhaseExecution[]> = new Map();
  private timestamps: number[] = [];

  constructor(config?: Partial<AnalyticsConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Record compilation telemetry.
   */
  record(telemetry: TelemetryData, executions: PhaseExecution[]): void {
    const now = Date.now();

    // Store telemetry
    this.compilationData.push(telemetry);
    this.timestamps.push(now);

    // Store phase executions
    for (const exec of executions) {
      if (!this.phaseExecutions.has(exec.phase)) {
        this.phaseExecutions.set(exec.phase, []);
      }
      this.phaseExecutions.get(exec.phase)!.push(exec);
    }

    // Clean up old data
    this.pruneOldData();
  }

  /**
   * Get analytics summary.
   */
  getSummary(): AnalyticsSummary {
    const totalCompilations = this.compilationData.length;
    const successfulCompilations = this.compilationData.filter(
      t => t.fallbacksUsed.length === 0 && t.retryCount === 0
    ).length;
    const failedCompilations = totalCompilations - successfulCompilations;
    const successRate = totalCompilations > 0
      ? successfulCompilations / totalCompilations
      : 0;

    const durations = this.compilationData.map(t => t.totalDuration);
    const averageDuration = this.mean(durations);

    // Analyze phases
    const phases = this.analyzePhases();

    // Identify bottlenecks
    const bottlenecks = this.identifyBottlenecks(phases);

    // Generate recommendations
    const recommendations = this.generateRecommendations(phases, bottlenecks);

    // Build time series
    const timeSeries = this.buildTimeSeries();

    return {
      totalCompilations,
      successfulCompilations,
      failedCompilations,
      successRate,
      averageDuration,
      phases,
      bottlenecks,
      recommendations,
      timeSeries,
    };
  }

  /**
   * Analyze phase performance.
   */
  private analyzePhases(): Map<PhaseId, PhaseAnalytics> {
    const analytics = new Map<PhaseId, PhaseAnalytics>();

    for (const [phase, executions] of this.phaseExecutions) {
      const successes = executions.filter(e => e.status === 'completed').length;
      const failures = executions.filter(e => e.status === 'failed').length;
      const durations = executions
        .filter(e => e.duration !== undefined)
        .map(e => e.duration!);

      analytics.set(phase, {
        phase,
        executions: executions.length,
        successes,
        failures,
        successRate: executions.length > 0 ? successes / executions.length : 0,
        timing: this.calculateStatistics(durations),
        lastExecution: executions[executions.length - 1],
      });
    }

    return analytics;
  }

  /**
   * Identify performance bottlenecks.
   */
  private identifyBottlenecks(phases: Map<PhaseId, PhaseAnalytics>): Bottleneck[] {
    const bottlenecks: Bottleneck[] = [];
    const totalTime = this.sum(
      Array.from(phases.values()).map(p => p.timing.sum)
    );

    for (const [phase, analytics] of phases) {
      const impact = totalTime > 0 ? analytics.timing.sum / totalTime : 0;

      // Check for slow phases
      if (analytics.timing.p90 > this.config.slowThreshold) {
        const severity = this.severityFromDuration(analytics.timing.p90);
        bottlenecks.push({
          phase,
          type: 'slow',
          severity,
          description: `Phase ${phase} has slow p90 latency (${analytics.timing.p90.toFixed(0)}ms)`,
          impact,
          occurrences: analytics.executions,
        });
      }

      // Check for flaky phases
      if (analytics.successRate < (1 - this.config.errorRateThreshold)) {
        const severity = this.severityFromErrorRate(1 - analytics.successRate);
        bottlenecks.push({
          phase,
          type: 'flaky',
          severity,
          description: `Phase ${phase} has high failure rate (${((1 - analytics.successRate) * 100).toFixed(1)}%)`,
          impact,
          occurrences: analytics.failures,
        });
      }

      // Check for high variance (potential timeout issues)
      if (analytics.timing.stdDev > analytics.timing.mean * 2) {
        bottlenecks.push({
          phase,
          type: 'timeout',
          severity: 'medium',
          description: `Phase ${phase} has high variance (stdDev: ${analytics.timing.stdDev.toFixed(0)}ms)`,
          impact,
          occurrences: analytics.executions,
        });
      }
    }

    // Sort by impact
    bottlenecks.sort((a, b) => b.impact - a.impact);

    return bottlenecks;
  }

  /**
   * Generate optimization recommendations.
   */
  private generateRecommendations(
    phases: Map<PhaseId, PhaseAnalytics>,
    bottlenecks: Bottleneck[]
  ): Recommendation[] {
    const recommendations: Recommendation[] = [];

    // Recommendations based on bottlenecks
    for (const bottleneck of bottlenecks) {
      if (bottleneck.type === 'slow') {
        recommendations.push({
          id: `perf-${bottleneck.phase}`,
          category: 'performance',
          priority: bottleneck.severity === 'critical' ? 'high' : 'medium',
          title: `Optimize ${bottleneck.phase} phase`,
          description: `The ${bottleneck.phase} phase is taking ${bottleneck.impact * 100}% of total compilation time.`,
          estimatedImpact: `Reduce compilation time by up to ${(bottleneck.impact * 50).toFixed(0)}%`,
          implementation: this.getOptimizationHint(bottleneck.phase),
        });
      }

      if (bottleneck.type === 'flaky') {
        recommendations.push({
          id: `reliability-${bottleneck.phase}`,
          category: 'reliability',
          priority: 'high',
          title: `Improve ${bottleneck.phase} reliability`,
          description: `The ${bottleneck.phase} phase has a ${(bottleneck.occurrences)} failures.`,
          estimatedImpact: 'Reduce retry overhead and improve success rate',
          implementation: 'Add better error handling and fallback strategies',
        });
      }
    }

    // General recommendations
    const avgRetries = this.mean(this.compilationData.map(t => t.retryCount));
    if (avgRetries > 0.5) {
      recommendations.push({
        id: 'reduce-retries',
        category: 'reliability',
        priority: 'medium',
        title: 'Reduce retry frequency',
        description: `Average retry count is ${avgRetries.toFixed(2)} per compilation.`,
        estimatedImpact: 'Improve reliability and reduce latency',
      });
    }

    const avgFallbacks = this.mean(this.compilationData.map(t => t.fallbacksUsed.length));
    if (avgFallbacks > 0.2) {
      recommendations.push({
        id: 'reduce-fallbacks',
        category: 'strategy',
        priority: 'medium',
        title: 'Improve strategy selection',
        description: `Fallback strategies are used ${(avgFallbacks * 100).toFixed(0)}% of the time.`,
        estimatedImpact: 'Better initial strategy selection can improve performance',
      });
    }

    return recommendations;
  }

  /**
   * Build time series data.
   */
  private buildTimeSeries(): AnalyticsSummary['timeSeries'] {
    const hourMs = 60 * 60 * 1000;
    const compilationsPerHour: DataPoint[] = [];
    const averageDurationPerHour: DataPoint[] = [];
    const errorRatePerHour: DataPoint[] = [];

    // Group by hour
    const hourBuckets = new Map<number, TelemetryData[]>();

    for (let i = 0; i < this.compilationData.length; i++) {
      const hourKey = Math.floor(this.timestamps[i] / hourMs) * hourMs;
      if (!hourBuckets.has(hourKey)) {
        hourBuckets.set(hourKey, []);
      }
      hourBuckets.get(hourKey)!.push(this.compilationData[i]);
    }

    // Calculate metrics per hour
    for (const [hour, data] of hourBuckets) {
      compilationsPerHour.push({
        timestamp: hour,
        value: data.length,
      });

      averageDurationPerHour.push({
        timestamp: hour,
        value: this.mean(data.map(t => t.totalDuration)),
      });

      const errors = data.filter(t => t.retryCount > 0 || t.fallbacksUsed.length > 0).length;
      errorRatePerHour.push({
        timestamp: hour,
        value: data.length > 0 ? errors / data.length : 0,
      });
    }

    return {
      compilationsPerHour,
      averageDurationPerHour,
      errorRatePerHour,
    };
  }

  /**
   * Calculate statistics for a dataset.
   */
  private calculateStatistics(values: number[]): Statistics {
    if (values.length === 0) {
      return {
        count: 0, sum: 0, min: 0, max: 0, mean: 0, median: 0, stdDev: 0,
        p50: 0, p90: 0, p95: 0, p99: 0,
      };
    }

    const sorted = [...values].sort((a, b) => a - b);
    const count = values.length;
    const sum = this.sum(values);
    const mean = sum / count;

    const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / count;
    const stdDev = Math.sqrt(variance);

    return {
      count,
      sum,
      min: sorted[0],
      max: sorted[count - 1],
      mean,
      median: this.percentile(sorted, 50),
      stdDev,
      p50: this.percentile(sorted, 50),
      p90: this.percentile(sorted, 90),
      p95: this.percentile(sorted, 95),
      p99: this.percentile(sorted, 99),
    };
  }

  /**
   * Calculate percentile.
   */
  private percentile(sorted: number[], p: number): number {
    if (sorted.length === 0) return 0;
    const idx = Math.ceil((p / 100) * sorted.length) - 1;
    return sorted[Math.max(0, idx)];
  }

  /**
   * Calculate sum.
   */
  private sum(values: number[]): number {
    return values.reduce((a, b) => a + b, 0);
  }

  /**
   * Calculate mean.
   */
  private mean(values: number[]): number {
    return values.length > 0 ? this.sum(values) / values.length : 0;
  }

  /**
   * Determine severity from duration.
   */
  private severityFromDuration(duration: number): Bottleneck['severity'] {
    if (duration > 30000) return 'critical';
    if (duration > 15000) return 'high';
    if (duration > 10000) return 'medium';
    return 'low';
  }

  /**
   * Determine severity from error rate.
   */
  private severityFromErrorRate(errorRate: number): Bottleneck['severity'] {
    if (errorRate > 0.5) return 'critical';
    if (errorRate > 0.25) return 'high';
    if (errorRate > 0.1) return 'medium';
    return 'low';
  }

  /**
   * Get optimization hint for phase.
   */
  private getOptimizationHint(phase: PhaseId): string {
    const hints: Record<PhaseId, string> = {
      perceive: 'Consider caching embeddings or using a faster encoder model',
      ground: 'Use incremental grounding and cache partial results',
      constrain: 'Apply constraint propagation early to prune search space',
      verify: 'Use incremental verification and parallelize gate checks',
      emit: 'Stream output generation and use efficient serialization',
      temporal: 'Index git history and use caching for repeated queries',
      solve: 'Apply learned clause caching and restart strategies',
    };
    return hints[phase] || 'Profile the phase to identify specific bottlenecks';
  }

  /**
   * Remove old data points.
   */
  private pruneOldData(): void {
    const cutoff = Date.now() - this.config.retentionPeriod;

    // Find first index to keep
    let firstKeep = 0;
    while (firstKeep < this.timestamps.length && this.timestamps[firstKeep] < cutoff) {
      firstKeep++;
    }

    // Prune main data
    if (firstKeep > 0) {
      this.timestamps = this.timestamps.slice(firstKeep);
      this.compilationData = this.compilationData.slice(firstKeep);
    }

    // Enforce max data points
    if (this.compilationData.length > this.config.maxDataPoints) {
      const excess = this.compilationData.length - this.config.maxDataPoints;
      this.timestamps = this.timestamps.slice(excess);
      this.compilationData = this.compilationData.slice(excess);
    }

    // Prune phase executions (keep last 1000 per phase)
    for (const [phase, executions] of this.phaseExecutions) {
      if (executions.length > 1000) {
        this.phaseExecutions.set(phase, executions.slice(-1000));
      }
    }
  }

  /**
   * Export analytics data.
   */
  export(): {
    compilations: TelemetryData[];
    phases: Record<string, PhaseExecution[]>;
    timestamps: number[];
  } {
    const phases: Record<string, PhaseExecution[]> = {};
    for (const [phase, execs] of this.phaseExecutions) {
      phases[phase] = execs;
    }

    return {
      compilations: [...this.compilationData],
      phases,
      timestamps: [...this.timestamps],
    };
  }

  /**
   * Import analytics data.
   */
  import(data: {
    compilations: TelemetryData[];
    phases: Record<string, PhaseExecution[]>;
    timestamps: number[];
  }): void {
    this.compilationData = [...data.compilations];
    this.timestamps = [...data.timestamps];
    this.phaseExecutions = new Map(
      Object.entries(data.phases) as [PhaseId, PhaseExecution[]][]
    );
  }

  /**
   * Reset analytics.
   */
  reset(): void {
    this.compilationData = [];
    this.timestamps = [];
    this.phaseExecutions.clear();
  }
}

/**
 * Create analytics instance.
 */
export function createAnalytics(config?: Partial<AnalyticsConfig>): CompilationAnalytics {
  return new CompilationAnalytics(config);
}
