import type { CompilationResult } from '@nesy/pipeline';
import type { AnalysisReport } from './types.js';

/**
 * Analyzes compilation results for quality metrics
 */
export class ResultAnalyzer {
  private results: CompilationResult[] = [];

  add(result: CompilationResult): void {
    this.results.push(result);
  }

  analyze(): AnalysisReport {
    const total = this.results.length;
    if (total === 0) {
      return {
        totalCompilations: 0,
        passed: 0,
        failed: 0,
        averageDurationMs: 0,
        gatePassRates: {},
        trustDistribution: {},
      };
    }

    const passed = this.results.filter(r => r.stages.verify.passed).length;
    const avgDuration = this.results.reduce((sum, r) => sum + r.metrics.totalDurationMs, 0) / total;

    return {
      totalCompilations: total,
      passed,
      failed: total - passed,
      averageDurationMs: avgDuration,
      gatePassRates: this.computeGateRates(),
      trustDistribution: {},
    };
  }

  private computeGateRates(): Record<string, number> {
    const total = this.results.length;
    if (total === 0) return {};

    const gates = ['provenance', 'effects', 'liveness', 'recovery', 'quality', 'omega'];
    const rates: Record<string, number> = {};

    for (const gate of gates) {
      const passCount = this.results.filter(r => {
        const proof = r.stages.verify.proof;
        const gateResult = proof[gate as keyof typeof proof];
        return gateResult && typeof gateResult === 'object' && 'status' in gateResult && gateResult.status === 'passed';
      }).length;
      rates[gate] = passCount / total;
    }

    return rates;
  }
}
