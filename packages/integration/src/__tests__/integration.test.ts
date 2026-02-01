import { describe, it, expect } from 'vitest';
import { ResultAnalyzer } from '../result-analyzer.js';

describe('ResultAnalyzer', () => {
  it('returns empty report with no results', () => {
    const analyzer = new ResultAnalyzer();
    const report = analyzer.analyze();
    expect(report.totalCompilations).toBe(0);
    expect(report.passed).toBe(0);
  });
});
