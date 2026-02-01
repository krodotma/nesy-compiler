import { describe, it, expect } from 'vitest';
import { ResultAnalyzer } from '../result-analyzer.js';
import { NeSyCompiler } from '@nesy/pipeline';

describe('ResultAnalyzer', () => {
  it('analyzes real results', async () => {
    const c = new NeSyCompiler();
    const a = new ResultAnalyzer();
    a.add(await c.compile({ mode: 'atom', intent: 'a' }));
    a.add(await c.compile({ mode: 'atom', intent: 'b' }));
    const r = a.analyze();
    expect(r.totalCompilations).toBe(2);
    expect(r.passed + r.failed).toBe(2);
  });
});
