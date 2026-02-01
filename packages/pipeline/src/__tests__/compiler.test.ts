import { describe, it, expect } from 'vitest';
import { NeSyCompiler, createCompiler, compileText, compileAndVerify } from '../compiler.js';

describe('NeSyCompiler', () => {
  it('compiles atom e2e', async () => {
    const r = await createCompiler({ trace: true }).compile({ mode: 'atom', intent: 'test' });
    expect(r.ir.kind).toBe('compiled');
    expect(r.stages.perceive).toBeDefined();
    expect(r.metrics.totalDurationMs).toBeGreaterThanOrEqual(0);
  });
  it('compiles genesis', async () => {
    const r = await createCompiler().compile({ mode: 'genesis', specification: { n: 1 } });
    expect(r.ir.kind).toBe('compiled');
  });
  it('compiles constraint', async () => {
    const r = await createCompiler().compile({
      mode: 'constraint', constraints: [{ type: 'equality', left: { type: 'variable', name: 'X' }, right: { type: 'constant', name: 'a' } }],
    });
    expect(r.ir.kind).toBe('compiled');
  });
  it('compiles seed', async () => {
    const r = await createCompiler().compile({ mode: 'seed', baseHolon: 'h1', mutations: [{}] });
    expect(r.ir.kind).toBe('compiled');
  });
  it('resets', () => {
    const c = new NeSyCompiler();
    c.reset();
    expect(c.getContext().trace).toEqual([]);
  });
});

describe('compileText', () => {
  it('works', async () => {
    expect((await compileText('hi')).ir.kind).toBe('compiled');
  });
});

describe('compileAndVerify', () => {
  it('returns verified flag', async () => {
    const { verified } = await compileAndVerify('test');
    expect(typeof verified).toBe('boolean');
  });
});
