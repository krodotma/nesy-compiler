import { describe, it, expect } from 'vitest';
import { createContext, addTrace, elevateContext, taintContext } from '../context.js';
import { TrustLevel } from '../types.js';

describe('createContext', () => {
  it('creates with defaults', () => {
    const ctx = createContext();
    expect(ctx.model).toBe('claude-opus-4-5');
    expect(ctx.trustLevel).toBe(TrustLevel.UNTRUSTED);
  });
  it('accepts overrides', () => {
    const ctx = createContext({ model: 'test', maxIterations: 50 });
    expect(ctx.model).toBe('test');
  });
});

describe('addTrace', () => {
  it('appends immutably', () => {
    const c1 = createContext();
    const c2 = addTrace(c1, 'perceive', 'in', 'out', 42);
    expect(c1.trace).toHaveLength(0);
    expect(c2.trace).toHaveLength(1);
  });
});

describe('elevateContext', () => {
  it('elevates trust', () => {
    const ctx = elevateContext(createContext(), TrustLevel.STANDARD);
    expect(ctx.trustLevel).toBe(TrustLevel.STANDARD);
  });
  it('refuses to lower', () => {
    const ctx = createContext({ trustLevel: TrustLevel.STANDARD });
    expect(elevateContext(ctx, TrustLevel.UNTRUSTED).trustLevel).toBe(TrustLevel.STANDARD);
  });
});

describe('taintContext', () => {
  it('adds taint', () => {
    expect(taintContext(createContext(), 'ext').taintVector).toContain('ext');
  });
});
