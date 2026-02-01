import { describe, it, expect } from 'vitest';
import { perceive } from '../stages/perceive.js';
import { ground } from '../stages/ground.js';
import { constrain } from '../stages/constrain.js';
import { verify } from '../stages/verify.js';
import { emit } from '../stages/emit.js';
import { createContext } from '@nesy/core';

describe('Full pipeline stages', () => {
  it('PERCEIVE->GROUND->CONSTRAIN->VERIFY->EMIT', async () => {
    const ctx = createContext();
    const p = await perceive({ text: 'test' }, ctx);
    expect(p.ir.kind).toBe('neural');
    const g = await ground({ perceiveIR: p.ir, features: p.features }, ctx);
    expect(g.ir.kind).toBe('grounded');
    const c = await constrain({ groundIR: g.ir, symbols: g.symbols }, ctx);
    expect(c.ir.kind).toBe('constrained');
    const v = await verify({ constrainIR: c.ir, satisfied: c.satisfied }, ctx);
    expect(v.ir.kind).toBe('verified');
    const e = await emit({ verifyIR: v.ir, stageHistory: [] }, ctx);
    expect(e.ir.kind).toBe('compiled');
  });
});
