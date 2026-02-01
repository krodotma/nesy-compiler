import { describe, it, expect } from 'vitest';
import { createIRNode, hashIR } from '../ir.js';

describe('createIRNode', () => {
  it('creates a node with correct kind', () => {
    const node = createIRNode('neural', 'abc', 'def');
    expect(node.kind).toBe('neural');
    expect(node.inputHash).toBe('abc');
    expect(node.outputHash).toBe('def');
    expect(node.id).toContain('neural_');
  });
});

describe('hashIR', () => {
  it('produces consistent hashes', () => {
    const node = createIRNode('grounded', 'a', 'b');
    const h1 = hashIR(node);
    const h2 = hashIR(node);
    expect(h1).toBe(h2);
  });

  it('produces different hashes for different nodes', () => {
    const n1 = createIRNode('neural', 'a', 'b');
    const n2 = createIRNode('compiled', 'a', 'b');
    expect(hashIR(n1)).not.toBe(hashIR(n2));
  });
});
