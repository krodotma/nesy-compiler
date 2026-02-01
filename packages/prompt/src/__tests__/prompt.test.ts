import { describe, it, expect } from 'vitest';
import { PromptBuilder } from '../builder.js';
import { PromptTemplate } from '../template.js';
import { ConstraintAdapter } from '../constraint-adapter.js';
import { TokenBudget } from '../token-budget.js';
import { FewShotManager } from '../few-shot.js';
import type { SymbolicStructure, Constraint } from '@nesy/core';

describe('PromptBuilder', () => {
  it('builds prompt from structure', () => {
    const structure: SymbolicStructure = {
      terms: [{ type: 'variable', name: 'X' }, { type: 'constant', name: 'foo' }],
      constraints: [{ type: 'equality', left: { type: 'variable', name: 'X' }, right: { type: 'constant', name: 'foo' } }],
    };
    const builder = new PromptBuilder();
    const prompt = builder.buildFromStructure(structure);
    expect(prompt).toContain('Terms');
    expect(prompt).toContain('?X');
    expect(prompt).toContain('Constraints');
  });
});

describe('PromptTemplate', () => {
  it('renders template with values', () => {
    const t = new PromptTemplate('Hello {{name}}, you are {{role}}', [
      { name: 'name', required: true },
      { name: 'role', required: false, defaultValue: 'user' },
    ]);
    expect(t.render({ name: 'Alice' })).toBe('Hello Alice, you are user');
    expect(t.render({ name: 'Bob', role: 'admin' })).toBe('Hello Bob, you are admin');
  });

  it('throws on missing required slot', () => {
    const t = new PromptTemplate('{{x}}', [{ name: 'x', required: true }]);
    expect(() => t.render({})).toThrow('Missing required slot');
  });
});

describe('ConstraintAdapter', () => {
  it('converts equality to natural language', () => {
    const adapter = new ConstraintAdapter();
    const c: Constraint = { type: 'equality', left: { type: 'variable', name: 'X' }, right: { type: 'constant', name: 'a' } };
    expect(adapter.toNaturalLanguage(c)).toContain('equals');
  });
});

describe('TokenBudget', () => {
  it('estimates tokens', () => {
    const budget = new TokenBudget(100);
    expect(budget.estimateTokens('hello world')).toBeGreaterThan(0);
  });

  it('truncates long text', () => {
    const budget = new TokenBudget(5);
    const result = budget.truncate('a'.repeat(1000));
    expect(result).toContain('[truncated]');
  });
});

describe('FewShotManager', () => {
  it('adds and selects examples', () => {
    const mgr = new FewShotManager();
    mgr.add({ input: 'a', output: 'b', tags: ['math'] });
    mgr.add({ input: 'c', output: 'd', tags: ['logic'] });
    expect(mgr.select(['math'])).toHaveLength(1);
    expect(mgr.getAll()).toHaveLength(2);
  });

  it('formats examples', () => {
    const mgr = new FewShotManager();
    mgr.add({ input: 'x', output: 'y' });
    const formatted = mgr.format(mgr.getAll());
    expect(formatted).toContain('Example 1');
  });
});
