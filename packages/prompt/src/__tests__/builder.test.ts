import { describe, it, expect } from 'vitest';
import { PromptBuilder } from '../builder.js';

describe('PromptBuilder', () => {
  it('handles empty structure', () => {
    expect(new PromptBuilder().buildFromStructure({ terms: [], constraints: [] })).toContain('Terms');
  });

  it('renders all constraint types', () => {
    const p = new PromptBuilder().buildFromStructure({
      terms: [],
      constraints: [
        { type: 'equality', left: { type: 'variable', name: 'X' }, right: { type: 'constant', name: 'a' } },
        { type: 'inequality', left: { type: 'variable', name: 'Y' }, right: { type: 'constant', name: 'b' } },
        { type: 'membership', element: { type: 'variable', name: 'Z' }, set: [{ type: 'constant', name: 'a' }] },
        { type: 'custom', name: 'special', args: [{ type: 'constant', name: 'c' }] },
      ],
    });
    expect(p).toContain('equals');
    expect(p).toContain('differs');
    expect(p).toContain('one of');
    expect(p).toContain('special');
  });

  it('truncates with budget', () => {
    const p = new PromptBuilder({ maxTokens: 5 }).buildFromStructure({
      terms: Array.from({ length: 100 }, (_, i) => ({ type: 'constant' as const, name: `t${i}` })),
      constraints: [],
    });
    expect(p).toContain('[truncated]');
  });
});
