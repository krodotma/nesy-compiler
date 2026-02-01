import type { SymbolicStructure, Constraint, Term } from '@nesy/core';
import { PromptTemplate, type TemplateSlot } from './template.js';
import { ConstraintAdapter } from './constraint-adapter.js';
import { TokenBudget } from './token-budget.js';

export interface PromptBuilderConfig {
  maxTokens?: number;
  includeConstraints?: boolean;
  includeExamples?: boolean;
}

/**
 * Builds structured prompts from symbolic structures
 */
export class PromptBuilder {
  private config: Required<PromptBuilderConfig>;
  private adapter: ConstraintAdapter;
  private budget: TokenBudget;

  constructor(config: PromptBuilderConfig = {}) {
    this.config = {
      maxTokens: 4096,
      includeConstraints: true,
      includeExamples: false,
      ...config,
    };
    this.adapter = new ConstraintAdapter();
    this.budget = new TokenBudget(this.config.maxTokens);
  }

  buildFromStructure(structure: SymbolicStructure): string {
    const parts: string[] = [];

    parts.push('## Terms');
    for (const term of structure.terms) {
      parts.push(`- ${this.termToString(term)}`);
    }

    if (this.config.includeConstraints && structure.constraints.length > 0) {
      parts.push('');
      parts.push('## Constraints');
      for (const c of structure.constraints) {
        parts.push(`- ${this.adapter.toNaturalLanguage(c)}`);
      }
    }

    const prompt = parts.join('\n');
    return this.budget.truncate(prompt);
  }

  private termToString(term: Term): string {
    switch (term.type) {
      case 'variable': return `?${term.name}`;
      case 'constant': return term.name;
      case 'compound': return `${term.functor}(${term.args.map(a => this.termToString(a)).join(', ')})`;
    }
  }
}
