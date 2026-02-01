import type { Constraint, Term } from '@nesy/core';

/**
 * Adapts symbolic constraints to natural language for prompts
 */
export class ConstraintAdapter {
  toNaturalLanguage(constraint: Constraint): string {
    switch (constraint.type) {
      case 'equality':
        return `${this.termStr(constraint.left)} equals ${this.termStr(constraint.right)}`;
      case 'inequality':
        return `${this.termStr(constraint.left)} differs from ${this.termStr(constraint.right)}`;
      case 'membership':
        return `${this.termStr(constraint.element)} is one of {${constraint.set.map(t => this.termStr(t)).join(', ')}}`;
      case 'custom':
        return `${constraint.name}(${constraint.args.map(t => this.termStr(t)).join(', ')})`;
    }
  }

  private termStr(term: Term): string {
    switch (term.type) {
      case 'variable': return `?${term.name}`;
      case 'constant': return term.name;
      case 'compound': return `${term.functor}(${term.args.map(a => this.termStr(a)).join(', ')})`;
    }
  }
}
