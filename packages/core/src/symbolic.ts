/**
 * Symbolic layer primitives
 *
 * - Terms: first-order term representation
 * - Unification: pattern matching
 * - Rewriting: term transformation
 */

import type { Term, Substitution, Constraint, SymbolicStructure } from './types';

// =============================================================================
// Term Operations
// =============================================================================

export function isVariable(term: Term): term is { type: 'variable'; name: string } {
  return term.type === 'variable';
}

export function isConstant(term: Term): term is { type: 'constant'; name: string; value?: unknown } {
  return term.type === 'constant';
}

export function isCompound(term: Term): term is { type: 'compound'; functor: string; args: Term[] } {
  return term.type === 'compound';
}

export function variable(name: string): Term {
  return { type: 'variable', name };
}

export function constant(name: string, value?: unknown): Term {
  return { type: 'constant', name, value };
}

export function compound(functor: string, ...args: Term[]): Term {
  return { type: 'compound', functor, args };
}

export function termToString(term: Term): string {
  switch (term.type) {
    case 'variable':
      return `?${term.name}`;
    case 'constant':
      return term.name;
    case 'compound':
      return `${term.functor}(${term.args.map(termToString).join(', ')})`;
  }
}

// =============================================================================
// Substitution Operations
// =============================================================================

export function emptySubstitution(): Substitution {
  return {};
}

export function applySubstitution(term: Term, sub: Substitution): Term {
  switch (term.type) {
    case 'variable': {
      const bound = sub[term.name];
      return bound ? applySubstitution(bound, sub) : term;
    }
    case 'constant':
      return term;
    case 'compound':
      return {
        type: 'compound',
        functor: term.functor,
        args: term.args.map(arg => applySubstitution(arg, sub)),
      };
  }
}

export function composeSubstitution(s1: Substitution, s2: Substitution): Substitution {
  const result: Substitution = {};

  // Apply s2 to all terms in s1
  for (const [k, v] of Object.entries(s1)) {
    result[k] = applySubstitution(v, s2);
  }

  // Add bindings from s2 that aren't in s1
  for (const [k, v] of Object.entries(s2)) {
    if (!(k in result)) {
      result[k] = v;
    }
  }

  return result;
}

// =============================================================================
// Unification
// =============================================================================

export function occursCheck(varName: string, term: Term): boolean {
  switch (term.type) {
    case 'variable':
      return term.name === varName;
    case 'constant':
      return false;
    case 'compound':
      return term.args.some(arg => occursCheck(varName, arg));
  }
}

export function unify(t1: Term, t2: Term, sub: Substitution = {}): Substitution | null {
  const s1 = applySubstitution(t1, sub);
  const s2 = applySubstitution(t2, sub);

  if (isVariable(s1)) {
    if (isVariable(s2) && s1.name === s2.name) {
      return sub;
    }
    if (occursCheck(s1.name, s2)) {
      return null; // Occurs check failure
    }
    return { ...sub, [s1.name]: s2 };
  }

  if (isVariable(s2)) {
    return unify(s2, s1, sub);
  }

  if (isConstant(s1) && isConstant(s2)) {
    return s1.name === s2.name ? sub : null;
  }

  if (isCompound(s1) && isCompound(s2)) {
    if (s1.functor !== s2.functor || s1.args.length !== s2.args.length) {
      return null;
    }

    let currentSub = sub;
    for (let i = 0; i < s1.args.length; i++) {
      const result = unify(s1.args[i], s2.args[i], currentSub);
      if (result === null) {
        return null;
      }
      currentSub = result;
    }
    return currentSub;
  }

  return null;
}

// =============================================================================
// Term Utilities
// =============================================================================

function hasVariables(term: Term): boolean {
  switch (term.type) {
    case 'variable':
      return true;
    case 'constant':
      return false;
    case 'compound':
      return term.args.some(hasVariables);
  }
}

export function isGround(term: Term): boolean {
  return !hasVariables(term);
}

export function extractVariables(term: Term): string[] {
  const vars: string[] = [];

  function extract(t: Term): void {
    switch (t.type) {
      case 'variable':
        if (!vars.includes(t.name)) {
          vars.push(t.name);
        }
        break;
      case 'constant':
        break;
      case 'compound':
        t.args.forEach(extract);
        break;
    }
  }

  extract(term);
  return vars;
}

// =============================================================================
// Constraint Operations
// =============================================================================

export function equalityConstraint(left: Term, right: Term): Constraint {
  return { type: 'equality', left, right };
}

export function inequalityConstraint(left: Term, right: Term): Constraint {
  return { type: 'inequality', left, right };
}

export function membershipConstraint(element: Term, set: Term[]): Constraint {
  return { type: 'membership', element, set };
}

export function customConstraint(name: string, args: Term[]): Constraint {
  return { type: 'custom', name, args };
}

// =============================================================================
// Symbolic Structure Operations
// =============================================================================

export function createSymbolicStructure(
  terms: Term[] = [],
  constraints: Constraint[] = []
): SymbolicStructure {
  return {
    terms,
    constraints,
    metadata: {},
  };
}

export function addTerm(
  structure: SymbolicStructure,
  term: Term
): SymbolicStructure {
  return {
    ...structure,
    terms: [...structure.terms, term],
  };
}

export function addConstraint(
  structure: SymbolicStructure,
  constraint: Constraint
): SymbolicStructure {
  return {
    ...structure,
    constraints: [...structure.constraints, constraint],
  };
}
