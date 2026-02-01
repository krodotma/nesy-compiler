import { describe, it, expect } from 'vitest';
import {
  variable, constant, compound,
  unify, applySubstitution, isGround, extractVariables,
  equalityConstraint, membershipConstraint,
  createSymbolicStructure, addTerm, addConstraint,
} from '../symbolic.js';

describe('Term constructors', () => {
  it('creates variable terms', () => {
    const v = variable('X');
    expect(v).toEqual({ type: 'variable', name: 'X' });
  });

  it('creates constant terms', () => {
    const c = constant('foo', 42);
    expect(c).toEqual({ type: 'constant', name: 'foo', value: 42 });
  });

  it('creates compound terms', () => {
    const t = compound('f', variable('X'), constant('a'));
    expect(t.type).toBe('compound');
    if (t.type === 'compound') {
      expect(t.functor).toBe('f');
      expect(t.args).toHaveLength(2);
    }
  });
});

describe('Unification', () => {
  it('unifies identical constants', () => {
    expect(unify(constant('a'), constant('a'))).toEqual({});
  });

  it('fails on different constants', () => {
    expect(unify(constant('a'), constant('b'))).toBeNull();
  });

  it('binds variable to constant', () => {
    const result = unify(variable('X'), constant('a'));
    expect(result).toEqual({ X: { type: 'constant', name: 'a' } });
  });

  it('unifies compound terms', () => {
    const t1 = compound('f', variable('X'), constant('b'));
    const t2 = compound('f', constant('a'), constant('b'));
    const result = unify(t1, t2);
    expect(result).not.toBeNull();
    expect(result!['X']).toEqual({ type: 'constant', name: 'a' });
  });

  it('fails on arity mismatch', () => {
    const t1 = compound('f', variable('X'));
    const t2 = compound('f', constant('a'), constant('b'));
    expect(unify(t1, t2)).toBeNull();
  });

  it('handles occurs check', () => {
    const t1 = variable('X');
    const t2 = compound('f', variable('X'));
    expect(unify(t1, t2)).toBeNull();
  });
});

describe('Substitution', () => {
  it('applies substitution to variable', () => {
    const result = applySubstitution(variable('X'), { X: constant('a') });
    expect(result).toEqual({ type: 'constant', name: 'a' });
  });

  it('leaves unbound variables unchanged', () => {
    const result = applySubstitution(variable('Y'), { X: constant('a') });
    expect(result).toEqual({ type: 'variable', name: 'Y' });
  });
});

describe('Term predicates', () => {
  it('isGround detects ground terms', () => {
    expect(isGround(constant('a'))).toBe(true);
    expect(isGround(variable('X'))).toBe(false);
    expect(isGround(compound('f', constant('a')))).toBe(true);
    expect(isGround(compound('f', variable('X')))).toBe(false);
  });

  it('extractVariables finds all variables', () => {
    const t = compound('f', variable('X'), compound('g', variable('Y'), variable('X')));
    expect(extractVariables(t)).toEqual(['X', 'Y']);
  });
});

describe('Constraints', () => {
  it('creates equality constraints', () => {
    const c = equalityConstraint(variable('X'), constant('a'));
    expect(c.type).toBe('equality');
  });

  it('creates membership constraints', () => {
    const c = membershipConstraint(variable('X'), [constant('a'), constant('b')]);
    expect(c.type).toBe('membership');
  });
});

describe('SymbolicStructure', () => {
  it('creates empty structure', () => {
    const s = createSymbolicStructure();
    expect(s.terms).toEqual([]);
    expect(s.constraints).toEqual([]);
  });

  it('adds terms and constraints', () => {
    let s = createSymbolicStructure();
    s = addTerm(s, variable('X'));
    s = addConstraint(s, equalityConstraint(variable('X'), constant('a')));
    expect(s.terms).toHaveLength(1);
    expect(s.constraints).toHaveLength(1);
  });
});
