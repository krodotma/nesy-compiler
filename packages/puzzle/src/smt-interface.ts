/**
 * SMT Interface - Satisfiability Modulo Theories
 *
 * Phase 3 Step 27: SMT Solver Integration
 *
 * Provides a Z3-style interface for expressing and solving constraints
 * involving theories like arithmetic, arrays, bitvectors, and datatypes.
 *
 * Key features:
 * - Theory-aware constraint encoding
 * - Incremental solving with push/pop
 * - Model extraction and interpolation
 * - UNSAT core computation
 */

import type { Term, Constraint } from '@nesy/core';
import { constant, compound, variable, termToString } from '@nesy/core';

/** SMT-LIB2 sorts (types) */
export type Sort =
  | { kind: 'bool' }
  | { kind: 'int' }
  | { kind: 'real' }
  | { kind: 'bitvec'; width: number }
  | { kind: 'array'; domain: Sort; range: Sort }
  | { kind: 'datatype'; name: string }
  | { kind: 'uninterpreted'; name: string };

/** SMT expression */
export interface SMTExpr {
  sort: Sort;
  sexp: string; // S-expression representation
}

/** SMT declaration */
export interface SMTDeclaration {
  name: string;
  sort: Sort;
  definition?: SMTExpr;
}

/** SMT assertion */
export interface SMTAssertion {
  id: string;
  expr: SMTExpr;
  label?: string;
}

/** Solver check result */
export type CheckResult = 'sat' | 'unsat' | 'unknown';

/** SMT model (satisfying assignment) */
export interface SMTModel {
  assignments: Map<string, SMTExpr>;
  interpretations: Map<string, (args: SMTExpr[]) => SMTExpr>;
}

/** UNSAT core */
export interface UnsatCore {
  assertions: string[];
  minimal: boolean;
}

/** Solver configuration */
export interface SMTConfig {
  logic: SMTLogic;
  timeout?: number;
  randomSeed?: number;
  produceModels: boolean;
  produceUnsatCores: boolean;
  incrementalMode: boolean;
}

/** Standard SMT-LIB logics */
export type SMTLogic =
  | 'QF_UF'      // Quantifier-free uninterpreted functions
  | 'QF_LIA'    // Quantifier-free linear integer arithmetic
  | 'QF_LRA'    // Quantifier-free linear real arithmetic
  | 'QF_BV'     // Quantifier-free bitvectors
  | 'QF_AUFLIA' // QF arrays + uninterpreted + linear int
  | 'ALL';      // All theories

const DEFAULT_CONFIG: SMTConfig = {
  logic: 'QF_UF',
  produceModels: true,
  produceUnsatCores: true,
  incrementalMode: true,
};

/**
 * SMT Solver Interface: Build and solve SMT formulas.
 */
export class SMTSolver {
  private config: SMTConfig;
  private declarations: Map<string, SMTDeclaration> = new Map();
  private assertions: SMTAssertion[] = [];
  private assertionStack: SMTAssertion[][] = [];
  private nextAssertionId = 0;

  constructor(config?: Partial<SMTConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  // ===== Sort Constructors =====

  static Bool(): Sort { return { kind: 'bool' }; }
  static Int(): Sort { return { kind: 'int' }; }
  static Real(): Sort { return { kind: 'real' }; }
  static BitVec(width: number): Sort { return { kind: 'bitvec', width }; }
  static Array(domain: Sort, range: Sort): Sort { return { kind: 'array', domain, range }; }

  // ===== Variable Declaration =====

  /**
   * Declare a constant (0-ary function).
   */
  declareConst(name: string, sort: Sort): SMTExpr {
    this.declarations.set(name, { name, sort });
    return { sort, sexp: name };
  }

  /**
   * Declare a function.
   */
  declareFunction(name: string, argSorts: Sort[], resultSort: Sort): void {
    // Functions are represented as uninterpreted in the declaration
    this.declarations.set(name, {
      name,
      sort: resultSort,
    });
  }

  /**
   * Define a constant with a value.
   */
  defineConst(name: string, value: SMTExpr): SMTExpr {
    this.declarations.set(name, { name, sort: value.sort, definition: value });
    return { sort: value.sort, sexp: name };
  }

  // ===== Expression Constructors =====

  // Boolean
  true_(): SMTExpr { return { sort: SMTSolver.Bool(), sexp: 'true' }; }
  false_(): SMTExpr { return { sort: SMTSolver.Bool(), sexp: 'false' }; }
  not(e: SMTExpr): SMTExpr { return { sort: SMTSolver.Bool(), sexp: `(not ${e.sexp})` }; }
  and(...es: SMTExpr[]): SMTExpr {
    if (es.length === 0) return this.true_();
    if (es.length === 1) return es[0];
    return { sort: SMTSolver.Bool(), sexp: `(and ${es.map(e => e.sexp).join(' ')})` };
  }
  or(...es: SMTExpr[]): SMTExpr {
    if (es.length === 0) return this.false_();
    if (es.length === 1) return es[0];
    return { sort: SMTSolver.Bool(), sexp: `(or ${es.map(e => e.sexp).join(' ')})` };
  }
  implies(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(=> ${a.sexp} ${b.sexp})` };
  }
  iff(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(= ${a.sexp} ${b.sexp})` };
  }
  ite(cond: SMTExpr, then_: SMTExpr, else_: SMTExpr): SMTExpr {
    return { sort: then_.sort, sexp: `(ite ${cond.sexp} ${then_.sexp} ${else_.sexp})` };
  }

  // Equality
  eq(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(= ${a.sexp} ${b.sexp})` };
  }
  distinct(...es: SMTExpr[]): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(distinct ${es.map(e => e.sexp).join(' ')})` };
  }

  // Arithmetic
  intVal(n: number): SMTExpr { return { sort: SMTSolver.Int(), sexp: String(n) }; }
  realVal(n: number): SMTExpr { return { sort: SMTSolver.Real(), sexp: n.toString() }; }
  add(...es: SMTExpr[]): SMTExpr {
    const sort = es[0]?.sort || SMTSolver.Int();
    return { sort, sexp: `(+ ${es.map(e => e.sexp).join(' ')})` };
  }
  sub(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: a.sort, sexp: `(- ${a.sexp} ${b.sexp})` };
  }
  mul(...es: SMTExpr[]): SMTExpr {
    const sort = es[0]?.sort || SMTSolver.Int();
    return { sort, sexp: `(* ${es.map(e => e.sexp).join(' ')})` };
  }
  div(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: a.sort, sexp: `(div ${a.sexp} ${b.sexp})` };
  }
  mod(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Int(), sexp: `(mod ${a.sexp} ${b.sexp})` };
  }
  neg(a: SMTExpr): SMTExpr {
    return { sort: a.sort, sexp: `(- ${a.sexp})` };
  }

  // Comparisons
  lt(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(< ${a.sexp} ${b.sexp})` };
  }
  le(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(<= ${a.sexp} ${b.sexp})` };
  }
  gt(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(> ${a.sexp} ${b.sexp})` };
  }
  ge(a: SMTExpr, b: SMTExpr): SMTExpr {
    return { sort: SMTSolver.Bool(), sexp: `(>= ${a.sexp} ${b.sexp})` };
  }

  // Arrays
  select(arr: SMTExpr, idx: SMTExpr): SMTExpr {
    if (arr.sort.kind !== 'array') {
      throw new Error('select requires array sort');
    }
    return { sort: arr.sort.range, sexp: `(select ${arr.sexp} ${idx.sexp})` };
  }
  store(arr: SMTExpr, idx: SMTExpr, val: SMTExpr): SMTExpr {
    return { sort: arr.sort, sexp: `(store ${arr.sexp} ${idx.sexp} ${val.sexp})` };
  }

  // Uninterpreted functions
  apply(fname: string, ...args: SMTExpr[]): SMTExpr {
    const decl = this.declarations.get(fname);
    const sort = decl?.sort || SMTSolver.Bool();
    if (args.length === 0) {
      return { sort, sexp: fname };
    }
    return { sort, sexp: `(${fname} ${args.map(a => a.sexp).join(' ')})` };
  }

  // ===== Assertions =====

  /**
   * Assert a formula.
   */
  assert(expr: SMTExpr, label?: string): string {
    const id = `a${this.nextAssertionId++}`;
    this.assertions.push({ id, expr, label });
    return id;
  }

  /**
   * Assert a named formula (for UNSAT core tracking).
   */
  assertNamed(expr: SMTExpr, name: string): string {
    const id = `a${this.nextAssertionId++}`;
    const namedExpr: SMTExpr = {
      sort: expr.sort,
      sexp: `(! ${expr.sexp} :named ${name})`,
    };
    this.assertions.push({ id, expr: namedExpr, label: name });
    return id;
  }

  // ===== Incremental Solving =====

  /**
   * Push assertion stack level.
   */
  push(): void {
    this.assertionStack.push([...this.assertions]);
  }

  /**
   * Pop assertion stack level.
   */
  pop(): void {
    const saved = this.assertionStack.pop();
    if (saved) {
      this.assertions = saved;
    }
  }

  /**
   * Reset solver state.
   */
  reset(): void {
    this.declarations.clear();
    this.assertions = [];
    this.assertionStack = [];
    this.nextAssertionId = 0;
  }

  // ===== Solving =====

  /**
   * Check satisfiability (stub - would invoke actual solver).
   */
  async check(): Promise<CheckResult> {
    // In a real implementation, this would:
    // 1. Generate SMT-LIB2 script
    // 2. Invoke Z3/CVC5/etc.
    // 3. Parse result

    // For now, return placeholder based on assertion count
    if (this.assertions.length === 0) {
      return 'sat';
    }

    // Simple contradiction detection
    const hasTrue = this.assertions.some(a => a.expr.sexp === 'true');
    const hasFalse = this.assertions.some(a => a.expr.sexp === 'false');
    if (hasFalse) return 'unsat';

    return 'unknown';
  }

  /**
   * Get model if satisfiable.
   */
  async getModel(): Promise<SMTModel | null> {
    const result = await this.check();
    if (result !== 'sat') {
      return null;
    }

    // Placeholder model
    return {
      assignments: new Map(),
      interpretations: new Map(),
    };
  }

  /**
   * Get UNSAT core if unsatisfiable.
   */
  async getUnsatCore(): Promise<UnsatCore | null> {
    const result = await this.check();
    if (result !== 'unsat') {
      return null;
    }

    // Return all named assertions as placeholder
    const namedAssertions = this.assertions
      .filter(a => a.label)
      .map(a => a.label!);

    return {
      assertions: namedAssertions,
      minimal: false,
    };
  }

  // ===== SMT-LIB2 Generation =====

  /**
   * Generate SMT-LIB2 script.
   */
  toSMTLIB2(): string {
    const lines: string[] = [];

    // Set logic
    lines.push(`(set-logic ${this.config.logic})`);

    // Options
    if (this.config.produceModels) {
      lines.push('(set-option :produce-models true)');
    }
    if (this.config.produceUnsatCores) {
      lines.push('(set-option :produce-unsat-cores true)');
    }

    // Declarations
    for (const [name, decl] of this.declarations) {
      const sortStr = this.sortToSMTLIB2(decl.sort);
      if (decl.definition) {
        lines.push(`(define-const ${name} ${sortStr} ${decl.definition.sexp})`);
      } else {
        lines.push(`(declare-const ${name} ${sortStr})`);
      }
    }

    // Assertions
    for (const assertion of this.assertions) {
      lines.push(`(assert ${assertion.expr.sexp})`);
    }

    // Check
    lines.push('(check-sat)');
    if (this.config.produceModels) {
      lines.push('(get-model)');
    }

    return lines.join('\n');
  }

  /**
   * Convert sort to SMT-LIB2 string.
   */
  private sortToSMTLIB2(sort: Sort): string {
    switch (sort.kind) {
      case 'bool': return 'Bool';
      case 'int': return 'Int';
      case 'real': return 'Real';
      case 'bitvec': return `(_ BitVec ${sort.width})`;
      case 'array': return `(Array ${this.sortToSMTLIB2(sort.domain)} ${this.sortToSMTLIB2(sort.range)})`;
      case 'datatype': return sort.name;
      case 'uninterpreted': return sort.name;
    }
  }

  // ===== Constraint Translation =====

  /**
   * Translate a NeSy constraint to SMT expression.
   */
  translateConstraint(constraint: Constraint): SMTExpr {
    switch (constraint.type) {
      case 'equality':
        return this.eq(
          this.translateTerm(constraint.left),
          this.translateTerm(constraint.right)
        );
      case 'inequality':
        return this.distinct(
          this.translateTerm(constraint.left),
          this.translateTerm(constraint.right)
        );
      case 'membership':
        // element ∈ set encoded as disjunction
        const elem = this.translateTerm(constraint.element);
        const memberExprs = constraint.set.map(s =>
          this.eq(elem, this.translateTerm(s))
        );
        return this.or(...memberExprs);
      case 'custom':
        // Custom constraint as uninterpreted predicate
        const args = constraint.args.map(a => this.translateTerm(a));
        return this.apply(constraint.name, ...args);
    }
  }

  /**
   * Translate a term to SMT expression.
   */
  translateTerm(term: Term): SMTExpr {
    const termStr = termToString(term);

    // Check if already declared
    if (!this.declarations.has(termStr)) {
      // Declare as uninterpreted constant
      this.declareConst(termStr, { kind: 'uninterpreted', name: 'Term' });
    }

    return { sort: { kind: 'uninterpreted', name: 'Term' }, sexp: termStr };
  }
}

/**
 * Quick function to check constraint satisfiability.
 */
export function checkConstraints(
  constraints: Constraint[],
  config?: Partial<SMTConfig>
): Promise<CheckResult> {
  const solver = new SMTSolver(config);

  for (const constraint of constraints) {
    solver.assert(solver.translateConstraint(constraint));
  }

  return solver.check();
}
