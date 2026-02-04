/**
 * SAT Encoder - Convert constraints to CNF (Conjunctive Normal Form)
 *
 * Phase 3 Step 26: SAT/SMT Foundation
 *
 * Transforms symbolic constraints into boolean satisfiability problems,
 * enabling integration with SAT solvers like MiniSat, CaDiCaL, or Kissat.
 *
 * Key concepts:
 * - Tseitin transformation for efficient CNF conversion
 * - Variable assignment for term equality
 * - Clause generation from constraints
 */

import type { Term, Constraint, SymbolicStructure } from '@nesy/core';
import { isVariable, isConstant, isCompound, termToString } from '@nesy/core';

/** A literal is a variable or its negation */
export interface Literal {
  variable: number;
  negated: boolean;
}

/** A clause is a disjunction of literals */
export type Clause = Literal[];

/** CNF formula is a conjunction of clauses */
export interface CNFFormula {
  clauses: Clause[];
  variableCount: number;
  variableNames: Map<number, string>;
  termToVariable: Map<string, number>;
}

/** SAT solver result */
export interface SATResult {
  satisfiable: boolean;
  assignment?: Map<number, boolean>;
  termAssignment?: Map<string, boolean>;
  unsatCore?: Clause[];
}

/** Encoding configuration */
export interface EncoderConfig {
  /** Use Tseitin transformation for complex formulas */
  useTseitin: boolean;
  /** Maximum clause length before introducing auxiliary variables */
  maxClauseLength: number;
  /** Track provenance for UNSAT core extraction */
  trackProvenance: boolean;
}

const DEFAULT_CONFIG: EncoderConfig = {
  useTseitin: true,
  maxClauseLength: 3,
  trackProvenance: true,
};

/**
 * SAT Encoder: Transform symbolic constraints to CNF.
 */
export class SATEncoder {
  private config: EncoderConfig;
  private nextVariable: number = 1;
  private variableNames: Map<number, string> = new Map();
  private termToVariable: Map<string, number> = new Map();
  private clauseProvenance: Map<number, string> = new Map();

  constructor(config?: Partial<EncoderConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Encode a symbolic structure to CNF.
   */
  encode(structure: SymbolicStructure): CNFFormula {
    this.reset();
    const clauses: Clause[] = [];

    // Encode all constraints
    for (const constraint of structure.constraints) {
      const constraintClauses = this.encodeConstraint(constraint);
      clauses.push(...constraintClauses);
    }

    // Add term groundedness constraints
    for (const term of structure.terms) {
      const termClauses = this.encodeTermGroundedness(term);
      clauses.push(...termClauses);
    }

    return {
      clauses,
      variableCount: this.nextVariable - 1,
      variableNames: new Map(this.variableNames),
      termToVariable: new Map(this.termToVariable),
    };
  }

  /**
   * Encode a single constraint to clauses.
   */
  encodeConstraint(constraint: Constraint): Clause[] {
    switch (constraint.type) {
      case 'equality':
        return this.encodeEquality(constraint.left, constraint.right);
      case 'inequality':
        return this.encodeInequality(constraint.left, constraint.right);
      case 'membership':
        return this.encodeMembership(constraint.element, constraint.set);
      case 'custom':
        return this.encodeCustom(constraint.name, constraint.args);
      default:
        return [];
    }
  }

  /**
   * Encode equality constraint: left = right
   *
   * Uses the equivalence: (a = b) ↔ (a → b) ∧ (b → a)
   */
  private encodeEquality(left: Term, right: Term): Clause[] {
    const leftVar = this.getOrCreateVariable(termToString(left));
    const rightVar = this.getOrCreateVariable(termToString(right));

    // a = b encoded as: (¬a ∨ b) ∧ (a ∨ ¬b)
    return [
      [{ variable: leftVar, negated: true }, { variable: rightVar, negated: false }],
      [{ variable: leftVar, negated: false }, { variable: rightVar, negated: true }],
    ];
  }

  /**
   * Encode inequality constraint: left ≠ right
   *
   * (a ≠ b) ↔ (a ∧ ¬b) ∨ (¬a ∧ b)
   * In CNF: (a ∨ b) ∧ (¬a ∨ ¬b)
   */
  private encodeInequality(left: Term, right: Term): Clause[] {
    const leftVar = this.getOrCreateVariable(termToString(left));
    const rightVar = this.getOrCreateVariable(termToString(right));

    // XOR encoding: (a ∨ b) ∧ (¬a ∨ ¬b)
    return [
      [{ variable: leftVar, negated: false }, { variable: rightVar, negated: false }],
      [{ variable: leftVar, negated: true }, { variable: rightVar, negated: true }],
    ];
  }

  /**
   * Encode membership constraint: element ∈ set
   *
   * At least one of the set elements must equal the element.
   */
  private encodeMembership(element: Term, set: Term[]): Clause[] {
    const clauses: Clause[] = [];
    const elementVar = this.getOrCreateVariable(termToString(element));

    // Create auxiliary variables for each membership check
    const memberVars: number[] = [];
    for (const setElement of set) {
      const auxVar = this.createAuxVariable(`member_${termToString(element)}_${termToString(setElement)}`);
      memberVars.push(auxVar);

      // auxVar ↔ (element = setElement)
      const setVar = this.getOrCreateVariable(termToString(setElement));
      clauses.push(
        [{ variable: auxVar, negated: true }, { variable: elementVar, negated: true }, { variable: setVar, negated: false }],
        [{ variable: auxVar, negated: true }, { variable: elementVar, negated: false }, { variable: setVar, negated: true }]
      );
    }

    // At least one membership must hold
    clauses.push(memberVars.map(v => ({ variable: v, negated: false })));

    return clauses;
  }

  /**
   * Encode custom constraint as a fresh variable with implications.
   */
  private encodeCustom(name: string, args: Term[]): Clause[] {
    const customVar = this.createAuxVariable(`custom_${name}`);

    // The custom constraint variable must be true
    return [[{ variable: customVar, negated: false }]];
  }

  /**
   * Encode term groundedness (no free variables in final assignment).
   */
  private encodeTermGroundedness(term: Term): Clause[] {
    if (isVariable(term)) {
      // Variables should be bound
      const varId = this.getOrCreateVariable(termToString(term));
      // No clause needed - just track the variable
      return [];
    }

    if (isCompound(term)) {
      // Recursively encode arguments
      const clauses: Clause[] = [];
      for (const arg of term.args) {
        clauses.push(...this.encodeTermGroundedness(arg));
      }
      return clauses;
    }

    return [];
  }

  /**
   * Apply Tseitin transformation to complex formula.
   *
   * Introduces auxiliary variables to keep clauses small.
   */
  tseitinTransform(clauses: Clause[]): Clause[] {
    if (!this.config.useTseitin) {
      return clauses;
    }

    const result: Clause[] = [];

    for (const clause of clauses) {
      if (clause.length <= this.config.maxClauseLength) {
        result.push(clause);
      } else {
        // Split large clause using auxiliary variables
        result.push(...this.splitClause(clause));
      }
    }

    return result;
  }

  /**
   * Split a large clause into smaller ones using auxiliary variables.
   *
   * (a ∨ b ∨ c ∨ d) becomes:
   * (a ∨ b ∨ x) ∧ (¬x ∨ c ∨ d)
   */
  private splitClause(clause: Clause): Clause[] {
    if (clause.length <= this.config.maxClauseLength) {
      return [clause];
    }

    const mid = Math.floor(clause.length / 2);
    const left = clause.slice(0, mid);
    const right = clause.slice(mid);

    const auxVar = this.createAuxVariable(`split_${this.nextVariable}`);

    // Left part with positive aux
    left.push({ variable: auxVar, negated: false });
    // Right part with negative aux
    right.unshift({ variable: auxVar, negated: true });

    // Recursively split if still too large
    return [...this.splitClause(left), ...this.splitClause(right)];
  }

  /**
   * Get or create a variable for a term.
   */
  private getOrCreateVariable(termStr: string): number {
    if (!this.termToVariable.has(termStr)) {
      const varId = this.nextVariable++;
      this.termToVariable.set(termStr, varId);
      this.variableNames.set(varId, termStr);
    }
    return this.termToVariable.get(termStr)!;
  }

  /**
   * Create an auxiliary variable.
   */
  private createAuxVariable(name: string): number {
    const varId = this.nextVariable++;
    this.variableNames.set(varId, `aux:${name}`);
    return varId;
  }

  /**
   * Reset encoder state.
   */
  private reset(): void {
    this.nextVariable = 1;
    this.variableNames.clear();
    this.termToVariable.clear();
    this.clauseProvenance.clear();
  }

  /**
   * Convert CNF to DIMACS format (standard SAT solver input).
   */
  toDIMACS(formula: CNFFormula): string {
    const lines: string[] = [];

    // Header
    lines.push(`p cnf ${formula.variableCount} ${formula.clauses.length}`);

    // Comments with variable names
    for (const [varId, name] of formula.variableNames) {
      lines.push(`c ${varId} ${name}`);
    }

    // Clauses
    for (const clause of formula.clauses) {
      const literals = clause.map(lit =>
        lit.negated ? -lit.variable : lit.variable
      );
      lines.push(`${literals.join(' ')} 0`);
    }

    return lines.join('\n');
  }

  /**
   * Parse DIMACS solution format.
   */
  parseDIMACSResult(output: string): SATResult {
    const lines = output.trim().split('\n');

    // Check satisfiability
    const statusLine = lines.find(l => l.startsWith('s '));
    if (!statusLine) {
      return { satisfiable: false };
    }

    if (statusLine.includes('UNSATISFIABLE')) {
      return { satisfiable: false };
    }

    if (!statusLine.includes('SATISFIABLE')) {
      return { satisfiable: false };
    }

    // Parse variable assignments
    const assignment = new Map<number, boolean>();
    const valueLine = lines.find(l => l.startsWith('v '));

    if (valueLine) {
      const values = valueLine.substring(2).trim().split(/\s+/);
      for (const v of values) {
        const num = parseInt(v, 10);
        if (num === 0) break;
        assignment.set(Math.abs(num), num > 0);
      }
    }

    // Convert to term assignments
    const termAssignment = new Map<string, boolean>();
    for (const [termStr, varId] of this.termToVariable) {
      if (assignment.has(varId)) {
        termAssignment.set(termStr, assignment.get(varId)!);
      }
    }

    return {
      satisfiable: true,
      assignment,
      termAssignment,
    };
  }
}

/**
 * Quick function to encode constraints to DIMACS.
 */
export function constraintsToDIMACS(
  constraints: Constraint[],
  terms: Term[] = [],
  config?: Partial<EncoderConfig>
): string {
  const encoder = new SATEncoder(config);
  const formula = encoder.encode({ terms, constraints });
  return encoder.toDIMACS(formula);
}
