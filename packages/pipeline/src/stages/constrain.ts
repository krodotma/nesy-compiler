/**
 * CONSTRAIN Stage
 *
 * Input: SymbolicStructure (from GROUND)
 * Output: Satisfied constraints + search trace
 *
 * This stage handles:
 * - Constraint propagation
 * - Backtracking search
 * - Conflict detection
 * - Solution enumeration
 */

import type { CompilationContext, SymbolicStructure, Term, Constraint } from '@nesy/core';
import { unify, applySubstitution } from '@nesy/core';
import type { GroundIR, ConstrainIR } from '../ir';
import { createIRNode } from '../ir';

export interface ConstrainInput {
  groundIR: GroundIR;
  symbols: SymbolicStructure;
  externalConstraints?: Constraint[];
}

export interface ConstrainOutput {
  ir: ConstrainIR;
  satisfied: SymbolicStructure;
  unsatisfied: string[];
}

/**
 * Main constraint satisfaction function
 *
 * Uses constraint propagation with backtracking to find satisfying assignments
 */
export async function constrain(
  input: ConstrainInput,
  context: CompilationContext
): Promise<ConstrainOutput> {
  const { symbols, externalConstraints = [] } = input;

  // Merge internal and external constraints
  const allConstraints: Constraint[] = [
    ...symbols.constraints,
    ...externalConstraints,
  ];

  // Initialize search state
  const state: SearchState = {
    terms: [...symbols.terms],
    substitution: new Map(),
    constraints: allConstraints,
    steps: 0,
    maxSteps: context.maxIterations,
  };

  // Run constraint propagation
  const result = await propagateConstraints(state, context);

  // Build satisfied structure
  const satisfiedTerms = result.terms.map(t =>
    applySubstitution(t, Object.fromEntries(result.substitution))
  );

  const satisfied: SymbolicStructure = {
    terms: satisfiedTerms,
    constraints: result.satisfiedConstraints,
    metadata: {
      ...symbols.metadata,
      searchSteps: result.steps,
      substitutionSize: result.substitution.size,
    },
  };

  const ir: ConstrainIR = {
    ...createIRNode('constrained', input.groundIR.outputHash, hashConstrained(satisfied)),
    kind: 'constrained',
    satisfied,
    unsatisfied: result.unsatisfiedConstraints.map(c => constraintToString(c)),
    searchSteps: result.steps,
  };

  return { ir, satisfied, unsatisfied: ir.unsatisfied };
}

// =============================================================================
// Search State
// =============================================================================

interface SearchState {
  terms: Term[];
  substitution: Map<string, Term>;
  constraints: Constraint[];
  steps: number;
  maxSteps: number;
}

interface PropagationResult {
  terms: Term[];
  substitution: Map<string, Term>;
  satisfiedConstraints: Constraint[];
  unsatisfiedConstraints: Constraint[];
  steps: number;
}

// =============================================================================
// Constraint Propagation
// =============================================================================

async function propagateConstraints(
  state: SearchState,
  context: CompilationContext
): Promise<PropagationResult> {
  const satisfied: Constraint[] = [];
  const unsatisfied: Constraint[] = [];
  let changed = true;

  while (changed && state.steps < state.maxSteps) {
    changed = false;
    state.steps++;

    for (const constraint of state.constraints) {
      if (satisfied.includes(constraint) || unsatisfied.includes(constraint)) {
        continue;
      }

      const result = tryConstraint(constraint, state);

      switch (result.status) {
        case 'satisfied':
          satisfied.push(constraint);
          // Merge any new bindings
          for (const [k, v] of result.bindings.entries()) {
            if (!state.substitution.has(k)) {
              state.substitution.set(k, v);
              changed = true;
            }
          }
          break;

        case 'violated':
          // Try backtracking
          const backtrackResult = await backtrack(constraint, state, context);
          if (backtrackResult.success) {
            state.substitution = backtrackResult.newSubstitution;
            changed = true;
          } else {
            unsatisfied.push(constraint);
          }
          break;

        case 'pending':
          // Need more information, skip for now
          break;
      }
    }
  }

  // Any remaining pending constraints are unsatisfied
  for (const constraint of state.constraints) {
    if (!satisfied.includes(constraint) && !unsatisfied.includes(constraint)) {
      unsatisfied.push(constraint);
    }
  }

  return {
    terms: state.terms,
    substitution: state.substitution,
    satisfiedConstraints: satisfied,
    unsatisfiedConstraints: unsatisfied,
    steps: state.steps,
  };
}

// =============================================================================
// Constraint Evaluation
// =============================================================================

interface ConstraintResult {
  status: 'satisfied' | 'violated' | 'pending';
  bindings: Map<string, Term>;
}

function tryConstraint(
  constraint: Constraint,
  state: SearchState
): ConstraintResult {
  switch (constraint.type) {
    case 'equality':
      return tryEqualityConstraint(constraint, state);

    case 'inequality':
      return tryInequalityConstraint(constraint, state);

    case 'membership':
      return tryMembershipConstraint(constraint, state);

    case 'custom':
      return tryCustomConstraint(constraint, state);

    default:
      return { status: 'pending', bindings: new Map() };
  }
}

function tryEqualityConstraint(
  constraint: Constraint & { type: 'equality' },
  state: SearchState
): ConstraintResult {
  const left = resolveTermWithSubst(constraint.left, state.substitution);
  const right = resolveTermWithSubst(constraint.right, state.substitution);

  const unifyResult = unify(left, right);

  if (unifyResult === null) {
    return { status: 'violated', bindings: new Map() };
  }

  // Check if we have enough ground terms
  if (hasUnboundVariables(left) && hasUnboundVariables(right)) {
    const bindings = new Map<string, Term>();
    for (const [k, v] of Object.entries(unifyResult)) {
      bindings.set(k, v);
    }
    return { status: 'satisfied', bindings };
  }

  return { status: 'satisfied', bindings: new Map() };
}

function tryInequalityConstraint(
  constraint: Constraint & { type: 'inequality' },
  state: SearchState
): ConstraintResult {
  const left = resolveTermWithSubst(constraint.left, state.substitution);
  const right = resolveTermWithSubst(constraint.right, state.substitution);

  // Can only check if both are ground
  if (hasUnboundVariables(left) || hasUnboundVariables(right)) {
    return { status: 'pending', bindings: new Map() };
  }

  const unifyResult = unify(left, right);

  // Inequality satisfied if unification fails
  if (unifyResult === null) {
    return { status: 'satisfied', bindings: new Map() };
  }

  // Terms are equal, constraint violated
  return { status: 'violated', bindings: new Map() };
}

function tryMembershipConstraint(
  constraint: Constraint & { type: 'membership' },
  state: SearchState
): ConstraintResult {
  const element = resolveTermWithSubst(constraint.element, state.substitution);

  if (hasUnboundVariables(element)) {
    // Try to bind to first member of set
    if (constraint.set.length > 0) {
      const bindings = new Map<string, Term>();
      if (element.type === 'variable') {
        bindings.set(element.name, constraint.set[0]);
      }
      return { status: 'satisfied', bindings };
    }
    return { status: 'pending', bindings: new Map() };
  }

  // Check membership
  for (const member of constraint.set) {
    const unifyResult = unify(element, member);
    if (unifyResult !== null) {
      return { status: 'satisfied', bindings: new Map() };
    }
  }

  return { status: 'violated', bindings: new Map() };
}

function tryCustomConstraint(
  constraint: Constraint & { type: 'custom' },
  state: SearchState
): ConstraintResult {
  // Custom constraints need external evaluation
  // For now, mark as pending
  return { status: 'pending', bindings: new Map() };
}

// =============================================================================
// Backtracking
// =============================================================================

interface BacktrackResult {
  success: boolean;
  newSubstitution: Map<string, Term>;
}

async function backtrack(
  failedConstraint: Constraint,
  state: SearchState,
  context: CompilationContext
): Promise<BacktrackResult> {
  // Simple backtracking: try alternative bindings for involved variables
  const involvedVars = extractVariables(failedConstraint);

  for (const varName of involvedVars) {
    const currentBinding = state.substitution.get(varName);
    if (!currentBinding) continue;

    // Try alternative bindings from terms in scope
    for (const term of state.terms) {
      if (term.type === 'constant' && term !== currentBinding) {
        const newSubst = new Map(state.substitution);
        newSubst.set(varName, term);

        // Test if this resolves the constraint
        const testResult = tryConstraint(failedConstraint, {
          ...state,
          substitution: newSubst,
        });

        if (testResult.status === 'satisfied') {
          return { success: true, newSubstitution: newSubst };
        }
      }
    }
  }

  return { success: false, newSubstitution: state.substitution };
}

// =============================================================================
// Utilities
// =============================================================================

function resolveTermWithSubst(
  term: Term,
  substitution: Map<string, Term>
): Term {
  if (term.type === 'variable') {
    const resolved = substitution.get(term.name);
    return resolved || term;
  }

  if (term.type === 'compound') {
    return {
      ...term,
      args: term.args.map(arg => resolveTermWithSubst(arg, substitution)),
    };
  }

  return term;
}

function hasUnboundVariables(term: Term): boolean {
  if (term.type === 'variable') return true;
  if (term.type === 'compound') {
    return term.args.some(hasUnboundVariables);
  }
  return false;
}

function extractVariables(constraint: Constraint): string[] {
  const vars: string[] = [];

  function extractFromTerm(term: Term): void {
    if (term.type === 'variable') {
      vars.push(term.name);
    } else if (term.type === 'compound') {
      term.args.forEach(extractFromTerm);
    }
  }

  if ('left' in constraint) extractFromTerm(constraint.left);
  if ('right' in constraint) extractFromTerm(constraint.right);
  if ('element' in constraint) extractFromTerm(constraint.element);

  return [...new Set(vars)];
}

function constraintToString(constraint: Constraint): string {
  switch (constraint.type) {
    case 'equality':
      return `${termToString(constraint.left)} = ${termToString(constraint.right)}`;
    case 'inequality':
      return `${termToString(constraint.left)} ≠ ${termToString(constraint.right)}`;
    case 'membership':
      return `${termToString(constraint.element)} ∈ {...}`;
    case 'custom':
      return `custom(${constraint.name})`;
    default:
      return 'unknown';
  }
}

function termToString(term: Term): string {
  switch (term.type) {
    case 'variable':
      return `?${term.name}`;
    case 'constant':
      return String(term.name);
    case 'compound':
      return `${term.functor}(${term.args.map(termToString).join(', ')})`;
  }
}

function hashConstrained(structure: SymbolicStructure): string {
  const content = JSON.stringify({
    termCount: structure.terms.length,
    constraintCount: structure.constraints.length,
  });

  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}
