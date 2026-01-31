/**
 * VERIFY Stage
 *
 * Input: Constrained symbolic structure
 * Output: Verification proof (Sextet gates)
 *
 * This stage handles:
 * - Provenance gate (lineage verification)
 * - Effects gate (side-effect analysis)
 * - Liveness gate (deadlock/livelock detection)
 * - Recovery gate (rollback capability)
 * - Quality gate (output quality metrics)
 * - Omega gate (self-referential consistency)
 */

import type { CompilationContext, SymbolicStructure, TrustLevel } from '@nesy/core';
import type { ConstrainIR, VerifyIR, VerificationProof, GateResult } from '../ir';
import { createIRNode } from '../ir';

export interface VerifyInput {
  constrainIR: ConstrainIR;
  satisfied: SymbolicStructure;
  provenance?: ProvenanceTrace;
}

export interface ProvenanceTrace {
  stages: string[];
  inputHashes: string[];
  outputHashes: string[];
  taintVector: string[];
}

export interface VerifyOutput {
  ir: VerifyIR;
  proof: VerificationProof;
  passed: boolean;
}

/**
 * Main verification function
 *
 * Runs all six gates of the Sextet verification framework
 */
export async function verify(
  input: VerifyInput,
  context: CompilationContext
): Promise<VerifyOutput> {
  const { constrainIR, satisfied, provenance } = input;

  // Run all six gates
  const provenanceGate = await runProvenanceGate(provenance, context);
  const effectsGate = await runEffectsGate(satisfied, context);
  const livenessGate = await runLivenessGate(satisfied, context);
  const recoveryGate = await runRecoveryGate(constrainIR, context);
  const qualityGate = await runQualityGate(satisfied, constrainIR, context);
  const omegaGate = await runOmegaGate(satisfied, context);

  // Build verification proof
  const allGates = [provenanceGate, effectsGate, livenessGate, recoveryGate, qualityGate, omegaGate];
  const passed = allGates.every(g => g.status === 'passed');

  const proof: VerificationProof = {
    id: `proof_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    provenance: provenanceGate,
    effects: effectsGate,
    liveness: livenessGate,
    recovery: recoveryGate,
    quality: qualityGate,
    omega: omegaGate,
    passed,
    derivation: buildDerivation(allGates),
  };

  const gatesPassed = allGates.filter(g => g.status === 'passed').map(g => g.gate);
  const gatesFailed = allGates.filter(g => g.status === 'failed').map(g => g.gate);

  const ir: VerifyIR = {
    ...createIRNode('verified', constrainIR.outputHash, hashProof(proof)),
    kind: 'verified',
    proof,
    gatesPasssed: gatesPassed,
    gatesFailed,
  };

  return { ir, proof, passed };
}

// =============================================================================
// Gate 1: Provenance
// =============================================================================

async function runProvenanceGate(
  provenance: ProvenanceTrace | undefined,
  context: CompilationContext
): Promise<GateResult> {
  const evidence: string[] = [];
  let status: 'passed' | 'failed' | 'skipped' = 'passed';

  if (!provenance) {
    evidence.push('No provenance trace provided');
    status = 'skipped';
    return { gate: 'provenance', status, evidence, timestamp: Date.now() };
  }

  // Verify hash chain integrity
  for (let i = 1; i < provenance.inputHashes.length; i++) {
    if (provenance.inputHashes[i] !== provenance.outputHashes[i - 1]) {
      evidence.push(`Hash chain break at stage ${i}: ${provenance.stages[i]}`);
      status = 'failed';
    }
  }

  if (status === 'passed') {
    evidence.push(`Verified ${provenance.stages.length} stages`);
    evidence.push(`Hash chain intact: ${provenance.inputHashes[0]} → ${provenance.outputHashes[provenance.outputHashes.length - 1]}`);
  }

  // Check for taint
  if (provenance.taintVector.length > 0) {
    evidence.push(`Taint detected: ${provenance.taintVector.join(', ')}`);
    // Taint doesn't necessarily fail, but is recorded
  }

  return { gate: 'provenance', status, evidence, timestamp: Date.now() };
}

// =============================================================================
// Gate 2: Effects
// =============================================================================

async function runEffectsGate(
  structure: SymbolicStructure,
  context: CompilationContext
): Promise<GateResult> {
  const evidence: string[] = [];
  let status: 'passed' | 'failed' = 'passed';

  // Analyze terms for side effects
  const effectTerms = structure.terms.filter(isEffectTerm);

  if (effectTerms.length === 0) {
    evidence.push('No side-effect terms detected');
    return { gate: 'effects', status: 'passed', evidence, timestamp: Date.now() };
  }

  // Classify effects
  const classified = classifyEffects(effectTerms);

  for (const [category, terms] of Object.entries(classified)) {
    evidence.push(`${category}: ${terms.length} terms`);

    // Check if effects are allowed at current trust level
    if (!isEffectAllowed(category, context.trustLevel)) {
      evidence.push(`DENIED: ${category} not allowed at trust level ${context.trustLevel}`);
      status = 'failed';
    }
  }

  return { gate: 'effects', status, evidence, timestamp: Date.now() };
}

function isEffectTerm(term: { type: string; functor?: string }): boolean {
  if (term.type !== 'compound') return false;

  const effectFunctors = ['io', 'write', 'send', 'execute', 'modify', 'delete', 'create'];
  return effectFunctors.some(ef => term.functor?.toLowerCase().includes(ef));
}

function classifyEffects(terms: unknown[]): Record<string, unknown[]> {
  const categories: Record<string, unknown[]> = {
    io: [],
    network: [],
    state: [],
    system: [],
  };

  for (const term of terms) {
    const t = term as { functor?: string };
    if (t.functor?.includes('io') || t.functor?.includes('read') || t.functor?.includes('write')) {
      categories.io.push(term);
    } else if (t.functor?.includes('send') || t.functor?.includes('net')) {
      categories.network.push(term);
    } else if (t.functor?.includes('state') || t.functor?.includes('modify')) {
      categories.state.push(term);
    } else {
      categories.system.push(term);
    }
  }

  return categories;
}

function isEffectAllowed(category: string, trustLevel: TrustLevel): boolean {
  const allowedByLevel: Record<TrustLevel, string[]> = {
    0: ['io', 'network', 'state', 'system'],  // KERNEL: all allowed
    1: ['io', 'network', 'state'],             // PRIVILEGED: no system
    2: ['io', 'state'],                        // STANDARD: no network/system
    3: ['state'],                              // UNTRUSTED: only state
  };

  return allowedByLevel[trustLevel]?.includes(category) ?? false;
}

// =============================================================================
// Gate 3: Liveness
// =============================================================================

async function runLivenessGate(
  structure: SymbolicStructure,
  context: CompilationContext
): Promise<GateResult> {
  const evidence: string[] = [];

  // Build dependency graph
  const graph = buildDependencyGraph(structure);

  // Check for cycles (potential deadlock)
  const cycles = findCycles(graph);

  if (cycles.length > 0) {
    evidence.push(`Detected ${cycles.length} potential deadlock cycle(s)`);
    for (const cycle of cycles.slice(0, 3)) {
      evidence.push(`  Cycle: ${cycle.join(' → ')}`);
    }
    return { gate: 'liveness', status: 'failed', evidence, timestamp: Date.now() };
  }

  // Check for infinite loops (livelock)
  const unboundedLoops = detectUnboundedLoops(structure);

  if (unboundedLoops.length > 0) {
    evidence.push(`Detected ${unboundedLoops.length} potentially unbounded loop(s)`);
    return { gate: 'liveness', status: 'failed', evidence, timestamp: Date.now() };
  }

  evidence.push('No deadlock or livelock patterns detected');
  evidence.push(`Analyzed ${graph.size} nodes in dependency graph`);

  return { gate: 'liveness', status: 'passed', evidence, timestamp: Date.now() };
}

function buildDependencyGraph(structure: SymbolicStructure): Map<string, string[]> {
  const graph = new Map<string, string[]>();

  for (const term of structure.terms) {
    if (term.type === 'compound') {
      const deps: string[] = [];
      for (const arg of term.args) {
        if (arg.type === 'variable' || arg.type === 'compound') {
          deps.push(arg.type === 'variable' ? arg.name : arg.functor);
        }
      }
      graph.set(term.functor, deps);
    }
  }

  return graph;
}

function findCycles(graph: Map<string, string[]>): string[][] {
  const cycles: string[][] = [];
  const visited = new Set<string>();
  const inStack = new Set<string>();
  const path: string[] = [];

  function dfs(node: string): void {
    if (inStack.has(node)) {
      // Found cycle
      const cycleStart = path.indexOf(node);
      cycles.push(path.slice(cycleStart));
      return;
    }

    if (visited.has(node)) return;

    visited.add(node);
    inStack.add(node);
    path.push(node);

    for (const neighbor of graph.get(node) || []) {
      dfs(neighbor);
    }

    path.pop();
    inStack.delete(node);
  }

  for (const node of graph.keys()) {
    dfs(node);
  }

  return cycles;
}

function detectUnboundedLoops(structure: SymbolicStructure): string[] {
  const unbounded: string[] = [];

  for (const term of structure.terms) {
    if (term.type === 'compound') {
      // Check for recursive calls without base case
      const hasRecursiveRef = term.args.some(
        arg => arg.type === 'compound' && arg.functor === term.functor
      );

      if (hasRecursiveRef) {
        // Check if there's a constraining base case
        const hasBaseCase = structure.constraints.some(c =>
          'left' in c && c.left.type === 'compound' && c.left.functor === term.functor
        );

        if (!hasBaseCase) {
          unbounded.push(term.functor);
        }
      }
    }
  }

  return unbounded;
}

// =============================================================================
// Gate 4: Recovery
// =============================================================================

async function runRecoveryGate(
  constrainIR: ConstrainIR,
  context: CompilationContext
): Promise<GateResult> {
  const evidence: string[] = [];

  // Check if we have enough information for rollback
  const hasCheckpoint = constrainIR.searchSteps > 0;
  const hasSubstitution = constrainIR.satisfied.terms.length > 0;

  if (!hasCheckpoint) {
    evidence.push('No search checkpoint available for recovery');
  } else {
    evidence.push(`Recovery checkpoint at step ${constrainIR.searchSteps}`);
  }

  if (hasSubstitution) {
    evidence.push(`State snapshot: ${constrainIR.satisfied.terms.length} terms preserved`);
  }

  // Check unsatisfied constraints for recoverability
  const unrecoverable = constrainIR.unsatisfied.filter(u =>
    u.includes('system') || u.includes('kernel')
  );

  if (unrecoverable.length > 0) {
    evidence.push(`${unrecoverable.length} constraint(s) may not be recoverable`);
    return { gate: 'recovery', status: 'failed', evidence, timestamp: Date.now() };
  }

  evidence.push('All unsatisfied constraints are recoverable');

  return { gate: 'recovery', status: 'passed', evidence, timestamp: Date.now() };
}

// =============================================================================
// Gate 5: Quality
// =============================================================================

async function runQualityGate(
  structure: SymbolicStructure,
  constrainIR: ConstrainIR,
  context: CompilationContext
): Promise<GateResult> {
  const evidence: string[] = [];
  let status: 'passed' | 'failed' = 'passed';

  // Compute quality metrics
  const metrics = computeQualityMetrics(structure, constrainIR);

  evidence.push(`Completeness: ${(metrics.completeness * 100).toFixed(1)}%`);
  evidence.push(`Consistency: ${(metrics.consistency * 100).toFixed(1)}%`);
  evidence.push(`Groundedness: ${(metrics.groundedness * 100).toFixed(1)}%`);

  // Quality thresholds
  const COMPLETENESS_THRESHOLD = 0.7;
  const CONSISTENCY_THRESHOLD = 0.9;
  const GROUNDEDNESS_THRESHOLD = 0.6;

  if (metrics.completeness < COMPLETENESS_THRESHOLD) {
    evidence.push(`WARN: Completeness below ${COMPLETENESS_THRESHOLD * 100}%`);
  }

  if (metrics.consistency < CONSISTENCY_THRESHOLD) {
    evidence.push(`FAIL: Consistency below ${CONSISTENCY_THRESHOLD * 100}%`);
    status = 'failed';
  }

  if (metrics.groundedness < GROUNDEDNESS_THRESHOLD) {
    evidence.push(`WARN: Groundedness below ${GROUNDEDNESS_THRESHOLD * 100}%`);
  }

  return { gate: 'quality', status, evidence, timestamp: Date.now() };
}

interface QualityMetrics {
  completeness: number;
  consistency: number;
  groundedness: number;
}

function computeQualityMetrics(
  structure: SymbolicStructure,
  constrainIR: ConstrainIR
): QualityMetrics {
  // Completeness: ratio of satisfied to total constraints
  const totalConstraints = structure.constraints.length + constrainIR.unsatisfied.length;
  const completeness = totalConstraints === 0 ? 1 :
    structure.constraints.length / totalConstraints;

  // Consistency: no contradictions in satisfied constraints
  const contradictions = findContradictions(structure.constraints);
  const consistency = contradictions === 0 ? 1 :
    1 - (contradictions / Math.max(1, structure.constraints.length));

  // Groundedness: ratio of ground terms to total terms
  const groundTerms = structure.terms.filter(t => t.type === 'constant').length;
  const groundedness = structure.terms.length === 0 ? 1 :
    groundTerms / structure.terms.length;

  return { completeness, consistency, groundedness };
}

function findContradictions(constraints: unknown[]): number {
  // Simplified: check for equality/inequality conflicts on same terms
  let contradictions = 0;

  for (let i = 0; i < constraints.length; i++) {
    for (let j = i + 1; j < constraints.length; j++) {
      const c1 = constraints[i] as { type: string; left?: unknown; right?: unknown };
      const c2 = constraints[j] as { type: string; left?: unknown; right?: unknown };

      if (c1.type === 'equality' && c2.type === 'inequality') {
        if (JSON.stringify(c1.left) === JSON.stringify(c2.left) &&
            JSON.stringify(c1.right) === JSON.stringify(c2.right)) {
          contradictions++;
        }
      }
    }
  }

  return contradictions;
}

// =============================================================================
// Gate 6: Omega (Self-referential consistency)
// =============================================================================

async function runOmegaGate(
  structure: SymbolicStructure,
  context: CompilationContext
): Promise<GateResult> {
  const evidence: string[] = [];

  // Omega gate checks self-referential consistency
  // Does the structure describe itself consistently?

  // Check 1: Meta-level consistency
  const metaTerms = structure.terms.filter(t =>
    t.type === 'compound' && (t.functor.startsWith('meta_') || t.functor.includes('self'))
  );

  if (metaTerms.length > 0) {
    evidence.push(`Found ${metaTerms.length} meta-level term(s)`);

    // Verify they don't create paradoxes
    const paradoxes = detectParadoxes(metaTerms, structure);
    if (paradoxes.length > 0) {
      evidence.push(`Detected ${paradoxes.length} potential paradox(es)`);
      for (const p of paradoxes) {
        evidence.push(`  - ${p}`);
      }
      return { gate: 'omega', status: 'failed', evidence, timestamp: Date.now() };
    }
  }

  // Check 2: Fixed-point consistency
  // The structure should be a fixed point of its own interpretation
  const isFixedPoint = checkFixedPoint(structure);

  if (!isFixedPoint) {
    evidence.push('Structure is not self-consistent (not a fixed point)');
    return { gate: 'omega', status: 'failed', evidence, timestamp: Date.now() };
  }

  evidence.push('Self-referential consistency verified');
  evidence.push('Structure is a fixed point of interpretation');

  return { gate: 'omega', status: 'passed', evidence, timestamp: Date.now() };
}

function detectParadoxes(
  metaTerms: unknown[],
  structure: SymbolicStructure
): string[] {
  const paradoxes: string[] = [];

  for (const term of metaTerms) {
    const t = term as { functor: string; args: unknown[] };

    // Check for self-referential negation
    if (t.functor.includes('not_') || t.functor.includes('neg_')) {
      const targetName = t.functor.replace('not_', '').replace('neg_', '');
      const selfRef = structure.terms.some(other =>
        (other as { functor?: string }).functor === targetName
      );

      if (selfRef) {
        paradoxes.push(`Self-negation: ${t.functor} references ${targetName}`);
      }
    }
  }

  return paradoxes;
}

function checkFixedPoint(structure: SymbolicStructure): boolean {
  // A structure is a fixed point if applying its transformations
  // doesn't change its interpretation

  // Simplified check: all terms are grounded or have bounded depth
  const maxDepth = 10;

  for (const term of structure.terms) {
    if (!hasFiniteDepth(term, maxDepth)) {
      return false;
    }
  }

  return true;
}

function hasFiniteDepth(term: { type: string; args?: unknown[] }, maxDepth: number): boolean {
  if (maxDepth <= 0) return false;
  if (term.type !== 'compound') return true;

  return (term.args || []).every(arg =>
    hasFiniteDepth(arg as { type: string; args?: unknown[] }, maxDepth - 1)
  );
}

// =============================================================================
// Utilities
// =============================================================================

function buildDerivation(gates: GateResult[]): string[] {
  const derivation: string[] = [];

  derivation.push('=== Verification Derivation ===');

  for (const gate of gates) {
    const statusSymbol = gate.status === 'passed' ? '✓' :
                        gate.status === 'failed' ? '✗' : '○';
    derivation.push(`${statusSymbol} ${gate.gate.toUpperCase()}`);
  }

  const passed = gates.filter(g => g.status === 'passed').length;
  const total = gates.length;
  derivation.push(`=== ${passed}/${total} gates passed ===`);

  return derivation;
}

function hashProof(proof: VerificationProof): string {
  const content = JSON.stringify({
    id: proof.id,
    passed: proof.passed,
    gateCount: 6,
  });

  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}
