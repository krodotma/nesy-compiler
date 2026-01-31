/**
 * EMIT Stage
 *
 * Input: Verified IR with proof
 * Output: CompiledArtifact (Pentad + Execution Plan)
 *
 * This stage handles:
 * - Pentad construction (5 Ws)
 * - Execution plan generation
 * - Trust level assignment
 * - Provenance chain finalization
 */

import type { CompilationContext, TrustLevel } from '@nesy/core';
import type {
  VerifyIR,
  CompiledIR,
  CompiledArtifact,
  ExecutionStep,
  ProvenanceChain,
  ProvenanceEntry,
} from '../ir';
import { createIRNode } from '../ir';

export interface EmitInput {
  verifyIR: VerifyIR;
  stageHistory: StageHistoryEntry[];
  intentMetadata?: IntentMetadata;
}

export interface StageHistoryEntry {
  stage: string;
  inputHash: string;
  outputHash: string;
  timestamp: number;
}

export interface IntentMetadata {
  why?: string;
  where?: string;
  who?: string;
  when?: string;
  what?: string;
}

export interface EmitOutput {
  ir: CompiledIR;
  artifact: CompiledArtifact;
}

/**
 * Main emit function
 *
 * Produces final compiled artifact with full provenance
 */
export async function emit(
  input: EmitInput,
  context: CompilationContext
): Promise<EmitOutput> {
  const { verifyIR, stageHistory, intentMetadata = {} } = input;

  // Build the Pentad (5 Ws)
  const pentad = buildPentad(verifyIR, intentMetadata, context);

  // Generate execution plan
  const plan = await generateExecutionPlan(verifyIR, context);

  // Determine final trust level
  const trust = determineTrustLevel(verifyIR, context);

  // Build artifact
  const artifact: CompiledArtifact = {
    pentad,
    plan,
    trust,
    compiledAt: Date.now(),
  };

  // Build provenance chain
  const provenance = buildProvenanceChain(stageHistory, context);

  const ir: CompiledIR = {
    ...createIRNode('compiled', verifyIR.outputHash, hashArtifact(artifact)),
    kind: 'compiled',
    artifact,
    provenance,
  };

  return { ir, artifact };
}

// =============================================================================
// Pentad Construction (5 Ws)
// =============================================================================

function buildPentad(
  verifyIR: VerifyIR,
  metadata: IntentMetadata,
  context: CompilationContext
): CompiledArtifact['pentad'] {
  return {
    // WHY: Purpose and motivation
    why: buildWhy(verifyIR, metadata.why),

    // WHERE: Spatial/contextual scope
    where: buildWhere(verifyIR, metadata.where),

    // WHO: Actors and principals
    who: buildWho(verifyIR, metadata.who, context),

    // WHEN: Temporal scope and constraints
    when: buildWhen(verifyIR, metadata.when),

    // WHAT: Actions and outputs
    what: buildWhat(verifyIR, metadata.what),
  };
}

function buildWhy(verifyIR: VerifyIR, provided?: string): unknown {
  const derivation = verifyIR.proof.derivation;

  return {
    stated: provided || 'Not specified',
    inferred: inferPurpose(verifyIR),
    derivation: derivation.slice(0, 5),
    verificationEvidence: {
      provenanceReason: verifyIR.proof.provenance.evidence[0] || 'N/A',
      qualityReason: verifyIR.proof.quality.evidence[0] || 'N/A',
    },
  };
}

function inferPurpose(verifyIR: VerifyIR): string {
  // Infer purpose from gate results
  if (verifyIR.proof.effects.evidence.some(e => e.includes('io'))) {
    return 'Data processing with I/O operations';
  }
  if (verifyIR.proof.effects.evidence.some(e => e.includes('network'))) {
    return 'Network communication';
  }
  if (verifyIR.proof.liveness.evidence.some(e => e.includes('loop'))) {
    return 'Iterative computation';
  }
  return 'General symbolic computation';
}

function buildWhere(verifyIR: VerifyIR, provided?: string): unknown {
  return {
    stated: provided || 'Local execution context',
    scope: {
      type: 'sandboxed',
      boundaries: extractBoundaries(verifyIR),
      restrictions: extractRestrictions(verifyIR),
    },
  };
}

function extractBoundaries(verifyIR: VerifyIR): string[] {
  const boundaries: string[] = [];

  if (verifyIR.proof.effects.status === 'passed') {
    boundaries.push('effects-contained');
  }
  if (verifyIR.proof.liveness.status === 'passed') {
    boundaries.push('termination-guaranteed');
  }

  return boundaries;
}

function extractRestrictions(verifyIR: VerifyIR): string[] {
  return verifyIR.gatesFailed.map(gate => `restricted:${gate}`);
}

function buildWho(
  verifyIR: VerifyIR,
  provided?: string,
  context: CompilationContext = {} as CompilationContext
): unknown {
  return {
    stated: provided || 'System',
    principal: {
      trustLevel: context.trustLevel,
      trustLabel: trustLevelLabel(context.trustLevel),
      taintVector: context.taintVector || [],
    },
    authorization: {
      verified: verifyIR.proof.passed,
      gates: verifyIR.gatesPasssed,
    },
  };
}

function trustLevelLabel(level: TrustLevel): string {
  const labels: Record<TrustLevel, string> = {
    0: 'KERNEL',
    1: 'PRIVILEGED',
    2: 'STANDARD',
    3: 'UNTRUSTED',
  };
  return labels[level] || 'UNKNOWN';
}

function buildWhen(verifyIR: VerifyIR, provided?: string): unknown {
  return {
    stated: provided || 'Immediate',
    temporal: {
      compiledAt: Date.now(),
      verifiedAt: verifyIR.timestamp,
      validUntil: null, // No expiry by default
    },
    ordering: {
      sequential: true,
      causal: true,
    },
  };
}

function buildWhat(verifyIR: VerifyIR, provided?: string): unknown {
  return {
    stated: provided || 'Execute compiled plan',
    actions: verifyIR.proof.derivation.filter(d => d.startsWith('✓')),
    outputs: {
      type: 'symbolic-result',
      verified: verifyIR.proof.passed,
    },
  };
}

// =============================================================================
// Execution Plan Generation
// =============================================================================

async function generateExecutionPlan(
  verifyIR: VerifyIR,
  context: CompilationContext
): Promise<ExecutionStep[]> {
  const steps: ExecutionStep[] = [];
  let stepCounter = 0;

  // Initialize step
  steps.push({
    id: `step_${stepCounter++}`,
    action: 'initialize',
    inputs: ['context'],
    outputs: ['state'],
    dependencies: [],
  });

  // Verification steps (one per passed gate)
  for (const gate of verifyIR.gatesPasssed) {
    steps.push({
      id: `step_${stepCounter++}`,
      action: `verify:${gate}`,
      inputs: ['state'],
      outputs: [`verified:${gate}`],
      dependencies: [steps[steps.length - 1].id],
    });
  }

  // Execution step
  steps.push({
    id: `step_${stepCounter++}`,
    action: 'execute',
    inputs: verifyIR.gatesPasssed.map(g => `verified:${g}`),
    outputs: ['result'],
    dependencies: steps.slice(1).map(s => s.id),
  });

  // Finalization step
  steps.push({
    id: `step_${stepCounter++}`,
    action: 'finalize',
    inputs: ['result'],
    outputs: ['output'],
    dependencies: [steps[steps.length - 1].id],
  });

  return steps;
}

// =============================================================================
// Trust Level Determination
// =============================================================================

function determineTrustLevel(
  verifyIR: VerifyIR,
  context: CompilationContext
): TrustLevel {
  // Start with context trust level
  let level = context.trustLevel;

  // Can only lower (increase trust) if all gates passed
  if (!verifyIR.proof.passed) {
    // Can't improve trust if verification failed
    return Math.max(level, 2) as TrustLevel; // At least STANDARD
  }

  // Omega gate passing might allow trust elevation
  if (verifyIR.proof.omega.status === 'passed' &&
      verifyIR.proof.provenance.status === 'passed') {
    // Self-referential consistency + provenance = can be trusted more
    if (level > 1) {
      level = (level - 1) as TrustLevel;
    }
  }

  // Effects gate failure forces lower trust
  if (verifyIR.proof.effects.status === 'failed') {
    level = 3; // UNTRUSTED
  }

  return level;
}

// =============================================================================
// Provenance Chain
// =============================================================================

function buildProvenanceChain(
  stageHistory: StageHistoryEntry[],
  context: CompilationContext
): ProvenanceChain {
  const entries: ProvenanceEntry[] = stageHistory.map(entry => ({
    stage: entry.stage,
    inputHash: entry.inputHash,
    outputHash: entry.outputHash,
    operation: stageToOperation(entry.stage),
    timestamp: entry.timestamp,
    taint: context.taintVector.filter(t => t.includes(entry.stage)),
  }));

  return { entries };
}

function stageToOperation(stage: string): string {
  const operations: Record<string, string> = {
    perceive: 'embed',
    ground: 'symbolize',
    constrain: 'satisfy',
    verify: 'prove',
    emit: 'compile',
  };
  return operations[stage.toLowerCase()] || 'transform';
}

// =============================================================================
// Utilities
// =============================================================================

function hashArtifact(artifact: CompiledArtifact): string {
  const content = JSON.stringify({
    trust: artifact.trust,
    stepCount: artifact.plan.length,
    compiledAt: artifact.compiledAt,
  });

  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}
