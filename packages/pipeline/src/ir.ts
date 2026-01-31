/**
 * Neurosymbolic Intermediate Representation (NeSy-IR)
 *
 * The common language between all pipeline stages
 */

import { z } from 'zod';
import type { NeuralFeatures, SymbolicStructure, TrustLevel } from '@nesy/core';

// =============================================================================
// IR Node Types
// =============================================================================

export const IRNodeKindSchema = z.enum([
  'neural',      // Raw neural features
  'grounded',    // Neural grounded to symbols
  'constrained', // Symbols with satisfied constraints
  'verified',    // Passed verification gates
  'compiled',    // Final compiled artifact
]);

export type IRNodeKind = z.infer<typeof IRNodeKindSchema>;

export interface IRNode {
  kind: IRNodeKind;
  id: string;
  inputHash: string;
  outputHash: string;
  timestamp: number;
  metadata: Record<string, unknown>;
}

// =============================================================================
// Stage-Specific IR
// =============================================================================

export interface PerceiveIR extends IRNode {
  kind: 'neural';
  features: NeuralFeatures;
  sourceText: string;
  modality: 'text' | 'image' | 'audio' | 'multimodal';
}

export interface GroundIR extends IRNode {
  kind: 'grounded';
  symbols: SymbolicStructure;
  groundingConfidence: number;
  ambiguities: string[];
}

export interface ConstrainIR extends IRNode {
  kind: 'constrained';
  satisfied: SymbolicStructure;
  unsatisfied: string[];  // Constraints that couldn't be satisfied
  searchSteps: number;
}

export interface VerifyIR extends IRNode {
  kind: 'verified';
  proof: VerificationProof;
  gatesPasssed: string[];
  gatesFailed: string[];
}

export interface CompiledIR extends IRNode {
  kind: 'compiled';
  artifact: CompiledArtifact;
  provenance: ProvenanceChain;
}

// =============================================================================
// Verification Proof
// =============================================================================

export interface VerificationProof {
  id: string;

  // Sextet gates
  provenance: GateResult;
  effects: GateResult;
  liveness: GateResult;
  recovery: GateResult;
  quality: GateResult;
  omega: GateResult;

  // Overall
  passed: boolean;
  derivation: string[];
}

export interface GateResult {
  gate: string;
  status: 'passed' | 'failed' | 'skipped';
  evidence: string[];
  timestamp: number;
}

// =============================================================================
// Compiled Artifact
// =============================================================================

export interface CompiledArtifact {
  // Pentad (5 Ws)
  pentad: {
    why: unknown;
    where: unknown;
    who: unknown;
    when: unknown;
    what: unknown;
  };

  // Execution plan
  plan: ExecutionStep[];

  // Trust level
  trust: TrustLevel;

  // Compiled at
  compiledAt: number;
}

export interface ExecutionStep {
  id: string;
  action: string;
  inputs: string[];
  outputs: string[];
  dependencies: string[];
}

// =============================================================================
// Provenance Chain
// =============================================================================

export interface ProvenanceChain {
  entries: ProvenanceEntry[];
}

export interface ProvenanceEntry {
  stage: string;
  inputHash: string;
  outputHash: string;
  operation: string;
  timestamp: number;
  taint: string[];
}

// =============================================================================
// IR Utilities
// =============================================================================

export function createIRNode(
  kind: IRNodeKind,
  inputHash: string,
  outputHash: string,
  metadata: Record<string, unknown> = {}
): IRNode {
  return {
    kind,
    id: `${kind}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    inputHash,
    outputHash,
    timestamp: Date.now(),
    metadata,
  };
}

export function hashIR(ir: IRNode): string {
  const content = JSON.stringify({
    kind: ir.kind,
    inputHash: ir.inputHash,
    outputHash: ir.outputHash,
  });

  // Simple hash for now (would use crypto in production)
  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}
