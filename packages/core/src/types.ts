/**
 * Core types for the NeSy Compiler
 */

import { z } from 'zod';

// =============================================================================
// Trust Levels (aligned with Holon rings)
// =============================================================================

export enum TrustLevel {
  KERNEL = 0,      // Ring 0: Highest trust
  PRIVILEGED = 1,  // Ring 1: Elevated trust
  STANDARD = 2,    // Ring 2: Normal trust
  UNTRUSTED = 3,   // Ring 3: Unverified
}

// =============================================================================
// Compilation Modes
// =============================================================================

export const GenesisRequestSchema = z.object({
  mode: z.literal('genesis'),
  specification: z.record(z.unknown()),
});

export const SeedRequestSchema = z.object({
  mode: z.literal('seed'),
  baseHolon: z.string(),  // Reference to existing holon
  mutations: z.array(z.record(z.unknown())),
});

export const AtomRequestSchema = z.object({
  mode: z.literal('atom'),
  intent: z.string(),
});

export const ConstraintRequestSchema = z.object({
  mode: z.literal('constraint'),
  constraints: z.array(z.unknown()),  // External constraints to apply
});

export const CompilationRequestSchema = z.discriminatedUnion('mode', [
  GenesisRequestSchema,
  SeedRequestSchema,
  AtomRequestSchema,
  ConstraintRequestSchema,
]);

export type CompilationRequest = z.infer<typeof CompilationRequestSchema>;

// =============================================================================
// Compilation Output
// =============================================================================

export const TransformationProofSchema = z.object({
  id: z.string(),
  stages: z.array(z.object({
    name: z.string(),
    input: z.string(),  // Hash
    output: z.string(), // Hash
    timestamp: z.number(),
  })),
  sextetReceipt: z.object({
    provenance: z.enum(['passed', 'failed', 'skipped']),
    effects: z.enum(['passed', 'failed', 'skipped']),
    liveness: z.enum(['passed', 'failed', 'skipped']),
    recovery: z.enum(['passed', 'failed', 'skipped']),
    quality: z.enum(['passed', 'failed', 'skipped']),
    omega: z.enum(['passed', 'failed', 'skipped']),
  }),
  deterministic: z.boolean(),
});

export type TransformationProof = z.infer<typeof TransformationProofSchema>;

export const CompiledHolonSchema = z.object({
  holon: z.record(z.unknown()),      // ActualizedHolon
  ir: z.record(z.unknown()),         // ExecutionPlan
  proof: TransformationProofSchema,
  provenance: z.object({
    trust: z.nativeEnum(TrustLevel),
    taint: z.array(z.string()),
    lineage: z.array(z.string()),
  }),
});

export type CompiledHolon = z.infer<typeof CompiledHolonSchema>;

// =============================================================================
// Neural Types
// =============================================================================

export interface Embedding {
  vector: Float32Array;
  dimension: number;
  model: string;
  timestamp: number;
}

export interface AttentionWeights {
  heads: Float32Array[];
  layers: number;
  normalized: boolean;
}

export interface NeuralFeatures {
  embedding: Embedding;
  attention: AttentionWeights;
  confidence: number;
}

// =============================================================================
// Symbolic Types
// =============================================================================

export type Term =
  | { type: 'variable'; name: string }
  | { type: 'constant'; name: string; value?: unknown }
  | { type: 'compound'; functor: string; args: Term[] };

export interface Substitution {
  [key: string]: Term;
}

// Constraint types for the constraint solver
export type Constraint =
  | { type: 'equality'; left: Term; right: Term }
  | { type: 'inequality'; left: Term; right: Term }
  | { type: 'membership'; element: Term; set: Term[] }
  | { type: 'custom'; name: string; args: Term[] };

export interface SymbolicStructure {
  terms: Term[];
  constraints: Constraint[];
  metadata?: Record<string, unknown>;
}

// =============================================================================
// Bridge Types
// =============================================================================

export interface GroundingResult {
  source: NeuralFeatures;
  target: SymbolicStructure;
  confidence: number;
  ambiguities: string[];
}

export interface LiftingResult {
  source: SymbolicStructure;
  target: Embedding;
  lossEstimate: number;
}

export interface DiscretizationConfig {
  temperature: number;  // Boltzmann temperature
  annealing: 'linear' | 'exponential' | 'cosine';
  threshold: number;
}
