/**
 * GROUND Stage
 *
 * Input: NeuralFeatures (from PERCEIVE)
 * Output: SymbolicStructure (grounded symbols)
 *
 * This stage handles:
 * - Symbol proposal from neural embeddings
 * - Grounding confidence estimation
 * - Ambiguity detection
 * - Hierarchical term construction
 */

import type { CompilationContext, NeuralFeatures, SymbolicStructure, Term } from '@nesy/core';
import type { PerceiveIR, GroundIR } from '../ir';
import { createIRNode } from '../ir';

export interface GroundInput {
  perceiveIR: PerceiveIR;
  features: NeuralFeatures;
}

export interface GroundOutput {
  ir: GroundIR;
  symbols: SymbolicStructure;
}

/**
 * Main grounding function
 *
 * Transforms continuous neural representations into discrete symbolic terms
 */
export async function ground(
  input: GroundInput,
  context: CompilationContext
): Promise<GroundOutput> {
  const { features } = input;

  // Extract candidate symbols from embedding
  const candidates = await extractSymbolCandidates(features, context);

  // Score and filter candidates
  const scored = scoreGroundingCandidates(candidates, features);

  // Build symbolic structure
  const { terms, ambiguities } = buildSymbolicStructure(scored, context);

  // Compute overall grounding confidence
  const groundingConfidence = computeGroundingConfidence(scored, ambiguities);

  const symbols: SymbolicStructure = {
    terms,
    constraints: [],
    metadata: {
      sourceModality: input.perceiveIR.modality,
      candidateCount: candidates.length,
    },
  };

  const ir: GroundIR = {
    ...createIRNode('grounded', input.perceiveIR.outputHash, hashSymbols(symbols)),
    kind: 'grounded',
    symbols,
    groundingConfidence,
    ambiguities,
  };

  return { ir, symbols };
}

// =============================================================================
// Symbol Candidate Extraction
// =============================================================================

interface SymbolCandidate {
  name: string;
  type: 'constant' | 'predicate' | 'function';
  arity: number;
  confidence: number;
  embeddingSlice: number[];
}

async function extractSymbolCandidates(
  features: NeuralFeatures,
  context: CompilationContext
): Promise<SymbolCandidate[]> {
  const { embedding, attention } = features;
  const candidates: SymbolCandidate[] = [];

  // Use attention heads to identify salient regions
  const salientIndices = findSalientRegions(attention);

  // For each salient region, propose a symbol
  for (let i = 0; i < salientIndices.length; i++) {
    const idx = salientIndices[i];
    const sliceStart = Math.floor((idx / attention.heads[0].length) * embedding.dimension);
    const sliceEnd = Math.min(sliceStart + 128, embedding.dimension);

    const slice: number[] = [];
    for (let j = sliceStart; j < sliceEnd; j++) {
      slice.push(embedding.vector[j]);
    }

    // Propose symbol based on embedding characteristics
    const candidate = proposeSymbol(slice, i, features.confidence);
    candidates.push(candidate);
  }

  return candidates;
}

function findSalientRegions(attention: { heads: Float32Array[]; layers: number }): number[] {
  const aggregated = new Float32Array(attention.heads[0].length);

  // Aggregate attention across heads
  for (const head of attention.heads) {
    for (let i = 0; i < head.length; i++) {
      aggregated[i] += head[i];
    }
  }

  // Normalize
  const total = aggregated.reduce((a, b) => a + b, 0);
  for (let i = 0; i < aggregated.length; i++) {
    aggregated[i] /= total;
  }

  // Find peaks (above mean + std)
  const mean = aggregated.reduce((a, b) => a + b, 0) / aggregated.length;
  const variance = aggregated.reduce((sum, x) => sum + (x - mean) ** 2, 0) / aggregated.length;
  const std = Math.sqrt(variance);
  const threshold = mean + std;

  const peaks: number[] = [];
  for (let i = 0; i < aggregated.length; i++) {
    if (aggregated[i] > threshold) {
      peaks.push(i);
    }
  }

  // Limit to top 10
  return peaks.slice(0, 10);
}

function proposeSymbol(
  embeddingSlice: number[],
  index: number,
  baseConfidence: number
): SymbolCandidate {
  // Analyze embedding slice to determine symbol type
  const mean = embeddingSlice.reduce((a, b) => a + b, 0) / embeddingSlice.length;
  const energy = embeddingSlice.reduce((a, b) => a + b * b, 0);

  // Heuristic: high mean suggests predicate, low suggests constant
  const type: 'constant' | 'predicate' | 'function' =
    mean > 0.05 ? 'predicate' :
    mean < -0.05 ? 'function' :
    'constant';

  // Arity from energy distribution
  const arity = type === 'constant' ? 0 : Math.max(1, Math.min(4, Math.floor(energy * 10)));

  return {
    name: `sym_${index}_${type.charAt(0)}`,
    type,
    arity,
    confidence: baseConfidence * (0.5 + Math.random() * 0.5), // Variance in grounding
    embeddingSlice,
  };
}

// =============================================================================
// Candidate Scoring
// =============================================================================

interface ScoredCandidate extends SymbolCandidate {
  groundingScore: number;
  ambiguityScore: number;
}

function scoreGroundingCandidates(
  candidates: SymbolCandidate[],
  features: NeuralFeatures
): ScoredCandidate[] {
  return candidates.map(candidate => {
    // Grounding score: how well does this symbol fit the neural representation?
    const groundingScore = computeGroundingScore(candidate, features);

    // Ambiguity score: could this be confused with other symbols?
    const ambiguityScore = computeAmbiguityScore(candidate, candidates);

    return {
      ...candidate,
      groundingScore,
      ambiguityScore,
    };
  });
}

function computeGroundingScore(
  candidate: SymbolCandidate,
  features: NeuralFeatures
): number {
  // Base score from candidate confidence
  let score = candidate.confidence;

  // Boost for higher-order symbols in high-confidence contexts
  if (features.confidence > 0.8 && candidate.type !== 'constant') {
    score *= 1.2;
  }

  // Penalize if arity seems inconsistent with attention pattern
  const expectedComplexity = features.attention.heads.length > 6 ? 'high' : 'low';
  if (expectedComplexity === 'high' && candidate.arity < 2) {
    score *= 0.8;
  }

  return Math.min(1, Math.max(0, score));
}

function computeAmbiguityScore(
  candidate: SymbolCandidate,
  allCandidates: SymbolCandidate[]
): number {
  // How many other candidates have similar embeddings?
  let similarCount = 0;

  for (const other of allCandidates) {
    if (other === candidate) continue;

    const similarity = cosineSimilarity(
      candidate.embeddingSlice,
      other.embeddingSlice
    );

    if (similarity > 0.7) {
      similarCount++;
    }
  }

  // More similar candidates = higher ambiguity
  return similarCount / Math.max(1, allCandidates.length - 1);
}

function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length || a.length === 0) return 0;

  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// =============================================================================
// Symbolic Structure Building
// =============================================================================

function buildSymbolicStructure(
  candidates: ScoredCandidate[],
  context: CompilationContext
): { terms: Term[]; ambiguities: string[] } {
  const terms: Term[] = [];
  const ambiguities: string[] = [];
  const threshold = context.discretization.threshold;

  for (const candidate of candidates) {
    // Skip low-confidence candidates
    if (candidate.groundingScore < threshold) {
      continue;
    }

    // Track ambiguities
    if (candidate.ambiguityScore > 0.3) {
      ambiguities.push(`${candidate.name}:ambiguous(${candidate.ambiguityScore.toFixed(2)})`);
    }

    // Build term
    const term = buildTerm(candidate);
    terms.push(term);
  }

  return { terms, ambiguities };
}

function buildTerm(candidate: ScoredCandidate): Term {
  switch (candidate.type) {
    case 'constant':
      return {
        type: 'constant',
        name: candidate.name,
        value: candidate.groundingScore,
      };

    case 'predicate':
    case 'function':
      // Create compound term with placeholder arguments
      const args: Term[] = [];
      for (let i = 0; i < candidate.arity; i++) {
        args.push({
          type: 'variable',
          name: `_arg${i}`,
        });
      }

      return {
        type: 'compound',
        functor: candidate.name,
        args,
      };
  }
}

// =============================================================================
// Confidence Computation
// =============================================================================

function computeGroundingConfidence(
  candidates: ScoredCandidate[],
  ambiguities: string[]
): number {
  if (candidates.length === 0) return 0;

  // Average grounding score
  const avgScore = candidates.reduce((sum, c) => sum + c.groundingScore, 0) / candidates.length;

  // Penalty for ambiguities
  const ambiguityPenalty = Math.min(0.5, ambiguities.length * 0.1);

  return Math.max(0, avgScore - ambiguityPenalty);
}

function hashSymbols(symbols: SymbolicStructure): string {
  const content = JSON.stringify({
    termCount: symbols.terms.length,
    constraintCount: symbols.constraints.length,
  });

  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}
