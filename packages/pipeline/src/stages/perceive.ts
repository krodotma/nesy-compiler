/**
 * PERCEIVE Stage
 *
 * Input: Raw input (text, image, audio)
 * Output: NeuralFeatures (embedding + attention + confidence)
 *
 * This stage handles:
 * - Modality detection
 * - Tokenization
 * - Embedding generation
 * - Attention extraction
 * - Confidence estimation
 */

import type { CompilationContext, NeuralFeatures, Embedding, AttentionWeights } from '@nesy/core';
import type { PerceiveIR } from '../ir';
import { createIRNode, hashIR } from '../ir';

export interface PerceiveInput {
  text?: string;
  image?: Uint8Array;
  audio?: Float32Array;
}

export interface PerceiveOutput {
  ir: PerceiveIR;
  features: NeuralFeatures;
}

/**
 * Main perceive function
 */
export async function perceive(
  input: PerceiveInput,
  context: CompilationContext
): Promise<PerceiveOutput> {
  const modality = detectModality(input);
  const sourceText = extractSourceText(input, modality);

  // Check cache
  const cacheKey = hashInput(input);
  const cached = context.embeddingCache.get(cacheKey);

  let embedding: Embedding;
  if (cached) {
    embedding = {
      vector: cached,
      dimension: cached.length,
      model: context.model,
      timestamp: Date.now(),
    };
  } else {
    // Generate embedding (mock for now)
    embedding = await generateEmbedding(sourceText, context);
    context.embeddingCache.set(cacheKey, embedding.vector);
  }

  // Extract attention (mock for now)
  const attention = await extractAttention(sourceText, context);

  // Compute confidence
  const confidence = computeConfidence(embedding, attention);

  const features: NeuralFeatures = {
    embedding,
    attention,
    confidence,
  };

  const baseNode = createIRNode('neural', cacheKey, hashFeatures(features));
  const ir: PerceiveIR = {
    ...baseNode,
    kind: 'neural',
    features,
    sourceText,
    modality,
  };

  return { ir, features };
}

// =============================================================================
// Helpers
// =============================================================================

function detectModality(input: PerceiveInput): 'text' | 'image' | 'audio' | 'multimodal' {
  const hasText = !!input.text;
  const hasImage = !!input.image;
  const hasAudio = !!input.audio;

  if (hasText && !hasImage && !hasAudio) return 'text';
  if (!hasText && hasImage && !hasAudio) return 'image';
  if (!hasText && !hasImage && hasAudio) return 'audio';
  return 'multimodal';
}

function extractSourceText(
  input: PerceiveInput,
  modality: 'text' | 'image' | 'audio' | 'multimodal'
): string {
  if (input.text) return input.text;
  if (modality === 'image') return '[IMAGE_PLACEHOLDER]';
  if (modality === 'audio') return '[AUDIO_PLACEHOLDER]';
  return '[MULTIMODAL_PLACEHOLDER]';
}

function hashInput(input: PerceiveInput): string {
  const content = JSON.stringify({
    text: input.text,
    imageLen: input.image?.length,
    audioLen: input.audio?.length,
  });

  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

function hashFeatures(features: NeuralFeatures): string {
  const content = JSON.stringify({
    embDim: features.embedding.dimension,
    confidence: features.confidence,
  });

  let hash = 0;
  for (let i = 0; i < content.length; i++) {
    const char = content.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

async function generateEmbedding(
  text: string,
  context: CompilationContext
): Promise<Embedding> {
  // Mock embedding generation
  // In production, this would call an embedding API
  const dimension = 1536;
  const vector = new Float32Array(dimension);

  // Simple deterministic pseudo-embedding based on text
  for (let i = 0; i < dimension; i++) {
    vector[i] = Math.sin(text.charCodeAt(i % text.length) * (i + 1)) * 0.1;
  }

  return {
    vector,
    dimension,
    model: context.model,
    timestamp: Date.now(),
  };
}

async function extractAttention(
  text: string,
  context: CompilationContext
): Promise<AttentionWeights> {
  // Mock attention extraction
  // In production, this would come from model internals
  const numHeads = 12;
  const seqLen = Math.min(text.length, 100);

  const heads: Float32Array[] = [];
  for (let h = 0; h < numHeads; h++) {
    const weights = new Float32Array(seqLen);
    // Uniform distribution for mock
    for (let i = 0; i < seqLen; i++) {
      weights[i] = 1 / seqLen;
    }
    heads.push(weights);
  }

  return {
    heads,
    layers: 12,
    normalized: true,
  };
}

function computeConfidence(embedding: Embedding, attention: AttentionWeights): number {
  // Heuristic: attention entropy-based confidence
  let totalEntropy = 0;

  for (const head of attention.heads) {
    let entropy = 0;
    for (const w of head) {
      if (w > 0) {
        entropy -= w * Math.log2(w);
      }
    }
    totalEntropy += entropy;
  }

  const avgEntropy = totalEntropy / attention.heads.length;
  const maxEntropy = Math.log2(attention.heads[0].length);

  // Lower entropy = higher confidence
  return Math.max(0, 1 - (avgEntropy / maxEntropy));
}
