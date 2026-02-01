/**
 * Bridge layer primitives
 *
 * - Grounding: Neural → Symbolic
 * - Lifting: Symbolic → Neural
 * - Discretization: Continuous → Discrete
 */

import type {
  NeuralFeatures,
  SymbolicStructure,
  GroundingResult,
  LiftingResult,
  DiscretizationConfig,
  Term,
  Constraint,
  Embedding,
} from './types';

// =============================================================================
// Grounding (Neural → Symbolic)
// =============================================================================

export interface GroundingStrategy {
  name: string;
  ground(features: NeuralFeatures): Promise<GroundingResult>;
}

/**
 * Threshold-based grounding
 *
 * Maps high-confidence neural features to symbolic constants,
 * low-confidence features to variables.
 */
export class ThresholdGrounding implements GroundingStrategy {
  name = 'threshold';

  constructor(private threshold: number = 0.7) {}

  async ground(features: NeuralFeatures): Promise<GroundingResult> {
    const terms: Term[] = [];
    const ambiguities: string[] = [];

    if (features.confidence >= this.threshold) {
      terms.push({
        type: 'constant',
        name: `grounded_${Date.now()}`,
        value: { embedding: Array.from(features.embedding.vector), confidence: features.confidence },
      });
    } else {
      terms.push({
        type: 'variable',
        name: `neural_${Date.now()}`,
      });
      ambiguities.push(`Low confidence (${features.confidence.toFixed(2)}) - created variable`);
    }

    return {
      source: features,
      target: { terms, constraints: [], metadata: { derivation: ['threshold_grounding'] } },
      confidence: features.confidence,
      ambiguities,
    };
  }
}

/**
 * Softened symbol grounding using Boltzmann distribution
 */
export class SoftGrounding implements GroundingStrategy {
  name = 'soft';

  constructor(private config: DiscretizationConfig) {}

  async ground(features: NeuralFeatures): Promise<GroundingResult> {
    const scaledConfidence = this.applyTemperature(features.confidence);

    const terms: Term[] = [];
    const ambiguities: string[] = [];

    if (scaledConfidence >= this.config.threshold) {
      terms.push({
        type: 'constant',
        name: `soft_grounded_${Date.now()}`,
        value: { embedding: Array.from(features.embedding.vector), scaled: scaledConfidence },
      });
    } else {
      terms.push({
        type: 'variable',
        name: `soft_${Date.now()}`,
      });
      ambiguities.push(`Below threshold after softening (${scaledConfidence.toFixed(2)})`);
    }

    return {
      source: features,
      target: { terms, constraints: [], metadata: { derivation: ['soft_grounding'] } },
      confidence: scaledConfidence,
      ambiguities,
    };
  }

  private applyTemperature(confidence: number): number {
    const scaled = Math.exp(confidence / this.config.temperature);
    return Math.min(1, scaled / Math.E);
  }
}

// =============================================================================
// Lifting (Symbolic → Neural)
// =============================================================================

export interface LiftingStrategy {
  name: string;
  lift(structure: SymbolicStructure): Promise<LiftingResult>;
}

/**
 * Concatenation-based lifting
 *
 * Encodes symbolic structure as string, then embeds
 */
export class ConcatLifting implements LiftingStrategy {
  name = 'concat';

  constructor(private embedder: (text: string) => Promise<Embedding>) {}

  async lift(structure: SymbolicStructure): Promise<LiftingResult> {
    const serialized = this.serialize(structure);
    const embedding = await this.embedder(serialized);
    const lossEstimate = this.estimateLoss(structure);

    return {
      source: structure,
      target: embedding,
      lossEstimate,
    };
  }

  private serialize(structure: SymbolicStructure): string {
    const termStrings = structure.terms.map(t => this.termToString(t));
    const constraintStrings = structure.constraints.map(c => this.constraintToString(c));
    return `TERMS: ${termStrings.join(', ')}\nCONSTRAINTS: ${constraintStrings.join(', ')}`;
  }

  private constraintToString(c: Constraint): string {
    switch (c.type) {
      case 'equality':
        return `${this.termToString(c.left)} = ${this.termToString(c.right)}`;
      case 'inequality':
        return `${this.termToString(c.left)} != ${this.termToString(c.right)}`;
      case 'membership':
        return `${this.termToString(c.element)} in {${c.set.map(t => this.termToString(t)).join(', ')}}`;
      case 'custom':
        return `${c.name}(${c.args.map(t => this.termToString(t)).join(', ')})`;
    }
  }

  private termToString(term: Term): string {
    switch (term.type) {
      case 'variable':
        return `?${term.name}`;
      case 'constant':
        return term.name;
      case 'compound':
        return `${term.functor}(${term.args.map(a => this.termToString(a)).join(', ')})`;
    }
  }

  private estimateLoss(structure: SymbolicStructure): number {
    const termComplexity = structure.terms.length;
    const constraintComplexity = structure.constraints.length * 2;
    const metaComplexity = structure.metadata ? Object.keys(structure.metadata).length * 0.5 : 0;

    return Math.min(1, (termComplexity + constraintComplexity + metaComplexity) / 100);
  }
}

// =============================================================================
// Discretization
// =============================================================================

export function discretize(
  continuous: Float32Array,
  config: DiscretizationConfig
): Int32Array {
  const discrete = new Int32Array(continuous.length);

  for (let i = 0; i < continuous.length; i++) {
    const scaled = continuous[i] / config.temperature;
    const prob = 1 / (1 + Math.exp(-scaled));
    discrete[i] = prob >= config.threshold ? 1 : 0;
  }

  return discrete;
}

export function anneal(
  config: DiscretizationConfig,
  step: number,
  maxSteps: number
): DiscretizationConfig {
  const progress = step / maxSteps;

  let newTemp: number;
  switch (config.annealing) {
    case 'linear':
      newTemp = config.temperature * (1 - progress);
      break;
    case 'exponential':
      newTemp = config.temperature * Math.exp(-3 * progress);
      break;
    case 'cosine':
      newTemp = config.temperature * (1 + Math.cos(Math.PI * progress)) / 2;
      break;
  }

  return {
    ...config,
    temperature: Math.max(0.01, newTemp),
  };
}
