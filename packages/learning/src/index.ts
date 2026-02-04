/**
 * @nesy/learning - Learning and adaptation for neurosymbolic compilation
 *
 * Generative learning components:
 * - RLCF: Reinforcement Learning from Compiler Feedback
 * - Generator: Draft/Refine Code Generation
 * - Training Loop: The Denoise Loop
 */

// Basic components
export { FeedbackCollector, type Feedback } from './feedback.js';
export { ExperienceBuffer, type Experience } from './experience.js';
export { AdaptationStrategy, type StrategyConfig } from './strategy.js';
export { MetricsTracker, type LearningMetrics } from './metrics.js';

// RLCF - Reinforcement Learning from Compiler Feedback
export {
  RLCFLoop,
  refineWithRLCF,
  type RLCFConfig,
  type RLCFState,
  type RLCFPolicy,
  type RLCFTransition,
  type CompilerDiagnostic,
  type FixStrategy,
} from './rlcf.js';

// Generator - Draft/Refine Code Generation
export {
  Generator,
  createGenerator,
  type GeneratorConfig,
  type GenerationRequest,
  type GenerationResult,
  type DraftResult,
  type RefineResult,
  type HoleSpec,
  type ContextItem,
  type GenerationConstraint,
  type SLMProvider,
} from './generator.js';

// Training Loop - The Denoise Loop
export {
  TrainingLoop,
  createTrainingLoop,
  type TrainingConfig,
  type TrainingState,
  type TrainingMetrics,
  type UncertaintyDecomposition,
  type EditTriplet,
} from './training-loop.js';

// Holon Exporter - Gold Standard Training Data Export
export {
  HolonExporter,
  type TrainingPair,
  type ExportConfig,
  type Holon,
} from './holon-exporter.js';

// Negative Sampler - Bad Examples from Rejected Thrash
export {
  NegativeSampler,
  type NegativeSample,
  type SamplingConfig,
} from './negative-sampler.js';

// Context Serializer - Graph Neighborhood to LLM Prompt
export {
  ContextSerializer,
  type GraphNeighborhood,
  type SerializedContext,
} from './context-serializer.js';
