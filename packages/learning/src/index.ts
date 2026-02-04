/**
 * @nesy/learning - Learning and adaptation for neurosymbolic compilation
 *
 * This package implements the generative learning components:
 *
 * 1. RLCF - Reinforcement Learning from Compiler Feedback
 *    - Compilers are perfect reward functions
 *    - Learn to fix errors iteratively
 *
 * 2. Generator - Draft/Refine Code Generation
 *    - PCFG skeleton (draft)
 *    - SLM fills logic (refine)
 *    - Compiler verification
 *
 * 3. PCFG Grammar - Probabilistic Context-Free Grammar
 *    - Learn code structure from examples
 *    - Convert to GBNF for constrained decoding
 *
 * 4. Training Loop - The Denoise Loop
 *    - observation → decompose(aleatoric, epistemic)
 *    - denoise_epistemic(learn more)
 *    - accept_aleatoric(maintain diversity)
 *    - evolve
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

// PCFG Grammar - Probabilistic Context-Free Grammar
export {
  PCFGLearner,
  PCFGGenerator,
  pcfgToGBNF,
  createTypeScriptPCFG,
  createPythonPCFG,
  type PCFGRule,
  type PCFGGrammar,
  type GBNFGrammar,
  type ParseTreeNode,
} from './pcfg-grammar.js';

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
