/**
 * Chromatic Agents Visualizer - Module Exports
 *
 * Prism metaphor visualization for parallel agent activity.
 */

// Main component
export { ChromaticVisualizer, default } from './ChromaticVisualizer';

// Core classes - NOTE: PrismCore and AgentTree are intentionally NOT exported here
// They contain static THREE.js imports (708KB) and should only be dynamically imported
// via ChromaticVisualizer's useVisibleTask$ to enable proper code splitting

// P5.js HUD Layer (Steps 9-11)
export { AgentHUD } from './AgentHUD';
export { Sparkline, createAgentSparklines, generateMockSparklineData } from './Sparkline';
export type { SparklineData, SparklineDataPoint } from './Sparkline';
export { MergeAnimation, createMergeAnimation, onPushSuccess } from './MergeAnimation';

// Bus Integration (Steps 16-17)
export {
  BusSubscription,
  getDefaultBusSubscription,
  createBusSubscription,
} from './BusSubscription';
export type {
  BusSubscriptionOptions,
  BusSubscriptionState,
  BusEventHandler,
  VisualizationEventHandler,
  StateChangeHandler,
} from './BusSubscription';

// Code Graph Parser (Steps 3 & 16-17)
export {
  parseDiffStat,
  parseNumstat,
  buildCodeGraph,
  parseAndBuildGraph,
  mergeGraphs,
  getGraphStats,
  detectDependencies,
  handleWorkerMessage,
} from './CodeGraphParser';
export type {
  DiffStatEntry,
  ParseOptions,
  WorkerMessage,
  WorkerResponse,
} from './CodeGraphParser';

// Mutation System (Steps 12-15)
export {
  MutationQueue,
  mutationQueue,
  MutationPriority,
} from './MutationQueue';
export type {
  ChromaticBusEvent,
  MutationType,
  QueuedMutation,
} from './MutationQueue';

export {
  AgentStateMachine,
  StateMachineManager,
  stateMachineManager,
} from './AgentStateMachine';
export type {
  StateTransitionEvent,
  StateMachineSnapshot,
} from './AgentStateMachine';

export {
  ParticleSystem,
  ParticlePool,
  ParticleEmitter,
  particleSystem,
  createParticleStream,
  PARTICLE_VERTEX_SHADER,
  PARTICLE_FRAGMENT_SHADER,
} from './ParticleSystem';
export type {
  Particle,
  StreamType,
  EmitterConfig,
  ParticleSystemStats,
} from './ParticleSystem';

// Types
export type {
  AgentId,
  AgentVisualEvent,
  AgentVisualData,
  ChromaticState,
  CodeGraph,
  CodeNode,
  PrismCoreState,
  TreeNodeVisual,
  VisualMutation,
  ParticleConfig,
  ParticleStream,
  CameraState,
  AgentHUDData,
  HUDState,
  VizTopic,
} from './types';

// Values (enums and constants)
export { AgentVisualState, VIZ_TOPICS } from './types';

// Utilities
export {
  AGENT_COLORS,
  getAgentColor,
  getAgentHue,
  getAgentRGB,
  getAgentRGBNormalized,
  getAgentColorWithIntensity,
  getAgentGlowColor,
  getAgentGhostColor,
  interpolateAgentColors,
  hexToThreeColor,
  getAgentThreeColor,
  getAgentOrbitalPosition,
  generateAgentCSSVariables,
  getAgentOrder,
  getAllAgentIds,
} from './utils/colorMap';

export {
  // Easing functions
  linear,
  easeOutQuad,
  easeInQuad,
  easeInOutQuad,
  easeOutCubic,
  easeOutElastic,
  easeOutBack,
  // Interpolation
  lerp,
  lerpEased,
  lerp3D,
  lerpHSL,
  // Spring physics
  updateSpring,
  isSpringSettled,
  Spring3D,
  // Decay
  exponentialDecay,
  smoothDamp,
  // Utilities
  clamp,
  mapRange,
  smoothstep,
  pingPong,
} from './utils/interpolation';

// =============================================================================
// Polish & Accessibility (Steps 18-20)
// =============================================================================

// Step 18: Reduced Motion & Accessibility
export {
  ReducedMotionManager,
  getReducedMotionManager,
  createReducedMotionHook,
  AGENT_SHAPES,
} from './ReducedMotion';
export type {
  A11yConfig,
  ShapeIndicator,
  ScreenReaderDescription,
  UseReducedMotionReturn,
} from './ReducedMotion';

// Step 19: Replay Controller
export {
  ReplayController,
  getReplayController,
  renderScrubberHTML,
} from './ReplayController';
export type {
  StateSnapshot,
  SerializedChromaticState,
  SerializedAgentData,
  ReplayConfig,
  PlaybackState,
  ScrubberState,
  ExportOptions,
  ScrubberProps,
} from './ReplayController';

// Step 20: GLSL Shaders (paths for dynamic loading)
export const SHADER_PATHS = {
  glow: './shaders/glow.frag',
  particle: './shaders/particle.vert',
  merge: './shaders/merge.frag',
} as const;
