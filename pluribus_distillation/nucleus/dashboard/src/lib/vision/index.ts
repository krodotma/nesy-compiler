/**
 * Vision Module - SUPERWORKERS Visual Context Injection System
 *
 * This barrel export provides unified access to:
 * - WebRTC screen capture with golden ratio optimization
 * - RTCPeerConnection stats collection with φ-weighted scoring
 * - VLM provider routing with Fibonacci token budgets
 *
 * Golden Ratio Philosophy:
 * All quality metrics use φ (1.618033988749895) as the fundamental
 * optimization constant. Quality thresholds follow the inverse powers:
 *   - Excellent: ≥ 1/φ^0 = 1.000 (or PHI for ultra)
 *   - Good:      ≥ 1/φ^1 = 0.618
 *   - Fair:      ≥ 1/φ^2 = 0.382
 *   - Poor:      ≥ 1/φ^3 = 0.236
 *
 * @module vision
 * @see SUPERWORKERS.md Section 6 for architecture
 */

// =============================================================================
// CONSTANTS - Re-export golden ratio fundamentals
// =============================================================================

export {
  PHI,
  FIBONACCI,
  GOLDEN_QUALITY_TIERS,
  DEFAULT_CONSTRAINTS,
} from './screen-capture';

// =============================================================================
// SCREEN CAPTURE - WebRTC getDisplayMedia
// =============================================================================

export {
  // Core functions
  captureScreen,
  captureFrameForVLM,
  streamScreenWithStats,
  isScreenCaptureSupported,

  // Golden utilities
  calculateGoldenScore,
  fibonacciTier,
  goldenDimensions,

  // Types
  type ScreenCaptureOptions,
  type CapturedFrame,
  type ScreenCaptureStats,
  type StreamController,
} from './screen-capture';

// =============================================================================
// WEBRTC STATS - RTCPeerConnection metrics
// =============================================================================

export {
  // Core class
  WebRTCStatsCollector,

  // Standalone functions
  collectStats,
  createQualityMonitor,

  // Types
  type WebRTCMetrics,
  type StatsCollectorOptions,
  type VideoMetrics,
  type AudioMetrics,
  type ConnectionMetrics,
} from './webrtc-stats';

// =============================================================================
// QUALITY CLASSIFICATION - Defined locally to avoid export issues
// =============================================================================

import { PHI as PHI_LOCAL } from './screen-capture';

/** Quality classification based on golden ratio thresholds */
export type QualityClassification = 'excellent' | 'good' | 'fair' | 'poor' | 'critical';

/** Golden ratio quality thresholds */
const QUALITY_THRESHOLDS = {
  excellent: 1 / PHI_LOCAL + 0.2,
  good: 1 / PHI_LOCAL,
  fair: 1 / (PHI_LOCAL * PHI_LOCAL),
  poor: 1 / (PHI_LOCAL * PHI_LOCAL * PHI_LOCAL),
} as const;

/**
 * Classify a golden score into a quality tier.
 * @param score - Golden score (0-PHI range)
 * @returns Quality classification
 */
export function classifyQuality(score: number): QualityClassification {
  if (score >= QUALITY_THRESHOLDS.excellent) return 'excellent';
  if (score >= QUALITY_THRESHOLDS.good) return 'good';
  if (score >= QUALITY_THRESHOLDS.fair) return 'fair';
  if (score >= QUALITY_THRESHOLDS.poor) return 'poor';
  return 'critical';
}

// =============================================================================
// VLM INTEGRATION - Vision Language Model routing
// =============================================================================

export {
  // Provider registry
  VLM_PROVIDERS,

  // Core functions
  processVisionRequest,
  injectVisionContext,
  selectVLMProvider,

  // Types
  type VLMProvider,
  type VLMProviderName,
  type VisionTaskType,
  type VisionRequest,
  type VisionResponse,
} from './vlm-integration';

// Aliases for backward compatibility
export { selectVLMProvider as selectProvider } from './vlm-integration';
export type { VisionTaskType as VisionTask } from './vlm-integration';

/** Options for vision context injection */
export interface VisionInjectionOptions {
  captureScreen?: boolean;
  imageUrl?: string;
  taskType?: import('./vlm-integration').VisionTaskType;
  ocrMode?: 'full' | 'selective';
}

/** Get the system prompt for a vision task type */
export function getTaskSystemPrompt(taskType: import('./vlm-integration').VisionTaskType): string {
  const prompts: Record<string, string> = {
    'ocr-chinese': 'You are a vision-enabled SUPERWORKER specialized in Chinese OCR.',
    'ocr-english': 'You are a vision-enabled SUPERWORKER specialized in English OCR.',
    'ocr-multilingual': 'You are a vision-enabled SUPERWORKER specialized in multilingual OCR.',
    'gui-automation': 'You are a vision-enabled SUPERWORKER for GUI automation.',
    'screenshot-to-code': 'You are a vision-enabled SUPERWORKER for screenshot-to-code conversion.',
    'diagram-parse': 'You are a vision-enabled SUPERWORKER for diagram analysis.',
    'error-debug': 'You are a vision-enabled SUPERWORKER for error debugging.',
    'code-screenshot': 'You are a vision-enabled SUPERWORKER for code analysis.',
    'document-analysis': 'You are a vision-enabled SUPERWORKER for document analysis.',
    'visual-qa': 'You are a vision-enabled SUPERWORKER for visual question answering.',
  };
  return prompts[taskType] || prompts['visual-qa'];
}

// =============================================================================
// VLM CLIENTS - Real API implementations
// =============================================================================

export {
  // Client classes
  GLM4VClient,
  Qwen3VLClient,
  ClaudeVisionClient,
  GPT4VisionClient,

  // Factory and router
  createVLMClient,
  VLMRouter,

  // Types
  type VLMClient,
  type VLMClientConfig,
  type VLMRequest as VLMClientRequest,
  type VLMResponse as VLMClientResponse,
  type VLMProviderType,
} from './vlm-clients';

// =============================================================================
// SUPERWORKER ORCHESTRATOR - Master coordination
// =============================================================================

export {
  // Main orchestrator class
  SuperworkerOrchestrator,

  // Query analysis
  analyzeQuery,

  // Gestalt comprehension
  aggregateGestalt,

  // Context injection
  buildLayerInjection,
  assembleInjection,

  // Constants
  TOKEN_BUDGETS,
  ATTENTION_WINDOWS,
  DEPTH_PATTERNS,
  LAYER_TEMPLATES,
  SILVER_RATIO,
  PLASTIC_CONSTANT,
  LUCAS,

  // Types
  type InjectionLayer,
  type DepthLevel,
  type EffectsBudget,
  type CoordinationLane,
  type QualityGates,
  type GestaltState,
  type InjectedContext,
  type SuperworkerState,
  type OrchestratorConfig,
  type QueryAnalysis,
  type ContextSource,
} from './superworker-orchestrator';

// =============================================================================
// UNIFIED TYPES
// =============================================================================

/**
 * SUPERWORKER quality tier based on golden ratio inverse powers.
 */
export type GoldenQualityTier = 'excellent' | 'good' | 'fair' | 'poor' | 'critical';

/**
 * Unified vision context for injection into chat messages.
 */
export interface VisionContext {
  /** Captured frame data (base64 PNG) */
  frame?: CapturedFrame;
  /** Current WebRTC stream metrics */
  metrics?: WebRTCMetrics;
  /** Selected VLM provider for processing */
  provider?: VLMProviderName;
  /** Computed golden quality score (0-PHI range) */
  goldenScore: number;
  /** Derived quality tier */
  qualityTier: GoldenQualityTier;
  /** ISO timestamp of context capture */
  capturedAt: string;
}

// Import types for VisionContext
import type { CapturedFrame } from './screen-capture';
import type { WebRTCMetrics } from './webrtc-stats';
import type { VLMProviderName } from './vlm-integration';

// =============================================================================
// UNIFIED ORCHESTRATION
// =============================================================================

/**
 * Capture and process a single frame for SUPERWORKER context injection.
 * Orchestrates screen capture → VLM analysis → golden scoring.
 *
 * @param options - Capture and processing options
 * @returns Complete vision context ready for injection
 */
export async function captureVisionContext(options: {
  /** Screen capture preferences */
  captureOptions?: import('./screen-capture').ScreenCaptureOptions;
  /** VLM task type for system prompt selection */
  task?: import('./vlm-integration').VisionTaskType;
  /** User query to contextualize the vision request */
  userQuery?: string;
  /** Minimum acceptable golden score */
  minGoldenScore?: number;
}): Promise<VisionContext> {
  const { captureFrameForVLM, PHI } = await import('./screen-capture');

  const minScore = options.minGoldenScore ?? 1 / (PHI * PHI); // Default: fair (0.382)

  // Capture frame
  const frame = await captureFrameForVLM(options.captureOptions);

  // Determine quality tier using local classifyQuality
  const qualityTier = classifyQuality(frame.goldenScore);

  // Validate minimum quality
  if (frame.goldenScore < minScore) {
    console.warn(
      `[vision] Frame quality ${frame.goldenScore.toFixed(3)} below threshold ${minScore.toFixed(3)}`
    );
  }

  return {
    frame,
    goldenScore: frame.goldenScore,
    qualityTier: qualityTier as GoldenQualityTier,
    capturedAt: new Date().toISOString(),
  };
}

/**
 * Start continuous vision monitoring with stats emission.
 * Returns controller for lifecycle management.
 */
export async function startVisionMonitoring(options: {
  /** Frame callback */
  onFrame?: (ctx: VisionContext) => void;
  /** Stats callback */
  onStats?: (metrics: WebRTCMetrics) => void;
  /** Quality alert callback (triggered on degradation) */
  onQualityAlert?: (tier: GoldenQualityTier, score: number) => void;
  /** Capture interval in milliseconds */
  intervalMs?: number;
  /** Screen capture preferences */
  captureOptions?: import('./screen-capture').ScreenCaptureOptions;
}) {
  const { streamScreenWithStats, PHI } = await import('./screen-capture');

  let lastQualityTier: GoldenQualityTier = 'good';

  return streamScreenWithStats(
    (frame) => {
      // Use local classifyQuality
      const qualityTier = classifyQuality(frame.goldenScore) as GoldenQualityTier;

      const ctx: VisionContext = {
        frame,
        goldenScore: frame.goldenScore,
        qualityTier,
        capturedAt: new Date().toISOString(),
      };

      options.onFrame?.(ctx);

      // Check for quality degradation
      if (
        qualityTier !== lastQualityTier &&
        frame.goldenScore < 1 / PHI // Below "good" threshold
      ) {
        options.onQualityAlert?.(qualityTier, frame.goldenScore);
      }
      lastQualityTier = qualityTier;
    },
    (stats) => {
      options.onStats?.(stats as unknown as WebRTCMetrics);
    },
    { intervalMs: options.intervalMs ?? 1000, ...options.captureOptions }
  );
}

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

import { SuperworkerOrchestrator } from './superworker-orchestrator';
import { VLMRouter } from './vlm-clients';

export default {
  // Constants
  PHI: 1.618033988749895,
  FIBONACCI: [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584],

  // Quality thresholds (inverse φ powers)
  QUALITY_THRESHOLDS: {
    excellent: 1.0,
    good: 0.618,
    fair: 0.382,
    poor: 0.236,
  },

  // Orchestration
  captureVisionContext,
  startVisionMonitoring,
  classifyQuality,

  // Classes (for instantiation)
  SuperworkerOrchestrator,
  VLMRouter,
};
