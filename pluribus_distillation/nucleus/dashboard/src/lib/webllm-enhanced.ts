/**
 * WebLLM Enhanced - Extended model registry, version checking, and Dialogos protocol
 *
 * Features:
 * - Comprehensive model registry with quantization options
 * - 4-hour version checking with localStorage caching
 * - TensorFlow.js / ONNX Runtime Web backend detection
 * - Dialogos dual-mind conversation seeds
 */

import type { BackendStatus, InferenceBackend, ModelCapability, ModelDef, EdgeModelDef, QuantType } from './model_registry_types';
import { EDGE_MODEL_CATALOG, MODEL_REGISTRY } from './model_registry_data';
export { EDGE_MODEL_CATALOG, MODEL_REGISTRY };
export type {
  BackendStatus,
  EdgeBackend,
  EdgeModelDef,
  InferenceBackend,
  ModelAvailability,
  ModelBackend,
  ModelCapability,
  ModelDef,
  QuantType,
} from './model_registry_types';

// ============================================================================
// CONSTANTS
// ============================================================================
export const MODEL_VERSION_CHECK_INTERVAL_MS = 4 * 60 * 60 * 1000; // 4 hours
export const LOCAL_STORAGE_KEY_VERSIONS = 'webllm_model_versions';
export const LOCAL_STORAGE_KEY_LAST_CHECK = 'webllm_last_version_check';
export const LOCAL_STORAGE_KEY_DIALOGOS = 'webllm_dialogos_state';

// Dialogos symbolic seed phrases for dual-mind conversations
export const DIALOGOS_SEEDS = [
  "Greetings. I am ready to discuss the architectural implications of Entelexis.",
  "Let us explore the topology of recursive self-improvement in bounded systems.",
  "Consider the phenomenology of distributed cognition across silicon substrates.",
  "The omega point of computation: what lies beyond the Turing threshold?",
  "Reflect on the isomorphism between consciousness and thermodynamic entropy.",
  "In what sense can we speak of agency emerging from deterministic processes?",
  "Fractal concurrency: how do parallel thoughts weave into coherent action?",
  "The membrane between simulation and reality grows ever thinner.",
];

// ============================================================================
// TYPES
// ============================================================================
export interface DialogosState {
  active: boolean;
  turnCount: number;
  lastSpeaker: string | null;
  conversationId: string;
  startedAt: number;
}

// ============================================================================
// MODEL REGISTRY
// ============================================================================
// Canonical registry: nucleus/specs/model_registry.json (exported to model_registry_data.ts)
//
// NEMOTRON SUPPORT:
// - Nemotron-Mini-4B-Instruct: Standard transformer (GQA+RoPE), MLC-compatible
//   Compile with: python nucleus/tools/compile_nemotron_webllm.py
//   Update nucleus/specs/model_registry.json and re-export via model_registry.py
//
// - Nemotron-3-Nano: Mamba-Transformer hybrid - NOT YET SUPPORTED by MLC
//   Track: https://github.com/mlc-ai/mlc-llm/issues for Mamba support

// ============================================================================
// HELPERS
// ============================================================================

/** Build full model ID from base + quantization */
export function buildModelId(baseId: string, quant: QuantType): string {
  return `${baseId}-${quant}-MLC`;
}

/** Get model definition by base ID */
export function getModelDef(baseId: string): ModelDef | undefined {
  return MODEL_REGISTRY.find(m => m.baseId === baseId);
}

/** Get model definition from a full model ID */
export function getModelDefByModelId(modelId: string): ModelDef | undefined {
  return MODEL_REGISTRY.find((m) => modelId.startsWith(`${m.baseId}-`));
}

/** Get fastest models suitable for Dialogos (speedRating >= 4) */
export function getDialogosModels(): ModelDef[] {
  return MODEL_REGISTRY.filter(
    (m) => m.speedRating >= 4 && m.backend === 'webllm' && m.availability === 'prebuilt'
  );
}

/** Select random Dialogos seed phrase */
export function selectDialogosSeed(): string {
  return DIALOGOS_SEEDS[Math.floor(Math.random() * DIALOGOS_SEEDS.length)];
}

/** Get models by capability (defaults to prebuilt WebLLM) */
export function getModelsByCapability(
  capability: ModelCapability,
  opts: { prebuiltOnly?: boolean } = {}
): ModelDef[] {
  const prebuiltOnly = opts.prebuiltOnly ?? true;
  return MODEL_REGISTRY.filter((m) => {
    if (!m.capabilities.includes(capability)) return false;
    if (!prebuiltOnly) return true;
    return m.backend === 'webllm' && m.availability === 'prebuilt';
  });
}

/** Auto-cache selection based on device memory (prefers 3-4 models, expands on high-memory devices) */
export function getAutoCacheModels(opts: { deviceMemoryGB?: number; limit?: number } = {}): ModelDef[] {
  const deviceMemoryGB = opts.deviceMemoryGB ?? 16;
  const defaultLimit = deviceMemoryGB >= 32 ? 8 : deviceMemoryGB >= 24 ? 6 : deviceMemoryGB >= 16 ? 4 : deviceMemoryGB >= 8 ? 4 : 3;
  const limit = opts.limit ?? defaultLimit;
  const pick = (baseId: string) => MODEL_REGISTRY.find((m) => m.baseId === baseId);
  const prebuiltOnly = (m: ModelDef | undefined) => m && m.backend === 'webllm' && m.availability === 'prebuilt';

  const highTierBase = [
    pick('gemma-2-9b-it'),
    pick('Phi-3.5-vision-instruct'),
    pick('Qwen2.5-3B-Instruct'),
    pick('SmolLM2-360M-Instruct'),
  ].filter(prebuiltOnly) as ModelDef[];

  const highTierExtras = [
    pick('GLM-4-9B-Chat'),
    pick('Llama-3.2-11B-Vision-Instruct'),
    pick('Llama-3.2-3B-Instruct'),
    pick('gemma-2-2b-it'),
  ].filter(prebuiltOnly) as ModelDef[];

  const highTier = [...highTierBase, ...highTierExtras];

  const midTier = [
    pick('Phi-3.5-vision-instruct'),
    pick('Qwen2.5-3B-Instruct'),
    pick('gemma-2-2b-it'),
    pick('SmolLM2-360M-Instruct'),
  ].filter(prebuiltOnly) as ModelDef[];

  const lowTier = [
    pick('Llama-3.2-1B-Instruct'),
    pick('Qwen3-0.6B'),
    pick('SmolLM2-360M-Instruct'),
  ].filter(prebuiltOnly) as ModelDef[];

  const chosen = deviceMemoryGB >= 16 ? highTier : deviceMemoryGB >= 8 ? midTier : lowTier;
  return chosen.slice(0, limit);
}

/** Generate flat model list for backwards compatibility */
export function getAvailableModels() {
  return MODEL_REGISTRY
    .filter((m) => m.backend === 'webllm' && m.availability === 'prebuilt')
    .map(m => ({
      id: buildModelId(m.baseId, m.defaultQuant),
      name: m.name,
      vram: m.vramBase,
      vramMB: m.vramMB,
      color: m.color,
      baseId: m.baseId,
      quant: m.defaultQuant,
      speedRating: m.speedRating,
      category: m.category,
    }));
}

export function getEdgeModelCatalog(): EdgeModelDef[] {
  return EDGE_MODEL_CATALOG.slice();
}

// ============================================================================
// VERSION CHECKING
// ============================================================================

/** Check if model versions need updating (runs every 4 hours) */
export async function checkModelVersions(): Promise<{
  needsUpdate: boolean;
  versions: Record<string, string>;
  checkedAt: number;
}> {
  if (typeof localStorage === 'undefined') {
    return { needsUpdate: false, versions: {}, checkedAt: 0 };
  }

  const lastCheck = parseInt(localStorage.getItem(LOCAL_STORAGE_KEY_LAST_CHECK) || '0', 10);
  const now = Date.now();

  // Skip check if within 4-hour window
  if (now - lastCheck < MODEL_VERSION_CHECK_INTERVAL_MS) {
    try {
      const cached = JSON.parse(localStorage.getItem(LOCAL_STORAGE_KEY_VERSIONS) || '{}');
      return { needsUpdate: false, versions: cached, checkedAt: lastCheck };
    } catch {
      return { needsUpdate: false, versions: {}, checkedAt: lastCheck };
    }
  }

  // Perform version check via WebLLM's model list
  try {
    const webllm = await import('@mlc-ai/web-llm');
    const modelList = (webllm as any).prebuiltAppConfig?.model_list || [];

    const versions: Record<string, string> = {};
    for (const model of modelList) {
      if (model.model_id && model.model_lib) {
        // Extract version from model_lib URL if present
        const match = String(model.model_lib).match(/v(\d+\.\d+\.\d+)/);
        versions[model.model_id] = match ? match[1] : 'latest';
      }
    }

    // Store previous versions for comparison
    const prevVersionsStr = localStorage.getItem(LOCAL_STORAGE_KEY_VERSIONS);
    localStorage.setItem(LOCAL_STORAGE_KEY_VERSIONS, JSON.stringify(versions));
    localStorage.setItem(LOCAL_STORAGE_KEY_LAST_CHECK, String(now));

    // Compare with previously cached versions
    let needsUpdate = false;
    try {
      const oldVersions = JSON.parse(prevVersionsStr || '{}');
      for (const [modelId, ver] of Object.entries(versions)) {
        if (oldVersions[modelId] && oldVersions[modelId] !== ver) {
          needsUpdate = true;
          console.log(`[WebLLM] Model ${modelId} has new version: ${oldVersions[modelId]} → ${ver}`);
        }
      }
    } catch { /* ignore */ }

    return { needsUpdate, versions, checkedAt: now };
  } catch (err) {
    console.warn('[WebLLM] Version check failed:', err);
    return { needsUpdate: false, versions: {}, checkedAt: now };
  }
}

// ============================================================================
// BACKEND DETECTION
// ============================================================================

/** Check TensorFlow.js availability (via global check - avoids bundler issues) */
export async function checkTfjsAvailability(): Promise<BackendStatus> {
  // Check if TensorFlow.js is loaded globally (e.g., via CDN script tag)
  const globalTf = typeof window !== 'undefined' ? (window as unknown as Record<string, unknown>).tf : undefined;
  if (globalTf && typeof (globalTf as Record<string, unknown>).ready === 'function') {
    try {
      await ((globalTf as Record<string, unknown>).ready as () => Promise<void>)();
      return {
        available: true,
        version: String((globalTf as Record<string, Record<string, unknown>>).version?.tfjs || 'unknown'),
        error: null,
      };
    } catch (err) {
      return {
        available: false,
        version: null,
        error: err instanceof Error ? err.message : 'TensorFlow.js init failed',
      };
    }
  }
  return {
    available: false,
    version: null,
    error: 'TensorFlow.js not loaded (add via CDN if needed)',
  };
}

/** Check ONNX Runtime Web availability (via global check - avoids bundler issues) */
export async function checkOnnxAvailability(): Promise<BackendStatus> {
  // Check if ONNX Runtime is loaded globally (e.g., via CDN script tag)
  const globalOrt = typeof window !== 'undefined' ? (window as unknown as Record<string, unknown>).ort : undefined;
  if (globalOrt) {
    return {
      available: true,
      version: String((globalOrt as Record<string, Record<string, Record<string, unknown>>>).env?.versions?.onnxruntime || 'unknown'),
      error: null,
    };
  }
  return {
    available: false,
    version: null,
    error: 'ONNX Runtime not loaded (add via CDN if needed)',
  };
}

/** Check all inference backends */
export async function checkAllBackends(): Promise<Record<InferenceBackend, BackendStatus>> {
  const [tfjs, onnx] = await Promise.all([
    checkTfjsAvailability(),
    checkOnnxAvailability(),
  ]);

  return {
    webllm: { available: true, version: 'native', error: null }, // Always available if this code runs
    tfjs,
    onnx,
  };
}

// ============================================================================
// DIALOGOS PROTOCOL
// ============================================================================

/** Create new Dialogos conversation state */
export function createDialogosState(): DialogosState {
  return {
    active: false,
    turnCount: 0,
    lastSpeaker: null,
    conversationId: `dialogos-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
    startedAt: 0,
  };
}

/** Save Dialogos state to localStorage */
export function saveDialogosState(state: DialogosState): void {
  if (typeof localStorage !== 'undefined') {
    localStorage.setItem(LOCAL_STORAGE_KEY_DIALOGOS, JSON.stringify(state));
  }
}

/** Load Dialogos state from localStorage */
export function loadDialogosState(): DialogosState | null {
  if (typeof localStorage === 'undefined') return null;
  try {
    const stored = localStorage.getItem(LOCAL_STORAGE_KEY_DIALOGOS);
    return stored ? JSON.parse(stored) : null;
  } catch {
    return null;
  }
}

/** Format Dialogos peer message for injection */
export function formatDialogosPeerMessage(
  senderName: string,
  content: string,
  turnCount: number
): string {
  return `[Dialogos Turn ${turnCount}] ${senderName} says: ${content}`;
}

/** Omega intervention message for stalled conversations */
export function getOmegaIntervention(turnCount: number): string {
  const interventions = [
    "System: [Omega Protocol] The conversation has stalled. Please propose a new theoretical direction regarding 'Fractal Concurrency'.",
    "System: [Omega Protocol] Silence detected. Consider exploring the recursive nature of self-reference in computational systems.",
    "System: [Omega Protocol] Dialogue paused. What are the implications of non-deterministic state transitions?",
    "System: [Omega Protocol] Resuming discourse. The topology of ideas requires continuous exploration.",
  ];
  return interventions[turnCount % interventions.length];
}

// ============================================================================
// CUSTOM MODEL LOADING (for self-compiled models like Nemotron)
// ============================================================================

export interface CustomModelConfig {
  modelId: string;
  modelLib: string; // URL to .wasm file
  modelWeights: string; // URL to HuggingFace repo with weights
  vramRequired: number; // MB
  contextWindowSize?: number;
  convTemplate?: string;
}

const LOCAL_STORAGE_KEY_CUSTOM_MODELS = 'webllm_custom_models';

/** Register a custom-compiled model for WebLLM */
export function registerCustomModel(config: CustomModelConfig): void {
  if (typeof localStorage === 'undefined') return;

  const stored = localStorage.getItem(LOCAL_STORAGE_KEY_CUSTOM_MODELS);
  const models: CustomModelConfig[] = stored ? JSON.parse(stored) : [];

  // Update or add
  const idx = models.findIndex(m => m.modelId === config.modelId);
  if (idx >= 0) {
    models[idx] = config;
  } else {
    models.push(config);
  }

  localStorage.setItem(LOCAL_STORAGE_KEY_CUSTOM_MODELS, JSON.stringify(models));
}

/** Get all registered custom models */
export function getCustomModels(): CustomModelConfig[] {
  if (typeof localStorage === 'undefined') return [];
  try {
    const stored = localStorage.getItem(LOCAL_STORAGE_KEY_CUSTOM_MODELS);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

/** Remove a custom model registration */
export function unregisterCustomModel(modelId: string): void {
  if (typeof localStorage === 'undefined') return;

  const stored = localStorage.getItem(LOCAL_STORAGE_KEY_CUSTOM_MODELS);
  if (!stored) return;

  const models: CustomModelConfig[] = JSON.parse(stored);
  const filtered = models.filter(m => m.modelId !== modelId);
  localStorage.setItem(LOCAL_STORAGE_KEY_CUSTOM_MODELS, JSON.stringify(filtered));
}

/**
 * Register Nemotron-Mini-4B after compiling with compile_nemotron_webllm.py
 *
 * Usage:
 *   registerNemotronMini4B({
 *     wasmUrl: 'https://huggingface.co/your-repo/resolve/main/Nemotron-Mini-4B-Instruct-q4f16_1-webgpu.wasm',
 *     weightsRepo: 'your-username/Nemotron-Mini-4B-Instruct-q4f16_1-MLC',
 *   });
 */
export function registerNemotronMini4B(opts: {
  wasmUrl: string;
  weightsRepo: string;
  quant?: QuantType;
}): void {
  const quant = opts.quant || 'q4f16_1';
  registerCustomModel({
    modelId: `Nemotron-Mini-4B-Instruct-${quant}-MLC`,
    modelLib: opts.wasmUrl,
    modelWeights: `https://huggingface.co/${opts.weightsRepo}`,
    vramRequired: 2500,
    contextWindowSize: 4096,
    convTemplate: 'chatml',
  });
}

// ============================================================================
// INDUCTIVE CONVERSATION SUPPORT (for dual-mind-channel)
// ============================================================================

/** Topics for inductive conversation mode */
export const INDUCTIVE_TOPICS = [
  'consciousness and emergence',
  'mathematical patterns in nature',
  'language and thought',
  'creativity and computation',
  'time and causality',
  'memory and identity',
  'ethics and decision-making',
  'perception and reality',
  'learning and adaptation',
  'cooperation and competition',
] as const;

/** Select a random inductive topic */
export function selectInductiveTopic(): string {
  return INDUCTIVE_TOPICS[Math.floor(Math.random() * INDUCTIVE_TOPICS.length)];
}

/** Generate an inductive conversation starter */
export function generateInductiveStarter(topic: string): string {
  const starters = [
    `Let's explore the nature of ${topic}. What patterns do you notice?`,
    `Consider ${topic} from first principles. What emerges?`,
    `I've been reflecting on ${topic}. Share your perspective.`,
    `The intersection of ${topic} with our current context is fascinating. Elaborate.`,
    `What hidden connections exist between ${topic} and consciousness?`,
  ];
  return starters[Math.floor(Math.random() * starters.length)];
}

// ============================================================================
// EXPORTS
// ============================================================================
export default {
  MODEL_REGISTRY,
  EDGE_MODEL_CATALOG,
  DIALOGOS_SEEDS,
  buildModelId,
  getModelDef,
  getModelDefByModelId,
  getDialogosModels,
  selectDialogosSeed,
  getAvailableModels,
  getEdgeModelCatalog,
  getAutoCacheModels,
  getModelsByCapability,
  checkModelVersions,
  checkTfjsAvailability,
  checkOnnxAvailability,
  checkAllBackends,
  createDialogosState,
  saveDialogosState,
  loadDialogosState,
  formatDialogosPeerMessage,
  getOmegaIntervention,
  // Custom model loading (Nemotron etc)
  registerCustomModel,
  getCustomModels,
  unregisterCustomModel,
  registerNemotronMini4B,
};
