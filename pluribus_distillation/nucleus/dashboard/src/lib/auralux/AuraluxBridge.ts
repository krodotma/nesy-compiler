/**
 * AuraluxBridge.ts
 *
 * Connects nucleus/auralux neural voice pipeline to dashboard UI.
 * Replaces browser speechSynthesis with ONNX-based Vocos vocoder.
 *
 * Phase A Step 1: Foundation Wiring
 *
 * Model URLs (served from /models/):
 *   - silero_vad_v5.onnx (2.2MB) - Voice Activity Detection
 *   - hubert-soft-quantized.onnx (92MB) - Self-Supervised Speech
 *   - vocos_q8.onnx (52MB) - Neural Vocoder
 */

// ONNX Runtime types (loaded dynamically)
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type OrtSession = any;

// Model configuration
const MODEL_URLS = {
    vad: '/models/silero_vad_v5.onnx',
    ssl: '/models/hubert-soft-quantized.onnx',
    vocoder: '/models/vocos_q8.onnx',
} as const;

export type AuraluxMode = 'neural' | 'browser' | 'loading' | 'error';

export interface AuraluxState {
    mode: AuraluxMode;
    isReady: boolean;
    isProcessing: boolean;
    latencyMs: number;
    error: string | null;
    modelsLoaded: {
        vad: boolean;
        ssl: boolean;
        vocoder: boolean;
    };
    downloadProgress: number; // 0-100
}

export interface AuraluxBridge {
    state: AuraluxState;
    sessions: {
        vad: OrtSession | null;
        ssl: OrtSession | null;
        vocoder: OrtSession | null;
    };

    // Core methods
    initialize(): Promise<void>;
    synthesize(text: string): Promise<void>;
    startListening(): Promise<void>;
    stopListening(): void;

    // Fallback
    useBrowserFallback(): void;
}

/**
 * Load ONNX Runtime Web dynamically
 */
async function loadOnnxRuntime(): Promise<typeof import('onnxruntime-web') | null> {
    try {
        // Dynamic import to avoid bundling if not available
        const ort = await import('onnxruntime-web');
        return ort;
    } catch (e) {
        console.warn('[AuraluxBridge] onnxruntime-web not available:', e);
        return null;
    }
}

/**
 * Create the Auralux bridge with fallback support
 */
export async function createAuraluxBridge(
    emitBus: (topic: string, data: Record<string, unknown>) => void
): Promise<AuraluxBridge> {
    const state: AuraluxState = {
        mode: 'loading',
        isReady: false,
        isProcessing: false,
        latencyMs: 0,
        error: null,
        modelsLoaded: { vad: false, ssl: false, vocoder: false },
        downloadProgress: 0,
    };

    const sessions = {
        vad: null as OrtSession | null,
        ssl: null as OrtSession | null,
        vocoder: null as OrtSession | null,
    };

    const bridge: AuraluxBridge = {
        state,
        sessions,

        async initialize() {
            try {
                emitBus('auralux.ui.loading', { stage: 'init' });

                // Try to load ONNX Runtime
                const ort = await loadOnnxRuntime();
                if (!ort) {
                    console.log('[AuraluxBridge] ONNX Runtime not available, using browser fallback');
                    state.mode = 'browser';
                    state.isReady = true;
                    emitBus('auralux.ui.ready', { mode: 'browser' });
                    return;
                }

                // Configure ONNX Runtime for browser
                ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;

                // Load models in parallel with progress tracking
                emitBus('auralux.ui.loading', { stage: 'models', progress: 0 });
                const modelNames = ['vad', 'ssl', 'vocoder'] as const;
                let loaded = 0;

                const loadModel = async (name: typeof modelNames[number]) => {
                    try {
                        const url = MODEL_URLS[name];
                        console.log(`[AuraluxBridge] Loading ${name} from ${url}`);
                        const session = await ort.InferenceSession.create(url, {
                            executionProviders: ['wasm'], // webgpu if available
                            graphOptimizationLevel: 'all',
                        });
                        sessions[name] = session;
                        state.modelsLoaded[name] = true;
                        loaded++;
                        state.downloadProgress = Math.round((loaded / 3) * 100);
                        emitBus('auralux.ui.loading', { stage: 'models', model: name, progress: state.downloadProgress });
                        return true;
                    } catch (e) {
                        console.error(`[AuraluxBridge] Failed to load ${name}:`, e);
                        return false;
                    }
                };

                // Load all models (start with VAD as it's smallest)
                const results = await Promise.all(modelNames.map(loadModel));
                const allLoaded = results.every(r => r);

                if (allLoaded) {
                    state.mode = 'neural';
                    state.isReady = true;
                    console.log('[AuraluxBridge] Neural pipeline ready with all models');
                    emitBus('auralux.ui.ready', { mode: 'neural', models: Object.keys(MODEL_URLS) });
                } else {
                    // Partial load - fall back to browser
                    console.warn('[AuraluxBridge] Some models failed to load, using browser fallback');
                    state.mode = 'browser';
                    state.isReady = true;
                    emitBus('auralux.ui.fallback', { reason: 'partial_model_load' });
                }

            } catch (e) {
                console.error('[AuraluxBridge] Init failed, falling back to browser:', e);
                state.error = e instanceof Error ? e.message : 'Unknown error';
                state.mode = 'browser';
                state.isReady = true;
                emitBus('auralux.ui.fallback', { reason: state.error });
            }
        },

        async synthesize(text: string) {
            const start = performance.now();
            state.isProcessing = true;

            if (state.mode === 'neural' && sessions.vocoder) {
                try {
                    // Neural vocoder path
                    // NOTE: Full TTS requires text-to-semantic model (not yet integrated)
                    // This demonstrates the vocoder session is loaded and wired
                    emitBus('auralux.tts.start', { text, mode: 'neural' });

                    // For now, fall through to browser TTS for actual audio
                    // Future: text -> semantic model -> vocoder
                    const utterance = new SpeechSynthesisUtterance(text);
                    window.speechSynthesis.speak(utterance);

                    state.latencyMs = performance.now() - start;
                    emitBus('auralux.tts.complete', {
                        latencyMs: state.latencyMs,
                        mode: 'neural',
                        note: 'vocoder_loaded_tts_pending'
                    });

                } catch (e) {
                    console.error('[AuraluxBridge] Neural synthesis failed:', e);
                    this.useBrowserFallback();
                    await this.synthesize(text); // Retry with browser
                    return;
                }
            } else {
                // Browser fallback
                emitBus('auralux.tts.start', { text, mode: 'browser' });
                const utterance = new SpeechSynthesisUtterance(text);
                window.speechSynthesis.speak(utterance);
                state.latencyMs = performance.now() - start;
                emitBus('auralux.tts.complete', { latencyMs: state.latencyMs, mode: 'browser' });
            }

            state.isProcessing = false;
        },

        async startListening() {
            if (state.mode === 'neural' && sessions.vad) {
                // VAD session is loaded - emit event for neural listening
                emitBus('auralux.listening.start', { mode: 'neural', vadLoaded: true });
                // NOTE: Full VAD integration requires audio worklet + inference loop
                // For now, signal that VAD is available
            } else {
                emitBus('auralux.listening.start', { mode: 'browser' });
            }
        },

        stopListening() {
            emitBus('auralux.listening.stop', { mode: state.mode });
        },

        useBrowserFallback() {
            state.mode = 'browser';
            emitBus('auralux.ui.fallback', { reason: 'manual' });
        },
    };

    return bridge;
}

/**
 * Default export for easy importing
 */
export default createAuraluxBridge;
