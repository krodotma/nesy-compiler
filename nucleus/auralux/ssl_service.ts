
import * as ort from 'onnxruntime-web';

export interface SSLConfig {
    modelUrl: string;
    inputSampleRate: number; // e.g. 16000
    outputDim: number; // e.g. 256 for HuBERT-soft / 768 for Base
    expectedFrameCount: number; // e.g. 1 for streaming or N
}

export interface SSLMetrics {
    lastInferenceMs: number;
    totalCalls: number;
    avgLatencyMs: number;
}

export class SSLService {
    private session: ort.InferenceSession | null = null;
    private config: SSLConfig;
    private initialized = false;

    // Performance Metrics
    private metrics: SSLMetrics = {
        lastInferenceMs: 0,
        totalCalls: 0,
        avgLatencyMs: 0,
    };
    private totalLatencySum = 0;

    constructor(config: SSLConfig) {
        this.config = config;
    }

    async init(): Promise<void> {
        try {
            this.session = await ort.InferenceSession.create(this.config.modelUrl, {
                executionProviders: ['webgpu', 'wasm'],
            });
            this.initialized = true;
            console.log('[SSLService] Initialized HuBERT-soft model');
        } catch (e) {
            console.error('[SSLService] Failed to initialize:', e);
            throw e;
        }
    }

    /**
     * Extract features from audio chunk.
     * HuBERT expect raw waveform.
     * 
     * @param audio Audio chunk (Float32Array). Size must match model expectations if static.
     * @returns Feature tensor (Float32Array flattened).
     */
    async extractFeatures(audio: Float32Array): Promise<Float32Array> {
        if (!this.initialized || !this.session) {
            throw new Error('SSLService not initialized');
        }

        const startTime = performance.now();

        // Prepare ONNX Input
        // Shape: [1, samples]
        const input = new ort.Tensor('float32', audio, [1, audio.length]);

        const feeds: Record<string, ort.Tensor> = {
            // Input name depends on specific ONNX export. 
            // Assuming 'source' or 'audio' or 'input_values'. 
            // For standard HuBERT export it's often 'input_values' or 'waverform'.
            // We will assume 'input_values' for HuggingFace exports.
            input_values: input,
        };

        try {
            const results = await this.session.run(feeds);

            // Output name also depends on export.
            // Usually 'last_hidden_state' or 'logits' or 'projection'.
            // For HuBERT-soft voice conversion exports it might be 'units'.
            // We assume 'last_hidden_state' for generic model or 'units' if soft-vc.
            // Let's assume a generic key check or specific key 'last_hidden_state'.

            // We grab the first output available if name is unknown, or specific.
            const outputKey = this.session.outputNames[0];
            const output = results[outputKey];

            // Update Metrics
            const latency = performance.now() - startTime;
            this.metrics.lastInferenceMs = latency;
            this.metrics.totalCalls++;
            this.totalLatencySum += latency;
            this.metrics.avgLatencyMs = this.totalLatencySum / this.metrics.totalCalls;

            return output.data as Float32Array;
        } catch (e) {
            console.error('[SSLService] Inference failed:', e);
            throw e;
        }
    }

    /**
     * Dispose of ONNX session and free resources.
     */
    async dispose(): Promise<void> {
        if (this.session) {
            await this.session.release();
            this.session = null;
            this.initialized = false;
            console.log('[SSLService] Disposed');
        }
    }

    /**
     * Get current performance metrics.
     */
    getMetrics(): SSLMetrics {
        return { ...this.metrics };
    }
}

