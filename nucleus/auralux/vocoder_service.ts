
import * as ort from 'onnxruntime-web';

export interface VocoderConfig {
    modelUrl: string; // e.g., vocos-stream-int8.onnx
    sampleRate: number; // e.g., 24000
    hopLength: number; // e.g., 256
}

export interface VocoderMetrics {
    lastSynthesisMs: number;
    totalSynthesisCalls: number;
    avgLatencyMs: number;
}

export class VocoderService {
    private session: ort.InferenceSession | null = null;
    private config: VocoderConfig;
    private correlationBuffer: Float32Array; // For overlap-add

    // Performance Metrics
    private metrics: VocoderMetrics = {
        lastSynthesisMs: 0,
        totalSynthesisCalls: 0,
        avgLatencyMs: 0,
    };
    private totalLatencySum = 0;

    constructor(config: VocoderConfig) {
        this.config = config;
        // Buffer for overlap-add. Size depends on fade length. 
        // Vocos isn't strictly overlap-add in some configs, 
        // but streaming artifacts usually require cross-fading edges.
        // We'll allocate a small buffer for now. 
        // Typically hopLength / 4 or similar.
        this.correlationBuffer = new Float32Array(config.hopLength);
    }

    async init(): Promise<void> {
        try {
            this.session = await ort.InferenceSession.create(this.config.modelUrl, {
                executionProviders: ['webgpu', 'wasm'],
            });
            console.log('[VocoderService] Initialized Vocos model');
        } catch (e) {
            console.error('[VocoderService] Failed to initialize:', e);
            throw e;
        }
    }

    /**
     * Synthesize audio from semantic features.
     * @param features Float32Array [1, frames, dim]
     * @returns Float32Array Audio PCM
     */
    async synthesize(features: Float32Array, frames: number, dim: number): Promise<Float32Array> {
        if (!this.session) throw new Error('Vocoder not initialized');

        const startTime = performance.now();

        // Prepare Input
        // Vocos expects [1, dim, frames] usually for Conv1d, or [1, frames, dim] depending on export.
        // The research doc says: `dummy_mel = torch.randn(1, 100, 100) # [batch, n_mels, frames]`
        // So likely [1, dim, frames].

        // We might need to transpose if input features are [frames, dim].
        // Assuming features is matching the expected layout (flat array). 

        // NOTE: If features comes from SSLService, check its layout! 
        // HuBERT usually [batch, frames, dim]. 
        // Vocos wants [batch, dim, frames] usually.
        // We will assume TRANSPOSE IS NEEDED or done before.
        // For this implementation, let's assume we pass it in correct shape OR reshape here.
        // Let's implement transpose for safety if dim is provided.

        const transposed = this.transposeFramesToChannels(features, frames, dim);

        const input = new ort.Tensor('float32', transposed, [1, dim, frames]);

        const feeds = {
            features: input, // Name from export
        };

        const results = await this.session.run(feeds);

        // Output: 'audio' -> [1, samples]
        const audioData = results.audio.data as Float32Array;

        // Update Metrics
        const latency = performance.now() - startTime;
        this.metrics.lastSynthesisMs = latency;
        this.metrics.totalSynthesisCalls++;
        this.totalLatencySum += latency;
        this.metrics.avgLatencyMs = this.totalLatencySum / this.metrics.totalSynthesisCalls;

        // Apply Overlap-Add / Smoothing if streaming
        // For Phase 3 prototype, we might just return raw chunks.
        return audioData;
    }

    private transposeFramesToChannels(data: Float32Array, frames: number, dim: number): Float32Array {
        // Input: [frames, dim] (row-major) -> Output: [dim, frames] (col-major essentially)
        const result = new Float32Array(data.length);
        for (let f = 0; f < frames; f++) {
            for (let d = 0; d < dim; d++) {
                // Source index: f * dim + d
                // Dest index: d * frames + f
                result[d * frames + f] = data[f * dim + d];
            }
        }
        return result;
    }

    /**
     * Dispose of ONNX session and free resources.
     */
    async dispose(): Promise<void> {
        if (this.session) {
            await this.session.release();
            this.session = null;
            console.log('[VocoderService] Disposed');
        }
    }

    /**
     * Get current performance metrics.
     */
    getMetrics(): VocoderMetrics {
        return { ...this.metrics };
    }
}

