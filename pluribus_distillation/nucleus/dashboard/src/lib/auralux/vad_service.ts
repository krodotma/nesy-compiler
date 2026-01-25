import { env, InferenceSession, Tensor } from 'onnxruntime-web';
import { AudioRingBuffer } from './ring-buffer';

export interface VADConfig {
    modelUrl: string;
    sampleRate: number;
    frameSize: number;
    threshold: number;
    minSpeechDurationMs: number;
    minSilenceDurationMs: number;
}

export interface VADEvent {
    type: 'speech_start' | 'speech_end' | 'silence';
    probability: number;
    ts: number;
}

export class VADService {
    private session: InferenceSession | null = null;
    private hn: Tensor;
    private cn: Tensor;
    private isSpeechActive = false;
    private isRunning = false;
    private processingInterval: any = null;

    constructor(
        private ringBuffer: AudioRingBuffer,
        private config: VADConfig
    ) {
        this.hn = new Tensor('float32', new Float32Array(2 * 1 * 64), [2, 1, 64]);
        this.cn = new Tensor('float32', new Float32Array(2 * 1 * 64), [2, 1, 64]);
    }

    async init() {
        // Only set env options if not already set or if strictly needed.
        // On recent onnxruntime-web versions, setting these after init might throw or be ignored.
        // We'll set them just in case, but wrap in try-catch if needed (omitted for brevity here).
        env.wasm.numThreads = 1;
        env.wasm.simd = true;

        this.session = await InferenceSession.create(this.config.modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        console.log('VADService initialized');
    }

    start(onEvent: (event: VADEvent) => void) {
        if (this.isRunning) return;
        this.isRunning = true;

        const frameSize = this.config.frameSize;
        const buffer = new Float32Array(frameSize);

        // Polling loop to read from ring buffer and process
        // In a real optimized system, this might be triggered by the ring buffer or worklet
        this.processingInterval = setInterval(async () => {
            if (!this.isRunning || !this.session) return;

            // Try to read a frame
            // Note: AudioRingBuffer.read might need to handle partial reads or we just skip
            // Assuming ringBuffer.read returns number of samples read
            const read = this.ringBuffer.read(buffer);
            
            if (read === frameSize) {
                await this.processFrame(buffer, onEvent);
            }
        }, 30); // ~32ms interval
    }

    stop() {
        this.isRunning = false;
        if (this.processingInterval) {
            clearInterval(this.processingInterval);
            this.processingInterval = null;
        }
    }

    private async processFrame(audioFrame: Float32Array, onEvent: (event: VADEvent) => void) {
        if (!this.session) return;

        const inputTensor = new Tensor('float32', audioFrame, [1, this.config.frameSize]);
        const srTensor = new Tensor('int64', BigInt64Array.from([BigInt(this.config.sampleRate)]));

        const feeds = { input: inputTensor, sr: srTensor, h: this.hn, c: this.cn };

        try {
            const results = await this.session.run(feeds);
            this.hn = results.hn;
            this.cn = results.cn;

            const probability = results.output.data[0] as number;
            const now = performance.now();

            // Simple Hysteresis
            if (probability > this.config.threshold && !this.isSpeechActive) {
                this.isSpeechActive = true;
                onEvent({ type: 'speech_start', probability, ts: now });
            } else if (probability < this.config.threshold - 0.15 && this.isSpeechActive) {
                this.isSpeechActive = false;
                onEvent({ type: 'speech_end', probability, ts: now });
            } else if (!this.isSpeechActive) {
                 // Optional: emit silence events if needed for visualization, 
                 // but throttle them to avoid bus flooding
                 // onEvent({ type: 'silence', probability, ts: now });
            }
        } catch (e) {
            console.error('VAD Inference error:', e);
        }
    }
}
