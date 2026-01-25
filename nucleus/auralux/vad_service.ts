
import { AudioRingBuffer } from './ring_buffer';
import * as ort from 'onnxruntime-web';

export type VADEvent =
    | { type: 'speech_start', timestamp: number }
    | { type: 'speech_end', timestamp: number };

export interface VADConfig {
    modelUrl: string;
    sampleRate: number;
    frameSize: number; // e.g. 512 samples
    threshold: number; // 0.5 default
    minSpeechDurationMs: number;
    minSilenceDurationMs: number;
}

export class VADService {
    private session: ort.InferenceSession | null = null;
    private ringBuffer: AudioRingBuffer;
    private config: VADConfig;
    private isRunning = false;

    // VAD State
    private h: ort.Tensor | null = null;
    private c: ort.Tensor | null = null;
    private frameBuffer: Float32Array;
    private isSpeaking = false;
    private speechStartTimestamp = 0;
    private silenceStartTimestamp = 0;

    constructor(ringBuffer: AudioRingBuffer, config: VADConfig) {
        this.ringBuffer = ringBuffer;
        this.config = config;
        this.frameBuffer = new Float32Array(config.frameSize);
    }

    async init(): Promise<void> {
        try {
            this.session = await ort.InferenceSession.create(this.config.modelUrl, {
                executionProviders: ['webgpu', 'wasm'],
            });

            // Initialize states (2, 1, 64) for Silero V4
            const stateShape = [2, 1, 64];
            const zeroState = new Float32Array(2 * 1 * 64).fill(0);
            this.h = new ort.Tensor('float32', zeroState, stateShape);
            this.c = new ort.Tensor('float32', zeroState, stateShape);

            console.log('[VADService] Initialized Silero VAD');
        } catch (e) {
            console.error('[VADService] Failed to initialize:', e);
            throw e;
        }
    }

    start(onEvent: (event: VADEvent) => void): void {
        if (this.isRunning) return;
        this.isRunning = true;
        this.processLoop(onEvent);
    }

    stop(): void {
        this.isRunning = false;
    }

    private async processLoop(onEvent: (event: VADEvent) => void) {
        while (this.isRunning) {
            // Check if enough data is available
            if (this.ringBuffer.availableRead() >= this.config.frameSize) {
                // Read frame
                this.ringBuffer.read(this.frameBuffer);

                // Run Inference
                const probability = await this.runInference(this.frameBuffer);

                // Logic
                this.handleProbability(probability, onEvent);
            } else {
                // Wait bit to avoid hot loop
                await new Promise(r => setTimeout(r, 10));
            }
        }
    }

    private async runInference(frame: Float32Array): Promise<number> {
        if (!this.session || !this.h || !this.c) return 0;

        // Silero VAD Inputs: input (1, N), sr (1), h (2,1,64), c (2,1,64)
        const input = new ort.Tensor('float32', frame, [1, frame.length]);
        const sr = new ort.Tensor('int64', BigInt(this.config.sampleRate), [1]);

        const feeds: Record<string, ort.Tensor> = {
            input: input,
            sr: sr,
            h: this.h,
            c: this.c,
        };

        const results = await this.session.run(feeds);

        // Update states
        this.h = results.hn;
        this.c = results.cn;

        // Output probability
        const output = results.output; // shape (1, 1)
        return output.data[0] as number;
    }

    private handleProbability(prob: number, onEvent: (event: VADEvent) => void) {
        const now = Date.now();

        if (prob >= this.config.threshold) {
            if (!this.isSpeaking) {
                this.isSpeaking = true;
                this.speechStartTimestamp = now;
                onEvent({ type: 'speech_start', timestamp: now });
            }
            this.silenceStartTimestamp = 0;
        } else {
            if (this.isSpeaking) {
                if (this.silenceStartTimestamp === 0) {
                    this.silenceStartTimestamp = now;
                } else if (now - this.silenceStartTimestamp >= this.config.minSilenceDurationMs) {
                    this.isSpeaking = false;
                    onEvent({ type: 'speech_end', timestamp: now });
                }
            }
        }
    }
}
