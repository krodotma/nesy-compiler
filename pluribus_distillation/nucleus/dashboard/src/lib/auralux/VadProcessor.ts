import { env, InferenceSession, Tensor } from 'onnxruntime-web';
import { RingBufferReader } from './ring-buffer';

/**
 * Auralux VAD Processor
 * Implements Silero VAD v4 with sliding window buffer.
 */
export class VadProcessor {
    private session: InferenceSession | null = null;
    private readonly sampleRate = 16000;
    private readonly frameSize = 512; // 32ms
    private readonly threshold = 0.5;

    private hn: Tensor;
    private cn: Tensor;
    private isSpeechActive = false;

    constructor(
        private inputRing: RingBufferReader,
        private outputPort: MessagePort
    ) {
        this.hn = new Tensor('float32', new Float32Array(2 * 1 * 64), [2, 1, 64]);
        this.cn = new Tensor('float32', new Float32Array(2 * 1 * 64), [2, 1, 64]);
    }

    async init(modelUrl: string) {
        env.wasm.numThreads = 1;
        env.wasm.simd = true;

        this.session = await InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        console.log('Auralux VAD initialized (WASM-SIMD)');
    }

    async processFrame(audioFrame: Float32Array) {
        if (!this.session) return;

        const inputTensor = new Tensor('float32', audioFrame, [1, this.frameSize]);
        const srTensor = new Tensor('int64', BigInt64Array.from([BigInt(this.sampleRate)]));

        const feeds = { input: inputTensor, sr: srTensor, h: this.hn, c: this.cn };

        const results = await this.session.run(feeds);
        this.hn = results.hn;
        this.cn = results.cn;

        const probability = results.output.data[0] as number;

        // Hysteresis trigger
        if (probability > this.threshold && !this.isSpeechActive) {
            this.isSpeechActive = true;
            this.outputPort.postMessage({
                type: 'vad_state',
                state: 'speech',
                probability,
                ts: performance.now()
            });
        } else if (probability < this.threshold - 0.15 && this.isSpeechActive) {
            this.isSpeechActive = false;
            this.outputPort.postMessage({
                type: 'vad_state',
                state: 'silence',
                probability,
                ts: performance.now()
            });
        }
    }
}
