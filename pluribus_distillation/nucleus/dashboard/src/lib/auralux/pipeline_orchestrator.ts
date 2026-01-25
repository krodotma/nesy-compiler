
import { RingBufferReader, RingBufferWriter, createRingBufferStorage } from './ring-buffer';
import { VADService, VADEvent } from './vad_service';
import { SSLService } from './ssl_service';
import { VocoderService } from './vocoder_service';
import { AudioMixer } from './audio_mixer';

export interface PipelineConfig {
    vadModelUrl: string;
    sslModelUrl: string;
    vocoderModelUrl: string; // New config
    workletUrl: string;
}

// Wrapper to match previous API expected by existing services
export class AudioRingBuffer {
    public sharedBuffer: SharedArrayBuffer;
    private writer: RingBufferWriter;
    private reader: RingBufferReader;

    constructor(capacity: number) {
        // Enforce power of 2 for capacity
        const pow2Capacity = Math.pow(2, Math.ceil(Math.log2(capacity)));
        this.sharedBuffer = createRingBufferStorage(pow2Capacity);
        this.writer = new RingBufferWriter(this.sharedBuffer);
        this.reader = new RingBufferReader(this.sharedBuffer);
    }

    read(output: Float32Array): number {
        // The reader.read() returns a new Float32Array
        // We need to copy it into the output buffer
        const data = this.reader.read(output.length);
        if (data.length === 0) return 0;
        
        output.set(data);
        return data.length;
    }

    write(input: Float32Array): number {
        return this.writer.write(input);
    }

    clear() {
        // Re-initialize? Or just let GC handle it if we drop references
    }
}

export class PipelineOrchestrator {
    private config: PipelineConfig;
    private audioContext: AudioContext | null = null;
    private mediaStream: MediaStream | null = null;
    private workletNode: AudioWorkletNode | null = null;
    private ringBuffer: AudioRingBuffer | null = null;

    private vadService: VADService | null = null;
    private sslService: SSLService | null = null;
    private vocoderService: VocoderService | null = null;
    private mixer: AudioMixer | null = null;

    // Pipeline State
    private isRunning = false;

    constructor(config: PipelineConfig) {
        this.config = config;
    }

    async start(
        onVadEvent: (event: VADEvent) => void,
        onFeatureEvent?: (features: Float32Array) => void
    ): Promise<void> {
        if (this.isRunning) return;

        try {
            // 1. Initialize Audio Context
            this.audioContext = new AudioContext({ sampleRate: 16000 });
            await this.audioContext.audioWorklet.addModule(this.config.workletUrl);

            // 2. Get User Media
            this.mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);

            // 3. Create Ring Buffer
            this.ringBuffer = new AudioRingBuffer(65536); // Power of 2 (approx 4s @ 16kHz)
            this.mixer = new AudioMixer(24000); // Vocos is 24kHz

            // 4. Create Worklet Node
            this.workletNode = new AudioWorkletNode(this.audioContext, 'auralux-capture');
            this.workletNode.port.postMessage({
                type: 'INIT',
                ringBuffer: this.ringBuffer.sharedBuffer,
                stateBuffer: new SharedArrayBuffer(1024) // Dummy state buffer
            });

            // 5. Connect Graph
            source.connect(this.workletNode);
            this.workletNode.connect(this.audioContext.destination);

            // 6. Initialize Services
            this.vadService = new VADService(this.ringBuffer, {
                modelUrl: this.config.vadModelUrl,
                sampleRate: this.audioContext.sampleRate,
                frameSize: 512,
                threshold: 0.5,
                minSpeechDurationMs: 64,
                minSilenceDurationMs: 300,
            });

            this.sslService = new SSLService({
                modelUrl: this.config.sslModelUrl,
                inputSampleRate: 16000,
                outputDim: 256,
                expectedFrameCount: 1
            });

            this.vocoderService = new VocoderService({
                modelUrl: this.config.vocoderModelUrl,
                sampleRate: 24000,
                hopLength: 256 // Matches HuBERT downsample usually
            });

            await Promise.all([
                this.vadService.init(),
                this.sslService.init(),
                this.vocoderService.init()
            ]);

            // 7. Start Loop
            this.vadService.start(async (event) => {
                onVadEvent(event);

                // Simulating Full Pipeline on Speech Start
                // In real system, this is continuous.
                // Here, we grab a chunk and process it end-to-end to verify flow.

                if (event.type === 'speech_start') {
                    // 1. Read Audio (Hypothetical - manual read for demo)
                    const frameSize = 3200; // 200ms
                    const audioChunk = new Float32Array(frameSize);
                    const read = this.ringBuffer?.read(audioChunk) || 0;

                    if (read > 0) {
                        // 2. Extract Features
                        const features = await this.sslService?.extractFeatures(audioChunk);

                        if (features) {
                            if (onFeatureEvent) onFeatureEvent(features);

                            // 3. Synthesize
                            // Frame count = audio_samples / 320 (HuBERT stride) approx.
                            // Let's assume features.length / 256 = frames.
                            const frames = features.length / 256;
                            const synthesized = await this.vocoderService?.synthesize(features, frames, 256);

                            // 4. Mix / Playback
                            if (synthesized) {
                                this.mixer?.enqueue(synthesized);
                                // Play: Create buffer source immediately for demo
                                this.playAudioChunk(synthesized);
                            }
                        }
                    }
                }
            });

            this.isRunning = true;
            console.log('[Orchestrator] Pipeline Started (Full Loopback)');

        } catch (e) {
            console.error('[Orchestrator] Failed to start pipeline:', e);
            this.stop();
            throw e;
        }
    }

    private playAudioChunk(data: Float32Array) {
        if (!this.audioContext) return;

        // Naive playback: create buffer source
        const buffer = this.audioContext.createBuffer(1, data.length, 24000);
        buffer.copyToChannel(data, 0);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);
        source.start();
    }

    async stop(): Promise<void> {
        this.isRunning = false;

        if (this.vadService) this.vadService.stop();
        if (this.sslService) await this.sslService.dispose();
        if (this.vocoderService) await this.vocoderService.dispose();
        if (this.mixer) this.mixer.clear();

        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        if (this.audioContext) {
            await this.audioContext.close();
            this.audioContext = null;
        }

        if (this.ringBuffer) {
            this.ringBuffer.clear();
        }

        console.log('[Orchestrator] Pipeline Stopped');
    }

    /**
     * Get pipeline performance metrics.
     */
    getMetrics(): { ssl: any; vocoder: any; mixer: any } | null {
        return {
            ssl: this.sslService?.getMetrics() || null,
            vocoder: this.vocoderService?.getMetrics() || null,
            mixer: this.mixer?.getStats() || null,
        };
    }
}
