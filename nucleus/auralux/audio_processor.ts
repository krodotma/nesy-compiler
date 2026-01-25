
import { AudioRingBuffer } from './ring_buffer';

class AuraluxProcessor extends AudioWorkletProcessor {
    private ringBuffer: AudioRingBuffer | null = null;
    private initialized = false;

    constructor() {
        super();
        this.port.onmessage = this.handleMessage.bind(this);
    }

    private handleMessage(event: MessageEvent) {
        if (event.data.type === 'INIT') {
            const { sharedBuffer } = event.data;
            this.ringBuffer = new AudioRingBuffer(sharedBuffer);
            this.initialized = true;
            console.log('[AuraluxProcessor] Initialized with SharedArrayBuffer');
        }
    }

    process(inputs: Float32Array[][], outputs: Float32Array[][], parameters: Record<string, Float32Array>): boolean {
        // We only care about the first input, first channel for now (mono VAD)
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const inputChannel = input[0];

        // Write to ring buffer if initialized
        if (this.initialized && this.ringBuffer) {
            const written = this.ringBuffer.write(inputChannel);
            if (written < inputChannel.length) {
                // Buffer overrun - this indicates the VAD worker is too slow
                // For now, we just drop the frames (overwrite behavior is implicit in ring buffer if we ignored write pointer, 
                // but here our safe implementation returns written count. 
                // We drop the rest.)
                // In a future robust version we might want to emit a warning via port
            }
        }

        // passthrough logic if needed (e.g. for monitoring), otherwise silence
        // For Auralux pipeline, we might want to mute the output to avoid feedback
        // so we don't copy input to output by default unless requested.

        return true;
    }
}

registerProcessor('auralux-processor', AuraluxProcessor);
