/**
 * Auralux Audio Preprocessor (Phase 2 Optimization)
 * 
 * Handles resampling and buffering between Input (44.1/48kHz) and SSL (16kHz).
 * Uses a polyphase FIR filter approach or simple linear interpolation depending on quality budget.
 */
export class AudioPreprocessor {
    private targetRate = 16000;
    private internalBuffer: Float32Array = new Float32Array(0);

    constructor() {}

    /**
     * Resamples audio frames to 16kHz for HuBERT
     * Using simple linear interpolation for low-latency browser performance
     */
    resample(input: Float32Array, inputRate: number): Float32Array {
        if (inputRate === this.targetRate) return input;

        const ratio = this.targetRate / inputRate;
        const newLength = Math.floor(input.length * ratio);
        const result = new Float32Array(newLength);

        for (let i = 0; i < newLength; i++) {
            const srcIdx = i / ratio;
            const srcIdxFloor = Math.floor(srcIdx);
            const srcIdxCeil = Math.min(srcIdxFloor + 1, input.length - 1);
            const t = srcIdx - srcIdxFloor;
            result[i] = input[srcIdxFloor] * (1 - t) + input[srcIdxCeil] * t;
        }

        return result;
    }

    /**
     * Batches small VAD chunks into larger SSL windows
     */
    batchFrames(input: Float32Array, requiredSize: number = 320): Float32Array[] {
        // Concatenate new input with existing buffer
        const combined = new Float32Array(this.internalBuffer.length + input.length);
        combined.set(this.internalBuffer);
        combined.set(input, this.internalBuffer.length);

        const batches: Float32Array[] = [];
        let offset = 0;

        while (offset + requiredSize <= combined.length) {
            batches.push(combined.slice(offset, offset + requiredSize));
            offset += requiredSize;
        }

        // Store remainder in internal buffer
        this.internalBuffer = combined.slice(offset);

        return batches;
    }

    /**
     * Clear the internal buffer
     */
    reset() {
        this.internalBuffer = new Float32Array(0);
    }
}