
/**
 * AudioMixer
 * 
 * Enhanced streaming audio output buffer with overflow protection
 * and seamless chunk concatenation for real-time playback.
 */
export class AudioMixer {
    private buffer: Float32Array;
    private writePos: number = 0;
    private readPos: number = 0;
    private sampleRate: number;
    private capacity: number;

    // Telemetry
    private overflowCount: number = 0;
    private underflowCount: number = 0;

    /**
     * @param sampleRate Output sample rate (e.g., 24000)
     * @param bufferDurationMs Buffer duration in milliseconds (default 2000ms)
     */
    constructor(sampleRate: number, bufferDurationMs: number = 2000) {
        this.sampleRate = sampleRate;
        this.capacity = Math.floor((sampleRate * bufferDurationMs) / 1000);
        this.buffer = new Float32Array(this.capacity);
    }

    /**
     * Enqueue synthesized audio chunk into circular buffer.
     * @param audio Float32Array of audio samples
     */
    enqueue(audio: Float32Array): void {
        for (let i = 0; i < audio.length; i++) {
            const nextWritePos = (this.writePos + 1) % this.capacity;

            // Overflow protection: don't overwrite unread data
            if (nextWritePos === this.readPos) {
                this.overflowCount++;
                console.warn('[AudioMixer] Buffer overflow, dropping samples');
                return;
            }

            this.buffer[this.writePos] = audio[i];
            this.writePos = nextWritePos;
        }
    }

    /**
     * Read audio from buffer for playback.
     * @param maxSamples Maximum samples to read
     * @returns Float32Array of available samples, or empty array if underflow
     */
    read(maxSamples: number): Float32Array {
        const available = this.availableSamples();

        if (available === 0) {
            this.underflowCount++;
            return new Float32Array(0);
        }

        const toRead = Math.min(maxSamples, available);
        const result = new Float32Array(toRead);

        for (let i = 0; i < toRead; i++) {
            result[i] = this.buffer[this.readPos];
            this.readPos = (this.readPos + 1) % this.capacity;
        }

        return result;
    }

    /**
     * Get number of samples available for reading.
     */
    availableSamples(): number {
        if (this.writePos >= this.readPos) {
            return this.writePos - this.readPos;
        }
        return this.capacity - this.readPos + this.writePos;
    }

    /**
     * Get buffer fullness as percentage.
     */
    get bufferLevel(): number {
        return (this.availableSamples() / this.capacity) * 100;
    }

    /**
     * Get telemetry stats.
     */
    getStats(): { overflowCount: number; underflowCount: number; bufferLevel: number } {
        return {
            overflowCount: this.overflowCount,
            underflowCount: this.underflowCount,
            bufferLevel: this.bufferLevel,
        };
    }

    /**
     * Clear the buffer.
     */
    clear(): void {
        this.writePos = 0;
        this.readPos = 0;
        this.buffer.fill(0);
    }
}

