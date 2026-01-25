
/**
 * AudioRingBuffer
 * 
 * A Single-Producer Single-Consumer (SPSC) lock-free ring buffer for audio data.
 * Uses SharedArrayBuffer and Atomics for thread safety between AudioWorklet and Main/Worker threads.
 */

export interface RingBufferState {
    buffer: SharedArrayBuffer;
    capacity: number;
    readPtr: Int32Array; // Shared atomic pointer
    writePtr: Int32Array; // Shared atomic pointer
    storage: Float32Array; // Shared data storage
}

export class AudioRingBuffer {
    private buffer: SharedArrayBuffer;
    private capacity: number;
    private readPtr: Int32Array;
    private writePtr: Int32Array;
    private storage: Float32Array;

    // Header size: 2 integers (readPtr, writePtr) * 4 bytes
    private static readonly HEADER_BYTES = 8;

    constructor(capacityOrBuffer: number | SharedArrayBuffer) {
        if (typeof capacityOrBuffer === 'number') {
            this.capacity = capacityOrBuffer;
            const dataBytes = this.capacity * 4; // Float32
            this.buffer = new SharedArrayBuffer(AudioRingBuffer.HEADER_BYTES + dataBytes);

            // Initialize pointers
            this.readPtr = new Int32Array(this.buffer, 0, 1);
            this.writePtr = new Int32Array(this.buffer, 4, 1);
            this.storage = new Float32Array(this.buffer, 8, this.capacity);

            Atomics.store(this.readPtr, 0, 0);
            Atomics.store(this.writePtr, 0, 0);
        } else {
            this.buffer = capacityOrBuffer;
            this.readPtr = new Int32Array(this.buffer, 0, 1);
            this.writePtr = new Int32Array(this.buffer, 4, 1);
            this.capacity = (this.buffer.byteLength - AudioRingBuffer.HEADER_BYTES) / 4;
            this.storage = new Float32Array(this.buffer, 8, this.capacity);
        }
    }

    get sharedBuffer(): SharedArrayBuffer {
        return this.buffer;
    }

    /**
     * Write data to the buffer.
     * @returns Number of items written.
     */
    write(data: Float32Array): number {
        const available = this.availableWrite();
        if (available === 0) return 0;

        const toWrite = Math.min(data.length, available);
        const writeIdx = Atomics.load(this.writePtr, 0);

        // Check for wrap-around
        const firstChunk = Math.min(toWrite, this.capacity - writeIdx);
        const secondChunk = toWrite - firstChunk;

        this.storage.set(data.subarray(0, firstChunk), writeIdx);
        if (secondChunk > 0) {
            this.storage.set(data.subarray(firstChunk, toWrite), 0);
        }

        // Update write pointer atomically
        const nextWriteIdx = (writeIdx + toWrite) % this.capacity;
        Atomics.store(this.writePtr, 0, nextWriteIdx);

        return toWrite;
    }

    /**
     * Read data from the buffer.
     * @returns Number of items read.
     */
    read(target: Float32Array): number {
        const available = this.availableRead();
        if (available === 0) return 0;

        const toRead = Math.min(target.length, available);
        const readIdx = Atomics.load(this.readPtr, 0);

        const firstChunk = Math.min(toRead, this.capacity - readIdx);
        const secondChunk = toRead - firstChunk;

        target.set(this.storage.subarray(readIdx, readIdx + firstChunk), 0);
        if (secondChunk > 0) {
            target.set(this.storage.subarray(0, secondChunk), firstChunk);
        }

        // Update read pointer atomically
        const nextReadIdx = (readIdx + toRead) % this.capacity;
        Atomics.store(this.readPtr, 0, nextReadIdx);

        return toRead;
    }

    availableRead(): number {
        const writeIdx = Atomics.load(this.writePtr, 0);
        const readIdx = Atomics.load(this.readPtr, 0);

        if (writeIdx >= readIdx) {
            return writeIdx - readIdx;
        } else {
            return (this.capacity - readIdx) + writeIdx;
        }
    }

    availableWrite(): number {
        // We keep one slot empty to distinguish full from empty
        return this.capacity - this.availableRead() - 1;
    }

    clear(): void {
        Atomics.store(this.readPtr, 0, 0);
        Atomics.store(this.writePtr, 0, 0);
    }
}
