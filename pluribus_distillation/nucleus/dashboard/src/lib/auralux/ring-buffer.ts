/**
 * Wait-Free SPSC Ring Buffer for Auralux
 *
 * Single-Producer Single-Consumer lock-free ring buffer using SharedArrayBuffer.
 * Based on ringbuf.js pattern by Paul Adenot (Mozilla).
 *
 * @see https://github.com/padenot/ringbuf.js/
 * @see https://blog.paul.cx/post/a-wait-free-spsc-ringbuffer-for-the-web/
 */

/**
 * Storage layout in SharedArrayBuffer:
 * - First 8 bytes: read/write indices (2 x Int32)
 * - Remaining bytes: actual ring buffer data (Float32Array)
 */
const INDEX_BYTES = 8; // 2 x Int32
const READ_INDEX_OFFSET = 0;
const WRITE_INDEX_OFFSET = 1;

export interface RingBufferOptions {
  /** Capacity in samples (must be power of 2) */
  capacity: number;
  /** Chunk size for atomic operations */
  chunkSize?: number;
}

/**
 * Create a SharedArrayBuffer for the ring buffer
 */
export function createRingBufferStorage(capacity: number): SharedArrayBuffer {
  // Ensure capacity is a power of 2
  const pow2Capacity = Math.pow(2, Math.ceil(Math.log2(capacity)));
  
  const dataBytes = pow2Capacity * Float32Array.BYTES_PER_ELEMENT;
  return new SharedArrayBuffer(INDEX_BYTES + dataBytes);
}

/**
 * Ring buffer writer (producer side)
 * Use this in AudioWorklet or main thread
 */
export class RingBufferWriter {
  private indices: Int32Array;
  private data: Float32Array;
  private capacity: number;
  private mask: number;

  constructor(sab: SharedArrayBuffer) {
    this.indices = new Int32Array(sab, 0, 2);
    this.data = new Float32Array(sab, INDEX_BYTES);
    this.capacity = this.data.length;
    this.mask = this.capacity - 1; // For fast modulo (capacity must be power of 2)
  }

  /**
   * Get available space for writing
   */
  availableWrite(): number {
    const read = Atomics.load(this.indices, READ_INDEX_OFFSET);
    const write = Atomics.load(this.indices, WRITE_INDEX_OFFSET);
    return this.capacity - (write - read);
  }

  /**
   * Write samples to ring buffer
   * @returns Number of samples actually written
   */
  write(samples: Float32Array): number {
    const available = this.availableWrite();
    const toWrite = Math.min(samples.length, available - 1); // Leave 1 slot empty

    if (toWrite <= 0) return 0;

    const writeIdx = Atomics.load(this.indices, WRITE_INDEX_OFFSET);

    // Write in up to 2 chunks (wrap-around)
    const firstPart = Math.min(toWrite, this.capacity - (writeIdx & this.mask));
    const secondPart = toWrite - firstPart;

    // First chunk
    const offset1 = writeIdx & this.mask;
    for (let i = 0; i < firstPart; i++) {
      this.data[offset1 + i] = samples[i];
    }

    // Second chunk (if wrap-around)
    if (secondPart > 0) {
      for (let i = 0; i < secondPart; i++) {
        this.data[i] = samples[firstPart + i];
      }
    }

    // Update write index (atomic)
    Atomics.store(this.indices, WRITE_INDEX_OFFSET, writeIdx + toWrite);

    return toWrite;
  }

  /**
   * Notify waiting readers
   */
  notify(): void {
    Atomics.notify(this.indices, WRITE_INDEX_OFFSET);
  }
}

/**
 * Ring buffer reader (consumer side)
 * Use this in Worker thread
 */
export class RingBufferReader {
  private indices: Int32Array;
  private data: Float32Array;
  private capacity: number;
  private mask: number;

  constructor(sab: SharedArrayBuffer) {
    this.indices = new Int32Array(sab, 0, 2);
    this.data = new Float32Array(sab, INDEX_BYTES);
    this.capacity = this.data.length;
    this.mask = this.capacity - 1;
  }

  /**
   * Get available samples for reading
   */
  availableRead(): number {
    const read = Atomics.load(this.indices, READ_INDEX_OFFSET);
    const write = Atomics.load(this.indices, WRITE_INDEX_OFFSET);
    return write - read;
  }

  /**
   * Read samples from ring buffer
   * @returns Float32Array with read samples (may be shorter than requested)
   */
  read(count: number): Float32Array {
    const available = this.availableRead();
    const toRead = Math.min(count, available);

    if (toRead <= 0) return new Float32Array(0);

    const result = new Float32Array(toRead);
    const readIdx = Atomics.load(this.indices, READ_INDEX_OFFSET);

    // Read in up to 2 chunks (wrap-around)
    const firstPart = Math.min(toRead, this.capacity - (readIdx & this.mask));
    const secondPart = toRead - firstPart;

    // First chunk
    const offset1 = readIdx & this.mask;
    for (let i = 0; i < firstPart; i++) {
      result[i] = this.data[offset1 + i];
    }

    // Second chunk (if wrap-around)
    if (secondPart > 0) {
      for (let i = 0; i < secondPart; i++) {
        result[firstPart + i] = this.data[i];
      }
    }

    // Update read index (atomic)
    Atomics.store(this.indices, READ_INDEX_OFFSET, readIdx + toRead);

    return result;
  }

  /**
   * Peek at samples without consuming them
   */
  peek(count: number): Float32Array {
    const available = this.availableRead();
    const toPeek = Math.min(count, available);

    if (toPeek <= 0) return new Float32Array(0);

    const result = new Float32Array(toPeek);
    const readIdx = Atomics.load(this.indices, READ_INDEX_OFFSET);

    const firstPart = Math.min(toPeek, this.capacity - (readIdx & this.mask));
    const secondPart = toPeek - firstPart;

    const offset1 = readIdx & this.mask;
    for (let i = 0; i < firstPart; i++) {
      result[i] = this.data[offset1 + i];
    }

    if (secondPart > 0) {
      for (let i = 0; i < secondPart; i++) {
        result[firstPart + i] = this.data[i];
      }
    }

    return result;
  }

  /**
   * Wait for data to be available (use in Worker only)
   * @returns Promise that resolves when data is available or timeout
   */
  async waitForData(timeoutMs: number = 100): Promise<boolean> {
    const currentWrite = Atomics.load(this.indices, WRITE_INDEX_OFFSET);
    const currentRead = Atomics.load(this.indices, READ_INDEX_OFFSET);

    if (currentWrite > currentRead) {
      return true; // Data already available
    }

    // Use Atomics.waitAsync for non-blocking wait in worker
    const result = Atomics.waitAsync(this.indices, WRITE_INDEX_OFFSET, currentWrite, timeoutMs);

    if (result.async) {
      const waitResult = await result.value;
      return waitResult === 'ok';
    }

    return result.value === 'not-equal'; // Data arrived before wait
  }
}

/**
 * Create state buffer for inter-thread communication
 * Layout: [captureWrite, captureRead, vadState, audioLevel, outputWrite, outputRead, ...]
 */
export function createStateBuffer(slots: number = 16): SharedArrayBuffer {
  return new SharedArrayBuffer(slots * Int32Array.BYTES_PER_ELEMENT);
}

/**
 * Utility: Check if SharedArrayBuffer is available
 */
export function isSharedArrayBufferAvailable(): boolean {
  return typeof SharedArrayBuffer !== 'undefined';
}

/**
 * Utility: Check required headers for SharedArrayBuffer
 */
export function checkCrossOriginIsolation(): { isolated: boolean; headers: string[] } {
  const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
  const missingHeaders: string[] = [];

  if (!isolated) {
    missingHeaders.push('Cross-Origin-Opener-Policy: same-origin');
    missingHeaders.push('Cross-Origin-Embedder-Policy: require-corp');
  }

  return { isolated, headers: missingHeaders };
}
