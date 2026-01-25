/**
 * Auralux Capture AudioWorklet Processor (JS Compiled)
 * Matches RingBuffer logic from nucleus/auralux/ring-buffer.ts
 */

class AuraluxProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.indices = null;
    this.data = null;
    this.capacity = 0;
    this.mask = 0;
    this.initialized = false;

    this.port.onmessage = (event) => {
      if (event.data.type === 'INIT') {
        const sab = event.data.sharedBuffer;
        
        // Match RingBufferWriter layout:
        // Indices: 2 x Int32 (Read @ 0, Write @ 1)
        // Data: Rest as Float32
        this.indices = new Int32Array(sab, 0, 2);
        this.data = new Float32Array(sab, 8); // 8 bytes offset
        this.capacity = this.data.length;
        this.mask = this.capacity - 1;
        
        this.initialized = true;
        // console.log('[AuraluxWorklet] Initialized. Capacity:', this.capacity);
      }
    };
  }

  process(inputs, outputs, parameters) {
    // 1. Check State
    if (!this.initialized || !this.indices || !this.data) return true;

    // 2. Get Input
    const input = inputs[0];
    const channel = input[0]; // Mono
    if (!channel || channel.length === 0) return true;

    // 3. Calculate Available Space
    // Read atomic indices
    const readIdx = Atomics.load(this.indices, 0); // READ_INDEX_OFFSET = 0
    const writeIdx = Atomics.load(this.indices, 1); // WRITE_INDEX_OFFSET = 1
    
    // Space = Capacity - (Write - Read)
    // -1 to keep one slot empty (standard ring buffer convention to distinguish full vs empty)
    const available = this.capacity - (writeIdx - readIdx);
    const toWrite = Math.min(channel.length, available - 1);

    if (toWrite <= 0) {
      // Buffer full or error
      return true;
    }

    // 4. Write Data (Handle Wrap-Around)
    const offset1 = writeIdx & this.mask;
    const firstPart = Math.min(toWrite, this.capacity - offset1);
    const secondPart = toWrite - firstPart;

    // First chunk
    for (let i = 0; i < firstPart; i++) {
      this.data[offset1 + i] = channel[i];
    }

    // Second chunk (wrap)
    if (secondPart > 0) {
      for (let i = 0; i < secondPart; i++) {
        this.data[i] = channel[firstPart + i];
      }
    }

    // 5. Update Pointer & Notify
    Atomics.store(this.indices, 1, writeIdx + toWrite);
    Atomics.notify(this.indices, 1);

    return true;
  }
}

registerProcessor('auralux-processor', AuraluxProcessor);