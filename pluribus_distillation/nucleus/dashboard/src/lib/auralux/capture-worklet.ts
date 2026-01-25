/**
 * Auralux Capture AudioWorklet Processor
 *
 * Real-time audio capture with wait-free ring buffer output.
 * CRITICAL: All code in process() must complete in <2.67ms (128 samples @ 48kHz)
 *
 * @see https://developer.chrome.com/blog/audio-worklet-design-pattern
 * @see https://github.com/padenot/ringbuf.js/
 */

// Constants
const RING_BUFFER_SIZE = 8192; // ~170ms @ 48kHz, power of 2 for efficient modulo
const RENDER_QUANTUM = 128;    // Web Audio API render quantum

interface CaptureProcessorOptions {
  ringBuffer: SharedArrayBuffer;
  stateBuffer: SharedArrayBuffer;
}

/**
 * State buffer layout (Int32Array):
 * [0]: captureWriteIndex  - where capture writes next
 * [1]: captureReadIndex   - where consumer reads next
 * [2]: vadState           - 0=inactive, 1=speech, 2=silence
 * [3]: audioLevel         - RMS level (scaled to int)
 */
const STATE_CAPTURE_WRITE = 0;
const STATE_CAPTURE_READ = 1;
const STATE_VAD = 2;
const STATE_AUDIO_LEVEL = 3;

class AuraluxCaptureProcessor extends AudioWorkletProcessor {
  private ringBuffer: Float32Array;
  private stateArray: Int32Array;
  private ringSize: number;
  private chunkSize: number;

  constructor(options: AudioWorkletNodeOptions) {
    super();

    const opts = options.processorOptions as CaptureProcessorOptions;

    // Map shared memory - NO ALLOCATIONS after this point in process()
    this.ringBuffer = new Float32Array(opts.ringBuffer);
    this.stateArray = new Int32Array(opts.stateBuffer);
    this.ringSize = this.ringBuffer.length;
    this.chunkSize = RENDER_QUANTUM;

    // Initialize write index
    Atomics.store(this.stateArray, STATE_CAPTURE_WRITE, 0);
  }

  /**
   * Process audio - MUST complete in <2.67ms
   * NO allocations, NO async, NO blocking calls
   */
  process(
    inputs: Float32Array[][],
    _outputs: Float32Array[][],
    _parameters: Record<string, Float32Array>
  ): boolean {
    const input = inputs[0]?.[0];

    // No input - keep processor alive
    if (!input || input.length === 0) {
      return true;
    }

    // Get current write position (atomic read)
    const writeIdx = Atomics.load(this.stateArray, STATE_CAPTURE_WRITE);

    // Calculate offset in ring buffer
    const offset = (writeIdx * this.chunkSize) % this.ringSize;

    // Copy samples to ring buffer (no allocation - direct write)
    // Manual loop is faster than .set() for small arrays
    for (let i = 0; i < this.chunkSize && i < input.length; i++) {
      this.ringBuffer[offset + i] = input[i];
    }

    // Calculate RMS level for metering (simple approximation)
    let sumSquares = 0;
    for (let i = 0; i < input.length; i += 4) { // Sample every 4th for speed
      sumSquares += input[i] * input[i];
    }
    const rms = Math.sqrt(sumSquares / (input.length / 4));
    const levelInt = Math.min(1000, Math.floor(rms * 10000)); // Scale to int
    Atomics.store(this.stateArray, STATE_AUDIO_LEVEL, levelInt);

    // Update write index (atomic)
    const nextIdx = (writeIdx + 1) % (this.ringSize / this.chunkSize);
    Atomics.store(this.stateArray, STATE_CAPTURE_WRITE, nextIdx);

    // Notify waiting consumers (non-blocking)
    Atomics.notify(this.stateArray, STATE_CAPTURE_WRITE);

    return true;
  }
}

registerProcessor('auralux-capture', AuraluxCaptureProcessor);

export { AuraluxCaptureProcessor, STATE_CAPTURE_WRITE, STATE_CAPTURE_READ, STATE_VAD, STATE_AUDIO_LEVEL };
