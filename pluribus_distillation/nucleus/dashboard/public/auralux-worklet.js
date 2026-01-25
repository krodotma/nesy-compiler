/**
 * Auralux Capture AudioWorklet Processor
 * Generated from capture-worklet.ts
 */

const RING_BUFFER_SIZE = 8192;
const RENDER_QUANTUM = 128;

const STATE_CAPTURE_WRITE = 0;
const STATE_CAPTURE_READ = 1;
const STATE_VAD = 2;
const STATE_AUDIO_LEVEL = 3;

class AuraluxCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const opts = options.processorOptions;
    this.ringBuffer = new Float32Array(opts.ringBuffer);
    this.stateArray = new Int32Array(opts.stateBuffer);
    this.ringSize = this.ringBuffer.length;
    this.chunkSize = RENDER_QUANTUM;
    Atomics.store(this.stateArray, STATE_CAPTURE_WRITE, 0);
  }

  process(inputs, _outputs, _parameters) {
    const input = inputs[0]?.[0];
    if (!input || input.length === 0) {
      return true;
    }
    const writeIdx = Atomics.load(this.stateArray, STATE_CAPTURE_WRITE);
    const offset = (writeIdx * this.chunkSize) % this.ringSize;
    for (let i = 0; i < this.chunkSize && i < input.length; i++) {
      this.ringBuffer[offset + i] = input[i];
    }
    let sumSquares = 0;
    for (let i = 0; i < input.length; i += 4) {
      sumSquares += input[i] * input[i];
    }
    const rms = Math.sqrt(sumSquares / (input.length / 4));
    const levelInt = Math.min(1000, Math.floor(rms * 10000));
    Atomics.store(this.stateArray, STATE_AUDIO_LEVEL, levelInt);
    const nextIdx = (writeIdx + 1) % (this.ringSize / this.chunkSize);
    Atomics.store(this.stateArray, STATE_CAPTURE_WRITE, nextIdx);
    Atomics.notify(this.stateArray, STATE_CAPTURE_WRITE);
    return true;
  }
}

registerProcessor('auralux-capture', AuraluxCaptureProcessor);
