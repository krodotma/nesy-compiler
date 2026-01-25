/**
 * Auralux Inference Worker
 *
 * Runs VAD and audio processing in a dedicated Worker thread.
 * Communicates with main thread via SharedArrayBuffer ring buffers.
 */

import { RingBufferReader } from './ring-buffer';
import { VadProcessor } from './VadProcessor';

// Worker message types
interface InitMessage {
  type: 'init';
  captureRing: SharedArrayBuffer;
  stateBuffer: SharedArrayBuffer;
  config: WorkerConfig;
}

interface StartMessage {
  type: 'start';
}

interface StopMessage {
  type: 'stop';
}

interface ConfigMessage {
  type: 'config';
  config: Partial<WorkerConfig>;
}

type WorkerMessage = InitMessage | StartMessage | StopMessage | ConfigMessage;

interface WorkerConfig {
  sampleRate: number;
  vadThreshold: number;
  chunkSizeMs: number;
  modelUrl?: string;
}

// State buffer indices
const STATE_VAD = 2;

// Worker state
let running = false;
let captureReader: RingBufferReader | null = null;
let stateArray: Int32Array | null = null;
let processor: VadProcessor | null = null;
let config: WorkerConfig = {
  sampleRate: 48000,
  vadThreshold: 0.5,
  chunkSizeMs: 32,
};

/**
 * Main processing loop
 */
async function processLoop(): Promise<void> {
  const chunkSamples = 512; // VadProcessor expects 512 (32ms @ 16kHz)

  while (running) {
    if (!captureReader || !stateArray || !processor) {
      await sleep(10);
      continue;
    }

    // Wait for data
    const hasData = await captureReader.waitForData(50);
    if (!hasData || !running) continue;

    // Read audio chunk (VAD expects 16kHz)
    // If input is 48kHz, we need 1536 samples to get 512 @ 16kHz
    const inputSamplesNeeded = Math.floor(chunkSamples * (config.sampleRate / 16000));
    const available = captureReader.availableRead();
    
    if (available < inputSamplesNeeded) continue;

    const audio = captureReader.read(inputSamplesNeeded);

    // Resample to 16kHz
    const audio16k = config.sampleRate === 16000 ? audio : resampleTo16k(audio, config.sampleRate);

    // Run VAD via VadProcessor
    await processor.processFrame(audio16k);
  }
}

/**
 * Simple resampling (linear interpolation)
 */
function resampleTo16k(audio: Float32Array, fromRate: number): Float32Array {
  const ratio = 16000 / fromRate;
  const newLength = 512;
  const result = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const srcIdx = i / ratio;
    const srcIdxFloor = Math.floor(srcIdx);
    const srcIdxCeil = Math.min(srcIdxFloor + 1, audio.length - 1);
    const t = srcIdx - srcIdxFloor;
    result[i] = audio[srcIdxFloor] * (1 - t) + audio[srcIdxCeil] * t;
  }

  return result;
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Message handler
 */
self.onmessage = async (event: MessageEvent<WorkerMessage>) => {
  const msg = event.data;

  switch (msg.type) {
    case 'init':
      captureReader = new RingBufferReader(msg.captureRing);
      stateArray = new Int32Array(msg.stateBuffer);
      config = { ...config, ...msg.config };

      // Initialize VadProcessor
      processor = new VadProcessor(captureReader, self as any);
      await processor.init(msg.config.modelUrl || '/auralux/models/silero_vad_v5.onnx');

      self.postMessage({ type: 'ready' });
      break;

    case 'start':
      running = true;
      processLoop();
      self.postMessage({ type: 'started' });
      break;

    case 'stop':
      running = false;
      self.postMessage({ type: 'stopped' });
      break;

    case 'config':
      config = { ...config, ...msg.config };
      break;
  }
};

export type { WorkerMessage, WorkerConfig };