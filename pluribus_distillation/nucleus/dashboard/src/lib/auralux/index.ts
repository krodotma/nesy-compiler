/**
 * Auralux VAD Module
 *
 * Browser-native Voice Activity Detection using:
 * - AudioWorklet for real-time capture
 * - SharedArrayBuffer ring buffers for lock-free IPC
 * - Silero VAD via ONNX Runtime Web
 * - Worker thread for inference
 *
 * Usage:
 * ```typescript
 * import { AuraluxVAD } from '@auralux/vad';
 *
 * const vad = new AuraluxVAD({
 *   onSpeechStart: () => console.log('Speech started'),
 *   onSpeechEnd: (audio) => console.log('Speech ended', audio.length),
 * });
 *
 * await vad.initialize();
 * await vad.start();
 * ```
 */

import {
  createRingBufferStorage,
  createStateBuffer,
  RingBufferWriter,
  RingBufferReader,
  isSharedArrayBufferAvailable,
  checkCrossOriginIsolation,
} from './ring-buffer';

import { VADManager, SimpleVAD, VADEvent } from './vad-manager';

// Re-export types
export { VADEvent } from './vad-manager';
export { RingBufferWriter, RingBufferReader } from './ring-buffer';

// Constants
const RING_BUFFER_CAPACITY = 16384; // ~340ms @ 48kHz
const STATE_BUFFER_SLOTS = 16;

export interface AuraluxVADConfig {
  /** Sample rate (default 48000) */
  sampleRate?: number;
  /** VAD threshold 0-1 (default 0.5) */
  vadThreshold?: number;
  /** Use @ricky0123/vad-web (default true) */
  useRicky0123VAD?: boolean;
  /** Path to Silero VAD ONNX model */
  vadModelUrl?: string;
  /** Callback when speech starts */
  onSpeechStart?: () => void;
  /** Callback when speech ends with audio buffer */
  onSpeechEnd?: (audio: Float32Array) => void;
  /** Callback for VAD state changes */
  onStateChange?: (state: 'idle' | 'speech' | 'silence') => void;
  /** Callback for audio level updates */
  onAudioLevel?: (level: number) => void;
  /** Bus emission callback */
  emitBus?: (topic: string, data: Record<string, unknown>) => void;
}

export type VADState = 'uninitialized' | 'initializing' | 'ready' | 'running' | 'error';

export class AuraluxVAD {
  private config: Required<AuraluxVADConfig>;
  private state: VADState = 'uninitialized';
  private audioContext: AudioContext | null = null;
  private captureWorklet: AudioWorkletNode | null = null;
  private mediaStream: MediaStream | null = null;
  private inferenceWorker: Worker | null = null;

  // Shared buffers
  private captureRing: SharedArrayBuffer | null = null;
  private stateBuffer: SharedArrayBuffer | null = null;

  // Fallback VAD (when SharedArrayBuffer unavailable)
  private vadManager: VADManager | null = null;
  private simpleVAD: SimpleVAD | null = null;

  // Audio level polling
  private levelPollInterval: number | null = null;

  constructor(config: AuraluxVADConfig = {}) {
    this.config = {
      sampleRate: 48000,
      vadThreshold: 0.5,
      useRicky0123VAD: true,
      vadModelUrl: '/auralux/models/silero_vad_v5.onnx',
      onSpeechStart: () => {},
      onSpeechEnd: () => {},
      onStateChange: () => {},
      onAudioLevel: () => {},
      emitBus: () => {},
      ...config,
    };
  }

  /**
   * Get current VAD state
   */
  getState(): VADState {
    return this.state;
  }

  /**
   * Check system capabilities
   */
  static checkCapabilities(): {
    supported: boolean;
    sharedArrayBuffer: boolean;
    audioWorklet: boolean;
    crossOriginIsolated: boolean;
    issues: string[];
  } {
    const issues: string[] = [];

    const sharedArrayBuffer = isSharedArrayBufferAvailable();
    if (!sharedArrayBuffer) {
      issues.push('SharedArrayBuffer not available');
    }

    const audioWorklet = typeof AudioWorkletNode !== 'undefined';
    if (!audioWorklet) {
      issues.push('AudioWorklet not supported');
    }

    const { isolated } = checkCrossOriginIsolation();
    if (!isolated) {
      issues.push('Cross-Origin-Isolation headers missing');
    }

    return {
      supported: issues.length === 0,
      sharedArrayBuffer,
      audioWorklet,
      crossOriginIsolated: isolated,
      issues,
    };
  }

  /**
   * Initialize the VAD system
   */
  async initialize(): Promise<void> {
    if (this.state !== 'uninitialized') {
      throw new Error(`Cannot initialize from state: ${this.state}`);
    }

    this.state = 'initializing';
    this.config.emitBus('auralux.vad.initializing', { timestamp: new Date().toISOString() });

    const caps = AuraluxVAD.checkCapabilities();

    if (!caps.supported) {
      // Fall back to @ricky0123/vad-web or simple VAD
      console.warn('[AuraluxVAD] Using fallback mode:', caps.issues.join(', '));

      if (this.config.useRicky0123VAD) {
        await this.initRicky0123VAD();
      } else {
        await this.initSimpleVAD();
      }
    } else {
      // Full SharedArrayBuffer + Worker mode
      await this.initFullPipeline();
    }

    this.state = 'ready';
    this.config.emitBus('auralux.vad.ready', {
      mode: caps.supported ? 'full' : 'fallback',
      timestamp: new Date().toISOString(),
    });
  }

  /**
   * Initialize full pipeline with SharedArrayBuffer
   */
  private async initFullPipeline(): Promise<void> {
    // Create shared buffers
    this.captureRing = createRingBufferStorage(RING_BUFFER_CAPACITY);
    this.stateBuffer = createStateBuffer(STATE_BUFFER_SLOTS);

    // Create audio context
    this.audioContext = new AudioContext({ sampleRate: this.config.sampleRate });

    // Load AudioWorklet module
    const workletUrl = new URL('./capture-worklet.js', import.meta.url).href;
    await this.audioContext.audioWorklet.addModule(workletUrl);

    // Create capture worklet node
    this.captureWorklet = new AudioWorkletNode(this.audioContext, 'auralux-capture', {
      processorOptions: {
        ringBuffer: this.captureRing,
        stateBuffer: this.stateBuffer,
      },
    });

    // Create inference worker
    const workerUrl = new URL('./inference-worker.js', import.meta.url).href;
    this.inferenceWorker = new Worker(workerUrl, { type: 'module' });

    // Set up worker message handling
    this.inferenceWorker.onmessage = (event) => {
      const msg = event.data;

      if (msg.type === 'vad_state') {
        this.config.onStateChange(msg.state);

        if (msg.state === 'speech') {
          this.config.onSpeechStart();
          this.config.emitBus('auralux.vad.speech_start', {
            probability: msg.probability,
            timestamp: new Date().toISOString(),
          });
        } else if (msg.state === 'silence') {
          // Note: Full audio buffer collection would require more logic
          this.config.onSpeechEnd(new Float32Array(0));
          this.config.emitBus('auralux.vad.speech_end', {
            timestamp: new Date().toISOString(),
          });
        }
      }
    };

    // Initialize worker
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('Worker init timeout')), 10000);

      this.inferenceWorker!.onmessage = (event) => {
        if (event.data.type === 'ready') {
          clearTimeout(timeout);
          resolve();
        }
      };

      this.inferenceWorker!.postMessage({
        type: 'init',
        captureRing: this.captureRing,
        stateBuffer: this.stateBuffer,
        config: {
          sampleRate: this.config.sampleRate,
          vadThreshold: this.config.vadThreshold,
          modelUrl: this.config.vadModelUrl,
        },
      });
    });
  }

  /**
   * Initialize @ricky0123/vad-web fallback
   */
  private async initRicky0123VAD(): Promise<void> {
    this.vadManager = new VADManager({
      positiveSpeechThreshold: this.config.vadThreshold,
      onEvent: (event) => {
        if (event.type === 'speech_start') {
          this.config.onSpeechStart();
          this.config.onStateChange('speech');
        } else if (event.type === 'speech_end') {
          this.config.onSpeechEnd(event.audio || new Float32Array(0));
          this.config.onStateChange('silence');
        }
      },
      emitBus: this.config.emitBus,
    });

    await this.vadManager.initialize();
  }

  /**
   * Initialize simple energy-based VAD fallback
   */
  private async initSimpleVAD(): Promise<void> {
    this.simpleVAD = new SimpleVAD(this.config.vadThreshold * 100);
  }

  /**
   * Start VAD processing
   */
  async start(): Promise<void> {
    if (this.state !== 'ready') {
      throw new Error(`Cannot start from state: ${this.state}`);
    }

    // Get microphone access
    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: this.config.sampleRate },
        channelCount: { ideal: 1 },
        echoCancellation: { ideal: true },
        noiseSuppression: { ideal: false },
        autoGainControl: { ideal: false },
      },
    });

    if (this.vadManager) {
      // @ricky0123/vad-web mode
      this.vadManager.start();
    } else if (this.simpleVAD) {
      // Simple VAD mode
      await this.simpleVAD.initialize(this.mediaStream);
      this.startSimpleVADLoop();
    } else if (this.audioContext && this.captureWorklet) {
      // Full pipeline mode
      const source = this.audioContext.createMediaStreamSource(this.mediaStream);
      source.connect(this.captureWorklet);

      this.inferenceWorker?.postMessage({ type: 'start' });

      // Start audio level polling
      this.startLevelPolling();
    }

    this.state = 'running';
    this.config.emitBus('auralux.vad.started', { timestamp: new Date().toISOString() });
  }

  /**
   * Stop VAD processing
   */
  stop(): void {
    if (this.vadManager) {
      this.vadManager.pause();
    }

    if (this.simpleVAD) {
      this.simpleVAD.destroy();
    }

    if (this.inferenceWorker) {
      this.inferenceWorker.postMessage({ type: 'stop' });
    }

    if (this.levelPollInterval) {
      clearInterval(this.levelPollInterval);
      this.levelPollInterval = null;
    }

    // Stop media stream
    this.mediaStream?.getTracks().forEach(track => track.stop());
    this.mediaStream = null;

    this.state = 'ready';
    this.config.emitBus('auralux.vad.stopped', { timestamp: new Date().toISOString() });
  }

  /**
   * Destroy and release all resources
   */
  destroy(): void {
    this.stop();

    this.vadManager?.destroy();
    this.vadManager = null;

    this.inferenceWorker?.terminate();
    this.inferenceWorker = null;

    this.captureWorklet?.disconnect();
    this.captureWorklet = null;

    this.audioContext?.close();
    this.audioContext = null;

    this.state = 'uninitialized';
    this.config.emitBus('auralux.vad.destroyed', { timestamp: new Date().toISOString() });
  }

  /**
   * Start audio level polling from state buffer
   */
  private startLevelPolling(): void {
    if (!this.stateBuffer) return;

    const stateArray = new Int32Array(this.stateBuffer);
    const STATE_AUDIO_LEVEL = 3;

    this.levelPollInterval = window.setInterval(() => {
      const level = Atomics.load(stateArray, STATE_AUDIO_LEVEL) / 1000;
      this.config.onAudioLevel(level);
    }, 50); // 20Hz update rate
  }

  /**
   * Simple VAD polling loop
   */
  private startSimpleVADLoop(): void {
    let lastState: 'speech' | 'silence' | null = null;

    const poll = () => {
      if (this.state !== 'running' || !this.simpleVAD) return;

      const { isSpeech, level, stateChanged } = this.simpleVAD.detect();
      this.config.onAudioLevel(level / 100);

      if (stateChanged) {
        const newState = isSpeech ? 'speech' : 'silence';
        this.config.onStateChange(newState);

        if (newState === 'speech' && lastState !== 'speech') {
          this.config.onSpeechStart();
        } else if (newState === 'silence' && lastState === 'speech') {
          this.config.onSpeechEnd(new Float32Array(0));
        }

        lastState = newState;
      }

      requestAnimationFrame(poll);
    };

    requestAnimationFrame(poll);
  }
}

// Default export
export default AuraluxVAD;
export { useAuraluxPipeline } from './useAuraluxPipeline';
