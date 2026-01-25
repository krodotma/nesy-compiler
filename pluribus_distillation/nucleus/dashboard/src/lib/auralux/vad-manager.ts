import { VadProcessor } from './VadProcessor';

/**
 * Auralux VAD Manager (Consolidated)
 *
 * Wraps @ricky0123/vad-web for browser-native Voice Activity Detection.
 * Uses Silero VAD model via ONNX Runtime Web.
 * 
 * Also serves as the interface for VadProcessor when running in main thread (if desired).
 */

// Types for @ricky0123/vad-web
interface MicVADOptions {
  onSpeechStart?: () => void;
  onSpeechEnd?: (audio: Float32Array) => void;
  onVADMisfire?: () => void;
  positiveSpeechThreshold?: number;
  negativeSpeechThreshold?: number;
  redemptionFrames?: number;
  frameSamples?: number;
  preSpeechPadFrames?: number;
  minSpeechFrames?: number;
}

interface MicVAD {
  start: () => void;
  pause: () => void;
  destroy: () => void;
  listening: boolean;
}

declare const vad: {
  MicVAD: {
    new: (options: MicVADOptions) => Promise<MicVAD>;
  };
};

export interface VADEvent {
  type: 'speech_start' | 'speech_end' | 'misfire';
  timestamp: number;
  audio?: Float32Array;
  duration?: number;
}

export interface VADManagerOptions {
  /** Speech detection threshold (0-1, default 0.5) */
  positiveSpeechThreshold?: number;
  /** End-of-speech threshold (0-1, default 0.35) */
  negativeSpeechThreshold?: number;
  /** Frames to confirm end of speech (default 8) */
  redemptionFrames?: number;
  /** Minimum speech frames to trigger (default 3) */
  minSpeechFrames?: number;
  /** Callback for VAD events */
  onEvent?: (event: VADEvent) => void;
  /** Callback for bus emission */
  emitBus?: (topic: string, data: Record<string, unknown>) => void;
}

export class VADManager {
  private micVAD: MicVAD | null = null;
  private options: VADManagerOptions;
  private speechStartTime: number = 0;
  private isInitialized: boolean = false;
  private processor: VadProcessor | null = null;
  private audioContext: AudioContext | null = null;

  constructor(options: VADManagerOptions = {}) {
    this.options = {
      positiveSpeechThreshold: 0.5,
      negativeSpeechThreshold: 0.35,
      redemptionFrames: 8,
      minSpeechFrames: 3,
      ...options,
    };
    // Initialize AudioContext if needed for VadProcessor
    if (typeof window !== 'undefined') {
        this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
  }

  static isAvailable(): boolean {
    return typeof vad !== 'undefined' && typeof vad.MicVAD !== 'undefined';
  }

  static async loadFromCDN(): Promise<void> {
    if (VADManager.isAvailable()) return;
    await loadScript('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.min.js');
    await loadScript('https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.29/dist/bundle.min.js');
    await new Promise(resolve => setTimeout(resolve, 100));
    if (!VADManager.isAvailable()) {
      throw new Error('Failed to load VAD library from CDN');
    }
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    if (!VADManager.isAvailable()) {
      await VADManager.loadFromCDN();
    }

    this.micVAD = await vad.MicVAD.new({
      positiveSpeechThreshold: this.options.positiveSpeechThreshold,
      negativeSpeechThreshold: this.options.negativeSpeechThreshold,
      redemptionFrames: this.options.redemptionFrames,
      minSpeechFrames: this.options.minSpeechFrames,

      onSpeechStart: () => {
        this.speechStartTime = performance.now();
        const event: VADEvent = { type: 'speech_start', timestamp: Date.now() };
        this.options.onEvent?.(event);
        this.options.emitBus?.('auralux.vad.speech_start', {
          confidence: this.options.positiveSpeechThreshold,
          timestamp: new Date().toISOString(),
        });
      },

      onSpeechEnd: (audio: Float32Array) => {
        const duration = performance.now() - this.speechStartTime;
        const event: VADEvent = { type: 'speech_end', timestamp: Date.now(), audio, duration };
        this.options.onEvent?.(event);
        this.options.emitBus?.('auralux.vad.speech_end', {
          duration_ms: Math.round(duration),
          samples: audio.length,
          timestamp: new Date().toISOString(),
        });
      },

      onVADMisfire: () => {
        const event: VADEvent = { type: 'misfire', timestamp: Date.now() };
        this.options.onEvent?.(event);
      },
    });

    this.isInitialized = true;
    this.options.emitBus?.('auralux.vad.initialized', {
      threshold: this.options.positiveSpeechThreshold,
      timestamp: new Date().toISOString(),
    });
  }

  start(): void {
    if (!this.micVAD) {
      throw new Error('VAD not initialized. Call initialize() first.');
    }
    this.micVAD.start();
    this.audioContext?.resume();
    this.options.emitBus?.('auralux.vad.started', {
      timestamp: new Date().toISOString(),
    });
  }

  pause(): void {
    this.micVAD?.pause();
    this.options.emitBus?.('auralux.vad.paused', {
      timestamp: new Date().toISOString(),
    });
  }

  get isListening(): boolean {
    return this.micVAD?.listening ?? false;
  }

  destroy(): void {
    this.micVAD?.destroy();
    this.micVAD = null;
    this.isInitialized = false;
    this.options.emitBus?.('auralux.vad.destroyed', {
      timestamp: new Date().toISOString(),
    });
  }
}

export class SimpleVAD {
  private analyser: AnalyserNode | null = null;
  private audioContext: AudioContext | null = null;
  private dataArray: Uint8Array | null = null;
  private threshold: number;
  private isSpeaking: boolean = false;
  private silenceFrames: number = 0;
  private readonly silenceThreshold: number = 10;

  constructor(threshold: number = 30) {
    this.threshold = threshold;
  }

  async initialize(stream: MediaStream): Promise<void> {
    this.audioContext = new AudioContext();
    const source = this.audioContext.createMediaStreamSource(stream);
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 256;
    this.analyser.smoothingTimeConstant = 0.3;
    source.connect(this.analyser);
    this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
  }

  detect(): { isSpeech: boolean; level: number; stateChanged: boolean } {
    if (!this.analyser || !this.dataArray) {
      return { isSpeech: false, level: 0, stateChanged: false };
    }
    this.analyser.getByteFrequencyData(this.dataArray);
    let sum = 0;
    for (let i = 0; i < this.dataArray.length; i++) {
      sum += this.dataArray[i];
    }
    const level = sum / this.dataArray.length;
    const wasSpeaking = this.isSpeaking;
    if (level > this.threshold) {
      this.isSpeaking = true;
      this.silenceFrames = 0;
    } else {
      this.silenceFrames++;
      if (this.silenceFrames > this.silenceThreshold) {
        this.isSpeaking = false;
      }
    }
    return { isSpeech: this.isSpeaking, level, stateChanged: wasSpeaking !== this.isSpeaking };
  }

  destroy(): void {
    this.audioContext?.close();
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
  }
}

function loadScript(src: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
}