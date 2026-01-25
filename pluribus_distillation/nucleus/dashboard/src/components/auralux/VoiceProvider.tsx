import {
  component$,
  useStore,
  useContextProvider,
  createContextId,
  useVisibleTask$,
  $,
  Slot,
  noSerialize,
  type NoSerialize,
} from '@builder.io/qwik';
import { PipelineOrchestrator } from '../../lib/auralux/pipeline_orchestrator';
import type { VADEvent } from '../../lib/auralux/vad_service';

export type AuraluxMode = 'neural' | 'browser' | 'sherpa' | 'loading' | 'error';
export type AuraluxBackend = 'browser-speech' | 'silero-vad' | 'sherpa-onnx';

// Browser SpeechRecognition type
type SpeechRecognitionType = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  abort(): void;
  onresult: ((event: any) => void) | null;
  onerror: ((event: any) => void) | null;
  onend: (() => void) | null;
  onspeechstart: (() => void) | null;
  onspeechend: (() => void) | null;
  onaudiostart: (() => void) | null;
  onaudioend: (() => void) | null;
};

// Audio analyzer for waveform
type AudioAnalyzerType = {
  analyser: AnalyserNode;
  dataArray: Uint8Array<ArrayBuffer>;
  source: MediaStreamAudioSourceNode;
  stream: MediaStream;
  context: AudioContext;
};

export interface VoiceStats {
  totalSpeechMs: number;
  transcriptCount: number;
  avgConfidence: number;
  sessionStart: number;
  lastActivityMs: number;
}

export interface VoiceState {
  status: 'init' | 'loading' | 'ready' | 'error' | 'active';
  isListening: boolean;
  isProcessing: boolean;
  isSpeaking: boolean;
  isTTSSpeaking: boolean;
  vadConfidence: number;
  error?: string;

  // Backend/Mode
  auraluxMode: AuraluxMode;
  activeBackend: AuraluxBackend;
  availableBackends: AuraluxBackend[];
  auraluxLatency: number;

  // Voice controls
  pitch: number;
  rate: number;
  volume: number;
  language: string;

  // Transcripts
  lastTranscript: string;
  interimTranscript: string;
  transcriptHistory: string[];

  // Audio visualization (0-255 levels)
  waveformData: number[];
  audioLevel: number;

  // Statistics
  stats: VoiceStats;

  // Keyboard shortcut enabled
  keyboardEnabled: boolean;

  // Internal (noSerialize)
  recognition: NoSerialize<SpeechRecognitionType>;
  audioAnalyzer: NoSerialize<AudioAnalyzerType>;
  ttsUtterance: NoSerialize<SpeechSynthesisUtterance>;
  pipeline: NoSerialize<PipelineOrchestrator>;
}

export const VoiceContext = createContextId<VoiceState>('pluribus.voice.v2');

// Available TTS voices cache
const voiceCache = { voices: [] as SpeechSynthesisVoice[] };

export const VoiceProvider = component$(() => {
  const state = useStore<VoiceState>({
    status: 'init',
    isListening: false,
    isProcessing: false,
    isSpeaking: false,
    isTTSSpeaking: false,
    vadConfidence: 0,
    auraluxMode: 'loading',
    activeBackend: 'browser-speech',
    availableBackends: ['browser-speech'],
    auraluxLatency: 0,
    pitch: 1.0,
    rate: 1.0,
    volume: 1.0,
    language: 'en-US',
    lastTranscript: '',
    interimTranscript: '',
    transcriptHistory: [],
    waveformData: new Array(32).fill(0),
    audioLevel: 0,
    stats: {
      totalSpeechMs: 0,
      transcriptCount: 0,
      avgConfidence: 0,
      sessionStart: Date.now(),
      lastActivityMs: 0,
    },
    keyboardEnabled: true,
    recognition: undefined,
    audioAnalyzer: undefined,
    ttsUtterance: undefined,
    pipeline: undefined,
  });

  // Client-side initialization
  useVisibleTask$(async ({ cleanup }) => {
    state.status = 'loading';
    const backends: AuraluxBackend[] = [];

    // Check for Web Speech API support
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

    if (SpeechRecognition) {
      backends.push('browser-speech');

      try {
        const recognition = new SpeechRecognition() as SpeechRecognitionType;
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = state.language;

        let speechStartTime = 0;
        let confidenceSum = 0;
        let confidenceCount = 0;

        recognition.onspeechstart = () => {
          speechStartTime = performance.now();
          state.isSpeaking = true;
          state.vadConfidence = 0.8;
          state.stats.lastActivityMs = Date.now();
        };

        recognition.onspeechend = () => {
          const duration = performance.now() - speechStartTime;
          state.stats.totalSpeechMs += duration;
          state.isSpeaking = false;
          state.vadConfidence = 0;
        };

        recognition.onaudiostart = () => {
          state.isProcessing = true;
        };

        recognition.onaudioend = () => {
          state.isProcessing = false;
        };

        recognition.onresult = (event: any) => {
          const results = event.results;
          if (results.length > 0) {
            const latest = results[results.length - 1];
            const transcript = latest[0].transcript;
            const confidence = latest[0].confidence || 0.5;

            if (latest.isFinal) {
              state.lastTranscript = transcript;
              state.interimTranscript = '';
              state.transcriptHistory = [...state.transcriptHistory.slice(-9), transcript];
              state.stats.transcriptCount++;

              confidenceSum += confidence;
              confidenceCount++;
              state.stats.avgConfidence = confidenceSum / confidenceCount;
            } else {
              state.interimTranscript = transcript;
            }

            state.vadConfidence = confidence;
            state.isSpeaking = !latest.isFinal;
            state.stats.lastActivityMs = Date.now();

            if (typeof window !== 'undefined') {
              window.dispatchEvent(new CustomEvent('pluribus:voice', {
                detail: {
                  topic: 'auralux.transcript',
                  data: { transcript, confidence, isFinal: latest.isFinal }
                }
              }));
            }
          }
        };

        recognition.onerror = (event: any) => {
          if (state.activeBackend !== 'browser-speech') return; // Ignore if not active

          if (event.error === 'not-allowed') {
            state.error = 'Microphone access denied';
          } else if (event.error !== 'no-speech' && event.error !== 'aborted') {
            state.error = `Speech error: ${event.error}`;
          }
          state.vadConfidence = 0;
        };

        recognition.onend = () => {
          if (state.isListening && state.activeBackend === 'browser-speech' && state.recognition) {
            try {
              setTimeout(() => {
                if (state.isListening && state.activeBackend === 'browser-speech' && state.recognition) {
                  state.recognition.start();
                }
              }, 100);
            } catch (e) {
              console.error('[VoiceProvider] Failed to restart:', e);
            }
          }
        };

        state.recognition = noSerialize(recognition);
      } catch (e) {
        console.error('[VoiceProvider] Failed to init recognition:', e);
      }
    }

    // Check for Auralux Neural support (ONNX models exist)
    // We assume yes if we are here, but can't strictly check FS from browser.
    // We trust the files are in /models/.
    backends.push('silero-vad');

    // Load TTS voices
    if (typeof speechSynthesis !== 'undefined') {
      const loadVoices = () => {
        voiceCache.voices = speechSynthesis.getVoices();
      };
      loadVoices();
      speechSynthesis.onvoiceschanged = loadVoices;
    }

    state.availableBackends = backends;
    state.activeBackend = 'browser-speech'; // Default to browser for safety
    state.status = backends.length > 0 ? 'ready' : 'error';
    state.auraluxMode = 'browser';

    if (backends.length === 0) {
      state.error = 'No speech backends available';
    }

    console.log('[VoiceProvider v2] Initialized with backends:', backends);

    const handleKeyDown = (e: KeyboardEvent) => {
      if (!state.keyboardEnabled) return;
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;

      if (e.key === 'v' || e.key === 'V') {
        if (e.shiftKey) {
          window.dispatchEvent(new CustomEvent('pluribus:voice:settings-toggle'));
        } else {
          if (state.isListening) {
            stopListeningInternal(state);
          } else {
            startListeningInternal(state);
          }
        }
        e.preventDefault();
      }

      if (e.key === 'Escape' && state.isListening) {
        stopListeningInternal(state);
      }
    };

    window.addEventListener('keydown', handleKeyDown);

    cleanup(() => {
      window.removeEventListener('keydown', handleKeyDown);
      if (state.audioAnalyzer) {
        state.audioAnalyzer.context.close();
        state.audioAnalyzer.stream.getTracks().forEach(t => t.stop());
      }
      if (state.pipeline) {
        state.pipeline.stop();
      }
    });
  });

  useContextProvider(VoiceContext, state);

  return <Slot />;
});

// Internal start function
async function startListeningInternal(state: VoiceState) {
  try {
    state.error = undefined;
    state.stats.sessionStart = Date.now();

    // Neural Mode (Auralux Pipeline)
    if (state.activeBackend === 'silero-vad') {
      if (!state.pipeline) {
        console.log('[VoiceProvider] Initializing Auralux Pipeline...');
        const orchestrator = new PipelineOrchestrator({
            vadModelUrl: '/models/silero_vad_v5.onnx',
            sslModelUrl: '/models/hubert-soft-quantized.onnx',
            vocoderModelUrl: '/models/vocos_q8.onnx',
            workletUrl: '/auralux-worklet.js'
        });
        state.pipeline = noSerialize(orchestrator);
      }

      if (state.pipeline) {
        await state.pipeline.start((event: VADEvent) => {
            // Handle VAD Events
            if (event.type === 'speech_start') {
                state.isSpeaking = true;
                state.vadConfidence = 1.0;
                state.stats.lastActivityMs = Date.now();
            } else if (event.type === 'speech_end') {
                state.isSpeaking = false;
                state.vadConfidence = 0.2;
                // Accumulate speech duration approximation? 
                // Currently handled by VAD logic internally usually
            } else if (event.type === 'silence') {
                state.vadConfidence = Math.max(0, state.vadConfidence - 0.05);
            }
        });
        state.isListening = true;
        state.status = 'active';
        state.isProcessing = true; // Pipeline is always processing audio
      }

    // Browser Mode
    } else if (state.activeBackend === 'browser-speech') {
      if (!state.recognition) {
        throw new Error('Speech recognition not initialized');
      }
      
      // Start audio analyzer for waveform visualization
      if (!state.audioAnalyzer && typeof navigator !== 'undefined' && navigator.mediaDevices) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const context = new AudioContext();
          const source = context.createMediaStreamSource(stream);
          const analyser = context.createAnalyser();
          analyser.fftSize = 64;
          analyser.smoothingTimeConstant = 0.8;
          source.connect(analyser);

          const dataArray = new Uint8Array(analyser.frequencyBinCount);
          state.audioAnalyzer = noSerialize({ analyser, dataArray, source, stream, context });

          const updateWaveform = () => {
            if (state.audioAnalyzer && state.isListening) {
              state.audioAnalyzer.analyser.getByteFrequencyData(state.audioAnalyzer.dataArray);
              const data = Array.from(state.audioAnalyzer.dataArray);
              state.waveformData = data;
              state.audioLevel = data.reduce((a, b) => a + b, 0) / data.length;
              requestAnimationFrame(updateWaveform);
            }
          };
          updateWaveform();
        } catch (e) {
          console.warn('[VoiceProvider] Audio analyzer failed:', e);
        }
      }
      
      state.recognition.start();
      state.isListening = true;
      state.status = 'active';
    }

    // Emit event
    if (typeof window !== 'undefined') {
        window.dispatchEvent(new CustomEvent('pluribus:voice', {
        detail: { topic: 'auralux.listening.start', backend: state.activeBackend }
        }));
    }

  } catch (e: any) {
    if (!e.message?.includes('already started')) {
      console.error('[VoiceProvider] Start failed:', e);
      state.error = `Failed to start: ${e.message}`;
      state.status = 'error';
    }
  }
}

// Internal stop function
async function stopListeningInternal(state: VoiceState) {
  // Stop Browser
  if (state.recognition) {
    try {
      state.recognition.stop();
    } catch (e) {
      // ignore
    }
  }

  // Stop Neural
  if (state.pipeline) {
      try {
          await state.pipeline.stop();
          // We don't nullify pipeline, we keep it warm for next toggle? 
          // Or we nullify to save memory. Let's keep it for now.
      } catch (e) {
          console.error('[VoiceProvider] Pipeline stop error:', e);
      }
  }

  state.isListening = false;
  state.isSpeaking = false;
  state.isProcessing = false;
  state.vadConfidence = 0;
  state.status = 'ready';
  state.waveformData = new Array(32).fill(0);
  state.audioLevel = 0;

  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent('pluribus:voice', {
        detail: { topic: 'auralux.listening.stop' }
    }));
  }
}

// Exported actions
export const startListening = $(async (state: VoiceState) => {
  await startListeningInternal(state);
});

export const stopListening = $(async (state: VoiceState) => {
  await stopListeningInternal(state);
});

export const toggleMode = $((state: VoiceState, mode: AuraluxMode) => {
  // Stop if listening
  if (state.isListening) {
      stopListeningInternal(state); // Fire and forget (it's async but we want immediate UI update)
  }

  state.auraluxMode = mode;
  // Map mode to backend
  if (mode === 'browser') state.activeBackend = 'browser-speech';
  else if (mode === 'neural') state.activeBackend = 'silero-vad';
  else if (mode === 'sherpa') state.activeBackend = 'sherpa-onnx';
});

export const setPitch = $((state: VoiceState, pitch: number) => {
  state.pitch = Math.max(0.5, Math.min(2, pitch));
});

export const setRate = $((state: VoiceState, rate: number) => {
  state.rate = Math.max(0.5, Math.min(2, rate));
});

export const setVolume = $((state: VoiceState, volume: number) => {
  state.volume = Math.max(0, Math.min(1, volume));
});

export const setLanguage = $((state: VoiceState, lang: string) => {
  state.language = lang;
  if (state.recognition) {
    state.recognition.lang = lang;
  }
});

export const toggleKeyboard = $((state: VoiceState, enabled: boolean) => {
  state.keyboardEnabled = enabled;
});

// TTS Speak function
export const speak = $((state: VoiceState, text: string) => {
  if (typeof speechSynthesis === 'undefined') {
    state.error = 'TTS not supported';
    return;
  }

  speechSynthesis.cancel();

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.pitch = state.pitch;
  utterance.rate = state.rate;
  utterance.volume = state.volume;
  utterance.lang = state.language;

  const voices = voiceCache.voices.length > 0 ? voiceCache.voices : speechSynthesis.getVoices();
  const preferredVoice = voices.find(v => v.lang.startsWith(state.language.split('-')[0]) && v.localService);
  if (preferredVoice) utterance.voice = preferredVoice;

  utterance.onstart = () => {
    state.isTTSSpeaking = true;
  };

  utterance.onend = () => {
    state.isTTSSpeaking = false;
  };

  utterance.onerror = () => {
    state.isTTSSpeaking = false;
  };

  state.ttsUtterance = noSerialize(utterance);
  speechSynthesis.speak(utterance);

  window.dispatchEvent(new CustomEvent('pluribus:voice', {
    detail: { topic: 'auralux.tts.speak', data: { text } }
  }));
});

export const stopSpeaking = $((state: VoiceState) => {
  if (typeof speechSynthesis !== 'undefined') {
    speechSynthesis.cancel();
  }
  state.isTTSSpeaking = false;
});

export const clearTranscripts = $((state: VoiceState) => {
  state.transcriptHistory = [];
  state.lastTranscript = '';
  state.interimTranscript = '';
});