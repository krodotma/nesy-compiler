import { component$, useSignal, useStore, useVisibleTask$, $, noSerialize, type NoSerialize, type QRL, type Signal } from '@builder.io/qwik';
import type { VPSSession } from '../../lib/state/types';

type SpeechRecognitionCtor = new () => any;

// Voice section uses Gentle Waves shader for calm ambiance
const VOICE_SHADER_ID = 'art-gentle_waves';
const VOICE_SHADER_NAME = 'Gentle Waves';
const VOICE_SHADER_PATH = 'nucleus/art_dept/collection/gentle_waves.glsl';

interface VoiceSpeechViewProps {
  session: Signal<VPSSession>;
  emitBus$: QRL<(topic: string, kind: string, data: Record<string, unknown>) => Promise<void>>;
}

interface VoiceSpeechState {
  sttSupported: boolean;
  ttsSupported: boolean;
  listening: boolean;
  interim: string;
  error: string | null;
  ttsText: string;
  voices: SpeechSynthesisVoice[];
  selectedVoiceUri: string | null;
  rate: number;
  pitch: number;
  volume: number;
}

function safeReqId(prefix: string): string {
  try {
    // eslint-disable-next-line @typescript-eslint/no-unnecessary-condition
    const id = (globalThis as any)?.crypto?.randomUUID?.();
    if (typeof id === 'string') return `${prefix}-${id}`;
  } catch {
    // ignore
  }
  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export const VoiceSpeechView = component$<VoiceSpeechViewProps>(({ session, emitBus$ }) => {
  const state = useStore<VoiceSpeechState>({
    sttSupported: false,
    ttsSupported: false,
    listening: false,
    interim: '',
    error: null,
    ttsText: '',
    voices: [],
    selectedVoiceUri: null,
    rate: 1,
    pitch: 1,
    volume: 1,
  });

  const transcript = useSignal('');
  const recognizer = useSignal<NoSerialize<any> | null>(null);

  const refreshVoices = $(() => {
    try {
      const voices = window.speechSynthesis?.getVoices?.() || [];
      state.voices = voices;
      if (!state.selectedVoiceUri && voices.length > 0) {
        state.selectedVoiceUri = voices[0].voiceURI;
      }
    } catch {
      // ignore
    }
  });

  useVisibleTask$(({ cleanup }) => {
    // Inject Gentle Waves shader for calm voice ambiance
    (async () => {
      try {
        const res = await fetch(`/api/fs/${VOICE_SHADER_PATH}`);
        if (res.ok) {
          const glsl = await res.text();
          const scene = { id: VOICE_SHADER_ID, name: VOICE_SHADER_NAME, glsl };
          (window as any).__PLURIBUS_LAST_ART_SCENE__ = { topic: 'art.scene.change', data: { scene } };
          window.dispatchEvent(new CustomEvent('pluribus:art', {
            detail: { topic: 'art.scene.change', data: { scene, tokens: {} } }
          }));
          // Set calm mood for voice section
          window.dispatchEvent(new CustomEvent('pluribus:mood:change', {
            detail: { mood: 'calm', source: 'VoiceSpeechView' }
          }));
        }
      } catch {
        // Shader injection is non-critical
      }
    })();

    state.sttSupported = false;
    state.ttsSupported = typeof window !== 'undefined' && typeof window.speechSynthesis !== 'undefined';
    refreshVoices();

    // Voices often populate asynchronously.
    const onVoices = () => refreshVoices();
    try {
      window.speechSynthesis?.addEventListener?.('voiceschanged', onVoices);
    } catch {
      // ignore
    }

    // SpeechRecognition is browser-specific (webkit prefix on Chrome).
    const Ctor: SpeechRecognitionCtor | null =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition ||
      null;

    if (Ctor) {
      state.sttSupported = true;
      const sr = new Ctor();
      sr.continuous = true;
      sr.interimResults = true;
      sr.lang = 'en-US';

      sr.onresult = (ev: any) => {
        try {
          let finalChunk = '';
          let interimChunk = '';
          for (let i = ev.resultIndex; i < ev.results.length; i++) {
            const res = ev.results[i];
            const text = res[0]?.transcript || '';
            if (res.isFinal) finalChunk += text;
            else interimChunk += text;
          }
          state.interim = interimChunk.trim();
          if (finalChunk.trim()) {
            transcript.value = (transcript.value + ' ' + finalChunk.trim()).trim();
            state.interim = '';
            void emitBus$(
              'voice.stt.final',
              'metric',
              { req_id: safeReqId('voice-stt'), chars: finalChunk.trim().length }
            );
          }
        } catch (e) {
          state.error = String(e);
        }
      };

      sr.onerror = (ev: any) => {
        state.error = String(ev?.error || 'speech recognition error');
        state.listening = false;
      };

      sr.onend = () => {
        state.listening = false;
      };

      recognizer.value = noSerialize(sr);
    } else {
      state.sttSupported = false;
    }

    cleanup(() => {
      try {
        window.speechSynthesis?.removeEventListener?.('voiceschanged', onVoices);
      } catch {
        // ignore
      }
      try {
        const sr = recognizer.value;
        sr?.stop?.();
      } catch {
        // ignore
      }
    });
  });

  const startListening = $(async () => {
    state.error = null;
    if (!state.sttSupported) {
      state.error = 'SpeechRecognition unsupported in this browser.';
      return;
    }
    const sr = recognizer.value;
    if (!sr) return;
    try {
      sr.start();
      state.listening = true;
      void emitBus$('voice.stt.start', 'metric', { req_id: safeReqId('voice-stt'), iso: new Date().toISOString() });
    } catch (e) {
      state.error = String(e);
      state.listening = false;
    }
  });

  const stopListening = $(async () => {
    const sr = recognizer.value;
    try {
      sr?.stop?.();
    } catch {
      // ignore
    }
    state.listening = false;
    void emitBus$('voice.stt.stop', 'metric', { req_id: safeReqId('voice-stt'), iso: new Date().toISOString() });
  });

  const clearTranscript = $(() => {
    transcript.value = '';
    state.interim = '';
    state.error = null;
  });

  const sendTranscript = $(async () => {
    const prompt = transcript.value.trim();
    if (!prompt) return;

    const providers = (() => {
      const active = session.value.activeFallback;
      if (active) return [active];
      const first = (session.value.fallbackOrder || []).find(p => p !== 'mock');
      return [first || 'chatgpt-web'];
    })();

    const req_id = safeReqId('voice-dialogos');
    await emitBus$('dialogos.submit', 'request', { req_id, mode: 'llm', providers, prompt });
    void emitBus$('voice.dialogos.submit', 'metric', { req_id, providers, chars: prompt.length });
  });

  const speakText = $(async () => {
    state.error = null;
    if (!state.ttsSupported) {
      state.error = 'speechSynthesis unsupported in this browser.';
      return;
    }
    try {
      window.speechSynthesis.cancel();
      const utter = new SpeechSynthesisUtterance(state.ttsText || transcript.value || '');
      utter.rate = state.rate;
      utter.pitch = state.pitch;
      utter.volume = state.volume;
      const voice = state.voices.find(v => v.voiceURI === state.selectedVoiceUri);
      if (voice) utter.voice = voice;
      window.speechSynthesis.speak(utter);
      void emitBus$('voice.tts.speak', 'metric', { req_id: safeReqId('voice-tts'), chars: utter.text.length });
    } catch (e) {
      state.error = String(e);
    }
  });

  const stopSpeaking = $(() => {
    try {
      window.speechSynthesis.cancel();
    } catch {
      // ignore
    }
    void emitBus$('voice.tts.stop', 'metric', { req_id: safeReqId('voice-tts'), iso: new Date().toISOString() });
  });

  return (
    <div class="h-full flex flex-col gap-4">
      <div class="glass-panel p-4">
        <div class="flex items-center justify-between gap-3">
          <div>
            <div class="text-lg font-semibold">Voice & Speech</div>
            <div class="text-xs text-muted-foreground">
              Browser-local STT/TTS. Emits bus evidence and can submit prompts via <code class="mono">dialogos.submit</code>.
            </div>
          </div>
          <div class="flex items-center gap-2 text-xs">
            <span class={`px-2 py-1 rounded border ${state.sttSupported ? 'border-green-500/30 bg-green-500/10 text-green-400' : 'border-amber-500/30 bg-amber-500/10 text-amber-400'}`}>
              STT {state.sttSupported ? 'OK' : 'n/a'}
            </span>
            <span class={`px-2 py-1 rounded border ${state.ttsSupported ? 'border-green-500/30 bg-green-500/10 text-green-400' : 'border-amber-500/30 bg-amber-500/10 text-amber-400'}`}>
              TTS {state.ttsSupported ? 'OK' : 'n/a'}
            </span>
          </div>
        </div>
        {state.error && (
          <div class="mt-3 text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2">
            {state.error}
          </div>
        )}
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-2 gap-4 flex-1 min-h-0">
        {/* STT */}
        <div class="rounded-lg border border-border bg-card flex flex-col min-h-0">
          <div class="p-4 border-b border-border flex items-center justify-between gap-2 flex-shrink-0">
            <div class="flex items-center gap-2">
              <span class="text-xl">🎙️</span>
              <div>
                <div class="text-sm font-semibold">Speech-to-Text</div>
                <div class="text-[10px] text-muted-foreground">Continuous, interim results</div>
              </div>
            </div>
            <div class="flex items-center gap-2">
              {!state.listening ? (
                <button
                  class="text-xs px-3 py-1.5 rounded bg-green-600 hover:bg-green-500 text-white disabled:opacity-50"
                  disabled={!state.sttSupported}
                  onClick$={startListening}
                >
                  Start
                </button>
              ) : (
                <button
                  class="text-xs px-3 py-1.5 rounded bg-red-600 hover:bg-red-500 text-white"
                  onClick$={stopListening}
                >
                  Stop
                </button>
              )}
              <button
                class="text-xs px-3 py-1.5 rounded bg-muted/30 hover:bg-muted/50"
                onClick$={clearTranscript}
              >
                Clear
              </button>
            </div>
          </div>
          <div class="p-4 flex-1 min-h-0 overflow-auto space-y-3">
            {state.interim && (
              <div class="text-xs text-amber-300 bg-amber-500/10 border border-amber-500/20 rounded p-2">
                <span class="font-semibold">Interim:</span> {state.interim}
              </div>
            )}
            <textarea
              class="w-full h-48 lg:h-full min-h-[180px] bg-muted/30 border border-border rounded p-3 text-sm font-mono resize-none"
              value={transcript.value}
              onInput$={(e) => { transcript.value = (e.target as HTMLTextAreaElement).value; }}
              placeholder="Transcript..."
            />
          </div>
          <div class="p-4 border-t border-border flex items-center justify-between gap-2 flex-shrink-0">
            <div class="text-[10px] text-muted-foreground">
              Providers: <span class="mono">{session.value.activeFallback || session.value.fallbackOrder?.find(p => p !== 'mock') || 'chatgpt-web'}</span>
            </div>
            <button
              class="text-xs px-3 py-1.5 rounded bg-primary/20 text-primary hover:bg-primary/30 disabled:opacity-50"
              disabled={!transcript.value.trim()}
              onClick$={sendTranscript}
              title="Submit transcript to dialogosd"
            >
              Send → Dialogos
            </button>
          </div>
        </div>

        {/* TTS */}
        <div class="rounded-lg border border-border bg-card flex flex-col min-h-0">
          <div class="p-4 border-b border-border flex items-center justify-between gap-2 flex-shrink-0">
            <div class="flex items-center gap-2">
              <span class="text-xl">🗣️</span>
              <div>
                <div class="text-sm font-semibold">Text-to-Speech</div>
                <div class="text-[10px] text-muted-foreground">speechSynthesis (local)</div>
              </div>
            </div>
            <div class="flex items-center gap-2">
              <button
                class="text-xs px-3 py-1.5 rounded bg-cyan-600 hover:bg-cyan-500 text-white disabled:opacity-50"
                disabled={!state.ttsSupported || !(state.ttsText || transcript.value).trim()}
                onClick$={speakText}
              >
                Speak
              </button>
              <button
                class="text-xs px-3 py-1.5 rounded bg-muted/30 hover:bg-muted/50 disabled:opacity-50"
                disabled={!state.ttsSupported}
                onClick$={stopSpeaking}
              >
                Stop
              </button>
            </div>
          </div>

          <div class="p-4 flex-1 min-h-0 overflow-auto space-y-3">
            <textarea
              class="w-full h-40 bg-muted/30 border border-border rounded p-3 text-sm resize-none"
              value={state.ttsText}
              onInput$={(e) => { state.ttsText = (e.target as HTMLTextAreaElement).value; }}
              placeholder="Text to speak (defaults to transcript)"
            />

            <div class="grid grid-cols-1 gap-3">
              <label class="text-xs text-muted-foreground">
                Voice
                <select
                  class="mt-1 w-full bg-muted/30 border border-border rounded px-2 py-1 text-xs"
                  value={state.selectedVoiceUri || ''}
                  onChange$={(e) => { state.selectedVoiceUri = (e.target as HTMLSelectElement).value; }}
                >
                  {state.voices.map(v => (
                    <option key={v.voiceURI} value={v.voiceURI}>
                      {`${v.name} (${v.lang})${v.default ? ' • default' : ''}`}
                    </option>
                  ))}
                </select>
              </label>

              <div class="grid grid-cols-3 gap-2">
                <label class="text-xs text-muted-foreground">
                  Rate
                  <input
                    type="range"
                    min="0.5"
                    max="2"
                    step="0.1"
                    value={state.rate}
                    onInput$={(e) => { state.rate = Number((e.target as HTMLInputElement).value); }}
                    class="w-full"
                  />
                </label>
                <label class="text-xs text-muted-foreground">
                  Pitch
                  <input
                    type="range"
                    min="0"
                    max="2"
                    step="0.1"
                    value={state.pitch}
                    onInput$={(e) => { state.pitch = Number((e.target as HTMLInputElement).value); }}
                    class="w-full"
                  />
                </label>
                <label class="text-xs text-muted-foreground">
                  Vol
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={state.volume}
                    onInput$={(e) => { state.volume = Number((e.target as HTMLInputElement).value); }}
                    class="w-full"
                  />
                </label>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

export default VoiceSpeechView;
