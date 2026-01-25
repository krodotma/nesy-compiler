import { component$, useSignal, useVisibleTask$, useStore, $ } from '@builder.io/qwik';
import { useVoice } from '../../lib/auralux/use-voice';
import { AuraluxModeIndicator } from '../auralux/AuraluxModeIndicator';
import { VadConfidenceOrb } from '../auralux/VadConfidenceOrb';
import { VadEventLog } from '../auralux/VadEventLog';
import { VoiceQualitySelector } from '../auralux/VoiceQualitySelector';

// Auralux uses Gentle Waves shader for calm voice ambiance
const VOICE_SHADER_ID = 'art-gentle_waves';
const VOICE_SHADER_NAME = 'Gentle Waves';
const VOICE_SHADER_PATH = 'nucleus/art_dept/collection/gentle_waves.glsl';

export const AuraluxConsole = component$(() => {
  const { state, start, stop } = useVoice();
  
  // Local UI state for visualization that isn't in global context
  const pitch = useSignal(1.0);
  const rate = useSignal(1.0);

  const statusLabel = state.error
    ? 'ERROR'
    : state.isRunning
      ? 'CAPTURING'
      : state.isReady
        ? 'READY'
        : 'LOADING';

  const statusClass = statusLabel === 'ERROR'
    ? 'bg-red-500/20 text-red-400'
    : statusLabel === 'CAPTURING'
      ? 'bg-red-500/20 text-red-400 animate-pulse'
      : statusLabel === 'READY'
        ? 'bg-green-500/20 text-green-400'
        : 'bg-amber-500/20 text-amber-400';

  // Inject Gentle Waves shader when entering voice section
  useVisibleTask$(() => {
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
          window.dispatchEvent(new CustomEvent('pluribus:mood:change', {
            detail: { mood: 'calm', source: 'AuraluxConsole' }
          }));
        }
      } catch {
        // Non-critical
      }
    })();
  });

  const toggleMic$ = $(async () => {
    if (state.isRunning) {
      await stop();
    } else {
      await start();
    }
  });

  const runTts$ = $(async () => {
    // TTS not fully wired in useVoice yet, placeholder
    console.log('TTS triggered');
  });

  return (
    <div class="h-full flex flex-col gap-6 p-6 glass-panel text-foreground">
      <div class="flex items-center justify-between gap-4">
        <div>
          <h2 class="text-2xl font-bold">AURALUX Voice Console</h2>
          <p class="text-muted-foreground">High-performance streaming voice pipeline</p>
        </div>
        <div class="flex items-center gap-3">
          <AuraluxModeIndicator mode="neural" latencyMs={0} />
          <div class={`px-4 py-2 rounded-full text-sm font-bold ${statusClass}`}>
            {statusLabel}
          </div>
        </div>
      </div>

      <div class="grid grid-cols-3 gap-4">
        <div class="p-4 rounded-lg bg-card border border-border">
          <div class="text-xs text-muted-foreground uppercase">Latency</div>
          <div class="text-2xl font-mono">--</div>
        </div>
        <div class="p-4 rounded-lg bg-card border border-border">
          <div class="text-xs text-muted-foreground uppercase">VAD</div>
          <div class="text-2xl font-mono">
            {state.vadState === 'speech_start' || state.vadState === 'speech_end' ? 'Active' : 'Silence'}
          </div>
        </div>
        <div class="p-4 rounded-lg bg-card border border-border">
          <div class="text-xs text-muted-foreground uppercase">Models</div>
          <div class="text-2xl font-mono">3/3</div>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-[1.2fr_1fr] gap-4">
        <div class="flex flex-col gap-4">
          <div class="flex items-center gap-6 rounded-lg bg-black/40 border border-border p-4">
            <VadConfidenceOrb
              confidence={state.vadState === 'speech_start' ? 1.0 : 0.0}
              isListening={state.isRunning}
              isSpeaking={state.vadState === 'speech_start'}
            />
            <div class="flex-1 min-w-0">
              <div class="text-xs text-muted-foreground uppercase">Listening State</div>
              <div class="text-lg font-mono">
                {state.isRunning ? 'ACTIVE' : 'STANDBY'}
              </div>
              <div class="text-[10px] text-muted-foreground mt-1">
                {state.isReady ? 'Neural engine online' : 'Initializing...'}
              </div>
            </div>
          </div>
          <VadEventLog events={[]} /> 
        </div>

        <div class="flex flex-col gap-4">
          <VoiceQualitySelector
            mode="neural"
            latencyMs={0}
            pitch={pitch.value}
            rate={rate.value}
            onModeChange$={() => {}}
            onPitchChange$={(value) => { pitch.value = value; }}
            onRateChange$={(value) => { rate.value = value; }}
          />
          {state.error && (
            <div class="rounded-lg border border-red-500/30 bg-red-500/10 text-red-200 text-xs p-3">
              {state.error}
            </div>
          )}
        </div>
      </div>

      <div class="flex gap-4">
        <button 
          onClick$={toggleMic$}
          disabled={!state.isReady}
          class={`flex-1 py-4 rounded-xl font-bold transition-all ${state.isRunning ? 'bg-red-600 hover:bg-red-500' : 'bg-green-600 hover:bg-green-500'} ${!state.isReady ? 'opacity-40 cursor-not-allowed' : ''}`}
        >
          {state.isRunning ? 'STOP MIC' : 'START MIC'}
        </button>
        <button
          onClick$={runTts$}
          disabled={true}
          class={`px-8 py-4 rounded-xl bg-primary/20 text-primary border border-primary/30 hover:bg-primary/30 font-bold opacity-50 cursor-not-allowed`}
        >
          TEST TTS
        </button>
      </div>
    </div>
  );
});

export default AuraluxConsole;
