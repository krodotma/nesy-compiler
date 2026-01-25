import {
  component$,
  useContext,
  useContextProvider,
  useStore,
  useVisibleTask$,
  createContextId,
  Slot,
  $,
  noSerialize,
  type NoSerialize
} from '@builder.io/qwik';
// We need to import the class. Qwik runs this in the browser, so standard import works.
// However, if the class uses non-serializable things (AudioContext), we must wrap it in noSerialize.
import { PipelineOrchestrator, type PipelineConfig } from '../../../../auralux/pipeline_orchestrator';

// Define State
export interface VoiceState {
  isRunning: boolean;
  isReady: boolean;
  isConsoleOpen: boolean;
  vadState: 'speech_start' | 'speech_end' | 'silence';
  error: string | null;
  metrics: { ssl: any; vocoder: any; mixer: any } | null;
  // The orchestrator instance (non-serializable)
  orchestrator: NoSerialize<PipelineOrchestrator> | undefined;
}

// Context ID
export const VoiceContextId = createContextId<VoiceState>('pluribus.auralux.voice');

// Default Config
const DEFAULT_CONFIG: PipelineConfig = {
  vadModelUrl: '/models/silero_vad_v5.onnx',
  sslModelUrl: '/models/hubert-soft-quantized.onnx',
  vocoderModelUrl: '/models/vocos_q8.onnx',
  workletUrl: '/worklets/capture-worklet.js',
};

// Provider Component
export const VoiceProvider = component$(() => {
  const state = useStore<VoiceState>({
    isRunning: false,
    isReady: false,
    isConsoleOpen: false,
    vadState: 'silence',
    error: null,
    metrics: null,
    orchestrator: undefined,
  });

  // Provide Context
  useContextProvider(VoiceContextId, state);

  // Initialize Orchestrator on Client
  useVisibleTask$(({ cleanup }) => {
    try {
      const orch = new PipelineOrchestrator(DEFAULT_CONFIG);
      state.orchestrator = noSerialize(orch);
      state.isReady = true;
      console.log('[VoiceProvider] Orchestrator Initialized');
    } catch (e) {
      console.error('[VoiceProvider] Init Failed:', e);
      state.error = e instanceof Error ? e.message : String(e);
    }

    cleanup(() => {
      if (state.orchestrator) {
        state.orchestrator.stop();
      }
    });
  });

  return <Slot />;
});

// Hook (Qwik Style - essentially a helper to get context and define actions)
export function useVoice() {
  const state = useContext(VoiceContextId);

  const start = $(async () => {
    if (!state.orchestrator) return;
    try {
      // We need to bind the callbacks.
      // Since QRLs are async and we are inside a Qwik event handler, 
      // we need to be careful about passing QRLs to the class if it's not expecting them.
      // The class expects direct functions.
      // But we are in the browser (implied by state.orchestrator existing), 
      // so we can pass regular functions if we are careful.
      
      // However, `state` proxy updates inside the callback will trigger re-renders.
      await state.orchestrator.start(
        (event) => {
          state.vadState = event.type === 'speech_start' ? 'speech_start' : 'speech_end';
        },
        (features) => {
          // Optional: Update metrics or visualization buffer
        }
      );
      state.isRunning = true;
      state.error = null;
    } catch (e) {
      console.error('[VoiceProvider] Start Failed:', e);
      state.error = e instanceof Error ? e.message : "Start failed";
      state.isRunning = false;
    }
  });

  const stop = $(async () => {
    if (!state.orchestrator) return;
    await state.orchestrator.stop();
    state.isRunning = false;
    state.vadState = 'silence';
  });

  const toggleConsole = $(() => {
    state.isConsoleOpen = !state.isConsoleOpen;
  });

  return { state, start, stop, toggleConsole };
}
