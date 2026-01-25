/**
 * LazyVoiceOverlay - lazy load the voice HUD/provider to avoid
 * pulling in onnxruntime-web on initial dashboard load.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { useVoice } from '../lib/auralux/use-voice';

export const LazyVoiceOverlay = component$(() => {
  const { state, toggleConsole } = useVoice();
  const ConsoleRef = useSignal<any>(null);

  useVisibleTask$(({ track }) => {
    track(() => state.isConsoleOpen);
    if (state.isConsoleOpen && !ConsoleRef.value) {
      import('./views/AuraluxConsole').then((module) => {
        ConsoleRef.value = module.AuraluxConsole;
      });
    }
  });

  if (!state.isConsoleOpen) return null;

  const AuraluxConsole = ConsoleRef.value;

  return (
    <div 
      class="fixed inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-md p-8 animate-in fade-in duration-200" 
      onClick$={(e, el) => { 
        // Close on backdrop click
        if (e.target === el) toggleConsole(); 
      }}
    >
      <div class="w-full max-w-6xl h-[85vh] bg-background/90 border border-white/10 rounded-2xl shadow-2xl overflow-hidden relative flex flex-col">
        <button 
          onClick$={toggleConsole}
          class="absolute top-4 right-4 z-10 p-2 hover:bg-white/10 rounded-full text-white/50 hover:text-white transition-colors"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
        </button>
        {AuraluxConsole ? <AuraluxConsole /> : (
            <div class="flex items-center justify-center h-full text-cyan-400 font-mono tracking-widest animate-pulse">
                INITIALIZING_NEURAL_INTERFACE...
            </div>
        )}
      </div>
    </div>
  );
});
