/**
 * DialogosShell.tsx
 * Author: gemini_ui_1
 * Context: Phase 1 Physics (Growable Shell)
 */

import { 
  component$, 
  useSignal, 
  useStore, 
  useVisibleTask$, 
  Slot, 
  $ 
} from '@builder.io/qwik';
import { isBrowser } from '@builder.io/qwik/build';

// Spring Physics Constants (Opus Architect spec)
const SPRING_STIFFNESS = 170;
const SPRING_DAMPING = 26;

type ViewState = 'rest' | 'active' | 'full';

export const DialogosShell = component$(() => {
  // State
  const viewState = useSignal<ViewState>('rest');
  const position = useStore({ x: 0, y: 0 }); // For drag physics
  const dimensions = useStore({ w: 600, h: 64 }); // Target dimensions
  
  // Refs
  const shellRef = useSignal<HTMLDivElement>();

  // Physics Engine (Gemini Optimizer)
  useVisibleTask$(({ track }) => {
    track(() => viewState.value);
    if (!isBrowser) return;

    const targetW = viewState.value === 'rest' ? 600 : viewState.value === 'active' ? 700 : window.innerWidth * 0.9;
    const targetH = viewState.value === 'rest' ? 64 : viewState.value === 'active' ? 500 : window.innerHeight * 0.9;
    
    // Simple FLIP-like transition for now, will upgrade to full Spring later
    // Using CSS transitions for "Good Enough" mvp, but planned for Spring hook.
    if (shellRef.value) {
      shellRef.value.style.width = `${targetW}px`;
      shellRef.value.style.height = `${targetH}px`;
      
      // Dynamic positioning
      if (viewState.value === 'rest') {
        shellRef.value.style.bottom = '2rem';
        shellRef.value.style.top = 'auto';
        shellRef.value.style.borderRadius = '32px';
      } else {
        shellRef.value.style.bottom = '50%';
        shellRef.value.style.top = 'auto';
        shellRef.value.style.transform = 'translate(-50%, 50%)'; // Center
        shellRef.value.style.borderRadius = '16px';
      }
    }
  });

  const toggleState = $(() => {
    if (viewState.value === 'rest') viewState.value = 'active';
    else if (viewState.value === 'active') viewState.value = 'rest'; // Toggle back for now
  });

  return (
    <div
      ref={shellRef}
      class="fixed left-1/2 -translate-x-1/2 z-[9000] 
             glass-panel glass-panel-glow
             transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)]
             flex flex-col overflow-hidden"
      style={{
        boxShadow: '0 20px 80px rgba(0,0,0,0.5)',
        border: '1px solid var(--glass-border-bright)'
      }}
    >
      {/* Liquid Background Layer */}
      <div class="absolute inset-0 opacity-20 pointer-events-none bg-gradient-to-br from-[var(--glass-accent-cyan)] to-[var(--glass-accent-magenta)] animate-gradient-shift" />

      {/* Main Content Area */}
      <div class="relative flex-1 flex flex-col z-10">
        
        {/* Header / Drag Handle */}
        <div 
          class="h-4 w-full cursor-grab active:cursor-grabbing flex justify-center items-center opacity-0 hover:opacity-100 transition-opacity"
          onClick$={toggleState}
        >
          <div class="w-12 h-1 rounded-full bg-white/20" />
        </div>

        {/* Input / Interaction Area */}
        <div class="flex-1 p-4 overflow-hidden">
          <Slot /> 
        </div>

        {/* Footer (Rest State UI) */}
        {viewState.value === 'rest' && (
          <div 
            class="absolute inset-0 flex items-center px-6 cursor-text"
            onClick$={toggleState}
          >
            <span class="text-xl mr-4 animate-pulse">✨</span>
            <span class="text-lg text-white/50 font-light">Ask Pluribus anything...</span>
            <div class="ml-auto flex gap-2">
               <div class="px-2 py-1 rounded bg-white/10 text-xs text-white/40 font-mono">Cmd+K</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
