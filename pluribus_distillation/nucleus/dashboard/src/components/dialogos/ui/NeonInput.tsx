/**
 * NeonInput.tsx
 * Author: gemini_components_1
 * Context: Phase 2 M3 Atoms
 */

import { component$, useSignal, $ } from '@builder.io/qwik';
import { EntelexisTracker } from '../logic/EntelexisTracker';

interface NeonInputProps {
  value: string;
  onInput$: (val: string) => void;
  onSubmit$: () => void;
  placeholder?: string;
  mode: 'rest' | 'active' | 'full';
  isThinking?: boolean;
}

export const NeonInput = component$<NeonInputProps>(({ value, onInput$, onSubmit$, placeholder, mode, isThinking }) => {
  const inputRef = useSignal<HTMLInputElement>();

  // Determine Visual State
  const visualState = isThinking ? 'actualizing' : value.length > 0 ? 'potential' : 'idle';
  const entState = visualState === 'actualizing' ? 'ENTELEXIS_INFERRING' : 
                   visualState === 'potential' ? 'ENTELEXIS_POTENTIAL_DETECTED' : 'ENTELEXIS_IDLE';
  
  const glowClass = EntelexisTracker.getVisualCue(entState);

  return (
    <div 
      class={`relative w-full group transition-all duration-300 ${glowClass}`}
      style={{
        height: mode === 'rest' ? '100%' : 'auto',
      }}
    >
      {/* Glowing Border / Underline */}
      <div class="absolute bottom-0 left-0 right-0 h-[2px] bg-white/10 group-focus-within:bg-[var(--glass-accent-cyan)] transition-colors duration-300 shadow-[0_0_10px_var(--glass-accent-cyan-subtle)]" />

      <input
        ref={inputRef}
        type="text"
        value={value}
        placeholder={placeholder || "Type to create..."}
        class="w-full bg-transparent border-none outline-none text-white placeholder-white/30 font-light text-lg px-4 py-3"
        onInput$={(e) => onInput$((e.target as HTMLInputElement).value)}
        onKeyDown$={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            onSubmit$();
          }
        }}
      />

      {/* Smart Actions (Right Side) */}
      <div class="absolute right-2 top-1/2 -translate-y-1/2 flex gap-2 opacity-0 group-focus-within:opacity-100 transition-opacity">
        <button 
          onClick$={onSubmit$}
          class="p-2 rounded-full hover:bg-white/10 text-[var(--glass-accent-cyan)]"
        >
          ⏎
        </button>
      </div>
    </div>
  );
});
