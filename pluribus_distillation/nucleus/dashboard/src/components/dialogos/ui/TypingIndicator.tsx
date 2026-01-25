/**
 * TypingIndicator.tsx
 * Author: gemini_components_1
 * Context: Phase 2 UI (Visual Feedback)
 * 
 * A "Thought Stream" visualizer.
 * Not just three dots, but a streaming neon pulse indicating active inference.
 */

import { component$, useStylesScoped$ } from '@builder.io/qwik';

export const TypingIndicator = component$(() => {
  useStylesScoped$(`
    .thought-stream {
      display: flex;
      gap: 4px;
      align-items: center;
      height: 20px;
    }
    .thought-dot {
      width: 4px;
      height: 4px;
      border-radius: 50%;
      background-color: var(--glass-accent-cyan);
      animation: pulse-wave 1.5s infinite ease-in-out;
    }
    .thought-dot:nth-child(1) { animation-delay: 0s; }
    .thought-dot:nth-child(2) { animation-delay: 0.2s; }
    .thought-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes pulse-wave {
      0%, 100% { transform: scale(0.8); opacity: 0.5; }
      50% { transform: scale(1.5); opacity: 1; box-shadow: 0 0 8px var(--glass-accent-cyan); }
    }
  `);

  return (
    <div class="glass-panel px-4 py-2 rounded-full inline-flex items-center gap-3 border border-[var(--glass-accent-cyan)]/30">
      <span class="text-xs text-[var(--glass-accent-cyan)] font-mono">THINKING</span>
      <div class="thought-stream">
        <div class="thought-dot"></div>
        <div class="thought-dot"></div>
        <div class="thought-dot"></div>
      </div>
    </div>
  );
});
