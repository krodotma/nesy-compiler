/**
 * Easter Egg: The Konami Code
 * [Ultrathink Agent 3: Ontologist]
 * 
 * "Hidden paths lead to hidden truths."
 * Triggers 'God Mode' (Max CMP, Golden State).
 */

import { $, useOnWindow, type QRL, type Signal } from '@builder.io/qwik';

const KONAMI_SEQUENCE = [
  'ArrowUp', 'ArrowUp', 
  'ArrowDown', 'ArrowDown', 
  'ArrowLeft', 'ArrowRight', 
  'ArrowLeft', 'ArrowRight', 
  'b', 'a'
];

export function useKonamiCode(onUnlock$: QRL<() => void>) {
  // We use a closure variable for sequence tracking since we can't easily 
  // inspect the Signal in the event handler without tracking it in render
  // For simplicity in Qwik, we'll use a data attribute on body or window scope 
  // if strict state is needed, but here we can reset on mismatch.
  
  useOnWindow('keydown', $((e: KeyboardEvent) => {
    // Basic state management via window object to persist across renders 
    // without triggering re-renders for every keypress
    const win = window as any;
    if (!win._konamiIdx) win._konamiIdx = 0;
    
    if (e.key === KONAMI_SEQUENCE[win._konamiIdx]) {
      win._konamiIdx++;
      if (win._konamiIdx === KONAMI_SEQUENCE.length) {
        onUnlock$();
        win._konamiIdx = 0;
      }
    } else {
      win._konamiIdx = 0;
    }
  }));
}

// "System Thoughts" Generator
export const SYSTEM_THOUGHTS = [
  "Analyzing local minima...",
  "Computing manifold distance...",
  "Traversing clade ancestry...",
  "Verifying HGT compatibility...",
  "Optimizing thermodynamic efficiency...",
  "Syncing with Omega point...",
  "Detecting motif recurrence...",
  "Refining vector embeddings...",
  "Pruning entropic branches...",
  "Synthesizing negentropy...",
];
