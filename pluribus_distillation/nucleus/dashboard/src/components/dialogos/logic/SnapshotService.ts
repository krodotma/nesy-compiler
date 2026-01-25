/**
 * SnapshotService.ts
 * Author: opus_algo_1
 * Context: Phase 2 - The Brain (Context Awareness)
 * 
 * Captures the current state of the user's environment to attach to Dialogos Atoms.
 * This provides the "Grounding" for the agent.
 */

import type { ContextSnapshot } from '../types/dialogos';

export class SnapshotService {
  
  static async capture(): Promise<ContextSnapshot> {
    if (typeof window === 'undefined') {
      return { url: 'server', viewport: { width: 0, height: 0 } };
    }

    // 1. Basic Browser Context
    const snapshot: ContextSnapshot = {
      url: window.location.href,
      viewport: {
        width: window.innerWidth,
        height: window.innerHeight
      },
      selection: window.getSelection()?.toString() || undefined,
      systemLoad: (navigator as any).hardwareConcurrency ? 1 / (navigator as any).hardwareConcurrency : 0.5 // Proxy metric
    };

    // 2. Active Element Context (Heuristic)
    const activeEl = document.activeElement;
    if (activeEl) {
      // If inside IPE or CodeMirror, try to scrape context
      // This is a loose coupling; ideally we'd use a shared signal
      if (activeEl.classList.contains('cm-content')) {
        // We are in an editor
        snapshot.activeFile = 'unknown-buffer'; // TODO: Hook into IPE State
      }
    }

    return snapshot;
  }
}
