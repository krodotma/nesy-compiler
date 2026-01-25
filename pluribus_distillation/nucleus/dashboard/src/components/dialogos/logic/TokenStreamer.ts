/**
 * TokenStreamer.ts
 * Author: opus_backend_1
 * Context: Phase 2 - Streaming Logic
 * 
 * Handles partial token updates from the bus to create a smooth typing effect.
 * Accumulates tokens into a buffer and updates the Atom in real-time.
 */

import type { DialogosAtom } from '../types/dialogos';

export class TokenStreamer {
  private buffer: string = '';
  private updateCallback: (text: string) => void;

  constructor(updateCallback: (text: string) => void) {
    this.updateCallback = updateCallback;
  }

  append(token: string) {
    this.buffer += token;
    // Throttle updates slightly for performance if needed
    this.updateCallback(this.buffer);
  }

  reset() {
    this.buffer = '';
  }
}
