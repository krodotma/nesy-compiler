/**
 * SystemListener.ts (The Ears)
 * Author: opus_integrator_1
 * Context: Phase 3 - Nervous System
 * 
 * Listens to the global bus for system events (Errors, Terminal Output)
 * and injects them into the Dialogos stream as context or alerts.
 */

import { createBusClient } from '../../../lib/bus/bus-client';
import type { DialogosAtom } from '../types/dialogos';

export class SystemListener {
  private client: any;
  private injectAtom: (atom: DialogosAtom) => void;

  constructor(injectAtom: (atom: DialogosAtom) => void) {
    this.client = createBusClient({ platform: 'browser' });
    this.injectAtom = injectAtom;
  }

  async init() {
    await this.client.connect();

    // Listen for Critical Errors
    this.client.subscribe('*', (event: any) => {
      if (event.level === 'error') {
        this.createAlertAtom(event);
      }
    });

    // Listen for Terminal Output (StdErr usually implies help needed)
    this.client.subscribe('terminal.stderr', (event: any) => {
       this.createTerminalAtom(event);
    });
  }

  private createAlertAtom(event: any) {
    const atom: DialogosAtom = {
      id: `sys-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      author: { id: 'system', name: 'System', role: 'system' },
      intent: 'clarification',
      content: { 
        type: 'text', 
        value: `System Alert (${event.actor}): ${event.data?.error || event.data?.message || 'Unknown Error'}` 
      },
      context: { url: 'system' },
      state: 'actualized',
      causes: [],
      effects: []
    };
    this.injectAtom(atom);
  }

  private createTerminalAtom(event: any) {
    // Only capture significant output
    const text = event.data?.text || '';
    if (text.length < 10) return;

    const atom: DialogosAtom = {
      id: `term-${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      author: { id: 'terminal', name: 'Terminal', role: 'system' },
      intent: 'query', // Treat as a query "What does this error mean?"
      content: { 
        type: 'code', 
        language: 'bash',
        value: text 
      },
      context: { url: 'terminal' },
      state: 'potential', // It's a potential problem to solve
      causes: [],
      effects: []
    };
    this.injectAtom(atom);
  }

  disconnect() {
    this.client.disconnect();
  }
}
