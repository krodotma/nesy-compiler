/**
 * IPE Bridge (The Hands)
 * Author: opus_integrator_1
 * Context: Phase 2 Integration
 * 
 * Connects Dialogos intentions to the In-Place Editor (CodeWarrior).
 * Dispatches code mutations and listens for context updates.
 */

import { createBusClient } from '../../../lib/bus/bus-client';
import type { DialogosAtom } from '../types/dialogos';

export class IPEBridge {
  private client: any;

  constructor() {
    this.client = createBusClient({ platform: 'browser' });
  }

  async init() {
    await this.client.connect();
  }

  /**
   * Dispatch a code mutation request to the IPE.
   * This triggers the "CodeWarrior" visualizer.
   */
  async dispatchMutation(atom: DialogosAtom) {
    if (atom.intent !== 'mutation' || atom.content.type !== 'code') return;

    await this.client.publish({
      topic: 'evolution.synthesizer.patch',
      kind: 'command',
      actor: 'dialogos-ingress',
      data: {
        target_file: atom.context.activeFile || 'unknown', // Best effort context
        diff: atom.content.value,
        language: atom.content.language,
        source_sota: 'dialogos-manual',
        rationale: 'User requested mutation via Dialogos',
        correlation_id: atom.id
      }
    });
  }

  disconnect() {
    this.client.disconnect();
  }
}
