/**
 * PBTSO Bridge (The Nervous System)
 * Author: opus_backend_1
 * Context: Phase 4 Integration - Connecting Mind (UI) to Body (Backend)
 *
 * This bridge listens for 'task' atoms in the Dialogos stream and
 * transmits them to the backend task ingress. Control plane remains
 * PBTSO tmux orchestration; bus events are evidence/telemetry.
 */
import type { DialogosAtom } from '../types/dialogos';
import { createBusClient } from '../../../lib/bus/bus-client';

export class PBTSOBridge {
  private client: any;
  private storeActions: any; // updateAtomState callback

  constructor(storeActions: { updateAtomState: (id: string, state: string) => void }) {
    this.client = createBusClient({ platform: 'browser' });
    this.storeActions = storeActions;
  }

  async init() {
    await this.client.connect();

    // Listen for Backend Acknowledgements (canonical + legacy)
    const handleAck = (event: any) => {
      const { correlationId, taskId } = event.data;
      if (correlationId) {
        // Transition Atom from 'actualizing' to 'actualized'
        this.storeActions.updateAtomState(correlationId, 'actualized');
        console.log(`[PBTSO] Task Confirmed: ${taskId} for Atom ${correlationId}`);
      }
    };

    this.client.subscribe('pbtso.task.created', handleAck);
    this.client.subscribe('tbtso.task.created', handleAck);
  }

  /**
   * Transmit a task atom to the backend.
   * "Mutation of State"
   */
  async dispatch(atom: DialogosAtom) {
    if (atom.intent !== 'task') return;

    // 1. Mark as processing in UI
    this.storeActions.updateAtomState(atom.id, 'actualizing');

    // 2. Emit to Bus (legacy evidence channel)
    await this.client.publish({
      topic: 'task.create',
      kind: 'command',
      actor: 'dialogos-ingress',
      data: {
        title: atom.content.type === 'task' ? atom.content.title : 'Untitled Task',
        status: 'todo',
        lane: atom.content.type === 'task' ? atom.content.laneId : 'inbox',
        source: 'dialogos',
        correlation_id: atom.id, // Critical for round-trip tracking
        context: atom.context
      }
    });
  }

  disconnect() {
    this.client.disconnect();
  }
}
