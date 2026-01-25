/**
 * useDialogos - React/Qwik hook for Dialogos dual-mind integration
 *
 * Listens to Dialogos bus events and triggers inference requests
 * in WebLLM sessions to orchestrate cross-model conversation.
 */

import { createBusClient } from './bus/bus-client';
import {
  selectDialogosSeed,
  formatDialogosPeerMessage,
  getOmegaIntervention,
  createDialogosState,
  type DialogosState,
} from './webllm-enhanced';

export interface DialogosHookOptions {
  /** Callback when a seed prompt should be injected */
  onSeedPrompt?: (sessionIndex: number, prompt: string) => void;
  /** Callback when a peer message should be relayed */
  onPeerMessage?: (targetSessionId: string, message: string) => void;
  /** Callback when Omega intervention needed */
  onOmegaIntervention?: (message: string) => void;
  /** Enable console logging */
  debug?: boolean;
}

export interface DialogosController {
  state: DialogosState;
  start: () => void;
  stop: () => void;
  isActive: () => boolean;
  getTurnCount: () => number;
}

/**
 * Initialize Dialogos dual-mind conversation orchestration
 */
export function initDialogos(options: DialogosHookOptions = {}): DialogosController {
  const { onSeedPrompt, onPeerMessage, onOmegaIntervention, debug = false } = options;

  let state = createDialogosState();
  let client: ReturnType<typeof createBusClient> | null = null;
  let connected = false;
  let unsubSeed: (() => void) | null = null;
  let unsubRelay: (() => void) | null = null;
  let stallTimer: ReturnType<typeof setInterval> | null = null;
  let lastActivityTime = 0;

  const log = (...args: any[]) => {
    if (debug) console.log('[Dialogos]', ...args);
  };

  const start = async () => {
    if (state.active) return;

    try {
      client = createBusClient({ platform: 'browser' });
      await client.connect();
      connected = true;

      state = {
        ...state,
        active: true,
        startedAt: Date.now(),
        conversationId: `dialogos-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      };
      lastActivityTime = Date.now();

      log('Started conversation:', state.conversationId);

      // Listen for seed events from DialogosContainer
      unsubSeed = client.subscribe('dialogos.seed', (ev: any) => {
        const data = ev?.data || {};
        if (data.conversation_id !== state.conversationId) return;

        const seedPrompt = data.seed_prompt || selectDialogosSeed();
        const targetSession = data.target_session ?? 0;

        log('Seed received for session', targetSession, ':', seedPrompt.slice(0, 50) + '...');
        lastActivityTime = Date.now();

        if (onSeedPrompt) {
          onSeedPrompt(targetSession, seedPrompt);
        }
      });

      // Listen for relay events (peer-to-peer messages)
      unsubRelay = client.subscribe('dialogos.relay', (ev: any) => {
        const data = ev?.data || {};
        if (data.conversation_id !== state.conversationId) return;

        const peerMessage = data.peer_message;
        const fromSession = data.from_session;

        log('Relay from', data.from_model, '- turn', data.turn);
        lastActivityTime = Date.now();

        state.turnCount = data.turn || state.turnCount + 1;
        state.lastSpeaker = fromSession;

        // Find a different session to relay to
        // (The DialogosContainer or WebLLMWidget should handle finding the right target)
        if (onPeerMessage) {
          onPeerMessage(fromSession, peerMessage);
        }
      });

      // Stall detection - check every 5 seconds
      stallTimer = setInterval(() => {
        if (!state.active) return;

        const elapsed = Date.now() - lastActivityTime;
        if (elapsed > 20000) { // 20 second stall threshold
          log('Stall detected, invoking Omega Protocol');
          const intervention = getOmegaIntervention(state.turnCount);

          if (onOmegaIntervention) {
            onOmegaIntervention(intervention);
          }

          // Emit on bus for other listeners
          if (connected && client) {
            client.publish({
              topic: 'dialogos.omega',
              kind: 'command',
              level: 'warn',
              actor: 'dialogos-hook',
              data: {
                conversation_id: state.conversationId,
                turn: state.turnCount,
                intervention,
                stall_duration_ms: elapsed,
              },
            });
          }

          lastActivityTime = Date.now(); // Reset to prevent spam
        }
      }, 5000);

      // Emit initial seed after short delay
      setTimeout(() => {
        if (!state.active || !connected || !client) return;

        const seed = selectDialogosSeed();
        log('Emitting initial seed...');

        client.publish({
          topic: 'dialogos.seed',
          kind: 'command',
          level: 'info',
          actor: 'dialogos-hook',
          data: {
            conversation_id: state.conversationId,
            seed_prompt: seed,
            target_session: 0,
          },
        });
      }, 2000);

    } catch (err) {
      log('Failed to start:', err);
      state.active = false;
    }
  };

  const stop = () => {
    state.active = false;

    if (stallTimer) {
      clearInterval(stallTimer);
      stallTimer = null;
    }

    try { unsubSeed?.(); } catch { /* ignore */ }
    try { unsubRelay?.(); } catch { /* ignore */ }
    try { client?.disconnect(); } catch { /* ignore */ }

    connected = false;
    client = null;

    log('Stopped');
  };

  return {
    get state() { return state; },
    start,
    stop,
    isActive: () => state.active,
    getTurnCount: () => state.turnCount,
  };
}

export default initDialogos;
