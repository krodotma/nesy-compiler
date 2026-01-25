/**
 * DialogosContainer - Dual-Mind Conversation Orchestrator
 *
 * Wraps WebLLMWidget with Dialogos protocol for:
 * - Auto-loading two cached "mind" agents on startup
 * - Injecting symbolic seed prompts to trigger conversation flow
 * - Cross-agent message relay via bus events (avoids Qwik SSR issues)
 * - Omega Protocol intervention for stalled conversations
 * - Model version checking every 4 hours
 *
 * DIALOGOS PROTOCOL:
 * Two local LLM "minds" engage in recursive self-reflective dialogue,
 * proving loop-viable inference within WebGPU browser-native edge environment.
 */

import { component$, useSignal, useVisibleTask$, $, noSerialize, type NoSerialize } from '@builder.io/qwik';
import { createBusClient } from '../lib/bus/bus-client';

// M3 Components - DialogosContainer
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';
import {
  selectDialogosSeed,
  formatDialogosPeerMessage,
  getOmegaIntervention,
  checkModelVersions,
  createDialogosState,
  saveDialogosState,
  getDialogosModels,
  type DialogosState,
} from '../lib/webllm-enhanced';

interface DialogosContainerProps {
  /** Auto-start Dialogos conversation on mount */
  autoStart?: boolean;
  /** Minimum ready sessions before starting */
  minSessions?: number;
  /** Milliseconds between turn checks */
  pollInterval?: number;
  /** Stall threshold in milliseconds */
  stallThreshold?: number;
}

export const DialogosContainer = component$<DialogosContainerProps>(({
  autoStart = true,
  minSessions = 2,
  pollInterval = 1000,
  stallThreshold = 45000,
}) => {
  const dialogosState = useSignal<DialogosState>(createDialogosState());
  const versionInfo = useSignal<{ checkedAt: number; hasUpdates: boolean }>({
    checkedAt: 0,
    hasUpdates: false,
  });
  const busConnected = useSignal(false);
  const lastActivityAt = useSignal(0);
  const busClientRef = useSignal<NoSerialize<ReturnType<typeof createBusClient>> | null>(null);

  // Version checking on mount (every 4 hours)
  useVisibleTask$(async () => {
    try {
      const result = await checkModelVersions();
      versionInfo.value = {
        checkedAt: result.checkedAt,
        hasUpdates: result.needsUpdate,
      };
      if (result.needsUpdate) {
        console.log('[Dialogos] Model updates available - consider clearing cache');
      }
    } catch (err) {
      console.warn('[Dialogos] Version check failed:', err);
    }
  });

  // Dialogos bus integration
  useVisibleTask$(({ cleanup }) => {
    if (!autoStart) return;

    let disposed = false;
    const client = createBusClient({ platform: 'browser' });

    const run = async () => {
      try {
        await client.connect();
        if (disposed) return;
        busConnected.value = true;
        busClientRef.value = noSerialize(client);

        // Subscribe to WebLLM status to track session readiness
        client.subscribe('webllm.widget.status', (ev: any) => {
          const data = ev?.data || {};
          const readySessions = data.sessions_ready || 0;

          // Check if we have enough sessions for Dialogos
          if (readySessions >= minSessions && !dialogosState.value.active) {
            console.log(`[Dialogos] ${readySessions} sessions ready, initiating dual-mind conversation...`);

            dialogosState.value = {
              ...dialogosState.value,
              active: true,
              startedAt: Date.now(),
              conversationId: `dialogos-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
            };
            lastActivityAt.value = Date.now();
            saveDialogosState(dialogosState.value);

            // Emit seed prompt via bus
            const seed = selectDialogosSeed();
            client.publish({
              topic: 'dialogos.seed',
              kind: 'command',
              level: 'info',
              actor: 'dialogos-container',
              data: {
                conversation_id: dialogosState.value.conversationId,
                seed_prompt: seed,
                target_session: 0, // First ready session
              },
            });
          }
        });

        // Subscribe to inference responses to orchestrate turn-taking
        client.subscribe('webllm.infer.response', (ev: any) => {
          const data = ev?.data || {};
          if (!data.ok || !dialogosState.value.active) return;

          const sessionId = data.session_id;
          const responseText = data.text || '';
          const modelName = data.model_name || 'Unknown';

          // Update turn state
          dialogosState.value = {
            ...dialogosState.value,
            turnCount: dialogosState.value.turnCount + 1,
            lastSpeaker: sessionId,
          };
          lastActivityAt.value = Date.now();
          saveDialogosState(dialogosState.value);

          // Emit peer message for other session(s)
          const peerMessage = formatDialogosPeerMessage(
            modelName,
            responseText,
            dialogosState.value.turnCount
          );

          client.publish({
            topic: 'dialogos.relay',
            kind: 'event',
            level: 'info',
            actor: 'dialogos-container',
            data: {
              conversation_id: dialogosState.value.conversationId,
              turn: dialogosState.value.turnCount,
              from_session: sessionId,
              from_model: modelName,
              peer_message: peerMessage,
            },
          });
        });

      } catch (err) {
        console.warn('[Dialogos] Bus connection failed:', err);
        busConnected.value = false;
      }
    };

    run();

    cleanup(() => {
      disposed = true;
      try { client.disconnect(); } catch { /* ignore */ }
      busConnected.value = false;
      busClientRef.value = null;
    });
  });

  // Omega Protocol stall detection
  useVisibleTask$(({ cleanup }) => {
    if (!autoStart) return;

    const interval = setInterval(() => {
      if (!dialogosState.value.active) return;

      const now = Date.now();
      const last = lastActivityAt.value || dialogosState.value.startedAt;
      const elapsedMs = now - last;

      if (elapsedMs > stallThreshold) {
        const intervention = getOmegaIntervention(dialogosState.value.turnCount);
        const client = busClientRef.value;
        if (client) {
          client.publish({
            topic: 'dialogos.omega',
            kind: 'command',
            level: 'warn',
            actor: 'dialogos-container',
            data: {
              conversation_id: dialogosState.value.conversationId,
              turn: dialogosState.value.turnCount,
              intervention,
              stall_duration_ms: elapsedMs,
            },
          });
        }
        lastActivityAt.value = now;
      }
    }, pollInterval);

    cleanup(() => clearInterval(interval));
  });

  // Recommended fast models for Dialogos
  const dialogosModels = getDialogosModels();

  return (
    <div class="dialogos-container p-4 border border-purple-500/30 rounded-lg bg-purple-500/5">
      <div class="flex items-center gap-3 mb-4">
        <span class={`w-3 h-3 rounded-full ${
          dialogosState.value.active ? 'bg-purple-400 animate-pulse' :
          busConnected.value ? 'bg-green-400' : 'bg-gray-400'
        }`} />
        <h3 class="text-lg font-semibold text-purple-300">🧠 Dialogos Dual-Mind</h3>
        {versionInfo.value.hasUpdates && (
          <span class="text-xs px-2 py-0.5 rounded bg-yellow-500/20 text-yellow-400">
            Updates Available
          </span>
        )}
      </div>

      <div class="space-y-2 text-sm">
        <div class="flex justify-between">
          <span class="text-muted-foreground">Status:</span>
          <span class={dialogosState.value.active ? 'text-purple-400' : 'text-gray-400'}>
            {dialogosState.value.active ? 'Active' : 'Waiting for sessions...'}
          </span>
        </div>

        {dialogosState.value.active && (
          <>
            <div class="flex justify-between">
              <span class="text-muted-foreground">Conversation ID:</span>
              <span class="text-xs font-mono text-cyan-400">
                {dialogosState.value.conversationId.slice(-12)}
              </span>
            </div>
            <div class="flex justify-between">
              <span class="text-muted-foreground">Turns:</span>
              <span class="text-green-400">{dialogosState.value.turnCount}</span>
            </div>
          </>
        )}

        <div class="mt-4 pt-4 border-t border-border/30">
          <div class="text-xs text-muted-foreground mb-2">Recommended Models for Dialogos:</div>
          <div class="flex flex-wrap gap-1">
            {dialogosModels.slice(0, 4).map((m) => (
              <span
                key={m.baseId}
                class="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300"
              >
                {m.name} (⚡{m.speedRating})
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

export default DialogosContainer;
