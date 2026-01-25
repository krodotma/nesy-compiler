/**
 * DialogosWidget.tsx (The Unified Ingress)
 * Author: opus_integrator_1
 * Context: Phase 3 Integration
 */

import { component$, $, useVisibleTask$, useComputed$ } from '@builder.io/qwik';
import { DialogosShell } from './ui/DialogosShell';
import { NeonInput } from './ui/NeonInput';
import { SmartChips } from './ui/SmartChips';
import { TypingIndicator } from './ui/TypingIndicator';
import { FeatureDiscovery } from './ui/FeatureDiscovery';
import { ContextHologram } from './ui/hologram/ContextHologram';
import { useDialogosStore } from './store/use-dialogos-store';
import { IntentRouter } from './logic/IntentRouter';

export const DialogosWidget = component$(() => {
  const { state, setMode$, submit$, currentIntent } = useDialogosStore();

  return (
    <>
      {/* Holographic Layer (The Superjump) */}
      {state.mode !== 'rest' && <ContextHologram />}

      {/* Backdrop for closing (Active Mode) */}
      {state.mode !== 'rest' && (
        <div
          class="fixed inset-0 z-[8999] bg-black/50 backdrop-blur-sm transition-opacity duration-300"
          onClick$={() => setMode$('rest')}
        />
      )}

      <DialogosShell>
        {/* State: Thinking Indicator (Subtle Header) */}
        {state.isThinking && (
          <div class="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-[var(--glass-accent-cyan)] to-transparent animate-shimmer z-50" />
        )}

        {/* Main Content: Timeline */}
        {state.mode !== 'rest' && (
          <div class="flex-1 overflow-y-auto space-y-4 p-4 scrollbar-hide">
            {state.timeline.length === 0 ? (
              <div class="flex flex-col items-center justify-center h-full">
                <div class="text-center text-white/30 mb-8">
                  <div class="text-6xl mb-4 opacity-50 animate-pulse">✨</div>
                  <div class="text-xl font-light">Dialogos Ingress</div>
                  <div class="text-sm mt-2">Unified Epistemic Workspace</div>
                </div>
                <FeatureDiscovery />
              </div>
            ) : (
              state.timeline.map((id) => {
                const atom = state.atoms[id];
                return (
                  <div key={id} class={`flex ${atom.author.role === 'human' ? 'justify-end' : 'justify-start'}`}>
                    <div class={`
                    max-w-[80%] p-3 rounded-2xl glass-panel
                    ${atom.author.role === 'human' ? 'bg-white/10' : 'bg-black/30 border border-[var(--glass-border-bright)]'}
                  `}>
                      <div class="text-xs text-white/40 mb-1">{atom.author.name}</div>
                      <div class="text-sm text-white/90 whitespace-pre-wrap">
                        {atom.content.type === 'text' ? atom.content.value : JSON.stringify(atom.content)}
                      </div>
                    </div>
                  </div>
                );
              })
            )}

            {/* Active Thinking Indicator at bottom of stream */}
            {state.isThinking && (
              <div class="flex justify-start">
                <TypingIndicator />
              </div>
            )}
          </div>
        )}

        {/* Footer: Input Area */}
        <div class="mt-auto flex flex-col">
          {/* Context-Aware Chips (Only active mode) */}
          {state.mode !== 'rest' && (
            <SmartChips
              intent={currentIntent.value}
              onSelect$={(val) => (state.inputDraft += val)}
            />
          )}

          <NeonInput
            value={state.inputDraft}
            mode={state.mode}
            isThinking={state.isThinking}
            onInput$={(val) => (state.inputDraft = val)}
            onSubmit$={() => submit$(state.inputDraft)}
          />
        </div>
      </DialogosShell>
    </>
  );
});
