import { component$, type Signal, $, type QRL } from '@builder.io/qwik';
import type { VPSSession, BusEvent, AgentStatus, STRpRequest } from '../../lib/state/types';
import { MetafizzyDeck } from '../MetafizzyDeck';

interface StudioViewProps {
  session: Signal<VPSSession>;
  events: Signal<BusEvent[]>;
  agents: Signal<AgentStatus[]>;
  requests: Signal<STRpRequest[]>;
  onOpenAuth$?: QRL<() => void>;
}

export const StudioView = component$<StudioViewProps>((props) => {
  const openView = $((view: string) => {
    try {
      window.dispatchEvent(new CustomEvent('pluribus:navigate', { detail: { view } }));
    } catch {
      // ignore
    }
  });

  const openAuth = props.onOpenAuth$ ?? $(() => openView('browser-auth'));

  const providers = props.session.value.providers || {};
  const availableProviders = Object.values(providers).filter((p) => (p as any)?.available).length;
  const totalProviders = Object.keys(providers).length;

  return (
    <div class="space-y-4">
      {/* Step 96: glass-surface applied to studio header panel */}
      <div class="glass-surface glass-interactive p-4 rounded-xl">
        <div class="flex items-center justify-between gap-3">
          <div>
            <div class="text-lg font-semibold">Studio</div>
            <div class="text-xs text-muted-foreground">
              Draggable capability tiles (Metafizzy Packery + Draggabilly) — a "pre-webui TUI", but spatial.
            </div>
          </div>
          {/* Step 97: glass-chip for stats display */}
          <div class="glass-chip glass-chip-accent-cyan text-xs mono">
            events:{props.events.value.length} agents:{props.agents.value.length} req:{props.requests.value.length} providers:{availableProviders}/{totalProviders}
          </div>
        </div>
      </div>

      <MetafizzyDeck
        title="Capability Deck"
        subtitle="Drag tiles; Open jumps to the canonical view. (Resumability stays intact: heavy panels still load on demand.)"
        items={[
          {
            id: 'deck.bus',
            title: 'Bus Observatory',
            subtitle: 'Evidence • Work • Control',
            icon: '🧭',
            size: 'lg',
            onOpen$: $(() => openView('bus')),
            body: 'Dense “where did my task go?” view. Domain/actor lenses, tasks snapshot, providers, and Ω signals.',
          },
          {
            id: 'deck.voice',
            title: 'Voice & Speech',
            subtitle: 'Audio I/O (browser)',
            icon: '🎙️',
            size: 'md',
            onOpen$: $(() => openView('voice')),
            body: 'Speech-to-text for commands + text-to-speech for outputs (secure/local; no cloud dependency by default).',
          },
          {
            id: 'deck.events',
            title: 'Events',
            subtitle: 'Raw bus timeline',
            icon: '📜',
            size: 'md',
            onOpen$: $(() => openView('events')),
            body: 'Full event cards + flowmap + filters. Use when you need forensic fidelity, not just abstraction.',
          },
          {
            id: 'deck.semops',
            title: 'SemOps',
            subtitle: 'Operators + CRUD',
            icon: '🧠',
            size: 'md',
            onOpen$: $(() => openView('semops')),
            body: 'Define/undefine operators; map aliases → tools/bus/UI/agents; invoke non-blocking plans via bus.',
          },
          {
            id: 'deck.browser',
            title: 'Browser Auth',
            subtitle: 'VNC / PBVW',
            icon: '🌐',
            size: 'sm',
            onOpen$: openAuth,
            body: 'Manual OAuth via VNC (current pivot) + provider control plane visibility.',
          },
          {
            id: 'deck.webllm',
            title: 'WebLLM',
            subtitle: 'Edge inference',
            icon: '🧩',
            size: 'lg',
            onOpen$: $(() => openView('webllm')),
            body: 'WebGPU local inference. Opt-in warm start; fullscreen surface for multi-session chats + bus-addressable inference.',
          },
        ]}
      />
    </div>
  );
});
