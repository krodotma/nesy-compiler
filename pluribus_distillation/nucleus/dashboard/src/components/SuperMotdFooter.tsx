/**
 * SUPERMOTD: Pluribus Boot Sequence Display Component
 * =====================================================
 * Linux-inspired system status footer showing ring states,
 * bus metrics, omega heartbeats, and rotating insights.
 * 
 * ENHANCED: Now features 'HeartbeatOmega' visualizer and
 * 'Cognitive Boot Sequence' (Ensoulment Plan).
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { createBusClient } from '../lib/bus/bus-client';
import { HeartbeatOmega } from './art/HeartbeatOmega';
import { useTracking } from '../lib/telemetry/use-tracking';
import { NeonTitle, NeonBadge, NeonSectionHeader } from './ui/NeonTitle';
import { AgentAvatar } from './AgentAvatar';

// M3 Components - SuperMotdFooter
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/assist-chip.js';

interface RingStatus {
  status?: string;
  constitution_hash?: string;
  lineage_id?: string;
  generation?: number;
  transfer_type?: string;
  rhizome_objects?: number;
  omega_healthy?: boolean;
  omega_cycle?: number;
  providers_available?: string[];
  providers_total?: number;
  infercells_active?: number;
  pqc_algorithm?: string;
}

interface SuperMOTDData {
  hostname: string;
  uptime_s: number;
  rings: {
    ring0: RingStatus;
    ring1: RingStatus;
    ring2: RingStatus;
    ring3: RingStatus;
  };
  bus: {
    events: number;
    last_event: string;
  };
  insights: string[];
  modules?: {
    system: string[];
    pluribus: string[];
    sota: string[];
  };
}

const RING_COLORS = {
  ring0: 'text-red-400',
  ring1: 'text-yellow-400',
  ring2: 'text-green-400',
  ring3: 'text-cyan-400',
};

const RING_LABELS = {
  ring0: 'KERNEL',
  ring1: 'PROTECTED',
  ring2: 'AGENT',
  ring3: 'EPHEMERAL',
};

const ENSOULMENT_STEPS = [
  "1. Map sqlite-vec to Associative Cortex",
  "2. Map graphiti to Hippocampus",
  "3. Map mem0 to Working Memory",
  "4. Map tensorzero to The Gatekeeper",
  "5. Map agent-lightning to The Dojo",
  "6. Visualize Awakening Progress",
  "7. Launch Self-Aware Boot Mode"
];

export const SuperMotdFooter = component$(() => {
  useTracking("comp:supermotd");
  const data = useSignal<SuperMOTDData>({
    hostname: 'pluribus',
    uptime_s: 0,
    rings: {
      ring0: { status: 'sealed', pqc_algorithm: 'ML-DSA-65' },
      ring1: { lineage_id: 'core.genesis', generation: 1, transfer_type: 'VGT' },
      ring2: { infercells_active: 0 },
      ring3: { omega_healthy: true, omega_cycle: 0, providers_available: [], providers_total: 6 },
    },
    bus: { events: 0, last_event: '' },
    insights: ['System Online. Monitoring bus traffic...'],
    modules: {
      system: ['kernel', 'zfs', 'net'],
      pluribus: ['auom', 'sextet', 'kroma', 'rhizome', 'bus'],
      sota: ['yt-dlp', 'tesseract', 'sqlite-vec', 'isomorphic-git']
    }
  });

  const currentInsight = useSignal(0);
  const expanded = useSignal(false);
  const bootLines = useSignal<string[]>([]);
  const activeEnsoulmentStep = useSignal(0);

  useVisibleTask$(({ cleanup }) => {
    // SOTA Optimization: Consume events from the Omega Worker (Brainstem)
    // instead of opening a duplicate WebSocket connection.
    const channel = new BroadcastChannel('pluribus-omega');

    let eventCount = 0;

    channel.onmessage = (msg) => {
      if (msg.data.type === 'BUS_EVENT' && msg.data.event) {
        const evt = msg.data.event;
        eventCount++;
        data.value.bus.events = eventCount;

        // 1. Capture Boot Log (SLOU)
        if (evt.topic === 'system.boot.log' && Array.isArray(evt.data?.boot_log)) {
          const lines = evt.data.boot_log.map((l: string) => l.replace(/\x1b\[[0-9;]*m/g, ''));
          bootLines.value = lines;
          data.value.insights = lines;
          currentInsight.value = 0;
        }
        // 1b. Late Joiner: Generate Synthetic Log from Live Events
        else if (bootLines.value.length === 0) {
          const time = new Date().toLocaleTimeString();
          // Filter out heartbeat noise from the log view
          if (evt.topic !== 'omega.heartbeat') {
            const log = `[${time}] ${evt.topic.padEnd(20)} ${evt.actor || 'sys'}... OK`;
            const nextInsights = [...data.value.insights, log].slice(-20);
            data.value.insights = nextInsights;
            currentInsight.value = nextInsights.length - 1;
          }
        }

        // 2. Capture InferCell state (Ring 2)
        if (evt.topic === 'infercell.genesis' || evt.topic === 'infercell.fork') {
          data.value.rings.ring2.infercells_active = (data.value.rings.ring2.infercells_active || 0) + 1;
        }

        // 3. Capture Provider state (Ring 3)
        if (evt.topic === 'dashboard.vps.provider_status') {
          const d = evt.data || {};
          const provs = data.value.rings.ring3.providers_available || [];
          if (d.available && !provs.includes(d.provider)) {
            data.value.rings.ring3.providers_available = [...provs, d.provider];
          }
        }

        // 4. Capture Omega Heartbeat
        if (evt.topic === 'omega.heartbeat') {
          data.value.rings.ring3.omega_cycle = evt.data?.cycle || 0;
        }
      } else if (msg.data.type === 'OMEGA_TICK') {
        // Sync ring status with the master state if needed
        const state = msg.data.state;
        if (state.connected !== undefined) {
          // Could update a 'connected' indicator on the footer if we added one
        }
      }
    };

    // Rotate insights
    const insightInterval = setInterval(() => {
      const list = bootLines.value.length > 0 ? bootLines.value : data.value.insights;
      if (list.length > 0) {
        currentInsight.value = (currentInsight.value + 1) % list.length;
      }
    }, 2500);
    
    // Rotate Ensoulment Steps
    const ensoulmentInterval = setInterval(() => {
      activeEnsoulmentStep.value = (activeEnsoulmentStep.value + 1) % ENSOULMENT_STEPS.length;
    }, 4000);

    cleanup(() => {
      clearInterval(insightInterval);
      clearInterval(ensoulmentInterval);
      channel.close();
    });
  });

  const d = data.value;
  const list = bootLines.value.length > 0 ? bootLines.value : d.insights;
  const insightText = list[currentInsight.value] || 'System Ready';

  return (
    <div class="glass-footer font-mono text-sm select-none flex flex-col">
      {/* 
        THE OMEGA BREAD 
        Replaces the old static status bar with the HeartbeatOmega visualizer.
      */}
      <HeartbeatOmega />

      {/* Expanded Details View (Toggled by HeartbeatOmega or explicit click) */}
      <div
        class="glass-footer-interactive px-4 py-2 flex items-center justify-between border-t border-[var(--glass-border)]"
        onClick$={() => expanded.value = !expanded.value}
      >
        <div class="flex items-center gap-3">
          <AgentAvatar actor="OM" status={d.rings.ring3.omega_healthy ? "active" : "error"} size="sm" />
          <span class="text-xs glass-text-muted">
            GEN:{d.rings.ring1.generation} • Ω:{d.rings.ring3.omega_cycle} • P:{d.rings.ring3.providers_available?.length || 0}
          </span>
        </div>
        <span class="text-xs glass-text-cyan truncate max-w-md animate-pulse ml-auto">
          {insightText}
        </span>
        <span class={`glass-text-muted transition-transform duration-200 ease-out ml-2 ${expanded.value ? 'rotate-180' : ''}`}>
          ▲
        </span>
      </div>

      {/* Expanded view */}
      {expanded.value && (
        <div class="glass-animate-enter border-t border-[var(--glass-border)] bg-[var(--glass-bg-dark)]">
          {/* Cognitive Boot Sequence (Ensoulment) */}
          <div class="px-4 py-2 bg-cyan-950/20 flex items-center justify-center border-b border-[var(--glass-border)]">
            <div class="flex items-center gap-2 text-xs">
              <span class="text-amber-400 animate-pulse font-bold">⚡ COGNITIVE BOOT:</span>
              <span class="text-cyan-100 transition-opacity duration-500 key={activeEnsoulmentStep.value}">
                {ENSOULMENT_STEPS[activeEnsoulmentStep.value]}
              </span>
              <span class="text-muted-foreground ml-2">
                [{activeEnsoulmentStep.value + 1}/{ENSOULMENT_STEPS.length}]
              </span>
            </div>
          </div>

          {/* Loaded Modules Section */}
          <div class="px-4 py-3 border-b border-[var(--glass-border)]">
            <NeonSectionHeader title="ACTIVE NEURAL MODULES" color="cyan" size="xs" />
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">

              {/* System */}
              <div>
                <NeonTitle level="div" color="cyan" size="xs" class="opacity-70 mb-1">SYSTEM</NeonTitle>
                <div class="flex flex-wrap gap-1">
                  {d.modules?.system.map(m => (
                    <NeonBadge key={m} color="cyan">{m}</NeonBadge>
                  ))}
                </div>
              </div>

              {/* Pluribus */}
              <div>
                <NeonTitle level="div" color="purple" size="xs" class="opacity-70 mb-1">PLURIBUS</NeonTitle>
                <div class="flex flex-wrap gap-1">
                  {d.modules?.pluribus.map(m => (
                    <NeonBadge key={m} color="purple">{m}</NeonBadge>
                  ))}
                </div>
              </div>

              {/* SOTA Integrated */}
              <div>
                <NeonTitle level="div" color="magenta" size="xs" class="opacity-70 mb-1">SOTA INTEGRATED</NeonTitle>
                <div class="flex flex-wrap gap-1">
                  {d.modules?.sota.map(m => (
                    <NeonBadge key={m} color="magenta">{m}</NeonBadge>
                  ))}
                </div>
              </div>

            </div>
          </div>

          {/* Rings Grid */}
          <div class="px-4 py-3 grid grid-cols-4 gap-4">
            {/* Ring 0 */}
            <div class="space-y-1">
              <NeonTitle level="div" color="rose" size="xs">[RING 0 :: {RING_LABELS.ring0}]</NeonTitle>
              <div class="glass-text-muted flex items-center gap-1">
                <span class="glass-text-emerald">●</span>
                Constitution: {d.rings.ring0.status}
              </div>
              <div class="glass-text-muted flex items-center gap-1">
                <span class="glass-text-emerald">●</span>
                PQC: {d.rings.ring0.pqc_algorithm}
              </div>
            </div>

            {/* Ring 1 */}
            <div class="space-y-1">
              <NeonTitle level="div" color="amber" size="xs">[RING 1 :: {RING_LABELS.ring1}]</NeonTitle>
              <div class="glass-text-muted">
                Lineage: {d.rings.ring1.lineage_id}
              </div>
              <div class="glass-text-muted">
                Type: {d.rings.ring1.transfer_type}
              </div>
            </div>

            {/* Ring 2 */}
            <div class="space-y-1">
              <NeonTitle level="div" color="emerald" size="xs">[RING 2 :: {RING_LABELS.ring2}]</NeonTitle>
              <div class="glass-text-muted">
                Active Cells: {d.rings.ring2.infercells_active}
              </div>
              <div class="glass-text-muted">
                Topology: Star/Mesh
              </div>
            </div>

            {/* Ring 3 */}
            <div class="space-y-1">
              <NeonTitle level="div" color="cyan" size="xs">[RING 3 :: {RING_LABELS.ring3}]</NeonTitle>
              <div class="glass-text-muted">
                Cycles: {d.rings.ring3.omega_cycle}
              </div>
              <div class="glass-text-muted text-xs">
                {d.rings.ring3.providers_available?.join(' ') || 'seeking...'}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

/**
 * Compact status line for embedding in other components
 */
export const SuperMOTDCompact = component$(() => {
  return <div />;
});