/**
 * SUPERMOTD: Pluribus Boot Sequence Display Component
 * =====================================================
 * Linux-inspired system status footer showing ring states,
 * bus metrics, omega heartbeats, and rotating insights.
 * 
 * ENHANCED: Now features 'HeartbeatOmega' visualizer.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { createBusClient } from '../lib/bus/bus-client';
import { HeartbeatOmega } from './art/HeartbeatOmega';

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

export const SuperMotdFooter = component$(() => {
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

    cleanup(() => {
      clearInterval(insightInterval);
      channel.close();
    });
  });

  const d = data.value;
  const list = bootLines.value.length > 0 ? bootLines.value : d.insights;
  const insightText = list[currentInsight.value] || 'System Ready';

  return (
    <div class="bg-card border-t border-border font-mono text-xs select-none z-50 relative flex flex-col">
      {/* 
        THE OMEGA BREAD 
        Replaces the old static status bar with the HeartbeatOmega visualizer.
      */}
      <HeartbeatOmega />

      {/* Expanded Details View (Toggled by HeartbeatOmega or explicit click) */}
      {/* 
         Note: HeartbeatOmega handles its own expansion visual, 
         but we can overlay the text details if needed.
         For now, we keep the text details below if the user wants to drill down.
         Actually, let's keep the details separate for now.
      */}
      <div
        class="px-4 py-1 flex items-center justify-between cursor-pointer hover:bg-muted/50 transition-colors border-t border-border"
        onClick$={() => expanded.value = !expanded.value}
      >
        <span class="text-[10px] text-muted-foreground">
          GEN:{d.rings.ring1.generation} • Ω:{d.rings.ring3.omega_cycle} • P:{d.rings.ring3.providers_available?.length || 0}
        </span>
        <span class="text-[10px] text-accent truncate max-w-md animate-pulse ml-auto">
          {insightText}
        </span>
        <span class={`text-muted-foreground transition-transform ml-2 ${expanded.value ? 'rotate-180' : ''}`}>
          ▲
        </span>
      </div>

      {/* Expanded view */}
      {expanded.value && (
        <div class="border-t border-border bg-background">
          {/* Loaded Modules Section */}
          <div class="px-4 py-3 border-b border-border">
            <div class="text-[10px] font-bold text-muted-foreground mb-2 tracking-wider">ACTIVE NEURAL MODULES</div>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">

              {/* System */}
              <div>
                <div class="text-[10px] text-cyan-500/70 mb-1">SYSTEM</div>
                <div class="flex flex-wrap gap-1">
                  {d.modules?.system.map(m => (
                    <span key={m} class="px-1.5 py-0.5 rounded bg-muted/20 text-primary border border-primary/30 text-[10px] shadow-sm">
                      {m}
                    </span>
                  ))}
                </div>
              </div>

              {/* Pluribus */}
              <div>
                <div class="text-[10px] text-purple-500/70 mb-1">PLURIBUS</div>
                <div class="flex flex-wrap gap-1">
                  {d.modules?.pluribus.map(m => (
                    <span key={m} class="px-1.5 py-0.5 rounded bg-muted/20 text-secondary border border-secondary/30 text-[10px] shadow-sm">
                      {m}
                    </span>
                  ))}
                </div>
              </div>

              {/* SOTA Integrated */}
              <div>
                <div class="text-[10px] text-pink-500/70 mb-1">SOTA INTEGRATED</div>
                <div class="flex flex-wrap gap-1">
                  {d.modules?.sota.map(m => (
                    <span key={m} class="px-1.5 py-0.5 rounded bg-muted/20 text-accent border border-accent/30 text-[10px] shadow-sm">
                      {m}
                    </span>
                  ))}
                </div>
              </div>

            </div>
          </div>

          {/* Rings Grid */}
          <div class="px-4 py-3 grid grid-cols-4 gap-4">
            {/* Ring 0 */}
            <div class="space-y-1">
              <div class={`${RING_COLORS.ring0} font-bold`}>[RING 0 :: {RING_LABELS.ring0}]</div>
              <div class="text-gray-500 flex items-center gap-1">
                <span class="text-green-500">●</span>
                Constitution: {d.rings.ring0.status}
              </div>
              <div class="text-gray-500 flex items-center gap-1">
                <span class="text-green-500">●</span>
                PQC: {d.rings.ring0.pqc_algorithm}
              </div>
            </div>

            {/* Ring 1 */}
            <div class="space-y-1">
              <div class={`${RING_COLORS.ring1} font-bold`}>[RING 1 :: {RING_LABELS.ring1}]</div>
              <div class="text-gray-500">
                Lineage: {d.rings.ring1.lineage_id}
              </div>
              <div class="text-gray-500">
                Type: {d.rings.ring1.transfer_type}
              </div>
            </div>

            {/* Ring 2 */}
            <div class="space-y-1">
              <div class={`${RING_COLORS.ring2} font-bold`}>[RING 2 :: {RING_LABELS.ring2}]</div>
              <div class="text-gray-500">
                Active Cells: {d.rings.ring2.infercells_active}
              </div>
              <div class="text-gray-500">
                Topology: Star/Mesh
              </div>
            </div>

            {/* Ring 3 */}
            <div class="space-y-1">
              <div class={`${RING_COLORS.ring3} font-bold`}>[RING 3 :: {RING_LABELS.ring3}]</div>
              <div class="text-gray-500">
                Cycles: {d.rings.ring3.omega_cycle}
              </div>
              <div class="text-gray-500 text-xs">
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