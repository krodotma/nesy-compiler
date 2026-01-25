/**
 * CodeWarriorTab.tsx
 *
 * The "Agentic" editor tab for IPE.
 * Visualizes Clade-Metaproductivity (CMP), Holon Lineage, and Evolutionary Mutations.
 * This is the interface for "Evolutionary Code Agency".
 */

import {
  component$,
  useSignal,
  $,
} from '@builder.io/qwik';
import type { IPEContext } from '../../../lib/ipe';
import { EntropyCanvas } from '../shaders/EntropyCanvas';
import { HolonJewel } from '../3d/HolonJewel';
import { MutationCockpit } from '../mutation/MutationCockpit';
import { CMPRadar } from '../viz/CMPRadar';
import { LatticeBackground } from '../viz/LatticeBackground';
import { COHORT_AVERAGE } from '../../../lib/holon/cmp';
import { useKonamiCode, SYSTEM_THOUGHTS } from '../../../lib/holon/easter-egg';
import { useCodeWarriorStream } from '../../../lib/holon/use-codewarrior-stream';
import { EvolutionClient } from '../../../lib/holon/evolution-client';

interface CodeWarriorTabProps {
  context: IPEContext;
}

export const CodeWarriorTab = component$<CodeWarriorTabProps>(({ context }) => {
  // Phase 6: Live Bus Connection
  const { node, vector, logs, isLive } = useCodeWarriorStream({
    instanceId: context.instanceId,
    name: context.componentName || context.selector.split(' > ').pop() || 'Element',
  });

  const optimizing = useSignal(false);
  const godMode = useSignal(false);
  const systemThought = useSignal(SYSTEM_THOUGHTS[0]);

  // Phase 5: Easter Egg - God Mode
  useKonamiCode($(() => {
    godMode.value = true;
    node.value = { ...node.value, cmp: 100 };
    vector.value = { velocity: 100, quality: 100, stability: 100, longevity: 100 };
    logs.value = ["Ω OMEGA SEQUENCE INITIATED", ...logs.value];
  }));

  const handleMutate = $((strategyId: string) => {
    if (optimizing.value) return;
    optimizing.value = true;

    // Request real mutation via Bus
    EvolutionClient.requestMutation(node.value.id, strategyId, {
      component: context.componentName,
      selector: context.selector
    });

    // Optimistic UI update (Spinner)
    // The real 'evolution.synthesizer.patch' event will arrive via the stream
    setTimeout(() => {
      optimizing.value = false;
    }, 2000); // Timeout if no backend response
  });

  return (
    <div class={`relative space-y-4 font-mono text-xs p-4 -m-4 min-h-[500px] ${godMode.value ? 'bg-amber-900/10' : ''}`}>
      {/* Phase 5: Lattice Background */}
      <LatticeBackground />

      <div class="relative z-10 space-y-4">

        {/* Phase 2: Holonic Identity (3D Jewel) */}
        <div class="flex justify-center relative">
          <HolonJewel node={node.value} width={340} height={180} />

          {/* Phase 4: CMP Radar Overlay (Floating HUD) */}
          <div class="absolute -bottom-4 -right-2 scale-75 origin-bottom-right opacity-90 hover:opacity-100 hover:scale-100 transition-all duration-300 z-10">
            <div class="relative">
              <CMPRadar current={vector.value} cohort={COHORT_AVERAGE} size={140} />
              <div class="absolute inset-0 flex items-center justify-center pointer-events-none">
                <div class="text-center">
                  <div class={`text-[8px] ${godMode.value ? 'text-amber-400' : 'text-cyan-500/80'}`}>CMP</div>
                  <div class={`text-lg font-bold ${godMode.value ? 'text-amber-300 drop-shadow-[0_0_8px_gold]' : 'text-white drop-shadow-[0_0_5px_cyan]'}`}>
                    {node.value.cmp.toFixed(0)}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Identity Strip */}
        <div class="flex justify-between items-center px-2 text-gray-500 text-[10px] border-b border-gray-800/50 pb-2">
          <div class="flex items-center gap-2">
            <span class="tracking-widest uppercase opacity-70">{node.value.name}</span>
            {/* Live Indicator */}
            <span class={`w-1.5 h-1.5 rounded-full ${isLive.value ? 'bg-green-500 animate-pulse' : 'bg-gray-600'}`} title={isLive.value ? "Live Bus Connection" : "Simulated"} />
          </div>
          <span class={`font-bold ${godMode.value ? 'text-amber-400' : 'text-blue-400'}`}>GEN-{node.value.generation}</span>
        </div>

        {/* Phase 3: Mutation Cockpit */}
        <div class="p-3 rounded-lg bg-gray-900/40 border border-gray-700/30 backdrop-blur-sm shadow-xl">
          <div class="flex items-center justify-between mb-3">
            <div class="text-[10px] text-gray-500 uppercase tracking-wider">Evolutionary Controls</div>
            <div class="flex gap-1">
               <div class={`w-2 h-2 rounded-full ${!optimizing.value ? (godMode.value ? 'bg-amber-500' : 'bg-green-500') : 'bg-amber-500 animate-pulse'}`} />
               <span class="text-[10px] text-gray-400 capitalize">{!optimizing.value ? (godMode.value ? 'Ascended' : 'Idle') : 'Evolving'}</span>
            </div>
          </div>

          <MutationCockpit
            onMutate$={handleMutate}
            isMutating={optimizing.value}
          />
        </div>

        {/* Phase 1: Entropy Shader */}
        <div class="p-3 rounded-lg bg-gray-900/40 border border-gray-700/30 backdrop-blur-sm shadow-xl">
          <div class="flex items-center justify-between mb-2">
            <div class="text-[10px] text-gray-500 uppercase tracking-wider">System Thermodynamics</div>
            <div class={`text-[10px] ${godMode.value ? 'text-amber-400' : 'text-blue-400'}`}>Negentropic Dominance</div>
          </div>

          <EntropyCanvas
            entropy={100 - node.value.cmp}
            negentropy={node.value.cmp}
            height={64}
          />

          <div class="flex justify-between mt-1 text-[9px] font-mono opacity-60">
            <span>Chaos (S)</span>
            <span>Order (-S)</span>
          </div>
        </div>

        {/* Log Stream (New in Phase 6) */}
        {logs.value.length > 0 && (
          <div class="max-h-20 overflow-y-auto text-[9px] font-mono text-gray-500 space-y-1 p-2 bg-black/20 rounded">
            {logs.value.map((log, i) => (
              <div key={i} class="truncate">{log}</div>
            ))}
          </div>
        )}

        {/* Phase 5: Flavor Footer */}
        <div class="text-center pt-2">
          <div class="text-[9px] text-cyan-500/40 animate-pulse font-mono tracking-wider">
            {godMode.value ? "/// OMEGA POINT REACHED ///" : `> ${systemThought.value}`}
          </div>
        </div>

      </div>
    </div>
  );
});
