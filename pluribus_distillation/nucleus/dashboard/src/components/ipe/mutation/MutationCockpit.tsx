/**
 * MutationCockpit.tsx
 * [Ultrathink Team]
 * 
 * The control surface for Evolutionary Code Operations (ECO).
 * Replaces the simple "Mutate" button with a comprehensive strategy selector
 * and execution environment.
 * 
 * Features:
 * - Strategy selection card (Cards vs List).
 * - Safety Latch integration.
 * - Risk/Reward visualization.
 */

import {
  component$,
  useSignal,
  $,
} from '@builder.io/qwik';
import { MUTATION_STRATEGIES, type MutationStrategy } from '../../../lib/holon/mutation';
import { SafetyLatch } from './SafetyLatch';

interface MutationCockpitProps {
  onMutate$: (strategyId: string) => void;
  isMutating: boolean;
}

export const MutationCockpit = component$<MutationCockpitProps>(({ onMutate$, isMutating }) => {
  const selectedStrategyId = useSignal<string | null>(null);
  const isArmed = useSignal(false);

  const handleArm = $(() => isArmed.value = true);
  const handleDisarm = $(() => isArmed.value = false);

  const handleExecute = $(() => {
    if (selectedStrategyId.value && isArmed.value) {
      onMutate$(selectedStrategyId.value);
      // Auto-disarm after firing
      isArmed.value = false;
    }
  });

  const selectedStrategy = MUTATION_STRATEGIES.find(s => s.id === selectedStrategyId.value);

  return (
    <div class="flex flex-col gap-4">
      {/* Strategy Grid */}
      <div class="grid grid-cols-2 gap-2">
        {MUTATION_STRATEGIES.map(strategy => (
          <div
            key={strategy.id}
            onClick$={() => { if(!isMutating) selectedStrategyId.value = strategy.id; }}
            class={`
              relative p-2 rounded border cursor-pointer transition-all duration-200
              ${selectedStrategyId.value === strategy.id 
                ? 'bg-blue-900/30 border-blue-500 ring-1 ring-blue-500/50' 
                : 'bg-gray-800/30 border-gray-700 hover:bg-gray-700/50'
              }
              ${isMutating ? 'opacity-50 pointer-events-none' : ''}
            `}
          >
            {/* Selection Indicator */}
            {selectedStrategyId.value === strategy.id && (
              <div class="absolute -top-1 -right-1 w-2 h-2 bg-blue-500 rounded-full shadow-[0_0_8px_rgba(59,130,246,0.8)]" />
            )}

            <div class="flex justify-between items-start mb-1">
              <div class="text-[10px] font-bold text-gray-200">{strategy.label}</div>
              <div class={`
                text-[8px] font-mono px-1 rounded
                ${strategy.risk === 'low' ? 'text-green-400 bg-green-900/30' : 
                  strategy.risk === 'medium' ? 'text-yellow-400 bg-yellow-900/30' : 
                  strategy.risk === 'high' ? 'text-orange-400 bg-orange-900/30' : 
                  'text-red-400 bg-red-900/30'}
              `}>
                {strategy.risk.toUpperCase()}
              </div>
            </div>
            
            <div class="text-[9px] text-gray-500 leading-tight mb-2">
              {strategy.description}
            </div>

            {/* Micro-metrics */}
            <div class="flex items-center gap-2 text-[8px] font-mono opacity-80">
              <span class="flex items-center gap-0.5 text-cyan-300">
                <span>⚡</span> {strategy.energyCost}
              </span>
              <span class="flex items-center gap-0.5 text-green-300">
                <span>▲</span> CMP+{strategy.cmpPotential}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Execution Controls */}
      <div class="mt-2 border-t border-gray-700/50 pt-4 flex flex-col items-center">
        {selectedStrategyId.value ? (
          <div class="w-full flex flex-col items-center gap-3 animate-in fade-in slide-in-from-bottom-2">
            
            {/* Contextual Warning */}
            <div class="text-[9px] text-center text-gray-400">
              Target: <span class="text-white font-bold">{selectedStrategy?.label}</span>
              <span class="mx-2">•</span>
              Predicted CMP: <span class="text-green-400">+{selectedStrategy?.cmpPotential}</span>
            </div>

            {/* The Big Switch */}
            <SafetyLatch 
              isArmed={isArmed.value} 
              onArm$={handleArm} 
              onDisarm$={handleDisarm} 
            />

            {/* Fire Button (Only active when armed) */}
            <button
              onClick$={handleExecute}
              disabled={!isArmed.value || isMutating}
              class={`
                w-full py-2 rounded text-xs font-bold font-mono tracking-widest transition-all duration-200
                ${isArmed.value 
                  ? 'bg-red-600 text-white shadow-[0_0_20px_rgba(220,38,38,0.4)] hover:bg-red-500 transform scale-105' 
                  : 'bg-gray-800 text-gray-600 cursor-not-allowed border border-gray-700'}
              `}
            >
              {isMutating ? 'MUTATION IN PROGRESS...' : isArmed.value ? 'INITIATE EVOLUTION' : 'SAFETY LOCKED'}
            </button>
          </div>
        ) : (
          <div class="text-[10px] text-gray-500 italic py-2">
            Select a mutation strategy to initialize sequence.
          </div>
        )}
      </div>
    </div>
  );
});
