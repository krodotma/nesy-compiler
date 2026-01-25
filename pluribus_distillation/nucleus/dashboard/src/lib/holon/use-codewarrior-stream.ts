/**
 * useCodeWarriorStream
 * [Ultrathink Agent 1: Architect]
 * 
 * A Qwik hook that binds the CodeWarrior UI to the live system bus.
 * Handles:
 * - Real-time CMP updates (`cmp.lineage.update`)
 * - Evolution proposals (`evolution.refiner.proposal`)
 * - Thermodynamic state (`cmp.entropy.weighted`)
 * 
 * Includes "Graceful Degradation" to simulation mode if backend is silent.
 */

import { useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { createBusClient } from '../bus/bus-client';
import type { HolonNode } from './types';
import type { CMPVector } from './cmp';
import { calculateAggregateCMP, COHORT_AVERAGE } from './cmp';

export interface CodeWarriorState {
  node: HolonNode;
  vector: CMPVector;
  logs: string[];
  isLive: boolean; // True if receiving real bus events
}

export function useCodeWarriorStream(initialContext: { instanceId: string, name: string }) {
  // State
  const node = useSignal<HolonNode>({
    id: initialContext.instanceId,
    name: initialContext.name,
    type: 'branch',
    cmp: 0,
    complexity: 45,
    stability: 50,
    generation: 1,
    parents: [],
    children: [],
    etymon: 'INIT-' + initialContext.instanceId.slice(0, 6).toUpperCase(),
    tags: ['loading'],
  });

  const vector = useSignal<CMPVector>({
    velocity: 0,
    quality: 0,
    stability: 0,
    longevity: 0,
  });

  const logs = useSignal<string[]>([]);
  const isLive = useSignal(false);
  const lastEventTime = useSignal(0);

  // Connect to Bus
  useVisibleTask$(({ cleanup }) => {
    const client = createBusClient({ platform: 'browser' });
    let cleanupSubs: (() => void)[] = [];

    const connect = async () => {
      try {
        await client.connect();
        
        // CMP Updates
        const sub1 = client.subscribe('cmp.lineage.update', (e) => {
          isLive.value = true;
          lastEventTime.value = Date.now();
          
          if (e.data) {
            const d = e.data;
            // Update Vector if present
            if (d.vector) {
              vector.value = d.vector;
            } else if (d.reward) {
              // Synthetic vector from scalar reward
              const val = Math.min(100, d.reward * 100);
              vector.value = { velocity: val, quality: val, stability: val, longevity: val };
            }
            
            // Update Node
            node.value = {
              ...node.value,
              cmp: calculateAggregateCMP(vector.value),
              generation: d.generation || node.value.generation,
              etymon: d.etymon || node.value.etymon
            };
          }
        });

        // Evolution Proposals
        const sub2 = client.subscribe('evolution.refiner.proposal', (e) => {
          isLive.value = true;
          lastEventTime.value = Date.now();
          
          const batch = e.data;
          const msg = `[PROPOSAL] ${batch.theme || 'optimization'} (${batch.proposal_count} items)`;
          logs.value = [msg, ...logs.value.slice(0, 49)];
        });

        // Evolution Status
        const sub3 = client.subscribe('evolution.synthesizer.patch', (e) => {
          isLive.value = true;
          lastEventTime.value = Date.now();
          const msg = `[PATCH] Applied to ${e.data?.target || 'unknown'}`;
          logs.value = [msg, ...logs.value.slice(0, 49)];
        });

        cleanupSubs.push(sub1, sub2, sub3);

      } catch (err) {
        console.warn('[CodeWarrior] Bus connection failed, falling back to sim', err);
      }
    };

    connect();

    // Simulation Fallback Loop (The "Dream State")
    // If no events received for 3s, gently animate to keep UI alive
    const simTimer = setInterval(() => {
      const silence = Date.now() - lastEventTime.value;
      
      if (!isLive.value || silence > 3000) {
        // Drifting simulation
        const drift = (Math.random() - 0.5) * 2;
        vector.value = {
          velocity: Math.max(0, Math.min(100, vector.value.velocity + drift)),
          quality: Math.max(0, Math.min(100, vector.value.quality + drift)),
          stability: Math.max(0, Math.min(100, vector.value.stability + drift)),
          longevity: Math.max(0, Math.min(100, vector.value.longevity + drift)),
        };
        node.value = { 
          ...node.value, 
          cmp: calculateAggregateCMP(vector.value),
          complexity: node.value.complexity + (Math.random() > 0.9 ? 1 : 0) // Occasional complexity growth
        };
      }
    }, 100);

    cleanup(() => {
      cleanupSubs.forEach(unsub => unsub());
      client.disconnect();
      clearInterval(simTimer);
    });
  });

  return { node, vector, logs, isLive };
}
