/**
 * CMPCladesWidget - Visualizes Agent Lineage (Clade) Metaproductivity
 * 
 * Subscribes to:
 * - cmp.lineage.update
 * - cmp.hgt.decision
 */

import { component$, useComputed$ } from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';

export interface CMPCladesWidgetProps {
  events: BusEvent[];
}

export const CMPCladesWidget = component$((props: CMPCladesWidgetProps) => {
  const clades = useComputed$(() => {
    const map = new Map<string, any>();
    
    // Process backwards to get latest state
    for (let i = props.events.length - 1; i >= 0; i--) {
      const e = props.events[i];
      if (e.topic === 'cmp.lineage.update') {
        const id = e.data.lineage_id;
        if (!map.has(id)) {
          map.set(id, {
            id,
            score: e.data.cmp_score,
            uy: e.data.u_y,
            parentId: e.data.parent_id,
            lastUpdate: e.ts,
            hgtCount: 0
          });
        }
      } else if (e.topic === 'cmp.hgt.decision' && e.data.decision === 'approve') {
        const id = e.data.target_lineage_id;
        const entry = map.get(id);
        if (entry) {
          entry.hgtCount = (entry.hgtCount || 0) + 1;
        }
      }
    }
    
    return Array.from(map.values()).sort((a, b) => b.score - a.score);
  });

  return (
    <div class="cmp-clades-widget p-4 bg-black/40 border border-cyan-900/50 rounded-lg text-xs font-mono">
      <div class="flex justify-between items-center mb-3 border-b border-cyan-900/30 pb-2">
        <h3 class="text-cyan-400 uppercase tracking-widest font-bold">Clade Metaproductivity (CMP)</h3>
        <span class="text-[10px] text-cyan-700">AGENT ALPHA</span>
      </div>
      
      <div class="space-y-2 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
        {clades.value.length === 0 && (
          <div class="text-cyan-900 italic py-4 text-center">Waiting for lineage signals...</div>
        )}
        
        {clades.value.map((c) => (
          <div key={c.id} class="bg-cyan-950/20 border border-cyan-900/20 p-2 rounded flex flex-col gap-1">
            <div class="flex justify-between items-start">
              <span class="text-cyan-100 truncate w-32" title={c.id}>{c.id}</span>
              <span class="text-cyan-400 font-bold">{(c.score * 100).toFixed(1)}%</span>
            </div>
            
            <div class="w-full bg-cyan-900/30 h-1.5 rounded-full overflow-hidden">
              <div 
                class="h-full bg-gradient-to-r from-cyan-600 to-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.5)]" 
                style={{ width: `${Math.min(100, c.score * 100)}%` }}
              ></div>
            </div>
            
            <div class="flex justify-between text-[10px] text-cyan-600">
              <span>U(Y): {c.uy?.toFixed(3) || '0.000'}</span>
              <span>{c.hgtCount > 0 ? `🧬 HGT: ${c.hgtCount}` : ''}</span>
              <span>{c.parentId ? `↑ ${c.parentId.slice(0, 8)}` : 'TRUNK'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});
