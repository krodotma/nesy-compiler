/**
 * Feature Discovery (The Guide)
 * Author: gemini_interaction_1
 * Context: Phase 4 Evolution
 * 
 * Renders "Empty State" hints to teach the user what the widget can do.
 * "Don't just show a blank box."
 */

import { component$ } from '@builder.io/qwik';

export const FeatureDiscovery = component$(() => {
  return (
    <div class="p-8 grid grid-cols-2 gap-4 opacity-70">
      <div class="glass-panel p-4 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group">
        <div class="text-[var(--glass-accent-cyan)] mb-2 group-hover:scale-110 transition-transform">🛠️</div>
        <div class="font-bold text-white/90 text-sm">Refactor Code</div>
        <div class="text-xs text-white/50">"/fix memory leak in main.ts"</div>
      </div>
      <div class="glass-panel p-4 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group">
        <div class="text-[var(--glass-accent-magenta)] mb-2 group-hover:scale-110 transition-transform">📅</div>
        <div class="font-bold text-white/90 text-sm">Plan Task</div>
        <div class="text-xs text-white/50">"/task Create new migration"</div>
      </div>
      <div class="glass-panel p-4 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group">
        <div class="text-[var(--chroma-warning)] mb-2 group-hover:scale-110 transition-transform">🧠</div>
        <div class="font-bold text-white/90 text-sm">Deep Research</div>
        <div class="text-xs text-white/50">"Analyze SOTA for RAG"</div>
      </div>
      <div class="glass-panel p-4 rounded-xl hover:bg-white/5 transition-colors cursor-pointer group">
        <div class="text-[var(--chroma-success)] mb-2 group-hover:scale-110 transition-transform">📊</div>
        <div class="font-bold text-white/90 text-sm">System Status</div>
        <div class="text-xs text-white/50">"Check bus health"</div>
      </div>
    </div>
  );
});
