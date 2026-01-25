/**
 * AgentsView - Agent Status Dashboard with Chromatic Visualizer
 *
 * Step 16: Dashboard Component Integration
 * Combines the 3D Chromatic Visualizer with agent status table.
 */

import { component$, useSignal, useVisibleTask$, type Signal } from '@builder.io/qwik';
import type { AgentStatus } from '../../lib/state/types';

// ChromaticVisualizer is lazy-loaded to avoid bundling THREE.js (708KB) on initial load

interface AgentsViewProps {
  agents: Signal<AgentStatus[]>;
}

export const AgentsView = component$<AgentsViewProps>(({ agents }) => {
  const showVisualizer = useSignal(true);
  const visualizerLoaded = useSignal(false);
  const VisualizerComponent = useSignal<any>(null);

  // Lazy-load ChromaticVisualizer when the view becomes visible
  useVisibleTask$(async () => {
    if (!visualizerLoaded.value) {
      try {
        const { ChromaticVisualizer } = await import('../chromatic/ChromaticVisualizer');
        VisualizerComponent.value = ChromaticVisualizer;
        visualizerLoaded.value = true;
      } catch (err) {
        console.error('[AgentsView] Failed to load ChromaticVisualizer:', err);
      }
    }
  });

  return (
    <div class="space-y-4 glass-animate-enter">
      {/* Chromatic Agents Visualizer - Prism Metaphor */}
      {showVisualizer.value && (
        <div class="relative">
          {visualizerLoaded.value && VisualizerComponent.value ? (
            <VisualizerComponent.value
              height={500}
              enableBloom={true}
              enableControls={true}
              autoFocus={true}
            />
          ) : (
            <div class="h-[500px] glass-surface rounded-lg flex items-center justify-center">
              <div class="text-center space-y-2">
                <div class="animate-pulse text-4xl">&#x1F48E;</div>
                <div class="text-sm glass-text-muted">Loading Chromatic Visualizer...</div>
              </div>
            </div>
          )}
          <button
            onClick$={() => showVisualizer.value = false}
            class="absolute top-3 right-16 text-xs px-2 py-1 glass-chip glass-hover-glow glass-transition-hover"
          >
            Hide 3D
          </button>
        </div>
      )}

      {/* Toggle visualizer button when hidden */}
      {!showVisualizer.value && (
        <button
          onClick$={() => showVisualizer.value = true}
          class="w-full p-3 glass-surface-subtle text-sm glass-text-muted glass-hover-glow glass-transition-hover flex items-center justify-center gap-2"
        >
          <span>&#x1F48E;</span>
          Show Chromatic Visualizer
        </button>
      )}

      {/* Agent Status Table */}
      <div class="glass-surface">
        <div class="p-4 border-b border-[var(--glass-border)] flex items-center justify-between">
          <div>
            <h2 class="font-semibold glass-text-title">Agent Status</h2>
            <p class="text-sm glass-text-muted">VOR metrics and health from pluribus.check.report</p>
          </div>
          <div class="flex items-center gap-2">
            {/* Agent color legend */}
            <div class="flex items-center gap-3 text-xs">
              <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full" style={{ backgroundColor: '#FF00FF' }} />
                Claude
              </span>
              <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full" style={{ backgroundColor: '#00FFFF' }} />
                Qwen
              </span>
              <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full" style={{ backgroundColor: '#FFFF00' }} />
                Gemini
              </span>
              <span class="flex items-center gap-1">
                <span class="w-2 h-2 rounded-full" style={{ backgroundColor: '#00FF00' }} />
                Codex
              </span>
            </div>
          </div>
        </div>
        <div class="overflow-auto">
          <table class="w-full text-sm">
            <thead class="glass-surface-subtle">
              <tr>
                <th class="text-left p-3 font-medium glass-text-label">Actor</th>
                <th class="text-left p-3 font-medium glass-text-label">Status</th>
                <th class="text-left p-3 font-medium glass-text-label">Health</th>
                <th class="text-left p-3 font-medium glass-text-label">Queue</th>
                <th class="text-left p-3 font-medium glass-text-label">Task</th>
                <th class="text-left p-3 font-medium glass-text-label">VOR CDI</th>
                <th class="text-left p-3 font-medium glass-text-label">Last Seen</th>
              </tr>
            </thead>
            <tbody>
              {agents.value.length === 0 ? (
                <tr>
                  <td colSpan={7} class="p-8 text-center text-muted-foreground">
                    No agent reports received yet
                  </td>
                </tr>
              ) : (
                agents.value.map((agent) => {
                  // Determine agent color based on actor name
                  const agentColor = agent.actor.toLowerCase().includes('claude') ? '#FF00FF' :
                                     agent.actor.toLowerCase().includes('qwen') ? '#00FFFF' :
                                     agent.actor.toLowerCase().includes('gemini') ? '#FFFF00' :
                                     agent.actor.toLowerCase().includes('codex') ? '#00FF00' :
                                     undefined;

                  return (
                    <tr key={agent.actor} class="border-b border-[var(--glass-border-subtle)] glass-hover-glow glass-transition-hover">
                      <td class="p-3 font-mono flex items-center gap-2 glass-text-body">
                        {agentColor && (
                          <span
                            class="w-2 h-2 rounded-full flex-shrink-0"
                            style={{ backgroundColor: agentColor }}
                          />
                        )}
                        {agent.actor}
                      </td>
                      <td class="p-3">
                        <span class={`glass-chip ${
                          agent.status === 'idle' ? 'glass-status-ok' :
                          agent.status === 'working' ? 'glass-status-info' :
                          'glass-chip-accent-amber'
                        }`}>
                          {agent.status}
                        </span>
                      </td>
                      <td class="p-3">
                        <span class={agent.health === 'ok' ? 'glass-status-ok' : 'glass-status-warning'}>
                          {agent.health}
                        </span>
                      </td>
                      <td class="p-3 glass-text-body">{agent.queue_depth}</td>
                      <td class="p-3 glass-text-muted max-w-[200px] truncate">{agent.current_task || '-'}</td>
                      <td class="p-3 font-mono glass-text-body">{agent.vor_cdi?.toFixed(2) || '-'}</td>
                      <td class="p-3 text-xs glass-text-muted">{agent.last_seen_iso?.slice(11, 19)}</td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
});
