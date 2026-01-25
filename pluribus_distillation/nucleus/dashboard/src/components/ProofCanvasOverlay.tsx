import { component$, useSignal, $, type QRL } from '@builder.io/qwik';

// M3 Components - ProofCanvasOverlay
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

interface ProofEvidence {
  id: string;
  source: string;
  kind: 'test' | 'log' | 'metric' | 'human_verification';
  message: string;
  timestamp: string;
  status: 'passed' | 'failed' | 'theoretical';
}

interface ProofCanvasOverlayProps {
  itemId: string;
  itemName: string;
  onClose$: QRL<() => void>;
}

export const ProofCanvasOverlay = component$<ProofCanvasOverlayProps>(({ itemId, itemName, onClose$ }) => {
  const activeTab = useSignal<'lattice' | 'evidence' | 'gaps'>('lattice');

  // Simulated evidence data
  const evidences: ProofEvidence[] = [
    { id: 'ev-1', source: 'PBTEST', kind: 'test', message: 'E2E verification of core loops', timestamp: '2025-12-28T01:00Z', status: 'passed' },
    { id: 'ev-2', source: 'Replisome', kind: 'log', message: 'Signed manifest detected in Rhizome', timestamp: '2025-12-28T00:45Z', status: 'passed' },
    { id: 'ev-3', source: 'Gemini SAGENT', kind: 'human_verification', message: 'Architecture review completed', timestamp: '2025-12-27T23:30Z', status: 'passed' },
    { id: 'ev-4', source: 'InferCell', kind: 'theoretical', message: 'Projected scalability in high-concurrency swarms', timestamp: '2025-12-28T02:00Z', status: 'theoretical' },
  ];

  return (
    <div class="fixed inset-0 z-[150] flex items-center justify-center bg-black/80 backdrop-blur-xl p-8">
      <div class="glass-surface-overlay bg-card/95 border border-[var(--glass-border)] rounded-xl w-full max-w-5xl h-full max-h-[800px] flex flex-col shadow-2xl overflow-hidden">
        {/* Header */}
        <div class="p-6 border-b border-[var(--glass-border)] flex items-center justify-between bg-muted/5">
          <div>
            <div class="text-[10px] text-primary font-mono tracking-widest uppercase">Proof Canvas</div>
            <h2 class="text-2xl font-bold tracking-tight">{itemName} <span class="text-muted-foreground font-mono text-sm opacity-50">[{itemId}]</span></h2>
          </div>
          <button onClick$={onClose$} class="p-2 rounded-full hover:bg-muted transition-colors text-muted-foreground hover:text-foreground">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
          </button>
        </div>

        {/* Tab Navigation */}
        <div class="px-6 py-2 border-b border-[var(--glass-border-subtle)] flex gap-4 text-xs font-mono">
          <button 
            onClick$={() => activeTab.value = 'lattice'}
            class={`pb-2 transition-all border-b-2 ${activeTab.value === 'lattice' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground'}`}
          >
            Proof Lattice (DAG)
          </button>
          <button 
            onClick$={() => activeTab.value = 'evidence'}
            class={`pb-2 transition-all border-b-2 ${activeTab.value === 'evidence' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground'}`}
          >
            Evidence Log ({evidences.length})
          </button>
          <button 
            onClick$={() => activeTab.value = 'gaps'}
            class={`pb-2 transition-all border-b-2 ${activeTab.value === 'gaps' ? 'border-primary text-primary' : 'border-transparent text-muted-foreground'}`}
          >
            Epistemic Gaps
          </button>
        </div>

        {/* Content */}
        <div class="flex-1 overflow-hidden flex">
          {/* Main Area */}
          <main class="flex-1 overflow-y-auto p-6">
            {activeTab.value === 'lattice' && (
              <div class="h-full flex items-center justify-center relative">
                <div class="text-center">
                  <div class="text-4xl mb-4">🕸️</div>
                  <div class="text-sm font-mono text-muted-foreground">Lattice Rendering Engine (WIP)</div>
                  <div class="text-[10px] text-muted-foreground mt-2 opacity-50">Visualizing causal dependencies between evidence nodes</div>
                </div>
                {/* Simulated DAG nodes */}
                <div class="absolute inset-0 pointer-events-none opacity-20">
                   <div class="absolute top-1/4 left-1/4 w-32 h-32 border border-primary rounded-full animate-pulse" />
                   <div class="absolute bottom-1/4 right-1/4 w-48 h-48 border border-cyan-500 rounded-full animate-pulse [animation-delay:1s]" />
                </div>
              </div>
            )}

            {activeTab.value === 'evidence' && (
              <div class="space-y-4">
                {evidences.map(ev => (
                  <div key={ev.id} class="p-4 rounded-lg bg-muted/10 border border-border/30 flex gap-4 items-start hover:bg-muted/20 transition-colors">
                    <div class={`mt-1 w-2 h-2 rounded-full ${ev.status === 'passed' ? 'bg-green-500' : 'bg-blue-500 animate-pulse'}`} />
                    <div class="flex-1">
                      <div class="flex items-center justify-between mb-1">
                        <span class="text-[10px] font-mono text-primary">{ev.source}</span>
                        <span class="text-[10px] text-muted-foreground">{new Date(ev.timestamp).toLocaleString()}</span>
                      </div>
                      <div class="text-sm font-medium mb-1">{ev.message}</div>
                      <div class="text-[10px] px-1.5 py-0.5 rounded bg-black/40 inline-block text-muted-foreground border border-[var(--glass-border-subtle)]">{ev.kind.toUpperCase()}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {activeTab.value === 'gaps' && (
              <div class="p-8 border-2 border-dashed border-amber-500/20 rounded-xl bg-amber-500/5 text-center">
                <div class="text-3xl mb-4">🕳️</div>
                <h3 class="text-lg font-bold text-amber-400 mb-2">Detected Epistemic Gaps</h3>
                <p class="text-sm text-muted-foreground max-w-md mx-auto mb-6">
                  The following areas lack sufficient hard evidence and rely on theoretical projections or parent lineage assumptions.
                </p>
                <ul class="text-left text-xs font-mono space-y-2 max-w-lg mx-auto">
                  <li class="p-2 bg-black/40 rounded border border-amber-500/20 text-amber-200/70">
                    ⚠️ CONTEXT_DRIFT: No local verification of SOTA integration latency.
                  </li>
                  <li class="p-2 bg-black/40 rounded border border-amber-500/20 text-amber-200/70">
                    ⚠️ RESOURCE_SHADOW: Actual token consumption metrics are 15m stale.
                  </li>
                </ul>
              </div>
            )}
          </main>

          {/* Sidebar */}
          <aside class="w-80 border-l border-border/50 p-6 bg-muted/5 flex flex-col gap-6 overflow-y-auto">
            <section>
              <h4 class="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-3">Overall Confidence</h4>
              <div class="flex items-end gap-2 mb-2">
                <div class="text-4xl font-bold text-primary">92%</div>
                <div class="text-xs text-green-500 mb-1">↑ 2.4%</div>
              </div>
              <div class="w-full bg-muted rounded-full h-1.5">
                <div class="bg-primary h-full rounded-full w-[92%]" />
              </div>
            </section>

            <section>
              <h4 class="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-3">Actions</h4>
              <div class="flex flex-col gap-2">
                <button class="w-full py-2 px-4 rounded bg-primary text-primary-foreground text-xs font-bold hover:opacity-90 transition-opacity">
                  FORCE VERIFICATION
                </button>
                <button class="w-full py-2 px-4 rounded border border-border text-xs font-bold hover:bg-muted transition-colors">
                  EXPORT RHIZOME PROOF
                </button>
              </div>
            </section>

            <section>
              <h4 class="text-[10px] font-mono text-muted-foreground uppercase tracking-widest mb-3">Lineage</h4>
              <div class="text-xs font-mono space-y-2">
                <div class="flex justify-between">
                  <span class="text-muted-foreground">Origin</span>
                  <span class="text-foreground">Sextet v1.2</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-muted-foreground">Gen</span>
                  <span class="text-foreground">42</span>
                </div>
                <div class="flex justify-between">
                  <span class="text-muted-foreground">HGT Splice</span>
                  <span class="text-cyan-400">active</span>
                </div>
              </div>
            </section>
          </aside>
        </div>
      </div>
    </div>
  );
});
