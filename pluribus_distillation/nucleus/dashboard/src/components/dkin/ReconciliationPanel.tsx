import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

// M3 Components - ReconciliationPanel
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';

interface ReconcileStats {
  orphans: number;
  patches: number;
  collisions: number;
  archives: number;
  lastScan: string;
}

export const ReconciliationPanel = component$(() => {
  const stats = useSignal<ReconcileStats>({
    orphans: 0,
    patches: 0,
    collisions: 0,
    archives: 0,
    lastScan: 'Pending...',
  });

  const isScanning = useSignal(false);

  // Poll for reconciliation state (mocked for now, or via shadow worker)
  useVisibleTask$(({ cleanup }) => {
    const channel = new BroadcastChannel('pluribus-omega');
    
    // Listen for live reconcile events
    channel.onmessage = (ev) => {
        if (ev.data.type === 'BUS_EVENT') {
            const topic = ev.data.event.topic;
            if (topic === 'reconcile.orphan.scanned') {
                stats.value.orphans = ev.data.event.data.count;
                stats.value.lastScan = new Date().toLocaleTimeString();
                isScanning.value = false;
            }
            if (ev.data.event.data?.reconcile_stats) {
                // If a heartbeat sends full stats
                const s = ev.data.event.data.reconcile_stats;
                stats.value = { ...stats.value, ...s };
            }
        }
    };

    // Initial fetch (via API if available, else relying on push)
    // For now we simulate the 'scanned' state from the CLI tool interaction we just saw
    stats.value.orphans = 1; 
    stats.value.patches = 1;
    stats.value.lastScan = new Date().toLocaleTimeString();

    cleanup(() => channel.close());
  });

  const triggerScan = $(async () => {
    isScanning.value = true;
    // In a real implementation, this would POST to /api/rpc/reconcile/scan
    // For now, we assume the operator is running or triggered via bus
    await fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            topic: 'cmd.reconcile.scan',
            kind: 'request',
            actor: 'dashboard-ui',
            data: {}
        })
    });
    
    // Simulate response delay
    setTimeout(() => isScanning.value = false, 2000);
  });

  return (
    <div class="p-4 border border-border rounded-lg bg-black/20 backdrop-blur-sm relative overflow-hidden">
      {/* v21 Badge */}
      <div class="absolute top-0 right-0 px-2 py-0.5 bg-purple-900/50 text-purple-200 text-[10px] font-mono border-l border-b border-purple-500/30 rounded-bl-lg">
        DKIN v21
      </div>

      <h3 class="text-sm font-semibold text-purple-400 mb-3 flex items-center gap-2">
        <span class="inline-block w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
        Lossless Reconciliation
      </h3>

      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div class="text-center p-2 rounded bg-white/5 border border-[var(--glass-border-subtle)]">
          <div class="text-xs text-muted-foreground uppercase tracking-wider">Orphans</div>
          <div class={`text-xl font-mono ${stats.value.orphans > 0 ? 'text-amber-400' : 'text-gray-400'}`}>
            {stats.value.orphans}
          </div>
        </div>
        <div class="text-center p-2 rounded bg-white/5 border border-[var(--glass-border-subtle)]">
          <div class="text-xs text-muted-foreground uppercase tracking-wider">Patches</div>
          <div class="text-xl font-mono text-cyan-400">
            {stats.value.patches}
          </div>
        </div>
        <div class="text-center p-2 rounded bg-white/5 border border-[var(--glass-border-subtle)]">
          <div class="text-xs text-muted-foreground uppercase tracking-wider">Collisions</div>
          <div class={`text-xl font-mono ${stats.value.collisions > 0 ? 'text-red-500' : 'text-green-400'}`}>
            {stats.value.collisions}
          </div>
        </div>
        <div class="text-center p-2 rounded bg-white/5 border border-[var(--glass-border-subtle)]">
          <div class="text-xs text-muted-foreground uppercase tracking-wider">Archives</div>
          <div class="text-xl font-mono text-gray-400">
            {stats.value.archives}
          </div>
        </div>
      </div>

      <div class="flex items-center justify-between">
        <div class="text-[10px] text-muted-foreground font-mono">
          Last Scan: {stats.value.lastScan}
        </div>
        <button
          onClick$={triggerScan}
          disabled={isScanning.value}
          class={`
            text-xs px-3 py-1.5 rounded transition-all flex items-center gap-2
            ${isScanning.value 
              ? 'bg-purple-500/20 text-purple-300 cursor-wait' 
              : 'bg-purple-600 hover:bg-purple-500 text-white shadow-[0_0_10px_rgba(168,85,247,0.4)]'
            }
          `}
        >
          {isScanning.value ? 'Scanning /tmp...' : 'Scan & Recover'}
        </button>
      </div>
      
      {/* Active Orphan Alert */}
      {stats.value.orphans > 0 && (
          <div class="mt-3 p-2 bg-amber-900/20 border border-amber-500/30 rounded text-xs text-amber-200 flex items-center gap-2">
              <span class="text-lg">⚠️</span>
              <div>
                  <strong>Uncommitted Work Detected:</strong> {stats.value.orphans} session(s) recovered.
                  <div class="opacity-70 text-[10px]">codex-1766356598 (1612 lines preserved)</div>
              </div>
          </div>
      )}
    </div>
  );
});
