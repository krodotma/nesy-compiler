import { component$, type QRL } from '@builder.io/qwik';
import { InferCellCard, InferCellGrid, type ModuleInfo, type InferCellSession, type InferCellStatus, type LiveModuleData } from '../InferCellCard';

interface DistillViewProps {
  goldenModules: ModuleInfo[];
  subsystemModules: ModuleInfo[];
  inferCellStatuses: Record<string, InferCellStatus>;
  inferCellSessions: Record<string, InferCellSession>;
  inferCellLiveData: Record<string, LiveModuleData>;
  handleInferCellAction: QRL<(action: string, module: ModuleInfo) => Promise<void>>;
}

export const DistillView = component$<DistillViewProps>((props) => {
  const {
    goldenModules,
    subsystemModules,
    inferCellStatuses,
    inferCellSessions,
    inferCellLiveData,
    handleInferCellAction
  } = props;

  return (
    <div class="space-y-6">
      {/* Step 99: glass-surface for Header Stats with glass-chip accents */}
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="glass-surface rounded-xl p-4 border border-green-500/30">
          <div class="text-3xl font-bold text-green-400">10/10</div>
          <div class="text-sm text-muted-foreground">Generations Complete</div>
          <div class="glass-progress-bar mt-2 h-1 rounded-full overflow-hidden">
            <div class="h-full bg-green-500 w-full"></div>
          </div>
        </div>
        <div class="glass-surface rounded-xl p-4 border border-blue-500/30">
          <div class="text-3xl font-bold text-blue-400">102</div>
          <div class="text-sm text-muted-foreground">BEAM Entries</div>
          <div class="glass-progress-bar mt-2 h-1 rounded-full overflow-hidden">
            <div class="h-full bg-blue-500 w-[85%]"></div>
          </div>
        </div>
        <div class="glass-surface rounded-xl p-4 border border-purple-500/30">
          <div class="text-3xl font-bold text-purple-400">651</div>
          <div class="text-sm text-muted-foreground">GOLDEN Lines</div>
          <div class="glass-progress-bar mt-2 h-1 rounded-full overflow-hidden">
            <div class="h-full bg-purple-500 w-[75%]"></div>
          </div>
        </div>
        <div class="glass-surface rounded-xl p-4 border border-yellow-500/30">
          <div class="text-3xl font-bold text-yellow-400">92%</div>
          <div class="text-sm text-muted-foreground">Verified (93/101)</div>
          {/* Step 101: glass-progress-bar for verification status */}
          <div class="glass-progress-bar mt-2 h-1 rounded-full overflow-hidden">
            <div class="h-full bg-yellow-500 w-[92%]"></div>
          </div>
        </div>
      </div>

      {/* Step 99: glass-surface-elevated for GOLDEN Synthesis Grid */}
      <div class="glass-surface-elevated rounded-xl">
        <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
          <span class="text-2xl">🏆</span>
          <h3 class="font-semibold">GOLDEN Synthesis (G1-G10)</h3>
          {/* Step 100: glass-chip for status indicators */}
          <span class="glass-chip glass-chip-accent-emerald text-xs">INTERACTIVE</span>
          <span class="text-xs text-muted-foreground ml-auto">Click to expand • Trigger actions</span>
        </div>
        <div class="p-4 space-y-2">
          {goldenModules.map((mod) => (
            <InferCellCard
              key={mod.goldenId}
              module={mod}
              status={inferCellStatuses[mod.name] || 'idle'}
              session={inferCellSessions[mod.name]}
              liveData={inferCellLiveData[mod.name]}
              onTrigger$={handleInferCellAction}
              compact={false}
            />
          ))}
        </div>
      </div>

      {/* Step 99: glass-surface for Subsystem Verification Matrix */}
      <div class="glass-surface rounded-xl">
        <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
          <span class="text-2xl">✅</span>
          <h3 class="font-semibold">Subsystem Verification Matrix</h3>
          {/* Step 100: glass-chip for verification count */}
          <span class="glass-chip glass-chip-accent-emerald text-xs">
            {Object.values(inferCellStatuses).filter(s => s === 'ok').length}/{subsystemModules.length} VERIFIED
          </span>
          {/* Step 97: glass-interactive for action button */}
          <button
            class="glass-chip glass-chip-accent-cyan text-xs ml-auto hover:scale-105 glass-transition-standard"
            onClick$={async () => {
              // Verify all subsystems
              for (const mod of subsystemModules) {
                await handleInferCellAction('verify', mod);
              }
            }}
          >
            🔄 Verify All
          </button>
        </div>
        <div class="p-4">
          <InferCellGrid
            modules={subsystemModules}
            statuses={inferCellStatuses}
            sessions={inferCellSessions}
            liveData={inferCellLiveData}
            onTrigger$={handleInferCellAction}
            compact={true}
            columns={5}
          />
        </div>
      </div>

      {/* Step 99: glass-surface for Lens/Collimator Status */}
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-surface rounded-xl">
          <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
            <span class="text-2xl">🔍</span>
            <h3 class="font-semibold">Lens/Collimator Routing</h3>
          </div>
          <div class="p-4 space-y-3">
            {/* Step 100: glass-chip for routing options */}
            <div class="glass-surface-subtle flex items-center justify-between p-2 rounded-lg">
              <span class="text-sm">Depth Classification</span>
              <span class="glass-chip glass-chip-accent-cyan text-xs">narrow | deep</span>
            </div>
            <div class="glass-surface-subtle flex items-center justify-between p-2 rounded-lg">
              <span class="text-sm">Lane Selection</span>
              <span class="glass-chip glass-chip-accent-purple text-xs">dialogos | pbpair</span>
            </div>
            <div class="glass-surface-subtle flex items-center justify-between p-2 rounded-lg">
              <span class="text-sm">Context Mode</span>
              <span class="glass-chip glass-chip-accent-emerald text-xs">min | lite | full</span>
            </div>
            <div class="glass-surface-subtle flex items-center justify-between p-2 rounded-lg">
              <span class="text-sm">Topology</span>
              <span class="glass-chip glass-chip-accent-amber text-xs">single | star | peer_debate</span>
            </div>
          </div>
        </div>

        {/* Step 99: glass-surface for Protocol Evolution panel */}
        <div class="glass-surface rounded-xl">
          <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
            <span class="text-2xl">📋</span>
            <h3 class="font-semibold">DKIN Protocol v12</h3>
          </div>
          <div class="p-4 space-y-2 text-sm">
            {/* Step 100: glass-chip style for version indicators */}
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v1</span><span>Minimal check-in</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v2</span><span>Drift guards</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v3</span><span>Enhanced dashboard</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v4</span><span>ITERATE operator</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v5</span><span>Silent monitoring</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v6</span><span>Shared-ledger sync</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v7</span><span>Gap detection</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v8</span><span>OITERATE omega-loop</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v9</span><span>DKIN alias established</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v10</span><span>Intelligent membrane</span></div>
            <div class="flex gap-2"><span class="glass-chip text-[10px] w-8 text-center">v11</span><span>MCP interop observability</span></div>
            <div class="flex gap-2"><span class="glass-chip glass-chip-accent-cyan text-[10px] w-8 text-center font-medium">v12</span><span class="text-primary font-medium">VLM solver observability</span></div>
            <div class="flex gap-2"><span class="glass-chip glass-chip-accent-amber text-[10px] w-10 text-center">draft</span><span>Parallel agent isolation (PAIP)</span></div>
          </div>
        </div>
      </div>

      {/* Step 99: glass-surface-elevated for BEAM Discourse Summary */}
      <div class="glass-surface-elevated rounded-xl">
        <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
          <span class="text-2xl">📝</span>
          <h3 class="font-semibold">BEAM Discourse Summary</h3>
          <span class="glass-chip text-xs">102 entries across 10 iterations</span>
        </div>
        <div class="p-4">
          {/* Step 100: glass-chip for step indicators */}
          <div class="grid grid-cols-5 md:grid-cols-10 gap-2">
            {[1,2,3,4,5,6,7,8,9,10].map((i) => (
              <div key={i} class="glass-surface-subtle glass-interactive rounded-lg p-2 text-center glass-transition-standard hover:scale-105">
                <div class="text-xs text-muted-foreground">Iter</div>
                <div class="text-lg font-bold text-primary">{i}</div>
                <div class="glass-chip glass-chip-accent-emerald text-[10px] mt-1">~10</div>
              </div>
            ))}
          </div>
          {/* Step 100: glass-chip for legend */}
          <div class="mt-4 flex flex-wrap gap-4 text-sm">
            <div class="flex items-center gap-2">
              <span class="glass-chip glass-chip-accent-emerald w-6 h-3"></span>
              <span>V - Verified</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="glass-chip glass-chip-accent-cyan w-6 h-3"></span>
              <span>R - Reported</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="glass-chip glass-chip-accent-purple w-6 h-3"></span>
              <span>I - Intent</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="glass-chip glass-chip-accent-amber w-6 h-3"></span>
              <span>G - Gap</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});
