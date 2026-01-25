import { component$, type Signal, type QRL, type PropFunction } from '@builder.io/qwik';
import type { VPSSession, BusEvent, ServiceDef } from '../../lib/state/types';
import type { ActionCell } from '../../lib/actions/types';
import { SuperMotd } from '../supermotd';
import { AgentTelemetryPanel } from '../AgentTelemetryPanel';
import { LazyWebLLM } from '../LazyWebLLM';
import { EdgeInferenceStatusWidget } from '../EdgeInferenceStatusWidget';
import { EdgeInferenceCatalog } from '../EdgeInferenceCatalog';
import { DialogosContainer } from '../DialogosContainer';
import { NotificationSidepanel } from '../NotificationSidepanel';
import { TimelineSparkline, EventStatsBadges, EnrichedEventCard, EventFlowmap } from '../EventVisualization';

interface HomeViewProps {
  session: Signal<VPSSession>;
  activeView: Signal<string>;
  workerCount: Signal<number>;
  connected: Signal<boolean>;
  events: Signal<BusEvent[]>;
  filteredEvents: Signal<BusEvent[]>;
  actionCells: ActionCell[]; // Store, so treated as array/object
  providerOptions: Signal<string[]>;
  selectedProviders: Signal<string[]>;
  commandInput: Signal<string>;
  providersList: Signal<[string, any][]>;
  eventFilter: Signal<string | null>;
  
  dispatchAction: QRL<(type: string, payload: Record<string, unknown>) => void>;
  setFlowMode: QRL<(mode: 'm' | 'A') => void>;
  toggleProvider: QRL<(provider: string) => void>;
  emitBus: QRL<(topic: string, kind: string, data: Record<string, unknown>) => Promise<void>>;
  cycleFilter: QRL<() => void>;
}

export const HomeView = component$<HomeViewProps>((props) => {
  const {
    session,
    activeView,
    workerCount,
    connected,
    events,
    filteredEvents,
    actionCells,
    providerOptions,
    selectedProviders,
    commandInput,
    providersList,
    eventFilter,
    dispatchAction,
    setFlowMode,
    toggleProvider,
    emitBus,
    cycleFilter
  } = props;

  return (
    <div class="space-y-6 glass-panel p-6">
      <div class="grid gap-6 lg:grid-cols-3">
        {/* Left Column - Controls */}
        <div class="space-y-6">
          {/* Flow Mode */}
          <div class="glass-surface glass-surface-1 p-4">
            <h3 class="glass-section-header -mx-4 -mt-4 mb-3">FLOW MODE</h3>
            <div class="flex gap-2">
              <button
                onClick$={() => setFlowMode('m')}
                class={`flex-1 py-3 rounded-lg font-medium transition-all ${
                  session.value.flowMode === 'm'
                    ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50'
                    : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                }`}
              >
                <div class="text-lg">[m]</div>
                <div class="text-xs">Monitor</div>
              </button>
              <button
                onClick$={() => setFlowMode('A')}
                class={`flex-1 py-3 rounded-lg font-medium transition-all ${
                  session.value.flowMode === 'A'
                    ? 'bg-green-500/20 text-green-400 border border-green-500/50'
                    : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                }`}
              >
                <div class="text-lg">[A]</div>
                <div class="text-xs">Auto</div>
              </button>
            </div>
          </div>

          {/* Quick Actions */}
          <div class="glass-surface glass-surface-1 p-4">
            <h3 class="glass-section-header -mx-4 -mt-4 mb-3">ACTIONS</h3>
            <div class="grid grid-cols-2 gap-2">
              <button
                onClick$={() => dispatchAction('curation.trigger', { source: 'dashboard' })}
                class="p-3 rounded-lg bg-green-500/20 text-green-400 hover:bg-green-500/30 transition-all text-sm font-medium"
              >
                <div>🔄</div>
                <div>Curate</div>
              </button>
              <button
                onClick$={() => dispatchAction('worker.spawn', { provider: session.value.activeFallback || session.value.fallbackOrder.find((x) => x !== 'mock') || 'chatgpt-web' })}
                class="p-3 rounded-lg bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-all text-sm font-medium"
              >
                <div>👷</div>
                <div>Worker</div>
              </button>
              <button
                onClick$={() => dispatchAction('verify.run', {})}
                class="p-3 rounded-lg bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-all text-sm font-medium"
              >
                <div>✓</div>
                <div>Verify</div>
              </button>
              <button
                onClick$={() => dispatchAction('command.send', { topic: 'pluribus.status', kind: 'request', data: {} })}
                class="p-3 rounded-lg bg-orange-500/20 text-orange-400 hover:bg-orange-500/30 transition-all text-sm font-medium"
              >
                <div>📊</div>
                <div>Status</div>
              </button>
            </div>
          </div>

          {/* Dialogos Stream (Action Results) */}
          {actionCells.length > 0 && (
            <div class="glass-surface glass-surface-1 p-4">
              <div class="flex items-center justify-between mb-3">
                <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">DIALOGOS STREAM</h3>
                <button
                  onClick$={() => { activeView.value = 'services'; }}
                  class="text-xs text-primary hover:underline"
                >
                  Expand
                </button>
              </div>
              <div class="space-y-2">
                {actionCells.slice(0, 3).map((cell) => (
                  <div key={cell.id} class="p-2 rounded bg-muted/30 flex items-center gap-2">
                    <div class={`w-2 h-2 rounded-full ${
                      cell.result?.status === 'success' ? 'bg-green-500' :
                      cell.result?.status === 'error' ? 'bg-red-500' :
                      'bg-yellow-500 animate-pulse'
                    }`} />
                    <span class="text-xs font-mono">{cell.request.type}</span>
                    <span class="text-xs text-muted-foreground ml-auto">
                      {cell.result?.outputs.length || 0} outputs
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* InferCell (Command Input) */}
          <div class="glass-surface glass-surface-1 p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">INFERCELL</h3>
              <span class="text-xs px-2 py-0.5 rounded glass-status-ok">
                Context: [New]
              </span>
            </div>
            
            {/* Provider Selection */}
            <div class="flex flex-wrap gap-2 mb-3">
              {providerOptions.value.map(p => (
                <label key={p} class="flex items-center gap-1 text-xs text-muted-foreground bg-muted/30 px-2 py-1 rounded border border-border/50">
                  <input
                    type="checkbox"
                    checked={selectedProviders.value.includes(p)}
                    onChange$={() => toggleProvider(p)}
                  />
                  <span class="mono">{p}</span>
                </label>
              ))}
            </div>

            <div class="flex gap-2">
              <input
                type="text"
                bind:value={commandInput}
                placeholder="Interject command to selected providers..."
                class="flex-1 bg-muted/30 border border-border rounded-lg px-3 py-2 text-sm mono focus:outline-none focus:border-primary"
                onKeyDown$={(e) => {
                  if (e.key === 'Enter') {
                     const providers = selectedProviders.value.length > 0
                       ? selectedProviders.value
                       : [session.value.activeFallback || session.value.fallbackOrder.find(x => x !== 'mock') || 'chatgpt-web'];
                     dispatchAction('command.send', {
                         topic: 'dialogos.submit',
                         kind: 'request',
                         data: { mode: 'llm', providers, prompt: commandInput.value }
                     });
                     commandInput.value = '';
                  }
                }}
              />
              <button 
                onClick$={() => {
                     const providers = selectedProviders.value.length > 0
                       ? selectedProviders.value
                       : [session.value.activeFallback || session.value.fallbackOrder.find(x => x !== 'mock') || 'chatgpt-web'];
                     dispatchAction('command.send', {
                         topic: 'dialogos.submit',
                         kind: 'request',
                         data: { mode: 'llm', providers, prompt: commandInput.value }
                     });
                     commandInput.value = '';
                }}
                class="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:bg-primary/90"
              >
                Send
              </button>
            </div>
            <div class="text-xs text-muted-foreground mt-2">
              Interjects prompt to selected inference engines (via `dialogosd`).
            </div>
          </div>
        </div>

        {/* Center Column - Providers (click to select for inference) */}
        <div class="space-y-6">
          <div class="glass-surface glass-surface-1 p-4">
            <div class="flex items-center justify-between mb-3">
              <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground">PROVIDERS</h3>
              {/* Browser Auth Quick Link */}
              {providersList.value.some(([name, status]) => {
                const isWebProvider = ['chatgpt-web', 'claude-web', 'gemini-web'].includes(name);
                return isWebProvider && !status.available && (
                  status.error?.toLowerCase().includes('login') ||
                  status.error?.toLowerCase().includes('bot') ||
                  status.error?.toLowerCase().includes('challenge') ||
                  status.error?.toLowerCase().includes('onboarding')
                );
              }) && (
                <button
                  onClick$={() => activeView.value = 'browser-auth'}
                  class="text-[10px] px-2 py-1 rounded bg-amber-500/20 text-amber-400 border border-amber-500/30 hover:bg-amber-500/30 transition-colors flex items-center gap-1"
                >
                  <span class="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse" />
                  VNC Auth
                </button>
              )}
            </div>
            <div class="space-y-2">
              {providersList.value.map(([name, status]) => {
                const isWebProvider = ['chatgpt-web', 'claude-web', 'gemini-web'].includes(name);
                const needsLogin = status.error?.toLowerCase().includes('login') || status.error?.toLowerCase().includes('needs_login');
                const blockedBot = status.error?.toLowerCase().includes('bot') || status.error?.toLowerCase().includes('challenge');
                const needsOnboarding = status.error?.toLowerCase().includes('onboarding');
                const webAuthIssue = isWebProvider && !status.available && (needsLogin || blockedBot || needsOnboarding);

                return (
                <div
                  key={name}
                  class={`p-3 rounded-lg border transition-all ${
                    status.available
                      ? 'border-green-500/30 bg-green-500/10'
                      : webAuthIssue
                      ? 'border-amber-500/30 bg-amber-500/10'
                      : 'border-border bg-muted/20'
                  }`}
                >
                  <div class="flex items-center justify-between">
                    <div class="flex items-center gap-2">
                      <span class={`status-dot ${status.available ? 'available' : webAuthIssue ? 'warning' : 'unavailable'}`} />
                      <span class="font-medium mono text-sm">{name}</span>
                      {isWebProvider && (
                        <span class="text-[9px] px-1 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30">
                          WEB
                        </span>
                      )}
                    </div>
                    <div class="flex items-center gap-1">
                      {webAuthIssue && (
                        <span class={`text-[9px] px-1.5 py-0.5 rounded ${
                          blockedBot
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                        }`}>
                          {blockedBot ? '🤖 BOT BLOCKED' : needsOnboarding ? '🎯 ONBOARD' : '🔐 AUTH REQUIRED'}
                        </span>
                      )}
                      {status.model && (
                        <span class="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
                          {status.model}
                        </span>
                      )}
                    </div>
                  </div>
                  {(status.note || status.error) && (
                    <div class={`text-xs mt-1 ml-5 ${webAuthIssue ? 'text-amber-400/80' : 'text-muted-foreground'}`}>
                      {status.available ? status.note : status.error}
                    </div>
                  )}
                  {webAuthIssue && (
                    <div class="text-[10px] text-muted-foreground mt-1 ml-5 italic">
                      {blockedBot
                        ? 'Cloudflare bot detection active - manual browser auth may help'
                        : needsLogin
                        ? 'Browser session requires OAuth login'
                        : 'Complete onboarding flow in browser'}
                    </div>
                  )}
                </div>
              )})}
            </div>
          </div>

          {/* Fallback Chain */}
          <div class="glass-surface glass-surface-1 p-4">
            <h3 class="text-xs font-bold uppercase tracking-widest text-muted-foreground mb-3">FALLBACK CHAIN</h3>
            <div class="flex flex-wrap gap-1 items-center">
              {session.value.fallbackOrder.filter((p) => p !== 'mock').map((provider, i) => {
                const status = session.value.providers[provider as keyof typeof session.value.providers];
                const isWebProvider = ['chatgpt-web', 'claude-web', 'gemini-web'].includes(provider);
                const needsAuth = isWebProvider && !status?.available && (
                  status?.error?.toLowerCase().includes('login') ||
                  status?.error?.toLowerCase().includes('bot') ||
                  status?.error?.toLowerCase().includes('challenge')
                );
                const isAvailable = status?.available;
                const isActive = provider === session.value.activeFallback;

                return (
                <span key={provider} class="flex items-center gap-1">
                  {i > 0 && <span class="text-muted-foreground text-xs">→</span>}
                  <span
                    class={`text-xs px-2 py-1 rounded relative ${
                      isActive
                        ? 'bg-primary text-primary-foreground ring-2 ring-primary/50'
                        : needsAuth
                        ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 opacity-60'
                        : isAvailable
                        ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                        : 'bg-muted/50 text-muted-foreground opacity-60'
                    }`}
                    title={needsAuth ? 'Auth required - OAuth login needed' : status?.error || ''}
                  >
                    {needsAuth && <span class="mr-0.5">🔐</span>}
                    {provider}
                  </span>
                </span>
              )})}
            </div>
            {/* Active fallback indicator */}
            {session.value.activeFallback && (
              <div class="mt-2 text-xs text-muted-foreground">
                Active: <span class="text-primary font-medium">{session.value.activeFallback}</span>
                {['chatgpt-web', 'claude-web', 'gemini-web'].some(p =>
                  session.value.fallbackOrder.indexOf(p) < session.value.fallbackOrder.indexOf(session.value.activeFallback || '')
                ) && (
                  <span class="ml-2 text-amber-400/80">(web providers bypassed - need auth)</span>
                )}
              </div>
            )}
          </div>

          {/* Edge Inference Status */}
          <EdgeInferenceStatusWidget />
          <EdgeInferenceCatalog compact={true} />

          {/* WebLLM Edge Inference - lazy loaded to avoid 5.3MB initial bundle */}
          <div class="min-h-[420px]">
            <LazyWebLLM />
          </div>
          <div class="flex items-center justify-between text-xs text-muted-foreground px-2">
            <span>Edge inference runs in-browser; full WebLLM view includes local server status.</span>
            <button
              class="px-2 py-1 rounded bg-muted/50 hover:bg-muted text-muted-foreground"
              onClick$={() => (activeView.value = 'webllm')}
            >
              Open WebLLM
            </button>
          </div>

          {/* Dialogos Dual-Mind Orchestrator */}
          <DialogosContainer autoStart={true} minSessions={2} />
          <div class="text-xs text-muted-foreground mt-2 px-4 italic">
            &gt; OMEGA: Server inference active; browser WebLLM autostart enabled (WebGPU).
          </div>
        </div>

        {/* Right Column - Live Events (Enhanced Mini Snapshot) */}
        <div class="glass-surface glass-surface-1 flex flex-col h-[500px]">
          <div class="glass-section-header flex items-center justify-between flex-shrink-0">
            <div class="flex items-center gap-2">
              <span>LIVE EVENTS</span>
              <span class="text-[9px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400">
                {events.value.length}
              </span>
            </div>
            <button
              onClick$={cycleFilter}
              class="text-xs px-2 py-1 rounded bg-muted/50 hover:bg-muted text-muted-foreground"
            >
              {eventFilter.value || 'all'}
            </button>
          </div>

          {/* Mini Timeline Sparkline */}
          <div class="px-3 py-2 border-b border-border/50">
            <TimelineSparkline events={events.value.slice(-200)} buckets={40} height={30} width={280} />
          </div>

          {/* Quick Domain Stats */}
          <div class="px-3 py-2 border-b border-border/50 flex flex-wrap gap-1">
            {(() => {
              const domains = new Map<string, number>();
              for (const e of events.value.slice(-100)) {
                const d = e.topic.split('.')[0];
                domains.set(d, (domains.get(d) || 0) + 1);
              }
              return Array.from(domains.entries())
                .sort((a, b) => b[1] - a[1])
                .slice(0, 5)
                .map(([domain, count]) => (
                  <span
                    key={domain}
                    class="text-[9px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 border border-cyan-500/20"
                  >
                    {domain}:{count}
                  </span>
                ));
            })()}
          </div>

          {/* Event List with Enhanced Rendering */}
          <div class="flex-1 overflow-auto p-2 font-mono text-xs space-y-1">
            {filteredEvents.value.slice(-30).reverse().map((event, i) => {
              const parts = event.topic.split('.');
              const domain = parts[0];
              const rest = parts.slice(1).join('.');
              const semantic = (event as any).semantic;

              return (
                <div
                  key={i}
                  class={`p-1.5 rounded transition-all hover:scale-[1.01] ${
                    event.level === 'error' ? 'bg-red-500/10 text-red-400 border-l-2 border-red-500' :
                    event.level === 'warn' ? 'bg-yellow-500/10 text-yellow-400 border-l-2 border-yellow-500' :
                    event.kind === 'artifact' ? 'bg-purple-500/5 text-muted-foreground border-l-2 border-purple-500/50' :
                    'bg-muted/20 text-muted-foreground'
                  }`}
                >
                  <div class="flex items-center gap-1">
                    <span class="text-muted-foreground/60 flex-shrink-0">{event.iso?.slice(11, 19)}</span>
                    <span class="text-cyan-400 font-semibold">{domain}</span>
                    {rest && <span class="text-primary/70">.{rest}</span>}
                  </div>
                  <div class="flex items-center justify-between mt-0.5">
                    <span class="text-muted-foreground/50">@{event.actor}</span>
                    {semantic?.impact && (
                      <span class={`text-[8px] px-1 rounded ${
                        semantic.impact === 'critical' ? 'bg-red-500/20 text-red-400' :
                        semantic.impact === 'high' ? 'bg-orange-500/20 text-orange-400' :
                        'bg-blue-500/20 text-blue-400'
                      }`}>
                        {semantic.impact}
                      </span>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
      <SuperMotd
        connected={connected.value}
        events={events.value}
        session={session.value}
        emitBus$={emitBus}
      />

      {/* Agent Telemetry Panel - Real-time debugging feedback */}
      <AgentTelemetryPanel
        events={events.value}
        maxHeight="320px"
      />
      <NotificationSidepanel />
    </div>
  );
});
