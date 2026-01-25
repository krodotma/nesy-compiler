import { component$, type Signal, type QRL, $ } from '@builder.io/qwik';
import type { ServiceDef } from '../../lib/state/types';
import type { ActionCell } from '../../lib/actions/types';
import { OutputCell } from '../../lib/actions/OutputCell';

interface ServicesViewProps {
  mergedServices: Signal<Array<ServiceDef & { instanceId?: string }>>;
  actionCells: ActionCell[];
  dispatchAction: QRL<(type: string, payload: Record<string, unknown>) => void>;
  clearActionCells: QRL<() => void>;
  toggleCellCollapse: QRL<(cellId: string) => void>;
}

export const ServicesView = component$<ServicesViewProps>((props) => {
  const { mergedServices, actionCells, dispatchAction, clearActionCells, toggleCellCollapse } = props;

  return (
    <div class="grid gap-6 lg:grid-cols-2 glass-animate-enter">
      {/* Left: Service Lists */}
      <div class="space-y-6">
        {/* Port Services */}
        <div class="glass-surface">
          <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
            <span class="text-primary">⚡</span>
            <h3 class="font-semibold glass-text-title">Port Services</h3>
            <span class="text-xs glass-text-muted">(HTTP/WS, MCP, APIs)</span>
          </div>
          <div class="divide-y divide-[var(--glass-border-subtle)]">
            {mergedServices.value.filter(s => s.kind === 'port').map((svc) => (
              <div key={svc.id} class="p-4 flex items-center justify-between glass-hover-glow glass-transition-hover">
                <div class="flex items-center gap-3">
                  <span class={`status-dot ${svc.status === 'running' ? 'glass-status-ok pulse' : 'glass-status-error'}`} />
                  <div>
                    <div class="font-medium glass-text-body">{svc.name}</div>
                    <div class="text-xs glass-text-muted">{svc.description}</div>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  {svc.port && <span class="mono text-sm text-primary">:{svc.port}</span>}
                  <div class="flex gap-1">
                    <button
                      onClick$={() => dispatchAction(svc.status === 'running' ? 'service.stop' : 'service.start', { serviceId: svc.id, instanceId: svc.instanceId })}
                      class={`glass-chip glass-transition-hover ${
                        svc.status === 'running'
                          ? 'glass-status-error'
                          : 'glass-status-ok'
                      }`}
                    >
                      {svc.status === 'running' ? 'Stop' : 'Start'}
                    </button>
                    <button
                      onClick$={() => dispatchAction('service.logs', { serviceId: svc.id, lines: 50 })}
                      class="glass-chip glass-hover-glow glass-transition-hover"
                    >
                      Logs
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Process Services */}
        <div class="glass-surface">
          <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
            <span class="text-primary">🔄</span>
            <h3 class="font-semibold glass-text-title">Process Services</h3>
            <span class="text-xs glass-text-muted">(Workers, Daemons, TUIs)</span>
          </div>
          <div class="divide-y divide-[var(--glass-border-subtle)]">
            {mergedServices.value.filter(s => s.kind === 'process').map((svc) => (
              <div key={svc.id} class="p-4 flex items-center justify-between glass-hover-glow glass-transition-hover">
                <div class="flex items-center gap-3">
                  <span class={`status-dot ${svc.status === 'running' ? 'glass-status-ok pulse' : 'glass-status-error'}`} />
                  <div>
                    <div class="font-medium glass-text-body">{svc.name}</div>
                    <div class="text-xs glass-text-muted">{svc.description}</div>
                  </div>
                </div>
                <div class="flex items-center gap-3">
                  <div class="flex gap-1">
                    {(svc.tags || []).slice(0, 2).map((tag) => (
                      <span key={tag} class="glass-chip">{tag}</span>
                    ))}
                  </div>
                  <div class="flex gap-1">
                    <button
                      onClick$={() => dispatchAction(svc.status === 'running' ? 'service.stop' : 'service.start', { serviceId: svc.id, instanceId: svc.instanceId })}
                      class={`glass-chip glass-transition-hover ${
                        svc.status === 'running'
                          ? 'glass-status-error'
                          : 'glass-status-ok'
                      }`}
                    >
                      {svc.status === 'running' ? 'Stop' : 'Start'}
                    </button>
                    <button
                      onClick$={() => dispatchAction('service.restart', { serviceId: svc.id, instanceId: svc.instanceId })}
                      class="glass-chip glass-chip-accent-amber glass-transition-hover"
                    >
                      Restart
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Compositions */}
        <div class="glass-surface">
          <div class="p-4 border-b border-[var(--glass-border)] flex items-center gap-3">
            <span class="text-primary">🔗</span>
            <h3 class="font-semibold glass-text-title">Compositions</h3>
            <span class="text-xs glass-text-muted">(Pipelines, Workflows)</span>
          </div>
          <div class="divide-y divide-[var(--glass-border-subtle)]">
            {mergedServices.value.filter(s => s.kind === 'composition').map((svc) => (
              <div key={svc.id} class="p-4 flex items-center justify-between glass-hover-glow glass-transition-hover">
                <div class="flex items-center gap-3">
                  <span class={`status-dot ${svc.status === 'running' ? 'glass-status-ok pulse' : 'glass-status-error'}`} />
                  <div>
                    <div class="font-medium glass-text-body">{svc.name}</div>
                    <div class="text-xs glass-text-muted">{svc.description}</div>
                  </div>
                </div>
                <button
                  onClick$={() => dispatchAction('composition.run', { compositionId: svc.id })}
                  class="glass-chip glass-chip-accent-purple glass-transition-hover"
                >
                  Run
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right: Dialogos Stream (Action Results Panel) */}
      <div class="glass-surface-elevated flex flex-col h-[calc(100vh-200px)]">
        <div class="p-4 border-b border-[var(--glass-border)] flex items-center justify-between flex-shrink-0">
          <h3 class="font-semibold glass-text-title">Dialogos</h3>
          <div class="flex items-center gap-2">
            <span class="text-xs glass-text-muted">{actionCells.length} actions</span>
            {actionCells.length > 0 && (
              <button
                onClick$={clearActionCells}
                class="glass-chip glass-hover-glow glass-transition-hover"
              >
                Clear
              </button>
            )}
          </div>
        </div>

        <div class="flex-1 overflow-auto p-4 space-y-4">
          {actionCells.length === 0 ? (
            <div class="text-center py-8 glass-text-muted">
              <div class="text-4xl mb-3">💬</div>
              <div>Dialogos Stream Empty</div>
              <div class="text-xs mt-1">Interject commands to begin the dialogue.</div>
            </div>
          ) : (
            actionCells.map((cell) => (
              <OutputCell
                key={cell.id}
                cell={cell}
                onToggleCollapse$={$(() => toggleCellCollapse(cell.id))}
              />
            ))
          )}
        </div>
      </div>
    </div>
  );
});
