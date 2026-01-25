import { component$, type Signal, type QRL } from '@builder.io/qwik';
import { useTracking } from '../../lib/telemetry/use-tracking';
import sotaMatrix from '../../data/sota_integration_matrix.json';

// M3 Components (Step 63 - SOTA Cards Upgrade)
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/button/text-button.js';

interface SOTAItem {
  id: string;
  url: string;
  title: string;
  org: string;
  region: string;
  type: string;
  priority: number;
  cadence_days: number;
  tags: string[];
  notes: string;
  distill_status?: 'idle' | 'queued' | 'running' | 'completed' | 'failed';
  distill_last_iso?: string;
  distill_req_id?: string;
  distill_artifact_path?: string;
  distill_snippet?: string;
}

interface SotaViewProps {
  sotaItems: Signal<SOTAItem[]>;
  sotaProvider: Signal<string>;
  sotaProviderOptions: readonly string[];
  sotaByType: Signal<Record<string, SOTAItem[]>>;
  dispatchAction: QRL<(type: string, payload: Record<string, unknown>) => void>;
}

export const SotaView = component$<SotaViewProps>((props) => {
  useTracking('comp:sota-view');
  const { sotaItems, sotaProvider, sotaProviderOptions, sotaByType, dispatchAction } = props;

  return (
    <div class="space-y-6">
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold">SOTA Catalog</h2>
          <p class="text-sm text-muted-foreground">Curated tools, sources, and research feeds</p>
        </div>
        <div class="flex items-center gap-3">
          <select
            class="px-3 py-2 rounded bg-muted/50 border border-border text-sm"
            value={sotaProvider.value}
            onChange$={(e) => { sotaProvider.value = (e.target as HTMLSelectElement).value; }}
          >
            {sotaProviderOptions.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
          <div class="text-sm text-muted-foreground">
            {sotaItems.value.length} items
          </div>
        </div>
      </div>

      {/* Integration Matrix Section */}
      <div class="sota-matrix-container rounded-lg border border-border bg-card">
        <md-elevation class="sota-matrix-elevation"></md-elevation>
        <div class="p-4 border-b border-border flex items-center gap-3">
          <span class="text-primary">🔌</span>
          <h3 class="font-semibold">Integration Matrix</h3>
          <span class="text-xs text-muted-foreground">(Pluribus Adoption Status)</span>
        </div>
        <div class="p-4 grid gap-4 lg:grid-cols-2">
          {sotaMatrix.map((tool) => (
            <div key={tool.id} class="sota-integration-card">
              <md-ripple class="sota-card-ripple"></md-ripple>
              <md-elevation class="sota-card-elevation"></md-elevation>
              <div class="flex justify-between items-start mb-2">
                <div>
                  <div class="font-medium text-primary">{tool.name}</div>
                  <div class="text-xs text-muted-foreground">{tool.pattern}</div>
                </div>
                <div class="text-right">
                  <span class={`sota-status-badge text-xs px-2 py-0.5 rounded ${
                    tool.status === 'active' ? 'status-active' :
                    tool.status === 'in-progress' ? 'status-progress' :
                    'status-planned'
                  }`}>
                    {tool.status}
                  </span>
                </div>
              </div>

              <div class="space-y-1">
                <div class="flex justify-between text-xs text-muted-foreground">
                  <span>Integration</span>
                  <span>{tool.integration_percent}%</span>
                </div>
                <md-linear-progress
                  class="sota-progress"
                  value={tool.integration_percent / 100}
                ></md-linear-progress>
              </div>

              <div class="mt-2 text-xs text-muted-foreground border-l-2 border-primary/30 pl-2">
                {tool.plan_details}
              </div>
            </div>
          ))}
        </div>
      </div>

      {Object.entries(sotaByType.value).map(([type, items]) => (
        <div key={type} class="sota-type-section rounded-lg border border-border bg-card">
          <md-elevation class="sota-section-elevation"></md-elevation>
          <div class="p-4 border-b border-border flex items-center gap-3">
            <span class="text-primary">
              {type === 'rss' ? '📰' : type === 'repo' ? '📦' : type === 'blog' ? '📝' : type === 'site' ? '🌐' : '📄'}
            </span>
            <h3 class="font-semibold capitalize">{type}</h3>
            <span class="text-xs text-muted-foreground">({items.length})</span>
          </div>
          <div class="divide-y divide-border/30">
            {items.map((item) => (
              <div key={item.id} class="sota-item-card">
                <md-ripple class="sota-item-ripple"></md-ripple>
                <div class="flex items-start justify-between gap-4">
                  <div class="flex-1">
                    <a
                      href={item.url}
                      target="_blank"
                      rel="noopener"
                      class="font-medium text-primary hover:underline"
                    >
                      {item.title}
                    </a>
                    <div class="text-sm text-muted-foreground mt-1">
                      {item.org} • {item.region}
                    </div>
                    {item.notes && (
                      <div class="text-xs text-muted-foreground mt-2">{item.notes}</div>
                    )}
                  </div>
                  <div class="flex flex-col items-end gap-2">
                    <span class={`sota-priority-badge text-xs px-2 py-0.5 rounded ${
                      item.priority === 1 ? 'priority-high' :
                      item.priority === 2 ? 'priority-medium' :
                      'priority-low'
                    }`}>
                      P{item.priority}
                    </span>
                    {item.distill_status && (
                      <span class={`sota-distill-badge text-xs px-2 py-0.5 rounded ${
                        item.distill_status === 'completed' ? 'distill-completed' :
                        item.distill_status === 'failed' ? 'distill-failed' :
                        item.distill_status === 'running' ? 'distill-running' :
                        item.distill_status === 'queued' ? 'distill-queued' :
                        'distill-idle'
                      }`}>
                        {item.distill_status}
                      </span>
                    )}
                    <div class="flex gap-1 flex-wrap justify-end">
                      {item.tags.slice(0, 3).map((tag) => (
                        <span key={tag} class="sota-tag text-xs px-1.5 py-0.5 rounded">
                          {tag}
                        </span>
                      ))}
                    </div>
                    <div class="flex gap-1 mt-1">
                      <md-text-button
                        class="sota-action-btn"
                        onClick$={() => dispatchAction('sota.distill', { itemId: item.id, provider: sotaProvider.value })}
                      >
                        Distill
                      </md-text-button>
                      <md-text-button
                        class="sota-action-btn"
                        onClick$={() => dispatchAction('sota.kg.add', { itemId: item.id, ref: item.distill_artifact_path || item.url })}
                      >
                        KG
                      </md-text-button>
                    </div>
                  </div>
                </div>
                {item.distill_artifact_path && (
                  <div class="mt-3 rounded bg-muted/30 border border-border/50 p-3">
                    <div class="text-xs text-muted-foreground">
                      Distillation artifact: <span class="mono text-primary">{item.distill_artifact_path}</span>
                    </div>
                    {item.distill_snippet && (
                      <pre class="mono text-xs whitespace-pre-wrap mt-2 max-h-40 overflow-auto">{item.distill_snippet}</pre>
                    )}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {sotaItems.value.length === 0 && (
        <div class="rounded-lg border border-border bg-card p-8 text-center text-muted-foreground">
          No SOTA items loaded. Check /api/sota or rebuild the index.
        </div>
      )}
    </div>
  );
});
