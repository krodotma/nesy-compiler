import { component$, useStore, useVisibleTask$, $ } from '@builder.io/qwik';
import { loadViewManifests, type ViewManifestSummary } from '../lib/view-registry';
import { Card } from './ui/Card';
import { Button } from './ui/Button';

interface ViewManifestState {
  manifests: ViewManifestSummary[];
  loading: boolean;
  error: string | null;
}

export const ViewManifestPanel = component$(() => {
  const state = useStore<ViewManifestState>({
    manifests: [],
    loading: true,
    error: null,
  });

  const refresh = $(async () => {
    state.loading = true;
    state.error = null;
    try {
      state.manifests = await loadViewManifests();
    } catch (err) {
      state.error = err instanceof Error ? err.message : 'Failed to load view manifests';
    } finally {
      state.loading = false;
    }
  });

  useVisibleTask$(() => {
    refresh();
  });

  return (
    <Card variant="outlined" padding="p-4">
      <div class="flex items-center justify-between">
        <div>
          <div class="text-sm font-semibold">View Manifests</div>
          <div class="text-xs text-muted-foreground">Tokenized view templates (web + TUI)</div>
        </div>
        <Button variant="tonal" class="h-7 text-xs" onClick$={refresh}>
          Refresh
        </Button>
      </div>
      <div class="mt-3 space-y-2">
        {state.loading && <div class="text-xs text-muted-foreground">Loading manifests...</div>}
        {state.error && <div class="text-xs text-red-400">{state.error}</div>}
        {!state.loading && !state.error && state.manifests.length === 0 && (
          <div class="text-xs text-muted-foreground">No manifests found.</div>
        )}
        {state.manifests.map((manifest) => (
          <div key={manifest.id} class="flex items-center justify-between text-xs">
            <div>
              <div class="font-medium">{manifest.name}</div>
              <div class="text-[11px] text-muted-foreground">{manifest.description || manifest.id}</div>
            </div>
            <div class="text-[11px] text-muted-foreground">v{manifest.version}</div>
          </div>
        ))}
      </div>
    </Card>
  );
});
