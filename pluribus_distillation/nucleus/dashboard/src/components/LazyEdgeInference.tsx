/**
 * LazyEdgeInference - Lazy-loading wrappers for Edge Inference components
 *
 * Defers loading of webllm-enhanced (5.3MB bundle) until the user actually
 * navigates to the WebLLM view. This dramatically improves initial page load.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';

/**
 * Lazy wrapper for EdgeInferenceStatusWidget
 */
export const LazyEdgeInferenceStatusWidget = component$(() => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);
  const loadError = useSignal<string | null>(null);

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('EdgeInferenceStatusWidget-bundle');
    const registry = (window as any).__loadingRegistry;
    registry?.entry('comp:lazy-edge-status');

    const load = async () => {
      try {
        const { EdgeInferenceStatusWidget } = await import('./EdgeInferenceStatusWidget');
        Widget.value = EdgeInferenceStatusWidget;
        isLoaded.value = true;
      } catch (err) {
        loadError.value = String(err);
        registry?.exit('comp:lazy-edge-status', String(err));
        console.error('[LazyEdgeInferenceStatusWidget] Failed to load:', err);
      } finally {
        stopTiming();
        registry?.exit('comp:lazy-edge-status');
      }
    };

    load();
  }, { strategy: 'document-ready' });

  if (loadError.value) {
    return (
      <div class="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
        Failed to load Edge Inference Status: {loadError.value}
      </div>
    );
  }

  if (!isLoaded.value) {
    return (
      <div class="p-4 bg-black/50 rounded-lg border border-cyan-500/20 animate-pulse">
        <div class="flex items-center gap-3">
          <div class="w-6 h-6 rounded-full bg-cyan-500/20" />
          <div class="flex-1 space-y-2">
            <div class="h-4 bg-cyan-500/10 rounded w-1/3" />
            <div class="h-3 bg-cyan-500/10 rounded w-1/2" />
          </div>
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget /> : null;
});

/**
 * Lazy wrapper for EdgeInferenceCatalog
 */
export interface LazyEdgeInferenceCatalogProps {
  compact?: boolean;
}

export const LazyEdgeInferenceCatalog = component$<LazyEdgeInferenceCatalogProps>(({ compact = false }) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);
  const loadError = useSignal<string | null>(null);

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('EdgeInferenceCatalog-bundle');
    const registry = (window as any).__loadingRegistry;
    registry?.entry('comp:lazy-edge-catalog');

    const load = async () => {
      try {
        const { EdgeInferenceCatalog } = await import('./EdgeInferenceCatalog');
        Widget.value = EdgeInferenceCatalog;
        isLoaded.value = true;
      } catch (err) {
        loadError.value = String(err);
        registry?.exit('comp:lazy-edge-catalog', String(err));
        console.error('[LazyEdgeInferenceCatalog] Failed to load:', err);
      } finally {
        stopTiming();
        registry?.exit('comp:lazy-edge-catalog');
      }
    };

    load();
  }, { strategy: 'document-ready' });

  if (loadError.value) {
    return (
      <div class="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
        Failed to load Edge Inference Catalog: {loadError.value}
      </div>
    );
  }

  if (!isLoaded.value) {
    return (
      <div class="p-4 bg-black/50 rounded-lg border border-cyan-500/20 space-y-3">
        <div class="h-4 bg-cyan-500/10 rounded w-1/4 animate-pulse" />
        <div class="grid grid-cols-2 gap-2">
          {[1, 2, 3, 4].map((i) => (
            <div key={i} class="h-16 bg-cyan-500/5 rounded animate-pulse" style={{ animationDelay: `${i * 100}ms` }} />
          ))}
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget compact={compact} /> : null;
});
