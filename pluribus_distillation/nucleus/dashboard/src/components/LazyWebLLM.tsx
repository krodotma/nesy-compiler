/**
 * LazyWebLLM - Lazy-loading wrapper for WebLLMWidget
 *
 * Defers loading of the 5.3MB WebLLM bundle until the user actually
 * navigates to a view that needs it. This dramatically improves initial
 * page load time.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';
import { trackEntry } from '../lib/telemetry/use-tracking';

export interface LazyWebLLMProps {
  fullScreen?: boolean;
}

export const LazyWebLLM = component$<LazyWebLLMProps>(({ fullScreen = false }) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);

  useVisibleTask$(() => {
    // Only load when component becomes visible
    const stopTiming = startComponentTiming('WebLLM-bundle');
    const stopTracking = trackEntry('comp:lazy-webllm');

    const load = async () => {
      try {
        const { WebLLMWidget } = await import('./WebLLMWidget');
        Widget.value = WebLLMWidget;
        isLoaded.value = true;
      } catch (err) {
        const registry = (window as any).__loadingRegistry;
        registry?.exit('comp:lazy-webllm', String(err));
        console.error('[LazyWebLLM] Failed to load:', err);
      } finally {
        stopTiming();
        stopTracking();
      }
    };

    load();
  }, { strategy: 'document-ready' });

  if (!isLoaded.value) {
    return (
      <div class="flex items-center justify-center h-full min-h-[200px] bg-black/50 rounded-lg border border-cyan-500/20">
        <div class="text-center space-y-3">
          <div class="animate-pulse text-4xl">🧩</div>
          <div class="text-sm text-muted-foreground">Loading WebLLM...</div>
          <div class="w-32 h-1 bg-muted rounded overflow-hidden mx-auto">
            <div class="h-full bg-cyan-500 animate-[loading_1.5s_ease-in-out_infinite]"
                 style="width: 30%; animation: loading 1.5s ease-in-out infinite;" />
          </div>
        </div>
      </div>
    );
  }

  // Render the loaded widget
  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget fullScreen={fullScreen} /> : null;
});
