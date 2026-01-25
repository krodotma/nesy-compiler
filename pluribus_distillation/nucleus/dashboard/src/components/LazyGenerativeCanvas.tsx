/**
 * LazyGenerativeCanvas - Lazy-loading wrapper for GenerativeCanvas
 *
 * Defers loading of three.js (673KB) until the canvas is visible.
 * Part of WebUI Performance Optimization Phase 2.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';
import { trackEntry } from '../lib/telemetry/use-tracking';

export interface LazyGenerativeCanvasProps {
  height?: string;
  width?: string;
}

export const LazyGenerativeCanvas = component$<LazyGenerativeCanvasProps>(({ height = '200px', width = '100%' }) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('GenerativeCanvas-bundle');
    const stopTracking = trackEntry('comp:generative-canvas');

    const load = async () => {
      try {
        const { GenerativeCanvas } = await import('./GenerativeCanvas');
        Widget.value = GenerativeCanvas;
        isLoaded.value = true;
      } catch (err) {
        const registry = (window as any).__loadingRegistry;
        registry?.exit('comp:generative-canvas', String(err));
        console.error('[LazyGenerativeCanvas] Failed to load:', err);
      } finally {
        stopTiming();
        stopTracking();
      }
    };

    load();
  });

  if (!isLoaded.value) {
    return (
      <div
        class="flex items-center justify-center bg-gradient-to-br from-purple-900/20 to-blue-900/20 rounded-lg border border-purple-500/20"
        style={{ height, width }}
      >
        <div class="text-center space-y-3">
          <div class="animate-pulse text-3xl">🎨</div>
          <div class="text-sm text-muted-foreground">Loading Canvas...</div>
          <div class="w-24 h-1 bg-muted rounded overflow-hidden mx-auto">
            <div class="h-full bg-purple-500 animate-[loading_1.5s_ease-in-out_infinite]" style="width: 30%;" />
          </div>
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? (
    <div style={{ animation: "fadeIn 0.7s ease-out forwards" }}>
      <LoadedWidget height={height} width={width} />
    </div>
  ) : null;
});
