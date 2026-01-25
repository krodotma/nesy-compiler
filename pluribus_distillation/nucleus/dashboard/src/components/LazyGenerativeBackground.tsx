/**
 * LazyGenerativeBackground - Lazy-loading wrapper for GenerativeBackground
 *
 * Defers loading of three.js (673KB) until the background is needed.
 * Improves initial page load by not blocking on 3D canvas initialization.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';

export interface LazyGenerativeBackgroundProps {
  entropy?: number;
  mood?: string;
  enabled?: boolean;
  wsUrl?: string;
  requestScene?: boolean;
}

export const LazyGenerativeBackground = component$<LazyGenerativeBackgroundProps>(({
  entropy = 0.1,
  mood = 'calm',
  enabled = true,
  wsUrl,
  requestScene = false,
}) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);
  const loadError = useSignal<string | null>(null);

  useVisibleTask$(() => {
    // Delay loading slightly to prioritize critical content
    const timeoutId = setTimeout(async () => {
      const stopTiming = startComponentTiming('GenerativeBackground-bundle');
      const registry = (window as any).__loadingRegistry;
      registry?.entry('comp:generative-bg');

      try {
        const { GenerativeBackground } = await import('./art/GenerativeBackground');
        Widget.value = GenerativeBackground;
        isLoaded.value = true;
      } catch (err) {
        loadError.value = String(err);
        registry?.exit('comp:generative-bg', String(err));
        console.error('[LazyGenerativeBackground] Failed to load:', err);
      } finally {
        stopTiming();
        registry?.exit('comp:generative-bg');
      }
    }, 100); // 100ms delay to let critical UI paint first

    return () => clearTimeout(timeoutId);
  }, { strategy: 'document-idle' }); // Use idle strategy for non-critical background

  // Return a simple gradient placeholder while loading
  // This doesn't block interactivity and provides visual continuity
  if (!isLoaded.value) {
    return (
      <div
        class="fixed inset-0 -z-10 pointer-events-none"
        style={{
          background: 'radial-gradient(ellipse at center, rgba(30, 20, 50, 0.8) 0%, rgba(10, 10, 20, 0.95) 100%)',
          opacity: enabled ? 1 : 0,
          transition: 'opacity 0.5s ease',
        }}
      />
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? (
    <LoadedWidget
      entropy={entropy}
      mood={mood}
      enabled={enabled}
      wsUrl={wsUrl}
      requestScene={requestScene}
    />
  ) : null;
});
