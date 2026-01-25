/**
 * LazyGitView - Lazy-loading wrapper for GitView
 *
 * Defers loading of GitView (41KB, 935 lines) until visible.
 * Part of WebUI Performance Optimization Phase 2.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';
import { trackEntry } from '../lib/telemetry/use-tracking';

export const LazyGitView = component$(() => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('GitView-bundle');
    const stopTracking = trackEntry('comp:git-view');

    const load = async () => {
      try {
        const { GitView } = await import('./git/GitView');
        Widget.value = GitView;
        isLoaded.value = true;
      } catch (err) {
        const registry = (window as any).__loadingRegistry;
        registry?.exit('comp:git-view', String(err));
        console.error('[LazyGitView] Failed to load:', err);
      } finally {
        stopTiming();
        stopTracking();
      }
    };

    load();
  });

  if (!isLoaded.value) {
    return (
      <div class="flex items-center justify-center h-full min-h-[300px] bg-black/50 rounded-lg border border-orange-500/20">
        <div class="text-center space-y-3">
          <div class="animate-pulse text-3xl">📊</div>
          <div class="text-sm text-muted-foreground">Loading Git View...</div>
          <div class="w-32 h-1 bg-muted rounded overflow-hidden mx-auto">
            <div class="h-full bg-orange-500 animate-[loading_1.5s_ease-in-out_infinite]" style="width: 30%;" />
          </div>
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget /> : null;
});
