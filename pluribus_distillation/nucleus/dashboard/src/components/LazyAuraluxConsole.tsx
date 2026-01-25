/**
 * LazyAuraluxConsole - Lazy-loading wrapper for AuraluxConsole
 *
 * Defers loading of onnxruntime-web dependencies until the voice view is active.
 */

import { component$, type QRL, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';

export interface LazyAuraluxConsoleProps {
  session: unknown;
  emitBus$?: QRL<(topic: string, kind: string, data: Record<string, unknown>) => Promise<void>>;
}

export const LazyAuraluxConsole = component$<LazyAuraluxConsoleProps>(({ session, emitBus$ }) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('AuraluxConsole-bundle');
    const registry = (window as any).__loadingRegistry;
    registry?.entry('comp:auralux');

    const load = async () => {
      try {
        const { AuraluxConsole } = await import('./views/AuraluxConsole');
        Widget.value = AuraluxConsole;
        isLoaded.value = true;
      } catch (err) {
        registry?.exit('comp:auralux', String(err));
        console.error('[LazyAuraluxConsole] Failed to load:', err);
      } finally {
        stopTiming();
        registry?.exit('comp:auralux');
      }
    };

    load();
  });

  if (!isLoaded.value) {
    return (
      <div class="flex items-center justify-center h-[420px] rounded-lg border border-amber-500/20 bg-black/60">
        <div class="text-center space-y-3">
          <div class="text-xl font-mono text-amber-400">AURALUX</div>
          <div class="text-sm text-muted-foreground">Loading voice engine...</div>
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget session={session} emitBus$={emitBus$} /> : null;
});
