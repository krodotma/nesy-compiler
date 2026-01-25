/**
 * LazyTerminal - Lazy-loading wrapper for Terminal components
 *
 * Defers loading of xterm.js (278KB) until the terminal is visible.
 * Part of WebUI Performance Optimization Phase 2.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import { startComponentTiming } from '../lib/telemetry/load-timing';
import { trackEntry } from '../lib/telemetry/use-tracking';

export interface LazyTerminalProps {
  height?: string;
  onCommand$?: (cmd: string) => void;
}

export const LazyTerminal = component$<LazyTerminalProps>(({ height = '300px', onCommand$ }) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);
  const surfaceStyle = {
    backgroundColor: 'var(--md-sys-color-surface, var(--mat-surface))',
    borderColor: 'var(--md-sys-color-outline-variant, var(--mat-border))',
  };
  const primaryAccent = {
    color: 'var(--md-sys-color-primary, var(--mat-primary))',
  };
  const primaryFill = {
    backgroundColor: 'var(--md-sys-color-primary, var(--mat-primary))',
  };

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('Terminal-bundle');
    const stopTracking = trackEntry('comp:terminal');

    const load = async () => {
      try {
        const { Terminal } = await import('./terminal');
        Widget.value = Terminal;
        isLoaded.value = true;
      } catch (err) {
        const registry = (window as any).__loadingRegistry;
        registry?.exit('comp:terminal', String(err));
        console.error('[LazyTerminal] Failed to load:', err);
      } finally {
        stopTiming();
        stopTracking();
      }
    };

    load();
  });

  if (!isLoaded.value) {
    return (
      <div class="flex items-center justify-center rounded-lg border" style={{ height, ...surfaceStyle }}>
        <div class="text-center space-y-3">
          <div class="animate-pulse text-3xl font-mono" style={primaryAccent}>_</div>
          <div class="text-sm text-muted-foreground">Loading Terminal...</div>
          <div class="w-32 h-1 bg-muted rounded overflow-hidden mx-auto">
            <div class="h-full animate-[loading_1.5s_ease-in-out_infinite]" style={{ width: '30%', ...primaryFill }} />
          </div>
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget height={height} onCommand$={onCommand$} /> : null;
});

export interface LazyPluriChatTerminalProps {
  height?: string;
}

export const LazyPluriChatTerminal = component$<LazyPluriChatTerminalProps>(({ height = '400px' }) => {
  const isLoaded = useSignal(false);
  const Widget = useSignal<any>(null);
  const surfaceStyle = {
    backgroundColor: 'var(--md-sys-color-surface, var(--mat-surface))',
    borderColor: 'var(--md-sys-color-outline-variant, var(--mat-border))',
  };
  const secondaryAccent = {
    color: 'var(--md-sys-color-secondary, var(--mat-secondary))',
  };

  useVisibleTask$(() => {
    const stopTiming = startComponentTiming('PluriChatTerminal-bundle');
    const stopTracking = trackEntry('comp:plurichat');

    const load = async () => {
      try {
        const { PluriChatTerminal } = await import('./terminal');
        Widget.value = PluriChatTerminal;
        isLoaded.value = true;
      } catch (err) {
        const registry = (window as any).__loadingRegistry;
        registry?.exit('comp:plurichat', String(err));
        console.error('[LazyPluriChatTerminal] Failed to load:', err);
      } finally {
        stopTiming();
        stopTracking();
      }
    };

    load();
  });

  if (!isLoaded.value) {
    return (
      <div class="flex items-center justify-center rounded-lg border" style={{ height, ...surfaceStyle }}>
        <div class="text-center space-y-3">
          <div class="animate-pulse text-3xl" style={secondaryAccent}>💬</div>
          <div class="text-sm text-muted-foreground">Loading Chat Terminal...</div>
        </div>
      </div>
    );
  }

  const LoadedWidget = Widget.value;
  return LoadedWidget ? <LoadedWidget height={height} /> : null;
});
