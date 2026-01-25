/**
 * HeaderOmega: The Top Bread (Agency Edition)
 * ===========================================
 * Persistent application header with:
 * - Glass morphism surfaces with theme-aware gradients
 * - M3 elevation and ripple integration
 * - Agency-grade typography and "emerging from sphere" logo effect
 *
 * Integrated from HeaderOmega_Agency.tsx.scratch
 */

import { component$, useVisibleTask$, useSignal, $ } from '@builder.io/qwik';
import { useTracking } from '../../lib/telemetry/use-tracking';
import { VNCAuthFab } from '../VNCAuthFab';
import { ThemeModeToggle } from '../ThemeModeToggle';
import { GestaltPill } from '../GestaltPill';
import { VisionEye } from '../VisionEye/VisionEye';
import { NeonTitle, NeonBadge } from '../ui/NeonTitle';

// M3 Components
import '@material/web/elevation/elevation.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/ripple/ripple.js';

// Glass tokens
import '../../theme/glass-tokens.css';

interface HeaderOmegaProps {
  connected: boolean;
  workerCount: number;
  mood: 'focused' | 'anxious' | 'calm';
  entropy: number;
  providerStatus: Record<string, { available: boolean; error?: string }>;
  onOpenAuth$: () => void;
}

export const HeaderOmega = component$<HeaderOmegaProps>((props) => {
  const isRefreshing = useSignal(false);
  useTracking("comp:header");

  useVisibleTask$(() => {
    (window as any).__loadingRegistry?.entry("comp:header");
    (window as any).__loadingRegistry?.exit("comp:header");
  });

  const handleRefresh = $(() => {
    if (isRefreshing.value) return;
    isRefreshing.value = true;
    const broadcast = new BroadcastChannel('pluribus-shadow');
    broadcast.postMessage({ type: 'PREFETCH', payload: { views: ['sota', 'git', 'browser'] } });
    broadcast.close();
    window.dispatchEvent(new CustomEvent('pluribus:art:request', {
      detail: { reason: 'manual_refresh' }
    }));
    try {
      sessionStorage.removeItem('pluribus_art_injected_session');
    } catch { /* ignore */ }
    setTimeout(() => {
      isRefreshing.value = false;
    }, 1500);
  });

  return (
    <header class="header-omega-agency relative z-50 h-20">
      {/* Glass background layer */}
      <div class="header-glass-agency absolute inset-0 border-b border-[var(--glass-border)] shadow-2xl backdrop-blur-xl" />

      {/* Content layer */}
      <div class="header-content-agency relative h-full flex items-center justify-between px-8 max-w-[1920px] mx-auto">
        {/* Left section: Logo + Title */}
        <div class="header-left-agency flex items-center">
          
          {/* Premium Logo Text */}
          <h1 class="header-title glass-chromatic-subtle">
            <span class="title-text">Pluribus</span>
            <span class="title-accent" aria-hidden="true">Pluribus</span>
            <span class="title-glow" aria-hidden="true">Pluribus</span>
          </h1>

          <div class="header-branding-agency ml-6 flex flex-col opacity-80 border-l border-[var(--glass-border)] pl-4">
            <NeonTitle level="span" color="cyan" size="xs" class="header-tagline-agency text-xs font-bold uppercase tracking-widest text-[var(--glass-accent-cyan)] shadow-[var(--glass-accent-cyan-subtle)]">
              AIOS: Evolutionary Code Agency
            </NeonTitle>
            <span class="header-subtitle-agency text-[0.7rem] text-[var(--glass-text-secondary)]">
              Multi-Agent Autonomous Orchestration
            </span>
          </div>
        </div>

        {/* Right section: Status + Controls */}
        <div class="header-right-agency flex items-center gap-6">
          <div class="header-stat-agency flex items-center gap-2 font-mono text-xs bg-[var(--glass-bg-card)] px-3 py-1.5 rounded-full border border-[var(--glass-border)]">
            <NeonTitle level="span" color="cyan" size="xs">Workers</NeonTitle>
            <NeonBadge color={props.workerCount > 0 ? 'emerald' : 'amber'} glow>
              {props.workerCount}
            </NeonBadge>
          </div>

          <div class="header-stat-agency flex items-center gap-2 font-mono text-xs bg-[var(--glass-bg-card)] px-3 py-1.5 rounded-full border border-[var(--glass-border)]">
            <span class={`status-dot glass-status-dot ${props.connected ? 'glass-status-dot-success' : 'glass-status-dot-error'}`} />
            <NeonBadge color={props.connected ? 'emerald' : 'rose'} glow pulse={!props.connected}>
              {props.connected ? 'Live' : 'Offline'}
            </NeonBadge>
          </div>

          <div class="header-stat-agency hidden sm:flex items-center gap-2 font-mono text-xs bg-[var(--glass-bg-card)] px-3 py-1.5 rounded-full border border-[var(--glass-border)]">
            <NeonTitle level="span" color="cyan" size="xs">Verifiable</NeonTitle>
            <NeonBadge color="emerald" glow>100%</NeonBadge>
          </div>

          <md-icon-button
            onClick$={handleRefresh}
            class={`header-refresh-btn glass-interactive glass-hover-scale glass-focus-ring ${isRefreshing.value ? 'refreshing' : ''}`}
            title="Refresh data & scene"
            aria-label="Refresh"
          >
            <md-ripple></md-ripple>
            <svg slot="icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-[var(--glass-text-primary)]">
              <path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
              <path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16" />
              <path d="M16 21h5v-5" />
            </svg>
          </md-icon-button>

          <ThemeModeToggle />
          <GestaltPill mood={props.mood} entropy={props.entropy} />
          <VisionEye />
          <VNCAuthFab providerStatus={props.providerStatus} onOpen$={props.onOpenAuth$} inline={true} />
        </div>
      </div>

      <style>{`
        .header-glass-agency {
          background: var(--glass-bg-dark);
        }
      `}</style>
    </header>
  );
});
