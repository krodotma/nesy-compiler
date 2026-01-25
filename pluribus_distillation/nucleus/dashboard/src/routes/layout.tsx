/**
 * OMEGA SANDWICH LAYOUT
 * =====================
 * Implements the "Bread" (Header/Footer) around the "Meat" (Routes).
 *
 * - Top Bread: HeaderOmega (Status, Auth, Theme)
 * - Meat: <Slot /> (The View)
 * - Bottom Bread: SuperMotdFooter (HeartbeatOmega, Bus Status)
 *
 * Glass System Integration (M3 + Glassmorphism):
 * - Uses glass-tokens.css for unified visual language
 * - Implements glass-aurora-bg for subtle animated background
 * - Applies glass-noise for texture depth
 * - Provides glass-page-transition hooks for route changes
 * - Supports IntersectionObserver-based glass-reveal animations
 */

import { component$, Slot, useSignal, useVisibleTask$, useComputed$, useContextProvider, $, useStore } from '@builder.io/qwik';
import { HeaderOmega } from '../components/layout/HeaderOmega';
import { SuperMotdFooter } from '../components/SuperMotdFooter';
import { VNCAuthOverlay } from '../components/VNCAuthOverlay';
import { ArtInjector } from '../components/art/ArtInjector';
// GenerativeBackground lazy-loaded to defer 673KB three.js bundle
import { LazyGenerativeBackground } from '../components/LazyGenerativeBackground';
import { ShaderControl } from '../components/art/ShaderControl';
import { LoadingStage } from '../components/LoadingStage';
import { DashboardLayoutContext, type ProviderStatusMap } from '../lib/state/dashboard_layout_context';
import { IPERoot } from '../components/ipe/IPERoot';
import { VoiceProvider } from '../lib/auralux/use-voice';
import { LazyVoiceOverlay } from '../components/LazyVoiceOverlay';


export default component$(() => {
  // --- OMEGA STATE (Lifted from Index) ---
  const connected = useSignal(false);
  const workerCount = useSignal(0);
  const entropy = useSignal(0.1); // Use signal for direct update
  const flowMode = useSignal<'m' | 'A'>('m');
  const mood = useComputed$(() => {
    if (workerCount.value > 2) return 'focused';
    // Logic for anxious/calm could be derived from entropy if needed
    return 'calm';
  });

  // Auth State
  const authOverlayOpen = useSignal(false);
  const providerStatus = useSignal<ProviderStatusMap>({});

  useContextProvider(DashboardLayoutContext, {
    authOverlayOpen,
    providerStatus,
    flowMode,
    setFlowMode$: $((mode: 'm' | 'A') => {
      flowMode.value = mode;
    }),
  });

  // Connect to Omega Worker Broadcast
  useVisibleTask$(({ cleanup }) => {
    const channel = new BroadcastChannel('pluribus-omega');
    channel.onmessage = (ev) => {
      if (ev.data.type === 'OMEGA_TICK') {
        const state = ev.data.state;
        connected.value = state.connected;
        workerCount.value = state.metrics.workerCount;
        entropy.value = state.entropy;
      }
    };
    return () => channel.close();
  });

  useVisibleTask$(({ cleanup }) => {
    if (__E2E__) return;
    if (typeof Worker === 'undefined') return;
    const worker = new Worker(new URL('../workers/shadow.worker.ts', import.meta.url), { type: 'module' });
    worker.postMessage({ type: 'INIT' });
    worker.postMessage({ type: 'PREFETCH', payload: { views: ['sota', 'git', 'browser'] } });
    cleanup(() => worker.terminate());
  });

  // Handlers
  const openAuthOverlay = $(() => authOverlayOpen.value = true);
  const closeAuthOverlay = $(() => authOverlayOpen.value = false);

  // --- Glass Page Transition State ---
  const pageTransition = useStore({
    transitioning: false,
    direction: 'enter' as 'enter' | 'exit',
  });

  // --- IntersectionObserver for glass-reveal elements (Step 51) ---
  useVisibleTask$(({ cleanup }) => {
    // Selector for all glass-reveal variants
    const REVEAL_SELECTOR = '.glass-reveal, .glass-reveal-left, .glass-reveal-right, .glass-reveal-scale';

    // Setup IntersectionObserver for glass-reveal animations
    const revealObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            entry.target.setAttribute('data-visible', 'true');
          }
        });
      },
      {
        root: null,
        rootMargin: '0px 0px -50px 0px', // Trigger 50px before element enters viewport
        threshold: 0.1,
      }
    );

    // Initial observation of existing elements
    const observeRevealElements = () => {
      const revealElements = document.querySelectorAll(REVEAL_SELECTOR);
      revealElements.forEach((el) => {
        // Only observe elements that haven't been revealed yet
        if (!el.classList.contains('visible')) {
          revealObserver.observe(el);
        }
      });
    };

    observeRevealElements();

    // MutationObserver to handle dynamically added glass-reveal elements
    const mutationObserver = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        mutation.addedNodes.forEach((node) => {
          if (node instanceof Element) {
            // Check if the added node itself matches
            if (node.matches(REVEAL_SELECTOR)) {
              revealObserver.observe(node);
            }
            // Check descendants
            node.querySelectorAll(REVEAL_SELECTOR).forEach((el) => {
              if (!el.classList.contains('visible')) {
                revealObserver.observe(el);
              }
            });
          }
        });
      });
    });

    mutationObserver.observe(document.body, {
      childList: true,
      subtree: true,
    });

    cleanup(() => {
      revealObserver.disconnect();
      mutationObserver.disconnect();
    });
  });

  return (
    <VoiceProvider>
      <div class="h-screen flex flex-col overflow-hidden relative glass-layout-root">
        {/* Layer 0: Aurora gradient background (subtle animated depth) */}
        <div
          class="glass-aurora-bg fixed inset-0 z-0 pointer-events-none"
          aria-hidden="true"
        />

        {/* Layer 1: Noise texture overlay (glass depth effect) */}
        <div
          class="glass-layout-noise fixed inset-0 z-[1] pointer-events-none opacity-[0.03]"
          aria-hidden="true"
        />

        {/* Layer 1.5: Glass atmosphere (subtle radial gradients for depth) */}
        <div
          class="glass-layout-atmosphere fixed inset-0 z-[2] pointer-events-none"
          aria-hidden="true"
        />

        {/* Layer 2: Generative shader background (lazy-loaded to defer three.js) */}
        <LazyGenerativeBackground entropy={entropy.value} mood={mood.value} requestScene={true} />
        <ArtInjector />

        {/* Layer 3: Content wrapper with glass atmosphere */}
        <div class="relative z-10 flex flex-col h-full glass-content-wrapper">
          {/* TOP BREAD - Header */}
          <LoadingStage id="comp:header">
            <HeaderOmega
              connected={connected.value}
              workerCount={workerCount.value}
              mood={mood.value}
              entropy={entropy.value}
              providerStatus={providerStatus.value}
              onOpenAuth$={openAuthOverlay}
              navItems={[
                { label: 'Portal', href: '/portal' },
                { label: 'ARK', href: '/ark', activeClass: 'text-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.5)]' },
                { label: 'Terminals', href: '/terminals' }
              ]}
            />
          </LoadingStage>

          {/* THE MEAT - Main content with page transition support */}
          <main
            class={[
              'flex-1 min-h-0 overflow-hidden relative',
              'glass-main-content',
              pageTransition.transitioning && pageTransition.direction === 'enter'
                ? 'glass-page-transition-enter'
                : '',
              pageTransition.transitioning && pageTransition.direction === 'exit'
                ? 'glass-page-transition-exit'
                : '',
            ].filter(Boolean).join(' ')}
          >
            <Slot />
          </main>

          {/* BOTTOM BREAD - Footer */}
          <SuperMotdFooter />
        </div>

        {/* Global Overlays */}
        <IPERoot enabled={true} />
        <LazyVoiceOverlay />
        <VNCAuthOverlay
          open={authOverlayOpen.value}
          providerStatus={providerStatus.value}
          onClose$={closeAuthOverlay}
        />

        {/* Shader A/B Testing Controls */}
        <ShaderControl />
      </div>
    </VoiceProvider>
  );
});
