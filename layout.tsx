/**
 * OMEGA SANDWICH LAYOUT
 * =====================
 * Implements the "Bread" (Header/Footer) around the "Meat" (Routes).
 * 
 * - Top Bread: HeaderOmega (Status, Auth, Theme)
 * - Meat: <Slot /> (The View)
 * - Bottom Bread: SuperMotdFooter (HeartbeatOmega, Bus Status)
 */

import { component$, Slot, useSignal, useVisibleTask$, useComputed$, useContextProvider, $ } from '@builder.io/qwik';
import { HeaderOmega } from '../components/layout/HeaderOmega';
import { SuperMotdFooter } from '../components/SuperMotdFooter';
import { VNCAuthOverlay } from '../components/VNCAuthOverlay';
import { ArtInjector } from '../components/art/ArtInjector';
import { GenerativeBackground } from '../components/art/GenerativeBackground';
import { LoadingStage } from '../components/LoadingStage';
import { DashboardLayoutContext, type ProviderStatusMap } from '../lib/state/dashboard_layout_context';


export default component$(() => {
  // --- OMEGA STATE (Lifted from Index) ---
  const connected = useSignal(false);
  const workerCount = useSignal(0);
  const entropy = useSignal(0.1); // Use signal for direct update
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

  return (
    <div class="h-screen bg-transparent flex flex-col overflow-hidden relative">
      <GenerativeBackground entropy={entropy.value} mood={mood.value} requestScene={true} />
      <ArtInjector />

      <div class="relative z-10 flex flex-col h-full">
        {/* TOP BREAD */}
        <LoadingStage id="comp:header">
          <HeaderOmega 
            connected={connected.value}
            workerCount={workerCount.value}
            mood={mood.value}
            entropy={entropy.value}
            providerStatus={providerStatus.value}
            onOpenAuth$={openAuthOverlay}
          />
        </LoadingStage>

        {/* THE MEAT */}
        <main class="flex-1 min-h-0 overflow-hidden relative">
           <Slot />
        </main>

        {/* BOTTOM BREAD */}
        <SuperMotdFooter />
      </div>

      {/* Global Overlays */}
      <VNCAuthOverlay
        open={authOverlayOpen.value}
        providerStatus={providerStatus.value}
        onClose$={closeAuthOverlay}
      />
    </div>
  );
});
