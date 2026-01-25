/**
 * Qwik City Root - Entry point for the Qwik application
 *
 * Initializes:
 * - Error telemetry collector (streams all client errors to bus)
 * - Service worker registration
 */

import { component$, useVisibleTask$ } from '@builder.io/qwik';
import {
  QwikCityProvider,
  RouterOutlet,
} from '@builder.io/qwik-city';
import { LoadingOverlay } from './components/LoadingOverlay';
import { ArtDeptInjector } from './components/art/ArtDeptInjector';
import { TokenKernelProvider } from './components/TokenKernelProvider';
import { DialogosWidget } from './components/dialogos/DialogosWidget';
// VoiceProvider disabled until auralux is properly bundled
// import { VoiceProvider } from './components/auralux/VoiceProvider';
import './lib/telemetry/LoadingRegistry';
import { tracker } from './lib/telemetry/verbose-tracker';

import './global.css';

export default component$(() => {

  // Initialize client-side error telemetry (deferred to reduce boot-time cost).
  useVisibleTask$(() => {
    // Ensure trackers are exposed to window for LoadingOverlay
    // Don't create a stub - let the verbose-tracker module create the real instance
    // The module's window-first singleton will handle this
    (window as any).__verboseTracker?.entry("sys:telemetry");
    (window as any).__loadingRegistry?.entry("hook:telemetry");
    // Version marker for PBTEST verification
    console.log('%c[PLURIBUS] v2025.12.27.007 | Load: ' + Math.round(performance.now()) + 'ms', 'color: #0f0; font-weight: bold');

    try {
      document.documentElement.dataset.qwikReady = '1';
    } catch {
      // ignore
    }

    const init = async () => {
      try {
        // Initialize error telemetry
        const { initErrorCollector } = await import('./lib/telemetry');
        initErrorCollector({
          endpoint: '/api/emit',
          batchSize: 3,
          flushInterval: 2000,
          interceptConsole: import.meta.env.DEV,
          interceptFetch: import.meta.env.DEV,
          actor: 'dashboard-client',
        });

        // Initialize load timing collector (captures Web Vitals, Navigation Timing)
        const { initLoadTiming, markHydrationEnd } = await import('./lib/telemetry/load-timing');
        initLoadTiming();
        markHydrationEnd(); // Mark Qwik hydration complete
        (window as any).__loadingRegistry?.exit("hook:telemetry");
        (window as any).__verboseTracker?.exit("sys:telemetry");
      } catch {
        (window as any).__loadingRegistry?.exit("hook:telemetry", "telemetry init failed");
        (window as any).__verboseTracker?.exit("sys:telemetry");
      }
    };

    const ric = (globalThis as any).requestIdleCallback as
      | undefined
      | ((cb: () => void, opts?: { timeout?: number }) => void);
    if (ric) ric(init, { timeout: 2000 });
    else setTimeout(init, 1200);
  });

  useVisibleTask$(() => {
    const registry = (window as any).__loadingRegistry;
    const verboseTracker = (window as any).__verboseTracker;
    verboseTracker?.entry('sys:service-worker');
    registry?.entry('hook:service-worker');
    if (typeof __E2E__ !== 'undefined' && __E2E__) {
      registry?.exit('hook:service-worker');
      verboseTracker?.exit('sys:service-worker');
      return;
    }
    if (!('serviceWorker' in navigator)) {
      registry?.exit('hook:service-worker');
      verboseTracker?.exit('sys:service-worker');
      return;
    }

    const swEnabled = import.meta.env.DEV;
    if (!swEnabled) {
      const unregister = async () => {
        try {
          const regs = await navigator.serviceWorker.getRegistrations();
          await Promise.all(regs.map((reg) => reg.unregister()));
        } catch {
          // ignore
        }
        try {
          if ('caches' in window) {
            const keys = await caches.keys();
            await Promise.all(keys.map((key) => caches.delete(key)));
          }
        } catch {
          // ignore
        }
      };
      unregister()
        .then(() => {
          registry?.exit('hook:service-worker');
          verboseTracker?.exit('sys:service-worker');
        })
        .catch(() => {
          registry?.exit('hook:service-worker');
          verboseTracker?.exit('sys:service-worker');
        });
      return;
    }

    navigator.serviceWorker.register('/service-worker.js')
      .then(() => navigator.serviceWorker.ready)
      .then(() => {
        registry?.exit('hook:service-worker');
        verboseTracker?.exit('sys:service-worker');
      })
      .catch((err) => {
        registry?.exit('hook:service-worker', String(err));
        verboseTracker?.exit('sys:service-worker');
      });
  });

  return (
    <QwikCityProvider>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        {/* RANDOM CHROMA - Pick uniform palette per session/reload */}
        <script dangerouslySetInnerHTML={`
(function(){
  // 12 cohesive palettes - each has harmonious success/warning/accent/info colors
  var palettes = [
    // Cyberpunk Cyan - electric blues
    {name:'cyber',s:'185 85% 45%',sl:'185 75% 55%',sd:'185 90% 35%',w:'50 90% 55%',wl:'50 80% 65%',wd:'50 95% 45%',a:'320 80% 55%',al:'320 70% 65%',ad:'320 85% 45%',i:'200 90% 50%',il:'200 80% 60%',id:'200 95% 40%'},
    // Ocean Depths - teals and aquas
    {name:'ocean',s:'175 70% 40%',sl:'175 60% 50%',sd:'175 80% 30%',w:'40 85% 50%',wl:'40 75% 60%',wd:'40 90% 40%',a:'260 65% 55%',al:'260 55% 65%',ad:'260 75% 45%',i:'195 85% 45%',il:'195 75% 55%',id:'195 90% 35%'},
    // Sunset Fire - warm oranges and reds
    {name:'sunset',s:'160 70% 45%',sl:'160 60% 55%',sd:'160 80% 35%',w:'25 90% 55%',wl:'25 80% 65%',wd:'25 95% 45%',a:'340 75% 55%',al:'340 65% 65%',ad:'340 85% 45%',i:'45 85% 55%',il:'45 75% 65%',id:'45 90% 45%'},
    // Forest Emerald - greens and earth tones
    {name:'forest',s:'145 65% 40%',sl:'145 55% 50%',sd:'145 75% 30%',w:'35 80% 50%',wl:'35 70% 60%',wd:'35 90% 40%',a:'280 55% 50%',al:'280 45% 60%',ad:'280 65% 40%',i:'170 75% 45%',il:'170 65% 55%',id:'170 85% 35%'},
    // Neon Nights - vibrant magentas
    {name:'neon',s:'170 90% 45%',sl:'170 80% 55%',sd:'170 95% 35%',w:'55 95% 50%',wl:'55 85% 60%',wd:'55 100% 40%',a:'310 90% 55%',al:'310 80% 65%',ad:'310 95% 45%',i:'190 95% 50%',il:'190 85% 60%',id:'190 100% 40%'},
    // Royal Purple - deep violets
    {name:'royal',s:'165 70% 45%',sl:'165 60% 55%',sd:'165 80% 35%',w:'45 85% 55%',wl:'45 75% 65%',wd:'45 90% 45%',a:'270 75% 55%',al:'270 65% 65%',ad:'270 85% 45%',i:'220 80% 55%',il:'220 70% 65%',id:'220 90% 45%'},
    // Arctic Ice - cool blues and silvers
    {name:'arctic',s:'190 60% 50%',sl:'190 50% 60%',sd:'190 70% 40%',w:'50 70% 55%',wl:'50 60% 65%',wd:'50 80% 45%',a:'240 55% 60%',al:'240 45% 70%',ad:'240 65% 50%',i:'205 75% 55%',il:'205 65% 65%',id:'205 85% 45%'},
    // Golden Hour - warm ambers
    {name:'golden',s:'155 65% 45%',sl:'155 55% 55%',sd:'155 75% 35%',w:'40 95% 50%',wl:'40 85% 60%',wd:'40 100% 40%',a:'15 80% 55%',al:'15 70% 65%',ad:'15 90% 45%',i:'60 80% 50%',il:'60 70% 60%',id:'60 90% 40%'},
    // Synthwave - retro 80s
    {name:'synth',s:'180 75% 45%',sl:'180 65% 55%',sd:'180 85% 35%',w:'45 90% 55%',wl:'45 80% 65%',wd:'45 95% 45%',a:'295 80% 55%',al:'295 70% 65%',ad:'295 90% 45%',i:'205 85% 50%',il:'205 75% 60%',id:'205 90% 40%'},
    // Mint Fresh - light greens
    {name:'mint',s:'160 55% 50%',sl:'160 45% 60%',sd:'160 65% 40%',w:'50 75% 55%',wl:'50 65% 65%',wd:'50 85% 45%',a:'290 50% 55%',al:'290 40% 65%',ad:'290 60% 45%',i:'175 65% 50%',il:'175 55% 60%',id:'175 75% 40%'},
    // Rose Gold - pinks and coppers
    {name:'rose',s:'170 60% 45%',sl:'170 50% 55%',sd:'170 70% 35%',w:'30 80% 55%',wl:'30 70% 65%',wd:'30 90% 45%',a:'350 65% 55%',al:'350 55% 65%',ad:'350 75% 45%',i:'340 55% 55%',il:'340 45% 65%',id:'340 65% 45%'},
    // Electric Lime - vivid greens
    {name:'lime',s:'140 80% 45%',sl:'140 70% 55%',sd:'140 90% 35%',w:'55 90% 50%',wl:'55 80% 60%',wd:'55 95% 40%',a:'280 70% 55%',al:'280 60% 65%',ad:'280 80% 45%',i:'100 75% 45%',il:'100 65% 55%',id:'100 85% 35%'}
  ];
  var p = palettes[Math.floor(Math.random() * palettes.length)];
  var r = document.documentElement;
  r.style.setProperty('--chroma-success', p.s);
  r.style.setProperty('--chroma-success-light', p.sl);
  r.style.setProperty('--chroma-success-dark', p.sd);
  r.style.setProperty('--chroma-warning', p.w);
  r.style.setProperty('--chroma-warning-light', p.wl);
  r.style.setProperty('--chroma-warning-dark', p.wd);
  r.style.setProperty('--chroma-accent', p.a);
  r.style.setProperty('--chroma-accent-light', p.al);
  r.style.setProperty('--chroma-accent-dark', p.ad);
  r.style.setProperty('--chroma-info', p.i);
  r.style.setProperty('--chroma-info-light', p.il);
  r.style.setProperty('--chroma-info-dark', p.id);
  r.dataset.chromaPalette = p.name;
  console.log('[Chroma] Palette:', p.name);
})();
`} />
        {/* Aggressive cache-busting for development only */}
        {import.meta.env.DEV && (
          <>
            <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate" />
            <meta http-equiv="Pragma" content="no-cache" />
            <meta http-equiv="Expires" content="0" />
          </>
        )}
        <link rel="manifest" href="/manifest.json" />
        <title>Pluribus Dashboard</title>
      </head>
      <body lang="en" class="bg-background text-foreground">
        <TokenKernelProvider>
          <LoadingOverlay />
          <ArtDeptInjector />
          <a
            href="#main"
            class="sr-only focus:not-sr-only focus:fixed focus:top-2 focus:left-2 focus:z-50 focus:rounded focus:bg-black/80 focus:px-3 focus:py-2 focus:text-white"
          >
            Skip to content
          </a>
          <main id="main" tabIndex={-1} class="min-h-screen">
            {/* VoiceProvider disabled until auralux is properly bundled */}
            <RouterOutlet />
          </main>
          {/* Unified Epistemic Ingress */}
          <DialogosWidget />
        </TokenKernelProvider>
      </body>
    </QwikCityProvider>
  );
});
