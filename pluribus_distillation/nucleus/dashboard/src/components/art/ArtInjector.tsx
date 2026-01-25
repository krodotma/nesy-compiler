import { component$, useVisibleTask$, useSignal, $ } from '@builder.io/qwik';
import { generateMaterial, applyMaterialToRoot, type MaterialDefinition } from '../../lib/art/visual-engine';

export const ArtInjector = component$(() => {
  // We use a signal to track the current ID for debugging/reactivity if needed
  const currentMaterialId = useSignal<string>('');
  const lastInjectionTime = useSignal<number>(Date.now());
  const lastMaterialId = useSignal<string>('');
  const keptEmitted = useSignal<boolean>(false);
  const breathingAngle = useSignal<number>(135);

  // Phase 2 Step 9: Breathing Gradients
  useVisibleTask$(({ cleanup }) => {
    let animationFrame: number;
    const animate = () => {
      breathingAngle.value = (breathingAngle.value + 0.05) % 360; // Approx 1deg per 20 seconds
      document.documentElement.style.setProperty('--art-gradient-angle', `${breathingAngle.value}deg`);
      animationFrame = requestAnimationFrame(animate);
    };
    animate();
    return () => cancelAnimationFrame(animationFrame);
  });

  // Helper: Emit preference to bus
  const emitPreference = $((status: 'skipped' | 'kept' | 'explicit', mat: MaterialDefinition) => {
    // 1. Implicit Metric (Phase 1 Step 1)
    if (status !== 'explicit') {
      fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: 'art.preference.implicit',
          kind: 'metric',
          level: 'info',
          actor: 'ArtInjector',
          data: { status, id: mat.id, name: mat.name, ts: Date.now() }
        })
      }).catch(() => {});
    }

    // 2. FalkorDB Sync (Phase 1 Step 2)
    // We emit user.preference.update which is caught by falkordb_bus_events.py
    fetch('/api/emit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic: 'user.preference.update',
        kind: 'request',
        level: 'info',
        actor: 'dashboard-user',
        payload: {
          chroma_id: mat.id,
          noise_intensity: mat.glass.noiseIntensity,
          gradient_angle: mat.glass.gradientAngle,
          material: mat // Full definition for context
        }
      })
    }).catch(() => {});
  });

  // Helper: Update Favicon (Phase 2 Step 13)
  const updateFavicon = $((color: string) => {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><circle cx="16" cy="16" r="14" fill="${color}" /></svg>`;
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    let link = document.querySelector('link[rel="icon"]');
    if (!link) {
      link = document.createElement('link');
      (link as HTMLLinkElement).rel = 'icon';
      document.head.appendChild(link);
    }
    link.setAttribute('href', url);
  });

  // The Core Injection Logic
  const injectMaterial = $((seed?: number) => {
    // 1. Generate (or could load from props)
    const mat = generateMaterial(seed);
    
    // 2. Track previous theme (Phase 1 Step 1)
    if (lastMaterialId.value) {
      const duration = Date.now() - lastInjectionTime.value;
      if (duration < 300000) {
        emitPreference('skipped', mat); // Implicitly skipping the previous one by accepting a new one early
      }
    }

    // 3. Apply to :root
    applyMaterialToRoot(mat);
    
    // 4. Update internal state
    lastMaterialId.value = currentMaterialId.value;
    currentMaterialId.value = mat.id;
    lastInjectionTime.value = Date.now();
    keptEmitted.value = false;

    // Phase 3 Step 16: Set Agent-specific colors in CSS
    const root = document.documentElement;
    root.style.setProperty('--agent-main-color', mat.colors.text);
    root.style.setProperty('--agent-claude-color', mat.colors.primary);
    root.style.setProperty('--agent-qwen-color', mat.colors.secondary);
    root.style.setProperty('--agent-gemini-color', mat.colors.primary);
    root.style.setProperty('--agent-codex-color', mat.colors.secondary);
    
    // 5. Persistence (SAGENT hook)
    localStorage.setItem('pluribus_visual_material', JSON.stringify(mat));

    // 6. Immediate Sync (Phase 1 Step 2)
    emitPreference('explicit', mat);

    // 7. Favicon Sync (Phase 2 Step 13)
    updateFavicon(mat.colors.primary);
    
    console.log(`[ArtInjector] Injected Material: ${mat.name} (${mat.id})`);
  });

  useVisibleTask$(({ track }) => {
    track(() => currentMaterialId.value);
    
    // Periodic check for "kept" status (Phase 1 Step 1)
    const timeout = setTimeout(() => {
      if (currentMaterialId.value && !keptEmitted.value) {
        // Recover full material from localStorage to get all fields
        const saved = localStorage.getItem('pluribus_visual_material');
        if (saved) {
          try {
            const mat = JSON.parse(saved) as MaterialDefinition;
            if (mat.id === currentMaterialId.value) {
              emitPreference('kept', mat);
              keptEmitted.value = true;
            }
          } catch (e) {}
        }
      }
    }, 300000); // 5 minutes

    return () => clearTimeout(timeout);
  });

  useVisibleTask$(({ cleanup }) => {
    // A. Initial Load
    // Try to recover last session's material for continuity, or generate new
    const saved = localStorage.getItem('pluribus_visual_material');
    if (saved) {
      try {
        const mat = JSON.parse(saved) as MaterialDefinition;
        applyMaterialToRoot(mat);
        currentMaterialId.value = mat.id;
        lastInjectionTime.value = Date.now();
      } catch (e) {
        injectMaterial();
      }
    } else {
      injectMaterial();
    }

    // Phase 1 Step 3: Performance Optimization (Web Worker)
    if (typeof Worker !== 'undefined') {
      const worker = new Worker(new URL('../../workers/chroma.worker.ts', import.meta.url), { type: 'module' });
      worker.postMessage({ type: 'GENERATE_REGISTRY' });
      worker.onmessage = (e) => {
        if (e.data.type === 'REGISTRY_GENERATED') {
          console.log(`[ArtInjector] Chroma Registry generated in background (${e.data.payload.duration.toFixed(2)}ms)`);
          // Note: In a real system, we'd sync this back to the singleton, 
          // but for now we just acknowledge it's ready.
        }
      };
      cleanup(() => worker.terminate());
    }

    // B. Bus Bridge Listener
    // This allows backend agents (Claude/SAGENT) to trigger visual updates via the Event Bus
    // The BusBridge component (elsewhere) translates WebSocket/SSE events to Window events.
    const updateHandler = (event: CustomEvent) => {
        const detail = event.detail || {};
        // If the agent provided a specific seed or parameters, use them
        // For now, we just re-roll with a new random seed
        injectMaterial(detail.seed ? Number(detail.seed) : Math.random());
    };

    window.addEventListener('pluribus:art:update', updateHandler as EventListener);
    window.addEventListener('pluribus:art:request', updateHandler as EventListener); // Alias

    cleanup(() => {
      window.removeEventListener('pluribus:art:update', updateHandler as EventListener);
      window.removeEventListener('pluribus:art:request', updateHandler as EventListener);
    });
  });

  // Render nothing - this is a logic-only component (headless)
  return <div style={{ display: 'none' }} data-material-id={currentMaterialId.value} />;
});
