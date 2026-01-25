import { component$, useVisibleTask$, useSignal } from '@builder.io/qwik';
import { fetchMetaSuggestion } from './src/skills/metaLearner';

interface ArtComposition {
  id: string;
  css?: string;
  tokens?: Record<string, string>;
  overlay?: string;
  shaderId?: string;
}

interface ArtItem {
  id: string;
  type: string;
  code_path: string;
  design_token?: string;
  content?: string; // If loaded
}

export const ArtInjector = component$(() => {
  const cssContent = useSignal<string>('');
  const overlayHtml = useSignal<string>('');

  useVisibleTask$(({ cleanup }) => {
    // --- Helper: Apply Composition ---
    const applyComposition = (comp: ArtComposition, mutationSeed: number) => {
      // 1. Inject CSS
      if (comp.css) cssContent.value = comp.css;

      // 2. Inject Tokens
      const root = document.documentElement;
      if (comp.tokens) {
        Object.entries(comp.tokens).forEach(([key, val]) => {
          root.style.setProperty(key, val);
          if (key === '--art-primary') {
            const raw = val.replace(/hsl\((.*)\)/, '$1').replace(/,/g, '');
            root.style.setProperty('--primary', raw);
            root.style.setProperty('--ring', raw);
          }
          if (key === '--art-secondary') {
            const raw = val.replace(/hsl\((.*)\)/, '$1').replace(/,/g, '');
            root.style.setProperty('--secondary', raw);
            root.style.setProperty('--accent', raw);
          }
        });
      }

      root.style.setProperty('--art-mutation-seed', mutationSeed.toString());
      if (comp.overlay) overlayHtml.value = comp.overlay;
    };

    // --- SIMPLE GRADIENT FALLBACK (Art Dept Disabled) ---
    const applyGradientFallback = () => {
      const root = document.documentElement;

      // Random chroma hue for accent (0-360)
      const randomHue = Math.floor(Math.random() * 360);

      // Light grey to dark grey gradient with subtle chroma in darker shade
      const gradientCss = `
        :root {
          --art-gradient-start: hsl(0 0% 92%);
          --art-gradient-mid: hsl(0 0% 75%);
          --art-gradient-end: hsl(${randomHue} 15% 25%);
          --art-primary: hsl(${randomHue} 40% 50%);
          --art-secondary: hsl(${(randomHue + 30) % 360} 30% 45%);
        }
        
        body {
          background: linear-gradient(
            145deg,
            var(--art-gradient-start) 0%,
            var(--art-gradient-mid) 40%,
            var(--art-gradient-end) 100%
          ) !important;
          background-attachment: fixed !important;
        }
        
        /* Disable any shader canvas that might be loading */
        #generative-bg-canvas,
        .generative-background,
        canvas[data-shader] {
          display: none !important;
        }
      `;

      // CRITICAL: Use DOM API to inject style, bypassing Qwik's JSX escaping
      const existingStyle = document.getElementById('art-gradient-fallback');
      if (existingStyle) existingStyle.remove();

      const styleEl = document.createElement('style');
      styleEl.id = 'art-gradient-fallback';
      styleEl.textContent = gradientCss;
      document.head.appendChild(styleEl);

      // Apply chroma tokens
      const primaryRaw = `${randomHue} 40% 50%`;
      const accentRaw = `${(randomHue + 30) % 360} 30% 45%`;
      root.style.setProperty('--primary', primaryRaw);
      root.style.setProperty('--ring', primaryRaw);
      root.style.setProperty('--secondary', accentRaw);
      root.style.setProperty('--accent', accentRaw);
      root.style.setProperty('--art-primary', `hsl(${primaryRaw})`);
      root.style.setProperty('--art-secondary', `hsl(${accentRaw})`);

      console.log('[ArtInjector] Applied gradient fallback with hue:', randomHue);
    };

    // Apply gradient immediately (Art Dept shaders disabled for performance)
    applyGradientFallback();

    // ... (Keep existing listener logic) ...
    const applyScene = (evt: any) => {
      const data = evt?.data || {};
      const tokens = data.tokens || {};
      const scene = data.scene || {};

      const primary = tokens.primary_color ? String(tokens.primary_color) : '';
      const accent = tokens.accent_color ? String(tokens.accent_color) : '';
      const blurPx = Number(tokens.blur_px || 0);
      const glassOpacity = Number(tokens.glass_opacity || 0);

      const root = document.documentElement;
      if (primary) root.style.setProperty('--art-primary', primary);
      if (accent) root.style.setProperty('--art-secondary', accent);
      if (primary) {
        const raw = primary.replace(/hsl\((.*)\)/, '$1').replace(/,/g, '').trim();
        root.style.setProperty('--primary', raw);
        root.style.setProperty('--ring', raw);
      }
      if (accent) {
        const raw = accent.replace(/hsl\((.*)\)/, '$1').replace(/,/g, '').trim();
        root.style.setProperty('--secondary', raw);
        root.style.setProperty('--accent', raw);
      }
      if (!Number.isNaN(blurPx) && blurPx > 0) root.style.setProperty('--art-blur-px', String(blurPx));
      if (!Number.isNaN(glassOpacity) && glassOpacity > 0) root.style.setProperty('--art-glass-opacity', String(glassOpacity));

      if (scene?.id) root.dataset.artSceneId = String(scene.id);
      if (scene?.name) root.dataset.artSceneName = String(scene.name);
    };

    const onArt = (ev: Event) => {
      const detail = (ev as CustomEvent).detail as any;
      if (detail?.topic === 'art.scene.change') applyScene(detail);
    };
    window.addEventListener('pluribus:art', onArt as any);
    cleanup(() => window.removeEventListener('pluribus:art', onArt as any));

    // --- MOOD-BASED DESIGN TOKEN APPLICATION ---
    const MOOD_COLORS: Record<string, { primary: string; accent: string }> = {
      calm: { primary: 'hsl(120 50% 50%)', accent: 'hsl(180 50% 50%)' },
      focused: { primary: 'hsl(220 70% 50%)', accent: 'hsl(200 60% 50%)' },
      anxious: { primary: 'hsl(0 70% 50%)', accent: 'hsl(30 60% 50%)' },
      chaotic: { primary: 'hsl(300 80% 50%)', accent: 'hsl(330 70% 50%)' },
      hyper: { primary: 'hsl(50 90% 55%)', accent: 'hsl(40 80% 50%)' },
      dormant: { primary: 'hsl(0 0% 45%)', accent: 'hsl(0 0% 55%)' },
    };

    const applyMood = (mood: string) => {
      const colors = MOOD_COLORS[mood] || MOOD_COLORS.calm;
      const root = document.documentElement;

      root.style.setProperty('--art-primary', colors.primary);
      root.style.setProperty('--art-secondary', colors.accent);

      const primaryRaw = colors.primary.replace(/hsl\((.*)\)/, '$1').replace(/,/g, '').trim();
      const accentRaw = colors.accent.replace(/hsl\((.*)\)/, '$1').replace(/,/g, '').trim();
      root.style.setProperty('--primary', primaryRaw);
      root.style.setProperty('--ring', primaryRaw);
      root.style.setProperty('--secondary', accentRaw);
      root.style.setProperty('--accent', accentRaw);

      root.dataset.artMood = mood;
    };

    const onMoodChange = (ev: Event) => {
      const detail = (ev as CustomEvent).detail || {};
      if (detail.mood) applyMood(detail.mood);
    };
    window.addEventListener('pluribus:mood:change', onMoodChange as any);
    cleanup(() => window.removeEventListener('pluribus:mood:change', onMoodChange as any));

  });

  return (
    <>
      <style dangerouslySetInnerHTML={cssContent.value} id="art-dept-injected-css" />
      <div
        id="art-dept-overlay-layer"
        class="fixed inset-0 pointer-events-none z-[9999] overflow-hidden mix-blend-overlay"
        dangerouslySetInnerHTML={overlayHtml.value}
      />
      <button
        className="ml-2 px-3 py-1 bg-indigo-600 text-white rounded"
        onClick={async () => {
          try {
            const suggestion = await fetchMetaSuggestion({ currentState: {} });
            // Apply tokens to root
            const root = document.documentElement;
            Object.entries(suggestion.tokens || {}).forEach(([k, v]) => {
              root.style.setProperty(k, v);
            });
          } catch (e) {
            console.error(e);
          }
        }}
      >
        Design Assist
      </button>
    </>
  );
});
