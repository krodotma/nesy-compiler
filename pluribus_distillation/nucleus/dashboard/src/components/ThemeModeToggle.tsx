import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { applyTheme, isThemeMode, resolveSystemTheme, type ThemeMode } from '../lib/theme';
import { Button } from './ui/Button';

const THEME_STORAGE_KEY = 'pluribus.dashboard.theme';
const CHROMA_SEED_KEY = 'pluribus.dashboard.chroma_seed';

export const ThemeModeToggle = component$(() => {
  // Default to 'chroma' to showcase the Art Dept V2 upgrades immediately
  const mode = useSignal<ThemeMode>('chroma');
  const chromaSeed = useSignal<number | null>(null);

  useVisibleTask$(() => {
    let storedMode: ThemeMode | null = null;
    let storedSeed: number | null = null;

    try {
      const rawMode = localStorage.getItem(THEME_STORAGE_KEY);
      if (isThemeMode(rawMode)) storedMode = rawMode;
      const rawSeed = localStorage.getItem(CHROMA_SEED_KEY);
      if (rawSeed) {
        const parsed = Number(rawSeed);
        storedSeed = Number.isFinite(parsed) ? parsed : null;
      }
    } catch {
      storedMode = null;
      storedSeed = null;
    }

    // Prefer stored mode, but default to 'chroma' if none set (first visit)
    const resolved = storedMode || 'chroma';
    mode.value = resolved;
    chromaSeed.value = storedSeed;
    console.log('[Theme] Initializing mode:', resolved);
    applyTheme(mode.value, { seed: chromaSeed.value ?? undefined, matchMedia });
  });

  const persist = $((next: ThemeMode, seed: number | null) => {
    try {
      localStorage.setItem(THEME_STORAGE_KEY, next);
      if (seed === null || seed === undefined) localStorage.removeItem(CHROMA_SEED_KEY);
      else localStorage.setItem(CHROMA_SEED_KEY, String(seed));
    } catch {
      // ignore persistence failures
    }
  });

  const setMode = $((next: ThemeMode) => {
    let seed = chromaSeed.value;
    if (next === 'chroma') {
      seed = __E2E__ ? 0.42 : Math.random();
      chromaSeed.value = seed;
      
      // Dispatch request to Art Director via GenerativeBackground bridge
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('pluribus:art:request', { detail: { reason: 'user_chroma_toggle' } })
        );
      }
    } else {
      seed = null;
    }
    mode.value = next;
    applyTheme(next, { seed: seed ?? undefined, matchMedia });
    persist(next, seed);
  });

  return (
    <div class="flex items-center gap-1 rounded-full border border-border bg-muted/40 p-1 backdrop-blur-md backdrop-saturate-150 shadow-lg">
      <Button 
        variant={mode.value === 'light' ? 'tonal' : 'text'} 
        onClick$={() => setMode('light')}
        class="h-6 text-[11px] px-3 rounded-full"
      >
        Light
      </Button>
      <Button 
        variant={mode.value === 'dark' ? 'tonal' : 'text'} 
        onClick$={() => setMode('dark')}
        class="h-6 text-[11px] px-3 rounded-full"
      >
        Dark
      </Button>
      <Button 
        variant={mode.value === 'chroma' ? 'tonal' : 'text'} 
        onClick$={() => setMode('chroma')}
        class="h-6 text-[11px] px-3 rounded-full"
      >
        Random Chroma
      </Button>
    </div>
  );
});
