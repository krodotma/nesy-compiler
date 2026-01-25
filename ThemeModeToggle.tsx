import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import { applyTheme, isThemeMode, resolveSystemTheme, type ThemeMode } from '../lib/theme';

const THEME_STORAGE_KEY = 'pluribus.dashboard.theme';
const CHROMA_SEED_KEY = 'pluribus.dashboard.chroma_seed';

export const ThemeModeToggle = component$(() => {
  const mode = useSignal<ThemeMode>('dark');
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

    const resolved = storedMode === 'system' ? resolveSystemTheme(matchMedia) : storedMode;
    mode.value = resolved || 'dark';
    chromaSeed.value = storedSeed;
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

  const getButtonClass = (target: ThemeMode) =>
    `px-3 py-1 text-[11px] font-semibold rounded-full transition-all ${
      mode.value === target
        ? 'bg-background text-foreground shadow border border-border'
        : 'text-muted-foreground hover:text-foreground'
    }`;

  return (
    <div class="flex items-center gap-1 rounded-full border border-border bg-muted/40 p-1">
      <button class={getButtonClass('light')} onClick$={() => setMode('light')}>
        Light
      </button>
      <button class={getButtonClass('dark')} onClick$={() => setMode('dark')}>
        Dark
      </button>
      <button class={getButtonClass('chroma')} onClick$={() => setMode('chroma')}>
        Random Chroma
      </button>
    </div>
  );
});
