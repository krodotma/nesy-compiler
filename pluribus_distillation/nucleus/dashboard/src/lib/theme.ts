export type ThemeMode = 'light' | 'dark' | 'system' | 'chroma';
export type ThemeTokens = Record<string, string>;

export const THEME_TOKEN_KEYS = [
  '--background',
  '--foreground',
  '--card',
  '--card-foreground',
  '--popover',
  '--popover-foreground',
  '--primary',
  '--primary-foreground',
  '--secondary',
  '--secondary-foreground',
  '--muted',
  '--muted-foreground',
  '--accent',
  '--accent-foreground',
  '--destructive',
  '--destructive-foreground',
  '--border',
  '--input',
  '--ring',
] as const;

// Cyberpunk Glass Theme - Electric cyan primary, hot magenta accent
// Clean dark backgrounds with neon highlights
const DARK_TOKENS: ThemeTokens = {
  '--background': 'hsl(220 20% 8%)',        // Deep blue-black
  '--foreground': 'hsl(200 20% 95%)',       // Cool white
  '--card': 'hsl(220 18% 12%)',             // Slightly lighter card bg
  '--card-foreground': 'hsl(200 20% 95%)',
  '--popover': 'hsl(220 18% 14%)',
  '--popover-foreground': 'hsl(200 20% 95%)',
  '--primary': 'hsl(185 100% 50%)',         // Electric cyan
  '--primary-foreground': 'hsl(220 20% 8%)',
  '--secondary': 'hsl(270 70% 60%)',        // Soft purple
  '--secondary-foreground': 'hsl(200 20% 95%)',
  '--muted': 'hsl(220 15% 18%)',
  '--muted-foreground': 'hsl(220 15% 65%)',
  '--accent': 'hsl(325 85% 55%)',           // Hot magenta
  '--accent-foreground': 'hsl(200 20% 95%)',
  '--destructive': 'hsl(0 80% 55%)',
  '--destructive-foreground': 'hsl(0 0% 100%)',
  '--border': 'hsl(220 15% 22%)',           // Subtle borders
  '--input': 'hsl(220 15% 16%)',
  '--ring': 'hsl(185 100% 50%)',            // Cyan focus ring
};

const LIGHT_TOKENS: ThemeTokens = {
  '--background': 'hsl(210 30% 98%)',
  '--foreground': 'hsl(222 35% 12%)',
  '--card': 'hsl(0 0% 100%)',
  '--card-foreground': 'hsl(222 35% 12%)',
  '--popover': 'hsl(0 0% 100%)',
  '--popover-foreground': 'hsl(222 35% 12%)',
  '--primary': 'hsl(210 90% 45%)',
  '--primary-foreground': 'hsl(0 0% 100%)',
  '--secondary': 'hsl(275 70% 60%)',
  '--secondary-foreground': 'hsl(0 0% 100%)',
  '--muted': 'hsl(210 24% 90%)',
  '--muted-foreground': 'hsl(215 18% 35%)',
  '--accent': 'hsl(330 85% 55%)',
  '--accent-foreground': 'hsl(0 0% 100%)',
  '--destructive': 'hsl(0 75% 52%)',
  '--destructive-foreground': 'hsl(0 0% 100%)',
  '--border': 'hsl(210 20% 86%)',
  '--input': 'hsl(210 20% 92%)',
  '--ring': 'hsl(210 90% 45%)',
};

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

const normalizeHue = (hue: number) => {
  const mod = hue % 360;
  return mod < 0 ? mod + 360 : mod;
};

const hsl = (hue: number, sat: number, light: number) =>
  `hsl(${Math.round(normalizeHue(hue))} ${Math.round(clamp(sat, 0, 100))}% ${Math.round(
    clamp(light, 0, 100)
  )}%)`;

const createRng = (seed?: number) => {
  if (seed === undefined || seed === null) {
    return Math.random;
  }

  let state = seed;
  if (seed > 0 && seed < 1) {
    state = seed * 0xffffffff;
  }

  let t = (state >>> 0) || 0x6d2b79f5;
  return () => {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
};

const hueDistance = (a: number, b: number) => {
  const diff = Math.abs(normalizeHue(a) - normalizeHue(b)) % 360;
  return Math.min(diff, 360 - diff);
};

export const generateChromaTokens = (seed?: number): ThemeTokens => {
  const rng = createRng(seed);
  const anchors = [200, 220, 250, 280, 320, 20, 40, 90, 140, 170];
  const anchor = anchors[Math.floor(rng() * anchors.length)];
  const baseHue = normalizeHue(anchor + (rng() - 0.5) * 14);

  const analogShift = (rng() < 0.5 ? 1 : -1) * (28 + rng() * 18);
  const secondaryHue = normalizeHue(baseHue + analogShift);
  let accentHue = normalizeHue(baseHue + 150 + rng() * 45);
  if (hueDistance(accentHue, secondaryHue) < 28) {
    accentHue = normalizeHue(accentHue + 40);
  }

  const bgHue = normalizeHue(baseHue + 210 + rng() * 20);
  const bgSat = 12 + rng() * 8;
  const bgLight = 10 + rng() * 6;
  const cardLight = bgLight + 4;
  const borderLight = bgLight + 18;
  const mutedLight = bgLight + 10;

  const primarySat = 74 + rng() * 14;
  const primaryLight = 58 + rng() * 8;
  const secondarySat = 62 + rng() * 16;
  const secondaryLight = 56 + rng() * 8;
  const accentSat = 70 + rng() * 18;
  const accentLight = 60 + rng() * 10;

  return {
    '--background': hsl(bgHue, bgSat, bgLight),
    '--foreground': hsl(bgHue, 18, 96),
    '--card': hsl(bgHue, bgSat + 4, cardLight),
    '--card-foreground': hsl(bgHue, 16, 96),
    '--popover': hsl(bgHue, bgSat + 4, cardLight),
    '--popover-foreground': hsl(bgHue, 16, 96),
    '--primary': hsl(baseHue, primarySat, primaryLight),
    '--primary-foreground': hsl(bgHue, 20, 12),
    '--secondary': hsl(secondaryHue, secondarySat, secondaryLight),
    '--secondary-foreground': hsl(bgHue, 20, 12),
    '--muted': hsl(bgHue, bgSat + 8, mutedLight),
    '--muted-foreground': hsl(bgHue, 12, 72),
    '--accent': hsl(accentHue, accentSat, accentLight),
    '--accent-foreground': hsl(bgHue, 18, 12),
    '--destructive': '0 85% 60%',
    '--destructive-foreground': '0 0% 100%',
    '--border': hsl(bgHue, bgSat + 6, borderLight),
    '--input': hsl(bgHue, bgSat + 6, borderLight - 2),
    '--ring': hsl(baseHue, primarySat, primaryLight),
  };
};

export const resolveSystemTheme = (
  matchMedia?: (query: string) => { matches: boolean }
): 'light' | 'dark' => {
  if (!matchMedia) return 'dark';
  return matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
};

export const getThemeTokens = (
  mode: ThemeMode,
  options?: { seed?: number; matchMedia?: (query: string) => { matches: boolean } }
): ThemeTokens => {
  if (mode === 'light') return { ...LIGHT_TOKENS };
  if (mode === 'dark') return { ...DARK_TOKENS };
  if (mode === 'chroma') return generateChromaTokens(options?.seed);
  const resolved = resolveSystemTheme(options?.matchMedia);
  return resolved === 'light' ? { ...LIGHT_TOKENS } : { ...DARK_TOKENS };
};

export type ThemeRoot = {
  dataset?: Record<string, string>;
  style: {
    setProperty: (key: string, value: string) => void;
    removeProperty?: (key: string) => void;
  };
};

export const applyThemeTokens = (root: ThemeRoot, tokens: ThemeTokens) => {
  Object.entries(tokens).forEach(([key, value]) => {
    root.style.setProperty(key, value);
  });
};

export const clearThemeTokens = (root: ThemeRoot) => {
  THEME_TOKEN_KEYS.forEach((key) => {
    if (root.style.removeProperty) root.style.removeProperty(key);
    else root.style.setProperty(key, '');
  });
};

export const applyTheme = (
  mode: ThemeMode,
  options?: {
    root?: ThemeRoot;
    seed?: number;
    matchMedia?: (query: string) => { matches: boolean };
  }
) => {
  const root =
    options?.root ??
    (typeof document !== 'undefined' ? (document.documentElement as ThemeRoot) : undefined);

  if (!root) {
    return {
      mode,
      resolvedMode: mode === 'system' ? resolveSystemTheme(options?.matchMedia) : mode,
      tokens: getThemeTokens(mode, options),
    };
  }

  if (mode === 'chroma') {
    const tokens = generateChromaTokens(options?.seed);
    if (root.dataset) root.dataset.theme = 'chroma';
    applyThemeTokens(root, tokens);
    return { mode, resolvedMode: 'chroma', tokens };
  }

  const resolvedMode = mode === 'system' ? resolveSystemTheme(options?.matchMedia) : mode;
  if (root.dataset) root.dataset.theme = resolvedMode;
  clearThemeTokens(root);
  const tokens = resolvedMode === 'light' ? { ...LIGHT_TOKENS } : { ...DARK_TOKENS };
  return { mode, resolvedMode, tokens };
};

export const isThemeMode = (value: string | null | undefined): value is ThemeMode =>
  value === 'light' || value === 'dark' || value === 'system' || value === 'chroma';
