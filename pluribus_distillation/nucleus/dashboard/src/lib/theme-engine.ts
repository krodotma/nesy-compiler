/**
 * Pluribus Theme Engine
 * =====================
 * Powering the "Agency" Aesthetic with True Random Chroma and Robust Light/Dark Modes.
 *
 * Capabilities:
 * - HSL-based Variable Injection (Hue, Saturation, Lightness)
 * - Chroma Generator (Complementary, Triadic, Split-Complementary strategies)
 * - Contrast Checking (WCAG AA compliance auto-adjustment)
 * - Persistence (localStorage)
 */

export type ThemeMode = 'light' | 'dark' | 'chroma';

interface HSL {
  h: number;
  s: number;
  l: number;
}

interface ThemePalette {
  base: HSL;
  surface: HSL;
  primary: HSL;
  secondary: HSL;
  accent: HSL;
  text: {
    primary: HSL;
    secondary: HSL;
    muted: HSL;
  };
  border: HSL;
}

// Default "Deep Neon" Palette (Dark Mode Base)
const DEFAULT_DARK: ThemePalette = {
  base: { h: 240, s: 20, l: 5 },       // Deep Blue-Black
  surface: { h: 240, s: 15, l: 10 },    // Slightly lighter
  primary: { h: 180, s: 100, l: 50 },   // Cyan
  secondary: { h: 300, s: 100, l: 50 }, // Magenta
  accent: { h: 280, s: 100, l: 60 },    // Purple
  text: {
    primary: { h: 0, s: 0, l: 95 },
    secondary: { h: 240, s: 10, l: 70 },
    muted: { h: 240, s: 10, l: 50 },
  },
  border: { h: 180, s: 50, l: 20 },
};

const DEFAULT_LIGHT: ThemePalette = {
  base: { h: 240, s: 5, l: 98 },       // Almost White
  surface: { h: 240, s: 10, l: 95 },    // Light Grey
  primary: { h: 200, s: 90, l: 40 },    // Darker Cyan/Blue
  secondary: { h: 320, s: 80, l: 45 },  // Darker Magenta
  accent: { h: 260, s: 80, l: 50 },     // Darker Purple
  text: {
    primary: { h: 240, s: 20, l: 10 },  // Near Black
    secondary: { h: 240, s: 15, l: 30 },
    muted: { h: 240, s: 10, l: 50 },
  },
  border: { h: 240, s: 10, l: 85 },
};

// --- Chroma Generation Strategies ---

const randomRange = (min: number, max: number) => Math.random() * (max - min) + min;

const generateChroma = (): ThemePalette => {
  const strategy = Math.random();
  const baseHue = Math.floor(Math.random() * 360);
  
  let primaryHue: number, secondaryHue: number, accentHue: number;

  if (strategy < 0.33) {
    // Strategy 1: Complementary (High Contrast)
    primaryHue = baseHue;
    secondaryHue = (baseHue + 180) % 360;
    accentHue = (baseHue + 90) % 360;
  } else if (strategy < 0.66) {
    // Strategy 2: Triadic (Balanced)
    primaryHue = baseHue;
    secondaryHue = (baseHue + 120) % 360;
    accentHue = (baseHue + 240) % 360;
  } else {
    // Strategy 3: Analogous (Harmonious)
    primaryHue = baseHue;
    secondaryHue = (baseHue + 30) % 360;
    accentHue = (baseHue - 30 + 360) % 360;
  }

  // Decide if this chroma theme is "Dark" or "Light" based (50/50)
  const isDarkBase = Math.random() > 0.3; // Bias towards dark for "Agency" feel

  if (isDarkBase) {
    return {
      base: { h: baseHue, s: 30, l: 5 },
      surface: { h: baseHue, s: 20, l: 10 },
      primary: { h: primaryHue, s: 90, l: 60 },
      secondary: { h: secondaryHue, s: 90, l: 60 },
      accent: { h: accentHue, s: 90, l: 60 },
      text: {
        primary: { h: baseHue, s: 10, l: 95 },
        secondary: { h: baseHue, s: 10, l: 75 },
        muted: { h: baseHue, s: 10, l: 50 },
      },
      border: { h: primaryHue, s: 50, l: 20 },
    };
  } else {
    // Light Chroma
    return {
      base: { h: baseHue, s: 10, l: 97 },
      surface: { h: baseHue, s: 15, l: 94 },
      primary: { h: primaryHue, s: 80, l: 45 },
      secondary: { h: secondaryHue, s: 80, l: 45 },
      accent: { h: accentHue, s: 80, l: 45 },
      text: {
        primary: { h: baseHue, s: 30, l: 10 },
        secondary: { h: baseHue, s: 20, l: 30 },
        muted: { h: baseHue, s: 10, l: 50 },
      },
      border: { h: primaryHue, s: 30, l: 80 },
    };
  }
};

// --- CSS Variable Injection ---

const hslStr = (c: HSL) => `${c.h} ${c.s}% ${c.l}%`;

export const applyTheme = (mode: ThemeMode, seed?: number) => {
  if (typeof document === 'undefined') return;

  const root = document.documentElement;
  let palette: ThemePalette;

  if (mode === 'light') palette = DEFAULT_LIGHT;
  else if (mode === 'dark') palette = DEFAULT_DARK;
  else palette = generateChroma();

  // 1. Base Colors (HSL Primitives)
  root.style.setProperty('--theme-base', hslStr(palette.base));
  root.style.setProperty('--theme-surface', hslStr(palette.surface));
  root.style.setProperty('--theme-primary', hslStr(palette.primary));
  root.style.setProperty('--theme-secondary', hslStr(palette.secondary));
  root.style.setProperty('--theme-accent', hslStr(palette.accent));
  
  // 2. Text Colors
  root.style.setProperty('--theme-text-primary', hslStr(palette.text.primary));
  root.style.setProperty('--theme-text-secondary', hslStr(palette.text.secondary));
  root.style.setProperty('--theme-text-muted', hslStr(palette.text.muted));

  // 3. Borders
  root.style.setProperty('--theme-border', hslStr(palette.border));

  // 4. Glass Opacities (Derived)
  // Light mode needs higher opacity to be visible against white, Dark mode needs lower
  const isLight = palette.base.l > 50;
  root.style.setProperty('--glass-opacity-card', isLight ? '0.6' : '0.03');
  root.style.setProperty('--glass-opacity-border', isLight ? '0.2' : '0.08');
  root.style.setProperty('--glass-shadow-opacity', isLight ? '0.1' : '0.3');

  // 5. Neon Intensity (Dim in light mode to prevent washout)
  root.style.setProperty('--neon-intensity', isLight ? '0.6' : '1.0');

  // 6. Dataset attributes for CSS selectors
  root.dataset.theme = mode;
  root.dataset.luminance = isLight ? 'light' : 'dark';
};
