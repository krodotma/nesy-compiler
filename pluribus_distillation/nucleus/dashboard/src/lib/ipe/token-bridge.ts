/**
 * IPE Token Bridge
 *
 * Bridges between IPE context CSS variables and tweakcn's ThemeStyleProps format.
 * Handles bidirectional conversion:
 *   IPEContext.cssVariables ←→ ThemeStyleProps
 */

import type { IPEContext } from './context-capture';

// ============================================================================
// Types (aligned with tweakcn's ThemeStyleProps)
// ============================================================================

export interface ThemeStyleProps {
  // Colors
  background: string;
  foreground: string;
  card: string;
  'card-foreground': string;
  popover: string;
  'popover-foreground': string;
  primary: string;
  'primary-foreground': string;
  secondary: string;
  'secondary-foreground': string;
  muted: string;
  'muted-foreground': string;
  accent: string;
  'accent-foreground': string;
  destructive: string;
  'destructive-foreground': string;
  border: string;
  input: string;
  ring: string;

  // Chart colors
  'chart-1': string;
  'chart-2': string;
  'chart-3': string;
  'chart-4': string;
  'chart-5': string;

  // Sidebar
  sidebar: string;
  'sidebar-foreground': string;
  'sidebar-primary': string;
  'sidebar-primary-foreground': string;
  'sidebar-accent': string;
  'sidebar-accent-foreground': string;
  'sidebar-border': string;
  'sidebar-ring': string;

  // Typography
  'font-sans': string;
  'font-serif': string;
  'font-mono': string;
  'letter-spacing': string;

  // Dimensions
  radius: string;
  spacing?: string;

  // Shadow
  'shadow-color': string;
  'shadow-opacity': string;
  'shadow-blur': string;
  'shadow-spread': string;
  'shadow-offset-x': string;
  'shadow-offset-y': string;
}

export interface ThemeStyles {
  light: ThemeStyleProps;
  dark: ThemeStyleProps;
}

// ============================================================================
// Constants
// ============================================================================

/** Map from CSS variable names to ThemeStyleProps keys */
const CSS_VAR_TO_THEME: Record<string, keyof ThemeStyleProps> = {
  '--background': 'background',
  '--foreground': 'foreground',
  '--card': 'card',
  '--card-foreground': 'card-foreground',
  '--popover': 'popover',
  '--popover-foreground': 'popover-foreground',
  '--primary': 'primary',
  '--primary-foreground': 'primary-foreground',
  '--secondary': 'secondary',
  '--secondary-foreground': 'secondary-foreground',
  '--muted': 'muted',
  '--muted-foreground': 'muted-foreground',
  '--accent': 'accent',
  '--accent-foreground': 'accent-foreground',
  '--destructive': 'destructive',
  '--destructive-foreground': 'destructive-foreground',
  '--border': 'border',
  '--input': 'input',
  '--ring': 'ring',
  '--chart-1': 'chart-1',
  '--chart-2': 'chart-2',
  '--chart-3': 'chart-3',
  '--chart-4': 'chart-4',
  '--chart-5': 'chart-5',
  '--sidebar': 'sidebar',
  '--sidebar-foreground': 'sidebar-foreground',
  '--sidebar-primary': 'sidebar-primary',
  '--sidebar-primary-foreground': 'sidebar-primary-foreground',
  '--sidebar-accent': 'sidebar-accent',
  '--sidebar-accent-foreground': 'sidebar-accent-foreground',
  '--sidebar-border': 'sidebar-border',
  '--sidebar-ring': 'sidebar-ring',
  '--font-sans': 'font-sans',
  '--font-serif': 'font-serif',
  '--font-mono': 'font-mono',
  '--radius': 'radius',
  '--spacing': 'spacing',
  '--letter-spacing': 'letter-spacing',
  '--shadow-color': 'shadow-color',
  '--shadow-opacity': 'shadow-opacity',
  '--shadow-blur': 'shadow-blur',
  '--shadow-spread': 'shadow-spread',
  '--shadow-offset-x': 'shadow-offset-x',
  '--shadow-offset-y': 'shadow-offset-y',
};

/** Reverse map from ThemeStyleProps keys to CSS variable names */
const THEME_TO_CSS_VAR: Record<keyof ThemeStyleProps, string> = Object.fromEntries(
  Object.entries(CSS_VAR_TO_THEME).map(([k, v]) => [v, k])
) as Record<keyof ThemeStyleProps, string>;

/** Default theme values (shadcn/ui defaults) */
const DEFAULT_THEME: ThemeStyleProps = {
  background: '0 0% 100%',
  foreground: '240 10% 3.9%',
  card: '0 0% 100%',
  'card-foreground': '240 10% 3.9%',
  popover: '0 0% 100%',
  'popover-foreground': '240 10% 3.9%',
  primary: '240 5.9% 10%',
  'primary-foreground': '0 0% 98%',
  secondary: '240 4.8% 95.9%',
  'secondary-foreground': '240 5.9% 10%',
  muted: '240 4.8% 95.9%',
  'muted-foreground': '240 3.8% 46.1%',
  accent: '240 4.8% 95.9%',
  'accent-foreground': '240 5.9% 10%',
  destructive: '0 84.2% 60.2%',
  'destructive-foreground': '0 0% 98%',
  border: '240 5.9% 90%',
  input: '240 5.9% 90%',
  ring: '240 5.9% 10%',
  'chart-1': '12 76% 61%',
  'chart-2': '173 58% 39%',
  'chart-3': '197 37% 24%',
  'chart-4': '43 74% 66%',
  'chart-5': '27 87% 67%',
  sidebar: '0 0% 98%',
  'sidebar-foreground': '240 5.3% 26.1%',
  'sidebar-primary': '240 5.9% 10%',
  'sidebar-primary-foreground': '0 0% 98%',
  'sidebar-accent': '240 4.8% 95.9%',
  'sidebar-accent-foreground': '240 5.9% 10%',
  'sidebar-border': '220 13% 91%',
  'sidebar-ring': '217.2 91.2% 59.8%',
  'font-sans': 'Inter, system-ui, sans-serif',
  'font-serif': 'Georgia, serif',
  'font-mono': 'JetBrains Mono, monospace',
  'letter-spacing': '0',
  radius: '0.5rem',
  'shadow-color': '0 0% 0%',
  'shadow-opacity': '0.1',
  'shadow-blur': '10px',
  'shadow-spread': '0px',
  'shadow-offset-x': '0px',
  'shadow-offset-y': '4px',
};

// ============================================================================
// Conversion Functions
// ============================================================================

/**
 * Extract design tokens from IPE context and convert to tweakcn format
 */
export function contextToThemeProps(
  context: IPEContext,
  mode: 'light' | 'dark' = 'dark'
): Partial<ThemeStyleProps> {
  const props: Partial<ThemeStyleProps> = {};

  for (const [cssVar, value] of Object.entries(context.cssVariables)) {
    const themeKey = CSS_VAR_TO_THEME[cssVar];
    if (themeKey) {
      props[themeKey] = value;
    }
  }

  return props;
}

/**
 * Convert tweakcn theme props back to CSS variables
 */
export function themePropsToCSS(
  props: Partial<ThemeStyleProps>
): Record<string, string> {
  const cssVars: Record<string, string> = {};

  for (const [key, value] of Object.entries(props)) {
    const cssVar = THEME_TO_CSS_VAR[key as keyof ThemeStyleProps];
    if (cssVar && value) {
      cssVars[cssVar] = value;
    }
  }

  return cssVars;
}

/**
 * Extract full theme styles from document root
 */
export function extractThemeFromRoot(mode: 'light' | 'dark' = 'dark'): ThemeStyleProps {
  const root = document.documentElement;
  const computed = getComputedStyle(root);
  const result = { ...DEFAULT_THEME };

  for (const [cssVar, themeKey] of Object.entries(CSS_VAR_TO_THEME)) {
    const value = computed.getPropertyValue(cssVar).trim();
    if (value) {
      result[themeKey] = value;
    }
  }

  return result;
}

/**
 * Apply theme props to document root
 */
export function applyThemeToRoot(props: Partial<ThemeStyleProps>): void {
  const root = document.documentElement;
  const cssVars = themePropsToCSS(props);

  for (const [cssVar, value] of Object.entries(cssVars)) {
    root.style.setProperty(cssVar, value);
  }
}

/**
 * Apply theme props to a specific element (instance scope)
 */
export function applyThemeToElement(
  element: Element,
  props: Partial<ThemeStyleProps>
): void {
  if (!(element instanceof HTMLElement)) return;

  const cssVars = themePropsToCSS(props);

  for (const [cssVar, value] of Object.entries(cssVars)) {
    element.style.setProperty(cssVar, value);
  }
}

/**
 * Generate CSS rule string for instance-scoped overrides
 */
export function generateInstanceCSS(
  instanceId: string,
  props: Partial<ThemeStyleProps>
): string {
  const cssVars = themePropsToCSS(props);
  const rules = Object.entries(cssVars)
    .map(([key, value]) => `  ${key}: ${value};`)
    .join('\n');

  return `[data-ipe-id="${instanceId}"] {\n${rules}\n}`;
}

/**
 * Parse HSL string to components
 */
export function parseHSL(hsl: string): { h: number; s: number; l: number } | null {
  // Format: "220 70% 50%" or "220deg 70% 50%"
  const match = hsl.match(/(\d+(?:\.\d+)?)\s*(?:deg)?\s+(\d+(?:\.\d+)?)%\s+(\d+(?:\.\d+)?)%/);
  if (!match) return null;

  return {
    h: parseFloat(match[1]),
    s: parseFloat(match[2]),
    l: parseFloat(match[3]),
  };
}

/**
 * Format HSL components to string
 */
export function formatHSL(h: number, s: number, l: number): string {
  return `${Math.round(h)} ${Math.round(s)}% ${Math.round(l)}%`;
}

/**
 * Convert HSL to RGB
 */
export function hslToRgb(h: number, s: number, l: number): { r: number; g: number; b: number } {
  s /= 100;
  l /= 100;

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;

  let r = 0, g = 0, b = 0;

  if (h < 60) { r = c; g = x; b = 0; }
  else if (h < 120) { r = x; g = c; b = 0; }
  else if (h < 180) { r = 0; g = c; b = x; }
  else if (h < 240) { r = 0; g = x; b = c; }
  else if (h < 300) { r = x; g = 0; b = c; }
  else { r = c; g = 0; b = x; }

  return {
    r: Math.round((r + m) * 255),
    g: Math.round((g + m) * 255),
    b: Math.round((b + m) * 255),
  };
}

/**
 * Calculate relative luminance for WCAG contrast
 */
export function relativeLuminance(r: number, g: number, b: number): number {
  const srgb = [r, g, b].map(c => {
    c /= 255;
    return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
  });
  return 0.2126 * srgb[0] + 0.7152 * srgb[1] + 0.0722 * srgb[2];
}

/**
 * Calculate contrast ratio between two colors
 */
export function contrastRatio(
  color1: { h: number; s: number; l: number },
  color2: { h: number; s: number; l: number }
): number {
  const rgb1 = hslToRgb(color1.h, color1.s, color1.l);
  const rgb2 = hslToRgb(color2.h, color2.s, color2.l);

  const lum1 = relativeLuminance(rgb1.r, rgb1.g, rgb1.b);
  const lum2 = relativeLuminance(rgb2.r, rgb2.g, rgb2.b);

  const lighter = Math.max(lum1, lum2);
  const darker = Math.min(lum1, lum2);

  return (lighter + 0.05) / (darker + 0.05);
}

/**
 * Check if contrast ratio meets WCAG AA standard
 */
export function meetsWCAGAA(
  foreground: string,
  background: string,
  largeText: boolean = false
): boolean {
  const fg = parseHSL(foreground);
  const bg = parseHSL(background);

  if (!fg || !bg) return true; // Assume passes if can't parse

  const ratio = contrastRatio(fg, bg);
  return largeText ? ratio >= 3 : ratio >= 4.5;
}

// ============================================================================
// Diff and Merge
// ============================================================================

/**
 * Compute diff between two theme props
 */
export function themePropsAdiff(
  original: Partial<ThemeStyleProps>,
  modified: Partial<ThemeStyleProps>
): Record<keyof ThemeStyleProps, { old: string; new: string }> {
  const diff: Record<string, { old: string; new: string }> = {};

  const allKeys = new Set([
    ...Object.keys(original),
    ...Object.keys(modified),
  ]) as Set<keyof ThemeStyleProps>;

  for (const key of allKeys) {
    const oldVal = original[key] ?? '';
    const newVal = modified[key] ?? '';
    if (oldVal !== newVal) {
      diff[key] = { old: oldVal, new: newVal };
    }
  }

  return diff as Record<keyof ThemeStyleProps, { old: string; new: string }>;
}

/**
 * Merge partial theme props with defaults
 */
export function mergeWithDefaults(partial: Partial<ThemeStyleProps>): ThemeStyleProps {
  return { ...DEFAULT_THEME, ...partial };
}

// ============================================================================
// Exports
// ============================================================================

export {
  CSS_VAR_TO_THEME,
  THEME_TO_CSS_VAR,
  DEFAULT_THEME,
};

export default {
  contextToThemeProps,
  themePropsToCSS,
  extractThemeFromRoot,
  applyThemeToRoot,
  applyThemeToElement,
  generateInstanceCSS,
  parseHSL,
  formatHSL,
  hslToRgb,
  contrastRatio,
  meetsWCAGAA,
  themePropsAdiff,
  mergeWithDefaults,
};
