
/**
 * Color Utilities for Visual Style Model 2.0
 * 
 * Implements OKLCH color space for perceptually uniform gradients.
 * OKLCH separates Lightness (L), Chroma (C), and Hue (H) much better than HSL.
 * 
 * L: 0.0 - 1.0 (Black to White)
 * C: 0.0 - 0.4 (Grey to Super-Saturated)
 * H: 0.0 - 360 (Hue angle)
 */

export type OKLCH = { l: number; c: number; h: number; a?: number };

export const oklchToString = (color: OKLCH): string => {
  const alpha = color.a !== undefined ? ` / ${color.a}` : '';
  return `oklch(${color.l.toFixed(3)} ${color.c.toFixed(3)} ${color.h.toFixed(1)}${alpha})`;
};

export const hexToOklch = (hex: string): OKLCH => {
  // Placeholder: In a real implementation, we'd need a math library.
  // For now, we rely on browser support for oklch() or simple HSL fallback if needed.
  // But modern browsers support oklch().
  // We will primarily generate in OKLCH directly.
  return { l: 0.5, c: 0.1, h: 0 }; 
};

// Generate a coherent palette from a seed hue
export const generatePalette = (seedHue: number, mode: 'dark' | 'light' = 'dark') => {
  const isDark = mode === 'dark';
  
  // Base Surface (Background)
  // Dark: Low Lightness, Low Chroma (Neutral)
  // Light: High Lightness, Low Chroma
  const surface: OKLCH = {
    l: isDark ? 0.12 : 0.98,
    c: 0.01, // Very slight tint
    h: seedHue
  };

  // Primary Accent
  // High chroma, medium lightness
  const primary: OKLCH = {
    l: isDark ? 0.65 : 0.55,
    c: 0.15, // Vibrant
    h: seedHue
  };

  // Secondary (Complimentary or Analogous)
  const secondary: OKLCH = {
    l: isDark ? 0.70 : 0.60,
    c: 0.12,
    h: (seedHue + 120) % 360
  };

  // The "Smoked Glass" Gradient Stops
  // Start: Deep, almost black
  const bgStart: OKLCH = {
    l: isDark ? 0.08 : 0.95,
    c: 0.02,
    h: seedHue
  };

  // End: Slightly lighter, shifted hue for depth
  const bgEnd: OKLCH = {
    l: isDark ? 0.05 : 0.90,
    c: 0.03,
    h: (seedHue + 40) % 360
  };

  return { surface, primary, secondary, bgStart, bgEnd };
};
