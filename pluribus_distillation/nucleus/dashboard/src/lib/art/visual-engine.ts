
import { type OKLCH, oklchToString, generatePalette, hexToOklch } from './color-utils';
import { getRandomChroma, type ChromaDefinition } from './chroma-registry';

/**
 * Visual Engine 2.0
 * The single source of truth for Pluribus aesthetics.
 * Replaces theme.ts and chroma-registry.ts
 */

export interface MaterialDefinition {
  id: string;
  name: string;
  category?: string;
  colors: {
    surface: string; // CSS value
    primary: string;
    secondary: string;
    bgStart: string;
    bgEnd: string;
    text: string;
    muted: string;
    border: string;
  };
  glass: {
    blur: number;
    roughness: number;
    refraction: number;
    gradientAngle: number; // New: Direction
    noiseIntensity: number; // New: Grain amount
  };
  shader: {
    noiseSpeed: number;
    noiseScale: number;
    bloomStrength: number;
  };
}

// Convert legacy ChromaDefinition to modern MaterialDefinition
export const chromaToMaterial = (chroma: ChromaDefinition): MaterialDefinition => {
  return {
    id: chroma.id,
    name: chroma.name,
    category: chroma.category,
    colors: {
      surface: chroma.colors.surface,
      primary: chroma.colors.primary,
      secondary: chroma.colors.secondary,
      bgStart: chroma.colors.bgGradientStart,
      bgEnd: chroma.colors.bgGradientEnd,
      text: 'oklch(0.95 0.01 240)', // Fallback text
      muted: 'oklch(0.60 0.01 240)',
      border: chroma.colors.primary.replace(')', ' / 0.2)'), // Add opacity if HSL
    },
    glass: {
      blur: chroma.glass.blur,
      roughness: 0.5,
      refraction: 1.0,
      gradientAngle: chroma.glass.gradientAngle,
      noiseIntensity: chroma.glass.noiseIntensity,
    },
    shader: {
      noiseSpeed: 0.2,
      noiseScale: 1.0,
      bloomStrength: 0.5,
    }
  };
};

export const generateMaterial = (seed?: number): MaterialDefinition => {
  // Try to use the chroma registry first for higher variety (500+ themes)
  try {
    const chroma = getRandomChroma();
    if (chroma) return chromaToMaterial(chroma);
  } catch (e) {
    console.warn('[VisualEngine] Failed to use chroma registry, falling back to procedural', e);
  }

  const seedVal = seed ?? Math.random();
  const hue = Math.floor(seedVal * 360);
  const palette = generatePalette(hue, 'dark'); // Forced Dark for now per plan

  return {
    id: `mat_${Date.now()}`,
    name: 'Procedural Smoked Glass',
    colors: {
      surface: oklchToString({ ...palette.surface, a: 0.6 }), // Transparent for glass
      primary: oklchToString(palette.primary),
      secondary: oklchToString(palette.secondary),
      bgStart: oklchToString(palette.bgStart),
      bgEnd: oklchToString(palette.bgEnd),
      text: oklchToString({ l: 0.95, c: 0.01, h: hue }), // White-ish
      muted: oklchToString({ l: 0.6, c: 0.01, h: hue }),
      border: oklchToString({ ...palette.primary, a: 0.2 }),
    },
    glass: {
      blur: 16,
      roughness: 0.4 + (Math.random() * 0.4),
      refraction: 1.0 + (Math.random() * 0.1),
      gradientAngle: 135 + Math.floor(Math.random() * 90), // Default top-left to bottom-right
      noiseIntensity: 0.4 + (Math.random() * 0.4),
    },
    shader: {
      noiseSpeed: 0.2 + (Math.random() * 0.5),
      noiseScale: 1.0 + (Math.random() * 2.0),
      bloomStrength: 0.5,
    }
  };
};

export const applyMaterialToRoot = (mat: MaterialDefinition) => {
  if (typeof document === 'undefined') return;
  
  const root = document.documentElement;
  
  // Set Colors
  root.style.setProperty('--mat-surface', mat.colors.surface);
  root.style.setProperty('--mat-primary', mat.colors.primary);
  root.style.setProperty('--mat-secondary', mat.colors.secondary);
  root.style.setProperty('--mat-bg-start', mat.colors.bgStart);
  root.style.setProperty('--mat-bg-end', mat.colors.bgEnd);
  root.style.setProperty('--mat-text', mat.colors.text);
  root.style.setProperty('--mat-muted', mat.colors.muted);
  root.style.setProperty('--mat-border', mat.colors.border);
  
  // Set Glass
  root.style.setProperty('--mat-blur', `${mat.glass.blur}px`);
  root.style.setProperty('--art-gradient-angle', `${mat.glass.gradientAngle}deg`);
  root.style.setProperty('--art-noise-intensity', String(mat.glass.noiseIntensity));
  
  // Backward Compatibility (Bridge to legacy theme.ts tokens)
  root.style.setProperty('--background', mat.colors.bgStart); 
  root.style.setProperty('--foreground', mat.colors.text);
  root.style.setProperty('--primary', mat.colors.primary);
};
