/**
 * LatticeBackground.tsx
 * [Ultrathink Agent 2: Artist]
 * 
 * A subtle, architectural background pattern for the CodeWarrior interface.
 * Represents the "Grid" of the simulation.
 */

import { component$ } from '@builder.io/qwik';

export const LatticeBackground = component$(() => {
  return (
    <div class="absolute inset-0 pointer-events-none z-0 overflow-hidden rounded-xl">
      <svg width="100%" height="100%" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern id="lattice-grid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255, 255, 255, 0.03)" stroke-width="1" />
          </pattern>
          <radialGradient id="vignette" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stop-color="rgba(0,0,0,0)" />
            <stop offset="100%" stop-color="rgba(0,0,0,0.6)" />
          </radialGradient>
        </defs>
        <rect width="100%" height="100%" fill="url(#lattice-grid)" />
        <rect width="100%" height="100%" fill="url(#vignette)" />
      </svg>
      
      {/* "Prism" Border Highlight (Top Shine) */}
      <div class="absolute top-0 left-0 right-0 h-[1px] bg-gradient-to-r from-transparent via-blue-500/50 to-transparent opacity-50" />
    </div>
  );
});
