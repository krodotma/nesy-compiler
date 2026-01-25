import { component$, useVisibleTask$, useSignal, useStylesScoped$ } from '@builder.io/qwik';
import { GenerativeBackground } from '../art/GenerativeBackground';

/**
 * VisualSynthesis: The "Hakim.se" Integration Layer.
 * 
 * Orchestrates:
 * 1. Generative Background (Shader/Canvas)
 * 2. 3D Transitions (CSS Perspective)
 * 3. Particle Systems (Three.js/P5 placeholder)
 */
export const VisualSynthesis = component$(() => {
  const containerRef = useSignal<HTMLDivElement>();

  useStylesScoped$(`
    .visual-synthesis-layer {
      position: fixed;
      inset: 0;
      z-index: 0; /* Behind content but in front of body bg */
      pointer-events: none;
      perspective: 1000px;
      overflow: hidden;
    }
    .particle-field {
      position: absolute;
      inset: 0;
      opacity: 0.3;
      mix-blend-mode: screen;
    }
  `);

  useVisibleTask$(({ track }) => {
    // Placeholder for P5/Three.js instance lifecycle
    // Real implementation would mount a sketch here if 'art.scene.change' requested 'particles'
  });

  return (
    <div ref={containerRef} class="visual-synthesis-layer">
      <GenerativeBackground enabled={true} />
      <div class="particle-field" id="particle-root" />
    </div>
  );
});
