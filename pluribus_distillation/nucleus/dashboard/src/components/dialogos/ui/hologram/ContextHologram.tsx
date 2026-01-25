/**
 * ContextHologram.tsx
 * Author: gemini_ui_1 (Ultrathink Mode)
 * Context: Phase 1 Superjump - The Holographic Layer
 * 
 * A full-screen SVG overlay that draws "Tethers" and "Spotlights" 
 * connecting the Dialogos Widget to relevant screen elements.
 */

import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

interface Tether {
  id: string;
  targetRect: DOMRect;
  color: string;
}

export const ContextHologram = component$(() => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const tethers = useSignal<Tether[]>([]);

  // Simulation: Listen for "Hover" events on SmartChips to draw tethers
  useVisibleTask$(({ cleanup }) => {
    const updateTethers = () => {
      // Find all elements marked with 'data-holon-target'
      const targets = document.querySelectorAll('[data-holon-target]');
      const newTethers: Tether[] = [];
      
      targets.forEach((el, i) => {
        // In a real implementation, we'd filter by 'active' state
        // For 'Superjump' demo, we tether to ALL marked targets
        const rect = el.getBoundingClientRect();
        newTethers.push({
          id: `tether-${i}`,
          targetRect: rect,
          color: 'rgba(0, 243, 255, 0.4)' // Cyan Neon
        });
      });
      tethers.value = newTethers;
    };

    // Poll for targets (Naïve implementation for Phase 1)
    const interval = setInterval(updateTethers, 500);
    cleanup(() => clearInterval(interval));
  });

  // Render Loop (60fps)
  useVisibleTask$(({ track }) => {
    track(() => tethers.value);
    const canvas = canvasRef.value;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Resize canvas
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const render = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Dialogos Input Position (Source)
      // Assuming fixed bottom center for now, ideally dynamic
      const sourceX = window.innerWidth / 2;
      const sourceY = window.innerHeight - 100; 

      tethers.value.forEach(t => {
        const targetX = t.targetRect.left + (t.targetRect.width / 2);
        const targetY = t.targetRect.top + (t.targetRect.height / 2);

        // Draw Tether (Curved Line)
        ctx.beginPath();
        ctx.moveTo(sourceX, sourceY);
        ctx.bezierCurveTo(
          sourceX, sourceY - 100, // Control Point 1
          targetX, targetY + 100, // Control Point 2
          targetX, targetY        // End
        );
        
        ctx.strokeStyle = t.color;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]); // Dashed "Data Flow" look
        ctx.stroke();

        // Draw "Spotlight" (Glow around target)
        ctx.beginPath();
        ctx.rect(t.targetRect.left - 5, t.targetRect.top - 5, t.targetRect.width + 10, t.targetRect.height + 10);
        ctx.strokeStyle = t.color;
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        ctx.shadowBlur = 15;
        ctx.shadowColor = 'cyan';
        ctx.stroke();
        ctx.shadowBlur = 0; // Reset
      });
      
      requestAnimationFrame(render);
    };

    const animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  });

  return (
    <canvas 
      ref={canvasRef}
      class="fixed inset-0 pointer-events-none z-[8000]"
    />
  );
});
