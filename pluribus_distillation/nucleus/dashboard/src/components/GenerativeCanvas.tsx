/**
 * GenerativeCanvas.tsx - Minimal generative surface (MVP)
 *
 * This is intentionally lightweight: it provides a stable UI surface for agents
 * to target with "ui.render" events, without requiring a full WebGL stack yet.
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';

// M3 Components - GenerativeCanvas
import '@material/web/elevation/elevation.js';

export const GenerativeCanvas = component$(() => {
  const canvasRef = useSignal<HTMLCanvasElement>();

  useVisibleTask$(({ cleanup }) => {
    const canvas = canvasRef.value;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    let t = 0;

    const draw = () => {
      t += 0.01;
      const w = canvas.width;
      const h = canvas.height;

      const grad = ctx.createLinearGradient(0, 0, w, h);
      grad.addColorStop(0, `rgba(34,211,238,${0.10 + 0.05 * Math.sin(t)})`);
      grad.addColorStop(0.5, `rgba(168,85,247,${0.08 + 0.04 * Math.cos(t * 1.3)})`);
      grad.addColorStop(1, `rgba(16,185,129,${0.10 + 0.05 * Math.sin(t * 0.7)})`);

      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = 'rgba(0,0,0,0.5)';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);

      // Minimal "signal trace" line
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x < w; x += 8) {
        const y = h / 2 + Math.sin(t * 2 + x * 0.01) * (h * 0.18);
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      raf = requestAnimationFrame(draw);
    };

    raf = requestAnimationFrame(draw);
    cleanup(() => cancelAnimationFrame(raf));
  });

  return (
    <div class="rounded-lg border border-border bg-card p-4 space-y-3">
      <div class="flex items-center justify-between">
        <div class="font-semibold">🎨 Generative Canvas</div>
        <div class="text-xs text-muted-foreground mono">MVP surface</div>
      </div>
      <canvas
        ref={canvasRef}
        width={900}
        height={360}
        class="w-full h-[320px] rounded border border-border bg-black/40"
      />
      <div class="text-xs text-muted-foreground">
        Planned: accept `ui.render` bus events (Lit/Three.js payloads), replayable via Rhizome artifacts.
      </div>
    </div>
  );
});

