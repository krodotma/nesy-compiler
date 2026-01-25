/**
 * EntropyCanvas.tsx
 *
 * Qwik component that renders the EntropyShader.
 * Manages the WebGL context and uniform binding.
 *
 * [Ultrathink Agent 1: Architect]
 * "Optimized for low overhead. Uses requestAnimationFrame and resizes intelligently."
 */

import {
  component$,
  useSignal,
  useVisibleTask$,
  $,
} from '@builder.io/qwik';
import { ENTROPY_VERTEX_SHADER, ENTROPY_FRAGMENT_SHADER } from './EntropyShader';

interface EntropyCanvasProps {
  entropy: number;    // 0-100
  negentropy: number; // 0-100
  height?: number;
}

export const EntropyCanvas = component$<EntropyCanvasProps>(({ entropy, negentropy, height = 60 }) => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const containerRef = useSignal<HTMLDivElement>();

  useVisibleTask$(({ cleanup }) => {
    if (!canvasRef.value || !containerRef.value) return;

    const canvas = canvasRef.value;
    const gl = canvas.getContext('webgl');
    if (!gl) return;

    // Shader compilation helper
    const createShader = (gl: WebGLRenderingContext, type: number, source: string) => {
      const shader = gl.createShader(type);
      if (!shader) return null;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(shader));
        gl.deleteShader(shader);
        return null;
      }
      return shader;
    };

    const vert = createShader(gl, gl.VERTEX_SHADER, ENTROPY_VERTEX_SHADER);
    const frag = createShader(gl, gl.FRAGMENT_SHADER, ENTROPY_FRAGMENT_SHADER);
    if (!vert || !frag) return;

    const program = gl.createProgram();
    if (!program) return;
    gl.attachShader(program, vert);
    gl.attachShader(program, frag);
    gl.linkProgram(program);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      console.error(gl.getProgramInfoLog(program));
      return;
    }

    gl.useProgram(program);

    // Buffers
    const positionBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]),
      gl.STATIC_DRAW
    );

    const positionLocation = gl.getAttribLocation(program, 'position');
    gl.enableVertexAttribArray(positionLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

    // Uniforms
    const uResolution = gl.getUniformLocation(program, 'u_resolution');
    const uTime = gl.getUniformLocation(program, 'u_time');
    const uEntropy = gl.getUniformLocation(program, 'u_entropy');
    const uNegentropy = gl.getUniformLocation(program, 'u_negentropy');
    const uMouse = gl.getUniformLocation(program, 'u_mouse');

    // Resize handler
    const resize = () => {
      if (!canvas || !containerRef.value) return;
      const rect = containerRef.value.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.uniform2f(uResolution, canvas.width, canvas.height);
    };
    window.addEventListener('resize', resize);
    resize();

    // Mouse tracking
    let mx = 0.5;
    let my = 0.5;
    const updateMouse = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      mx = (e.clientX - rect.left) / rect.width;
      my = 1.0 - (e.clientY - rect.top) / rect.height;
    };
    canvas.addEventListener('mousemove', updateMouse);

    // Animation Loop
    let running = true;
    const startTime = performance.now();

    const frame = (now: number) => {
      if (!running) return;
      const t = (now - startTime) / 1000;

      gl.uniform1f(uTime, t);
      gl.uniform1f(uEntropy, entropy / 100);
      gl.uniform1f(uNegentropy, negentropy / 100);
      gl.uniform2f(uMouse, mx, my);

      gl.drawArrays(gl.TRIANGLES, 0, 6);
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);

    cleanup(() => {
      running = false;
      window.removeEventListener('resize', resize);
      canvas.removeEventListener('mousemove', updateMouse);
      gl.deleteProgram(program);
    });
  });

  return (
    <div 
      ref={containerRef} 
      class="relative w-full rounded-md overflow-hidden border border-[var(--glass-border)] shadow-inner"
      style={{ height: `${height}px` }}
    >
      <canvas 
        ref={canvasRef} 
        class="w-full h-full block"
      />
      
      {/* Overlay Labels */}
      <div class="absolute inset-0 pointer-events-none flex justify-between items-end p-2 text-[9px] font-mono font-bold tracking-widest mix-blend-overlay">
        <span class="text-red-300 drop-shadow-md">ENTROPY</span>
        <span class="text-cyan-300 drop-shadow-md">NEGENTROPY</span>
      </div>
    </div>
  );
});
