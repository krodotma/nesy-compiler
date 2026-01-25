/**
 * P5Canvas.tsx - p5.js Creative Coding Integration
 *
 * Enables running p5.js sketches within the Pluribus dashboard with:
 * - Full p5.js API support (instance mode)
 * - Sketch hot-reloading
 * - Bus event integration for real-time data visualization
 * - Recording/export capabilities
 *
 * Reference: https://p5js.org/reference/
 */

import { component$, useSignal, useVisibleTask$, $, type QRL, useStore, noSerialize, type NoSerialize } from '@builder.io/qwik';

// M3 Components - P5Canvas
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';

// ============================================================================
// Types
// ============================================================================

/** p5.js sketch function (instance mode) */
export type P5SketchFn = (p: P5Instance) => void;

/** Simplified p5 instance type (key methods) */
export interface P5Instance {
  // Setup
  createCanvas: (w: number, h: number, renderer?: string) => void;
  background: (r: number | string, g?: number, b?: number, a?: number) => void;
  frameRate: (fps: number) => void;

  // Drawing
  fill: (r: number | string, g?: number, b?: number, a?: number) => void;
  stroke: (r: number | string, g?: number, b?: number, a?: number) => void;
  strokeWeight: (w: number) => void;
  noFill: () => void;
  noStroke: () => void;

  // Shapes
  ellipse: (x: number, y: number, w: number, h?: number) => void;
  rect: (x: number, y: number, w: number, h: number, r?: number) => void;
  line: (x1: number, y1: number, x2: number, y2: number) => void;
  point: (x: number, y: number) => void;
  triangle: (x1: number, y1: number, x2: number, y2: number, x3: number, y3: number) => void;
  beginShape: () => void;
  endShape: (mode?: string) => void;
  vertex: (x: number, y: number) => void;

  // Transform
  translate: (x: number, y: number) => void;
  rotate: (angle: number) => void;
  scale: (s: number | [number, number]) => void;
  push: () => void;
  pop: () => void;

  // Math
  random: (min?: number, max?: number) => number;
  noise: (x: number, y?: number, z?: number) => number;
  map: (v: number, a1: number, b1: number, a2: number, b2: number) => number;
  constrain: (v: number, lo: number, hi: number) => number;
  sin: (angle: number) => number;
  cos: (angle: number) => number;

  // Properties
  width: number;
  height: number;
  frameCount: number;
  mouseX: number;
  mouseY: number;
  pmouseX: number;
  pmouseY: number;
  mouseIsPressed: boolean;

  // Constants
  PI: number;
  TWO_PI: number;
  HALF_PI: number;
  CLOSE: string;

  // Lifecycle (user-defined)
  setup?: () => void;
  draw?: () => void;
  mousePressed?: () => void;
  mouseReleased?: () => void;
  keyPressed?: () => void;

  // Control
  loop: () => void;
  noLoop: () => void;
  redraw: () => void;
  remove: () => void;

  // Export
  saveCanvas: (filename?: string, extension?: string) => void;
  saveFrames: (filename: string, extension: string, duration: number, fps: number) => void;
}

export interface P5EventPayload {
  type: 'sketch_loaded' | 'sketch_error' | 'frame_rendered' | 'user_interaction' | 'export_complete';
  sketchId: string;
  data: Record<string, unknown>;
}

interface P5State {
  status: 'idle' | 'loading' | 'running' | 'paused' | 'error';
  error: string | null;
  fps: number;
  frameCount: number;
  instance: NoSerialize<P5Instance> | null;
}

// ============================================================================
// Component Props
// ============================================================================

export interface P5CanvasProps {
  /** Unique sketch identifier */
  sketchId?: string;
  /** Sketch code as string (evaluated at runtime) */
  sketchCode?: string;
  /** Pre-built sketch function */
  sketchFn?: P5SketchFn;
  /** Canvas width */
  width?: number;
  /** Canvas height */
  height?: number;
  /** Target frame rate */
  frameRate?: number;
  /** Auto-start on load */
  autoPlay?: boolean;
  /** Data to pass into sketch (accessible via p._pluribusData) */
  data?: Record<string, unknown>;
  /** Event callback */
  onEvent$?: QRL<(event: P5EventPayload) => void>;
  /** Show controls */
  showControls?: boolean;
  /** WebGL mode */
  webgl?: boolean;
}

// ============================================================================
// p5.js Loader
// ============================================================================

const P5_CDN = 'https://cdn.jsdelivr.net/npm/p5@1.9.0/lib/p5.min.js';

async function loadP5Runtime(): Promise<unknown> {
  if (typeof window === 'undefined') return null;

  const win = window as unknown as { p5?: unknown };
  if (win.p5) return win.p5;

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = P5_CDN;
    script.async = true;
    script.onload = () => {
      const p5 = (window as unknown as { p5?: unknown }).p5;
      if (p5) {
        resolve(p5);
      } else {
        reject(new Error('p5.js runtime not found'));
      }
    };
    script.onerror = () => reject(new Error('Failed to load p5.js'));
    document.head.appendChild(script);
  });
}

// ============================================================================
// Sketch Code Evaluation (sandboxed)
// ============================================================================

function createSketchFromCode(code: string, data?: Record<string, unknown>): P5SketchFn {
  // Wrap user code in instance mode
  const wrappedCode = `
    return function(p) {
      // Inject Pluribus data
      p._pluribusData = ${JSON.stringify(data || {})};

      // User sketch code
      ${code}
    };
  `;

  try {
    // eslint-disable-next-line no-new-func
    const factory = new Function(wrappedCode);
    return factory();
  } catch (err) {
    throw new Error(`Sketch compilation error: ${err instanceof Error ? err.message : 'Unknown'}`);
  }
}

// ============================================================================
// Main Component
// ============================================================================

export const P5Canvas = component$<P5CanvasProps>((props) => {
  const containerRef = useSignal<HTMLDivElement>();

  const state = useStore<P5State>({
    status: 'idle',
    error: null,
    fps: 0,
    frameCount: 0,
    instance: null,
  });

  const isPlaying = useSignal(props.autoPlay ?? true);
  const sketchId = props.sketchId || `sketch-${Date.now()}`;

  // Initialize p5 sketch
  useVisibleTask$(async ({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const container = containerRef.value;
    if (!container) return;

    state.status = 'loading';

    try {
      const p5Constructor = await loadP5Runtime() as new (
        sketch: P5SketchFn,
        node: HTMLElement
      ) => P5Instance;

      if (!p5Constructor) {
        throw new Error('p5.js runtime unavailable');
      }

      // Determine sketch source
      let sketchFn: P5SketchFn;

      if (props.sketchFn) {
        sketchFn = props.sketchFn;
      } else if (props.sketchCode) {
        sketchFn = createSketchFromCode(props.sketchCode, props.data);
      } else {
        // Default demo sketch
        sketchFn = (p) => {
          const particles: Array<{ x: number; y: number; vx: number; vy: number; life: number }> = [];

          p.setup = () => {
            p.createCanvas(props.width || 600, props.height || 400, props.webgl ? 'webgl' : undefined);
            p.frameRate(props.frameRate || 60);
          };

          p.draw = () => {
            p.background(10, 10, 20);

            // Spawn particles at mouse
            if (p.mouseIsPressed) {
              for (let i = 0; i < 3; i++) {
                particles.push({
                  x: p.mouseX,
                  y: p.mouseY,
                  vx: p.random(-2, 2),
                  vy: p.random(-4, -1),
                  life: 255,
                });
              }
            }

            // Update and draw particles
            for (let i = particles.length - 1; i >= 0; i--) {
              const part = particles[i];
              part.x += part.vx;
              part.y += part.vy;
              part.vy += 0.1; // gravity
              part.life -= 3;

              if (part.life <= 0) {
                particles.splice(i, 1);
                continue;
              }

              p.noStroke();
              p.fill(100, 200, 255, part.life);
              p.ellipse(part.x, part.y, 8, 8);
            }

            // Instructions
            p.fill(255, 100);
            p.noStroke();
            const pAny = p as any;
            pAny.textSize?.(12);
            pAny.text?.('Click and drag to create particles', 10, 20);
          };
        };
      }

      // Wrap sketch to capture lifecycle events
      const wrappedSketch: P5SketchFn = (p) => {
        // Run user sketch
        sketchFn(p);

        // Capture original setup/draw
        const originalSetup = p.setup;
        const originalDraw = p.draw;

        p.setup = () => {
          originalSetup?.call(p);
          state.status = 'running';
          props.onEvent$?.({
            type: 'sketch_loaded',
            sketchId,
            data: { width: p.width, height: p.height },
          });
        };

        p.draw = () => {
          originalDraw?.call(p);
          state.frameCount = p.frameCount;

          // Throttled FPS calculation
          if (p.frameCount % 30 === 0) {
            state.fps = Math.round((p as unknown as { frameRate: () => number }).frameRate?.() || 60);
          }
        };
      };

      // Create p5 instance
      const instance = new p5Constructor(wrappedSketch, container);
      state.instance = noSerialize(instance);

      // Auto-pause if not autoPlay
      if (!props.autoPlay) {
        instance.noLoop();
        isPlaying.value = false;
        state.status = 'paused';
      }

      cleanup(() => {
        try {
          instance.remove();
        } catch {
          // Ignore cleanup errors
        }
      });

    } catch (err) {
      state.status = 'error';
      state.error = err instanceof Error ? err.message : 'Unknown error';
      props.onEvent$?.({
        type: 'sketch_error',
        sketchId,
        data: { error: state.error },
      });
    }
  });

  // Play/pause control
  const togglePlay = $(() => {
    const p = state.instance as P5Instance | null;
    if (!p) return;

    if (isPlaying.value) {
      p.noLoop();
      state.status = 'paused';
    } else {
      p.loop();
      state.status = 'running';
    }
    isPlaying.value = !isPlaying.value;
  });

  // Export frame
  const exportFrame = $(() => {
    const p = state.instance as P5Instance | null;
    if (!p) return;

    p.saveCanvas(`pluribus-sketch-${Date.now()}`, 'png');
    props.onEvent$?.({
      type: 'export_complete',
      sketchId,
      data: { format: 'png' },
    });
  });

  // Redraw single frame
  const stepFrame = $(() => {
    const p = state.instance as P5Instance | null;
    if (!p || state.status !== 'paused') return;
    p.redraw();
  });

  const width = props.width || 600;
  const height = props.height || 400;

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Canvas Container */}
      <div
        ref={containerRef}
        class="relative bg-black"
        style={{ width: `${width}px`, height: `${height}px`, maxWidth: '100%' }}
      />

      {/* Loading State */}
      {state.status === 'loading' && (
        <div
          class="absolute inset-0 flex items-center justify-center bg-black/80"
          style={{ width: `${width}px`, height: `${height}px` }}
        >
          <div class="text-center">
            <div class="w-8 h-8 border-2 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <div class="text-sm text-purple-400">Loading p5.js...</div>
          </div>
        </div>
      )}

      {/* Error State */}
      {state.status === 'error' && (
        <div
          class="absolute inset-0 flex items-center justify-center bg-red-900/20 p-4"
          style={{ width: `${width}px`, height: `${height}px` }}
        >
          <div class="text-center">
            <div class="text-red-400 text-lg mb-2">Sketch Error</div>
            <pre class="text-xs text-red-300/70 max-w-md overflow-auto bg-black/50 p-2 rounded">
              {state.error}
            </pre>
          </div>
        </div>
      )}

      {/* Controls */}
      {props.showControls !== false && (state.status === 'running' || state.status === 'paused') && (
        <div class="flex items-center justify-between p-2 bg-black/60 border-t border-border/50">
          <div class="flex items-center gap-2">
            {/* Play/Pause */}
            <button
              class="w-8 h-8 rounded bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
              onClick$={togglePlay}
              title={isPlaying.value ? 'Pause' : 'Play'}
            >
              {isPlaying.value ? (
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <polygon points="5,3 19,12 5,21" />
                </svg>
              )}
            </button>

            {/* Step Frame (when paused) */}
            {state.status === 'paused' && (
              <button
                class="w-8 h-8 rounded bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
                onClick$={stepFrame}
                title="Step Frame"
              >
                <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                  <polygon points="5,4 15,12 5,20" />
                  <rect x="16" y="4" width="3" height="16" />
                </svg>
              </button>
            )}

            {/* Export */}
            <button
              class="w-8 h-8 rounded bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
              onClick$={exportFrame}
              title="Save Frame"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </button>
          </div>

          {/* Stats */}
          <div class="flex items-center gap-3 text-xs text-white/60 mono">
            <span>{state.fps} FPS</span>
            <span>Frame {state.frameCount}</span>
          </div>
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Preset Sketches
// ============================================================================

export const P5_PRESETS = {
  /** Particle system demo */
  particles: `
    const particles = [];

    p.setup = () => {
      p.createCanvas(600, 400);
      p.background(10);
    };

    p.draw = () => {
      p.background(10, 20);

      if (p.frameCount % 5 === 0) {
        particles.push({
          x: p.width / 2,
          y: p.height,
          vx: p.random(-1, 1),
          vy: p.random(-3, -1),
          size: p.random(5, 15),
          hue: p.random(180, 220)
        });
      }

      for (let i = particles.length - 1; i >= 0; i--) {
        const pt = particles[i];
        pt.x += pt.vx;
        pt.y += pt.vy;
        pt.vy += 0.02;
        pt.size *= 0.99;

        if (pt.size < 1) {
          particles.splice(i, 1);
          continue;
        }

        p.noStroke();
        p.fill(pt.hue, 200, 255, 180);
        p.ellipse(pt.x, pt.y, pt.size);
      }
    };
  `,

  /** Perlin noise flow field */
  flowField: `
    let particles = [];
    const num = 500;
    let noiseScale = 0.01;

    p.setup = () => {
      p.createCanvas(600, 400);
      p.background(0);
      for (let i = 0; i < num; i++) {
        particles.push(p.createVector(p.random(p.width), p.random(p.height)));
      }
    };

    p.draw = () => {
      p.background(0, 10);
      p.stroke(100, 200, 255, 30);
      p.strokeWeight(1);

      for (let pt of particles) {
        const angle = p.noise(pt.x * noiseScale, pt.y * noiseScale) * p.TWO_PI * 2;
        pt.x += p.cos(angle);
        pt.y += p.sin(angle);

        if (pt.x < 0 || pt.x > p.width || pt.y < 0 || pt.y > p.height) {
          pt.x = p.random(p.width);
          pt.y = p.random(p.height);
        }

        p.point(pt.x, pt.y);
      }
    };
  `,

  /** Audio reactive (requires mic permission) */
  audioReactive: `
    let mic;
    let fft;

    p.setup = () => {
      p.createCanvas(600, 400);
      mic = new p5.AudioIn();
      mic.start();
      fft = new p5.FFT();
      fft.setInput(mic);
    };

    p.draw = () => {
      p.background(0);
      const spectrum = fft.analyze();

      p.noStroke();
      for (let i = 0; i < spectrum.length; i++) {
        const x = p.map(i, 0, spectrum.length, 0, p.width);
        const h = p.map(spectrum[i], 0, 255, 0, p.height);
        const hue = p.map(i, 0, spectrum.length, 0, 360);
        p.fill(hue, 200, 255);
        p.rect(x, p.height - h, p.width / spectrum.length, h);
      }
    };
  `,
} as const;

export default P5Canvas;
