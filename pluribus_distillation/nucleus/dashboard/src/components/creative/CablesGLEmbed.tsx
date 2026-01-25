/**
 * CablesGLEmbed.tsx - Cables.gl WebGL Visual Editor Integration
 *
 * Provides embedding and event bridging for Cables.gl patches within the
 * Pluribus dashboard. Supports loading patches by URL or inline JSON,
 * bidirectional event communication with the Pluribus bus, and variable
 * binding for real-time parameter control.
 *
 * Reference: https://cables.gl/api/
 */

import { component$, useSignal, useVisibleTask$, $, type QRL, useStore, noSerialize, type NoSerialize } from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface CablesPatch {
  /** URL to a cables.gl patch export (JSON or direct link) */
  url?: string;
  /** Inline patch definition (cables export JSON) */
  inline?: Record<string, unknown>;
  /** Patch ID from cables.gl (e.g., "abcd1234") */
  patchId?: string;
}

export interface CablesVariable {
  name: string;
  type: 'number' | 'string' | 'boolean' | 'trigger' | 'array';
  value: unknown;
}

export interface CablesEventPayload {
  type: 'variable_change' | 'render_frame' | 'patch_loaded' | 'patch_error' | 'op_trigger';
  patchId: string;
  data: Record<string, unknown>;
}

interface CablesState {
  status: 'idle' | 'loading' | 'ready' | 'error';
  error: string | null;
  fps: number;
  frameCount: number;
  variables: CablesVariable[];
  patch: NoSerialize<unknown> | null;
}

// ============================================================================
// Component Props
// ============================================================================

export interface CablesGLEmbedProps {
  /** Patch source configuration */
  patch: CablesPatch;
  /** Canvas width (px or %) */
  width?: string;
  /** Canvas height (px or %) */
  height?: string;
  /** Whether to auto-play on load */
  autoPlay?: boolean;
  /** Variables to inject into the patch */
  variables?: CablesVariable[];
  /** Callback when patch events occur (bridged to bus) */
  onEvent$?: QRL<(event: CablesEventPayload) => void>;
  /** Whether to show controls overlay */
  showControls?: boolean;
}

// ============================================================================
// Cables.gl Loader (runtime injection)
// ============================================================================

const CABLES_CDN = 'https://cables.gl/api/ops/latest/cables.min.js';

async function loadCablesRuntime(): Promise<unknown> {
  if (typeof window === 'undefined') return null;

  // Check if already loaded
  const win = window as unknown as { CABLES?: unknown };
  if (win.CABLES) return win.CABLES;

  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = CABLES_CDN;
    script.async = true;
    script.onload = () => {
      const cables = (window as unknown as { CABLES?: unknown }).CABLES;
      if (cables) {
        resolve(cables);
      } else {
        reject(new Error('Cables.gl runtime not found after load'));
      }
    };
    script.onerror = () => reject(new Error('Failed to load Cables.gl runtime'));
    document.head.appendChild(script);
  });
}

// ============================================================================
// Main Component
// ============================================================================

export const CablesGLEmbed = component$<CablesGLEmbedProps>((props) => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const containerRef = useSignal<HTMLDivElement>();

  const state = useStore<CablesState>({
    status: 'idle',
    error: null,
    fps: 0,
    frameCount: 0,
    variables: [],
    patch: null,
  });

  const isPlaying = useSignal(props.autoPlay ?? true);
  const showVars = useSignal(false);

  // Initialize Cables.gl patch
  useVisibleTask$(async ({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const canvas = canvasRef.value;
    if (!canvas) return;

    state.status = 'loading';

    try {
      const CABLES = await loadCablesRuntime() as {
        Patch: new (config: {
          canvas: HTMLCanvasElement;
          glCanvasResizeToWindow?: boolean;
          onError?: (err: Error) => void;
          onFinishedLoading?: () => void;
          onFirstFrameRendered?: () => void;
        }) => {
          config: { patchFile?: Record<string, unknown> };
          setVariable: (name: string, value: unknown) => void;
          getVariable: (name: string) => unknown;
          pause: () => void;
          resume: () => void;
          dispose: () => void;
          cgl: { fps: number };
        };
      };

      if (!CABLES) {
        throw new Error('Cables.gl runtime unavailable');
      }

      // Determine patch source
      let patchConfig: Record<string, unknown> | null = null;

      if (props.patch.inline) {
        patchConfig = props.patch.inline;
      } else if (props.patch.url) {
        const resp = await fetch(props.patch.url);
        if (!resp.ok) throw new Error(`Failed to fetch patch: ${resp.status}`);
        patchConfig = await resp.json();
      } else if (props.patch.patchId) {
        // Fetch from cables.gl API
        const apiUrl = `https://cables.gl/api/p/${props.patch.patchId}/json`;
        const resp = await fetch(apiUrl);
        if (!resp.ok) throw new Error(`Failed to fetch patch ${props.patch.patchId}`);
        patchConfig = await resp.json();
      }

      if (!patchConfig) {
        throw new Error('No patch source provided');
      }

      // Create patch instance
      const patch = new CABLES.Patch({
        canvas,
        glCanvasResizeToWindow: false,
        onError: (err: Error) => {
          state.error = err.message;
          state.status = 'error';
          props.onEvent$?.({
            type: 'patch_error',
            patchId: props.patch.patchId || 'inline',
            data: { error: err.message },
          });
        },
        onFinishedLoading: () => {
          state.status = 'ready';
          props.onEvent$?.({
            type: 'patch_loaded',
            patchId: props.patch.patchId || 'inline',
            data: { status: 'ready' },
          });
        },
        onFirstFrameRendered: () => {
          state.frameCount++;
        },
      });

      // Load patch configuration
      patch.config.patchFile = patchConfig;

      // Inject initial variables
      if (props.variables) {
        for (const v of props.variables) {
          patch.setVariable(v.name, v.value);
          state.variables.push({ ...v });
        }
      }

      state.patch = noSerialize(patch);

      // FPS tracking
      const fpsInterval = setInterval(() => {
        if (patch.cgl) {
          state.fps = Math.round(patch.cgl.fps);
        }
      }, 1000);

      cleanup(() => {
        clearInterval(fpsInterval);
        try {
          patch.dispose();
        } catch {
          // Ignore disposal errors
        }
      });

    } catch (err) {
      state.status = 'error';
      state.error = err instanceof Error ? err.message : 'Unknown error loading patch';
    }
  });

  // Play/pause control
  const togglePlay = $(() => {
    const patch = state.patch as unknown as { pause: () => void; resume: () => void } | null;
    if (!patch) return;

    if (isPlaying.value) {
      patch.pause();
    } else {
      patch.resume();
    }
    isPlaying.value = !isPlaying.value;
  });

  // Set variable
  const setVariable = $((name: string, value: unknown) => {
    const patch = state.patch as unknown as { setVariable: (n: string, v: unknown) => void } | null;
    if (!patch) return;

    patch.setVariable(name, value);

    // Update local state
    const idx = state.variables.findIndex(v => v.name === name);
    if (idx >= 0) {
      state.variables[idx].value = value;
    }

    // Emit bus event
    props.onEvent$?.({
      type: 'variable_change',
      patchId: props.patch.patchId || 'inline',
      data: { name, value },
    });
  });

  const width = props.width || '100%';
  const height = props.height || '400px';

  return (
    <div
      ref={containerRef}
      class="relative rounded-lg border border-border bg-black overflow-hidden"
      style={{ width, height }}
    >
      {/* WebGL Canvas */}
      <canvas
        ref={canvasRef}
        class="w-full h-full"
        style={{ display: state.status === 'error' ? 'none' : 'block' }}
      />

      {/* Loading Overlay */}
      {state.status === 'loading' && (
        <div class="absolute inset-0 flex items-center justify-center bg-black/80">
          <div class="text-center">
            <div class="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto mb-2" />
            <div class="text-sm text-cyan-400">Loading Cables.gl patch...</div>
          </div>
        </div>
      )}

      {/* Error Overlay */}
      {state.status === 'error' && (
        <div class="absolute inset-0 flex items-center justify-center bg-red-900/20 p-4">
          <div class="text-center">
            <div class="text-red-400 text-lg mb-2">Patch Error</div>
            <div class="text-sm text-red-300/70 max-w-md">{state.error}</div>
          </div>
        </div>
      )}

      {/* Controls Overlay */}
      {props.showControls !== false && state.status === 'ready' && (
        <div class="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-3">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-3">
              {/* Play/Pause */}
              <button
                class="w-8 h-8 rounded bg-white/10 hover:bg-white/20 flex items-center justify-center text-white transition-colors"
                onClick$={togglePlay}
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

              {/* FPS Display */}
              <div class="text-xs text-white/60 mono">
                {state.fps} FPS
              </div>
            </div>

            {/* Variables Toggle */}
            {state.variables.length > 0 && (
              <button
                class={`px-2 py-1 rounded text-xs transition-colors ${
                  showVars.value
                    ? 'bg-cyan-500/30 text-cyan-300'
                    : 'bg-white/10 text-white/60 hover:bg-white/20'
                }`}
                onClick$={() => { showVars.value = !showVars.value; }}
              >
                Variables ({state.variables.length})
              </button>
            )}
          </div>

          {/* Variables Panel */}
          {showVars.value && state.variables.length > 0 && (
            <div class="mt-2 p-2 bg-black/60 rounded space-y-2 max-h-32 overflow-y-auto">
              {state.variables.map((v) => (
                <div key={v.name} class="flex items-center gap-2 text-xs">
                  <span class="text-white/60 w-24 truncate">{v.name}</span>
                  {v.type === 'number' && (
                    <input
                      type="range"
                      class="flex-1 h-1 bg-white/20 rounded appearance-none cursor-pointer"
                      min="0"
                      max="100"
                      value={Number(v.value) || 0}
                      onInput$={(e) => {
                        const val = parseFloat((e.target as HTMLInputElement).value);
                        setVariable(v.name, val);
                      }}
                    />
                  )}
                  {v.type === 'boolean' && (
                    <input
                      type="checkbox"
                      checked={Boolean(v.value)}
                      onChange$={(e) => {
                        setVariable(v.name, (e.target as HTMLInputElement).checked);
                      }}
                    />
                  )}
                  {v.type === 'trigger' && (
                    <button
                      class="px-2 py-0.5 bg-cyan-600 hover:bg-cyan-500 rounded text-white"
                      onClick$={() => setVariable(v.name, true)}
                    >
                      Trigger
                    </button>
                  )}
                  <span class="text-cyan-400 mono w-12 text-right truncate">
                    {String(v.value).slice(0, 8)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Cables.gl Attribution */}
      <div class="absolute top-2 right-2 text-[10px] text-white/30 mono">
        cables.gl
      </div>
    </div>
  );
});

// ============================================================================
// Preset Patches (Built-in demos)
// ============================================================================

export const CABLES_PRESETS = {
  /** Simple audio visualizer */
  audioVisualizer: {
    patchId: 'jEdwWB',
    name: 'Audio Visualizer',
  },
  /** Particle system */
  particles: {
    patchId: '5qLHWQ',
    name: 'Particle System',
  },
  /** Generative noise */
  noise: {
    patchId: 'YN9wBZ',
    name: 'Generative Noise',
  },
} as const;

export default CablesGLEmbed;
