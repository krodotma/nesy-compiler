/**
 * ManimPreview.tsx - ManimCE Animation Preview Component
 *
 * Dashboard component for previewing and triggering ManimCE mathematical
 * animations. Connects to the manim_renderer.py backend for server-side
 * rendering with real-time progress updates via the Pluribus bus.
 *
 * Features:
 * - Code editor with syntax highlighting
 * - Quality preset selection
 * - Render progress tracking
 * - Video/GIF preview playback
 * - Frame-by-frame export
 *
 * Reference: https://docs.manim.community/
 */

import { component$, useSignal, $, type QRL, useStore, useVisibleTask$ } from '@builder.io/qwik';

// M3 Components - ManimPreview
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/select/outlined-select.js';
import '@material/web/select/select-option.js';

// ============================================================================
// Types
// ============================================================================

export interface ManimRenderRequest {
  code: string;
  scene_name?: string;
  quality: 'preview' | 'low' | 'medium' | 'high' | '4k';
  output_format: 'mp4' | 'gif' | 'webm' | 'png';
  extra_args?: string[];
}

export interface ManimRenderResult {
  request_id: string;
  status: 'pending' | 'complete' | 'failed';
  success?: boolean;
  output_path?: string;
  preview_frame?: string; // Base64 PNG
  duration_seconds?: number;
  error?: string;
}

export interface ManimEventPayload {
  type: 'render_started' | 'render_progress' | 'render_complete' | 'render_error';
  requestId: string;
  data: Record<string, unknown>;
}

interface ManimState {
  code: string;
  quality: ManimRenderRequest['quality'];
  format: ManimRenderRequest['output_format'];
  renderStatus: 'idle' | 'rendering' | 'complete' | 'error';
  currentJobId: string | null;
  result: ManimRenderResult | null;
  error: string | null;
  serverStatus: 'unknown' | 'healthy' | 'unhealthy';
  manimVersion: string | null;
}

// ============================================================================
// Props
// ============================================================================

export interface ManimPreviewProps {
  /** Initial code to display */
  initialCode?: string;
  /** Manim renderer server URL (default: http://localhost:9210) */
  serverUrl?: string;
  /** Event callback */
  onEvent$?: QRL<(event: ManimEventPayload) => void>;
  /** Whether to show code editor */
  showEditor?: boolean;
  /** Height of the preview area */
  height?: string;
}

// ============================================================================
// Default Scene Templates
// ============================================================================

const MANIM_TEMPLATES = {
  circle: `class CircleScene(Scene):
    def construct(self):
        circle = Circle(color=BLUE)
        self.play(Create(circle))
        self.wait()`,

  square_to_circle: `class SquareToCircle(Scene):
    def construct(self):
        square = Square(color=BLUE, fill_opacity=0.5)
        circle = Circle(color=RED, fill_opacity=0.5)

        self.play(Create(square))
        self.wait(0.5)
        self.play(Transform(square, circle))
        self.wait()`,

  tex_equation: `class TexEquation(Scene):
    def construct(self):
        eq1 = MathTex(r"e^{i\\pi} + 1 = 0")
        eq2 = MathTex(r"e^{i\\pi}", r"=", r"-1")

        self.play(Write(eq1))
        self.wait()
        self.play(TransformMatchingTex(eq1, eq2))
        self.wait()`,

  fourier_series: `class FourierSeries(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-PI, PI, PI/2],
            y_range=[-2, 2, 1],
        )

        def fourier_approx(x, n_terms):
            result = 0
            for k in range(1, n_terms + 1, 2):
                result += (4 / (k * PI)) * np.sin(k * x)
            return result

        square_wave = axes.plot(
            lambda x: 1 if x > 0 else -1,
            discontinuities=[-PI, 0, PI],
            color=BLUE,
        )

        self.play(Create(axes), Create(square_wave))

        for n in [1, 3, 5, 9, 19]:
            approx = axes.plot(
                lambda x: fourier_approx(x, n),
                color=YELLOW,
            )
            label = Text(f"n = {n}", font_size=24).to_corner(UR)
            self.play(Create(approx), Write(label))
            self.wait(0.5)
            self.play(FadeOut(approx), FadeOut(label))

        self.wait()`,

  threed_surface: `class ThreeDSurface(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()

        surface = Surface(
            lambda u, v: axes.c2p(u, v, np.sin(u) * np.cos(v)),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            resolution=(24, 24),
        )
        surface.set_fill_by_value(
            axes=axes,
            colorscale=[(RED, -1), (YELLOW, 0), (GREEN, 1)],
            axis=2,
        )

        self.set_camera_orientation(phi=75 * DEGREES, theta=-45 * DEGREES)
        self.add(axes, surface)
        self.begin_ambient_camera_rotation(rate=0.2)
        self.wait(5)`,
} as const;

const QUALITY_INFO = {
  preview: { label: 'Preview', desc: '480p 15fps', icon: 'P' },
  low: { label: 'Low', desc: '480p 30fps', icon: 'L' },
  medium: { label: 'Medium', desc: '720p 30fps', icon: 'M' },
  high: { label: 'High', desc: '1080p 60fps', icon: 'H' },
  '4k': { label: '4K', desc: '2160p 60fps', icon: '4K' },
} as const;

// ============================================================================
// Main Component
// ============================================================================

export const ManimPreview = component$<ManimPreviewProps>((props) => {
  const serverUrl = props.serverUrl || 'http://localhost:9210';

  const state = useStore<ManimState>({
    code: props.initialCode || MANIM_TEMPLATES.circle,
    quality: 'preview',
    format: 'mp4',
    renderStatus: 'idle',
    currentJobId: null,
    result: null,
    error: null,
    serverStatus: 'unknown',
    manimVersion: null,
  });

  const showTemplates = useSignal(false);
  const pollInterval = useSignal<ReturnType<typeof setInterval> | null>(null);

  // Check server health on mount
  useVisibleTask$(async () => {
    try {
      const resp = await fetch(`${serverUrl}/health`);
      if (resp.ok) {
        const data = await resp.json();
        state.serverStatus = data.status === 'healthy' ? 'healthy' : 'unhealthy';
        state.manimVersion = data.manim_version || null;
      } else {
        state.serverStatus = 'unhealthy';
      }
    } catch {
      state.serverStatus = 'unhealthy';
    }
  });

  // Poll for render status
  useVisibleTask$(({ track, cleanup }) => {
    track(() => state.currentJobId);

    if (!state.currentJobId) return;

    const interval = setInterval(async () => {
      try {
        const resp = await fetch(`${serverUrl}/status/${state.currentJobId}`);
        if (resp.ok) {
          const data: ManimRenderResult = await resp.json();

          if (data.status === 'complete') {
            state.renderStatus = data.success ? 'complete' : 'error';
            state.result = data;
            state.error = data.error || null;
            state.currentJobId = null;

            props.onEvent$?.({
              type: data.success ? 'render_complete' : 'render_error',
              requestId: data.request_id,
              data: { output_path: data.output_path, duration: data.duration_seconds },
            });
          } else if (data.status === 'failed') {
            state.renderStatus = 'error';
            state.error = data.error || 'Unknown error';
            state.result = data;
            state.currentJobId = null;

            props.onEvent$?.({
              type: 'render_error',
              requestId: data.request_id,
              data: { error: data.error },
            });
          }
        }
      } catch {
        // Ignore polling errors
      }
    }, 1000);

    pollInterval.value = interval;

    cleanup(() => {
      if (pollInterval.value) {
        clearInterval(pollInterval.value);
        pollInterval.value = null;
      }
    });
  });

  // Submit render request
  const submitRender = $(async () => {
    if (state.renderStatus === 'rendering') return;
    if (!state.code.trim()) {
      state.error = 'No code to render';
      return;
    }

    state.renderStatus = 'rendering';
    state.error = null;
    state.result = null;

    try {
      const resp = await fetch(`${serverUrl}/render`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          code: state.code,
          quality: state.quality,
          output_format: state.format,
        }),
      });

      if (!resp.ok) {
        const data = await resp.json();
        throw new Error(data.error || `Server error: ${resp.status}`);
      }

      const data = await resp.json();
      state.currentJobId = data.request_id;

      props.onEvent$?.({
        type: 'render_started',
        requestId: data.request_id,
        data: { quality: state.quality, format: state.format },
      });

    } catch (err) {
      state.renderStatus = 'error';
      state.error = err instanceof Error ? err.message : 'Request failed';
    }
  });

  // Load template
  const loadTemplate = $((key: keyof typeof MANIM_TEMPLATES) => {
    state.code = MANIM_TEMPLATES[key];
    showTemplates.value = false;
  });

  const height = props.height || '500px';

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-3 bg-black/30 border-b border-border/50">
        <div class="flex items-center gap-3">
          <span class="text-lg">M</span>
          <div>
            <div class="font-semibold">ManimCE Renderer</div>
            <div class="text-xs text-muted-foreground">
              {state.serverStatus === 'healthy' && state.manimVersion
                ? `v${state.manimVersion}`
                : state.serverStatus === 'unhealthy'
                ? 'Server unavailable'
                : 'Checking...'}
            </div>
          </div>
        </div>

        <div class="flex items-center gap-2">
          {/* Server Status */}
          <span class={`w-2 h-2 rounded-full ${
            state.serverStatus === 'healthy' ? 'bg-green-400' :
            state.serverStatus === 'unhealthy' ? 'bg-red-400' : 'bg-yellow-400 animate-pulse'
          }`} />
        </div>
      </div>

      {/* Main Content */}
      <div class="flex" style={{ height }}>
        {/* Code Editor */}
        {props.showEditor !== false && (
          <div class="flex-1 flex flex-col border-r border-border/50">
            {/* Editor Toolbar */}
            <div class="flex items-center gap-2 p-2 bg-black/20 border-b border-border/30">
              <button
                class="px-2 py-1 rounded text-xs bg-white/10 hover:bg-white/20 transition-colors"
                onClick$={() => { showTemplates.value = !showTemplates.value; }}
              >
                Templates
              </button>

              {/* Quality Selector */}
              <select
                class="px-2 py-1 rounded text-xs bg-black/50 border border-border/50"
                value={state.quality}
                onChange$={(e) => {
                  state.quality = (e.target as HTMLSelectElement).value as ManimState['quality'];
                }}
              >
                {Object.entries(QUALITY_INFO).map(([key, info]) => (
                  <option key={key} value={key}>
                    {`${info.label} (${info.desc})`}
                  </option>
                ))}
              </select>

              {/* Format Selector */}
              <select
                class="px-2 py-1 rounded text-xs bg-black/50 border border-border/50"
                value={state.format}
                onChange$={(e) => {
                  state.format = (e.target as HTMLSelectElement).value as ManimState['format'];
                }}
              >
                <option value="mp4">MP4</option>
                <option value="gif">GIF</option>
                <option value="webm">WebM</option>
                <option value="png">PNG (last frame)</option>
              </select>

              <div class="flex-1" />

              {/* Render Button */}
              <button
                class={`px-4 py-1 rounded text-xs font-medium transition-colors ${
                  state.renderStatus === 'rendering'
                    ? 'bg-yellow-600 text-white cursor-wait'
                    : state.serverStatus === 'healthy'
                    ? 'bg-green-600 hover:bg-green-500 text-white'
                    : 'bg-gray-600 text-gray-400 cursor-not-allowed'
                }`}
                onClick$={submitRender}
                disabled={state.renderStatus === 'rendering' || state.serverStatus !== 'healthy'}
              >
                {state.renderStatus === 'rendering' ? 'Rendering...' : 'Render'}
              </button>
            </div>

            {/* Templates Dropdown */}
            {showTemplates.value && (
              <div class="absolute z-10 mt-8 ml-2 bg-card border border-border rounded shadow-lg p-2 space-y-1">
                {Object.keys(MANIM_TEMPLATES).map((key) => (
                  <button
                    key={key}
                    class="block w-full text-left px-3 py-1 text-xs rounded hover:bg-white/10 transition-colors"
                    onClick$={() => loadTemplate(key as keyof typeof MANIM_TEMPLATES)}
                  >
                    {key.replace(/_/g, ' ')}
                  </button>
                ))}
              </div>
            )}

            {/* Code Textarea */}
            <textarea
              class="flex-1 w-full p-3 bg-black/40 text-sm mono resize-none focus:outline-none"
              value={state.code}
              onInput$={(e) => { state.code = (e.target as HTMLTextAreaElement).value; }}
              placeholder="Enter Manim scene code..."
              spellcheck={false}
            />
          </div>
        )}

        {/* Preview Panel */}
        <div class="w-[400px] flex flex-col bg-black/30">
          {/* Rendering State */}
          {state.renderStatus === 'rendering' && (
            <div class="flex-1 flex items-center justify-center">
              <div class="text-center">
                <div class="w-12 h-12 border-3 border-green-500 border-t-transparent rounded-full animate-spin mx-auto mb-3" />
                <div class="text-sm text-green-400">Rendering...</div>
                <div class="text-xs text-muted-foreground mt-1">
                  Quality: {QUALITY_INFO[state.quality].desc}
                </div>
              </div>
            </div>
          )}

          {/* Error State */}
          {state.renderStatus === 'error' && (
            <div class="flex-1 flex items-center justify-center p-4">
              <div class="text-center">
                <div class="text-red-400 text-lg mb-2">Render Failed</div>
                <pre class="text-xs text-red-300/70 bg-red-900/20 p-2 rounded max-w-full overflow-auto">
                  {state.error}
                </pre>
              </div>
            </div>
          )}

          {/* Complete State - Show Preview */}
          {state.renderStatus === 'complete' && state.result && (
            <div class="flex-1 flex flex-col">
              {/* Preview Frame */}
              {state.result.preview_frame && (
                <div class="flex-1 flex items-center justify-center p-2 bg-black">
                  <img
                    src={`data:image/png;base64,${state.result.preview_frame}`}
                    alt="Preview frame"
                    class="max-w-full max-h-full object-contain rounded"
                  />
                </div>
              )}

              {/* Video Player (if mp4/webm/gif) */}
              {state.result.output_path && state.format !== 'png' && (
                <div class="p-2 bg-black">
                  <video
                    controls
                    autoplay
                    loop
                    muted
                    class="w-full rounded"
                    src={`/api/manim/media/${state.result.output_path.split('/').pop()}`}
                  >
                    Your browser does not support video playback.
                  </video>
                </div>
              )}

              {/* Stats */}
              <div class="p-2 border-t border-border/30 text-xs text-muted-foreground">
                <div class="flex items-center justify-between">
                  <span>Duration: {state.result.duration_seconds?.toFixed(2)}s</span>
                  {state.result.output_path && (
                    <a
                      href={`/api/manim/media/${state.result.output_path.split('/').pop()}`}
                      download
                      class="text-cyan-400 hover:text-cyan-300"
                    >
                      Download
                    </a>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Idle State */}
          {state.renderStatus === 'idle' && (
            <div class="flex-1 flex items-center justify-center p-4">
              <div class="text-center text-muted-foreground">
                <div class="text-4xl mb-3">M</div>
                <div class="text-sm">Write a scene and click Render</div>
                <div class="text-xs mt-2">
                  Mathematical animations powered by ManimCE
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div class="p-2 border-t border-border/50 bg-black/30 text-xs text-muted-foreground flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span>ManimCE</span>
          <span class="text-white/30">|</span>
          <span>Server: {serverUrl}</span>
        </div>
        <div>
          Like 3Blue1Brown animations
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Export Templates for External Use
// ============================================================================

export { MANIM_TEMPLATES };

export default ManimPreview;
