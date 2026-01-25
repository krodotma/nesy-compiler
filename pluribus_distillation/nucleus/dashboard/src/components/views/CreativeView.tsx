/**
 * CreativeView.tsx - Unified Creative Tools View
 *
 * Dashboard view that integrates all creative visualization and animation tools:
 * - Cables.gl: Node-based WebGL visuals
 * - p5.js: Creative coding sketches
 * - ManimCE: Mathematical animations
 *
 * Supports tab-based navigation and tool-specific configurations.
 */

import { component$, useSignal, useStore, $ } from '@builder.io/qwik';
import { CablesGLEmbed, CABLES_PRESETS } from '../creative/CablesGLEmbed';
import { P5Canvas, P5_PRESETS } from '../creative/P5Canvas';
import { ManimPreview, MANIM_TEMPLATES } from '../creative/ManimPreview';

// ============================================================================
// Types
// ============================================================================

type CreativeTool = 'cables' | 'p5' | 'manim' | 'overview';

interface CreativeViewState {
  activeTool: CreativeTool;
  cablesPreset: keyof typeof CABLES_PRESETS | 'custom';
  p5Preset: keyof typeof P5_PRESETS | 'custom';
  manimTemplate: keyof typeof MANIM_TEMPLATES;
  customCablesUrl: string;
  customP5Code: string;
}

// ============================================================================
// Tool Info
// ============================================================================

const TOOL_INFO = {
  cables: {
    name: 'Cables.gl',
    icon: 'C',
    description: 'Node-based WebGL visual programming',
    color: 'cyan',
    docs: 'https://cables.gl/docs',
  },
  p5: {
    name: 'p5.js',
    icon: 'P',
    description: 'Creative coding for artists',
    color: 'purple',
    docs: 'https://p5js.org/reference/',
  },
  manim: {
    name: 'ManimCE',
    icon: 'M',
    description: 'Mathematical animations (3Blue1Brown style)',
    color: 'green',
    docs: 'https://docs.manim.community/',
  },
  overview: {
    name: 'Overview',
    icon: 'O',
    description: 'Creative tools dashboard',
    color: 'white',
    docs: '',
  },
} as const;

// ============================================================================
// Main Component
// ============================================================================

export const CreativeView = component$(() => {
  const state = useStore<CreativeViewState>({
    activeTool: 'overview',
    cablesPreset: 'audioVisualizer',
    p5Preset: 'particles',
    manimTemplate: 'circle',
    customCablesUrl: '',
    customP5Code: '',
  });

  const setTool = $((tool: CreativeTool) => {
    state.activeTool = tool;
  });

  return (
    <div class="h-full flex flex-col">
      {/* Header with Tool Tabs */}
      <div class="flex items-center gap-1 p-2 border-b border-border/50 bg-black/30">
        {(Object.keys(TOOL_INFO) as CreativeTool[]).map((tool) => {
          const info = TOOL_INFO[tool];
          const isActive = state.activeTool === tool;

          return (
            <button
              key={tool}
              class={`px-4 py-2 rounded-t text-sm font-medium transition-colors flex items-center gap-2 ${
                isActive
                  ? `bg-card border border-b-0 border-border/50 text-${info.color}-400`
                  : 'text-muted-foreground hover:text-foreground hover:bg-white/5'
              }`}
              onClick$={() => setTool(tool)}
            >
              <span class={`w-5 h-5 rounded flex items-center justify-center text-xs font-bold ${
                isActive ? `bg-${info.color}-500/20` : 'bg-white/10'
              }`}>
                {info.icon}
              </span>
              {info.name}
            </button>
          );
        })}

        <div class="flex-1" />

        {/* Active Tool Docs Link */}
        {state.activeTool !== 'overview' && TOOL_INFO[state.activeTool].docs && (
          <a
            href={TOOL_INFO[state.activeTool].docs}
            target="_blank"
            rel="noopener noreferrer"
            class="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            Docs
          </a>
        )}
      </div>

      {/* Content Area */}
      <div class="flex-1 overflow-auto p-4">
        {/* Overview */}
        {state.activeTool === 'overview' && (
          <div class="space-y-6">
            <div class="text-center py-8">
              <h1 class="text-3xl font-bold mb-2">Creative Tools</h1>
              <p class="text-muted-foreground max-w-xl mx-auto">
                Visualization and animation tools integrated into the Pluribus dashboard.
                Use these to create interactive visuals, creative coding sketches, and
                mathematical animations.
              </p>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Cables.gl Card */}
              <div
                class="p-6 rounded-lg border border-cyan-500/30 bg-cyan-500/5 hover:bg-cyan-500/10 cursor-pointer transition-colors"
                onClick$={() => setTool('cables')}
              >
                <div class="flex items-center gap-3 mb-4">
                  <span class="w-10 h-10 rounded-lg bg-cyan-500/20 flex items-center justify-center text-cyan-400 text-xl font-bold">
                    C
                  </span>
                  <div>
                    <div class="font-semibold text-cyan-400">Cables.gl</div>
                    <div class="text-xs text-muted-foreground">WebGL Visual Editor</div>
                  </div>
                </div>
                <p class="text-sm text-muted-foreground mb-4">
                  Node-based visual programming for creating interactive WebGL experiences.
                  Load patches by URL or embed inline JSON.
                </p>
                <div class="text-xs text-cyan-400/70">
                  Click to open
                </div>
              </div>

              {/* p5.js Card */}
              <div
                class="p-6 rounded-lg border border-purple-500/30 bg-purple-500/5 hover:bg-purple-500/10 cursor-pointer transition-colors"
                onClick$={() => setTool('p5')}
              >
                <div class="flex items-center gap-3 mb-4">
                  <span class="w-10 h-10 rounded-lg bg-purple-500/20 flex items-center justify-center text-purple-400 text-xl font-bold">
                    P
                  </span>
                  <div>
                    <div class="font-semibold text-purple-400">p5.js</div>
                    <div class="text-xs text-muted-foreground">Creative Coding</div>
                  </div>
                </div>
                <p class="text-sm text-muted-foreground mb-4">
                  JavaScript library for creative coding. Write sketches to visualize
                  data, create generative art, or build interactive experiences.
                </p>
                <div class="text-xs text-purple-400/70">
                  Click to open
                </div>
              </div>

              {/* ManimCE Card */}
              <div
                class="p-6 rounded-lg border border-green-500/30 bg-green-500/5 hover:bg-green-500/10 cursor-pointer transition-colors"
                onClick$={() => setTool('manim')}
              >
                <div class="flex items-center gap-3 mb-4">
                  <span class="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center text-green-400 text-xl font-bold">
                    M
                  </span>
                  <div>
                    <div class="font-semibold text-green-400">ManimCE</div>
                    <div class="text-xs text-muted-foreground">Math Animations</div>
                  </div>
                </div>
                <p class="text-sm text-muted-foreground mb-4">
                  Create mathematical animations like 3Blue1Brown. Server-side
                  rendering with video/GIF export.
                </p>
                <div class="text-xs text-green-400/70">
                  Click to open
                </div>
              </div>
            </div>

            {/* Integration Info */}
            <div class="mt-8 p-4 rounded-lg border border-border/50 bg-black/20">
              <h3 class="font-semibold mb-2">Bus Integration</h3>
              <p class="text-sm text-muted-foreground mb-3">
                All creative tools emit events to the Pluribus bus, enabling:
              </p>
              <ul class="text-sm text-muted-foreground space-y-1 list-disc list-inside">
                <li>Real-time data visualization from bus events</li>
                <li>Trigger renders based on agent actions</li>
                <li>Export artifacts to Rhizome</li>
                <li>Coordinate animations across tools</li>
              </ul>
            </div>
          </div>
        )}

        {/* Cables.gl View */}
        {state.activeTool === 'cables' && (
          <div class="space-y-4">
            {/* Preset Selector */}
            <div class="flex items-center gap-4">
              <label class="text-sm font-medium">Preset:</label>
              <select
                class="px-3 py-1.5 rounded bg-black/50 border border-border/50 text-sm"
                value={state.cablesPreset}
                onChange$={(e) => {
                  state.cablesPreset = (e.target as HTMLSelectElement).value as keyof typeof CABLES_PRESETS | 'custom';
                }}
              >
                {Object.entries(CABLES_PRESETS).map(([key, preset]) => (
                  <option key={key} value={key}>
                    {preset.name}
                  </option>
                ))}
                <option value="custom">Custom URL</option>
              </select>

              {state.cablesPreset === 'custom' && (
                <input
                  type="text"
                  placeholder="Patch URL or ID"
                  class="flex-1 px-3 py-1.5 rounded bg-black/50 border border-border/50 text-sm"
                  value={state.customCablesUrl}
                  onInput$={(e) => {
                    state.customCablesUrl = (e.target as HTMLInputElement).value;
                  }}
                />
              )}
            </div>

            {/* Cables Embed */}
            <CablesGLEmbed
              patch={
                state.cablesPreset === 'custom'
                  ? { url: state.customCablesUrl }
                  : { patchId: CABLES_PRESETS[state.cablesPreset].patchId }
              }
              width="100%"
              height="500px"
              autoPlay={true}
              showControls={true}
              onEvent$={$((event) => {
                console.log('[CreativeView] Cables event:', event);
              })}
            />
          </div>
        )}

        {/* p5.js View */}
        {state.activeTool === 'p5' && (
          <div class="space-y-4">
            {/* Preset Selector */}
            <div class="flex items-center gap-4">
              <label class="text-sm font-medium">Preset:</label>
              <select
                class="px-3 py-1.5 rounded bg-black/50 border border-border/50 text-sm"
                value={state.p5Preset}
                onChange$={(e) => {
                  state.p5Preset = (e.target as HTMLSelectElement).value as keyof typeof P5_PRESETS | 'custom';
                }}
              >
                {Object.keys(P5_PRESETS).map((key) => (
                  <option key={key} value={key}>
                    {key.replace(/([A-Z])/g, ' $1').trim()}
                  </option>
                ))}
                <option value="custom">Custom Code</option>
              </select>
            </div>

            {/* p5 Canvas */}
            <P5Canvas
              sketchId={`preset-${state.p5Preset}`}
              sketchCode={
                state.p5Preset === 'custom'
                  ? state.customP5Code
                  : P5_PRESETS[state.p5Preset]
              }
              width={800}
              height={500}
              autoPlay={true}
              showControls={true}
              onEvent$={$((event) => {
                console.log('[CreativeView] p5 event:', event);
              })}
            />

            {/* Custom Code Editor */}
            {state.p5Preset === 'custom' && (
              <div class="space-y-2">
                <label class="text-sm font-medium">Custom Sketch Code:</label>
                <textarea
                  class="w-full h-48 p-3 rounded bg-black/50 border border-border/50 mono text-sm resize-none"
                  value={state.customP5Code}
                  onInput$={(e) => {
                    state.customP5Code = (e.target as HTMLTextAreaElement).value;
                  }}
                  placeholder={`// p5.js sketch (instance mode)
p.setup = () => {
  p.createCanvas(800, 500);
};

p.draw = () => {
  p.background(0);
  p.fill(255);
  p.ellipse(p.mouseX, p.mouseY, 50, 50);
};`}
                />
              </div>
            )}
          </div>
        )}

        {/* ManimCE View */}
        {state.activeTool === 'manim' && (
          <div class="space-y-4">
            <ManimPreview
              initialCode={MANIM_TEMPLATES[state.manimTemplate]}
              serverUrl="http://localhost:9210"
              showEditor={true}
              height="600px"
              onEvent$={$((event) => {
                console.log('[CreativeView] Manim event:', event);
              })}
            />
          </div>
        )}
      </div>
    </div>
  );
});

export default CreativeView;
