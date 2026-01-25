/**
 * Vision3DWidget - 3D Scene Inference Dashboard Component
 *
 * Integrates SOTA vision tools:
 * - VGGT: Single-image 3D inference (depth + normals + mesh)
 * - GEN3C: 3D-consistent video generation
 * - Live2Diff: Real-time video stylization (~16 FPS)
 *
 * Features:
 * - Image upload for 3D inference
 * - Depth/Normal/Mesh visualization
 * - Style preview for Live2Diff
 * - Service status monitoring
 */

import { component$, useSignal, useStore, $, useVisibleTask$ } from '@builder.io/qwik';

// M3 Components - Vision3DWidget
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/progress/circular-progress.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

// Types
interface ServiceStatus {
  id: string;
  name: string;
  port: number;
  status: 'unknown' | 'checking' | 'online' | 'offline';
  gpu: boolean | null;
  error: string | null;
}

interface VGGTResult {
  depth_map_b64: string | null;
  normal_map_b64: string | null;
  mesh_obj: string | null;
  inference_time_ms: number;
  status: string;
  error: string | null;
  metadata: Record<string, unknown>;
}

interface Live2DiffStyles {
  [key: string]: string;
}

interface WidgetState {
  services: ServiceStatus[];
  selectedImage: string | null;
  vggtResult: VGGTResult | null;
  selectedStyle: string;
  availableStyles: Live2DiffStyles;
  isProcessing: boolean;
  error: string | null;
}

const VISION_SERVICES: ServiceStatus[] = [
  { id: 'vggt-inference', name: 'VGGT 3D', port: 9301, status: 'unknown', gpu: null, error: null },
  { id: 'gen3c-video', name: 'GEN3C', port: 9302, status: 'unknown', gpu: null, error: null },
  { id: 'live2diff-stream', name: 'Live2Diff', port: 9303, status: 'unknown', gpu: null, error: null },
];

const DEFAULT_STYLES: Live2DiffStyles = {
  anime: 'Anime/Cartoon style',
  watercolor: 'Soft watercolor',
  oil_paint: 'Oil painting',
  sketch: 'Pencil sketch',
  neon: 'Cyberpunk neon',
  vintage: 'Sepia vintage',
  comic: 'Comic book',
  pixel: 'Pixel art',
};

export const Vision3DWidget = component$(() => {
  const state = useStore<WidgetState>({
    services: [...VISION_SERVICES],
    selectedImage: null,
    vggtResult: null,
    selectedStyle: 'anime',
    availableStyles: DEFAULT_STYLES,
    isProcessing: false,
    error: null,
  });

  const fileInputRef = useSignal<HTMLInputElement>();
  const activeTab = useSignal<'vggt' | 'gen3c' | 'live2diff'>('vggt');

  // Check service health on mount
  useVisibleTask$(({ cleanup }) => {
    const checkHealth = async () => {
      for (const service of state.services) {
        service.status = 'checking';
        try {
          const response = await fetch(`http://localhost:${service.port}/health`, {
            method: 'GET',
            signal: AbortSignal.timeout(3000),
          });
          if (response.ok) {
            const data = await response.json();
            service.status = 'online';
            service.gpu = data.gpu ?? null;
            service.error = null;
            // Update styles for Live2Diff
            if (service.id === 'live2diff-stream' && data.styles) {
              state.availableStyles = data.styles.reduce((acc: Live2DiffStyles, s: string) => {
                acc[s] = DEFAULT_STYLES[s] || s;
                return acc;
              }, {});
            }
          } else {
            service.status = 'offline';
          }
        } catch (e) {
          service.status = 'offline';
          service.error = e instanceof Error ? e.message : 'Connection failed';
        }
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    cleanup(() => clearInterval(interval));
  });

  // Handle file selection
  const handleFileSelect = $((event: Event) => {
    const target = event.target as HTMLInputElement;
    const file = target.files?.[0];
    if (file && file.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onload = (e) => {
        state.selectedImage = e.target?.result as string;
        state.vggtResult = null;
        state.error = null;
      };
      reader.readAsDataURL(file);
    }
  });

  // Run VGGT inference
  const runVGGTInference = $(async () => {
    if (!state.selectedImage) {
      state.error = 'Please select an image first';
      return;
    }

    const vggtService = state.services.find(s => s.id === 'vggt-inference');
    if (vggtService?.status !== 'online') {
      state.error = 'VGGT service is not available';
      return;
    }

    state.isProcessing = true;
    state.error = null;

    try {
      // For actual deployment, we'd send to the service
      // For now, show mock result structure
      const response = await fetch(`http://localhost:${vggtService.port}/infer`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: state.selectedImage.split(',')[1], // Remove data:image/... prefix
        }),
        signal: AbortSignal.timeout(30000),
      });

      if (response.ok) {
        state.vggtResult = await response.json();
      } else {
        state.error = 'Inference failed';
      }
    } catch (e) {
      state.error = e instanceof Error ? e.message : 'Request failed';
    } finally {
      state.isProcessing = false;
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="p-4 border-b border-border">
        <div class="flex items-center gap-3">
          <span class="text-2xl">🔮</span>
          <h3 class="font-semibold">Vision & 3D Tools</h3>
          <span class="text-xs px-2 py-0.5 rounded bg-purple-500/20 text-purple-400">SOTA</span>
        </div>
      </div>

      {/* Service Status Row */}
      <div class="p-4 border-b border-border">
        <div class="flex gap-4 flex-wrap">
          {state.services.map((service) => (
            <div
              key={service.id}
              class={[
                'flex items-center gap-2 px-3 py-1.5 rounded text-sm',
                service.status === 'online' ? 'bg-green-500/10 border border-green-500/30' :
                service.status === 'checking' ? 'bg-yellow-500/10 border border-yellow-500/30' :
                'bg-red-500/10 border border-red-500/30',
              ].join(' ')}
            >
              <span class={[
                'w-2 h-2 rounded-full',
                service.status === 'online' ? 'bg-green-500' :
                service.status === 'checking' ? 'bg-yellow-500 animate-pulse' :
                'bg-red-500',
              ].join(' ')} />
              <span>{service.name}</span>
              {service.gpu !== null && (
                <span class="text-xs opacity-60">
                  {service.gpu ? 'GPU' : 'CPU'}
                </span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Tab Navigation */}
      <div class="flex border-b border-border">
        <button
          class={[
            'px-4 py-2 text-sm font-medium',
            activeTab.value === 'vggt'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground hover:text-foreground',
          ].join(' ')}
          onClick$={() => activeTab.value = 'vggt'}
        >
          VGGT 3D Inference
        </button>
        <button
          class={[
            'px-4 py-2 text-sm font-medium',
            activeTab.value === 'gen3c'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground hover:text-foreground',
          ].join(' ')}
          onClick$={() => activeTab.value = 'gen3c'}
        >
          GEN3C Video
        </button>
        <button
          class={[
            'px-4 py-2 text-sm font-medium',
            activeTab.value === 'live2diff'
              ? 'border-b-2 border-primary text-primary'
              : 'text-muted-foreground hover:text-foreground',
          ].join(' ')}
          onClick$={() => activeTab.value = 'live2diff'}
        >
          Live2Diff
        </button>
      </div>

      {/* Tab Content */}
      <div class="p-4">
        {/* VGGT Tab */}
        {activeTab.value === 'vggt' && (
          <div class="space-y-4">
            <div class="text-sm text-muted-foreground">
              Upload an image to generate depth map, surface normals, and 3D mesh.
            </div>

            {/* Image Upload */}
            <div class="flex gap-4">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                class="hidden"
                onChange$={handleFileSelect}
              />
              <button
                class="px-4 py-2 rounded border border-border bg-muted/20 hover:bg-muted/40 text-sm"
                onClick$={() => fileInputRef.value?.click()}
              >
                Select Image
              </button>
              <button
                class="px-4 py-2 rounded bg-primary text-primary-foreground hover:bg-primary/90 text-sm disabled:opacity-50"
                disabled={!state.selectedImage || state.isProcessing}
                onClick$={runVGGTInference}
              >
                {state.isProcessing ? 'Processing...' : 'Run VGGT Inference'}
              </button>
            </div>

            {/* Error Display */}
            {state.error && (
              <div class="p-3 rounded bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
                {state.error}
              </div>
            )}

            {/* Results Grid */}
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
              {/* Original Image */}
              <div class="rounded border border-border overflow-hidden">
                <div class="p-2 bg-muted/20 text-xs font-medium">Input</div>
                <div class="aspect-square bg-muted/10 flex items-center justify-center">
                  {state.selectedImage ? (
                    <img src={state.selectedImage} alt="Input" class="w-full h-full object-contain" />
                  ) : (
                    <span class="text-muted-foreground text-sm">No image</span>
                  )}
                </div>
              </div>

              {/* Depth Map */}
              <div class="rounded border border-border overflow-hidden">
                <div class="p-2 bg-muted/20 text-xs font-medium">Depth</div>
                <div class="aspect-square bg-muted/10 flex items-center justify-center">
                  {state.vggtResult?.depth_map_b64 ? (
                    <img
                      src={`data:image/png;base64,${state.vggtResult.depth_map_b64}`}
                      alt="Depth"
                      class="w-full h-full object-contain"
                    />
                  ) : (
                    <span class="text-muted-foreground text-sm">--</span>
                  )}
                </div>
              </div>

              {/* Normal Map */}
              <div class="rounded border border-border overflow-hidden">
                <div class="p-2 bg-muted/20 text-xs font-medium">Normals</div>
                <div class="aspect-square bg-muted/10 flex items-center justify-center">
                  {state.vggtResult?.normal_map_b64 ? (
                    <img
                      src={`data:image/png;base64,${state.vggtResult.normal_map_b64}`}
                      alt="Normals"
                      class="w-full h-full object-contain"
                    />
                  ) : (
                    <span class="text-muted-foreground text-sm">--</span>
                  )}
                </div>
              </div>

              {/* Mesh Info */}
              <div class="rounded border border-border overflow-hidden">
                <div class="p-2 bg-muted/20 text-xs font-medium">Mesh</div>
                <div class="aspect-square bg-muted/10 flex items-center justify-center p-2">
                  {state.vggtResult?.mesh_obj ? (
                    <div class="text-center">
                      <div class="text-2xl">🗿</div>
                      <div class="text-xs text-muted-foreground mt-1">
                        {state.vggtResult.mesh_obj.split('\n').length} lines
                      </div>
                      <button
                        class="mt-2 px-2 py-1 text-xs rounded bg-muted/20 hover:bg-muted/40"
                        onClick$={() => {
                          if (state.vggtResult?.mesh_obj) {
                            const blob = new Blob([state.vggtResult.mesh_obj], { type: 'text/plain' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'mesh.obj';
                            a.click();
                            URL.revokeObjectURL(url);
                          }
                        }}
                      >
                        Download OBJ
                      </button>
                    </div>
                  ) : (
                    <span class="text-muted-foreground text-sm">--</span>
                  )}
                </div>
              </div>
            </div>

            {/* Inference Stats */}
            {state.vggtResult && (
              <div class="flex gap-4 text-sm text-muted-foreground">
                <span>Time: {state.vggtResult.inference_time_ms.toFixed(1)}ms</span>
                <span>Mode: {String(state.vggtResult.metadata?.mode || 'unknown')}</span>
                <span>Status: {state.vggtResult.status}</span>
              </div>
            )}
          </div>
        )}

        {/* GEN3C Tab */}
        {activeTab.value === 'gen3c' && (
          <div class="space-y-4">
            <div class="text-sm text-muted-foreground">
              Generate 3D-consistent video from sparse view images.
            </div>
            <div class="p-8 rounded border border-dashed border-border text-center">
              <div class="text-4xl mb-2">🎬</div>
              <div class="text-muted-foreground">
                GEN3C video generation interface coming soon.
              </div>
              <div class="text-xs text-muted-foreground mt-2">
                Upload multiple view images to generate fly-through videos.
              </div>
            </div>
          </div>
        )}

        {/* Live2Diff Tab */}
        {activeTab.value === 'live2diff' && (
          <div class="space-y-4">
            <div class="text-sm text-muted-foreground">
              Real-time video stylization at ~16 FPS.
            </div>

            {/* Style Selector */}
            <div class="space-y-2">
              <div class="text-sm font-medium">Available Styles</div>
              <div class="grid grid-cols-4 md:grid-cols-8 gap-2">
                {Object.entries(state.availableStyles).map(([style, desc]) => (
                  <button
                    key={style}
                    class={[
                      'px-3 py-2 rounded text-sm border transition-all',
                      state.selectedStyle === style
                        ? 'bg-primary text-primary-foreground border-primary'
                        : 'bg-muted/20 border-border hover:bg-muted/40',
                    ].join(' ')}
                    onClick$={() => state.selectedStyle = style}
                    title={String(desc)}
                  >
                    {style}
                  </button>
                ))}
              </div>
            </div>

            <div class="p-8 rounded border border-dashed border-border text-center">
              <div class="text-4xl mb-2">🎨</div>
              <div class="text-muted-foreground">
                Live video stylization preview coming soon.
              </div>
              <div class="text-xs text-muted-foreground mt-2">
                Connect webcam or upload video for real-time style transfer.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});

export default Vision3DWidget;
