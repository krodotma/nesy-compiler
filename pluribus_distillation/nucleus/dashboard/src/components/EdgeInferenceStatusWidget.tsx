import { component$, useStore, useVisibleTask$ } from '@builder.io/qwik';
import { checkAllBackends, getAvailableModels } from '../lib/webllm-enhanced';

// M3 Components - EdgeInferenceStatusWidget
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/progress/circular-progress.js';
import '@material/web/chips/assist-chip.js';

interface BackendStatus {
  available: boolean;
  version: string | null;
  error: string | null;
}

interface State {
  loading: boolean;
  error: string | null;
  webgpuAvailable: boolean;
  webgpuInfo: string | null;
  backends: Record<string, BackendStatus>;
  modelCount: number;
  deviceMemoryGB: number | null;
  cores: number | null;
}

export const EdgeInferenceStatusWidget = component$(() => {
  const state = useStore<State>({
    loading: true,
    error: null,
    webgpuAvailable: false,
    webgpuInfo: null,
    backends: {},
    modelCount: 0,
    deviceMemoryGB: null,
    cores: null,
  });

  useVisibleTask$(async () => {
    state.loading = true;
    state.error = null;

    try {
      const models = getAvailableModels();
      state.modelCount = models.length;
    } catch (err) {
      state.error = err instanceof Error ? err.message : String(err);
    }

    try {
      const memory = typeof navigator !== 'undefined'
        ? Number((navigator as Navigator & { deviceMemory?: number }).deviceMemory || 0)
        : 0;
      state.deviceMemoryGB = memory > 0 ? memory : null;
      state.cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency : null;
    } catch {
      state.deviceMemoryGB = null;
      state.cores = null;
    }

    try {
      const hasWebGPU = typeof navigator !== 'undefined' && (navigator as any).gpu;
      state.webgpuAvailable = Boolean(hasWebGPU);
      if (hasWebGPU && (navigator as any).gpu.requestAdapter) {
        const adapter = await (navigator as any).gpu.requestAdapter();
        if (adapter && typeof adapter.requestAdapterInfo === 'function') {
          const info = await adapter.requestAdapterInfo();
          const desc = String(info?.description || info?.vendor || info?.architecture || '').trim();
          state.webgpuInfo = desc || null;
        }
      }
    } catch {
      state.webgpuInfo = null;
    }

    try {
      state.backends = await checkAllBackends();
    } catch (err) {
      state.error = err instanceof Error ? err.message : String(err);
    } finally {
      state.loading = false;
    }
  }, { strategy: 'document-idle' });

  const tfjs = state.backends['tfjs'];
  const onnx = state.backends['onnx'];

  return (
    <div class="rounded-lg border border-border bg-card p-4" data-testid="edge-inference-status">
      <div class="flex items-center gap-2 mb-2">
        <div class="text-sm font-semibold text-muted-foreground">EDGE INFERENCE (BROWSER)</div>
        {state.loading && <div class="text-[10px] text-muted-foreground">checking…</div>}
        <div class="ml-auto text-[10px] text-muted-foreground">
          {state.modelCount} WebLLM models
        </div>
      </div>

      {state.error && (
        <div class="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2 mb-2">
          {state.error}
        </div>
      )}

      <div class="grid grid-cols-2 gap-2 text-xs">
        <div class={`rounded border border-border/60 p-2 ${state.webgpuAvailable ? 'bg-cyan-500/5' : 'bg-muted/20'}`}>
          <div class="flex items-center gap-2">
            <span class={`w-2 h-2 rounded-full ${state.webgpuAvailable ? 'bg-cyan-400' : 'bg-gray-400'}`} />
            <span class="mono">webgpu</span>
          </div>
          <div class="text-[10px] text-muted-foreground mt-1">
            {state.webgpuAvailable ? (state.webgpuInfo || 'available') : 'unavailable'}
          </div>
        </div>

        <div class="rounded border border-border/60 p-2 bg-muted/20">
          <div class="flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-green-400" />
            <span class="mono">webllm</span>
          </div>
          <div class="text-[10px] text-muted-foreground mt-1">
            native (prebuilt)
          </div>
        </div>

        <div class={`rounded border border-border/60 p-2 ${tfjs?.available ? 'bg-green-500/5' : 'bg-muted/20'}`}>
          <div class="flex items-center gap-2">
            <span class={`w-2 h-2 rounded-full ${tfjs?.available ? 'bg-green-400' : 'bg-gray-400'}`} />
            <span class="mono">tfjs</span>
          </div>
          <div class="text-[10px] text-muted-foreground mt-1">
            {tfjs?.available ? `v${tfjs.version || 'unknown'}` : (tfjs?.error || 'not loaded')}
          </div>
        </div>

        <div class={`rounded border border-border/60 p-2 ${onnx?.available ? 'bg-green-500/5' : 'bg-muted/20'}`}>
          <div class="flex items-center gap-2">
            <span class={`w-2 h-2 rounded-full ${onnx?.available ? 'bg-green-400' : 'bg-gray-400'}`} />
            <span class="mono">onnx</span>
          </div>
          <div class="text-[10px] text-muted-foreground mt-1">
            {onnx?.available ? `v${onnx.version || 'unknown'}` : (onnx?.error || 'not loaded')}
          </div>
        </div>
      </div>

      <div class="text-[10px] text-muted-foreground mt-2">
        {state.deviceMemoryGB ? `${state.deviceMemoryGB}GB` : 'device memory unknown'}
        {state.cores ? ` · ${state.cores} cores` : ''}
      </div>
    </div>
  );
});

export default EdgeInferenceStatusWidget;
