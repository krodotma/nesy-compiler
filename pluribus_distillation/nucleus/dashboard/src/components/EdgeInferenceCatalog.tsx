import { component$ } from '@builder.io/qwik';
import { getAvailableModels, getEdgeModelCatalog, getModelDefByModelId } from '../lib/webllm-enhanced';

export interface EdgeInferenceCatalogProps {
  compact?: boolean;
}

function renderCaps(caps: string[] | undefined) {
  if (!caps || caps.length === 0) return null;
  return caps.slice(0, 3).map((cap) => (
    <span key={cap} class="px-1.5 py-0.5 rounded bg-muted/40 text-[10px] uppercase tracking-wide">
      {cap}
    </span>
  ));
}

export const EdgeInferenceCatalog = component$<EdgeInferenceCatalogProps>(({ compact = false }) => {
  const webllm = getAvailableModels()
    .map((model) => ({ model, def: getModelDefByModelId(model.id) }))
    .sort((a, b) => (b.def?.speedRating || 0) - (a.def?.speedRating || 0));
  const edge = getEdgeModelCatalog();
  const vllm = edge.filter((m) => m.backend === 'vllm');
  const onnx = edge.filter((m) => m.backend === 'onnx');
  const tfjs = edge.filter((m) => m.backend === 'tfjs');
  const planned = edge.filter((m) => m.backend === 'planned');
  const tfjsCatalog = [...tfjs, ...planned];

  const webllmDisplay = compact ? webllm.slice(0, 6) : webllm;
  const vllmDisplay = compact ? vllm.slice(0, 3) : vllm;
  const onnxDisplay = compact ? onnx.slice(0, 3) : onnx;
  const tfjsDisplay = compact ? tfjsCatalog.slice(0, 2) : tfjsCatalog;

  return (
    <div class="rounded-lg border border-border bg-card p-4" data-testid="edge-inference-catalog">
      <div class="flex items-center gap-2 mb-3">
        <div class="text-sm font-semibold text-muted-foreground">EDGE MODEL CATALOG</div>
        <div class="ml-auto text-[10px] text-muted-foreground">
          {webllm.length} WebLLM · {vllm.length} vLLM · {onnx.length} ONNX · {tfjsCatalog.length} TFJS/Planned
        </div>
      </div>

      <div class="grid gap-3 lg:grid-cols-4">
        <div class="rounded border border-border/60 p-3 bg-muted/20">
          <div class="text-xs font-semibold text-cyan-400 mb-2">WebLLM (WebGPU)</div>
          <div class="space-y-2">
            {webllmDisplay.map(({ model, def }) => (
              <div key={model.id} class="flex items-center justify-between gap-2">
                <div>
                  <div class="text-xs font-medium">{model.name}</div>
                  <div class="flex items-center gap-1 mt-1 text-[10px] text-muted-foreground">
                    <span>{model.vram}</span>
                    {renderCaps(def?.capabilities as string[] | undefined)}
                  </div>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded bg-cyan-500/10 text-cyan-300">
                  prebuilt
                </span>
              </div>
            ))}
          </div>
        </div>

        <div class="rounded border border-border/60 p-3 bg-muted/20">
          <div class="text-xs font-semibold text-blue-300 mb-2">vLLM (Server)</div>
          <div class="space-y-2">
            {vllmDisplay.map((model) => (
              <div key={model.id} class="flex items-center justify-between gap-2">
                <div>
                  <div class="text-xs font-medium">{model.name}</div>
                  <div class="flex items-center gap-1 mt-1 text-[10px] text-muted-foreground">
                    {renderCaps(model.capabilities)}
                  </div>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-300">
                  {model.availability}
                </span>
              </div>
            ))}
          </div>
        </div>

        <div class="rounded border border-border/60 p-3 bg-muted/20">
          <div class="text-xs font-semibold text-emerald-300 mb-2">ONNX Runtime (Browser)</div>
          <div class="space-y-2">
            {onnxDisplay.map((model) => (
              <div key={model.id} class="flex items-center justify-between gap-2">
                <div>
                  <div class="text-xs font-medium">{model.name}</div>
                  <div class="flex items-center gap-1 mt-1 text-[10px] text-muted-foreground">
                    {typeof model.sizeMB === 'number' && <span>{model.sizeMB}MB</span>}
                    {renderCaps(model.capabilities)}
                  </div>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-300">
                  custom
                </span>
              </div>
            ))}
          </div>
        </div>

        <div class="rounded border border-border/60 p-3 bg-muted/20">
          <div class="text-xs font-semibold text-amber-300 mb-2">TFJS (Browser/Planned)</div>
          <div class="space-y-2">
            {tfjsDisplay.map((model) => (
              <div key={model.id} class="flex items-center justify-between gap-2">
                <div>
                  <div class="text-xs font-medium">{model.name}</div>
                  <div class="flex items-center gap-1 mt-1 text-[10px] text-muted-foreground">
                    {typeof model.sizeMB === 'number' && <span>{model.sizeMB}MB</span>}
                    {renderCaps(model.capabilities)}
                  </div>
                </div>
                <span class="text-[10px] px-2 py-0.5 rounded bg-amber-500/10 text-amber-300">
                  {model.availability}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
});

export default EdgeInferenceCatalog;
