import { component$, useSignal, useVisibleTask$, $, type QRL } from '@builder.io/qwik';

import type { IngressFragment } from './PortalIngressSelector';
import {
  loadPortalAssetIndex,
  purgeExpiredPortalAssets,
  stagePortalAsset,
  startPortalAssetBridge,
  type PortalAssetMeta,
} from '../../lib/portal/ingest-assets';

interface PortalIngestDropzoneProps {
  onStage$?: QRL<(fragment: IngressFragment, asset: PortalAssetMeta) => void>;
}

const formatBytes = (bytes: number): string => {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  let idx = 0;
  let value = bytes;
  while (value >= 1024 && idx < units.length - 1) {
    value /= 1024;
    idx += 1;
  }
  return `${value.toFixed(value >= 10 || idx === 0 ? 0 : 1)} ${units[idx]}`;
};

export const PortalIngestDropzone = component$<PortalIngestDropzoneProps>((props) => {
  const assets = useSignal<PortalAssetMeta[]>([]);
  const isDragging = useSignal(false);
  const error = useSignal<string | null>(null);
  const inputRef = useSignal<HTMLInputElement>();

  useVisibleTask$(async ({ cleanup }) => {
    await purgeExpiredPortalAssets();
    assets.value = loadPortalAssetIndex().assets;
    const stop = startPortalAssetBridge({
      onIndexUpdate: (index) => {
        assets.value = index.assets;
      },
    });
    cleanup(() => stop());
  });

  const handleFiles = $(async (fileList: FileList | File[]) => {
    const list = Array.from(fileList);
    if (list.length === 0) return;
    error.value = null;
    for (const file of list) {
      try {
        const meta = await stagePortalAsset(file);
        assets.value = loadPortalAssetIndex().assets;
        if (props.onStage$) {
          const fragment: IngressFragment = {
            id: meta.id,
            content_preview: `${meta.name} (${formatBytes(meta.byte_size)})`,
            source_type: 'file',
            source_uri: meta.name,
            byte_size: meta.byte_size,
            created_iso: meta.created_iso,
            asset_id: meta.id,
          };
          await props.onStage$(fragment, meta);
        }
      } catch (err) {
        error.value = String(err);
      }
    }
  });

  const handleDrop = $((event: DragEvent) => {
    event.preventDefault();
    isDragging.value = false;
    if (!event.dataTransfer?.files?.length) return;
    void handleFiles(event.dataTransfer.files);
  });

  const handleInputChange = $((event: Event) => {
    const target = event.target as HTMLInputElement | null;
    if (!target?.files) return;
    void handleFiles(target.files);
    target.value = '';
  });

  return (
    <div class="space-y-2">
      <div
        class={[
          'rounded-lg border border-dashed px-3 py-2 transition-colors',
          'bg-card/70 backdrop-blur-sm',
          isDragging.value ? 'border-cyan-400/80 bg-cyan-500/10' : 'border-border/60',
        ]}
        onDragOver$={(event) => {
          event.preventDefault();
          isDragging.value = true;
        }}
        onDragLeave$={() => {
          isDragging.value = false;
        }}
        onDrop$={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          class="hidden"
          onChange$={handleInputChange}
        />
        <div class="flex items-center justify-between gap-3">
          <div>
            <p class="text-xs font-semibold text-muted-foreground">Drop assets for Portal ingest</p>
            <p class="text-[10px] text-muted-foreground/70">
              Stored in browser cache, announced on bus, pulled into 24h analysis cache.
            </p>
          </div>
          <button
            class="text-[10px] px-2 py-1 rounded glass-chip glass-chip-accent-cyan uppercase tracking-widest"
            onClick$={() => inputRef.value?.click()}
          >
            Select
          </button>
        </div>
      </div>

      {error.value && (
        <div class="text-[10px] text-red-400">
          {error.value}
        </div>
      )}

      <div class="flex items-center justify-between text-[10px] text-muted-foreground">
        <span>staged assets: {assets.value.length}</span>
        {assets.value.length > 0 && (
          <span>latest: {assets.value[assets.value.length - 1].name}</span>
        )}
      </div>

      {assets.value.length > 0 && (
        <div class="space-y-1">
          {assets.value.slice(-3).map((asset) => (
            <div
              key={asset.id}
              class="flex items-center justify-between rounded-md border border-border/40 px-2 py-1 text-[10px]"
            >
              <span class="truncate">{asset.name}</span>
              <span class="text-muted-foreground">{formatBytes(asset.byte_size)} · {asset.status}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
});

export default PortalIngestDropzone;
