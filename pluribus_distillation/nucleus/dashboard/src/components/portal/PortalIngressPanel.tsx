import { component$, useComputed$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';
import type { BusEvent } from '../../lib/state/types';
import {
  PortalIngressSelector,
  type IngressFragment,
  type IngressMode,
} from './PortalIngressSelector';
import { PortalIngestDropzone } from './PortalIngestDropzone';
import {
  PortalDestinationSelector,
  type IngressSelection,
} from './PortalDestinationSelector';
import { NeonTitle, NeonBadge, NeonSectionHeader } from '../ui/NeonTitle';

interface PortalIngressPanelProps {
  events: BusEvent[];
}

const PORTAL_TOPIC_PREFIXES = [
  'portal.ingress',
  'portal.inception',
  'portal.ingest',
  'portal.a2ui',
  'strp.lead.action.ingest',
];

const MODE_BADGE: Record<IngressMode, { label: string; color: 'amber' | 'purple' | 'cyan' }> = {
  AM: { label: 'Actualized', color: 'amber' },
  SM: { label: 'Shadow', color: 'purple' },
  AUTO: { label: 'Auto', color: 'cyan' },
};

function pickString(data: Record<string, unknown>, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = data[key];
    if (typeof value === 'string' && value.trim().length > 0) {
      return value.trim();
    }
  }
  return undefined;
}

function pickNumber(data: Record<string, unknown>, keys: string[]): number | undefined {
  for (const key of keys) {
    const value = data[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return undefined;
}

function inferSourceType(
  data: Record<string, unknown>,
  sourceUri?: string
): IngressFragment['source_type'] {
  const explicit = data.source_type;
  if (
    explicit === 'url' ||
    explicit === 'file' ||
    explicit === 'text' ||
    explicit === 'arxiv' ||
    explicit === 'github'
  ) {
    return explicit;
  }
  const normalized = (sourceUri || '').toLowerCase();
  if (normalized.startsWith('http')) return 'url';
  if (normalized.includes('arxiv')) return 'arxiv';
  if (normalized.includes('github.com')) return 'github';
  if (normalized) return 'file';
  return 'text';
}

function extractPreview(event?: BusEvent): string {
  if (!event) return 'Awaiting portal ingress signal.';
  const data = (event.data || {}) as Record<string, unknown>;
  const fragment = (data.fragment || {}) as Record<string, unknown>;
  const preview =
    pickString(data, ['content_preview', 'content', 'text', 'summary', 'message', 'prompt', 'name']) ||
    pickString(fragment, ['content', 'content_preview']) ||
    event.topic;
  return preview.length > 140 ? `${preview.slice(0, 137)}...` : preview;
}

function findLatestPortalEvent(events: BusEvent[]): BusEvent | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (PORTAL_TOPIC_PREFIXES.some((prefix) => event.topic.startsWith(prefix))) {
      return event;
    }
  }
  return undefined;
}

function buildFragment(event?: BusEvent): IngressFragment {
  const data = (event?.data || {}) as Record<string, unknown>;
  const fragment = (data.fragment || {}) as Record<string, unknown>;
  const sourceUri =
    pickString(data, ['source_uri', 'source_ref', 'url']) ||
    pickString(fragment, ['source_uri']);
  const preview = extractPreview(event);
  const byteSize =
    pickNumber(data, ['byte_size', 'length_bytes', 'size']) ||
    pickNumber(fragment, ['length_bytes', 'byte_size']) ||
    preview.length;
  const fragmentId =
    pickString(data, ['fragment_id', 'ingress_id', 'req_id']) ||
    pickString(fragment, ['id']) ||
    event?.id ||
    'fragment-unknown';
  const assetId =
    pickString(data, ['asset_id']) ||
    pickString(fragment, ['asset_id']);
  const createdIso = event?.iso || new Date().toISOString();

  return {
    id: fragmentId,
    content_preview: preview,
    source_type: inferSourceType(data, sourceUri),
    source_uri: sourceUri,
    texture_density: pickNumber(data, ['texture_density']),
    byte_size: byteSize,
    created_iso: createdIso,
    asset_id: assetId,
  };
}

export const PortalIngressPanel = component$<PortalIngressPanelProps>(({ events }) => {
  const lastEvent = useComputed$(() => findLatestPortalEvent(events));
  const stagedFragment = useSignal<IngressFragment | null>(null);
  const selectedMode = useSignal<IngressMode | null>(null);
  const selectionToast = useSignal<string | null>(null);
  const toastTimeout = useSignal<number | null>(null);
  const fragment = useComputed$(() => stagedFragment.value ?? buildFragment(lastEvent.value));
  const contentTitle = useComputed$(
    () => fragment.value.source_uri || fragment.value.content_preview
  );
  const handleStage = $((next: IngressFragment) => {
    stagedFragment.value = next;
    selectedMode.value = null;
    selectionToast.value = null;
    if (toastTimeout.value && typeof window !== 'undefined') {
      window.clearTimeout(toastTimeout.value);
      toastTimeout.value = null;
    }
  });
  const handleModeSelected = $((mode: IngressMode) => {
    selectedMode.value = mode;
  });
  const handleDestinationSelect = $(async (selection: IngressSelection) => {
    const event = {
      topic: 'portal.ingress.select',
      kind: 'action',
      level: 'info' as const,
      actor: 'portal-destination-selector',
      data: {
        selection,
        fragment: fragment.value,
        mode: selectedMode.value,
        selected_at: new Date().toISOString(),
      },
    };

    await fetch('/api/emit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(event),
    }).catch(() => {/* Best effort */});

    if (toastTimeout.value && typeof window !== 'undefined') {
      window.clearTimeout(toastTimeout.value);
    }
    selectionToast.value = `Destination queued: ${selection.destinationId}`;
    if (typeof window !== 'undefined') {
      toastTimeout.value = window.setTimeout(() => {
        selectionToast.value = null;
        toastTimeout.value = null;
      }, 3000);
    }
  });
  useVisibleTask$(({ track }) => {
    track(() => fragment.value.id);
    selectedMode.value = null;
    selectionToast.value = null;
    if (toastTimeout.value && typeof window !== 'undefined') {
      window.clearTimeout(toastTimeout.value);
      toastTimeout.value = null;
    }
  });

  return (
    <div class="space-y-3">
      <div class="flex items-center justify-between">
        <div>
          <NeonTitle level="h3" color="magenta" size="sm" animation="flicker">
            Portal Ingest
          </NeonTitle>
          <p class="text-[10px] text-muted-foreground mt-1">
            {lastEvent.value
              ? `Last signal: ${lastEvent.value.topic}`
              : 'No portal ingest events yet.'}
          </p>
        </div>
        <NeonBadge color="cyan" glow>ingest</NeonBadge>
      </div>

      <PortalIngestDropzone onStage$={handleStage} />
      <PortalIngressSelector fragment={fragment.value} onModeSelected$={handleModeSelected} />
      <NeonSectionHeader title="Destination" color="cyan" size="xs" />
      {selectedMode.value ? (
        <div class="flex items-center gap-2 -mt-2 mb-2 text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          <span>Mode</span>
          <NeonBadge color={MODE_BADGE[selectedMode.value].color} glow>
            {MODE_BADGE[selectedMode.value].label}
          </NeonBadge>
        </div>
      ) : (
        <div class="text-[10px] text-muted-foreground -mt-2 mb-2">
          No ingress mode selected.
        </div>
      )}
      {selectedMode.value ? (
        <PortalDestinationSelector
          contentId={fragment.value.id}
          contentTitle={contentTitle.value}
          onSelect$={handleDestinationSelect}
        />
      ) : (
        <div class="rounded-lg border border-border/40 bg-muted/10 p-3 text-xs text-muted-foreground">
          Select an ingress mode to unlock destination routing.
        </div>
      )}
      {selectionToast.value && (
        <div class="rounded-md border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200">
          {selectionToast.value}
        </div>
      )}
    </div>
  );
});

export default PortalIngressPanel;
