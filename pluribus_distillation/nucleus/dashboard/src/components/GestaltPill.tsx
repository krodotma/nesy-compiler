import { component$, useComputed$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';
import GestaltOverlay from './GestaltOverlay';

export interface GestaltPillProps {
  mood?: string | null;
  entropy?: number | null;
}

function fmtHHMM(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return '--:--';
  const hh = String(d.getUTCHours()).padStart(2, '0');
  const mm = String(d.getUTCMinutes()).padStart(2, '0');
  return `${hh}:${mm}`;
}

function moodDotClass(mood: string | null | undefined): string {
  switch ((mood || '').toLowerCase()) {
    case 'focused':
      return 'bg-cyan-400';
    case 'anxious':
      return 'bg-amber-300';
    case 'chaotic':
    case 'hyper':
      return 'bg-fuchsia-400';
    case 'dormant':
      return 'bg-gray-400';
    case 'calm':
    default:
      return 'bg-green-400';
  }
}

export const GestaltPill = component$<GestaltPillProps>(({ mood = null, entropy = null }) => {
  const open = useSignal(false);
  const state = useStore<{
    generation: number | null;
    lineageId: string | null;
    dagId: string | null;
    loaded: boolean;
  }>({
    generation: null,
    lineageId: null,
    dagId: null,
    loaded: false,
  });

  const buildTimeIso = __BUILD_TIME__;
  const buildCommit = __BUILD_COMMIT__;

  const buildHHMM = useComputed$(() => fmtHHMM(buildTimeIso));

  const toggle = $(() => {
    open.value = !open.value;
  });

  const close = $(() => {
    open.value = false;
  });

  const loadLineage = $(async () => {
    try {
      const res = await fetch('/api/fs/.pluribus/lineage.json', { cache: 'no-store' });
      const data = await res.json().catch(() => ({}));
      state.generation = typeof data?.generation === 'number' ? data.generation : null;
      state.lineageId = typeof data?.lineage_id === 'string' ? data.lineage_id : null;
      state.dagId = typeof data?.dag_id === 'string' ? data.dag_id : null;
      state.loaded = true;
    } catch {
      state.loaded = true;
    }
  });

  useVisibleTask$(({ cleanup }) => {
    if (!__E2E__) loadLineage();

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (!open.value) return;
      e.preventDefault();
      open.value = false;
    };
    // Capture phase so we can close even if nested widgets stop propagation.
    window.addEventListener('keydown', onKeyDown, true);
    cleanup(() => window.removeEventListener('keydown', onKeyDown, true));
  });

  const dot = useComputed$(() => moodDotClass(mood));
  const gen = useComputed$(() => (typeof state.generation === 'number' ? `GEN:${state.generation}` : 'GEN:—'));

  const ariaLabel = useComputed$(() => {
    const moodLabel = (mood || 'unknown').toString();
    return `Gestalt snapshot, ${gen.value}, mood ${moodLabel}. Click to expand.`;
  });

  return (
    <>
      <button
        type="button"
        class="flex items-center gap-2 text-xs px-3 py-1.5 rounded-full border border-border bg-muted/30 hover:bg-muted/50 transition-colors"
        onClick$={toggle}
        data-testid="gestalt-pill"
        aria-label={ariaLabel.value}
      >
        <span class="text-sm">📄</span>
        <span class="mono">{gen.value}</span>
        <span class="text-muted-foreground">|</span>
        <span class="mono">{buildHHMM.value}</span>
        <span class="text-muted-foreground">|</span>
        <span class={`inline-block w-2 h-2 rounded-full ${dot.value}`} />
        <span class="mono">{(mood || 'calm').toString()}</span>
      </button>

      <GestaltOverlay
        open={open.value}
        onClose$={close}
        buildTimeIso={buildTimeIso}
        buildCommit={buildCommit}
        generation={state.generation}
        lineageId={state.lineageId}
        dagId={state.dagId}
        mood={mood}
        entropy={entropy}
      />
    </>
  );
});

export default GestaltPill;
