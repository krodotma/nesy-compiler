import { component$, type QRL, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

interface DeckItem {
  id: string;
  title: string;
  subtitle?: string;
  icon?: string;
  colorClass?: string;
  size?: 'sm' | 'md' | 'lg';
  onOpen$?: QRL<() => void>;
  body?: string;
}

interface MetafizzyDeckProps {
  items: DeckItem[];
  title?: string;
  subtitle?: string;
}

function sizeClass(size: DeckItem['size']): string {
  switch (size) {
    case 'sm':
      return 'w-[260px] h-[180px]';
    case 'lg':
      return 'w-[520px] h-[380px]';
    case 'md':
    default:
      return 'w-[360px] h-[240px]';
  }
}

export const MetafizzyDeck = component$<MetafizzyDeckProps>((props) => {
  const containerRef = useSignal<HTMLDivElement>();
  const status = useSignal<'idle' | 'ready' | 'unavailable'>('idle');
  const error = useSignal<string | null>(null);

  const relayout = $(() => {
    try {
      const el = containerRef.value;
      const packery = (el as any)?.__packery as any;
      packery?.layout?.();
    } catch {
      // ignore
    }
  });

  useVisibleTask$(async ({ cleanup }) => {
    const container = containerRef.value;
    if (!container) return;

    // Metafizzy libs are browser-only (depend on window). Load lazily.
    let Packery: any = null;
    let Draggabilly: any = null;
    try {
      const modPackery: any = await import('packery');
      const modDraggabilly: any = await import('draggabilly');
      Packery = modPackery?.default || modPackery;
      Draggabilly = modDraggabilly?.default || modDraggabilly;
      if (!Packery || !Draggabilly) throw new Error('missing packery/draggabilly exports');
    } catch (e) {
      status.value = 'unavailable';
      error.value = String(e);
      return;
    }

    const packery = new Packery(container, {
      itemSelector: '.deck-item',
      gutter: 12,
      percentPosition: false,
      transitionDuration: '0.25s',
    });

    (container as any).__packery = packery;

    const draggies: any[] = [];
    const bind = () => {
      try {
        const elems: HTMLElement[] = Array.from(container.querySelectorAll('.deck-item'));
        for (const elem of elems) {
          if ((elem as any).__draggie) continue;
          const draggie = new Draggabilly(elem, { handle: '.deck-handle' });
          (elem as any).__draggie = draggie;
          packery.bindDraggabillyEvents(draggie);
          draggies.push(draggie);
        }
      } catch {
        // ignore
      }
    };

    bind();
    packery.layout();
    status.value = 'ready';

    const observer = new MutationObserver(() => {
      bind();
      packery.reloadItems();
      packery.layout();
    });
    observer.observe(container, { childList: true, subtree: false });

    const onResize = () => {
      packery.layout();
    };
    window.addEventListener('resize', onResize);

    cleanup(() => {
      window.removeEventListener('resize', onResize);
      observer.disconnect();
      for (const d of draggies) {
        try {
          d?.destroy?.();
        } catch {
          // ignore
        }
      }
      try {
        packery?.destroy?.();
      } catch {
        // ignore
      }
      try {
        delete (container as any).__packery;
      } catch {
        // ignore
      }
    });
  });

  return (
    <div class="space-y-3">
      {/* Step 97: glass-surface-elevated for deck toolbar */}
      <div class="glass-surface-elevated rounded-xl p-3 flex items-center justify-between gap-3">
        <div>
          <div class="text-sm font-semibold">{props.title || 'Studio Deck'}</div>
          {props.subtitle && <div class="text-xs text-muted-foreground">{props.subtitle}</div>}
        </div>
        <div class="flex items-center gap-2 text-xs text-muted-foreground">
          <span class="mono">metafizzy</span>
          {/* Step 97: glass-chip for status badge */}
          <span class={`glass-chip ${status.value === 'ready' ? 'glass-chip-accent-emerald' : status.value === 'unavailable' ? 'glass-chip-accent-magenta' : ''}`}>
            {status.value}
          </span>
          {/* Step 97: glass-interactive for relayout button */}
          <button
            onClick$={relayout}
            class="glass-interactive px-2 py-1 rounded-lg text-xs"
            title="Force a Packery relayout"
          >
            Relayout
          </button>
        </div>
      </div>

      {status.value === 'unavailable' && error.value && (
        <div class="glass-chip glass-chip-accent-magenta text-xs p-2 w-full">
          Deck unavailable: {error.value}
        </div>
      )}

      {/* Step 98: glass-depth-layers for canvas area */}
      <div
        ref={containerRef}
        class="glass-surface-subtle relative w-full min-h-[520px] rounded-xl p-3 overflow-hidden"
      >
        {/* Step 98: glass-surface for deck items with glass-interactive handles */}
        {props.items.map((item) => (
          <div
            key={item.id}
            class={`deck-item absolute ${sizeClass(item.size)} glass-surface-elevated rounded-xl shadow-xl glass-transition-standard ${
              item.colorClass || ''
            }`}
            style={{ transform: 'translate3d(0,0,0)' }}
          >
            <div class="deck-handle glass-interactive cursor-move select-none p-3 border-b border-[var(--glass-border)] flex items-center justify-between rounded-t-xl">
              <div class="flex items-center gap-2">
                <span class="text-lg">{item.icon || '▣'}</span>
                <div>
                  <div class="text-sm font-semibold">{item.title}</div>
                  {item.subtitle && <div class="text-[11px] text-muted-foreground">{item.subtitle}</div>}
                </div>
              </div>
              {item.onOpen$ && (
                <button
                  onClick$={item.onOpen$}
                  class="glass-chip glass-chip-accent-cyan text-xs px-2 py-1 hover:scale-105 transition-transform"
                >
                  Open
                </button>
              )}
            </div>
            <div class="p-3 text-xs text-muted-foreground space-y-2">
              {item.body && <div class="leading-relaxed">{item.body}</div>}
              {!item.body && (
                <div class="opacity-70">
                  Drag me. Pin me. Treat UI as a capability surface.
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
});

