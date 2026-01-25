import { component$, $, useSignal, useVisibleTask$, useStore, useComputed$, noSerialize, type NoSerialize } from '@builder.io/qwik';
import { NeonTitle, NeonBadge, NeonSectionHeader } from './ui/NeonTitle';

// M3 Components - GestaltOverlay
import '@material/web/elevation/elevation.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/slider/slider.js';

// Real registry item from art_dept
interface RegistryItem {
  id: string;
  type: string;
  name: string;
  code_path: string;
  tags: string[];
  provenance?: string;
  design_token?: string;
}

// Current scene from art director
interface ArtScene {
  id: string;
  name: string;
  code_path?: string;
  tags?: string[];
  glsl?: string;
}

interface ArtState {
  mood: string;
  entropy: number;
  velocity: number;
  anxiety: number;
}

export interface GestaltOverlayProps {
  open: boolean;
  onClose$?: () => void;
  buildTimeIso: string;
  buildCommit: string;
  generation?: number | null;
  lineageId?: string | null;
  dagId?: string | null;
  mood?: string | null;
  entropy?: number | null;
}

function fmtShortCommit(commit: string): string {
  const c = (commit || '').trim();
  if (!c) return 'dev';
  return c.length > 12 ? c.slice(0, 12) : c;
}

const MOOD_OPTIONS = ['calm', 'focused', 'hyper', 'anxious', 'chaotic', 'dormant'] as const;

export const GestaltOverlay = component$<GestaltOverlayProps>(
  ({ open, onClose$, buildTimeIso, buildCommit, generation, lineageId, dagId, mood: propMood, entropy: propEntropy }) => {
    const handleClose = $(() => onClose$?.());

    // Art registry state
    const registry = useSignal<RegistryItem[]>([]);
    const currentScene = useSignal<ArtScene | null>(null);
    const artState = useSignal<ArtState | null>(null);
    const isLoading = useSignal(false);
    const selectedMood = useSignal<string>(propMood || 'calm');
    const statusMessage = useSignal<string>('');
    const searchQuery = useSignal<string>('');
    const visibleCount = useSignal<number>(12);

    const filteredRegistry = useComputed$(() => {
        const query = searchQuery.value.toLowerCase();
        return registry.value.filter(item => 
            item.name.toLowerCase().includes(query) || 
            item.tags.some(t => t.toLowerCase().includes(query))
        );
    });

    const visibleRegistry = useComputed$(() => {
        return filteredRegistry.value.slice(0, visibleCount.value);
    });

    const safeEntropy = typeof propEntropy === 'number' ? Math.max(0, Math.min(1, propEntropy)) : null;

    // ... (keep initialize tasks)
    
    const loadMore = $(() => {
        visibleCount.value += 12;
    });

    // Load registry when overlay opens
    useVisibleTask$(({ track, cleanup }) => {
      track(() => open);
      if (!open) return;

      // Load registry
      fetch('/api/fs/nucleus/art_dept/artifacts/registry.ndjson')
        .then(r => r.ok ? r.text() : '')
        .then(text => {
          const items: RegistryItem[] = [];
          const lines = text.split('\n').filter(Boolean);
          for (const line of lines) {
            try {
              const item = JSON.parse(line);
              if (item.type === 'shader') items.push(item);
            } catch {}
          }
          registry.value = items;
        })
        .catch(() => {});

      // Get current scene from window state
      try {
        const last = (window as any).__PLURIBUS_LAST_ART_SCENE__;
        if (last?.data?.scene) {
          currentScene.value = last.data.scene;
        }
      } catch {}

      // Listen for art events
      const onArt = (ev: Event) => {
        const detail = (ev as CustomEvent).detail;
        if (detail?.topic === 'art.scene.change' && detail?.data?.scene) {
          currentScene.value = detail.data.scene;
          statusMessage.value = `Scene changed: ${detail.data.scene.name || detail.data.scene.id}`;
          setTimeout(() => statusMessage.value = '', 3000);
        }
      };
      window.addEventListener('pluribus:art', onArt);
      cleanup(() => window.removeEventListener('pluribus:art', onArt));
    });

    // Request new scene from art director
    const requestScene = $(async (reason: string) => {
      isLoading.value = true;
      statusMessage.value = 'Requesting new scene...';
      try {
        const seed = typeof crypto !== 'undefined' && crypto.getRandomValues
          ? crypto.getRandomValues(new Uint32Array(1))[0]
          : Math.floor(Math.random() * 2 ** 32);

        await fetch('/api/emit', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            topic: 'art.scene.request',
            kind: 'request',
            level: 'info',
            actor: 'gestalt-overlay',
            data: {
              source: 'GestaltOverlay',
              seed,
              reason,
              mood: selectedMood.value,
              at: new Date().toISOString()
            }
          })
        });
        statusMessage.value = 'Scene request sent. Waiting for art director...';
      } catch (e) {
        statusMessage.value = 'Failed to request scene';
      } finally {
        isLoading.value = false;
      }
    });

    // Apply a specific shader from registry
    const applyShader = $(async (item: RegistryItem) => {
      isLoading.value = true;
      statusMessage.value = `Loading ${item.name}...`;
      try {
        // Fetch the GLSL code
        const res = await fetch(`/api/fs/${item.code_path}`);
        if (!res.ok) throw new Error('Failed to load shader');
        const glsl = await res.text();

        // Emit scene change event directly (bypass director for testing)
        const scene = { ...item, glsl };
        (window as any).__PLURIBUS_LAST_ART_SCENE__ = { topic: 'art.scene.change', data: { scene } };
        window.dispatchEvent(new CustomEvent('pluribus:art', {
          detail: { topic: 'art.scene.change', data: { scene, tokens: {} } }
        }));

        currentScene.value = scene;
        statusMessage.value = `Applied: ${item.name}`;
      } catch (e) {
        statusMessage.value = `Error: ${String(e)}`;
      } finally {
        isLoading.value = false;
      }
    });

    // Cycle to random shader
    const cycleRandom = $(async () => {
      if (registry.value.length === 0) {
        statusMessage.value = 'No shaders in registry';
        return;
      }
      const items = registry.value.filter(i => i.type === 'shader');
      const random = items[Math.floor(Math.random() * items.length)];
      await applyShader(random);
    });

    return (
      <div
        class={`fixed inset-0 z-[55] ${open ? 'glass-animate-enter' : 'hidden'}`}
        data-testid="gestalt-overlay"
        aria-modal="true"
        role="dialog"
      >
        <div class="absolute inset-0 glass-surface-overlay" onClick$={handleClose} />
        <div class="absolute right-4 top-4 left-4 md:left-auto md:w-[720px] rounded-xl glass-surface-elevated shadow-glass-elevated overflow-hidden glass-animate-enter" style={{ '--stagger': 1 } as any}>
          <div class="p-3 border-b border-glass-border-subtle flex items-center gap-2 glass-surface-subtle">
            <span class="text-lg">🎨</span>
            <NeonTitle level="div" color="magenta" size="lg" animation="flicker">Gestalt Art Director</NeonTitle>
            <NeonBadge color="purple" glow class="ml-2">
              {registry.value.length} shaders
            </NeonBadge>
            <button
              type="button"
              class="ml-auto text-xs px-3 py-1.5 rounded bg-primary text-primary-foreground hover:bg-primary/90"
              onClick$={handleClose}
            >
              Close
            </button>
          </div>

          <div class="p-3 text-xs space-y-3 max-h-[80vh] overflow-y-auto">
            {/* Status Message */}
            {statusMessage.value && (
              <div class="rounded border border-primary/30 bg-primary/10 px-3 py-2 text-primary">
                {statusMessage.value}
              </div>
            )}

            {/* System Evolution */}
            <div class="rounded-lg glass-surface glass-animate-enter" style={{ '--stagger': 2 } as any}>
              <div class="px-3 py-2 border-b border-glass-border-subtle">
                <NeonTitle level="div" color="cyan" size="sm">System Evolution</NeonTitle>
              </div>
              <div class="p-3 grid grid-cols-2 gap-x-3 gap-y-2 text-muted-foreground">
                <div class="font-mono">BUILD</div>
                <div class="font-mono text-foreground">{buildTimeIso}</div>
                <div class="font-mono">COMMIT</div>
                <div class="font-mono text-foreground">{fmtShortCommit(buildCommit)}</div>
                <div class="font-mono">GEN</div>
                <div class="font-mono text-foreground">{typeof generation === 'number' ? generation : '—'}</div>
                <div class="font-mono">LINEAGE</div>
                <div class="font-mono text-foreground truncate">{lineageId || '—'}</div>
                <div class="font-mono">DAG</div>
                <div class="font-mono text-foreground truncate">{dagId || '—'}</div>
              </div>
            </div>

            {/* Current Scene */}
            <div class="rounded-lg glass-surface glass-animate-enter" style={{ '--stagger': 3 } as any}>
              <div class="px-3 py-2 border-b border-glass-border-subtle">
                <NeonTitle level="div" color="amber" size="sm">Current Scene</NeonTitle>
              </div>
              <div class="p-3 space-y-2">
                {currentScene.value ? (
                  <>
                    <div class="flex items-center gap-2">
                      <span class="font-mono text-foreground">{currentScene.value.name || currentScene.value.id}</span>
                      {currentScene.value.tags && (
                        <div class="flex gap-1">
                          {currentScene.value.tags.slice(0, 3).map(tag => (
                            <span key={tag} class="px-1.5 py-0.5 rounded bg-accent/20 text-accent text-[10px]">{tag}</span>
                          ))}
                        </div>
                      )}
                    </div>
                    <div class="font-mono text-muted-foreground text-[10px]">{currentScene.value.code_path || '—'}</div>
                  </>
                ) : (
                  <div class="text-muted-foreground">No scene active</div>
                )}
              </div>
            </div>

            {/* Art Controls */}
            <div class="rounded-lg glass-surface glass-animate-enter" style={{ '--stagger': 4 } as any}>
              <div class="px-3 py-2 border-b border-glass-border-subtle">
                <NeonTitle level="div" color="purple" size="sm">Art Controls</NeonTitle>
              </div>
              <div class="p-3 space-y-3">
                {/* Mood Selector */}
                <div class="flex items-center gap-2">
                  <span class="font-mono text-muted-foreground w-16">MOOD</span>
                  <div class="flex gap-1 flex-wrap">
                    {MOOD_OPTIONS.map(m => (
                      <button
                        key={m}
                        type="button"
                        class={`px-2 py-1 rounded text-[10px] transition-colors ${
                          selectedMood.value === m
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-secondary/50 text-secondary-foreground hover:bg-secondary'
                        }`}
                        onClick$={async () => {
                          selectedMood.value = m;
                          // Dispatch mood change event for immediate visual feedback
                          if (typeof window !== 'undefined') {
                            window.dispatchEvent(new CustomEvent('pluribus:mood:change', {
                              detail: { mood: m, timestamp: Date.now() }
                            }));
                            // Also persist to sessionStorage
                            try { sessionStorage.setItem('gestalt_mood', m); } catch {}
                          }
                        }}
                      >
                        {m}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Entropy Display */}
                <div class="flex items-center gap-2">
                  <span class="font-mono text-muted-foreground w-16">ENTROPY</span>
                  <div class="flex-1 h-2 bg-muted rounded overflow-hidden">
                    <div
                      class="h-full bg-gradient-to-r from-primary to-accent transition-all"
                      style={{ width: `${(safeEntropy ?? 0.5) * 100}%` }}
                    />
                  </div>
                  <span class="font-mono text-foreground w-12 text-right">
                    {safeEntropy === null ? '—' : safeEntropy.toFixed(2)}
                  </span>
                </div>

                {/* Action Buttons */}
                <div class="flex gap-2 flex-wrap">
                  <button
                    type="button"
                    class={`px-3 py-1.5 rounded bg-primary text-primary-foreground hover:bg-primary/90 text-xs ${isLoading.value ? 'opacity-50 cursor-not-allowed' : ''}`}
                    onClick$={() => requestScene('user_chroma_toggle')}
                    disabled={isLoading.value}
                  >
                    {isLoading.value ? 'Requesting...' : 'Request New Scene'}
                  </button>
                  <button
                    type="button"
                    class={`px-3 py-1.5 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80 text-xs ${isLoading.value ? 'opacity-50 cursor-not-allowed' : ''}`}
                    onClick$={cycleRandom}
                    disabled={isLoading.value}
                  >
                    Random Shader
                  </button>
                </div>
              </div>
            </div>

            {/* Shader Gallery */}
            <div class="rounded-lg glass-surface glass-animate-enter" style={{ '--stagger': 5 } as any}>
              <div class="px-3 py-2 border-b border-glass-border-subtle flex items-center justify-between">
                <div class="flex items-center gap-2">
                  <NeonTitle level="span" color="emerald" size="sm">Shader Gallery</NeonTitle>
                  <NeonBadge color="emerald" glow>{filteredRegistry.value.length}</NeonBadge>
                </div>
                <input 
                    type="text" 
                    placeholder="Search shaders..." 
                    class="bg-background border border-[var(--glass-border)] rounded px-2 py-0.5 text-[10px] w-32 focus:outline-none focus:ring-1 focus:ring-primary"
                    onInput$={(e) => searchQuery.value = (e.target as HTMLInputElement).value}
                />
              </div>
              <div class="p-3 max-h-80 overflow-y-auto custom-scrollbar" onScroll$={(e) => {
                  const target = e.target as HTMLElement;
                  if (target.scrollHeight - target.scrollTop <= target.clientHeight + 10) {
                      loadMore();
                  }
              }}>
                <div class="columns-2 md:columns-3 gap-3 space-y-3">
                  {visibleRegistry.value.map((item) => (
                    <button
                      key={item.id}
                      type="button"
                      class={`w-full text-left rounded-xl p-3 glass-transition-all break-inside-avoid glass-surface glass-hover-scale ${
                        currentScene.value?.id === item.id
                          ? 'ring-1 ring-glass-accent/50 shadow-glass-glow'
                          : 'glass-hover-glow'
                      }`}
                      onClick$={() => applyShader(item)}
                    >
                      <div class="flex items-center gap-2 mb-1.5">
                        <span class="text-[10px] opacity-50">#</span>
                        <div class="font-bold truncate text-[11px] uppercase tracking-wider text-foreground">{item.name}</div>
                      </div>
                      <div class="flex flex-wrap gap-1">
                        {item.tags?.slice(0, 3).map(tag => (
                          <span key={tag} class="text-[8px] px-1.5 py-0.5 rounded-full bg-[var(--glass-bg-card)] border border-[var(--glass-border)] text-[var(--glass-text-tertiary)]">{tag}</span>
                        ))}
                      </div>
                    </button>
                  ))}
                </div>
                {visibleCount.value < filteredRegistry.value.length && (
                    <div class="mt-3 text-center">
                        <button 
                            type="button" 
                            class="text-[10px] text-primary hover:underline"
                            onClick$={loadMore}
                        >
                            Load More (+{filteredRegistry.value.length - visibleCount.value})
                        </button>
                    </div>
                )}
              </div>
            </div>

            <div class="text-[11px] text-muted-foreground">
              Tip: press <span class="font-mono">Esc</span> to close. Art Director continuously studies kroma.live for inspiration.
            </div>
          </div>
        </div>
      </div>
    );
  });

export default GestaltOverlay;
