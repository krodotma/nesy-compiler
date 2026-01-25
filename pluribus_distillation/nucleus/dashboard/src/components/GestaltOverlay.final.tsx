import { component$, $, useSignal, useVisibleTask$, useStore, noSerialize, type NoSerialize } from '@builder.io/qwik';

// Simple interface for artwork objects
interface Artwork {
  id: string;
  title: string;
  style: string;
  tags: string[];
  createdAt: string;
  previewUrl?: string;
}

// Simple ArtDirector class for demonstration
class ArtDirector {
  async getRecentArtworks(): Promise<Artwork[]> {
    // In a real implementation, this would fetch from the art department
    return [
      {
        id: '1',
        title: 'Quantum Entanglement Visualization',
        style: 'Abstract Digital',
        tags: ['quantum', 'visualization', 'tech'],
        createdAt: new Date().toISOString()
      },
      {
        id: '2',
        title: 'Neural Pathways',
        style: 'Cyberpunk',
        tags: ['neural', 'network', 'ai'],
        createdAt: new Date(Date.now() - 3600000).toISOString()
      }
    ];
  }

  async generateArtwork(options: { mood: string; entropy: number; context: string; targetElement: string }): Promise<Artwork> {
    // In a real implementation, this would generate actual artwork
    return {
      id: Math.random().toString(36).substring(2, 9),
      title: `Generated Art - ${options.mood} Mood`,
      style: 'Dynamic Abstract',
      tags: [options.mood, 'generated', options.context],
      createdAt: new Date().toISOString()
    };
  }

  async getRandomArtwork(options: { mood: string; entropy: number }): Promise<Artwork> {
    return {
      id: Math.random().toString(36).substring(2, 9),
      title: `Random ${options.mood} Art`,
      style: 'Chaotic Beauty',
      tags: [options.mood, 'random', 'dynamic'],
      createdAt: new Date().toISOString()
    };
  }

  async applyArtworkToUI(artwork: Artwork): Promise<void> {
    // In a real implementation, this would apply the artwork to the UI
    console.log('Applying artwork to UI:', artwork);
  }

  async previewArtwork(artwork: Artwork): Promise<void> {
    // In a real implementation, this would preview the artwork
    console.log('Previewing artwork:', artwork);
  }
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

export const GestaltOverlay = component$<GestaltOverlayProps>(
  ({ open, onClose$, buildTimeIso, buildCommit, generation, lineageId, dagId, mood, entropy }) => {
    const handleClose = $(() => onClose$?.());

    // Use useStore to hold non-serializable objects
    const artDirectorState = useStore({
      artDirector: noSerialize(new ArtDirector()) as NoSerialize<ArtDirector>,
    });

    const currentArtwork = useSignal<Artwork | null>(null);
    const artHistory = useSignal<Artwork[]>([]);
    const isGenerating = useSignal(false);

    const safeEntropy = typeof entropy === 'number' ? Math.max(0, Math.min(1, entropy)) : null;

    // Initialize art director when overlay opens
    useVisibleTask$(({ track }) => {
      track(() => open);
      if (open && artDirectorState.artDirector) {
        // Load recent artwork history
        artDirectorState.artDirector.getRecentArtworks().then(history => {
          artHistory.value = history;
          if (history.length > 0) {
            currentArtwork.value = history[0];
          }
        });
      }
    });

    const generateArt = $(async () => {
      if (!artDirectorState.artDirector) return;

      isGenerating.value = true;
      try {
        const newArtwork = await artDirectorState.artDirector.generateArtwork({
          mood: mood || 'neutral',
          entropy: safeEntropy || 0.5,
          context: 'gestalt-overlay',
          targetElement: 'dashboard'
        });
        currentArtwork.value = newArtwork;
        artHistory.value = [newArtwork, ...artHistory.value.slice(0, 9)]; // Keep last 10
      } catch (error) {
        console.error('Art generation failed:', error);
      } finally {
        isGenerating.value = false;
      }
    });

    const applyArtwork = $(async (artwork: Artwork) => {
      if (!artDirectorState.artDirector) return;

      try {
        await artDirectorState.artDirector.applyArtworkToUI(artwork);
        // Close overlay after applying
        handleClose();
      } catch (error) {
        console.error('Failed to apply artwork:', error);
      }
    });

    const randomizeArt = $(async () => {
      if (!artDirectorState.artDirector) return;

      isGenerating.value = true;
      try {
        const randomArtwork = await artDirectorState.artDirector.getRandomArtwork({
          mood: mood || 'neutral',
          entropy: safeEntropy || 0.5
        });
        currentArtwork.value = randomArtwork;
      } catch (error) {
        console.error('Random artwork selection failed:', error);
      } finally {
        isGenerating.value = false;
      }
    });

    return (
      <div
        class={`fixed inset-0 z-[55] ${open ? '' : 'hidden'}`}
        data-testid="gestalt-overlay"
        aria-modal="true"
        role="dialog"
      >
        <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick$={handleClose} />
        <div class="absolute right-4 top-4 left-4 md:left-auto md:w-[680px] rounded-xl border border-border bg-card/90 shadow-2xl overflow-hidden">
          <div class="p-3 border-b border-border flex items-center gap-2">
            <span class="text-lg">🎨</span>
            <div class="text-sm font-semibold">Gestalt Art Director</div>
            <button
              type="button"
              class="ml-auto text-xs px-3 py-1.5 rounded bg-primary text-primary-foreground hover:bg-primary/90"
              onClick$={handleClose}
            >
              Close
            </button>
          </div>

          <div class="p-3 text-xs space-y-3 max-h-[80vh] overflow-y-auto">
            <div class="rounded border border-border bg-muted/20">
              <div class="px-3 py-2 border-b border-border font-semibold">System Evolution</div>
              <div class="p-3 grid grid-cols-2 gap-x-3 gap-y-2 text-muted-foreground">
                <div class="mono">BUILD</div>
                <div class="mono text-foreground">{buildTimeIso}</div>
                <div class="mono">COMMIT</div>
                <div class="mono text-foreground">{fmtShortCommit(buildCommit)}</div>
                <div class="mono">GEN</div>
                <div class="mono text-foreground">{typeof generation === 'number' ? generation : '—'}</div>
                <div class="mono">LINEAGE</div>
                <div class="mono text-foreground truncate">{lineageId || '—'}</div>
                <div class="mono">DAG</div>
                <div class="mono text-foreground truncate">{dagId || '—'}</div>
              </div>
            </div>

            <div class="rounded border border-border bg-muted/20">
              <div class="px-3 py-2 border-b border-border font-semibold">Art Department</div>
              <div class="p-3 grid grid-cols-2 gap-x-3 gap-y-2 text-muted-foreground">
                <div class="mono">MOOD</div>
                <div class="mono text-foreground">{mood || '—'}</div>
                <div class="mono">ENTROPY</div>
                <div class="mono text-foreground">{safeEntropy === null ? '—' : safeEntropy.toFixed(2)}</div>
                <div class="mono">ARTWORKS</div>
                <div class="mono text-foreground">{artHistory.value.length}</div>
                <div class="mono">STATUS</div>
                <div class="mono text-foreground">{isGenerating.value ? 'Generating...' : 'Ready'}</div>
              </div>
            </div>

            <div class="rounded border border-border bg-muted/20">
              <div class="px-3 py-2 border-b border-border font-semibold">Art Controls</div>
              <div class="p-3 space-y-3">
                <div class="flex gap-2">
                  <button
                    type="button"
                    class={`px-3 py-1.5 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80 text-xs ${isGenerating.value ? 'opacity-50 cursor-not-allowed' : ''}`}
                    onClick$={generateArt}
                    disabled={isGenerating.value}
                  >
                    {isGenerating.value ? 'Generating...' : 'Generate New Art'}
                  </button>
                  <button
                    type="button"
                    class={`px-3 py-1.5 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80 text-xs ${isGenerating.value ? 'opacity-50 cursor-not-allowed' : ''}`}
                    onClick$={randomizeArt}
                    disabled={isGenerating.value}
                  >
                    Random Art
                  </button>
                </div>

                {currentArtwork.value && (
                  <div class="mt-3">
                    <div class="text-sm font-medium mb-2">Current Artwork</div>
                    <div class="border rounded p-2 bg-background">
                      <div class="text-xs mb-1">Title: {currentArtwork.value.title}</div>
                      <div class="text-xs mb-2">Style: {currentArtwork.value.style}</div>
                      <div class="text-xs mb-2">Tags: {currentArtwork.value.tags.join(', ')}</div>
                      <div class="flex gap-2 mt-2">
                        <button
                          type="button"
                          class="px-2 py-1 rounded bg-primary text-primary-foreground hover:bg-primary/90 text-xs"
                          onClick$={() => applyArtwork(currentArtwork.value!)}
                        >
                          Apply to UI
                        </button>
                        <button
                          type="button"
                          class="px-2 py-1 rounded bg-secondary text-secondary-foreground hover:bg-secondary/80 text-xs"
                          onClick$={() => {
                            // Preview the artwork
                            if (artDirectorState.artDirector) {
                              artDirectorState.artDirector.previewArtwork(currentArtwork.value!);
                            }
                          }}
                        >
                          Preview
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {artHistory.value.length > 0 && (
              <div class="rounded border border-border bg-muted/20">
                <div class="px-3 py-2 border-b border-border font-semibold">Recent Artworks</div>
                <div class="p-3 max-h-40 overflow-y-auto">
                  <div class="grid grid-cols-3 gap-2">
                    {artHistory.value.slice(0, 6).map((art) => (
                      <div 
                        key={art.id} 
                        class="border rounded p-1 text-xs cursor-pointer hover:bg-accent"
                        onClick$={() => currentArtwork.value = art}
                      >
                        <div class="truncate">{art.title || 'Untitled'}</div>
                        <div class="text-[10px] text-muted-foreground">{art.style || 'Mixed'}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            <div class="text-[11px] text-muted-foreground">
              Tip: press <span class="mono">Esc</span> to close. Art Director continuously studies kroma.live for inspiration.
            </div>
          </div>
        </div>
      </div>
    );
  }
);

export default GestaltOverlay;