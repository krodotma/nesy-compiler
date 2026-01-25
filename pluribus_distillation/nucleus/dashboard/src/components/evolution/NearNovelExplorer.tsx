import { component$, useSignal, useTask$, $ } from '@builder.io/qwik';

/**
 * NearNovelExplorer - E14: Edge-of-Chaos Exploration Seeds
 * 
 * Displays exploration seeds from near_but_novel.py:
 * - Goldilocks zone visualization (0.3-0.7 distance)
 * - Seed priority ranking
 * - Explore/Discard actions
 */

interface ExplorationSeed {
    seed_id: string;
    source_relation_id: string;
    seed_type: string;  // hgt_candidate, motif_merge, exploration_prompt
    description: string;
    distance: number;   // 0-1, Goldilocks = 0.3-0.7
    priority: number;
    status: string;     // pending, explored, discarded
    created_at: string;
}

export const NearNovelExplorer = component$(() => {
    const seeds = useSignal<ExplorationSeed[]>([]);
    const loading = useSignal(true);
    const error = useSignal<string | null>(null);
    const processingId = useSignal<string | null>(null);

    // Fetch seeds on mount
    useTask$(async () => {
        try {
            const res = await fetch('/api/evolution/seeds');
            if (res.ok) {
                const data = await res.json();
                seeds.value = data.seeds || [];
            } else {
                error.value = `Failed to fetch seeds: ${res.status}`;
            }
        } catch (e) {
            error.value = String(e);
        } finally {
            loading.value = false;
        }
    });

    const handleExplore = $(async (seedId: string) => {
        processingId.value = seedId;
        try {
            const res = await fetch(`/api/evolution/seeds/${seedId}/explore`, { method: 'POST' });
            if (res.ok) {
                // Update local state
                seeds.value = seeds.value.map(s =>
                    s.seed_id === seedId ? { ...s, status: 'explored' } : s
                );
            }
        } finally {
            processingId.value = null;
        }
    });

    const handleDiscard = $(async (seedId: string) => {
        processingId.value = seedId;
        try {
            const res = await fetch(`/api/evolution/seeds/${seedId}/discard`, { method: 'POST' });
            if (res.ok) {
                seeds.value = seeds.value.map(s =>
                    s.seed_id === seedId ? { ...s, status: 'discarded' } : s
                );
            }
        } finally {
            processingId.value = null;
        }
    });

    const getGoldilocksClass = (distance: number) => {
        if (distance < 0.3) return 'zone-stagnation';
        if (distance > 0.7) return 'zone-chaos';
        return 'zone-goldilocks';
    };

    const getZoneLabel = (distance: number) => {
        if (distance < 0.3) return 'Too Similar (Stagnation)';
        if (distance > 0.7) return 'Too Different (Chaos)';
        return 'Goldilocks Zone ✓';
    };

    const pendingSeeds = seeds.value.filter(s => s.status === 'pending');
    const exploredSeeds = seeds.value.filter(s => s.status === 'explored');
    const discardedSeeds = seeds.value.filter(s => s.status === 'discarded');

    return (
        <div class="near-novel-explorer">
            <div class="explorer-header">
                <h3>🔮 Near-But-Novel Explorer (E14)</h3>
                <p class="tagline">Edge-of-chaos exploration seeds</p>
            </div>

            {loading.value && <div class="loading">Loading exploration seeds...</div>}
            {error.value && <div class="error">{error.value}</div>}

            {/* Goldilocks Zone Legend */}
            <div class="goldilocks-legend">
                <div class="zone-indicator zone-stagnation">
                    <span class="zone-range">0.0 - 0.3</span>
                    <span class="zone-name">Stagnation</span>
                </div>
                <div class="zone-indicator zone-goldilocks">
                    <span class="zone-range">0.3 - 0.7</span>
                    <span class="zone-name">Goldilocks ✓</span>
                </div>
                <div class="zone-indicator zone-chaos">
                    <span class="zone-range">0.7 - 1.0</span>
                    <span class="zone-name">Chaos</span>
                </div>
            </div>

            {/* Pending Seeds */}
            <div class="seeds-section">
                <h4>Pending Seeds ({pendingSeeds.length})</h4>
                {pendingSeeds.length === 0 ? (
                    <p class="empty">No pending exploration seeds</p>
                ) : (
                    <div class="seeds-list">
                        {pendingSeeds.map((seed) => (
                            <div key={seed.seed_id} class={`seed-card ${getGoldilocksClass(seed.distance)}`}>
                                <div class="seed-header">
                                    <span class="seed-type">{seed.seed_type}</span>
                                    <span class="seed-priority">P{seed.priority}</span>
                                </div>
                                <div class="seed-description">{seed.description}</div>
                                <div class="seed-metrics">
                                    <div class="distance-bar">
                                        <div
                                            class="distance-marker"
                                            style={{ left: `${seed.distance * 100}%` }}
                                            title={`Distance: ${seed.distance.toFixed(2)}`}
                                        />
                                    </div>
                                    <span class="zone-label">{getZoneLabel(seed.distance)}</span>
                                </div>
                                <div class="seed-actions">
                                    <button
                                        class="btn-explore"
                                        onClick$={() => handleExplore(seed.seed_id)}
                                        disabled={processingId.value === seed.seed_id}
                                    >
                                        {processingId.value === seed.seed_id ? '...' : '🔍 Explore'}
                                    </button>
                                    <button
                                        class="btn-discard"
                                        onClick$={() => handleDiscard(seed.seed_id)}
                                        disabled={processingId.value === seed.seed_id}
                                    >
                                        ✕ Discard
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Explored Seeds */}
            {exploredSeeds.length > 0 && (
                <div class="seeds-section explored">
                    <h4>Explored ({exploredSeeds.length})</h4>
                    <div class="seeds-list compact">
                        {exploredSeeds.slice(0, 5).map((seed) => (
                            <div key={seed.seed_id} class="seed-card mini">
                                <span class="seed-type">{seed.seed_type}</span>
                                <span class="seed-desc">{seed.description.slice(0, 50)}...</span>
                                <span class="status-badge explored">✓ Explored</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <style>{`
        .near-novel-explorer {
          padding: 1rem;
          background: var(--color-surface);
          border-radius: 8px;
        }
        .explorer-header h3 {
          margin: 0;
          color: var(--color-text);
        }
        .tagline {
          color: var(--color-text-muted);
          font-size: 0.875rem;
          margin: 0.25rem 0 1rem;
        }
        .goldilocks-legend {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1rem;
          padding: 0.75rem;
          background: var(--color-surface-elevated);
          border-radius: 6px;
        }
        .zone-indicator {
          flex: 1;
          text-align: center;
          padding: 0.5rem;
          border-radius: 4px;
          font-size: 0.75rem;
        }
        .zone-stagnation {
          background: rgba(234, 179, 8, 0.2);
          border-left: 3px solid #eab308;
        }
        .zone-goldilocks {
          background: rgba(34, 197, 94, 0.2);
          border-left: 3px solid #22c55e;
        }
        .zone-chaos {
          background: rgba(239, 68, 68, 0.2);
          border-left: 3px solid #ef4444;
        }
        .zone-range {
          display: block;
          font-family: monospace;
          font-weight: bold;
        }
        .zone-name {
          display: block;
          color: var(--color-text-muted);
        }
        .seeds-section {
          margin-top: 1rem;
        }
        .seeds-section h4 {
          color: var(--color-text-muted);
          margin-bottom: 0.5rem;
        }
        .seeds-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        .seed-card {
          background: var(--color-surface-elevated);
          padding: 1rem;
          border-radius: 6px;
          border-left: 3px solid var(--color-border);
        }
        .seed-card.zone-goldilocks {
          border-left-color: #22c55e;
        }
        .seed-card.zone-stagnation {
          border-left-color: #eab308;
        }
        .seed-card.zone-chaos {
          border-left-color: #ef4444;
        }
        .seed-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.5rem;
        }
        .seed-type {
          font-size: 0.75rem;
          text-transform: uppercase;
          color: var(--color-primary);
        }
        .seed-priority {
          font-size: 0.75rem;
          background: var(--color-primary-muted);
          padding: 0.125rem 0.375rem;
          border-radius: 3px;
        }
        .seed-description {
          color: var(--color-text);
          margin-bottom: 0.75rem;
        }
        .seed-metrics {
          margin-bottom: 0.75rem;
        }
        .distance-bar {
          height: 8px;
          background: linear-gradient(90deg, #eab308 0%, #eab308 30%, #22c55e 30%, #22c55e 70%, #ef4444 70%, #ef4444 100%);
          border-radius: 4px;
          position: relative;
        }
        .distance-marker {
          position: absolute;
          top: -4px;
          width: 4px;
          height: 16px;
          background: white;
          border: 2px solid var(--color-text);
          border-radius: 2px;
          transform: translateX(-50%);
        }
        .zone-label {
          display: block;
          text-align: center;
          font-size: 0.7rem;
          color: var(--color-text-muted);
          margin-top: 0.25rem;
        }
        .seed-actions {
          display: flex;
          gap: 0.5rem;
        }
        .btn-explore, .btn-discard {
          padding: 0.375rem 0.75rem;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.8rem;
        }
        .btn-explore {
          background: var(--color-primary);
          color: white;
        }
        .btn-discard {
          background: transparent;
          border: 1px solid var(--color-border);
          color: var(--color-text-muted);
        }
        .btn-explore:disabled, .btn-discard:disabled {
          opacity: 0.5;
        }
        .seed-card.mini {
          display: flex;
          align-items: center;
          gap: 0.75rem;
          padding: 0.5rem;
        }
        .status-badge.explored {
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
          padding: 0.125rem 0.375rem;
          border-radius: 3px;
          font-size: 0.7rem;
        }
        .empty {
          color: var(--color-text-muted);
          font-style: italic;
        }
      `}</style>
        </div>
    );
});

export default NearNovelExplorer;
