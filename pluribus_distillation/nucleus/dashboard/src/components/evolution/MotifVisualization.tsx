import { component$, useSignal, useTask$, $ } from '@builder.io/qwik';

/**
 * MotifVisualization - Display ω-motifs from motif_extractor
 * 
 * Shows discovered patterns with:
 * - Force-directed graph visualization
 * - Stability score coloring
 * - Büchi acceptance badges
 */

interface Motif {
    motif_id: string;
    topic_sequence: string[];
    occurrence_count: number;
    stability_score: number;
    buchi_accepted: boolean;
}

interface MetaMotif {
    meta_id: string;
    constituent_motifs: string[];
    occurrence_count: number;
    buchi_accepted: boolean;
    emergence_score: number;
}

export const MotifVisualization = component$(() => {
    const motifs = useSignal<Motif[]>([]);
    const metaMotifs = useSignal<MetaMotif[]>([]);
    const loading = useSignal(true);
    const error = useSignal<string | null>(null);
    const extracting = useSignal(false);

    // Fetch motifs on mount
    useTask$(async () => {
        try {
            const [motifRes, metaRes] = await Promise.all([
                fetch('/api/evolution/motifs'),
                fetch('/api/evolution/meta-motifs')
            ]);

            if (motifRes.ok) {
                const data = await motifRes.json();
                motifs.value = data.motifs || [];
            }

            if (metaRes.ok) {
                const data = await metaRes.json();
                metaMotifs.value = data.meta_motifs || [];
            }
        } catch (e) {
            error.value = String(e);
        } finally {
            loading.value = false;
        }
    });

    const handleExtract = $(async () => {
        extracting.value = true;
        try {
            const res = await fetch('/api/evolution/motifs/extract', { method: 'POST' });
            if (res.ok) {
                // Refresh motifs
                const motifRes = await fetch('/api/evolution/motifs');
                if (motifRes.ok) {
                    const data = await motifRes.json();
                    motifs.value = data.motifs || [];
                }
            }
        } finally {
            extracting.value = false;
        }
    });

    const getStabilityColor = (score: number) => {
        if (score >= 0.8) return 'var(--color-success)';
        if (score >= 0.5) return 'var(--color-warning)';
        return 'var(--color-text-muted)';
    };

    return (
        <div class="motif-visualization">
            <div class="motif-header">
                <h3>ω-Motif Discovery</h3>
                <button
                    onClick$={handleExtract}
                    disabled={extracting.value}
                    class="btn-primary"
                >
                    {extracting.value ? 'Extracting...' : 'Extract Motifs'}
                </button>
            </div>

            {loading.value && <div class="loading">Loading motifs...</div>}
            {error.value && <div class="error">{error.value}</div>}

            <div class="motif-grid">
                {/* First-order motifs */}
                <div class="motif-section">
                    <h4>First-Order Motifs ({motifs.value.length})</h4>
                    <div class="motif-list">
                        {motifs.value.map((motif) => (
                            <div key={motif.motif_id} class="motif-card">
                                <div class="motif-id">{motif.motif_id.slice(0, 8)}...</div>
                                <div class="motif-topics">
                                    {motif.topic_sequence.map((t, i) => (
                                        <span key={i} class="topic-badge">{t}</span>
                                    ))}
                                </div>
                                <div class="motif-stats">
                                    <span class="occurrences">×{motif.occurrence_count}</span>
                                    <span
                                        class="stability"
                                        style={{ color: getStabilityColor(motif.stability_score) }}
                                    >
                                        φ={motif.stability_score.toFixed(2)}
                                    </span>
                                    {motif.buchi_accepted && (
                                        <span class="buchi-badge" title="Büchi Accepted (Recurring Forever)">ω</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Second-order meta-motifs (E13) */}
                <div class="motif-section">
                    <h4>Meta-Motifs (E13: Second-Order) ({metaMotifs.value.length})</h4>
                    <div class="motif-list">
                        {metaMotifs.value.map((meta) => (
                            <div key={meta.meta_id} class="motif-card meta">
                                <div class="motif-id">{meta.meta_id.slice(0, 8)}...</div>
                                <div class="constituent-motifs">
                                    {meta.constituent_motifs.map((m, i) => (
                                        <span key={i} class="constituent-badge">{m.slice(0, 6)}</span>
                                    ))}
                                </div>
                                <div class="motif-stats">
                                    <span class="occurrences">×{meta.occurrence_count}</span>
                                    <span class="emergence">E={meta.emergence_score.toFixed(2)}</span>
                                    {meta.buchi_accepted && (
                                        <span class="buchi-badge" title="Meta-Truth (Büchi Accepted)">Ω</span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            <style>{`
        .motif-visualization {
          padding: 1rem;
          background: var(--color-surface);
          border-radius: 8px;
        }
        .motif-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 1rem;
        }
        .motif-header h3 {
          margin: 0;
          color: var(--color-text);
        }
        .btn-primary {
          background: var(--color-primary);
          color: white;
          border: none;
          padding: 0.5rem 1rem;
          border-radius: 4px;
          cursor: pointer;
        }
        .btn-primary:disabled {
          opacity: 0.5;
        }
        .motif-grid {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 1rem;
        }
        .motif-section h4 {
          color: var(--color-text-muted);
          margin-bottom: 0.5rem;
        }
        .motif-list {
          display: flex;
          flex-direction: column;
          gap: 0.5rem;
        }
        .motif-card {
          background: var(--color-surface-elevated);
          padding: 0.75rem;
          border-radius: 6px;
          border: 1px solid var(--color-border);
        }
        .motif-card.meta {
          border-left: 3px solid var(--color-accent);
        }
        .motif-id {
          font-family: monospace;
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }
        .motif-topics, .constituent-motifs {
          display: flex;
          flex-wrap: wrap;
          gap: 0.25rem;
          margin: 0.5rem 0;
        }
        .topic-badge, .constituent-badge {
          background: var(--color-primary-muted);
          padding: 0.125rem 0.375rem;
          border-radius: 3px;
          font-size: 0.7rem;
        }
        .motif-stats {
          display: flex;
          gap: 0.5rem;
          font-size: 0.75rem;
        }
        .buchi-badge {
          background: var(--color-success);
          color: white;
          padding: 0.125rem 0.375rem;
          border-radius: 3px;
          font-weight: bold;
        }
      `}</style>
        </div>
    );
});

export default MotifVisualization;
