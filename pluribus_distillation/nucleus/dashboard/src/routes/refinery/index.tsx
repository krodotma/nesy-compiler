import { component$, useSignal, useTask$ } from '@builder.io/qwik';
import MotifVisualization from '../../components/evolution/MotifVisualization';
import NearNovelExplorer from '../../components/evolution/NearNovelExplorer';
import HexisPromotionWorkflow from '../../components/evolution/HexisPromotionWorkflow';

/**
 * RefineryView - Entropy/Negentropy Transmutation Dashboard
 * 
 * Route: /refinery
 * 
 * Displays:
 * - Real-time entropy gauge
 * - CMP evolution chart
 * - Transmutation phase indicator
 * - Population health heatmap
 */

interface EntropyState {
    entropy: number;
    negentropy: number;
    phase: string;
    ts: string;
}

interface CMPDataPoint {
    ts: string;
    cmp: number;
    lineage_id?: string;
}

export default component$(() => {
    const entropyState = useSignal<EntropyState | null>(null);
    const cmpTimeline = useSignal<CMPDataPoint[]>([]);
    const loading = useSignal(true);
    const activeTab = useSignal<'entropy' | 'motifs' | 'seeds' | 'promotion'>('entropy');

    // Fetch entropy state
    useTask$(async () => {
        try {
            const [entropyRes, cmpRes] = await Promise.all([
                fetch('/api/evolution/entropy'),
                fetch('/api/evolution/cmp/timeline')
            ]);

            if (entropyRes.ok) {
                entropyState.value = await entropyRes.json();
            }

            if (cmpRes.ok) {
                const data = await cmpRes.json();
                cmpTimeline.value = data.timeline || [];
            }
        } catch (e) {
            console.error('Failed to fetch evolution data:', e);
        } finally {
            loading.value = false;
        }
    });

    const getPhaseColor = (phase: string) => {
        switch (phase) {
            case 'equilibrium': return 'var(--color-success)';
            case 'transmuting': return 'var(--color-warning)';
            case 'chaos': return 'var(--color-error)';
            default: return 'var(--color-text-muted)';
        }
    };

    return (
        <div class="refinery-view">
            <header class="refinery-header">
                <h1>🔮 Refinery</h1>
                <p class="subtitle">DUALITY-BIND Evolution Observatory</p>
            </header>

            <nav class="refinery-tabs">
                <button
                    class={`tab ${activeTab.value === 'entropy' ? 'active' : ''}`}
                    onClick$={() => activeTab.value = 'entropy'}
                >
                    Entropy
                </button>
                <button
                    class={`tab ${activeTab.value === 'motifs' ? 'active' : ''}`}
                    onClick$={() => activeTab.value = 'motifs'}
                >
                    Motifs
                </button>
                <button
                    class={`tab ${activeTab.value === 'seeds' ? 'active' : ''}`}
                    onClick$={() => activeTab.value = 'seeds'}
                >
                    Near-Novel
                </button>
                <button
                    class={`tab ${activeTab.value === 'promotion' ? 'active' : ''}`}
                    onClick$={() => activeTab.value = 'promotion'}
                >
                    Promotion
                </button>
            </nav>

            <main class="refinery-content">
                {activeTab.value === 'entropy' && (
                    <div class="entropy-panel">
                        {loading.value ? (
                            <div class="loading">Loading entropy state...</div>
                        ) : entropyState.value ? (
                            <>
                                <div class="entropy-gauges">
                                    <div class="gauge">
                                        <div class="gauge-label">Entropy (Chaos)</div>
                                        <div class="gauge-bar">
                                            <div
                                                class="gauge-fill entropy"
                                                style={{ width: `${entropyState.value.entropy * 100}%` }}
                                            />
                                        </div>
                                        <div class="gauge-value">{(entropyState.value.entropy * 100).toFixed(1)}%</div>
                                    </div>

                                    <div class="gauge">
                                        <div class="gauge-label">Negentropy (Order)</div>
                                        <div class="gauge-bar">
                                            <div
                                                class="gauge-fill negentropy"
                                                style={{ width: `${entropyState.value.negentropy * 100}%` }}
                                            />
                                        </div>
                                        <div class="gauge-value">{(entropyState.value.negentropy * 100).toFixed(1)}%</div>
                                    </div>
                                </div>

                                <div class="phase-indicator">
                                    <span class="phase-label">Transmutation Phase:</span>
                                    <span
                                        class="phase-value"
                                        style={{ color: getPhaseColor(entropyState.value.phase) }}
                                    >
                                        {entropyState.value.phase.toUpperCase()}
                                    </span>
                                </div>

                                <div class="cmp-chart">
                                    <h4>CMP Evolution Timeline</h4>
                                    <div class="chart-placeholder">
                                        {cmpTimeline.value.length > 0 ? (
                                            <div class="mini-chart">
                                                {cmpTimeline.value.slice(-20).map((point, i) => (
                                                    <div
                                                        key={i}
                                                        class="chart-bar"
                                                        style={{ height: `${point.cmp * 100}%` }}
                                                        title={`CMP: ${point.cmp.toFixed(3)} at ${point.ts}`}
                                                    />
                                                ))}
                                            </div>
                                        ) : (
                                            <p>No CMP data available</p>
                                        )}
                                    </div>
                                </div>
                            </>
                        ) : (
                            <div class="error">Failed to load entropy state</div>
                        )}
                    </div>
                )}

                {activeTab.value === 'motifs' && (
                    <MotifVisualization />
                )}

                {activeTab.value === 'seeds' && (
                    <div class="seeds-panel">
                        <h3>Near-but-Novel Exploration Seeds (E14)</h3>
                        <p>Edge-of-chaos exploration candidates will appear here.</p>
                        {/* NearNovelExplorer component will go here */}
                    </div>
                )}

                {activeTab.value === 'promotion' && (
                    <div class="promotion-panel">
                        <h3>Hexis→Entelechy Promotion (E12)</h3>
                        <p>Pattern promotion candidates will appear here.</p>
                        {/* HexisPromotionWorkflow component will go here */}
                    </div>
                )}
            </main>

            <style>{`
        .refinery-view {
          min-height: 100vh;
          background: var(--color-background);
          color: var(--color-text);
          padding: 2rem;
        }
        .refinery-header {
          margin-bottom: 2rem;
        }
        .refinery-header h1 {
          margin: 0;
          font-size: 2rem;
        }
        .subtitle {
          color: var(--color-text-muted);
          margin: 0.5rem 0 0;
        }
        .refinery-tabs {
          display: flex;
          gap: 0.5rem;
          margin-bottom: 1.5rem;
          border-bottom: 1px solid var(--color-border);
          padding-bottom: 0.5rem;
        }
        .tab {
          background: transparent;
          border: none;
          color: var(--color-text-muted);
          padding: 0.5rem 1rem;
          cursor: pointer;
          border-radius: 4px 4px 0 0;
          transition: all 0.2s;
        }
        .tab:hover {
          background: var(--color-surface);
        }
        .tab.active {
          background: var(--color-primary);
          color: white;
        }
        .entropy-panel {
          display: flex;
          flex-direction: column;
          gap: 1.5rem;
        }
        .entropy-gauges {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 2rem;
        }
        .gauge {
          background: var(--color-surface);
          padding: 1rem;
          border-radius: 8px;
        }
        .gauge-label {
          font-size: 0.875rem;
          color: var(--color-text-muted);
          margin-bottom: 0.5rem;
        }
        .gauge-bar {
          height: 24px;
          background: var(--color-surface-elevated);
          border-radius: 12px;
          overflow: hidden;
        }
        .gauge-fill {
          height: 100%;
          border-radius: 12px;
          transition: width 0.5s ease;
        }
        .gauge-fill.entropy {
          background: linear-gradient(90deg, #f59e0b, #ef4444);
        }
        .gauge-fill.negentropy {
          background: linear-gradient(90deg, #10b981, #3b82f6);
        }
        .gauge-value {
          text-align: right;
          margin-top: 0.25rem;
          font-family: monospace;
        }
        .phase-indicator {
          background: var(--color-surface);
          padding: 1rem;
          border-radius: 8px;
          text-align: center;
        }
        .phase-value {
          font-size: 1.5rem;
          font-weight: bold;
          margin-left: 1rem;
        }
        .cmp-chart {
          background: var(--color-surface);
          padding: 1rem;
          border-radius: 8px;
        }
        .cmp-chart h4 {
          margin: 0 0 1rem;
        }
        .mini-chart {
          display: flex;
          align-items: flex-end;
          height: 100px;
          gap: 2px;
        }
        .chart-bar {
          flex: 1;
          background: var(--color-primary);
          border-radius: 2px 2px 0 0;
          min-height: 4px;
        }
        .seeds-panel, .promotion-panel {
          background: var(--color-surface);
          padding: 1.5rem;
          border-radius: 8px;
        }
      `}</style>
        </div>
    );
});
