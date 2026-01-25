import { component$, useSignal, useTask$, $ } from '@builder.io/qwik';

/**
 * HexisPromotionWorkflow - E12: Hexis→Entelechy Pattern Promotion
 * 
 * Approval workflow for promoting ephemeral patterns to permanent motifs:
 * - Candidate list with stability scores
 * - Approve/Reject actions
 * - Promotion history log
 */

interface PromotionCandidate {
    candidate_id: string;
    source_pattern_id: string;
    proposed_motif_id: string;
    stability_evidence: number;  // 0-1
    recurrence_count: number;
    pattern_preview: string[];   // Topic sequence preview
    status: string;              // pending, approved, rejected
    created_at: string;
}

export const HexisPromotionWorkflow = component$(() => {
    const candidates = useSignal<PromotionCandidate[]>([]);
    const loading = useSignal(true);
    const error = useSignal<string | null>(null);
    const processingId = useSignal<string | null>(null);
    const showHistory = useSignal(false);

    // Fetch candidates on mount
    useTask$(async () => {
        try {
            const res = await fetch('/api/evolution/candidates');
            if (res.ok) {
                const data = await res.json();
                candidates.value = data.candidates || [];
            } else {
                error.value = `Failed to fetch candidates: ${res.status}`;
            }
        } catch (e) {
            error.value = String(e);
        } finally {
            loading.value = false;
        }
    });

    const handleApprove = $(async (candidateId: string) => {
        processingId.value = candidateId;
        try {
            const res = await fetch(`/api/evolution/candidates/${candidateId}/approve`, {
                method: 'POST'
            });
            if (res.ok) {
                candidates.value = candidates.value.map(c =>
                    c.candidate_id === candidateId ? { ...c, status: 'approved' } : c
                );
            }
        } finally {
            processingId.value = null;
        }
    });

    const handleReject = $(async (candidateId: string) => {
        processingId.value = candidateId;
        try {
            const res = await fetch(`/api/evolution/candidates/${candidateId}/reject`, {
                method: 'POST'
            });
            if (res.ok) {
                candidates.value = candidates.value.map(c =>
                    c.candidate_id === candidateId ? { ...c, status: 'rejected' } : c
                );
            }
        } finally {
            processingId.value = null;
        }
    });

    const getStabilityColor = (stability: number) => {
        if (stability >= 0.8) return '#22c55e';  // High stability = green
        if (stability >= 0.5) return '#eab308';  // Medium = yellow
        return '#ef4444';  // Low = red
    };

    const pendingCandidates = candidates.value.filter(c => c.status === 'pending');
    const approvedCandidates = candidates.value.filter(c => c.status === 'approved');
    const rejectedCandidates = candidates.value.filter(c => c.status === 'rejected');

    return (
        <div class="hexis-promotion-workflow">
            <div class="workflow-header">
                <div class="header-content">
                    <h3>⬆️ Hexis→Entelechy Promotion (E12)</h3>
                    <p class="tagline">Promote stable ephemeral patterns to permanent motifs</p>
                </div>
                <div class="header-stats">
                    <span class="stat pending">{pendingCandidates.length} pending</span>
                    <span class="stat approved">{approvedCandidates.length} approved</span>
                    <span class="stat rejected">{rejectedCandidates.length} rejected</span>
                </div>
            </div>

            {loading.value && <div class="loading">Loading promotion candidates...</div>}
            {error.value && <div class="error">{error.value}</div>}

            {/* Pending Candidates */}
            <div class="candidates-section">
                <h4>Pending Approval ({pendingCandidates.length})</h4>
                {pendingCandidates.length === 0 ? (
                    <p class="empty">No patterns awaiting promotion</p>
                ) : (
                    <div class="candidates-list">
                        {pendingCandidates.map((candidate) => (
                            <div key={candidate.candidate_id} class="candidate-card">
                                <div class="candidate-header">
                                    <span class="candidate-id">
                                        {candidate.candidate_id.slice(0, 8)}...
                                    </span>
                                    <span class="recurrence">
                                        ×{candidate.recurrence_count} recurrences
                                    </span>
                                </div>

                                <div class="pattern-preview">
                                    {candidate.pattern_preview.map((topic, i) => (
                                        <span key={i} class="topic-badge">{topic}</span>
                                    ))}
                                </div>

                                <div class="stability-section">
                                    <span class="stability-label">Stability Evidence:</span>
                                    <div class="stability-bar">
                                        <div
                                            class="stability-fill"
                                            style={{
                                                width: `${candidate.stability_evidence * 100}%`,
                                                background: getStabilityColor(candidate.stability_evidence)
                                            }}
                                        />
                                    </div>
                                    <span class="stability-value">
                                        {(candidate.stability_evidence * 100).toFixed(0)}%
                                    </span>
                                </div>

                                <div class="promotion-info">
                                    <div class="info-row">
                                        <span class="label">Source:</span>
                                        <code>{candidate.source_pattern_id.slice(0, 12)}...</code>
                                    </div>
                                    <div class="info-row">
                                        <span class="label">→ Motif:</span>
                                        <code>{candidate.proposed_motif_id.slice(0, 12)}...</code>
                                    </div>
                                </div>

                                <div class="candidate-actions">
                                    <button
                                        class="btn-approve"
                                        onClick$={() => handleApprove(candidate.candidate_id)}
                                        disabled={processingId.value === candidate.candidate_id}
                                    >
                                        {processingId.value === candidate.candidate_id
                                            ? '...'
                                            : '✓ Approve Promotion'}
                                    </button>
                                    <button
                                        class="btn-reject"
                                        onClick$={() => handleReject(candidate.candidate_id)}
                                        disabled={processingId.value === candidate.candidate_id}
                                    >
                                        ✕ Reject
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* History Toggle */}
            <div class="history-toggle">
                <button onClick$={() => showHistory.value = !showHistory.value}>
                    {showHistory.value ? '▼ Hide History' : '▶ Show History'}
                </button>
            </div>

            {/* Recent History */}
            {showHistory.value && (
                <div class="history-section">
                    {approvedCandidates.length > 0 && (
                        <div class="history-group">
                            <h5>Recently Approved</h5>
                            {approvedCandidates.slice(0, 5).map((c) => (
                                <div key={c.candidate_id} class="history-item approved">
                                    <span class="status-icon">✓</span>
                                    <span class="pattern-summary">
                                        {c.pattern_preview.slice(0, 3).join(' → ')}
                                    </span>
                                    <span class="stability-badge">
                                        {(c.stability_evidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}

                    {rejectedCandidates.length > 0 && (
                        <div class="history-group">
                            <h5>Recently Rejected</h5>
                            {rejectedCandidates.slice(0, 5).map((c) => (
                                <div key={c.candidate_id} class="history-item rejected">
                                    <span class="status-icon">✕</span>
                                    <span class="pattern-summary">
                                        {c.pattern_preview.slice(0, 3).join(' → ')}
                                    </span>
                                    <span class="stability-badge">
                                        {(c.stability_evidence * 100).toFixed(0)}%
                                    </span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}

            <style>{`
        .hexis-promotion-workflow {
          padding: 1rem;
          background: var(--color-surface);
          border-radius: 8px;
        }
        .workflow-header {
          display: flex;
          justify-content: space-between;
          align-items: flex-start;
          margin-bottom: 1rem;
        }
        .header-content h3 {
          margin: 0;
          color: var(--color-text);
        }
        .tagline {
          color: var(--color-text-muted);
          font-size: 0.875rem;
          margin: 0.25rem 0 0;
        }
        .header-stats {
          display: flex;
          gap: 0.5rem;
        }
        .stat {
          font-size: 0.75rem;
          padding: 0.25rem 0.5rem;
          border-radius: 4px;
        }
        .stat.pending {
          background: rgba(99, 102, 241, 0.2);
          color: #6366f1;
        }
        .stat.approved {
          background: rgba(34, 197, 94, 0.2);
          color: #22c55e;
        }
        .stat.rejected {
          background: rgba(239, 68, 68, 0.2);
          color: #ef4444;
        }
        .candidates-section h4 {
          color: var(--color-text-muted);
          margin-bottom: 0.75rem;
        }
        .candidates-list {
          display: flex;
          flex-direction: column;
          gap: 0.75rem;
        }
        .candidate-card {
          background: var(--color-surface-elevated);
          padding: 1rem;
          border-radius: 8px;
          border: 1px solid var(--color-border);
        }
        .candidate-header {
          display: flex;
          justify-content: space-between;
          margin-bottom: 0.75rem;
        }
        .candidate-id {
          font-family: monospace;
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }
        .recurrence {
          font-size: 0.75rem;
          color: var(--color-primary);
        }
        .pattern-preview {
          display: flex;
          flex-wrap: wrap;
          gap: 0.25rem;
          margin-bottom: 0.75rem;
        }
        .topic-badge {
          background: var(--color-primary-muted);
          padding: 0.125rem 0.5rem;
          border-radius: 3px;
          font-size: 0.7rem;
        }
        .stability-section {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          margin-bottom: 0.75rem;
        }
        .stability-label {
          font-size: 0.75rem;
          color: var(--color-text-muted);
        }
        .stability-bar {
          flex: 1;
          height: 8px;
          background: var(--color-surface);
          border-radius: 4px;
          overflow: hidden;
        }
        .stability-fill {
          height: 100%;
          border-radius: 4px;
          transition: width 0.3s;
        }
        .stability-value {
          font-family: monospace;
          font-size: 0.75rem;
          min-width: 35px;
        }
        .promotion-info {
          background: var(--color-surface);
          padding: 0.5rem;
          border-radius: 4px;
          margin-bottom: 0.75rem;
          font-size: 0.75rem;
        }
        .info-row {
          display: flex;
          gap: 0.5rem;
        }
        .info-row .label {
          color: var(--color-text-muted);
        }
        .info-row code {
          font-family: monospace;
          color: var(--color-text);
        }
        .candidate-actions {
          display: flex;
          gap: 0.5rem;
        }
        .btn-approve, .btn-reject {
          padding: 0.5rem 1rem;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 0.8rem;
        }
        .btn-approve {
          background: #22c55e;
          color: white;
          flex: 1;
        }
        .btn-reject {
          background: transparent;
          border: 1px solid var(--color-border);
          color: var(--color-text-muted);
        }
        .btn-approve:disabled, .btn-reject:disabled {
          opacity: 0.5;
        }
        .history-toggle {
          margin-top: 1rem;
          padding-top: 1rem;
          border-top: 1px solid var(--color-border);
        }
        .history-toggle button {
          background: transparent;
          border: none;
          color: var(--color-text-muted);
          cursor: pointer;
          font-size: 0.875rem;
        }
        .history-section {
          margin-top: 1rem;
        }
        .history-group {
          margin-bottom: 1rem;
        }
        .history-group h5 {
          color: var(--color-text-muted);
          font-size: 0.75rem;
          margin-bottom: 0.5rem;
        }
        .history-item {
          display: flex;
          align-items: center;
          gap: 0.5rem;
          padding: 0.375rem 0.5rem;
          background: var(--color-surface-elevated);
          border-radius: 4px;
          margin-bottom: 0.25rem;
          font-size: 0.75rem;
        }
        .history-item.approved .status-icon {
          color: #22c55e;
        }
        .history-item.rejected .status-icon {
          color: #ef4444;
        }
        .pattern-summary {
          flex: 1;
          color: var(--color-text);
        }
        .stability-badge {
          font-family: monospace;
          color: var(--color-text-muted);
        }
        .empty {
          color: var(--color-text-muted);
          font-style: italic;
        }
      `}</style>
        </div>
    );
});

export default HexisPromotionWorkflow;
