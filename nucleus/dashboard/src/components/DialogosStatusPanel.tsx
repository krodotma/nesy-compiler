/**
 * DialogosStatusPanel.tsx
 * Real-time Dialogos streaming status and priority queue visualization.
 * Shows active provider cells and queue depth per priority level.
 */
import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface StreamingCell {
    reqId: string;
    provider: string;
    progress: number;  // 0-100
    latencyMs: number;
    status: 'streaming' | 'complete' | 'error';
}

interface PriorityQueueStats {
    CRITICAL: number;
    HIGH: number;
    NORMAL: number;
    LOW: number;
    BULK: number;
}

interface DialogosStats {
    activeStreams: number;
    maxConcurrent: number;
    queuedTotal: number;
    byPriority: PriorityQueueStats;
    saturated: Record<string, boolean>;
}

// ============================================================================
// Styles
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
    container: {
        background: 'rgba(0, 0, 0, 0.85)',
        borderRadius: '12px',
        padding: '16px',
        color: '#fff',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        minWidth: '280px',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
    },
    header: {
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '16px',
        borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
        paddingBottom: '12px',
    },
    title: {
        fontSize: '14px',
        fontWeight: 600,
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
    },
    section: {
        marginBottom: '16px',
    },
    sectionTitle: {
        fontSize: '11px',
        color: 'rgba(255, 255, 255, 0.5)',
        textTransform: 'uppercase' as const,
        letterSpacing: '0.5px',
        marginBottom: '8px',
    },
    cellGrid: {
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fill, minmax(60px, 1fr))',
        gap: '8px',
    },
    cell: {
        padding: '8px',
        borderRadius: '8px',
        background: 'rgba(255, 255, 255, 0.05)',
        textAlign: 'center' as const,
        fontSize: '10px',
    },
    queueBar: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        marginBottom: '6px',
    },
    queueLabel: {
        width: '60px',
        fontSize: '10px',
        fontFamily: 'monospace',
    },
    queueTrack: {
        flex: 1,
        height: '8px',
        background: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '4px',
        overflow: 'hidden',
    },
    queueFill: {
        height: '100%',
        borderRadius: '4px',
        transition: 'width 0.3s ease',
    },
    queueCount: {
        width: '30px',
        textAlign: 'right' as const,
        fontSize: '10px',
        fontFamily: 'monospace',
    },
    empty: {
        color: 'rgba(255, 255, 255, 0.4)',
        textAlign: 'center' as const,
        padding: '12px',
        fontSize: '11px',
    },
};

// ============================================================================
// Helpers
// ============================================================================

const PRIORITY_COLORS: Record<string, string> = {
    CRITICAL: '#ef4444',
    HIGH: '#f59e0b',
    NORMAL: '#22c55e',
    LOW: '#3b82f6',
    BULK: '#6b7280',
};

const PRIORITY_MAX: Record<string, number> = {
    CRITICAL: 100,
    HIGH: 50,
    NORMAL: 200,
    LOW: 100,
    BULK: 500,
};

const PROVIDER_ICONS: Record<string, string> = {
    'claude-opus': '🟣',
    'claude-sonnet': '🔵',
    'gemini-2': '💎',
    'qwen-plus': '🟢',
};

// ============================================================================
// Component
// ============================================================================

interface DialogosStatusPanelProps {
    pollInterval?: number;
    apiEndpoint?: string;
}

export function DialogosStatusPanel({
    pollInterval = 1000,
    apiEndpoint = '/api/dialogos/status',
}: DialogosStatusPanelProps) {
    const [cells, setCells] = useState<StreamingCell[]>([]);
    const [stats, setStats] = useState<DialogosStats | null>(null);
    const [isConnected, setIsConnected] = useState(false);

    const fetchStatus = useCallback(async () => {
        try {
            const res = await fetch(apiEndpoint);
            if (res.ok) {
                const data = await res.json();
                setCells(data.cells || []);
                setStats(data.stats || null);
                setIsConnected(true);
            }
        } catch {
            setIsConnected(false);
        }
    }, [apiEndpoint]);

    useEffect(() => {
        fetchStatus();
        const interval = setInterval(fetchStatus, pollInterval);
        return () => clearInterval(interval);
    }, [fetchStatus, pollInterval]);

    const priorityKeys = ['CRITICAL', 'HIGH', 'NORMAL', 'LOW', 'BULK'] as const;

    return (
        <div style={styles.container}>
            {/* Header */}
            <div style={styles.header}>
                <div style={styles.title}>
                    <span>🎯</span>
                    <span>Dialogos</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span style={{ fontSize: '10px', color: 'rgba(255,255,255,0.5)' }}>
                        {stats?.activeStreams || 0}/{stats?.maxConcurrent || 5}
                    </span>
                    <div style={{
                        width: 8,
                        height: 8,
                        borderRadius: '50%',
                        background: isConnected ? '#22c55e' : '#ef4444',
                    }} />
                </div>
            </div>

            {/* Active Cells */}
            <div style={styles.section}>
                <div style={styles.sectionTitle}>Active Cells</div>
                {cells.length === 0 ? (
                    <div style={styles.empty}>No active streams</div>
                ) : (
                    <div style={styles.cellGrid}>
                        {cells.map(cell => (
                            <div key={cell.reqId} style={{
                                ...styles.cell,
                                borderLeft: `3px solid ${cell.status === 'streaming' ? '#22c55e' : cell.status === 'error' ? '#ef4444' : '#3b82f6'}`,
                            }}>
                                <div>{PROVIDER_ICONS[cell.provider] || '⚪'}</div>
                                <div style={{ marginTop: 4 }}>{cell.latencyMs}ms</div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {/* Priority Queue */}
            <div style={styles.section}>
                <div style={styles.sectionTitle}>Priority Queue</div>
                {priorityKeys.map(priority => {
                    const count = stats?.byPriority?.[priority] || 0;
                    const max = PRIORITY_MAX[priority];
                    const pct = Math.min((count / max) * 100, 100);
                    const saturated = stats?.saturated?.[priority];

                    return (
                        <div key={priority} style={styles.queueBar}>
                            <div style={{
                                ...styles.queueLabel,
                                color: saturated ? '#ef4444' : 'rgba(255,255,255,0.7)',
                            }}>
                                {priority}
                            </div>
                            <div style={styles.queueTrack}>
                                <div style={{
                                    ...styles.queueFill,
                                    width: `${pct}%`,
                                    background: saturated
                                        ? `repeating-linear-gradient(45deg, ${PRIORITY_COLORS[priority]}, ${PRIORITY_COLORS[priority]} 5px, #ef4444 5px, #ef4444 10px)`
                                        : PRIORITY_COLORS[priority],
                                }} />
                            </div>
                            <div style={styles.queueCount}>{count}</div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}

export default DialogosStatusPanel;
