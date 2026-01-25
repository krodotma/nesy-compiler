/**
 * A2ATaskMonitor.tsx
 * Real-time A2A dispatcher task monitoring panel.
 * Displays Star Topology task flow and P/E/L/R/Q gate states.
 */
import React, { useState, useEffect, useCallback } from 'react';

// ============================================================================
// Types
// ============================================================================

interface A2ATask {
    id: string;
    actor: string;
    target: string;
    gate: 'P' | 'E' | 'L' | 'R' | 'Q';
    status: 'pending' | 'dispatched' | 'in_progress' | 'completed' | 'failed' | 'vetoed';
    topic: string;
    timestamp: number;
}

interface A2AStats {
    total: number;
    pending: number;
    active: number;
    completed: number;
    failed: number;
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
        minWidth: '320px',
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
    statsRow: {
        display: 'flex',
        gap: '12px',
        marginBottom: '16px',
    },
    statBadge: {
        padding: '4px 8px',
        borderRadius: '6px',
        fontSize: '11px',
        fontFamily: 'monospace',
    },
    taskList: {
        maxHeight: '200px',
        overflowY: 'auto' as const,
        display: 'flex',
        flexDirection: 'column' as const,
        gap: '8px',
    },
    taskItem: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '8px',
        background: 'rgba(255, 255, 255, 0.05)',
        borderRadius: '8px',
        fontSize: '11px',
    },
    gateBadge: {
        width: '24px',
        height: '24px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontWeight: 700,
        fontSize: '12px',
    },
    taskMeta: {
        flex: 1,
        display: 'flex',
        flexDirection: 'column' as const,
        gap: '2px',
    },
    empty: {
        color: 'rgba(255, 255, 255, 0.4)',
        textAlign: 'center' as const,
        padding: '20px',
        fontSize: '12px',
    },
};

// ============================================================================
// Helpers
// ============================================================================

const GATE_COLORS: Record<string, string> = {
    P: '#8b5cf6', // Propose - purple
    E: '#22c55e', // Execute - green
    L: '#3b82f6', // Log - blue
    R: '#f59e0b', // Review - amber
    Q: '#6b7280', // Queue - gray
};

const STATUS_COLORS: Record<string, string> = {
    pending: '#6b7280',
    dispatched: '#8b5cf6',
    in_progress: '#22c55e',
    completed: '#3b82f6',
    failed: '#ef4444',
    vetoed: '#f59e0b',
};

function formatTimestamp(ts: number): string {
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

// ============================================================================
// Component
// ============================================================================

interface A2ATaskMonitorProps {
    /** Poll interval in ms */
    pollInterval?: number;
    /** Max tasks to display */
    maxTasks?: number;
    /** API endpoint for fetching tasks */
    apiEndpoint?: string;
}

export function A2ATaskMonitor({
    pollInterval = 2000,
    maxTasks = 10,
    apiEndpoint = '/api/a2a/tasks',
}: A2ATaskMonitorProps) {
    const [tasks, setTasks] = useState<A2ATask[]>([]);
    const [stats, setStats] = useState<A2AStats>({
        total: 0,
        pending: 0,
        active: 0,
        completed: 0,
        failed: 0,
    });
    const [isConnected, setIsConnected] = useState(false);

    // Fetch tasks from API
    const fetchTasks = useCallback(async () => {
        try {
            const res = await fetch(apiEndpoint);
            if (res.ok) {
                const data = await res.json();
                setTasks(data.tasks?.slice(0, maxTasks) || []);
                setStats(data.stats || stats);
                setIsConnected(true);
            }
        } catch {
            setIsConnected(false);
        }
    }, [apiEndpoint, maxTasks]);

    // Poll for updates
    useEffect(() => {
        fetchTasks();
        const interval = setInterval(fetchTasks, pollInterval);
        return () => clearInterval(interval);
    }, [fetchTasks, pollInterval]);

    return (
        <div style={styles.container}>
            {/* Header */}
            <div style={styles.header}>
                <div style={styles.title}>
                    <span>🌟</span>
                    <span>A2A Task Monitor</span>
                </div>
                <div style={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    background: isConnected ? '#22c55e' : '#ef4444',
                }} />
            </div>

            {/* Stats */}
            <div style={styles.statsRow}>
                <div style={{ ...styles.statBadge, background: 'rgba(139, 92, 246, 0.2)', color: '#a78bfa' }}>
                    Q: {stats.pending}
                </div>
                <div style={{ ...styles.statBadge, background: 'rgba(34, 197, 94, 0.2)', color: '#4ade80' }}>
                    ▶ {stats.active}
                </div>
                <div style={{ ...styles.statBadge, background: 'rgba(59, 130, 246, 0.2)', color: '#60a5fa' }}>
                    ✓ {stats.completed}
                </div>
                {stats.failed > 0 && (
                    <div style={{ ...styles.statBadge, background: 'rgba(239, 68, 68, 0.2)', color: '#f87171' }}>
                        ✗ {stats.failed}
                    </div>
                )}
            </div>

            {/* Task List */}
            <div style={styles.taskList}>
                {tasks.length === 0 ? (
                    <div style={styles.empty}>No active tasks</div>
                ) : (
                    tasks.map(task => (
                        <div key={task.id} style={styles.taskItem}>
                            <div style={{
                                ...styles.gateBadge,
                                background: GATE_COLORS[task.gate] || '#6b7280',
                            }}>
                                {task.gate}
                            </div>
                            <div style={styles.taskMeta}>
                                <div style={{ fontWeight: 500 }}>
                                    {task.actor} → {task.target}
                                </div>
                                <div style={{ color: 'rgba(255,255,255,0.5)' }}>
                                    {task.topic} • {formatTimestamp(task.timestamp)}
                                </div>
                            </div>
                            <div style={{
                                padding: '2px 6px',
                                borderRadius: '4px',
                                background: STATUS_COLORS[task.status] || '#6b7280',
                                fontSize: '9px',
                                textTransform: 'uppercase',
                            }}>
                                {task.status}
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}

export default A2ATaskMonitor;
