/**
 * VisionSynthesis - Real-time Visual Synthesis Display Component
 *
 * Shows program grammar visualization, inference state, and active learning feedback.
 * Connects to Theia WebSocket bridge for real-time updates.
 *
 * @module components/VisionSynthesis
 */

import { component$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';

// =============================================================================
// TYPES
// =============================================================================

interface SynthesisState {
    pattern: string;
    confidence: number;
    grammar: string;
    tokens: string[];
    lastUpdate: number;
}

interface InferenceResult {
    framesAnalyzed: number;
    synthesis: SynthesisState;
    timestamp: number;
}

interface ConnectionState {
    connected: boolean;
    reconnecting: boolean;
    error: string | null;
    lastHeartbeat: number;
}

// =============================================================================
// CONSTANTS
// =============================================================================

const WS_URL = 'ws://localhost:8091';
const PHI = 1.618033988749895; // Golden ratio

// =============================================================================
// COMPONENT
// =============================================================================

export const VisionSynthesis = component$(() => {
    // Connection state
    const connection = useStore<ConnectionState>({
        connected: false,
        reconnecting: false,
        error: null,
        lastHeartbeat: 0,
    });

    // Synthesis state
    const synthesis = useStore<SynthesisState>({
        pattern: 'awaiting_inference',
        confidence: 0,
        grammar: 'S -> ε',
        tokens: [],
        lastUpdate: 0,
    });

    // Frame stats
    const frameStats = useStore({
        sent: 0,
        buffered: 0,
        analyzed: 0,
    });

    // WebSocket reference (not reactive)
    const wsRef = useSignal<WebSocket | null>(null);

    // Connect to WebSocket
    const connect = $(async () => {
        if (wsRef.value && wsRef.value.readyState === WebSocket.OPEN) {
            return; // Already connected
        }

        connection.reconnecting = true;
        connection.error = null;

        try {
            const ws = new WebSocket(WS_URL);

            ws.onopen = () => {
                connection.connected = true;
                connection.reconnecting = false;
                connection.error = null;
                console.log('[VisionSynthesis] Connected to Theia WebSocket');
            };

            ws.onmessage = (event) => {
                try {
                    const msg = JSON.parse(event.data);
                    handleMessage(msg);
                } catch (e) {
                    console.error('[VisionSynthesis] Parse error:', e);
                }
            };

            ws.onerror = (error) => {
                connection.error = 'Connection error';
                console.error('[VisionSynthesis] WebSocket error:', error);
            };

            ws.onclose = () => {
                connection.connected = false;
                wsRef.value = null;
                console.log('[VisionSynthesis] Disconnected');
            };

            wsRef.value = ws;
        } catch (e) {
            connection.error = `Failed to connect: ${e}`;
            connection.reconnecting = false;
        }
    });

    // Handle incoming messages
    const handleMessage = (msg: Record<string, unknown>) => {
        switch (msg.type) {
            case 'welcome':
                console.log('[VisionSynthesis] Welcome:', msg.client_id);
                break;

            case 'inference_result':
                const result = msg as unknown as { frames_analyzed: number; synthesis: SynthesisState; ts: number };
                synthesis.pattern = result.synthesis.pattern;
                synthesis.confidence = result.synthesis.confidence;
                synthesis.grammar = result.synthesis.grammar;
                synthesis.tokens = result.synthesis.tokens;
                synthesis.lastUpdate = Date.now();
                frameStats.analyzed += result.frames_analyzed;
                break;

            case 'frame_ack':
                const ack = msg as { count: number; buffer_size: number };
                frameStats.sent = ack.count;
                frameStats.buffered = ack.buffer_size;
                break;

            case 'synthesis_update':
                const state = (msg as { state: SynthesisState }).state;
                synthesis.pattern = state.pattern;
                synthesis.confidence = state.confidence;
                synthesis.grammar = state.grammar;
                synthesis.tokens = state.tokens;
                synthesis.lastUpdate = Date.now();
                break;

            case 'a2a_heartbeat':
            case 'heartbeat_ack':
                connection.lastHeartbeat = Date.now();
                break;

            case 'pong':
                // Ping response
                break;

            default:
                console.log('[VisionSynthesis] Unknown message:', msg.type);
        }
    };

    // Request analysis
    const requestAnalysis = $(async () => {
        if (!wsRef.value || wsRef.value.readyState !== WebSocket.OPEN) {
            console.warn('[VisionSynthesis] Not connected');
            return;
        }

        wsRef.value.send(JSON.stringify({
            type: 'analyze',
            count: 10,
        }));
    });

    // Auto-connect on mount
    useVisibleTask$(() => {
        // Don't auto-connect, let user initiate
    });

    // Render confidence bar
    const renderConfidenceBar = (value: number) => {
        const filled = Math.round(value * 12);
        return '█'.repeat(filled) + '░'.repeat(12 - filled);
    };

    // Get quality color based on confidence (golden ratio thresholds)
    const getQualityColor = (confidence: number): string => {
        if (confidence >= 1 / PHI + 0.2) return '#00ff88'; // Excellent
        if (confidence >= 1 / PHI) return '#88ff00';       // Good
        if (confidence >= 1 / (PHI * PHI)) return '#ffaa00'; // Fair
        return '#ff4444'; // Poor
    };

    // ==========================================================================
    // RENDER
    // ==========================================================================

    return (
        <div class="vision-synthesis" style={{
            fontFamily: 'system-ui, sans-serif',
            padding: '16px',
            backgroundColor: '#1a1a2e',
            borderRadius: '8px',
            color: '#e0e0e0',
            maxWidth: '400px',
            border: '1px solid #333',
        }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                marginBottom: '12px',
                paddingBottom: '8px',
                borderBottom: '1px solid #333',
            }}>
                <h4 style={{ margin: 0, color: '#8844ff', fontSize: '14px' }}>
                    Visual Synthesis
                </h4>
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                }}>
                    <span style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: connection.connected ? '#00ff88' : '#ff4444',
                    }} />
                    <span style={{ fontSize: '10px', opacity: 0.7 }}>
                        {connection.connected ? 'Live' : 'Disconnected'}
                    </span>
                </div>
            </div>

            {/* Connection Controls */}
            {!connection.connected && (
                <button
                    onClick$={connect}
                    disabled={connection.reconnecting}
                    style={{
                        width: '100%',
                        padding: '8px',
                        marginBottom: '12px',
                        backgroundColor: connection.reconnecting ? '#444' : '#8844ff',
                        color: '#fff',
                        border: 'none',
                        borderRadius: '4px',
                        cursor: connection.reconnecting ? 'wait' : 'pointer',
                        fontSize: '12px',
                    }}
                >
                    {connection.reconnecting ? 'Connecting...' : 'Connect to Theia'}
                </button>
            )}

            {/* Error display */}
            {connection.error && (
                <div style={{
                    padding: '8px',
                    backgroundColor: '#ff444433',
                    border: '1px solid #ff4444',
                    borderRadius: '4px',
                    marginBottom: '12px',
                    fontSize: '11px',
                }}>
                    {connection.error}
                </div>
            )}

            {/* Grammar Display */}
            <div style={{
                backgroundColor: '#2a2a4e',
                borderRadius: '6px',
                padding: '12px',
                marginBottom: '12px',
            }}>
                <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '4px' }}>
                    GRAMMAR
                </div>
                <div style={{
                    fontFamily: 'monospace',
                    fontSize: '12px',
                    color: '#00ff88',
                    wordBreak: 'break-all',
                }}>
                    {synthesis.grammar}
                </div>
            </div>

            {/* Pattern & Confidence */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr',
                gap: '12px',
                marginBottom: '12px',
            }}>
                <div style={{ backgroundColor: '#2a2a4e', borderRadius: '6px', padding: '12px' }}>
                    <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '4px' }}>
                        PATTERN
                    </div>
                    <div style={{ fontSize: '12px', fontWeight: 'bold' }}>
                        {synthesis.pattern}
                    </div>
                </div>

                <div style={{ backgroundColor: '#2a2a4e', borderRadius: '6px', padding: '12px' }}>
                    <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '4px' }}>
                        CONFIDENCE
                    </div>
                    <div style={{
                        fontFamily: 'monospace',
                        fontSize: '11px',
                        color: getQualityColor(synthesis.confidence),
                    }}>
                        {renderConfidenceBar(synthesis.confidence)} {(synthesis.confidence * 100).toFixed(0)}%
                    </div>
                </div>
            </div>

            {/* Active Tokens */}
            {synthesis.tokens.length > 0 && (
                <div style={{
                    backgroundColor: '#2a2a4e',
                    borderRadius: '6px',
                    padding: '12px',
                    marginBottom: '12px',
                }}>
                    <div style={{ fontSize: '10px', opacity: 0.7, marginBottom: '8px' }}>
                        ACTIVE TOKENS
                    </div>
                    <div style={{ display: 'flex', flexWrap: 'wrap', gap: '4px' }}>
                        {synthesis.tokens.map((token) => (
                            <span
                                key={token}
                                style={{
                                    padding: '2px 8px',
                                    backgroundColor: '#8844ff33',
                                    border: '1px solid #8844ff',
                                    borderRadius: '4px',
                                    fontSize: '10px',
                                    fontFamily: 'monospace',
                                }}
                            >
                                {token}
                            </span>
                        ))}
                    </div>
                </div>
            )}

            {/* Frame Stats */}
            <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '8px',
                marginBottom: '12px',
            }}>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '16px', fontWeight: 'bold' }}>{frameStats.sent}</div>
                    <div style={{ fontSize: '9px', opacity: 0.6 }}>SENT</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '16px', fontWeight: 'bold' }}>{frameStats.buffered}</div>
                    <div style={{ fontSize: '9px', opacity: 0.6 }}>BUFFERED</div>
                </div>
                <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '16px', fontWeight: 'bold' }}>{frameStats.analyzed}</div>
                    <div style={{ fontSize: '9px', opacity: 0.6 }}>ANALYZED</div>
                </div>
            </div>

            {/* Analyze Button */}
            {connection.connected && (
                <button
                    onClick$={requestAnalysis}
                    style={{
                        width: '100%',
                        padding: '10px',
                        backgroundColor: '#00ff8833',
                        border: '1px solid #00ff88',
                        color: '#00ff88',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '12px',
                        fontWeight: 'bold',
                    }}
                >
                    Request Analysis
                </button>
            )}

            {/* Status Bar */}
            <div style={{
                marginTop: '12px',
                paddingTop: '8px',
                borderTop: '1px solid #333',
                fontSize: '9px',
                opacity: 0.5,
                display: 'flex',
                justifyContent: 'space-between',
            }}>
                <span>Last update: {synthesis.lastUpdate ? new Date(synthesis.lastUpdate).toLocaleTimeString() : '—'}</span>
                <span>PLURIBUS v1</span>
            </div>
        </div>
    );
});

export default VisionSynthesis;
