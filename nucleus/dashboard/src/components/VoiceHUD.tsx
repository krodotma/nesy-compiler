/**
 * VoiceHUD.tsx
 * Voice pipeline heads-up display component.
 * Displays VAD state, pipeline latency, and audio waveform visualization.
 * Supports minimized/expanded states with localStorage persistence.
 */
import React, { useEffect, useRef, useState } from 'react';
import { useAuralux } from '../hooks/useAuralux';

// ============================================================================
// Styles
// ============================================================================

const styles: Record<string, React.CSSProperties> = {
    container: {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: '8px',
        padding: '16px',
        background: 'rgba(0, 0, 0, 0.8)',
        borderRadius: '16px',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        zIndex: 1000,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        transition: 'all 0.3s ease',
    },
    containerMinimized: {
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        padding: '8px',
        background: 'rgba(0, 0, 0, 0.8)',
        borderRadius: '50%',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(255, 255, 255, 0.1)',
        zIndex: 1000,
        cursor: 'pointer',
        transition: 'all 0.3s ease',
    },
    statusRing: {
        width: '64px',
        height: '64px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        transition: 'all 0.3s ease',
    },
    statusRingMini: {
        width: '40px',
        height: '40px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        transition: 'all 0.3s ease',
    },
    innerCircle: {
        width: '48px',
        height: '48px',
        borderRadius: '50%',
        background: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    innerCircleMini: {
        width: '32px',
        height: '32px',
        borderRadius: '50%',
        background: 'rgba(0, 0, 0, 0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    micIcon: {
        fontSize: '24px',
    },
    micIconMini: {
        fontSize: '16px',
    },
    vadIndicator: {
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        color: 'rgba(255, 255, 255, 0.8)',
        fontSize: '12px',
    },
    stateDot: {
        width: '8px',
        height: '8px',
        borderRadius: '50%',
        transition: 'background 0.2s ease',
    },
    latencyBadge: {
        padding: '4px 8px',
        background: 'rgba(255, 255, 255, 0.1)',
        borderRadius: '4px',
        fontSize: '10px',
        color: 'rgba(255, 255, 255, 0.6)',
        fontFamily: 'monospace',
    },
    waveformCanvas: {
        width: '100px',
        height: '24px',
        borderRadius: '4px',
        background: 'rgba(255, 255, 255, 0.05)',
    },
    controls: {
        display: 'flex',
        gap: '8px',
        marginTop: '8px',
    },
    button: {
        padding: '8px 16px',
        borderRadius: '8px',
        border: 'none',
        cursor: 'pointer',
        fontSize: '12px',
        fontWeight: 600,
        transition: 'all 0.2s ease',
    },
    minimizeButton: {
        position: 'absolute' as const,
        top: '4px',
        right: '4px',
        width: '20px',
        height: '20px',
        borderRadius: '50%',
        border: 'none',
        background: 'rgba(255, 255, 255, 0.1)',
        color: 'rgba(255, 255, 255, 0.6)',
        cursor: 'pointer',
        fontSize: '10px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
    },
    errorText: {
        color: '#ff6b6b',
        fontSize: '10px',
        maxWidth: '120px',
        textAlign: 'center' as const,
    },
};

// ============================================================================
// Helper: Status Ring Colors
// ============================================================================

function getStatusRingStyle(vadState: string, isListening: boolean, mini: boolean): React.CSSProperties {
    const baseStyle = mini ? styles.statusRingMini : styles.statusRing;

    if (!isListening) {
        return {
            ...baseStyle,
            background: 'linear-gradient(135deg, #333 0%, #222 100%)',
            boxShadow: 'none',
        };
    }

    switch (vadState) {
        case 'speech':
            return {
                ...baseStyle,
                background: 'linear-gradient(135deg, #00d4aa 0%, #00a085 100%)',
                boxShadow: '0 0 20px rgba(0, 212, 170, 0.5)',
                animation: 'pulse 1s infinite',
            };
        case 'transition':
            return {
                ...baseStyle,
                background: 'linear-gradient(135deg, #ffd93d 0%, #e6b800 100%)',
                boxShadow: '0 0 15px rgba(255, 217, 61, 0.3)',
            };
        default: // silence
            return {
                ...baseStyle,
                background: 'linear-gradient(135deg, #4a5568 0%, #2d3748 100%)',
                boxShadow: '0 0 10px rgba(74, 85, 104, 0.3)',
            };
    }
}

function getStateDotColor(vadState: string): string {
    switch (vadState) {
        case 'speech': return '#00d4aa';
        case 'transition': return '#ffd93d';
        default: return '#4a5568';
    }
}

// ============================================================================
// Component
// ============================================================================

export interface VoiceHUDProps {
    /** Start in minimized state */
    defaultMinimized?: boolean;
    /** Hide minimize button */
    hideMinimizeButton?: boolean;
}

export function VoiceHUD({
    defaultMinimized = false,
    hideMinimizeButton = false,
}: VoiceHUDProps) {
    const {
        isListening,
        isSpeaking,
        vadState,
        metrics,
        error,
        startPipeline,
        stopPipeline,
    } = useAuralux();

    // Minimized state with localStorage persistence
    const [isMinimized, setIsMinimized] = useState(() => {
        if (typeof window === 'undefined') return defaultMinimized;
        const stored = localStorage.getItem('voicehud-minimized');
        return stored !== null ? stored === 'true' : defaultMinimized;
    });

    const canvasRef = useRef<HTMLCanvasElement>(null);

    // Persist minimized state
    useEffect(() => {
        if (typeof window !== 'undefined') {
            localStorage.setItem('voicehud-minimized', String(isMinimized));
        }
    }, [isMinimized]);

    // Simple waveform visualization
    useEffect(() => {
        if (!canvasRef.current || !isListening || isMinimized) return;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        let animationId: number;
        const draw = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = vadState === 'speech' ? '#00d4aa' : '#4a5568';

            const bars = 20;
            const barWidth = canvas.width / bars;

            for (let i = 0; i < bars; i++) {
                const amplitude = isSpeaking
                    ? Math.random() * 0.8 + 0.2
                    : Math.random() * 0.2;
                const barHeight = canvas.height * amplitude;
                const y = (canvas.height - barHeight) / 2;
                ctx.fillRect(i * barWidth + 1, y, barWidth - 2, barHeight);
            }

            animationId = requestAnimationFrame(draw);
        };

        draw();
        return () => cancelAnimationFrame(animationId);
    }, [isListening, isSpeaking, vadState, isMinimized]);

    // Calculate E2E latency
    const e2eLatency = metrics
        ? ((metrics.ssl?.avgLatencyMs || 0) + (metrics.vocoder?.avgLatencyMs || 0)).toFixed(0)
        : '--';

    // Minimized view
    if (isMinimized) {
        return (
            <div
                style={styles.containerMinimized}
                onClick={() => setIsMinimized(false)}
                title="Click to expand VoiceHUD"
            >
                <div style={getStatusRingStyle(vadState, isListening, true)}>
                    <div style={styles.innerCircleMini}>
                        <span style={styles.micIconMini}>
                            {isListening ? '🎤' : '🎙️'}
                        </span>
                    </div>
                </div>
            </div>
        );
    }

    // Expanded view
    return (
        <div style={{ ...styles.container, position: 'relative' }}>
            {/* Minimize Button */}
            {!hideMinimizeButton && (
                <button
                    style={styles.minimizeButton}
                    onClick={() => setIsMinimized(true)}
                    title="Minimize"
                >
                    −
                </button>
            )}

            {/* Status Ring */}
            <div style={getStatusRingStyle(vadState, isListening, false)}>
                <div style={styles.innerCircle}>
                    <span style={styles.micIcon}>
                        {isListening ? '🎤' : '🎙️'}
                    </span>
                </div>
            </div>

            {/* VAD Indicator */}
            <div style={styles.vadIndicator}>
                <div style={{ ...styles.stateDot, background: getStateDotColor(vadState) }} />
                <span>{vadState.toUpperCase()}</span>
            </div>

            {/* Waveform */}
            {isListening && (
                <canvas
                    ref={canvasRef}
                    width={100}
                    height={24}
                    style={styles.waveformCanvas}
                />
            )}

            {/* Latency Badge */}
            <div style={styles.latencyBadge}>
                E2E: {e2eLatency}ms
            </div>

            {/* Error Display */}
            {error && (
                <div style={styles.errorText}>{error}</div>
            )}

            {/* Controls */}
            <div style={styles.controls}>
                {!isListening ? (
                    <button
                        style={{
                            ...styles.button,
                            background: 'linear-gradient(135deg, #00d4aa 0%, #00a085 100%)',
                            color: '#fff',
                        }}
                        onClick={startPipeline}
                    >
                        Start
                    </button>
                ) : (
                    <button
                        style={{
                            ...styles.button,
                            background: 'linear-gradient(135deg, #ff6b6b 0%, #ee5a5a 100%)',
                            color: '#fff',
                        }}
                        onClick={stopPipeline}
                    >
                        Stop
                    </button>
                )}
            </div>
        </div>
    );
}

export default VoiceHUD;

