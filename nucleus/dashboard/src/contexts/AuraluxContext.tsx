/**
 * AuraluxContext.tsx
 * React context for Auralux voice pipeline state management.
 * Provides pipeline state, metrics, and controls to all child components.
 */
import React, { createContext, useContext, useReducer, useCallback, useRef, ReactNode } from 'react';
import { PipelineOrchestrator, PipelineConfig } from '../../auralux/pipeline_orchestrator';
import { VADEvent } from '../../auralux/vad_service';

// ============================================================================
// Types
// ============================================================================

export type VADState = 'silence' | 'speech' | 'transition';

export interface PipelineMetrics {
    ssl: {
        lastInferenceMs: number;
        totalCalls: number;
        avgLatencyMs: number;
    } | null;
    vocoder: {
        lastSynthesisMs: number;
        totalSynthesisCalls: number;
        avgLatencyMs: number;
    } | null;
    mixer: {
        overflowCount: number;
        underflowCount: number;
        bufferedSamples: number;
    } | null;
}

export interface AuraluxState {
    isListening: boolean;
    isSpeaking: boolean;
    vadState: VADState;
    metrics: PipelineMetrics | null;
    error: string | null;
    lastVadEvent: VADEvent | null;
}

export interface AuraluxContextValue extends AuraluxState {
    startPipeline: () => Promise<void>;
    stopPipeline: () => Promise<void>;
    refreshMetrics: () => void;
}

// ============================================================================
// Reducer
// ============================================================================

type AuraluxAction =
    | { type: 'START_LISTENING' }
    | { type: 'STOP_LISTENING' }
    | { type: 'VAD_EVENT'; event: VADEvent }
    | { type: 'UPDATE_METRICS'; metrics: PipelineMetrics }
    | { type: 'SET_ERROR'; error: string }
    | { type: 'CLEAR_ERROR' };

const initialState: AuraluxState = {
    isListening: false,
    isSpeaking: false,
    vadState: 'silence',
    metrics: null,
    error: null,
    lastVadEvent: null,
};

function auraluxReducer(state: AuraluxState, action: AuraluxAction): AuraluxState {
    switch (action.type) {
        case 'START_LISTENING':
            return { ...state, isListening: true, error: null };
        case 'STOP_LISTENING':
            return { ...state, isListening: false, isSpeaking: false, vadState: 'silence' };
        case 'VAD_EVENT':
            const vadState: VADState =
                action.event.type === 'speech_start' ? 'speech' :
                    action.event.type === 'speech_end' ? 'silence' : 'transition';
            return {
                ...state,
                vadState,
                isSpeaking: action.event.type === 'speech_start',
                lastVadEvent: action.event,
            };
        case 'UPDATE_METRICS':
            return { ...state, metrics: action.metrics };
        case 'SET_ERROR':
            return { ...state, error: action.error, isListening: false };
        case 'CLEAR_ERROR':
            return { ...state, error: null };
        default:
            return state;
    }
}

// ============================================================================
// Context
// ============================================================================

const AuraluxContext = createContext<AuraluxContextValue | null>(null);

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: PipelineConfig = {
    vadModelUrl: '/models/silero_vad_v5.onnx',
    sslModelUrl: '/models/hubert-soft-quantized.onnx',
    vocoderModelUrl: '/models/vocos_q8.onnx',
    workletUrl: '/worklets/auralux-processor.js',
};

// ============================================================================
// Provider
// ============================================================================

interface AuraluxProviderProps {
    children: ReactNode;
    config?: Partial<PipelineConfig>;
}

export function AuraluxProvider({ children, config }: AuraluxProviderProps) {
    const [state, dispatch] = useReducer(auraluxReducer, initialState);
    const orchestratorRef = useRef<PipelineOrchestrator | null>(null);
    const metricsIntervalRef = useRef<number | null>(null);

    const fullConfig: PipelineConfig = { ...DEFAULT_CONFIG, ...config };

    const startPipeline = useCallback(async () => {
        if (orchestratorRef.current) return;

        try {
            const orchestrator = new PipelineOrchestrator(fullConfig);
            orchestratorRef.current = orchestrator;

            await orchestrator.start(
                // VAD callback
                (event: VADEvent) => {
                    dispatch({ type: 'VAD_EVENT', event });
                },
                // Feature callback (optional logging)
                (_features: Float32Array) => {
                    // Could dispatch feature event for visualization
                }
            );

            dispatch({ type: 'START_LISTENING' });

            // Start metrics polling
            metricsIntervalRef.current = window.setInterval(() => {
                const metrics = orchestratorRef.current?.getMetrics();
                if (metrics) {
                    dispatch({ type: 'UPDATE_METRICS', metrics: metrics as PipelineMetrics });
                }
            }, 1000);

        } catch (e) {
            dispatch({ type: 'SET_ERROR', error: e instanceof Error ? e.message : 'Pipeline start failed' });
            orchestratorRef.current = null;
        }
    }, [fullConfig]);

    const stopPipeline = useCallback(async () => {
        if (!orchestratorRef.current) return;

        if (metricsIntervalRef.current) {
            clearInterval(metricsIntervalRef.current);
            metricsIntervalRef.current = null;
        }

        await orchestratorRef.current.stop();
        orchestratorRef.current = null;
        dispatch({ type: 'STOP_LISTENING' });
    }, []);

    const refreshMetrics = useCallback(() => {
        const metrics = orchestratorRef.current?.getMetrics();
        if (metrics) {
            dispatch({ type: 'UPDATE_METRICS', metrics: metrics as PipelineMetrics });
        }
    }, []);

    const value: AuraluxContextValue = {
        ...state,
        startPipeline,
        stopPipeline,
        refreshMetrics,
    };

    return (
        <AuraluxContext.Provider value={value}>
            {children}
        </AuraluxContext.Provider>
    );
}

// ============================================================================
// Hook
// ============================================================================

export function useAuralux(): AuraluxContextValue {
    const context = useContext(AuraluxContext);
    if (!context) {
        throw new Error('useAuralux must be used within an AuraluxProvider');
    }
    return context;
}

export { AuraluxContext };
