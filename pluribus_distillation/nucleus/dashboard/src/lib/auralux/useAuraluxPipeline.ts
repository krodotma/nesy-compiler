/**
 * useAuraluxPipeline.ts
 * 
 * Qwik hook for accessing Auralux neural voice pipeline.
 * Phase A Step 3: Create useAuraluxPipeline() Qwik hook
 */

import { useStore, useVisibleTask$, $, type QRL, noSerialize, type NoSerialize } from '@builder.io/qwik';
import { createAuraluxBridge, type AuraluxState, type AuraluxBridge } from './AuraluxBridge';

export interface UseAuraluxPipelineOptions {
    emitBus$: QRL<(topic: string, data: Record<string, unknown>) => Promise<void>>;
    autoInit?: boolean;
}

export interface UseAuraluxPipelineResult {
    state: AuraluxState;
    isNeural: boolean;
    isBrowser: boolean;
    isLoading: boolean;

    // Actions
    initialize$: QRL<() => Promise<void>>;
    synthesize$: QRL<(text: string) => Promise<void>>;
    startListening$: QRL<() => Promise<void>>;
    stopListening$: QRL<() => void>;
    switchToBrowser$: QRL<() => void>;
}

interface InternalState extends AuraluxState {
    bridge: NoSerialize<AuraluxBridge> | undefined;
}

export function useAuraluxPipeline(options: UseAuraluxPipelineOptions): UseAuraluxPipelineResult {
    const state = useStore<InternalState>({
        mode: 'loading',
        isReady: false,
        isProcessing: false,
        latencyMs: 0,
        error: null,
        modelsLoaded: { vad: false, ssl: false, vocoder: false },
        downloadProgress: 0,
        bridge: undefined,
    });

    const initialize$ = $(async () => {
        const emitBus = async (topic: string, data: Record<string, unknown>) => {
            await options.emitBus$(topic, data);
        };

        const bridge = await createAuraluxBridge(emitBus);
        await bridge.initialize();

        // Store bridge and sync state
        state.bridge = noSerialize(bridge);
        state.mode = bridge.state.mode;
        state.isReady = bridge.state.isReady;
        state.isProcessing = bridge.state.isProcessing;
        state.latencyMs = bridge.state.latencyMs;
        state.error = bridge.state.error;
        state.modelsLoaded = bridge.state.modelsLoaded;
        state.downloadProgress = bridge.state.downloadProgress;
    });

    const synthesize$ = $(async (text: string) => {
        if (state.bridge) {
            await state.bridge.synthesize(text);
            state.mode = state.bridge.state.mode;
            state.isProcessing = state.bridge.state.isProcessing;
            state.latencyMs = state.bridge.state.latencyMs;
        }
    });

    const startListening$ = $(async () => {
        if (state.bridge) {
            await state.bridge.startListening();
        }
    });

    const stopListening$ = $(() => {
        if (state.bridge) {
            state.bridge.stopListening();
        }
    });

    const switchToBrowser$ = $(() => {
        if (state.bridge) {
            state.bridge.useBrowserFallback();
            state.mode = state.bridge.state.mode;
        }
    });

    // Auto-init if requested
    useVisibleTask$(({ cleanup }) => {
        if (options.autoInit !== false) {
            initialize$();
        }

        cleanup(() => {
            if (state.bridge) {
                state.bridge.stopListening();
            }
        });
    });

    return {
        state,
        isNeural: state.mode === 'neural',
        isBrowser: state.mode === 'browser',
        isLoading: state.mode === 'loading',
        initialize$,
        synthesize$,
        startListening$,
        stopListening$,
        switchToBrowser$,
    };
}

export default useAuraluxPipeline;
