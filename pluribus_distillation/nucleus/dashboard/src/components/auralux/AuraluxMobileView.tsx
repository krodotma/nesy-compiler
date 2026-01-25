/**
 * AuraluxMobileView.tsx
 * 
 * Phase E Step 47: Mobile responsive controls
 * Touch-friendly, responsive layout for AuraluxStudio
 */

import { component$, useSignal, $ } from '@builder.io/qwik';
import { VadConfidenceOrb } from './VadConfidenceOrb';
import { VoiceQualitySelector } from './VoiceQualitySelector';
import type { AuraluxState } from '../../lib/auralux';

interface AuraluxMobileViewProps {
    state: AuraluxState;
    isListening: boolean;
    isSpeaking: boolean;
    confidence: number;
    onToggleListening$: () => void;
    emitBus$: (topic: string, data: Record<string, unknown>) => Promise<void>;
}

export const AuraluxMobileView = component$<AuraluxMobileViewProps>(({
    state,
    isListening,
    isSpeaking,
    confidence,
    onToggleListening$,
    emitBus$
}) => {
    const activeTab = useSignal<'voice' | 'settings'>('voice');
    const pitch = useSignal(1.0);
    const rate = useSignal(1.0);

    return (
        <div class="min-h-screen bg-gradient-to-b from-black to-slate-950 flex flex-col">
            {/* Header */}
            <header class="flex items-center justify-between px-4 py-3 border-b border-[var(--glass-border)]">
                <h1 class="text-lg font-semibold text-white flex items-center gap-2">
                    <span class="text-cyan-400">⚡</span>
                    Auralux
                </h1>
                <div class={`px-2 py-1 rounded-full text-xs ${state.isReady ? 'bg-green-500/20 text-green-400' : 'bg-amber-500/20 text-amber-400'
                    }`}>
                    {state.mode === 'neural' ? 'Neural' : 'Browser'}
                </div>
            </header>

            {/* Main content */}
            <main class="flex-1 flex flex-col items-center justify-center p-6 space-y-8">
                {/* Large touch-friendly orb */}
                <div class="relative">
                    <VadConfidenceOrb
                        confidence={confidence}
                        isListening={isListening}
                        isSpeaking={isSpeaking}
                    />
                </div>

                {/* Big listen button */}
                <button
                    onClick$={onToggleListening$}
                    class={`w-full max-w-xs py-4 px-8 rounded-2xl text-lg font-semibold transition-all active:scale-95 touch-manipulation ${isListening
                            ? 'bg-red-500 text-white shadow-lg shadow-red-500/30'
                            : 'bg-gradient-to-r from-cyan-500 to-blue-500 text-white shadow-lg shadow-cyan-500/30'
                        }`}
                >
                    {isListening ? 'Stop Listening' : 'Start Listening'}
                </button>

                {/* Status text */}
                <p class="text-sm text-white/50 text-center">
                    {isListening
                        ? isSpeaking
                            ? 'Hearing you clearly...'
                            : 'Listening for speech...'
                        : 'Tap to begin'}
                </p>
            </main>

            {/* Tab bar */}
            <nav class="border-t border-[var(--glass-border)] bg-black/50 backdrop-blur-xl">
                <div class="flex">
                    <button
                        onClick$={() => activeTab.value = 'voice'}
                        class={`flex-1 py-4 text-center text-sm font-medium transition-colors ${activeTab.value === 'voice'
                                ? 'text-cyan-400 border-t-2 border-cyan-400 -mt-px'
                                : 'text-white/50'
                            }`}
                    >
                        🎤 Voice
                    </button>
                    <button
                        onClick$={() => activeTab.value = 'settings'}
                        class={`flex-1 py-4 text-center text-sm font-medium transition-colors ${activeTab.value === 'settings'
                                ? 'text-cyan-400 border-t-2 border-cyan-400 -mt-px'
                                : 'text-white/50'
                            }`}
                    >
                        ⚙️ Settings
                    </button>
                </div>
            </nav>

            {/* Settings panel (slide up) */}
            {activeTab.value === 'settings' && (
                <div class="fixed inset-x-0 bottom-16 bg-black/95 backdrop-blur-xl border-t border-[var(--glass-border)] p-4 rounded-t-3xl animate-slideUp">
                    <VoiceQualitySelector
                        mode={state.mode}
                        latencyMs={state.latencyMs}
                        pitch={pitch.value}
                        rate={rate.value}
                        onModeChange$={(m) => state.mode = m}
                        onPitchChange$={(v) => pitch.value = v}
                        onRateChange$={(v) => rate.value = v}
                    />
                </div>
            )}
        </div>
    );
});

export default AuraluxMobileView;
