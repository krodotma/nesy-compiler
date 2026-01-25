/**
 * AuraluxStudioView.tsx
 * 
 * Phase E Step 41: AuraluxStudio view route
 * PBDESIGN: Premium voice studio combining all Auralux components
 */

import { component$, useSignal, useStore, $, useVisibleTask$ } from '@builder.io/qwik';
import { VadConfidenceOrb } from './VadConfidenceOrb';
import { AudioLevelMeter } from './AudioLevelMeter';
import { VoiceQualitySelector } from './VoiceQualitySelector';
import { SpeechSegmentTimeline } from './SpeechSegmentTimeline';
import { AuraluxDashboard } from './AuraluxDashboard';
import { VadSensitivityControl } from './VadSensitivityControl';
import { ListeningModeOverlay } from './ListeningModeOverlay';
import { SslSpectrogramViewer } from './SslSpectrogramViewer';
import { AuraluxOnboarding } from './AuraluxOnboarding';
import type { AuraluxState } from '../../lib/auralux';

interface AuraluxStudioViewProps {
    emitBus$: (topic: string, data: Record<string, unknown>) => Promise<void>;
}

export const AuraluxStudioView = component$<AuraluxStudioViewProps>(({ emitBus$ }) => {
    // State
    const state = useStore<AuraluxState>({
        mode: 'neural',
        isReady: true,
        isProcessing: false,
        latencyMs: 45,
        error: null,
        modelsLoaded: { vad: true, ssl: true, vocoder: true },
        downloadProgress: 100,
    });

    const isListening = useSignal(false);
    const isSpeaking = useSignal(false);
    const confidence = useSignal(0.7);
    const audioLevel = useSignal(0.5);
    const showFullscreen = useSignal(false);
    const sensitivity = useSignal(0.5);
    const noiseSuppressionEnabled = useSignal(true);
    const pitch = useSignal(1.0);
    const rate = useSignal(1.0);
    const segments = useSignal<any[]>([]);
    const sslFeatures = useSignal<Float32Array | null>(null);
    const showOnboarding = useSignal(false);

    const latency = { vad: 8, ssl: 35, vocoder: 28, total: 71 };

    const toggleListening = $(() => {
        isListening.value = !isListening.value;
        emitBus$('auralux.studio.listening', { active: isListening.value });
    });

    // Keyboard shortcuts - Phase E Step 46
    useVisibleTask$(({ cleanup }) => {
        // Check if first visit
        const hasSeenOnboarding = localStorage.getItem('auralux_onboarding_complete');
        if (!hasSeenOnboarding) {
            showOnboarding.value = true;
        }

        const handleKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement;
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') return;

            switch (event.code) {
                case 'Space':
                    event.preventDefault();
                    isListening.value = !isListening.value;
                    emitBus$('auralux.studio.listening', { active: isListening.value });
                    break;
                case 'Escape':
                    event.preventDefault();
                    if (showFullscreen.value) {
                        showFullscreen.value = false;
                    } else if (isListening.value) {
                        isListening.value = false;
                        emitBus$('auralux.studio.listening', { active: false });
                    }
                    break;
                case 'KeyF':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        showFullscreen.value = !showFullscreen.value;
                    }
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        cleanup(() => window.removeEventListener('keydown', handleKeyDown));
    });

    const completeOnboarding = $(() => {
        showOnboarding.value = false;
        localStorage.setItem('auralux_onboarding_complete', 'true');
    });

    return (
        <div class="min-h-screen bg-gradient-to-br from-black via-slate-950 to-black p-6 space-y-6">
            {/* Step 96: glass-surface-elevated for Header */}
            <div class="glass-surface-elevated rounded-xl p-4 flex items-center justify-between">
                <div>
                    <h1 class="text-2xl font-bold text-white tracking-tight flex items-center gap-3">
                        <span class="text-cyan-400">⚡</span>
                        Auralux Studio
                    </h1>
                    <p class="text-sm text-white/50 mt-1">Neural Voice Synthesis Control Center</p>
                </div>

                {/* Step 97: glass-interactive for toolbar buttons */}
                <div class="flex items-center gap-3">
                    <button
                        onClick$={() => showOnboarding.value = true}
                        class="glass-interactive p-2 text-white/60 hover:text-white rounded-xl glass-transition-standard"
                        title="Show tutorial"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <circle cx="12" cy="12" r="10"></circle>
                            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"></path>
                            <line x1="12" y1="17" x2="12.01" y2="17"></line>
                        </svg>
                    </button>
                    <button
                        onClick$={() => showFullscreen.value = true}
                        class="glass-chip glass-chip-accent-cyan px-4 py-2 text-sm font-medium glass-transition-standard hover:scale-105"
                    >
                        Enter Focus Mode
                    </button>
                </div>
            </div>

            {/* Main Grid */}
            <div class="grid grid-cols-12 gap-6">
                {/* Left Column - Controls */}
                <div class="col-span-4 space-y-4">
                    {/* Step 98: glass-surface for VAD Orb panel */}
                    <div class="glass-surface rounded-2xl p-6 flex flex-col items-center">
                        <VadConfidenceOrb
                            confidence={confidence.value}
                            isListening={isListening.value}
                            isSpeaking={isSpeaking.value}
                        />
                        <button
                            onClick$={toggleListening}
                            class={`mt-6 px-6 py-2.5 rounded-xl font-medium glass-transition-standard ${isListening.value
                                    ? 'glass-chip glass-chip-accent-magenta'
                                    : 'glass-chip glass-chip-accent-cyan'
                                }`}
                        >
                            {isListening.value ? 'Stop Listening' : 'Start Listening'}
                        </button>
                    </div>

                    {/* Step 98: glass-surface for Audio Level panel */}
                    <div class="glass-surface rounded-2xl p-4">
                        <div class="flex items-center justify-between mb-3">
                            <span class="text-xs text-white/60">Input Level</span>
                            <span class="glass-chip glass-chip-accent-cyan text-xs">{(audioLevel.value * 100).toFixed(0)}%</span>
                        </div>
                        <AudioLevelMeter level={audioLevel.value} peak={audioLevel.value * 1.1} />
                    </div>

                    {/* Sensitivity */}
                    <VadSensitivityControl
                        sensitivity={sensitivity.value}
                        noiseSuppressionEnabled={noiseSuppressionEnabled.value}
                        onSensitivityChange$={(v) => sensitivity.value = v}
                        onNoiseSuppressionToggle$={(v) => noiseSuppressionEnabled.value = v}
                    />
                </div>

                {/* Center Column - Visualizations */}
                <div class="col-span-5 space-y-4">
                    {/* Timeline */}
                    <SpeechSegmentTimeline
                        segments={segments.value}
                        isListening={isListening.value}
                    />

                    {/* SSL Spectrogram - Phase D Step 39 */}
                    <SslSpectrogramViewer
                        features={sslFeatures.value}
                        isActive={isListening.value && state.modelsLoaded.ssl}
                    />

                    {/* Pipeline Dashboard */}
                    <AuraluxDashboard state={state} latency={latency} />
                </div>

                {/* Right Column - Quality */}
                <div class="col-span-3 space-y-4">
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
            </div>

            {/* Fullscreen Overlay */}
            <ListeningModeOverlay
                isActive={showFullscreen.value}
                isSpeaking={isSpeaking.value}
                confidence={confidence.value}
                onClose$={() => showFullscreen.value = false}
            />

            {/* Onboarding - Phase E Step 45 */}
            <AuraluxOnboarding
                isOpen={showOnboarding.value}
                onComplete$={completeOnboarding}
                onSkip$={completeOnboarding}
            />

            {/* Keyboard hint */}
            <div class="fixed bottom-4 right-4 text-[10px] text-white/30 space-x-4">
                <span>Space: Toggle</span>
                <span>Esc: Cancel</span>
                <span>Ctrl+F: Focus</span>
            </div>
        </div>
    );
});

export default AuraluxStudioView;
