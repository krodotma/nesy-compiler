/**
 * VoiceQualitySelector.tsx
 * 
 * Phase B Steps 16-17: Quality tier selector + A/B toggle
 * Allows switching between Neural (Vocos) and Browser (speechSynthesis)
 */

import { component$, useSignal, $ } from '@builder.io/qwik';
import type { AuraluxMode } from './VoiceProvider';

interface VoiceQualitySelectorProps {
    mode: AuraluxMode;
    latencyMs: number;
    onModeChange$: (mode: 'neural' | 'browser') => void;
    pitch: number;
    rate: number;
    volume?: number;
    onPitchChange$: (value: number) => void;
    onRateChange$: (value: number) => void;
    onVolumeChange$?: (value: number) => void;
}

export const VoiceQualitySelector = component$<VoiceQualitySelectorProps>((props) => {
    const showAdvanced = useSignal(false);

    const handleModeToggle = $(() => {
        props.onModeChange$(props.mode === 'neural' ? 'browser' : 'neural');
    });

    return (
        <div class="bg-black/40 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] p-4 space-y-4">
            {/* Header */}
            <div class="flex items-center justify-between">
                <h3 class="text-sm font-semibold text-white/80">Voice Engine</h3>
                <span class="text-xs text-cyan-400">{props.latencyMs.toFixed(0)}ms</span>
            </div>

            {/* A/B Toggle: Neural vs Browser */}
            <div class="flex items-center gap-3 p-3 bg-white/5 rounded-xl">
                <button
                    onClick$={handleModeToggle}
                    class={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all duration-300 ${props.mode === 'neural'
                            ? 'bg-gradient-to-r from-green-500/20 to-cyan-500/20 text-green-400 border border-green-500/30'
                            : 'bg-transparent text-white/40 hover:text-white/60'
                        }`}
                >
                    ⚡ Neural (Vocos)
                </button>
                <button
                    onClick$={handleModeToggle}
                    class={`flex-1 py-2 px-4 rounded-lg text-sm font-medium transition-all duration-300 ${props.mode === 'browser'
                            ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                            : 'bg-transparent text-white/40 hover:text-white/60'
                        }`}
                >
                    🔊 Browser
                </button>
            </div>

            {/* Quality Description */}
            <p class="text-xs text-white/50">
                {props.mode === 'neural'
                    ? 'Using Vocos neural vocoder with HuBERT features. High quality, low latency.'
                    : 'Using browser speechSynthesis API. Basic quality, universal support.'}
            </p>

            {/* Advanced Controls Toggle */}
            <button
                onClick$={() => showAdvanced.value = !showAdvanced.value}
                class="text-xs text-cyan-400 hover:text-cyan-300"
            >
                {showAdvanced.value ? '▼ Hide Controls' : '▶ Voice Controls'}
            </button>

            {/* Advanced Sliders */}
            {showAdvanced.value && (
                <div class="space-y-3 pt-2 border-t border-[var(--glass-border)]">
                    {/* Volume Slider */}
                    {props.volume !== undefined && props.onVolumeChange$ && (
                      <div class="space-y-1">
                          <div class="flex justify-between text-xs">
                              <span class="text-white/60">Volume</span>
                              <span class="text-white/40">{Math.round(props.volume * 100)}%</span>
                          </div>
                          <input
                              type="range"
                              min="0"
                              max="1"
                              step="0.05"
                              value={props.volume}
                              onInput$={(e) => props.onVolumeChange$!(parseFloat((e.target as HTMLInputElement).value))}
                              class="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-green-400"
                          />
                      </div>
                    )}

                    {/* Pitch Slider */}
                    <div class="space-y-1">
                        <div class="flex justify-between text-xs">
                            <span class="text-white/60">Pitch</span>
                            <span class="text-white/40">{props.pitch.toFixed(1)}x</span>
                        </div>
                        <input
                            type="range"
                            min="0.5"
                            max="2"
                            step="0.1"
                            value={props.pitch}
                            onInput$={(e) => props.onPitchChange$(parseFloat((e.target as HTMLInputElement).value))}
                            class="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                        />
                    </div>

                    {/* Rate Slider */}
                    <div class="space-y-1">
                        <div class="flex justify-between text-xs">
                            <span class="text-white/60">Speed</span>
                            <span class="text-white/40">{props.rate.toFixed(1)}x</span>
                        </div>
                        <input
                            type="range"
                            min="0.5"
                            max="2"
                            step="0.1"
                            value={props.rate}
                            onInput$={(e) => props.onRateChange$(parseFloat((e.target as HTMLInputElement).value))}
                            class="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                        />
                    </div>
                </div>
            )}
        </div>
    );
});

export default VoiceQualitySelector;
