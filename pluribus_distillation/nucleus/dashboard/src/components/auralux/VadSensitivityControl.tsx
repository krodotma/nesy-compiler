/**
 * VadSensitivityControl.tsx
 * 
 * Phase C Steps 26, 29: Noise suppression toggle + VAD sensitivity slider
 * PBDESIGN Aesthetic: Minimalist sliders with glow accents
 */

import { component$, useSignal } from '@builder.io/qwik';

interface VadSensitivityControlProps {
    sensitivity: number; // 0-1
    noiseSuppressionEnabled: boolean;
    onSensitivityChange$: (value: number) => void;
    onNoiseSuppressionToggle$: (enabled: boolean) => void;
}

export const VadSensitivityControl = component$<VadSensitivityControlProps>(({
    sensitivity,
    noiseSuppressionEnabled,
    onSensitivityChange$,
    onNoiseSuppressionToggle$
}) => {
    const presets = [
        { label: 'Quiet', value: 0.3, desc: 'Low background noise' },
        { label: 'Normal', value: 0.5, desc: 'Standard environments' },
        { label: 'Noisy', value: 0.7, desc: 'High background noise' },
    ];

    return (
        <div class="bg-black/40 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] p-4 space-y-4">
            {/* Header */}
            <div class="flex items-center justify-between">
                <h3 class="text-sm font-semibold text-white/80">VAD Settings</h3>
            </div>

            {/* Sensitivity Slider */}
            <div class="space-y-3">
                <div class="flex items-center justify-between">
                    <span class="text-xs text-white/60">Detection Sensitivity</span>
                    <span class="text-xs font-mono text-cyan-400">
                        {(sensitivity * 100).toFixed(0)}%
                    </span>
                </div>

                {/* Slider */}
                <div class="relative">
                    <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={sensitivity}
                        onInput$={(e) => onSensitivityChange$(parseFloat((e.target as HTMLInputElement).value))}
                        class="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                        style={{
                            background: `linear-gradient(90deg, rgb(6, 182, 212) ${sensitivity * 100}%, rgba(255,255,255,0.1) ${sensitivity * 100}%)`
                        }}
                    />
                    {/* Scale markers */}
                    <div class="flex justify-between text-[9px] text-white/30 mt-1 px-0.5">
                        <span>Low</span>
                        <span>Med</span>
                        <span>High</span>
                    </div>
                </div>

                {/* Presets */}
                <div class="flex gap-2">
                    {presets.map((preset) => (
                        <button
                            key={preset.label}
                            onClick$={() => onSensitivityChange$(preset.value)}
                            class={`flex-1 py-1.5 px-2 rounded-lg text-xs transition-all ${Math.abs(sensitivity - preset.value) < 0.1
                                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                                    : 'bg-white/5 text-white/50 hover:bg-white/10 hover:text-white/70 border border-transparent'
                                }`}
                        >
                            {preset.label}
                        </button>
                    ))}
                </div>
            </div>

            {/* Divider */}
            <div class="h-px bg-white/10" />

            {/* Noise Suppression Toggle */}
            <div class="flex items-center justify-between">
                <div class="space-y-0.5">
                    <span class="text-sm text-white/80">Noise Suppression</span>
                    <p class="text-[10px] text-white/40">Reduce background noise</p>
                </div>

                {/* Toggle Switch */}
                <button
                    onClick$={() => onNoiseSuppressionToggle$(!noiseSuppressionEnabled)}
                    class={`relative w-12 h-6 rounded-full transition-all duration-300 ${noiseSuppressionEnabled
                            ? 'bg-gradient-to-r from-green-500 to-cyan-500 shadow-[0_0_12px_rgba(34,197,94,0.3)]'
                            : 'bg-white/10'
                        }`}
                >
                    <div
                        class={`absolute top-1 w-4 h-4 rounded-full bg-white shadow-md transition-all duration-300 ${noiseSuppressionEnabled ? 'left-7' : 'left-1'
                            }`}
                    />
                </button>
            </div>

            {/* Active indicator */}
            {noiseSuppressionEnabled && (
                <div class="flex items-center gap-2 px-3 py-2 bg-green-500/10 rounded-lg border border-green-500/20">
                    <div class="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                    <span class="text-xs text-green-400">RNNoise active</span>
                </div>
            )}
        </div>
    );
});

export default VadSensitivityControl;
