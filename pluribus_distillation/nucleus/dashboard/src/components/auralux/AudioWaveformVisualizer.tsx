/**
 * AudioWaveformVisualizer.tsx
 * 
 * Phase B Step 13: Waveform visualizer for synthesized audio
 * Displays real-time frequency bars during voice synthesis
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';

interface AudioWaveformVisualizerProps {
    isActive: boolean;
    audioLevel?: number; // 0-1 normalized level
    barCount?: number;
}

export const AudioWaveformVisualizer = component$<AudioWaveformVisualizerProps>(({
    isActive,
    audioLevel = 0.5,
    barCount = 12
}) => {
    const bars = useSignal<number[]>(Array(barCount).fill(0));

    // Animate bars when active
    useVisibleTask$(({ track, cleanup }) => {
        track(() => isActive);
        track(() => audioLevel);

        if (!isActive) {
            bars.value = Array(barCount).fill(0);
            return;
        }

        const interval = setInterval(() => {
            bars.value = Array(barCount).fill(0).map(() => {
                const base = audioLevel * 0.7;
                const variance = Math.random() * 0.3;
                return Math.min(base + variance, 1);
            });
        }, 50);

        cleanup(() => clearInterval(interval));
    });

    return (
        <div class="flex items-end justify-center gap-1 h-8 px-2">
            {bars.value.map((height, i) => (
                <div
                    key={i}
                    class="w-1 rounded-full transition-all duration-75"
                    style={{
                        height: `${height * 100}%`,
                        minHeight: isActive ? '4px' : '2px',
                        background: isActive
                            ? `linear-gradient(to top, rgba(34, 197, 94, 0.8), rgba(6, 182, 212, 0.6))`
                            : 'rgba(255, 255, 255, 0.2)',
                        boxShadow: isActive && height > 0.5
                            ? '0 0 4px rgba(34, 197, 94, 0.5)'
                            : 'none',
                    }}
                />
            ))}
        </div>
    );
});

export default AudioWaveformVisualizer;
