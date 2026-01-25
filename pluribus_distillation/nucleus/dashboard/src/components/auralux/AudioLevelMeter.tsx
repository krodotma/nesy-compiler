/**
 * AudioLevelMeter.tsx
 * 
 * Phase C Step 25: Audio level meter (input gain visualization)
 * Shows microphone input level with color-coded gradient
 */

import { component$ } from '@builder.io/qwik';

interface AudioLevelMeterProps {
    level: number; // 0-1 normalized
    peak?: number; // 0-1 peak hold
    orientation?: 'horizontal' | 'vertical';
    showPeak?: boolean;
}

export const AudioLevelMeter = component$<AudioLevelMeterProps>(({
    level,
    peak = 0,
    orientation = 'horizontal',
    showPeak = true
}) => {
    // Color based on level (green → yellow → red)
    const getColor = (l: number) => {
        if (l < 0.6) return 'rgb(34, 197, 94)'; // Green
        if (l < 0.85) return 'rgb(251, 191, 36)'; // Yellow/Amber
        return 'rgb(239, 68, 68)'; // Red
    };

    const isVertical = orientation === 'vertical';
    const clampedLevel = Math.min(Math.max(level, 0), 1);
    const clampedPeak = Math.min(Math.max(peak, 0), 1);

    return (
        <div
            class={`relative rounded-full overflow-hidden bg-white/10 ${isVertical ? 'w-3 h-24' : 'w-full h-3'
                }`}
        >
            {/* Background gradient segments */}
            <div
                class="absolute inset-0 opacity-30"
                style={{
                    background: isVertical
                        ? 'linear-gradient(to top, rgb(34, 197, 94), rgb(251, 191, 36), rgb(239, 68, 68))'
                        : 'linear-gradient(to right, rgb(34, 197, 94), rgb(251, 191, 36), rgb(239, 68, 68))'
                }}
            />

            {/* Active level fill */}
            <div
                class="absolute transition-all duration-75 rounded-full"
                style={{
                    background: getColor(clampedLevel),
                    boxShadow: `0 0 8px ${getColor(clampedLevel)}`,
                    ...(isVertical
                        ? { bottom: 0, left: 0, right: 0, height: `${clampedLevel * 100}%` }
                        : { top: 0, bottom: 0, left: 0, width: `${clampedLevel * 100}%` }
                    )
                }}
            />

            {/* Peak hold indicator */}
            {showPeak && clampedPeak > 0 && (
                <div
                    class="absolute transition-all duration-300"
                    style={{
                        background: getColor(clampedPeak),
                        ...(isVertical
                            ? { left: 0, right: 0, bottom: `${clampedPeak * 100}%`, height: '2px' }
                            : { top: 0, bottom: 0, left: `${clampedPeak * 100}%`, width: '2px' }
                        )
                    }}
                />
            )}
        </div>
    );
});

export default AudioLevelMeter;
