/**
 * SpeechSegmentTimeline.tsx
 * 
 * Phase C Step 23: Speech segment timeline visualization
 * PBDESIGN Aesthetic: Neuro-Mesh Arena with glassmorphism
 * 
 * Shows recent VAD-detected speech segments as glowing bars on a timeline
 */

import { component$, useStore, useVisibleTask$ } from '@builder.io/qwik';

interface SpeechSegment {
    id: string;
    startTime: number;
    endTime: number;
    confidence: number;
}

interface SpeechSegmentTimelineProps {
    segments: SpeechSegment[];
    maxDurationMs?: number;
    isListening: boolean;
}

export const SpeechSegmentTimeline = component$<SpeechSegmentTimelineProps>(({
    segments,
    maxDurationMs = 10000,
    isListening
}) => {
    const now = useStore({ time: Date.now() });

    // Update current time for animation
    useVisibleTask$(({ cleanup }) => {
        const interval = setInterval(() => {
            now.time = Date.now();
        }, 100);
        cleanup(() => clearInterval(interval));
    });

    // Calculate segment position on timeline
    const getSegmentStyle = (segment: SpeechSegment) => {
        const age = now.time - segment.startTime;
        const duration = segment.endTime - segment.startTime;

        // Position from right (0%) to left (100%) based on age
        const rightPercent = (age / maxDurationMs) * 100;
        const widthPercent = (duration / maxDurationMs) * 100;

        // Fade out as segments age
        const opacity = Math.max(0, 1 - (age / maxDurationMs));

        return {
            right: `${rightPercent}%`,
            width: `${Math.max(widthPercent, 2)}%`,
            opacity,
        };
    };

    return (
        <div class="relative h-16 bg-black/40 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] overflow-hidden">
            {/* Background gradient grid */}
            <div
                class="absolute inset-0 opacity-20"
                style={{
                    backgroundImage: `linear-gradient(90deg, rgba(6, 182, 212, 0.1) 1px, transparent 1px)`,
                    backgroundSize: '10% 100%'
                }}
            />

            {/* Timeline label */}
            <div class="absolute top-2 left-3 flex items-center gap-2">
                <div class={`w-2 h-2 rounded-full ${isListening ? 'bg-green-400 animate-pulse' : 'bg-white/30'}`} />
                <span class="text-[10px] font-medium text-white/50 uppercase tracking-wider">
                    Speech Timeline
                </span>
            </div>

            {/* Time markers */}
            <div class="absolute bottom-1.5 left-0 right-0 flex justify-between px-3">
                <span class="text-[9px] text-white/30">-10s</span>
                <span class="text-[9px] text-white/30">-5s</span>
                <span class="text-[9px] text-cyan-400/60">now</span>
            </div>

            {/* Segments container */}
            <div class="absolute inset-x-3 top-8 bottom-4">
                {/* Current time indicator */}
                <div class="absolute right-0 top-0 bottom-0 w-0.5 bg-gradient-to-b from-cyan-400 to-cyan-400/30 rounded-full shadow-[0_0_8px_rgba(6,182,212,0.5)]" />

                {/* Speech segments */}
                {segments.map((segment) => {
                    const style = getSegmentStyle(segment);
                    return (
                        <div
                            key={segment.id}
                            class="absolute top-0 bottom-0 rounded-lg transition-all duration-100"
                            style={{
                                ...style,
                                background: `linear-gradient(90deg, 
                  rgba(34, 197, 94, ${segment.confidence * 0.8}), 
                  rgba(6, 182, 212, ${segment.confidence * 0.6})
                )`,
                                boxShadow: `0 0 ${12 * segment.confidence}px rgba(34, 197, 94, ${segment.confidence * 0.4})`,
                            }}
                        />
                    );
                })}
            </div>

            {/* Empty state */}
            {segments.length === 0 && isListening && (
                <div class="absolute inset-0 flex items-center justify-center">
                    <span class="text-xs text-white/30 italic">Waiting for speech...</span>
                </div>
            )}
        </div>
    );
});

export default SpeechSegmentTimeline;
