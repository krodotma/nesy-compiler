/**
 * useVisemeLipSync.ts
 * Hook for driving avatar morph targets from viseme frames.
 * Provides 60fps animation loop with smooth weight interpolation.
 */
import { useEffect, useRef, useCallback } from 'react';
import { VisemeFrame, getVisemeWeightsAtTime, VISEME_NAMES, VisemeName } from '../../auralux/viseme_mapper';

// ============================================================================
// Types
// ============================================================================

export interface MorphTargetRef {
    setInfluence: (name: string, weight: number) => void;
}

export interface LipSyncOptions {
    /** Morph target controller (e.g., Three.js mesh morphTargetInfluences) */
    morphTargetRef: React.RefObject<MorphTargetRef | null>;
    /** Viseme frames to animate */
    frames: VisemeFrame[];
    /** Whether animation is playing */
    isPlaying: boolean;
    /** Start time offset in ms */
    startTimeMs?: number;
    /** Callback when animation completes */
    onComplete?: () => void;
}

// ============================================================================
// Hook
// ============================================================================

export function useVisemeLipSync({
    morphTargetRef,
    frames,
    isPlaying,
    startTimeMs = 0,
    onComplete,
}: LipSyncOptions) {
    const animationRef = useRef<number | null>(null);
    const startRef = useRef<number>(0);

    // Calculate total duration
    const totalDuration = frames.reduce((max, frame) => {
        return Math.max(max, frame.startMs + frame.durationMs);
    }, 0);

    const animate = useCallback((timestamp: number) => {
        if (!morphTargetRef.current) {
            animationRef.current = requestAnimationFrame(animate);
            return;
        }

        // Initialize start time on first frame
        if (startRef.current === 0) {
            startRef.current = timestamp;
        }

        const elapsed = timestamp - startRef.current + startTimeMs;

        // Check if animation complete
        if (elapsed > totalDuration && totalDuration > 0) {
            // Reset all visemes to 0
            VISEME_NAMES.forEach((name) => {
                morphTargetRef.current?.setInfluence(name, 0);
            });

            animationRef.current = null;
            onComplete?.();
            return;
        }

        // Get current weights
        const weights = getVisemeWeightsAtTime(frames, elapsed);

        // Apply weights to morph targets
        VISEME_NAMES.forEach((name) => {
            const weight = weights[name] || 0;
            morphTargetRef.current?.setInfluence(name, weight);
        });

        // Continue animation
        animationRef.current = requestAnimationFrame(animate);
    }, [frames, morphTargetRef, startTimeMs, totalDuration, onComplete]);

    useEffect(() => {
        if (isPlaying && frames.length > 0) {
            startRef.current = 0; // Reset start time
            animationRef.current = requestAnimationFrame(animate);
        } else {
            // Stop and reset
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
                animationRef.current = null;
            }
            // Reset visemes
            VISEME_NAMES.forEach((name) => {
                morphTargetRef.current?.setInfluence(name, 0);
            });
        }

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [isPlaying, frames, animate, morphTargetRef]);

    return {
        isAnimating: animationRef.current !== null,
        totalDurationMs: totalDuration,
        visemeNames: VISEME_NAMES,
    };
}

// ============================================================================
// Utility: Create Three.js Morph Target Adapter
// ============================================================================

/**
 * Creates a MorphTargetRef adapter for Three.js meshes.
 * Usage: const morphRef = useRef(createThreeMorphAdapter(mesh));
 */
export function createThreeMorphAdapter(
    mesh: { morphTargetDictionary?: Record<string, number>; morphTargetInfluences?: number[] } | null
): MorphTargetRef | null {
    if (!mesh || !mesh.morphTargetDictionary || !mesh.morphTargetInfluences) {
        return null;
    }

    return {
        setInfluence: (name: string, weight: number) => {
            const index = mesh.morphTargetDictionary![name];
            if (index !== undefined) {
                mesh.morphTargetInfluences![index] = weight;
            }
        },
    };
}

export default useVisemeLipSync;
