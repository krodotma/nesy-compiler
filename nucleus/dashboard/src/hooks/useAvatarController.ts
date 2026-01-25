/**
 * useAvatarController.ts
 * React hook for integrating AvatarController with component lifecycle.
 */
import { useEffect, useRef, useCallback, useState } from 'react';
import { AvatarController, AvatarControllerConfig, AvatarState, MorphTargetMesh } from '../../auralux/avatar_controller';
import { VisemeFrame } from '../../auralux/viseme_mapper';

// ============================================================================
// Hook Options
// ============================================================================

export interface UseAvatarControllerOptions {
    /** Avatar controller configuration */
    config?: Partial<AvatarControllerConfig>;
    /** Auto-play when visemes are loaded */
    autoPlay?: boolean;
    /** Callback when playback completes */
    onComplete?: () => void;
}

// ============================================================================
// Hook Return Type
// ============================================================================

export interface UseAvatarControllerReturn {
    /** Attach controller to a mesh */
    attachMesh: (mesh: MorphTargetMesh) => void;
    /** Load viseme frames for playback */
    loadVisemes: (frames: VisemeFrame[]) => void;
    /** Start playback */
    play: () => void;
    /** Stop playback */
    stop: () => void;
    /** Current animation state */
    state: AvatarState;
    /** Whether mesh is attached */
    isAttached: boolean;
    /** Controller instance ref */
    controllerRef: React.RefObject<AvatarController>;
}

// ============================================================================
// Hook
// ============================================================================

export function useAvatarController(
    options: UseAvatarControllerOptions = {}
): UseAvatarControllerReturn {
    const { config, autoPlay = false, onComplete } = options;

    const controllerRef = useRef<AvatarController>(new AvatarController(config));
    const [state, setState] = useState<AvatarState>(controllerRef.current.getState());
    const [isAttached, setIsAttached] = useState(false);
    const stateIntervalRef = useRef<number | null>(null);

    // State polling (60fps)
    useEffect(() => {
        stateIntervalRef.current = window.setInterval(() => {
            const newState = controllerRef.current.getState();
            setState(newState);

            // Check for playback completion
            if (!newState.isAnimating && state.isAnimating && onComplete) {
                onComplete();
            }
        }, 16);

        return () => {
            if (stateIntervalRef.current) {
                clearInterval(stateIntervalRef.current);
            }
        };
    }, [onComplete, state.isAnimating]);

    // Cleanup on unmount
    useEffect(() => {
        return () => {
            controllerRef.current.stop();
        };
    }, []);

    const attachMesh = useCallback((mesh: MorphTargetMesh) => {
        controllerRef.current.attach(mesh);
        setIsAttached(true);
    }, []);

    const loadVisemes = useCallback((frames: VisemeFrame[]) => {
        controllerRef.current.loadVisemes(frames);
        if (autoPlay) {
            controllerRef.current.play();
        }
    }, [autoPlay]);

    const play = useCallback(() => {
        controllerRef.current.play();
    }, []);

    const stop = useCallback(() => {
        controllerRef.current.stop();
    }, []);

    return {
        attachMesh,
        loadVisemes,
        play,
        stop,
        state,
        isAttached,
        controllerRef,
    };
}

export default useAvatarController;
