/**
 * AvatarController.ts
 * WebGL avatar controller for lip-sync animation.
 * Manages morph target blending synchronized with viseme frames.
 *
 * Implements:
 * - 4.4: WebGL morph target manipulation
 * - 4.5: Phoneme-viseme onset synchronization
 * - 4.6: Exponential smoothing for transitions
 * - 4.7: Idle state (eye-blink, micro-expressions)
 */

import { VisemeFrame, getVisemeWeightsAtTime, VISEME_NAMES, VisemeName } from './viseme_mapper';

// ============================================================================
// Types
// ============================================================================

export interface MorphTargetMesh {
    getMorphTargetIndex(name: string): number;
    setMorphTargetInfluence(index: number, weight: number): void;
}

export interface AvatarControllerConfig {
    /** Target animation frame rate */
    targetFps: number;
    /** Exponential smoothing factor (0-1, higher = faster response) */
    smoothingFactor: number;
    /** Enable idle animations (blinks, micro-expressions) */
    enableIdle: boolean;
    /** Average blink interval in ms */
    blinkIntervalMs: number;
    /** Blink duration in ms */
    blinkDurationMs: number;
}

export interface AvatarState {
    isAnimating: boolean;
    currentFrame: number;
    visemeWeights: Record<VisemeName, number>;
    idleState: {
        eyesClosed: number;
        microExpression: number;
    };
}

// ============================================================================
// Default Config
// ============================================================================

const DEFAULT_CONFIG: AvatarControllerConfig = {
    targetFps: 60,
    smoothingFactor: 0.3,
    enableIdle: true,
    blinkIntervalMs: 4000,
    blinkDurationMs: 150,
};

// ============================================================================
// Avatar Controller
// ============================================================================

export class AvatarController {
    private config: AvatarControllerConfig;
    private mesh: MorphTargetMesh | null = null;
    private visemeFrames: VisemeFrame[] = [];
    private playbackStartTime: number = 0;
    private animationHandle: number | null = null;

    // Smoothed weights (current state)
    private smoothedWeights: Record<string, number> = {};

    // Idle animation state
    private lastBlinkTime: number = 0;
    private isBlinking: boolean = false;
    private blinkProgress: number = 0;

    private state: AvatarState = {
        isAnimating: false,
        currentFrame: 0,
        visemeWeights: {} as Record<VisemeName, number>,
        idleState: {
            eyesClosed: 0,
            microExpression: 0,
        },
    };

    constructor(config: Partial<AvatarControllerConfig> = {}) {
        this.config = { ...DEFAULT_CONFIG, ...config };

        // Initialize weights
        for (const viseme of VISEME_NAMES) {
            this.smoothedWeights[viseme] = 0;
            this.state.visemeWeights[viseme] = 0;
        }
    }

    /**
     * Attach the controller to a mesh with morph targets.
     */
    attach(mesh: MorphTargetMesh): void {
        this.mesh = mesh;
        console.log('[AvatarController] Attached to mesh');
    }

    /**
     * Detach from current mesh.
     */
    detach(): void {
        this.stop();
        this.mesh = null;
    }

    /**
     * Load viseme frames for playback.
     */
    loadVisemes(frames: VisemeFrame[]): void {
        this.visemeFrames = frames;
        this.state.currentFrame = 0;
    }

    /**
     * Start synchronized playback.
     */
    play(): void {
        if (this.state.isAnimating) return;

        this.playbackStartTime = performance.now();
        this.state.isAnimating = true;
        this.lastBlinkTime = this.playbackStartTime;
        this.tick();

        console.log('[AvatarController] Playback started');
    }

    /**
     * Stop playback.
     */
    stop(): void {
        this.state.isAnimating = false;
        if (this.animationHandle !== null) {
            cancelAnimationFrame(this.animationHandle);
            this.animationHandle = null;
        }

        // Reset to neutral
        this.resetWeights();
        console.log('[AvatarController] Playback stopped');
    }

    /**
     * Get current animation state.
     */
    getState(): AvatarState {
        return { ...this.state };
    }

    // ========================================================================
    // Animation Loop
    // ========================================================================

    private tick = (): void => {
        if (!this.state.isAnimating) return;

        const now = performance.now();
        const elapsed = now - this.playbackStartTime;

        // Get target weights from viseme frames
        const targetWeights = getVisemeWeightsAtTime(this.visemeFrames, elapsed);

        // Apply exponential smoothing
        this.applySmoothing(targetWeights);

        // Apply idle animations
        if (this.config.enableIdle) {
            this.updateIdle(now);
        }

        // Apply to mesh
        this.applyToMesh();

        // Update state
        this.state.currentFrame++;

        // Schedule next frame
        this.animationHandle = requestAnimationFrame(this.tick);
    };

    // ========================================================================
    // Exponential Smoothing (Phase 4.6)
    // ========================================================================

    private applySmoothing(targetWeights: Record<string, number>): void {
        const alpha = this.config.smoothingFactor;

        for (const viseme of VISEME_NAMES) {
            const target = targetWeights[viseme] ?? 0;
            const current = this.smoothedWeights[viseme] ?? 0;

            // Exponential moving average: new = alpha * target + (1-alpha) * current
            this.smoothedWeights[viseme] = alpha * target + (1 - alpha) * current;
            this.state.visemeWeights[viseme] = this.smoothedWeights[viseme];
        }
    }

    // ========================================================================
    // Idle Animations (Phase 4.7)
    // ========================================================================

    private updateIdle(now: number): void {
        // Eye blink logic
        const timeSinceLastBlink = now - this.lastBlinkTime;

        if (!this.isBlinking && timeSinceLastBlink > this.getNextBlinkInterval()) {
            this.isBlinking = true;
            this.blinkProgress = 0;
        }

        if (this.isBlinking) {
            this.blinkProgress += (1000 / this.config.targetFps) / this.config.blinkDurationMs;

            if (this.blinkProgress >= 1) {
                this.isBlinking = false;
                this.blinkProgress = 0;
                this.lastBlinkTime = now;
                this.state.idleState.eyesClosed = 0;
            } else {
                // Bell curve: closes then opens
                this.state.idleState.eyesClosed = Math.sin(this.blinkProgress * Math.PI);
            }
        }

        // Subtle micro-expressions (random low-amplitude noise)
        this.state.idleState.microExpression = (Math.random() - 0.5) * 0.02;
    }

    private getNextBlinkInterval(): number {
        // Add variance: 50% to 150% of base interval
        const variance = 0.5 + Math.random();
        return this.config.blinkIntervalMs * variance;
    }

    // ========================================================================
    // Mesh Application
    // ========================================================================

    private applyToMesh(): void {
        if (!this.mesh) return;

        for (const viseme of VISEME_NAMES) {
            const index = this.mesh.getMorphTargetIndex(viseme);
            if (index >= 0) {
                this.mesh.setMorphTargetInfluence(index, this.smoothedWeights[viseme] ?? 0);
            }
        }

        // Apply blink to eye close morphs (if available)
        const eyeCloseIndex = this.mesh.getMorphTargetIndex('eyesClosed');
        if (eyeCloseIndex >= 0) {
            this.mesh.setMorphTargetInfluence(eyeCloseIndex, this.state.idleState.eyesClosed);
        }
    }

    private resetWeights(): void {
        for (const viseme of VISEME_NAMES) {
            this.smoothedWeights[viseme] = 0;
            this.state.visemeWeights[viseme] = 0;
        }
        this.state.idleState.eyesClosed = 0;
        this.state.idleState.microExpression = 0;
        this.applyToMesh();
    }

    // ========================================================================
    // Metrics
    // ========================================================================

    getMetrics(): { fps: number; frameCount: number; isAnimating: boolean } {
        return {
            fps: this.config.targetFps,
            frameCount: this.state.currentFrame,
            isAnimating: this.state.isAnimating,
        };
    }
}

export default AvatarController;
