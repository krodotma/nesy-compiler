/**
 * AvatarLipSync.tsx
 * Example component demonstrating avatar lip-sync integration.
 * Uses useAvatarController with Three.js mesh.
 */
import React, { useRef, useEffect, useState } from 'react';
import { useAvatarController } from '../hooks/useAvatarController';
import { mapPhonemeSequence, PhonemeInput } from '../../auralux/viseme_mapper';
import { MorphTargetMesh } from '../../auralux/avatar_controller';

// ============================================================================
// Props
// ============================================================================

interface AvatarLipSyncProps {
    /** Three.js mesh with morph targets (from useRef) */
    meshRef: React.RefObject<MorphTargetMesh | null>;
    /** Phoneme sequence to animate */
    phonemes?: PhonemeInput[];
    /** Whether animation is playing */
    isPlaying?: boolean;
    /** Auto-play when phonemes change */
    autoPlay?: boolean;
    /** Callback when animation completes */
    onComplete?: () => void;
}

// ============================================================================
// Component
// ============================================================================

export function AvatarLipSync({
    meshRef,
    phonemes = [],
    isPlaying = false,
    autoPlay = true,
    onComplete,
}: AvatarLipSyncProps) {
    const {
        attachMesh,
        loadVisemes,
        play,
        stop,
        state,
        isAttached,
    } = useAvatarController({ autoPlay, onComplete });

    // Attach mesh when available
    useEffect(() => {
        if (meshRef.current && !isAttached) {
            attachMesh(meshRef.current);
        }
    }, [meshRef.current, isAttached, attachMesh]);

    // Load visemes when phonemes change
    useEffect(() => {
        if (phonemes.length > 0) {
            const frames = mapPhonemeSequence(phonemes);
            loadVisemes(frames);
        }
    }, [phonemes, loadVisemes]);

    // Control playback
    useEffect(() => {
        if (isPlaying) {
            play();
        } else {
            stop();
        }
    }, [isPlaying, play, stop]);

    // Debug display (optional - remove in production)
    return (
        <div style={{
            position: 'absolute',
            bottom: 100,
            right: 20,
            padding: 8,
            background: 'rgba(0,0,0,0.5)',
            borderRadius: 8,
            color: '#fff',
            fontSize: 10,
            fontFamily: 'monospace',
        }}>
            <div>Frame: {state.currentFrame}</div>
            <div>Animating: {state.isAnimating ? '▶️' : '⏸️'}</div>
            <div>Blink: {(state.idleState.eyesClosed * 100).toFixed(0)}%</div>
            <div style={{ marginTop: 4 }}>
                {Object.entries(state.visemeWeights)
                    .filter(([_, v]) => v > 0.01)
                    .map(([name, weight]) => (
                        <div key={name} style={{ display: 'flex', gap: 4 }}>
                            <span style={{ width: 24 }}>{name}</span>
                            <div style={{
                                width: 50,
                                height: 8,
                                background: '#333',
                                borderRadius: 4,
                            }}>
                                <div style={{
                                    width: `${weight * 100}%`,
                                    height: '100%',
                                    background: '#00d4aa',
                                    borderRadius: 4,
                                }} />
                            </div>
                        </div>
                    ))}
            </div>
        </div>
    );
}

export default AvatarLipSync;
