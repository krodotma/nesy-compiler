/**
 * VadConfidenceOrb.tsx - 200% ENHANCED VERSION
 * 
 * Premium features:
 * - Particle effects around orb when speaking
 * - Breathing animation when idle
 * - Multi-layer glow with depth
 * - Smooth 60fps transitions
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';

interface VadConfidenceOrbProps {
    confidence: number; // 0-1
    isListening: boolean;
    isSpeaking: boolean;
}

interface Particle {
    id: number;
    x: number;
    y: number;
    vx: number;
    vy: number;
    life: number;
    size: number;
}

export const VadConfidenceOrb = component$<VadConfidenceOrbProps>(({ confidence, isListening, isSpeaking }) => {
    const particles = useSignal<Particle[]>([]);
    const breathScale = useSignal(1);

    // Particle system for speaking state
    useVisibleTask$(({ track, cleanup }) => {
        track(() => isSpeaking);
        track(() => confidence);

        if (!isSpeaking) {
            particles.value = [];
            return;
        }

        const interval = setInterval(() => {
            // Spawn new particles
            const newParticles: Particle[] = [];
            for (let i = 0; i < 3; i++) {
                const angle = Math.random() * Math.PI * 2;
                newParticles.push({
                    id: Date.now() + i,
                    x: Math.cos(angle) * 30,
                    y: Math.sin(angle) * 30,
                    vx: Math.cos(angle) * (2 + Math.random() * 2),
                    vy: Math.sin(angle) * (2 + Math.random() * 2),
                    life: 1,
                    size: 2 + Math.random() * 3,
                });
            }

            // Update and filter
            particles.value = [...particles.value, ...newParticles]
                .map(p => ({
                    ...p,
                    x: p.x + p.vx,
                    y: p.y + p.vy,
                    life: p.life - 0.05,
                }))
                .filter(p => p.life > 0)
                .slice(-50); // Max 50 particles
        }, 50);

        cleanup(() => clearInterval(interval));
    });

    // Breathing animation when idle
    useVisibleTask$(({ track, cleanup }) => {
        track(() => isListening);
        track(() => isSpeaking);

        if (!isListening || isSpeaking) {
            breathScale.value = 1;
            return;
        }

        let frame = 0;
        const animate = () => {
            frame++;
            breathScale.value = 1 + Math.sin(frame * 0.03) * 0.05;
            requestAnimationFrame(animate);
        };
        const id = requestAnimationFrame(animate);
        cleanup(() => cancelAnimationFrame(id));
    });

    const glowIntensity = Math.min(confidence * 1.5, 1);
    const orbSize = 48 + (confidence * 16);

    return (
        <div class="relative flex items-center justify-center" style={{ width: '120px', height: '120px' }}>
            {/* Particles */}
            {particles.value.map((p) => (
                <div
                    key={p.id}
                    class="absolute rounded-full pointer-events-none"
                    style={{
                        left: `calc(50% + ${p.x}px)`,
                        top: `calc(50% + ${p.y}px)`,
                        width: `${p.size}px`,
                        height: `${p.size}px`,
                        opacity: p.life,
                        background: `radial-gradient(circle, rgba(34, 197, 94, ${p.life}), transparent)`,
                        transform: 'translate(-50%, -50%)',
                    }}
                />
            ))}

            {/* Outer ambient glow */}
            <div
                class="absolute rounded-full transition-all duration-500"
                style={{
                    width: `${orbSize + 40}px`,
                    height: `${orbSize + 40}px`,
                    background: isSpeaking
                        ? `radial-gradient(circle, rgba(34, 197, 94, ${glowIntensity * 0.25}) 0%, transparent 70%)`
                        : `radial-gradient(circle, rgba(6, 182, 212, ${glowIntensity * 0.15}) 0%, transparent 70%)`,
                }}
            />

            {/* Middle pulse ring */}
            {isSpeaking && (
                <div
                    class="absolute rounded-full animate-ping"
                    style={{
                        width: `${orbSize + 20}px`,
                        height: `${orbSize + 20}px`,
                        backgroundColor: `rgba(34, 197, 94, ${glowIntensity * 0.3})`,
                        animationDuration: '1.5s',
                    }}
                />
            )}

            {/* Inner pulse ring */}
            {isSpeaking && (
                <div
                    class="absolute rounded-full"
                    style={{
                        width: `${orbSize + 10}px`,
                        height: `${orbSize + 10}px`,
                        background: `conic-gradient(from 0deg, transparent, rgba(34, 197, 94, ${glowIntensity * 0.4}), transparent)`,
                        animation: 'spin 2s linear infinite',
                    }}
                />
            )}

            {/* Core orb */}
            <div
                class="relative rounded-full flex items-center justify-center transition-all duration-200"
                style={{
                    width: `${orbSize}px`,
                    height: `${orbSize}px`,
                    transform: `scale(${breathScale.value})`,
                    background: isSpeaking
                        ? `linear-gradient(135deg, rgba(34, 197, 94, 0.9), rgba(6, 182, 212, 0.7))`
                        : isListening
                            ? `linear-gradient(135deg, rgba(6, 182, 212, 0.6), rgba(99, 102, 241, 0.4))`
                            : `linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05))`,
                    boxShadow: isSpeaking
                        ? `0 0 ${40 * glowIntensity}px rgba(34, 197, 94, ${glowIntensity * 0.5}), 
               0 0 ${80 * glowIntensity}px rgba(34, 197, 94, ${glowIntensity * 0.3}), 
               inset 0 -10px 30px rgba(0, 0, 0, 0.2),
               inset 0 10px 30px rgba(255, 255, 255, 0.1)`
                        : isListening
                            ? `0 0 ${30 * glowIntensity}px rgba(6, 182, 212, ${glowIntensity * 0.3}), 
                 inset 0 0 20px rgba(255, 255, 255, 0.05)`
                            : `inset 0 0 10px rgba(255, 255, 255, 0.05)`,
                    border: `2px solid ${isSpeaking ? 'rgba(34, 197, 94, 0.6)' : isListening ? 'rgba(6, 182, 212, 0.4)' : 'rgba(255, 255, 255, 0.2)'}`,
                }}
            >
                {/* Glass reflection */}
                <div
                    class="absolute top-1 left-1/4 w-1/2 h-1/4 rounded-full opacity-30"
                    style={{
                        background: 'linear-gradient(to bottom, rgba(255,255,255,0.4), transparent)'
                    }}
                />

                {/* Microphone Icon */}
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    stroke-width="2"
                    stroke-linecap="round"
                    stroke-linejoin="round"
                    class={`transition-all duration-300 ${isSpeaking ? 'text-white scale-110'
                            : isListening ? 'text-cyan-300'
                                : 'text-white/40'
                        }`}
                    style={{
                        filter: isSpeaking ? 'drop-shadow(0 0 8px rgba(255,255,255,0.5))' : 'none'
                    }}
                >
                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                    <line x1="12" y1="19" x2="12" y2="23" />
                    <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
            </div>

            {/* Confidence label */}
            {isListening && (
                <div class="absolute -bottom-8 text-xs font-mono text-white/50 flex items-center gap-1">
                    <div class={`w-1.5 h-1.5 rounded-full ${isSpeaking ? 'bg-green-400 animate-pulse' : 'bg-cyan-400'}`} />
                    {(confidence * 100).toFixed(0)}%
                </div>
            )}

            {/* Spin animation style */}
            <style>
                {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
            </style>
        </div>
    );
});

export default VadConfidenceOrb;
