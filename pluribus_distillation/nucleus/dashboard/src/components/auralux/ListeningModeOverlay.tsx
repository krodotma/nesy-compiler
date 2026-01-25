/**
 * ListeningModeOverlay.tsx
 * 
 * Phase C Step 27: Full-screen "Listening Mode" overlay
 * PBDESIGN Aesthetic: Immersive neural ambient with radial glow
 * 
 * Creates an immersive listening experience with focus on the VAD orb
 */

import { component$, $ } from '@builder.io/qwik';

interface ListeningModeOverlayProps {
    isActive: boolean;
    isSpeaking: boolean;
    confidence: number;
    onClose$: () => void;
}

export const ListeningModeOverlay = component$<ListeningModeOverlayProps>(({
    isActive,
    isSpeaking,
    confidence,
    onClose$
}) => {
    if (!isActive) return null;

    const handleKeyDown = $((event: KeyboardEvent) => {
        if (event.key === 'Escape') {
            onClose$();
        }
    });

    return (
        <div
            class="fixed inset-0 z-[100] flex flex-col items-center justify-center"
            onKeyDown$={handleKeyDown}
            tabIndex={0}
        >
            {/* Backdrop with gradient */}
            <div
                class="absolute inset-0 transition-all duration-500"
                style={{
                    background: isSpeaking
                        ? `radial-gradient(ellipse at center, 
                rgba(34, 197, 94, 0.15) 0%, 
                rgba(0, 0, 0, 0.95) 50%,
                rgba(0, 0, 0, 0.98) 100%)`
                        : `radial-gradient(ellipse at center, 
                rgba(6, 182, 212, 0.1) 0%, 
                rgba(0, 0, 0, 0.95) 50%,
                rgba(0, 0, 0, 0.98) 100%)`
                }}
            />

            {/* Ambient particles (optional decorative) */}
            <div class="absolute inset-0 overflow-hidden pointer-events-none">
                {[...Array(20)].map((_, i) => (
                    <div
                        key={i}
                        class="absolute w-1 h-1 rounded-full bg-cyan-400/20 animate-pulse"
                        style={{
                            left: `${Math.random() * 100}%`,
                            top: `${Math.random() * 100}%`,
                            animationDelay: `${Math.random() * 2}s`,
                            animationDuration: `${2 + Math.random() * 3}s`,
                        }}
                    />
                ))}
            </div>

            {/* Close button */}
            <button
                onClick$={onClose$}
                class="absolute top-8 right-8 w-12 h-12 rounded-full bg-white/5 border border-[var(--glass-border)] flex items-center justify-center text-white/60 hover:text-white hover:bg-white/10 transition-all"
            >
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
            </button>

            {/* Main content */}
            <div class="relative z-10 flex flex-col items-center gap-8">
                {/* Giant VAD Orb */}
                <div class="relative">
                    {/* Outer glow rings */}
                    {isSpeaking && (
                        <>
                            <div
                                class="absolute inset-0 rounded-full animate-ping"
                                style={{
                                    width: '200px',
                                    height: '200px',
                                    marginLeft: '-100px',
                                    marginTop: '-100px',
                                    left: '50%',
                                    top: '50%',
                                    background: `rgba(34, 197, 94, ${confidence * 0.2})`,
                                    animationDuration: '2s',
                                }}
                            />
                            <div
                                class="absolute rounded-full"
                                style={{
                                    width: '240px',
                                    height: '240px',
                                    marginLeft: '-120px',
                                    marginTop: '-120px',
                                    left: '50%',
                                    top: '50%',
                                    background: `radial-gradient(circle, rgba(34, 197, 94, ${confidence * 0.15}) 0%, transparent 70%)`,
                                }}
                            />
                        </>
                    )}

                    {/* Core orb */}
                    <div
                        class="relative w-32 h-32 rounded-full flex items-center justify-center transition-all duration-300"
                        style={{
                            background: isSpeaking
                                ? `linear-gradient(135deg, rgba(34, 197, 94, 0.9), rgba(6, 182, 212, 0.7))`
                                : `linear-gradient(135deg, rgba(6, 182, 212, 0.4), rgba(99, 102, 241, 0.3))`,
                            boxShadow: isSpeaking
                                ? `0 0 60px rgba(34, 197, 94, ${confidence * 0.6}), 0 0 120px rgba(34, 197, 94, ${confidence * 0.3}), inset 0 0 30px rgba(255, 255, 255, 0.1)`
                                : `0 0 40px rgba(6, 182, 212, 0.2), inset 0 0 20px rgba(255, 255, 255, 0.05)`,
                            transform: `scale(${1 + confidence * 0.1})`,
                        }}
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" class={`transition-all ${isSpeaking ? 'text-white' : 'text-white/60'}`}>
                            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                            <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                            <line x1="12" y1="19" x2="12" y2="23" />
                            <line x1="8" y1="23" x2="16" y2="23" />
                        </svg>
                    </div>
                </div>

                {/* Status text */}
                <div class="text-center space-y-2">
                    <h2 class={`text-2xl font-light tracking-wide transition-colors ${isSpeaking ? 'text-green-400' : 'text-white/60'
                        }`}>
                        {isSpeaking ? 'Listening...' : 'Speak now'}
                    </h2>
                    <p class="text-sm text-white/40">
                        Press <kbd class="px-1.5 py-0.5 bg-white/10 rounded text-xs">ESC</kbd> to exit
                    </p>
                </div>

                {/* Confidence meter */}
                <div class="w-64 space-y-2">
                    <div class="flex justify-between text-xs text-white/40">
                        <span>Confidence</span>
                        <span class={isSpeaking ? 'text-green-400' : 'text-white/40'}>
                            {(confidence * 100).toFixed(0)}%
                        </span>
                    </div>
                    <div class="h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                            class="h-full rounded-full transition-all duration-150"
                            style={{
                                width: `${confidence * 100}%`,
                                background: isSpeaking
                                    ? 'linear-gradient(90deg, rgb(34, 197, 94), rgb(6, 182, 212))'
                                    : 'rgba(255, 255, 255, 0.3)',
                            }}
                        />
                    </div>
                </div>
            </div>
        </div>
    );
});

export default ListeningModeOverlay;
