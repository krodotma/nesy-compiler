/**
 * MicPermissionModal.tsx
 * 
 * Phase C Step 28: Microphone permission flow with styled modal
 * PBDESIGN Aesthetic: Premium modal with subtle animations
 */

import { component$, $ } from '@builder.io/qwik';

interface MicPermissionModalProps {
    isOpen: boolean;
    status: 'prompt' | 'requesting' | 'granted' | 'denied';
    onRequestPermission$: () => Promise<void>;
    onClose$: () => void;
}

export const MicPermissionModal = component$<MicPermissionModalProps>(({
    isOpen,
    status,
    onRequestPermission$,
    onClose$
}) => {
    if (!isOpen) return null;

    return (
        <div class="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                class="absolute inset-0 bg-black/80 backdrop-blur-sm"
                onClick$={onClose$}
            />

            {/* Modal */}
            <div class="relative z-10 w-full max-w-md mx-4 bg-gradient-to-br from-black/90 to-black/80 backdrop-blur-xl rounded-3xl border border-[var(--glass-border)] overflow-hidden shadow-2xl">
                {/* Glow effect */}
                <div class="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-purple-500/5 pointer-events-none" />

                {/* Content */}
                <div class="relative p-8 space-y-6">
                    {/* Icon */}
                    <div class="flex justify-center">
                        <div class={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-500 ${status === 'granted'
                                ? 'bg-green-500/20 ring-2 ring-green-500/50'
                                : status === 'denied'
                                    ? 'bg-red-500/20 ring-2 ring-red-500/50'
                                    : 'bg-cyan-500/20 ring-2 ring-cyan-500/30'
                            }`}>
                            {status === 'granted' ? (
                                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-green-400">
                                    <polyline points="20 6 9 17 4 12" />
                                </svg>
                            ) : status === 'denied' ? (
                                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-red-400">
                                    <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            ) : (
                                <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-cyan-400">
                                    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                                    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                                    <line x1="12" y1="19" x2="12" y2="23" />
                                    <line x1="8" y1="23" x2="16" y2="23" />
                                </svg>
                            )}
                        </div>
                    </div>

                    {/* Title & Description */}
                    <div class="text-center space-y-2">
                        <h2 class="text-xl font-semibold text-white">
                            {status === 'granted' ? 'Microphone Enabled'
                                : status === 'denied' ? 'Permission Denied'
                                    : 'Enable Microphone'}
                        </h2>
                        <p class="text-sm text-white/60 leading-relaxed">
                            {status === 'granted'
                                ? 'Your microphone is ready. Auralux can now listen for your voice.'
                                : status === 'denied'
                                    ? 'Please enable microphone access in your browser settings to use voice features.'
                                    : 'Auralux needs microphone access to enable voice input and VAD detection.'}
                        </p>
                    </div>

                    {/* Actions */}
                    <div class="flex flex-col gap-3">
                        {status === 'prompt' && (
                            <button
                                onClick$={onRequestPermission$}
                                class="w-full py-3 px-6 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-medium rounded-xl transition-all duration-300 shadow-lg shadow-cyan-500/20 hover:shadow-cyan-500/30"
                            >
                                Allow Microphone Access
                            </button>
                        )}
                        {status === 'requesting' && (
                            <div class="w-full py-3 px-6 bg-white/5 text-white/60 font-medium rounded-xl text-center flex items-center justify-center gap-2">
                                <div class="w-4 h-4 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                                Requesting access...
                            </div>
                        )}
                        <button
                            onClick$={onClose$}
                            class="w-full py-3 px-6 bg-white/5 hover:bg-white/10 text-white/70 hover:text-white font-medium rounded-xl transition-all duration-300 border border-[var(--glass-border)]"
                        >
                            {status === 'granted' ? 'Continue' : 'Cancel'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
});

export default MicPermissionModal;
