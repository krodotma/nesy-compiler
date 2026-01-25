/**
 * AuraluxOnboarding.tsx
 * 
 * Phase E Step 45: Onboarding tutorial overlay
 * Guides new users through Auralux features
 */

import { component$, useSignal, $ } from '@builder.io/qwik';

interface AuraluxOnboardingProps {
    isOpen: boolean;
    onComplete$: () => void;
    onSkip$: () => void;
}

const steps = [
    {
        title: 'Welcome to Auralux',
        description: 'Neural voice synthesis powered by state-of-the-art AI models.',
        icon: '⚡',
    },
    {
        title: 'Voice Detection',
        description: 'The glowing orb responds to your voice in real-time using Silero VAD.',
        icon: '🎤',
    },
    {
        title: 'Neural Quality',
        description: 'Toggle between neural (Vocos) and browser TTS for quality comparison.',
        icon: '🧠',
    },
    {
        title: 'Keyboard Shortcuts',
        description: 'Press Space to toggle listening, Esc to cancel, Ctrl+F for focus mode.',
        icon: '⌨️',
    },
];

export const AuraluxOnboarding = component$<AuraluxOnboardingProps>(({
    isOpen,
    onComplete$,
    onSkip$
}) => {
    const currentStep = useSignal(0);

    const nextStep = $(() => {
        if (currentStep.value < steps.length - 1) {
            currentStep.value++;
        } else {
            onComplete$();
        }
    });

    const prevStep = $(() => {
        if (currentStep.value > 0) {
            currentStep.value--;
        }
    });

    if (!isOpen) return null;

    const step = steps[currentStep.value];

    return (
        <div class="fixed inset-0 z-[100] flex items-center justify-center">
            {/* Backdrop */}
            <div class="absolute inset-0 bg-black/90 backdrop-blur-sm" />

            {/* Modal */}
            <div class="relative z-10 w-full max-w-lg mx-4 bg-gradient-to-br from-slate-900 to-slate-950 rounded-3xl border border-[var(--glass-border)] overflow-hidden shadow-2xl">
                {/* Progress dots */}
                <div class="flex justify-center gap-2 pt-6">
                    {steps.map((_, i) => (
                        <div
                            key={i}
                            class={`w-2 h-2 rounded-full transition-all ${i === currentStep.value
                                    ? 'bg-cyan-400 w-6'
                                    : i < currentStep.value
                                        ? 'bg-cyan-400/50'
                                        : 'bg-white/20'
                                }`}
                        />
                    ))}
                </div>

                {/* Content */}
                <div class="p-8 text-center space-y-6">
                    {/* Icon */}
                    <div class="text-6xl animate-bounce">{step.icon}</div>

                    {/* Text */}
                    <div class="space-y-3">
                        <h2 class="text-2xl font-semibold text-white">{step.title}</h2>
                        <p class="text-white/60 leading-relaxed">{step.description}</p>
                    </div>
                </div>

                {/* Actions */}
                <div class="flex items-center justify-between p-6 border-t border-[var(--glass-border)] bg-black/20">
                    <button
                        onClick$={onSkip$}
                        class="px-4 py-2 text-sm text-white/50 hover:text-white/80 transition-colors"
                    >
                        Skip Tutorial
                    </button>

                    <div class="flex gap-3">
                        {currentStep.value > 0 && (
                            <button
                                onClick$={prevStep}
                                class="px-4 py-2 bg-white/5 hover:bg-white/10 text-white/70 rounded-xl transition-colors"
                            >
                                Back
                            </button>
                        )}
                        <button
                            onClick$={nextStep}
                            class="px-6 py-2 bg-gradient-to-r from-cyan-500 to-blue-500 hover:from-cyan-400 hover:to-blue-400 text-white font-medium rounded-xl transition-all shadow-lg shadow-cyan-500/20"
                        >
                            {currentStep.value === steps.length - 1 ? 'Get Started' : 'Next'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
});

export default AuraluxOnboarding;
