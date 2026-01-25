/**
 * AuraluxAccessibility.tsx
 * 
 * Phase E Step 48: Accessibility - screen reader announcements
 * ARIA live regions for voice status updates
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';

interface AuraluxAccessibilityProps {
    isListening: boolean;
    isSpeaking: boolean;
    mode: 'neural' | 'browser' | 'loading' | 'error';
    error: string | null;
}

export const AuraluxAccessibility = component$<AuraluxAccessibilityProps>(({
    isListening,
    isSpeaking,
    mode,
    error
}) => {
    const announcement = useSignal('');
    const lastState = useSignal({ isListening: false, isSpeaking: false, mode: 'loading' });

    // Announce state changes
    useVisibleTask$(({ track }) => {
        track(() => isListening);
        track(() => isSpeaking);
        track(() => mode);
        track(() => error);

        // Detect changes and announce
        if (isListening !== lastState.value.isListening) {
            announcement.value = isListening
                ? 'Auralux is now listening for speech'
                : 'Auralux has stopped listening';
        }

        if (isSpeaking !== lastState.value.isSpeaking) {
            announcement.value = isSpeaking
                ? 'Speech detected'
                : 'Speech ended';
        }

        if (mode !== lastState.value.mode) {
            switch (mode) {
                case 'neural':
                    announcement.value = 'Neural voice engine active';
                    break;
                case 'browser':
                    announcement.value = 'Using browser voice engine';
                    break;
                case 'loading':
                    announcement.value = 'Loading voice models';
                    break;
                case 'error':
                    announcement.value = `Voice engine error: ${error || 'Unknown error'}`;
                    break;
            }
        }

        lastState.value = { isListening, isSpeaking, mode };
    });

    return (
        <>
            {/* Screen reader only live region */}
            <div
                role="status"
                aria-live="polite"
                aria-atomic="true"
                class="sr-only"
            >
                {announcement.value}
            </div>

            {/* Labels for controls (visually hidden but accessible) */}
            <div class="sr-only">
                <span id="auralux-listen-label">Toggle voice listening</span>
                <span id="auralux-mode-label">Voice engine mode selector</span>
                <span id="auralux-status-label">
                    Current status: {isListening ? 'Listening' : 'Idle'},
                    Mode: {mode},
                    {isSpeaking ? 'Speech detected' : 'No speech'}
                </span>
            </div>

            {/* CSS for sr-only class */}
            <style>
                {`
          .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
          }
        `}
            </style>
        </>
    );
});

export default AuraluxAccessibility;
