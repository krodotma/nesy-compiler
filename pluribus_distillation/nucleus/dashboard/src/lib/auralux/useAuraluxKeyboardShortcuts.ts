/**
 * useAuraluxKeyboardShortcuts.ts
 * 
 * Phase E Step 46: Keyboard shortcuts for AuraluxStudio
 * Space = toggle listening, Esc = cancel/close overlay
 */

import { $, useVisibleTask$ } from '@builder.io/qwik';

interface KeyboardShortcutHandlers {
    onToggleListening$: () => void;
    onCancel$: () => void;
    onFocusMode$: () => void;
}

export function useAuraluxKeyboardShortcuts(handlers: KeyboardShortcutHandlers) {
    useVisibleTask$(({ cleanup }) => {
        const handleKeyDown = (event: KeyboardEvent) => {
            // Ignore if typing in input/textarea
            const target = event.target as HTMLElement;
            if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA') {
                return;
            }

            switch (event.code) {
                case 'Space':
                    event.preventDefault();
                    handlers.onToggleListening$();
                    break;
                case 'Escape':
                    event.preventDefault();
                    handlers.onCancel$();
                    break;
                case 'KeyF':
                    if (event.ctrlKey || event.metaKey) {
                        event.preventDefault();
                        handlers.onFocusMode$();
                    }
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        cleanup(() => window.removeEventListener('keydown', handleKeyDown));
    });
}

export default useAuraluxKeyboardShortcuts;
