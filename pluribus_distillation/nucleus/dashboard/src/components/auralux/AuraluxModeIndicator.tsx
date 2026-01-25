/**
 * AuraluxModeIndicator.tsx
 *
 * Visual indicator showing Neural vs Browser voice mode.
 * CLICKABLE - toggles between modes when clicked.
 * Phase A Step 10: Verify VoiceHUD shows neural vs browser mode
 */

import { component$, type QRL } from '@builder.io/qwik';
import type { AuraluxMode } from './VoiceProvider';

interface AuraluxModeIndicatorProps {
    mode: AuraluxMode;
    latencyMs?: number;
    onToggle$?: QRL<() => void>;
}

export const AuraluxModeIndicator = component$<AuraluxModeIndicatorProps>(({ mode, latencyMs, onToggle$ }) => {
    const isNeural = mode === 'neural';
    const isBrowser = mode === 'browser';
    const isLoading = mode === 'loading';
    const isError = mode === 'error';
    const canToggle = !isLoading && !isError && onToggle$;

    return (
        <button
            onClick$={onToggle$}
            disabled={!canToggle}
            class={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium transition-all duration-300 ${
                canToggle ? 'cursor-pointer hover:scale-105 active:scale-95' : 'cursor-default'
            }`}
            style={{
                background: isNeural
                    ? 'linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(6, 182, 212, 0.2))'
                    : isBrowser
                        ? 'rgba(251, 191, 36, 0.2)'
                        : isError
                            ? 'rgba(239, 68, 68, 0.2)'
                            : 'rgba(255, 255, 255, 0.1)',
                border: `1px solid ${isNeural ? 'rgba(34, 197, 94, 0.4)'
                        : isBrowser ? 'rgba(251, 191, 36, 0.4)'
                            : isError ? 'rgba(239, 68, 68, 0.4)'
                                : 'rgba(255, 255, 255, 0.2)'
                    }`
            }}
            title={canToggle ? `Click to switch to ${isNeural ? 'Browser' : 'Neural'} mode` : undefined}
        >
            {/* Mode Icon */}
            <div class={`w-2 h-2 rounded-full ${isNeural ? 'bg-green-400 animate-pulse'
                    : isBrowser ? 'bg-amber-400'
                        : isError ? 'bg-red-400'
                            : 'bg-white/50'
                }`} />

            {/* Mode Label */}
            <span class={`${isNeural ? 'text-green-400'
                    : isBrowser ? 'text-amber-400'
                        : isError ? 'text-red-400'
                            : 'text-white/70'
                }`}>
                {isLoading && 'Loading...'}
                {isNeural && '⚡ Neural'}
                {isBrowser && '🔊 Browser'}
                {isError && '⚠️ Error'}
            </span>

            {/* Latency Badge (only when neural) */}
            {isNeural && latencyMs !== undefined && (
                <span class="text-cyan-400 ml-1">
                    {latencyMs.toFixed(0)}ms
                </span>
            )}

            {/* Toggle hint */}
            {canToggle && (
                <svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-white/40">
                    <polyline points="6 9 12 15 18 9"></polyline>
                </svg>
            )}
        </button>
    );
});

export default AuraluxModeIndicator;
