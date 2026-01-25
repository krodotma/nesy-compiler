/**
 * VadEventLog.tsx
 * 
 * Phase C Step 24: VAD trigger/release events mini-log
 * PBDESIGN Aesthetic: Terminal-inspired with neon accents
 * 
 * Shows recent VAD events (START/END) with timestamps
 */

import { component$ } from '@builder.io/qwik';

interface VadEvent {
    id: string;
    type: 'start' | 'end' | 'misfire';
    timestamp: number;
    confidence?: number;
    duration?: number;
}

interface VadEventLogProps {
    events: VadEvent[];
    maxEvents?: number;
}

export const VadEventLog = component$<VadEventLogProps>(({ events, maxEvents = 8 }) => {
    const recentEvents = events.slice(-maxEvents);

    const formatTime = (ts: number) => {
        const date = new Date(ts);
        return date.toLocaleTimeString('en-US', {
            hour12: false,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    };

    const getEventStyle = (type: VadEvent['type']) => {
        switch (type) {
            case 'start':
                return {
                    icon: '▶',
                    color: 'text-green-400',
                    bg: 'bg-green-500/10',
                    label: 'SPEECH_START'
                };
            case 'end':
                return {
                    icon: '■',
                    color: 'text-cyan-400',
                    bg: 'bg-cyan-500/10',
                    label: 'SPEECH_END'
                };
            case 'misfire':
                return {
                    icon: '⚠',
                    color: 'text-amber-400',
                    bg: 'bg-amber-500/10',
                    label: 'VAD_MISFIRE'
                };
        }
    };

    return (
        <div class="bg-black/60 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] overflow-hidden">
            {/* Header */}
            <div class="px-3 py-2 border-b border-[var(--glass-border-subtle)] flex items-center justify-between">
                <span class="text-[10px] font-medium text-white/50 uppercase tracking-wider">
                    VAD Events
                </span>
                <span class="text-[10px] text-white/30 font-mono">
                    {events.length} total
                </span>
            </div>

            {/* Event list */}
            <div class="max-h-40 overflow-y-auto scrollbar-thin scrollbar-thumb-white/10">
                {recentEvents.length === 0 ? (
                    <div class="px-3 py-4 text-center text-xs text-white/30 italic">
                        No events yet
                    </div>
                ) : (
                    <div class="divide-y divide-white/5">
                        {recentEvents.map((event) => {
                            const style = getEventStyle(event.type);
                            return (
                                <div
                                    key={event.id}
                                    class={`px-3 py-2 flex items-center gap-3 ${style.bg} transition-colors`}
                                >
                                    {/* Icon */}
                                    <span class={`text-sm ${style.color}`}>{style.icon}</span>

                                    {/* Event info */}
                                    <div class="flex-1 min-w-0">
                                        <div class="flex items-center gap-2">
                                            <span class={`text-xs font-mono ${style.color}`}>
                                                {style.label}
                                            </span>
                                            {event.confidence !== undefined && (
                                                <span class="text-[10px] text-white/30">
                                                    {(event.confidence * 100).toFixed(0)}%
                                                </span>
                                            )}
                                        </div>
                                        {event.duration !== undefined && (
                                            <span class="text-[10px] text-white/40">
                                                Duration: {event.duration.toFixed(0)}ms
                                            </span>
                                        )}
                                    </div>

                                    {/* Timestamp */}
                                    <span class="text-[10px] font-mono text-white/30">
                                        {formatTime(event.timestamp)}
                                    </span>
                                </div>
                            );
                        })}
                    </div>
                )}
            </div>
        </div>
    );
});

export default VadEventLog;
