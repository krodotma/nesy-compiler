/**
 * AuraluxDashboard.tsx - 200% ENHANCED VERSION
 * 
 * Premium features:
 * - Animated latency history sparkline
 * - Real-time stage indicators with pulse
 * - Memory usage bar with gradient
 * - WebGPU/WASM backend badge with tooltip
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import type { AuraluxState } from '../../lib/auralux';

interface StageLatency {
    vad: number;
    ssl: number;
    vocoder: number;
    total: number;
}

interface AuraluxDashboardProps {
    state: AuraluxState;
    latency: StageLatency;
    memoryMB?: number;
    backend?: 'webgpu' | 'wasm' | 'cpu';
}

export const AuraluxDashboard = component$<AuraluxDashboardProps>(({
    state,
    latency,
    memoryMB = 0,
    backend = 'wasm'
}) => {
    const latencyHistory = useSignal<number[]>([]);
    const showTooltip = useSignal(false);

    // Track latency history for sparkline
    useVisibleTask$(({ track }) => {
        track(() => latency.total);
        latencyHistory.value = [...latencyHistory.value.slice(-29), latency.total];
    });

    const pipelineStages = [
        { name: 'VAD', icon: '🎤', ms: latency.vad, target: 10, loaded: state.modelsLoaded.vad },
        { name: 'SSL', icon: '🧠', ms: latency.ssl, target: 40, loaded: state.modelsLoaded.ssl },
        { name: 'Vocoder', icon: '🔊', ms: latency.vocoder, target: 30, loaded: state.modelsLoaded.vocoder },
    ];

    const getLatencyColor = (ms: number, target: number) => {
        const ratio = ms / target;
        if (ratio <= 1) return { text: 'text-green-400', bg: 'from-green-500 to-cyan-500' };
        if (ratio <= 1.5) return { text: 'text-amber-400', bg: 'from-amber-500 to-orange-500' };
        return { text: 'text-red-400', bg: 'from-red-500 to-pink-500' };
    };

    const backendLabels = {
        webgpu: { label: 'WebGPU', color: 'from-green-500 to-emerald-500', desc: 'GPU-accelerated inference' },
        wasm: { label: 'WASM-SIMD', color: 'from-cyan-500 to-blue-500', desc: 'Optimized WebAssembly' },
        cpu: { label: 'CPU', color: 'from-gray-500 to-slate-500', desc: 'Fallback CPU inference' },
    };

    const backendInfo = backendLabels[backend];

    // Sparkline max for scaling
    const sparklineMax = Math.max(...latencyHistory.value, 100);

    return (
        <div class="bg-black/60 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] p-4 space-y-4 relative overflow-hidden">
            {/* Subtle animated background */}
            <div
                class="absolute inset-0 opacity-5 pointer-events-none"
                style={{
                    background: 'radial-gradient(ellipse at 50% 0%, rgba(6, 182, 212, 0.3), transparent 50%)'
                }}
            />

            {/* Header */}
            <div class="relative flex items-center justify-between">
                <h2 class="text-sm font-semibold text-white/80 flex items-center gap-2">
                    <span class="text-cyan-400">⚡</span>
                    Auralux Pipeline
                </h2>
                <div class="flex items-center gap-2">
                    {/* Backend badge with tooltip */}
                    <div
                        class="relative"
                        onMouseEnter$={() => showTooltip.value = true}
                        onMouseLeave$={() => showTooltip.value = false}
                    >
                        <div class={`px-2 py-1 rounded-full text-[10px] font-medium bg-gradient-to-r ${backendInfo.color} text-white`}>
                            {backendInfo.label}
                        </div>
                        {showTooltip.value && (
                            <div class="absolute top-full mt-1 right-0 px-2 py-1 bg-black/90 rounded text-[10px] text-white/70 whitespace-nowrap z-10">
                                {backendInfo.desc}
                            </div>
                        )}
                    </div>

                    <div class={`px-2 py-1 rounded-full text-xs font-medium ${state.isReady ? 'bg-green-500/20 text-green-400' : 'bg-amber-500/20 text-amber-400'
                        }`}>
                        {state.isReady ? 'Ready' : 'Loading...'}
                    </div>
                </div>
            </div>

            {/* Pipeline Stages with animated indicators */}
            <div class="relative flex items-center justify-between gap-2">
                {pipelineStages.map((stage, i) => {
                    const colors = getLatencyColor(stage.ms, stage.target);
                    return (
                        <div key={stage.name} class="flex items-center gap-2">
                            <div class={`relative flex flex-col items-center p-3 rounded-xl transition-all duration-300 ${stage.loaded
                                    ? 'bg-white/5 border border-[var(--glass-border)]'
                                    : 'bg-white/5 border border-dashed border-[var(--glass-border-hover)] opacity-50'
                                }`}>
                                {/* Active pulse indicator */}
                                {stage.loaded && state.isProcessing && (
                                    <div class="absolute -top-1 -right-1 w-3 h-3">
                                        <div class="absolute inset-0 bg-green-400 rounded-full animate-ping opacity-75" />
                                        <div class="absolute inset-0 bg-green-400 rounded-full" />
                                    </div>
                                )}

                                <span class="text-lg">{stage.icon}</span>
                                <span class="text-xs text-white/60 mt-1">{stage.name}</span>
                                <span class={`text-xs font-mono mt-1 ${colors.text}`}>
                                    {stage.ms.toFixed(0)}ms
                                </span>
                            </div>

                            {i < pipelineStages.length - 1 && (
                                <div class="flex items-center">
                                    <div class={`h-0.5 w-6 bg-gradient-to-r ${stage.loaded ? 'from-white/20 to-white/40' : 'from-white/5 to-white/10'
                                        }`} />
                                    <span class="text-white/30 text-xs">→</span>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>

            {/* Latency Sparkline */}
            <div class="space-y-1">
                <div class="flex justify-between text-xs">
                    <span class="text-white/50">E2E Latency History</span>
                    <span class={`font-mono ${getLatencyColor(latency.total, 100).text}`}>
                        {latency.total.toFixed(0)}ms
                    </span>
                </div>
                <div class="h-10 flex items-end gap-0.5">
                    {latencyHistory.value.map((l, i) => {
                        const height = (l / sparklineMax) * 100;
                        const colors = getLatencyColor(l, 100);
                        return (
                            <div
                                key={i}
                                class={`flex-1 rounded-t bg-gradient-to-t ${colors.bg} transition-all duration-150`}
                                style={{ height: `${Math.max(height, 5)}%`, opacity: 0.3 + (i / latencyHistory.value.length) * 0.7 }}
                            />
                        );
                    })}
                </div>
                <div class="flex justify-between text-[9px] text-white/30">
                    <span>30 samples</span>
                    <span class="text-green-400/50">100ms target</span>
                </div>
            </div>

            {/* Memory Usage Bar */}
            {memoryMB > 0 && (
                <div class="space-y-1">
                    <div class="flex justify-between text-xs">
                        <span class="text-white/50">Memory</span>
                        <span class="text-white/40">{memoryMB.toFixed(0)} MB</span>
                    </div>
                    <div class="h-1.5 bg-white/10 rounded-full overflow-hidden">
                        <div
                            class="h-full rounded-full transition-all duration-500"
                            style={{
                                width: `${Math.min(memoryMB / 200 * 100, 100)}%`,
                                background: memoryMB < 100
                                    ? 'linear-gradient(90deg, rgb(34, 197, 94), rgb(6, 182, 212))'
                                    : memoryMB < 150
                                        ? 'linear-gradient(90deg, rgb(251, 191, 36), rgb(245, 158, 11))'
                                        : 'linear-gradient(90deg, rgb(239, 68, 68), rgb(220, 38, 38))'
                            }}
                        />
                    </div>
                </div>
            )}

            {/* Download Progress */}
            {state.downloadProgress > 0 && state.downloadProgress < 100 && (
                <div class="space-y-1">
                    <div class="flex justify-between text-xs">
                        <span class="text-white/50">Downloading models...</span>
                        <span class="text-cyan-400">{state.downloadProgress}%</span>
                    </div>
                    <div class="h-1 bg-white/10 rounded-full overflow-hidden">
                        <div
                            class="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
                            style={{ width: `${state.downloadProgress}%` }}
                        />
                    </div>
                </div>
            )}

            {/* Mode Indicator + Export */}
            <div class="flex items-center justify-between pt-2 border-t border-[var(--glass-border)]">
                <div class="flex items-center gap-2">
                    <div class={`w-2 h-2 rounded-full ${state.mode === 'neural' ? 'bg-green-400 animate-pulse' : 'bg-amber-400'
                        }`} />
                    <span class="text-xs text-white/50">
                        {state.mode === 'neural' ? 'Neural Engine Active' : 'Browser Fallback'}
                    </span>
                </div>
                <div class="flex items-center gap-2">
                    {/* Export Metrics Button */}
                    <button
                        onClick$={() => {
                            const metrics = {
                                timestamp: new Date().toISOString(),
                                mode: state.mode,
                                backend,
                                latency: { ...latency },
                                latencyHistory: [...latencyHistory.value],
                                modelsLoaded: { ...state.modelsLoaded },
                                memoryMB,
                                isReady: state.isReady
                            };
                            const blob = new Blob([JSON.stringify(metrics, null, 2)], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `auralux-metrics-${Date.now()}.json`;
                            a.click();
                            URL.revokeObjectURL(url);
                        }}
                        class="p-1.5 hover:bg-white/10 rounded-lg transition-colors group"
                        title="Export metrics as JSON"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="text-white/30 group-hover:text-cyan-400 transition-colors">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                            <polyline points="7 10 12 15 17 10"></polyline>
                            <line x1="12" y1="15" x2="12" y2="3"></line>
                        </svg>
                    </button>
                    <span class="text-[10px] text-white/30 font-mono">
                        AURALUX v2.0
                    </span>
                </div>
            </div>
        </div>
    );
});

export default AuraluxDashboard;
