/**
 * SslSpectrogramViewer.tsx
 * 
 * Phase D Step 39: SSL feature spectrogram visualization
 * Shows HuBERT/ContentVec feature maps in real-time
 */

import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';

type ColormapType = 'viridis' | 'plasma' | 'inferno' | 'cool' | 'ocean';

interface SslSpectrogramViewerProps {
    features: Float32Array | null;
    isActive: boolean;
    frameCount?: number;
    hiddenSize?: number;
    colormap?: ColormapType;
}

// Colormap definitions
const colormaps: Record<ColormapType, (v: number) => [number, number, number]> = {
    viridis: (v) => [
        Math.floor(68 + v * 120),
        Math.floor(1 + v * 180),
        Math.floor(84 + v * 100)
    ],
    plasma: (v) => [
        Math.floor(13 + v * 227),
        Math.floor(8 + v * 92),
        Math.floor(135 + v * (-25))
    ],
    inferno: (v) => [
        Math.floor(v * 252),
        Math.floor(v * 163),
        Math.floor(69 + v * (-40))
    ],
    cool: (v) => [
        Math.floor(34 + v * 0),
        Math.floor(100 + v * 155),
        Math.floor(100 + v * 112)
    ],
    ocean: (v) => [
        Math.floor(v * 50),
        Math.floor(80 + v * 120),
        Math.floor(120 + v * 135)
    ]
};

export const SslSpectrogramViewer = component$<SslSpectrogramViewerProps>(({
    features,
    isActive,
    frameCount = 50,
    hiddenSize = 256,
    colormap: initialColormap = 'cool'
}) => {
    const canvasRef = useSignal<HTMLCanvasElement>();
    const selectedColormap = useSignal<ColormapType>(initialColormap);

    useVisibleTask$(({ track }) => {
        track(() => features);
        track(() => isActive);
        track(() => selectedColormap.value);

        const canvas = canvasRef.value;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Clear canvas
        ctx.fillStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        if (!isActive || !features) {
            // Draw empty state
            ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.font = '12px monospace';
            ctx.textAlign = 'center';
            ctx.fillText('Waiting for features...', canvas.width / 2, canvas.height / 2);
            return;
        }

        // Draw spectrogram with selected colormap
        const cellWidth = canvas.width / frameCount;
        const cellHeight = canvas.height / Math.min(hiddenSize, 64);
        const getColor = colormaps[selectedColormap.value];

        for (let f = 0; f < frameCount && f * hiddenSize < features.length; f++) {
            for (let h = 0; h < Math.min(hiddenSize, 64); h++) {
                const idx = f * hiddenSize + h;
                const value = features[idx] || 0;
                const normalized = Math.min(Math.abs(value) / 2, 1);
                const [r, g, b] = getColor(normalized);

                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(
                    f * cellWidth,
                    (63 - h) * cellHeight,
                    cellWidth,
                    cellHeight
                );
            }
        }

        // Draw frame markers
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 0.5;
        for (let i = 0; i <= frameCount; i += 10) {
            ctx.beginPath();
            ctx.moveTo(i * cellWidth, 0);
            ctx.lineTo(i * cellWidth, canvas.height);
            ctx.stroke();
        }
    });

    const colormapLabels: Record<ColormapType, string> = {
        viridis: 'Viridis',
        plasma: 'Plasma',
        inferno: 'Inferno',
        cool: 'Cool',
        ocean: 'Ocean'
    };

    return (
        <div class="bg-black/60 backdrop-blur-xl rounded-2xl border border-[var(--glass-border)] p-4 space-y-3">
            {/* Header */}
            <div class="flex items-center justify-between">
                <h3 class="text-xs font-semibold text-white/70 flex items-center gap-2">
                    <span class="text-cyan-400">🧠</span>
                    SSL Features (HuBERT)
                </h3>
                <div class="flex items-center gap-2">
                    {/* Colormap selector */}
                    <select
                        value={selectedColormap.value}
                        onChange$={(e) => selectedColormap.value = (e.target as HTMLSelectElement).value as ColormapType}
                        class="bg-white/5 border border-[var(--glass-border)] rounded px-1.5 py-0.5 text-[10px] text-white/60 hover:bg-white/10 transition-colors cursor-pointer"
                    >
                        {(Object.keys(colormapLabels) as ColormapType[]).map(key => (
                            <option key={key} value={key} class="bg-slate-900">{colormapLabels[key]}</option>
                        ))}
                    </select>
                    <div class={`w-2 h-2 rounded-full ${isActive ? 'bg-green-400 animate-pulse' : 'bg-white/30'}`} />
                </div>
            </div>

            {/* Canvas */}
            <canvas
                ref={canvasRef}
                width={400}
                height={128}
                class="w-full h-32 rounded-lg"
            />

            {/* Legend */}
            <div class="flex justify-between text-[9px] text-white/40">
                <span>Time →</span>
                <div class="flex items-center gap-2">
                    <div class={`w-12 h-1.5 rounded bg-gradient-to-r ${
                        selectedColormap.value === 'viridis' ? 'from-[#440154] via-[#21908c] to-[#fde725]' :
                        selectedColormap.value === 'plasma' ? 'from-[#0d0887] via-[#cc4778] to-[#f0f921]' :
                        selectedColormap.value === 'inferno' ? 'from-[#000004] via-[#bc3754] to-[#fcffa4]' :
                        selectedColormap.value === 'ocean' ? 'from-[#003366] to-[#00ccff]' :
                        'from-cyan-600 to-green-400'
                    }`} />
                    <span>Feature magnitude</span>
                </div>
            </div>
        </div>
    );
});

export default SslSpectrogramViewer;
