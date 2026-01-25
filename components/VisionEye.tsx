import { component$, useSignal, useVisibleTask$, $ } from '@builder.io/qwik';

export const VisionEye = component$(() => {
    const videoRef = useSignal<HTMLVideoElement>();
    const canvasRef = useSignal<HTMLCanvasElement>();
    const active = useSignal(false);
    const bufferCount = useSignal(0);
    const status = useSignal('Ready');

    // Ring Buffer State (Closure restricted to this component instance)
    // We use a ref-like pattern since Qwik component logic runs on server/client boundary
    const ringBuffer = useSignal<{ ts: number, data: string, meta?: any }[]>([]);
    const BUFFER_SIZE = 60;

    const startCapture = $(async () => {
        if (!videoRef.value || !canvasRef.value) return;

        try {
            const stream = await navigator.mediaDevices.getDisplayMedia({
                video: { cursor: "always" },
                audio: false
            });
            videoRef.value.srcObject = stream;
            active.value = true;
            status.value = 'Buffering...';

            // Start Ring Buffer Loop
            const ctx = canvasRef.value.getContext('2d');
            if (!ctx) return;

            const interval = setInterval(() => {
                if (!videoRef.value || !active.value) {
                    clearInterval(interval);
                    return;
                }

                // Draw to canvas (downscaled)
                const w = videoRef.value.videoWidth / 2;
                const h = videoRef.value.videoHeight / 2;

                if (w && h) {
                    canvasRef.value!.width = w;
                    canvasRef.value!.height = h;
                    ctx.drawImage(videoRef.value, 0, 0, w, h);

                    const frame = canvasRef.value!.toDataURL('image/jpeg', 0.7);

                    // Push to buffer
                    const newBuffer = [...ringBuffer.value];
                    if (newBuffer.length >= BUFFER_SIZE) newBuffer.shift();
                    newBuffer.push({ ts: Date.now(), data: frame });

                    ringBuffer.value = newBuffer;
                    bufferCount.value = newBuffer.length;
                }
            }, 1000); // 1 FPS

            // Cleanup stream on end
            stream.getVideoTracks()[0].onended = () => {
                active.value = false;
                clearInterval(interval);
                status.value = 'Stream ended';
            };

        } catch (err) {
            console.error(err);
            status.value = 'Error: ' + err;
        }
    });

    const sendSnapshot = $(async () => {
        status.value = 'Sending...';
        try {
            const payload = {
                timestamp: Date.now(),
                frames: ringBuffer.value.map(f => f.data),
                meta: { count: ringBuffer.value.length }
            };

            const res = await fetch('http://localhost:8080/v1/eyes/ingest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const json = await res.json();
            status.value = `Sent ${json.count} frames`;
        } catch (err) {
            status.value = 'Upload Failed';
            console.error(err);
        }
    });

    return (
        <div class="flex items-center gap-2 px-3 py-1 bg-slate-800/50 rounded-full border border-slate-700/50 backdrop-blur-sm">
            <div class={`w-2 h-2 rounded-full ${active.value ? 'bg-green-500 animate-pulse' : 'bg-slate-500'}`} />

            {!active.value ? (
                <button
                    onClick$={startCapture}
                    class="text-xs font-semibold text-slate-300 hover:text-white transition-colors"
                >
                    Enable Vision
                </button>
            ) : (
                <div class="flex items-center gap-2">
                    <span class="text-xs text-slate-400 font-mono">{bufferCount.value}f</span>
                    <button
                        onClick$={sendSnapshot}
                        class="text-xs font-bold text-blue-400 hover:text-blue-300 transition-colors"
                    >
                        SNAPSHOT
                    </button>
                </div>
            )}

            {/* Hidden Capture Elements */}
            <video ref={videoRef} autoPlay playsInline muted class="hidden" />
            <canvas ref={canvasRef} class="hidden" />
        </div>
    );
});
