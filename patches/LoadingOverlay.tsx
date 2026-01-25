/**
 * LoadingOverlay - 60% opacity overlay until hydration complete
 * Shows progress indicator and fades out when app is interactive
 */
import { component$, useSignal, useVisibleTask$ } from "@builder.io/qwik";

declare global {
    interface Window { __pluribusReady?: boolean; }
}

export const LoadingOverlay = component$(() => {
    const visible = useSignal(true);
    const progress = useSignal(0);

    useVisibleTask$(({ cleanup }) => {
        // Animate progress bar
        const pi = setInterval(() => {
            if (progress.value < 90) progress.value += Math.random() * 15;
        }, 200);

        // Check for hydration complete
        const ri = setInterval(() => {
            if (window.__pluribusReady) {
                progress.value = 100;
                setTimeout(() => {
                    visible.value = false;
                    document.body.classList.add("hydrated");
                }, 300);
                clearInterval(ri);
                clearInterval(pi);
            }
        }, 50);

        // Fallback: hide after 10s max
        const t = setTimeout(() => {
            visible.value = false;
            document.body.classList.add("hydrated");
        }, 10000);

        cleanup(() => {
            clearInterval(pi);
            clearInterval(ri);
            clearTimeout(t);
        });
    });

    if (!visible.value) return null;

    return (
        <div class="loading-overlay" aria-live="polite" aria-busy="true">
            <div class="loading-content">
                <div class="loading-orb" />
                <div class="loading-progress">
                    <div
                        class="loading-progress-bar"
                        style={{ width: `${Math.min(progress.value, 100)}%` }}
                    />
                </div>
                <div class="loading-status">
                    {progress.value < 30 && "Initializing..."}
                    {progress.value >= 30 && progress.value < 60 && "Loading..."}
                    {progress.value >= 60 && progress.value < 90 && "Hydrating..."}
                    {progress.value >= 90 && "Almost ready..."}
                </div>
            </div>
        </div>
    );
});
