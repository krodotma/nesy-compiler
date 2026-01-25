import { $, component$, useSignal, useVisibleTask$, useStore } from "@builder.io/qwik";
import { tracker, type TrackerState } from "../lib/telemetry/verbose-tracker";
import { LoadingOrbShader } from "./LoadingOrbShader";

interface DisplayItem {
    desc: string;
    seg: string;
    time: string;
    id: string;
}

export const LoadingOverlay = component$(() => {
    const visible = useSignal(true);
    const shaderReady = useSignal(false);

    const state = useStore({
        percent: 0,
        count: 0,
        total: 39,
        visibleItems: [] as DisplayItem[],
        queueIndex: 0,
        lastSwitchTime: 0,
        ready: false
    });

    useVisibleTask$(({ cleanup }) => {
        const getTracker = () => (window as any).__verboseTracker || tracker;

        const loop = setInterval(() => {
            const t = getTracker();
            if (!t) return;

            const s = t.state as TrackerState;
            const now = performance.now();

            // Update stats - ensure reasonable bounds
            state.percent = Math.min(100, Math.max(0, s.progressPercent || 0));
            state.count = s.completedCount || 0;
            state.total = Math.max(s.totalCount || 39, state.count); // Total should never be less than count

            // Visualization queue
            const queue = s.recentlyCompleted || [];
            const itemsPending = queue.length - state.queueIndex;
            const minTime = itemsPending > 5 ? 100 : 500;

            if (state.queueIndex < queue.length) {
                if ((now - state.lastSwitchTime) > minTime) {
                    const nextItem = queue[state.queueIndex];
                    if (nextItem) {
                        const newItem = {
                            desc: nextItem.meta?.name || "Loading...",
                            seg: nextItem.meta?.id || "item",
                            time: ((nextItem.duration || 0)).toFixed(0) + "ms",
                            id: nextItem.meta?.id || `item-${state.queueIndex}`
                        };
                        const newList = [...state.visibleItems, newItem];
                        if (newList.length > 8) newList.shift();
                        state.visibleItems = newList;

                        state.queueIndex++;
                        state.lastSwitchTime = now;
                    }
                }
            } else if (state.visibleItems.length === 0 && (s.activeItems?.length || 0) > 0 && (now - state.lastSwitchTime > 2000)) {
                state.visibleItems = [{
                    desc: s.activeItems[0],
                    seg: "active",
                    time: "...",
                    id: "active"
                }];
            }

            // Completion
            if (s.progressPercent >= 100 && state.queueIndex >= queue.length) {
                if (!state.ready) {
                    state.ready = true;
                    setTimeout(() => {
                        visible.value = false;
                        document.body.classList.add("hydrated");
                    }, 1200);
                }
            }

            // Safety timeout
            if ((window as any).__pluribusReady && !state.ready) {
                state.percent = 100;
                state.ready = true;
                setTimeout(() => { visible.value = false; }, 600);
            }

        }, 50);

        const timeout = setTimeout(() => { visible.value = false; }, 15000);

        cleanup(() => {
            clearInterval(loop);
            clearTimeout(timeout);
        });
    });

    if (!visible.value) return null;

    return (
        <div class="loading-overlay">
            <div class="loading-content">

                {/* Animated Logo Orb - Shader */}
                <div class="loading-orb">
                    <div class={`loading-glow ${shaderReady.value ? 'visible' : ''}`} />
                    <LoadingOrbShader onReady$={$(() => shaderReady.value = true)} />
                </div>

                {/* Title */}
                <h2 class="loading-title">Pluribus</h2>

                {/* Modern Progress Bar */}
                <div class="loading-progress">
                    <div class="loading-progress-track">
                        <div
                            class="loading-progress-fill"
                            style={{ width: `${state.percent}%` }}
                        />
                        <div class="loading-progress-glow" style={{ left: `${state.percent}%` }} />
                    </div>
                </div>

                {/* Stats - Fixed format */}
                <div class="loading-stats">
                    <span class="loading-count">{state.count} / {state.total}</span>
                    <span class="loading-percent">{Math.round(state.percent)}%</span>
                </div>

                {/* Activity List */}
                <div class="loading-activity">
                    {state.visibleItems.length > 0 ? (
                        state.visibleItems.map((item, idx) => (
                            <div
                                class="activity-row"
                                key={item.id + idx}
                                style={{ opacity: 0.5 + (idx / state.visibleItems.length) * 0.5 }}
                            >
                                <span class="activity-name">{item.desc}</span>
                                <span class="activity-time">{item.time}</span>
                            </div>
                        ))
                    ) : (
                        <div class="activity-row">
                            <span class="activity-name">Initializing...</span>
                            <span class="activity-time">0ms</span>
                        </div>
                    )}
                </div>

            </div>
        </div>
    );
});
