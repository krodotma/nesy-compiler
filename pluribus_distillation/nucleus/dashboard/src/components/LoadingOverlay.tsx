import { component$, useSignal, useVisibleTask$, useStore, noSerialize } from "@builder.io/qwik";
import { tracker, type TrackerState } from "../lib/telemetry/verbose-tracker";
import { LoadingRegistry } from "../lib/telemetry/LoadingRegistry";
import { MANIFEST_STATS } from "../lib/telemetry/LoadingManifest";
import { CliffordTorusShader } from "./art/CliffordTorusShader";
import subsystemRegistry from "../../../specs/subsystem_registry.json";

// M3 Components - LoadingOverlay
import '@material/web/progress/linear-progress.js';
import '@material/web/elevation/elevation.js';

interface DisplayItem {
    desc: string;
    seg: string;
    time: string;
    id: string;
}

interface BootLine {
    id: string;
    status: "ok" | "warn" | "wait" | "fail";
    label: string;
    detail?: string;
    meta?: string;
}

interface RegistryLine {
    id: string;
    name: string;
    status: "pending" | "loading" | "complete" | "error";
    duration?: number;
    error?: string;
}

interface BusEventLine {
    topic?: string;
    iso?: string;
    actor?: string;
}

interface SotaItem {
    id: string;
    title?: string;
    type?: string;
    priority?: number;
    tags?: string[];
    org?: string;
}

interface SkillItem {
    id: string;
    name: string;
    description?: string;
    source: string;
    tower_level?: number;
    mastery?: number;
    tags?: string[];
    status: "ok" | "warn" | "wait";
}

interface SubsystemCategory {
    label?: string;
    description?: string;
}

interface SubsystemEntry {
    id: string;
    name: string;
    description?: string;
}

interface SubsystemRegistry {
    categories?: Record<string, SubsystemCategory>;
    subsystems?: Record<string, SubsystemEntry[]>;
    version?: string;
    updated_iso?: string;
}

export const LoadingOverlay = component$(() => {
    const visible = useSignal(true);
    const exiting = useSignal(false); // Exit animation state
    const entered = useSignal(false); // Entry animation complete
    const shaderReady = useSignal(false);
    const SOTA_PAGE_SIZE = 12;
    const SUBSYS_PAGE_SIZE = 12;
    const trackerRef = noSerialize(tracker);
    const registryRef = noSerialize(LoadingRegistry);

    const state = useStore({
        percent: 0,
        count: 0,
        total: MANIFEST_STATS.total,
        visibleItems: [] as DisplayItem[],
        registryLines: [] as RegistryLine[],
        queueIndex: 0,
        lastSwitchTime: 0,
        lastRegistryUpdate: 0,
        ready: false,
        startedAt: 0,
        sotaPage: 0,
        sotaCycleComplete: false,
        subsystemPage: 0,
        subsystemCycleComplete: false
    });

    const telemetry = useStore({
        bus: {
            totalEvents: 0,
            velocity: 0,
            errorRate: 0,
            topics: 0,
            agents: 0,
            loaded: false
        },
        dialogos: {
            status: "unknown",
            records: 0,
            loaded: false
        },
        falkor: {
            status: "unknown",
            nodes: 0,
            edges: 0,
            sync: 0,
            latency: 0,
            loaded: false
        },
        semops: {
            operators: 0,
            commands: 0,
            loaded: false
        },
        metatest: {
            totalTests: 0,
            coverage: 0,
            criticalGaps: 0,
            loaded: false,
            lastRun: null as null | { status?: string; scope?: string; finished_at?: string; started_at?: string; duration_s?: number }
        },
        learning: [] as { id: string; status: string; label: string; detail: string; meta: string; uncertainty: number }[],
        sota: [] as SotaItem[],
        busEvents: [] as BusEventLine[],
        skills: {
            total: 0,
            sources: { pbskills: 0, registries: 0, oss: 0 },
            items: [] as SkillItem[],
            loaded: false
        }
    });

    // Trigger enter animation after mount
    useVisibleTask$(() => {
        requestAnimationFrame(() => {
            entered.value = true;
            // Show glow after shader has time to initialize
            setTimeout(() => {
                shaderReady.value = true;
            }, 500);
        });
    });

    useVisibleTask$(({ cleanup }) => {
        state.startedAt = performance.now();
        const SOTA_ROTATE_MS = 320;
        const SUBSYS_ROTATE_MS = 280;
        // Move to state or verify scope. It's used in 2nd visible task.
        // Actually, let's just re-calculate or move it up. Re-calculating is cheap.
        const subsystemLinesCount = Object.keys((subsystemRegistry as SubsystemRegistry).subsystems || {}).length;

        const fetchJson = async (url: string, timeoutMs = 1200) => {
            const timeout = new Promise<null>((resolve) => {
                setTimeout(() => resolve(null), timeoutMs);
            });
            const fetchPromise = fetch(url, { cache: "no-store" })
                .then(async (res) => {
                    if (!res.ok) return null;
                    const raw = await res.text();
                    try {
                        return JSON.parse(raw);
                    } catch {
                        return null;
                    }
                })
                .catch(() => null);
            return Promise.race([fetchPromise, timeout]);
        };

        const hydrateTelemetry = async () => {
            const [metrics, dialogos, falkor, semops, metatest, learning, sota, busEvents, skills] = await Promise.allSettled([
                fetchJson("/api/metrics/snapshot?window=60"),
                fetchJson("/api/dialogos/health"),
                fetchJson("/api/falkordb/health"),
                fetchJson("/api/semops"),
                fetchJson("/api/metatest/inventory"),
                fetchJson("/api/learning/status"),
                fetchJson("/api/sota"),
                fetchJson("/api/bus/events?limit=12"),
                fetchJson("/api/skills")
            ]);

            if (metrics.status === "fulfilled" && metrics.value) {
                telemetry.bus.totalEvents = Number(metrics.value?.kpis?.total_events || 0);
                telemetry.bus.velocity = Number(metrics.value?.kpis?.velocity || 0);
                telemetry.bus.errorRate = Number(metrics.value?.kpis?.error_rate || 0);
                telemetry.bus.topics = Number(metrics.value?.topics?.count || 0);
                telemetry.bus.agents = Number(metrics.value?.agents?.count || 0);
                telemetry.bus.loaded = true;
            }

            if (dialogos.status === "fulfilled" && dialogos.value) {
                telemetry.dialogos.status = String(dialogos.value?.status || "unknown");
                telemetry.dialogos.records = Number(dialogos.value?.records_indexed_total || 0);
                telemetry.dialogos.loaded = true;
            }

            if (falkor.status === "fulfilled" && falkor.value) {
                telemetry.falkor.status = String(falkor.value?.status || "unknown");
                telemetry.falkor.nodes = Number(falkor.value?.nodes || 0);
                telemetry.falkor.edges = Number(falkor.value?.edges || 0);
                telemetry.falkor.sync = Number(falkor.value?.sync_pct || 0);
                telemetry.falkor.latency = Number(falkor.value?.latency_ms || 0);
                telemetry.falkor.loaded = true;
            }

            if (semops.status === "fulfilled" && semops.value) {
                telemetry.semops.operators = Object.keys(semops.value?.operators || {}).length;
                telemetry.semops.commands = Array.isArray(semops.value?.commands) ? semops.value.commands.length : 0;
                telemetry.semops.loaded = true;
            }

            if (metatest.status === "fulfilled" && metatest.value) {
                telemetry.metatest.totalTests = Number(metatest.value?.summary?.total_tests || 0);
                telemetry.metatest.coverage = Number(metatest.value?.summary?.coverage_percent || 0);
                telemetry.metatest.criticalGaps = Number(metatest.value?.summary?.critical_gaps || 0);
                const runs = metatest.value?.runs;
                const lastRun = runs?.last?.all || runs?.recent?.[0] || null;
                telemetry.metatest.lastRun = lastRun
                    ? {
                        status: lastRun.status,
                        scope: lastRun.scope,
                        finished_at: lastRun.finished_at,
                        started_at: lastRun.started_at,
                        duration_s: lastRun.duration_s
                    }
                    : null;
                telemetry.metatest.loaded = true;
            }

            if (learning.status === "fulfilled" && learning.value) {
                telemetry.learning = Array.isArray(learning.value?.systems) ? learning.value.systems : [];
            }

            if (sota.status === "fulfilled" && sota.value) {
                telemetry.sota = Array.isArray(sota.value?.items) ? sota.value.items : [];
                state.sotaPage = 0;
                state.sotaCycleComplete = telemetry.sota.length <= SOTA_PAGE_SIZE;
            }

            if (busEvents.status === "fulfilled" && busEvents.value) {
                const payload = Array.isArray(busEvents.value) ? busEvents.value : busEvents.value?.events;
                telemetry.busEvents = Array.isArray(payload) ? payload.slice(0, 8) : [];
            }

            if (skills.status === "fulfilled" && skills.value) {
                telemetry.skills.total = Number(skills.value?.total || 0);
                telemetry.skills.sources = skills.value?.sources || { pbskills: 0, registries: 0, oss: 0 };
                telemetry.skills.items = Array.isArray(skills.value?.skills) ? skills.value.skills : [];
                telemetry.skills.loaded = true;
            }
        };

        hydrateTelemetry();

        const subsysTotalPages = Math.max(1, Math.ceil(subsystemLinesCount / SUBSYS_PAGE_SIZE));
        state.subsystemCycleComplete = subsysTotalPages <= 1;

        const rotateTimer = setInterval(() => {
            const totalPages = Math.max(1, Math.ceil(telemetry.sota.length / SOTA_PAGE_SIZE));
            if (totalPages <= 1) return;
            state.sotaPage = (state.sotaPage + 1) % totalPages;
            if (state.sotaPage === 0) {
                state.sotaCycleComplete = true;
            }
        }, SOTA_ROTATE_MS);

        const subsysTimer = setInterval(() => {
            if (subsysTotalPages <= 1) return;
            state.subsystemPage = (state.subsystemPage + 1) % subsysTotalPages;
            if (state.subsystemPage === 0) {
                state.subsystemCycleComplete = true;
            }
        }, SUBSYS_ROTATE_MS);

        cleanup(() => {
            clearInterval(rotateTimer);
            clearInterval(subsysTimer);
        });
    });

    useVisibleTask$(({ cleanup }) => {
        // Force-init the registry onto window if not already there
        if (!(window as any).__loadingRegistry) {
            (window as any).__loadingRegistry = registryRef;
        }
        const getTracker = () => (window as any).__verboseTracker || trackerRef;
        const getRegistry = () => (window as any).__loadingRegistry || registryRef;

        const loop = setInterval(() => {
            const t = getTracker();
            if (!t) return;

            const s = t.state as TrackerState;
            const now = performance.now();
            const reg = getRegistry();

            // Update stats - ensure reasonable bounds
            // Merge progress from LoadingRegistry (early phases) and verbose-tracker
            const regProgress = reg ? reg.getProgress() : null;
            state.percent = Math.min(100, Math.max(0, regProgress?.percent || s.progressPercent || 0));
            state.count = regProgress?.completed || s.completedCount || 0;
            state.total = regProgress?.total || Math.max(s.totalCount || MANIFEST_STATS.total, state.count); // Total should never be less than count

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

            if (reg && (now - state.lastRegistryUpdate > 200)) {
                const stages = Array.from(reg.getStages().values());
                const sorted = stages.sort((a: any, b: any) => {
                    const rank = (status: string) => {
                        if (status === "loading") return 0;
                        if (status === "error") return 1;
                        if (status === "pending") return 2;
                        return 3;
                    };
                    return rank(a.status) - rank(b.status);
                });
                state.registryLines = sorted.slice(0, 12).map((stage: any) => ({
                    id: stage.stage.id,
                    name: stage.stage.name,
                    status: stage.status,
                    duration: stage.duration,
                    error: stage.error
                }));
                state.lastRegistryUpdate = now;
            }

            // Completion - trigger exit animation
            const elapsed = now - state.startedAt;
            const minDisplay = 1400;
            const maxDisplay = 3800;
            const currentSubsystemLineCount = Object.keys((subsystemRegistry as SubsystemRegistry).subsystems || {}).length;

            const canExit = elapsed >= minDisplay
                && (state.sotaCycleComplete || telemetry.sota.length <= SOTA_PAGE_SIZE)
                && (state.subsystemCycleComplete || currentSubsystemLineCount <= SUBSYS_PAGE_SIZE);

            if ((s.progressPercent >= 100 && state.queueIndex >= queue.length && canExit) || elapsed >= maxDisplay) {
                if (!state.ready) {
                    state.ready = true;
                    // Start exit animation
                    setTimeout(() => {
                        exiting.value = true;
                        // After exit animation completes, hide completely
                        setTimeout(() => {
                            visible.value = false;
                            document.body.classList.add("hydrated");
                        }, 800); // Match CSS exit animation duration
                    }, 400);
                }
            }

            // Safety timeout
            if ((window as any).__pluribusReady && !state.ready) {
                state.percent = 100;
                state.ready = true;
                exiting.value = true;
                setTimeout(() => { visible.value = false; }, 800);
            }

        }, 50);

        const timeout = setTimeout(() => {
            exiting.value = true;
            setTimeout(() => {
                visible.value = false;
                document.body.classList.add("hydrated");
            }, 800);
        }, 4200);

        cleanup(() => {
            clearInterval(loop);
            clearTimeout(timeout);
        });
    });

    if (!visible.value) return null;

    const overlayClass = `loading-overlay ${entered.value ? 'entered' : ''} ${exiting.value ? 'exiting' : ''}`;

    const statusFrom = (value: string, loaded: boolean): BootLine["status"] => {
        if (!loaded) return "wait";
        const normalized = value.toLowerCase();
        if (normalized === "healthy" || normalized === "ok") return "ok";
        if (normalized === "degraded" || normalized === "warning") return "warn";
        if (normalized === "error" || normalized === "failed") return "fail";
        return "wait";
    };

    const formatMeta = (line: BootLine) => {
        if (line.detail && line.meta) return `${line.detail} | ${line.meta}`;
        return line.detail || line.meta || "";
    };

    const formatIsoShort = (value?: string) => value ? value.slice(11, 19) : "";
    const formatDurationShort = (seconds?: number) => {
        if (typeof seconds !== "number" || Number.isNaN(seconds)) return "";
        if (seconds < 60) return `${Math.round(seconds)}s`;
        const minutes = Math.floor(seconds / 60);
        const secs = Math.round(seconds % 60);
        return `${minutes}m${secs}s`;
    };

    const bootLines: BootLine[] = [
        {
            id: "boot-progress",
            status: state.percent >= 100 ? "ok" : "wait",
            label: "boot-seq",
            detail: `${Math.round(state.percent)}%`,
            meta: `${state.count}/${state.total}`
        } as BootLine,
        ...state.registryLines.map((line) => ({
            id: `reg-${line.id}`,
            status: (line.status === "complete" ? "ok" : line.status === "error" ? "fail" : line.status === "loading" ? "wait" : "warn") as any,
            label: line.name,
            detail: line.duration ? `${Math.round(line.duration)}ms` : line.status,
            meta: line.error ? String(line.error).slice(0, 20) : undefined
        } as BootLine)),
        ...state.visibleItems.map((item) => ({
            id: `trace-${item.id}`,
            status: "ok",
            label: item.desc,
            detail: item.time,
            meta: item.seg
        } as BootLine))
    ].slice(0, 15);

    const metatestRun = telemetry.metatest.lastRun;
    const metatestMetaParts = [
        metatestRun?.scope,
        metatestRun?.status,
        formatIsoShort(metatestRun?.finished_at || metatestRun?.started_at),
        formatDurationShort(metatestRun?.duration_s)
    ].filter(Boolean) as string[];
    const metatestMeta = metatestMetaParts.length > 0 ? metatestMetaParts.join(" ") : undefined;

    // Protocol version entries for verbose System State
    const protocolLines: BootLine[] = [
        { id: "proto-holon", status: "ok", label: "HOLON", detail: "v2", meta: "envelope" } as BootLine,
        { id: "proto-dkin", status: "ok", label: "DKIN", detail: "v30", meta: "kernel" } as BootLine,
        { id: "proto-citizen", status: "ok", label: "CITIZEN", detail: "v2", meta: "compliance" } as BootLine,
        { id: "proto-paip", status: "ok", label: "PAIP", detail: "v16", meta: "isolation" } as BootLine,
        { id: "proto-uniform", status: "ok", label: "UNIFORM", detail: "v2.1", meta: "header" } as BootLine,
    ];

    const systemLines: BootLine[] = [
        ...protocolLines,
        {
            id: "bus-events",
            status: telemetry.bus.loaded ? "ok" : "wait",
            label: "bus",
            detail: `${telemetry.bus.totalEvents} ev`,
            meta: `vel=${telemetry.bus.velocity.toFixed(1)}/m t=${telemetry.bus.topics} a=${telemetry.bus.agents}`
        } as BootLine,
        {
            id: "dialogos",
            status: statusFrom(telemetry.dialogos.status, telemetry.dialogos.loaded),
            label: "dialogos",
            detail: telemetry.dialogos.status,
            meta: `${telemetry.dialogos.records} rec`
        } as BootLine,
        {
            id: "falkor",
            status: statusFrom(telemetry.falkor.status, telemetry.falkor.loaded),
            label: "falkordb",
            detail: telemetry.falkor.status,
            meta: `n=${telemetry.falkor.nodes} e=${telemetry.falkor.edges}`
        } as BootLine,
        {
            id: "semops",
            status: telemetry.semops.loaded ? "ok" : "wait",
            label: "semops",
            detail: `${telemetry.semops.operators} ops`,
            meta: `${telemetry.semops.commands} cmd`
        } as BootLine,
        {
            id: "metatest",
            status: telemetry.metatest.loaded ? "ok" : "wait",
            label: "metatest",
            detail: `${telemetry.metatest.totalTests} tests`,
            meta: metatestMeta || `cov=${telemetry.metatest.coverage.toFixed(1)}%`
        } as BootLine
    ];

    const subsystemRegistryData = subsystemRegistry as SubsystemRegistry;
    const subsystemCategories = subsystemRegistryData.categories || {};
    const subsystemMap = subsystemRegistryData.subsystems || {};
    const subsystemLines: BootLine[] = Object.entries(subsystemMap).map(([key, entries]) => {
        const list = Array.isArray(entries) ? entries : [];
        const category = subsystemCategories[key] || {};
        const label = (category.label || key).toLowerCase().replace(/\s+/g, "-");
        return {
            id: `subsys-${key}`,
            status: list.length > 0 ? "ok" : "wait",
            label,
            detail: `${list.length} units`,
            meta: category.label || key
        } as BootLine;
    });

    const subsystemStart = state.subsystemPage * SUBSYS_PAGE_SIZE;
    const subsystemSlice = subsystemLines.slice(subsystemStart, subsystemStart + SUBSYS_PAGE_SIZE);
    const aiosLines = [...systemLines, ...subsystemSlice].slice(0, 15);

    const sotaPageSize = SOTA_PAGE_SIZE;
    const sotaStart = state.sotaPage * sotaPageSize;
    const sotaSlice = telemetry.sota.slice(sotaStart, sotaStart + sotaPageSize);
    const sotaLines: BootLine[] = [
        {
            id: "sota-count",
            status: telemetry.sota.length > 0 ? "ok" : "wait",
            label: "sota",
            detail: `${telemetry.sota.length} items`,
            meta: state.sotaCycleComplete ? "cycle ok" : "streaming"
        } as BootLine,
        ...telemetry.busEvents.slice(0, 4).map((event, idx) => ({
            id: `bus-${idx}`,
            status: "ok" as any,
            label: event.topic || "event",
            detail: event.actor || "bus",
            meta: event.iso ? event.iso.slice(11, 19) : undefined
        } as BootLine)),
        ...sotaSlice.map((item, idx) => ({
            id: `sota-${item.id}-${idx}`,
            status: "ok" as any,
            label: item.title || item.id,
            detail: item.type || "tool",
            meta: typeof item.priority === "number"
                ? `p${item.priority}`
                : (item.tags && item.tags.length > 0
                    ? item.tags.slice(0, 2).join("/")
                    : (item.org || undefined))
        } as BootLine))
    ].slice(0, 15);

    // Skills Registry column - leveraging registry_protocol_v1 patterns
    const skillLines: BootLine[] = [
        {
            id: "skills-summary",
            status: telemetry.skills.loaded ? "ok" : "wait",
            label: "skills",
            detail: `${telemetry.skills.total} total`,
            meta: `pb=${telemetry.skills.sources.pbskills} reg=${telemetry.skills.sources.registries} oss=${telemetry.skills.sources.oss}`
        } as BootLine,
        ...telemetry.skills.items.slice(0, 12).map((skill, idx) => ({
            id: `skill-${skill.id}-${idx}`,
            status: skill.status || "ok",
            label: skill.name.slice(0, 18),
            detail: skill.source || "skill",
            meta: skill.tower_level ? `L${skill.tower_level}` : (skill.tags?.slice(0, 2).join("/") || undefined)
        } as BootLine))
    ].slice(0, 15);

    return (
        <div class={overlayClass}>
            {/* M3 Elevation */}
            <md-elevation class="loading-overlay-elevation"></md-elevation>
            <div class="loading-content">

                {/* Animated Logo Orb - Shader */}
                <div class="loading-orb juicy-element" style="--stagger: 0">
                    <div class={`loading-glow ${shaderReady.value ? 'visible' : ''}`} />
                    <CliffordTorusShader />
                </div>

                {/* Title */}
                <h2 class="loading-title juicy-element" style="--stagger: 1">Pluribus</h2>

                {/* M3 Progress Bar */}
                <div class="loading-progress juicy-element" style="--stagger: 2">
                    <md-linear-progress
                        class="loading-progress-m3"
                        value={state.percent / 100}
                    ></md-linear-progress>
                    <div class="loading-progress-glow-m3" style={{ left: `${state.percent}%` }} />
                </div>

                {/* Stats - Fixed format */}
                <div class="loading-stats juicy-element" style="--stagger: 3">
                    <span class="loading-count">{state.count} / {state.total}</span>
                    <span class="loading-percent">{Math.round(state.percent)}%</span>
                </div>

                {/* Boot Matrix */}
                <div class="loading-matrix juicy-element" style="--stagger: 4">
                    <div class="loading-column">
                        <div class="column-title">Boot Trace</div>
                        {bootLines.length > 0 ? bootLines.map((line) => (
                            <div class={`boot-line ${line.status}`} key={line.id}>
                                <span class={`boot-status ${line.status}`}>{line.status}</span>
                                <span class="boot-label">{line.label}</span>
                                <span class="boot-meta">{formatMeta(line)}</span>
                            </div>
                        )) : (
                            <div class="boot-line wait">
                                <span class="boot-status wait">wait</span>
                                <span class="boot-label">Initializing</span>
                                <span class="boot-meta">...</span>
                            </div>
                        )}
                    </div>
                    <div class="loading-column">
                        <div class="column-title">System State</div>
                        {aiosLines.map((line) => (
                            <div class={`boot-line ${line.status}`} key={line.id}>
                                <span class={`boot-status ${line.status}`}>{line.status}</span>
                                <span class="boot-label">{line.label}</span>
                                <span class="boot-meta">{formatMeta(line)}</span>
                            </div>
                        ))}
                    </div>
                    <div class="loading-column">
                        <div class="column-title">Catalog Stream</div>
                        {sotaLines.map((line) => (
                            <div class={`boot-line ${line.status}`} key={line.id}>
                                <span class={`boot-status ${line.status}`}>{line.status}</span>
                                <span class="boot-label">{line.label}</span>
                                <span class="boot-meta">{formatMeta(line)}</span>
                            </div>
                        ))}
                    </div>
                    <div class="loading-column">
                        <div class="column-title">Learning Systems</div>
                        {telemetry.learning.length > 0 ? telemetry.learning.map((line) => (
                            <div class={`boot-line ${statusFrom(line.status, true)}`} key={line.id}
                                style={{ opacity: Math.max(0.4, 1.0 - (line.uncertainty * 2)) }}
                            >
                                <span class={`boot-status ${statusFrom(line.status, true)}`}>{line.status}</span>
                                <span class="boot-label" title={`Uncertainty: ${line.uncertainty.toFixed(2)}`}>{line.label}</span>
                                <span class="boot-meta">{line.detail} | {line.meta}</span>
                            </div>
                        )) : (
                            <div class="boot-line wait">
                                <span class="boot-status wait">wait</span>
                                <span class="boot-label">Connecting</span>
                                <span class="boot-meta">...</span>
                            </div>
                        )}
                    </div>
                    <div class="loading-column">
                        <div class="column-title">Skills Registry</div>
                        {skillLines.length > 0 ? skillLines.map((line) => (
                            <div class={`boot-line ${line.status}`} key={line.id}>
                                <span class={`boot-status ${line.status}`}>{line.status}</span>
                                <span class="boot-label">{line.label}</span>
                                <span class="boot-meta">{formatMeta(line)}</span>
                            </div>
                        )) : (
                            <div class="boot-line wait">
                                <span class="boot-status wait">wait</span>
                                <span class="boot-label">Discovering</span>
                                <span class="boot-meta">...</span>
                            </div>
                        )}
                    </div>
                </div>

            </div>
        </div>
    );
});
