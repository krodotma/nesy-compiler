/**
 * OMEGA WORKER (The Bread)
 * ========================
 * Independent "Brainstem" process for Pluribus.
 * Maintains Bus connection, Heartbeat, and Entropy calculation
 * regardless of the UI thread's state.
 */

// Worker context definitions
const ctx: Worker = self as any;

type PressureState = 'nominal' | 'elevated' | 'critical' | 'unknown';

interface MemoryTelemetry {
    usedPct: number;
    swapUsedPct: number;
    psiSome: number;
    psiFull: number;
    pressure: PressureState;
}

interface OmegaState {
    timestamp: number;
    connected: boolean;
    rings: [number, number, number, number]; // 0=Const, 1=Git, 2=Agent, 3=Bus
    entropy: number;
    beat: number;
    metrics: {
        eventRate: number;
        workerCount: number;
        lastError: string | null;
        memory: MemoryTelemetry;
    };
}

let ws: WebSocket | null = null;
let state: OmegaState = {
    timestamp: Date.now(),
    connected: false,
    rings: [1.0, 1.0, 1.0, 1.0],
    entropy: 0.1,
    beat: 0.0,
    metrics: {
        eventRate: 0,
        workerCount: 0,
        lastError: null,
        memory: { usedPct: 0, swapUsedPct: 0, psiSome: 0, psiFull: 0, pressure: 'unknown' }
    }
};

const broadcast = new BroadcastChannel('pluribus-omega');
const broadcastMessage = (payload: unknown) => {
    try {
        broadcast.postMessage(payload);
    } catch {
        // ignore
    }
};

// Bus Event Buffer for Entropy Calculation
const eventBuffer: number[] = []; // timestamps
let errorCount = 0;
const ENTROPY_WINDOW_MS = 10000;

// Heartbeat Loop (60fps)
// We calculate the 'beat' phase here to ensure smooth animation even if UI lags.
let lastFrame = performance.now();
const BEAT_FREQ = 1.0; // Hz

function loop() {
    const now = performance.now();
    const dt = (now - lastFrame) / 1000;
    lastFrame = now;

    // Pulse Logic (Logistic Beat)
    // t is monotonic time
    const t = now / 1000 * BEAT_FREQ;
    const x = t % 1.0;
    // Systolic pulse: sharp attack, exponential decay
    state.beat = (x < 0.2) 
        ? (x / 0.2) // Attack
        : Math.exp(-(x - 0.2) * 5.0); // Decay

    // Entropy Decay
    if (Math.random() < 0.01) {
        state.entropy = Math.max(0.1, state.entropy * 0.99); // Slow recovery
    }

    // Broadcast to both the owning UI thread and any broadcast listeners.
    postMessage({ type: 'OMEGA_TICK', state });
    broadcastMessage({ type: 'OMEGA_TICK', state });
    
    requestAnimationFrame(loop);
}

// Connect to Bus
function connect() {
    // Determine protocol
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    // When running in a worker, location.host might be the CDN or blob, but usually relative works if served same-origin.
    // However, for Vite dev, we might need to be explicit.
    const wsUrl = `${protocol}//${location.host}/ws/bus`;

    try {
        ws = new WebSocket(wsUrl);
        
        ws.onopen = () => {
            state.connected = true;
            state.rings[3] = 1.0; // Bus is Ring 3
            console.log('[Omega] Connected to Bus');
        };

        ws.onclose = () => {
            state.connected = false;
            state.rings[3] = 0.0;
            setTimeout(connect, 2000);
        };

        ws.onmessage = (msg) => {
            try {
                const data = JSON.parse(msg.data);
                if (data.type === 'event' && data.event) {
                    const ev = data.event;
                    const now = Date.now();

                    const updateMemory = (memory: any) => {
                        if (!memory || typeof memory !== 'object') return;
                        const toNumber = (value: unknown, fallback: number) =>
                            typeof value === 'number' && !Number.isNaN(value) ? value : fallback;
                        const pressure =
                            typeof memory.pressure_state === 'string'
                                ? (memory.pressure_state as PressureState)
                                : state.metrics.memory.pressure;
                        state.metrics.memory = {
                            usedPct: toNumber(memory.mem_used_pct, state.metrics.memory.usedPct),
                            swapUsedPct: toNumber(memory.swap_used_pct, state.metrics.memory.swapUsedPct),
                            psiSome: toNumber(memory.psi_some_avg10, state.metrics.memory.psiSome),
                            psiFull: toNumber(memory.psi_full_avg10, state.metrics.memory.psiFull),
                            pressure
                        };
                    };
                    
                    // Update Metrics
                    eventBuffer.push(now);
                    // Prune old
                    while (eventBuffer.length > 0 && eventBuffer[0] < now - ENTROPY_WINDOW_MS) {
                        eventBuffer.shift();
                    }
                    state.metrics.eventRate = eventBuffer.length / (ENTROPY_WINDOW_MS / 1000);

                    if (ev.topic === 'ohm.status' && ev.data && ev.data.memory) {
                        updateMemory(ev.data.memory);
                    }
                    if (ev.topic === 'ohm.memory.pressure' && ev.data) {
                        updateMemory(ev.data);
                    }

                    // Entropy Injection
                    if (ev.level === 'error') {
                        state.entropy = Math.min(1.0, state.entropy + 0.1);
                        state.rings[0] *= 0.9; // Penalty to Ring 0 (Constitution/Safety)
                        state.metrics.lastError = ev.topic;
                    }

                    // Ring 2: Workers
                    if (ev.topic.includes('strp.worker.start')) state.metrics.workerCount++;
                    if (ev.topic.includes('strp.worker.end')) state.metrics.workerCount--;
                    state.rings[2] = state.metrics.workerCount > 0 ? 1.0 : 0.5;

                    // Ring 0: Omega Heartbeat (The real signal)
                    if (ev.topic === 'omega.heartbeat') {
                        // Resync beat phase if needed, or just acknowledge
                        state.rings[0] = 1.0;
                    }

                    // System Reload Trigger
                    if (ev.topic === 'system.reload' || ev.topic === 'nucleus.version.cycle') {
                        console.log('[Omega] Version cycle detected. Triggering UI reload.');
                        broadcastMessage({ type: 'OMEGA_RELOAD_REQUIRED', reason: ev.topic });
                    }

                    // Relay specific events to UI via BroadcastChannel (SOTA: Single Socket)
                    const relayTopics = [
                        'system.boot.log', 
                        'infercell.genesis', 
                        'infercell.fork', 
                        'dashboard.vps.provider_status',
                        'omega.heartbeat'
                    ];
                    if (relayTopics.includes(ev.topic)) {
                        broadcastMessage({ type: 'BUS_EVENT', event: ev });
                    }
                }
            } catch (e) {
                // ignore
            }
        };
    } catch (e) {
        console.error('[Omega] Connection failed', e);
        setTimeout(connect, 5000);
    }
}

// Start
connect();
requestAnimationFrame(loop);

// Listen for commands from Main Thread
ctx.onmessage = (ev) => {
    if (ev.data.type === 'PING') {
        postMessage({ type: 'PONG' });
    }
};
