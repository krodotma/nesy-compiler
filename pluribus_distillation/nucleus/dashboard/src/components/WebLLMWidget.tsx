/**
 * WebLLMWidget - Multi-Session Browser LLM Inference
 *
 * Features:
 * - Up to 3 concurrent model sessions
 * - Models can converse with each other
 * - Full-width chat spanning 30-40% viewport height
 * - WebGPU acceleration with cache detection
 * - Each session maintains independent context
 */

import { component$, useSignal, useStore, $, useVisibleTask$, noSerialize, type NoSerialize } from '@builder.io/qwik';
import { createBusClient } from '../lib/bus/bus-client';
import type { BusEvent } from '../lib/state/types';
import {
  buildModelId,
  formatDialogosPeerMessage,
  getAutoCacheModels,
  getAvailableModels,
  getDialogosModels,
  getModelDefByModelId,
  getOmegaIntervention,
  selectDialogosSeed,
} from '../lib/webllm-enhanced';
import { DualMindChannel, type ConversationMode } from '../lib/dual-mind-channel';

// Types
interface GPUAdapterInfo {
  vendor?: string;
  device?: string;
  description?: string;
  architecture?: string;
}

interface GPUAdapter {
  name?: string;
  requestAdapterInfo?: () => Promise<GPUAdapterInfo>;
}

interface GPU {
  requestAdapter(options?: { powerPreference?: string }): Promise<GPUAdapter | null>;
}

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

interface ChatSession {
  id: string;
  modelId: string;
  modelName: string;
  engine: NoSerialize<unknown> | null;
  messages: ChatMessage[];
  status: 'idle' | 'loading' | 'ready' | 'generating';
  loadProgress: number;
  loadStage: string;
  error: string | null;
}

interface WebLLMState {
  webgpuStatus: 'checking' | 'ready' | 'unsupported';
  gpuInfo: string | null;
  error: string | null;
  cachedModels: string[];
  sessions: ChatSession[];
  conversationMode: boolean;
  currentMode: ConversationMode;
  turnCount: number;
  autoCacheEnabled: boolean;
  autoCacheStatus: 'idle' | 'running' | 'complete' | 'error';
  autoCacheTargets: string[];
  autoCacheQueue: string[];
  autoCacheCompleted: string[];
  autoCacheProgress: number;
  autoCacheStage: string;
  autoCacheError: string | null;
  deviceMemoryGB: number | null;
}

const AVAILABLE_MODELS = getAvailableModels();
const AUTO_CHAT_CADENCE_MS = 15_000;
const AUTO_CHAT_STALL_MS = 45_000;
const AUTO_CHAT_POLL_MS = 1_000;
const WEBLLM_PREF_KEY = 'webllm_enabled';
const MAX_SESSION_HISTORY = 50; // Prevent memory degradation on long conversations

function readWebLLMPreference(): boolean {
  if (typeof localStorage === 'undefined') return true;
  try {
    const raw = localStorage.getItem(WEBLLM_PREF_KEY);
    if (raw === null) {
      localStorage.setItem(WEBLLM_PREF_KEY, 'true');
      return true;
    }
    return raw === 'true' || raw === '1';
  } catch {
    return true;
  }
}

function writeWebLLMPreference(enabled: boolean): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(WEBLLM_PREF_KEY, enabled ? 'true' : 'false');
  } catch {
    // ignore storage failures
  }
}

function getDeviceMemoryGB(): number {
  const memory = typeof navigator !== 'undefined'
    ? Number((navigator as Navigator & { deviceMemory?: number }).deviceMemory || 0)
    : 0;
  return memory > 0 ? memory : 6;
}

function uniq(items: string[]): string[] {
  return Array.from(new Set(items));
}

const MODEL_COLORS: Record<string, string> = {
  cyan: 'border-cyan-500/50 bg-cyan-500/10',
  purple: 'border-purple-500/50 bg-purple-500/10',
  blue: 'border-blue-500/50 bg-blue-500/10',
  green: 'border-green-500/50 bg-green-500/10',
  orange: 'border-orange-500/50 bg-orange-500/10',
};

const MODEL_TEXT_COLORS: Record<string, string> = {
  cyan: 'text-cyan-400',
  purple: 'text-purple-400',
  blue: 'text-blue-400',
  green: 'text-green-400',
  orange: 'text-orange-400',
};

// Cache detection
async function getCachedModels(): Promise<string[]> {
  if (typeof caches === 'undefined') return [];
  try {
    const cache = await caches.open('webllm/model');
    const keys = await cache.keys();
    return AVAILABLE_MODELS
      .filter(m => keys.some(req => req.url.includes(m.id)))
      .map(m => m.id);
  } catch {
    return [];
  }
}

function createSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
}

export interface WebLLMWidgetProps {
  /** Expand to fill the parent container (use with an absolute inset wrapper). */
  fullScreen?: boolean;
}

export const WebLLMWidget = component$<WebLLMWidgetProps>(({ fullScreen = false }) => {
  const state = useStore<WebLLMState>({
    webgpuStatus: 'checking',
    gpuInfo: null,
    error: null,
    cachedModels: [],
    sessions: [],
    conversationMode: true,
    currentMode: 'exploration',
    turnCount: 0,
    autoCacheEnabled: true,
    autoCacheStatus: 'idle',
    autoCacheTargets: [],
    autoCacheQueue: [],
    autoCacheCompleted: [],
    autoCacheProgress: 0,
    autoCacheStage: '',
    autoCacheError: null,
    deviceMemoryGB: null,
  });

  const expanded = useSignal(true); // Visible by default to fulfill always-on narrative
  const enabled = useSignal(true);
  const busBound = useSignal(false);
  const autostarted = useSignal(false);
  const autoCacheStarted = useSignal(false);
  const dialogosActive = useSignal(false);
  const dialogosPending = useSignal<{ sessionId: string; prompt: string; origin: 'dialogos' | 'auto' } | null>(null);
  const dialogosTimer = useSignal<NoSerialize<ReturnType<typeof setTimeout>> | null>(null);
  const lastDialogosSendAt = useSignal(0);
  const lastAutoRelaySignature = useSignal('');
  const autoChatTurn = useSignal(0);
  const publishRef = useSignal<NoSerialize<(event: Omit<BusEvent, 'ts' | 'iso'>) => Promise<void>> | null>(null);
  const channelRef = useSignal<NoSerialize<DualMindChannel> | null>(null);

  // Always-on by default; honor user override if they disable it.
  useVisibleTask$(() => {
    const preferEnabled = readWebLLMPreference();
    enabled.value = preferEnabled;
    if (preferEnabled && typeof navigator !== 'undefined' && (navigator as any).gpu) {
      console.log('[WebLLM] WebGPU detected, always-on inference enabled.');
    }
  }, { strategy: 'document-ready' });

  const enableWebLLM = $(() => {
    if (enabled.value) return;
    enabled.value = true;
    writeWebLLMPreference(true);
    state.webgpuStatus = 'checking';
    state.error = null;
  });

  const emitBus = $((event: Omit<BusEvent, 'ts' | 'iso'>) => {
    const publish = publishRef.value;
    if (!publish) return;
    void publish(event);
  });

  const computeAutoCacheTargets = $(() => {
    const memoryGB = getDeviceMemoryGB();
    state.deviceMemoryGB = memoryGB;
    const models = getAutoCacheModels({ deviceMemoryGB: memoryGB });
    const targetIds = models.map((m) => buildModelId(m.baseId, m.defaultQuant));
    state.autoCacheTargets = targetIds;
    state.autoCacheCompleted = targetIds.filter((id) => state.cachedModels.includes(id));
    state.autoCacheQueue = targetIds.filter((id) => !state.cachedModels.includes(id));
    state.autoCacheProgress = targetIds.length
      ? Math.round((state.autoCacheCompleted.length / targetIds.length) * 100)
      : 0;
  });

  const ensureAutoSessions = $(() => {
    const dialogosTargets = getDialogosModels().map((m) => buildModelId(m.baseId, m.defaultQuant));
    const desired = dialogosTargets.length > 0
      ? dialogosTargets
      : state.autoCacheTargets.length > 0
        ? state.autoCacheTargets
        : AVAILABLE_MODELS.map((m) => m.id);
    const targetIds = desired.slice(0, 2);
    const used = new Set(state.sessions.map((s) => s.modelId));

    for (const modelId of targetIds) {
      if (state.sessions.length >= 2) break;
      if (used.has(modelId)) continue;
      const model = AVAILABLE_MODELS.find((m) => m.id === modelId) || AVAILABLE_MODELS[0];
      if (!model) break;
      state.sessions.push({
        id: createSessionId(),
        modelId: model.id,
        modelName: model.name,
        engine: null,
        messages: [],
        status: 'idle',
        loadProgress: 0,
        loadStage: '',
        error: null,
      });
      used.add(model.id);
    }
  });

  const unloadEngine = $((engine: unknown) => {
    try {
      const unload = (engine as { unload?: () => Promise<void> }).unload;
      if (typeof unload === 'function') {
        const result = unload();
        if (result && typeof (result as Promise<void>).catch === 'function') {
          void (result as Promise<void>).catch(() => {});
        }
      }
    } catch {
      // ignore unload failures
    }
  });

  const disableWebLLM = $(() => {
    if (!enabled.value) return;
    enabled.value = false;
    writeWebLLMPreference(false);

    autostarted.value = false;
    autoCacheStarted.value = false;
    dialogosActive.value = false;
    dialogosPending.value = null;
    lastDialogosSendAt.value = 0;
    lastAutoRelaySignature.value = '';
    autoChatTurn.value = 0;
    if (dialogosTimer.value) {
      clearTimeout(dialogosTimer.value);
      dialogosTimer.value = null;
    }

    for (const session of state.sessions) {
      if (session.engine) void unloadEngine(session.engine);
    }

    state.sessions = [];
    state.webgpuStatus = 'checking';
    state.error = null;
    state.autoCacheStatus = 'idle';
    state.autoCacheStage = '';
    state.autoCacheError = null;
  });

  const warmCache = $(async (modelIds: string[]) => {
    if (!modelIds.length || !state.autoCacheEnabled) {
      state.autoCacheStatus = 'complete';
      return;
    }

    state.autoCacheStatus = 'running';
    state.autoCacheError = null;

    for (const modelId of modelIds) {
      if (!state.autoCacheEnabled) break;
      if (state.cachedModels.includes(modelId)) continue;

      const model = AVAILABLE_MODELS.find((m) => m.id === modelId);
      state.autoCacheStage = model ? `Caching ${model.name}...` : `Caching ${modelId}...`;
      void emitBus({
        topic: 'webllm.cache.start',
        kind: 'metric',
        level: 'info',
        actor: 'webllm-widget',
        data: { model_id: modelId, model_name: model?.name || null },
      });

      try {
        const webllm = await import('@mlc-ai/web-llm');
        const engine = await webllm.CreateMLCEngine(modelId, {
          initProgressCallback: (progress: { text: string; progress: number }) => {
            state.autoCacheProgress = Math.round(progress.progress * 100);
            state.autoCacheStage = progress.text;
          },
        });
        await engine.unload();

        state.cachedModels = uniq([...state.cachedModels, modelId]);
        state.autoCacheCompleted = uniq([...state.autoCacheCompleted, modelId]);
        void emitBus({
          topic: 'webllm.cache.ready',
          kind: 'artifact',
          level: 'info',
          actor: 'webllm-widget',
          data: { model_id: modelId, model_name: model?.name || null },
        });
      } catch (err) {
        // Log error but continue caching other models
        const errorMsg = err instanceof Error ? err.message : 'Cache warm failed';
        console.warn(`[WebLLM] Failed to cache ${modelId}:`, errorMsg);
        
        void emitBus({
          topic: 'webllm.cache.error',
          kind: 'artifact',
          level: 'error',
          actor: 'webllm-widget',
          data: { model_id: modelId, error: errorMsg },
        });
      }
    }

    void computeAutoCacheTargets();
    state.autoCacheStatus = 'complete';
    state.autoCacheStage = state.autoCacheQueue.length === 0 ? 'Cache ready' : 'Cache cycle complete';
  });

  const startAutoSessions = $(async () => {
    if (autostarted.value) return;
    autostarted.value = true;

    const sessionsToLoad = state.sessions.slice(0, 2);
    for (const session of sessionsToLoad) {
      await loadModel(session.id);
      void emitBus({
        topic: 'webllm.session.auto_started',
        kind: 'artifact',
        level: 'info',
        actor: 'webllm-widget',
        data: { session_id: session.id, model_id: session.modelId, model_name: session.modelName },
      });
    }
  });

  const startAutoCache = $(async () => {
    if (!state.autoCacheEnabled || autoCacheStarted.value) return;
    autoCacheStarted.value = true;
    if (state.autoCacheQueue.length === 0) {
      state.autoCacheStatus = 'complete';
      return;
    }
    await warmCache(state.autoCacheQueue);
  });

  // Initialize WebGPU and cache detection
  useVisibleTask$(async ({ track }) => {
    track(() => enabled.value);
    if (!enabled.value) return;

    if (typeof window === 'undefined' || typeof navigator === 'undefined') {
      state.webgpuStatus = 'unsupported';
      state.error = 'Not in browser context';
      return;
    }

    const isSecureContext = window.isSecureContext;
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';

    if (!isSecureContext && !isLocalhost) {
      state.webgpuStatus = 'unsupported';
      state.error = 'WebGPU requires HTTPS';
      return;
    }

    const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
    if (!gpu) {
      state.webgpuStatus = 'unsupported';
      state.error = 'WebGPU API not available';
      return;
    }

    try {
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (adapter) {
        const info = await adapter.requestAdapterInfo?.() || {};
        state.gpuInfo = info.description || info.device || info.vendor || 'WebGPU';
        state.webgpuStatus = 'ready';

        // Check cache
        state.cachedModels = await getCachedModels();
        void computeAutoCacheTargets();
        void ensureAutoSessions();

        if (!autostarted.value || (state.autoCacheEnabled && !autoCacheStarted.value)) {
          queueMicrotask(() => {
            void (async () => {
              await startAutoSessions();
              void computeAutoCacheTargets();
              await startAutoCache();
            })();
          });
        }
      } else {
        state.webgpuStatus = 'unsupported';
        state.error = 'No WebGPU adapter found';
      }
    } catch (err) {
      state.webgpuStatus = 'unsupported';
      state.error = err instanceof Error ? err.message : 'WebGPU init failed';
    }
  }, { strategy: 'document-idle' });

  // Supersymmetric binding: browser WebLLM can act as a bus-addressable inference surface.
  useVisibleTask$(({ track, cleanup }) => {
    track(() => enabled.value);
    if (!enabled.value) {
      busBound.value = false;
      dialogosActive.value = false;
      return;
    }

    let disposed = false;
    const client = createBusClient({ platform: 'browser' });
    let unsubInfer: (() => void) | null = null;
    let unsubStatus: (() => void) | null = null;
    let unsubDialogosSeed: (() => void) | null = null;
    let unsubDialogosRelay: (() => void) | null = null;
    let unsubDialogosOmega: (() => void) | null = null;
    let statusInterval: ReturnType<typeof setInterval> | null = null;

    const publish = async (event: Omit<BusEvent, 'ts' | 'iso'>) => {
      try {
        await client.publish(event);
      } catch {
        // Non-fatal: bus bridge may be down.
      }
    };

    const selectReadySession = (preferredId?: string | null) => {
      if (preferredId) {
        const s = state.sessions.find((x) => x.id === preferredId);
        if (s && s.status === 'ready' && s.engine) return s;
      }
      return state.sessions.find((s) => s.status === 'ready' && s.engine) || null;
    };

    const inferOnce = async (prompt: string, opts: { sessionId?: string | null; temperature?: number; maxTokens?: number }) => {
      const session = selectReadySession(opts.sessionId || null);
      if (!session || !session.engine || session.status !== 'ready') {
        throw new Error('webllm not ready (enable WebLLM and load a model session)');
      }
      const engine = session.engine as unknown as {
        chat: {
          completions: {
            create: (args: {
              messages: Array<{ role: string; content: string }>;
              temperature: number;
              max_tokens: number;
            }) => Promise<{ choices: Array<{ message: { content: string } }> }>;
          };
        };
      };
      const t = typeof opts.temperature === 'number' ? opts.temperature : 0.7;
      const max = typeof opts.maxTokens === 'number' ? opts.maxTokens : 256;
      const response = await engine.chat.completions.create({
        messages: [{ role: 'user', content: prompt }],
        temperature: t,
        max_tokens: max,
      });
      return {
        text: response.choices[0]?.message?.content || '',
        sessionId: session.id,
        modelId: session.modelId,
        modelName: session.modelName,
      };
    };

    const run = async () => {
      try {
        await client.connect();
        if (disposed) return;
        busBound.value = true;
        publishRef.value = noSerialize(publish);

        const scheduleDialogosSend = (sessionId: string, prompt: string, origin: 'dialogos' | 'auto') => {
          const now = Date.now();
          const elapsed = now - lastDialogosSendAt.value;

          const dispatch = (payload: { sessionId: string; prompt: string; origin: 'dialogos' | 'auto' }) => {
            lastDialogosSendAt.value = Date.now();
            void sendMessage(payload.sessionId, payload.prompt, { origin: payload.origin, emitBus: true });
          };

          if (elapsed >= AUTO_CHAT_CADENCE_MS && !dialogosTimer.value) {
            dispatch({ sessionId, prompt, origin });
            return;
          }

          dialogosPending.value = { sessionId, prompt, origin };
          if (!dialogosTimer.value) {
            const delay = Math.max(0, AUTO_CHAT_CADENCE_MS - elapsed);
            dialogosTimer.value = noSerialize(setTimeout(() => {
              const pending = dialogosPending.value;
              dialogosPending.value = null;
              dialogosTimer.value = null;
              if (pending) dispatch(pending);
            }, delay));
          }
        };

        unsubInfer = client.subscribe('webllm.infer.request', (ev) => {
          const data = (ev as any)?.data || {};
          const reqId = String((data as any)?.req_id || '');
          const prompt = String((data as any)?.prompt || '');
          const sessionId = typeof (data as any)?.session_id === 'string' ? String((data as any).session_id) : null;
          const temperature = typeof (data as any)?.temperature === 'number' ? (data as any).temperature : undefined;
          const maxTokens = typeof (data as any)?.max_tokens === 'number' ? (data as any).max_tokens : undefined;

          if (!reqId || !prompt) {
            publish({
              topic: 'webllm.infer.response',
              kind: 'response',
              level: 'error',
              actor: 'webllm-widget',
              data: { req_id: reqId || null, ok: false, error: 'missing req_id or prompt' },
            });
            return;
          }

          const start = performance.now();
          inferOnce(prompt, { sessionId, temperature, maxTokens })
            .then((out) => {
              publish({
                topic: 'webllm.infer.response',
                kind: 'response',
                level: 'info',
                actor: 'webllm-widget',
                data: {
                  req_id: reqId,
                  ok: true,
                  text: out.text,
                  latency_ms: Math.round(performance.now() - start),
                  session_id: out.sessionId,
                  model_id: out.modelId,
                  model_name: out.modelName,
                },
              });
            })
            .catch((err) => {
              publish({
                topic: 'webllm.infer.response',
                kind: 'response',
                level: 'error',
                actor: 'webllm-widget',
                data: { req_id: reqId, ok: false, error: String(err instanceof Error ? err.message : err) },
              });
            });
        });

        unsubDialogosSeed = client.subscribe('dialogos.seed', (ev) => {
          const data = (ev as any)?.data || {};
          const seed = String(data.seed_prompt || selectDialogosSeed());
          const targetIndex = Number.isFinite(data.target_session) ? Number(data.target_session) : 0;
          const readySessions = state.sessions.filter((s) => s.status === 'ready' && s.engine);
          const target = readySessions[targetIndex] || readySessions[0];
          if (!target) return;
          dialogosActive.value = true;
          state.conversationMode = true;
          scheduleDialogosSend(target.id, seed, 'dialogos');
        });

        unsubDialogosRelay = client.subscribe('dialogos.relay', (ev) => {
          const data = (ev as any)?.data || {};
          const peerMessage = String(data.peer_message || '');
          if (!peerMessage) return;
          const fromSession = String(data.from_session || '');
          const readySessions = state.sessions.filter((s) => s.status === 'ready' && s.engine);
          const target = readySessions.find((s) => s.id !== fromSession) || readySessions[0];
          if (!target) return;
          dialogosActive.value = true;
          state.conversationMode = true;
          scheduleDialogosSend(target.id, peerMessage, 'dialogos');
        });

        unsubDialogosOmega = client.subscribe('dialogos.omega', (ev) => {
          const data = (ev as any)?.data || {};
          const intervention = String(data.intervention || getOmegaIntervention(autoChatTurn.value));
          const readySessions = state.sessions.filter((s) => s.status === 'ready' && s.engine);
          const target = readySessions[0];
          if (!target) return;
          dialogosActive.value = true;
          state.conversationMode = true;
          scheduleDialogosSend(target.id, intervention, 'dialogos');
        });

        unsubStatus = client.subscribe('webllm.status.request', (ev) => {
          const data = (ev as any)?.data || {};
          const reqId = String((data as any)?.req_id || '');
          publish({
            topic: 'webllm.status.response',
            kind: 'response',
            level: 'info',
            actor: 'webllm-widget',
            data: {
              req_id: reqId || null,
              ok: true,
              enabled: enabled.value,
              webgpu_status: state.webgpuStatus,
              gpu_info: state.gpuInfo,
              cached_models: state.cachedModels,
              sessions: state.sessions.map((s) => ({
                id: s.id,
                model_id: s.modelId,
                model_name: s.modelName,
                status: s.status,
              })),
              bound: busBound.value,
              auto_cache: {
                enabled: state.autoCacheEnabled,
                status: state.autoCacheStatus,
                targets: state.autoCacheTargets.length,
                completed: state.autoCacheCompleted.length,
              },
              at: new Date().toISOString(),
            },
          });
        });

        statusInterval = setInterval(() => {
          if (!busBound.value) return;
          publish({
            topic: 'webllm.widget.status',
            kind: 'metric',
            level: 'info',
            actor: 'webllm-widget',
            data: {
              enabled: enabled.value,
              webgpu_status: state.webgpuStatus,
              gpu_info: state.gpuInfo,
              cached_models: state.cachedModels.length,
              sessions_total: state.sessions.length,
              sessions_ready: state.sessions.filter((s) => s.status === 'ready' && s.engine).length,
              auto_cache_status: state.autoCacheStatus,
              auto_cache_completed: state.autoCacheCompleted.length,
              auto_cache_targets: state.autoCacheTargets.length,
              at: new Date().toISOString(),
            },
          });
        }, 10000);
      } catch {
        busBound.value = false;
      }
    };

    run();

    cleanup(() => {
      disposed = true;
      if (statusInterval) clearInterval(statusInterval);
      try { unsubInfer?.(); } catch { /* ignore */ }
      try { unsubStatus?.(); } catch { /* ignore */ }
      try { unsubDialogosSeed?.(); } catch { /* ignore */ }
      try { unsubDialogosRelay?.(); } catch { /* ignore */ }
      try { unsubDialogosOmega?.(); } catch { /* ignore */ }
      if (dialogosTimer.value) {
        clearTimeout(dialogosTimer.value);
        dialogosTimer.value = null;
      }
      dialogosPending.value = null;
      try { client.disconnect(); } catch { /* ignore */ }
      busBound.value = false;
      publishRef.value = null;
    });
  }, { strategy: 'document-idle' });

  // OMEGA MONITOR & DUAL-MIND ORCHESTRATION
  useVisibleTask$(({ track, cleanup }) => {
    track(() => state.conversationMode);
    track(() => enabled.value);

    if (!enabled.value || !state.conversationMode) {
      return;
    }

    let cancelled = false;
    let activeChannel: DualMindChannel | null = null;
    let startInterval: ReturnType<typeof setInterval> | null = null;
    let starting = false;

    const tryStartChannel = async () => {
      if (cancelled || starting) return;
      if (!enabled.value || !state.conversationMode) return;
      if (channelRef.value) return;

      const readySessions = state.sessions.filter(s => s.status === 'ready' && s.engine);
      if (readySessions.length < 2) return;

      starting = true;
      const channel = new DualMindChannel({
        turnDelayMs: AUTO_CHAT_CADENCE_MS / 5, // Faster internal logic, external pacing controlled by cadence
        stallThresholdMs: AUTO_CHAT_STALL_MS,
      });

      channel.setInferenceFunction(async (mindIndex, prompt) => {
        const readyNow = state.sessions.filter(s => s.status === 'ready' && s.engine);
        const session = readyNow[mindIndex] || readyNow[0];
        if (!session) throw new Error('No ready mind for channel');

        // Bridge class logic to Qwik async sendMessage
        await sendMessage(session.id, prompt, { origin: 'auto', emitBus: true });

        // Return latest assistant response for the channel state
        const lastMsg = session.messages[session.messages.length - 1];
        return lastMsg?.role === 'assistant' ? lastMsg.content : '';
      });

      channel.onModeChange((from, to) => {
        state.currentMode = to;
        void emitBus({
          topic: 'dialogos.mode_transition',
          kind: 'metric',
          level: 'info',
          actor: 'webllm-arena',
          data: { from, to, turn: state.turnCount }
        });
      });

      channel.onMessageReceived((msg) => {
        state.turnCount = msg.turnNumber;
        autoChatTurn.value = msg.turnNumber;
      });

      try {
        await channel.start();
        if (cancelled) {
          channel.stop();
          return;
        }
        activeChannel = channel;
        channelRef.value = noSerialize(channel);
        if (startInterval) {
          clearInterval(startInterval);
          startInterval = null;
        }
      } catch (err) {
        console.error('[Arena] Channel failed:', err);
      } finally {
        starting = false;
      }
    };

    startInterval = setInterval(() => {
      void tryStartChannel();
    }, 1000);

    void tryStartChannel();

    cleanup(() => {
      cancelled = true;
      if (startInterval) clearInterval(startInterval);
      const existing = (channelRef.value as unknown as DualMindChannel | null) || activeChannel;
      if (existing) existing.stop();
      channelRef.value = null;
    });
  }, { strategy: 'document-idle' });

  // FALLBACK AUTO-CONVERSE LOOP (Simple multi-mind pattern when DualMindChannel unavailable)
  // Ported from WIP wip_webllm_autostart_2025-12-19_round2/WebLLMWidget.tsx lines 362-424
  useVisibleTask$(({ track, cleanup }) => {
    track(() => state.conversationMode);
    track(() => enabled.value);

    if (!enabled.value || !state.conversationMode) {
      return;
    }

    let cancelled = false;
    let lastFallbackActorId = '';
    let lastFallbackMsgTime = 0;
    let fallbackTurn = 0;

    const interval = setInterval(async () => {
      if (cancelled) return;
      if (!state.conversationMode || !enabled.value) return;

      // Only run fallback when DualMindChannel is NOT active
      if (channelRef.value) return;

      // Filter ready sessions
      const readySessions = state.sessions.filter(s => s.status === 'ready' && s.engine);
      if (readySessions.length < 2) return; // Need at least 2 to tango

      const now = Date.now();

      // Check global activity across all ready sessions
      let lastMsgTime = 0;
      let lastMsgContent = '';
      let lastActorId = '';

      readySessions.forEach(s => {
        if (s.messages.length > 0) {
          const m = s.messages[s.messages.length - 1];
          if (m.timestamp > lastMsgTime) {
            lastMsgTime = m.timestamp;
            lastMsgContent = m.content;
            lastActorId = s.id;
          }
        }
      });

      const timeSinceLast = now - lastMsgTime;

      // Case 1: Start fresh (Seed) - no messages yet
      if (lastMsgTime === 0) {
        const seed = selectDialogosSeed();
        console.log('[Omega Fallback] Seeding chat with:', seed.slice(0, 50) + '...');
        try {
          await sendMessage(readySessions[0].id, seed, { origin: 'auto', emitBus: true });
          fallbackTurn++;
          autoChatTurn.value = fallbackTurn;
        } catch (err) {
          console.error('[Omega Fallback] Seed failed:', err);
        }
        return;
      }

      // Case 2: Reply (Turn-taking) - wait 3-10s before replying to avoid spam
      if (timeSinceLast > 3000 && timeSinceLast < 10000 && lastActorId !== lastFallbackActorId) {
        // Find someone who ISN'T the last actor
        const nextAgent = readySessions.find(s => s.id !== lastActorId);
        if (nextAgent && nextAgent.status === 'ready') {
          console.log(`[Omega Fallback] ${nextAgent.modelName} replying to peer...`);
          const prompt = formatDialogosPeerMessage(lastMsgContent);
          try {
            await sendMessage(nextAgent.id, prompt, { origin: 'auto', emitBus: true });
            lastFallbackActorId = nextAgent.id;
            lastFallbackMsgTime = now;
            fallbackTurn++;
            autoChatTurn.value = fallbackTurn;
          } catch (err) {
            console.error('[Omega Fallback] Turn failed:', err);
          }
        }
      }

      // Case 3: Omega Intervention (Stall detection > 20s)
      if (timeSinceLast > 20000) {
        console.log('[Omega Fallback] Stall detected. Intervening...');
        const nextAgent = readySessions.find(s => s.id !== lastActorId) || readySessions[0];
        const intervention = getOmegaIntervention(fallbackTurn);
        try {
          await sendMessage(nextAgent.id, intervention, { origin: 'auto', emitBus: true });
          lastFallbackActorId = nextAgent.id;
          lastFallbackMsgTime = now;
          fallbackTurn++;
          autoChatTurn.value = fallbackTurn;
        } catch (err) {
          console.error('[Omega Fallback] Intervention failed:', err);
        }
      }
    }, AUTO_CHAT_POLL_MS);

    cleanup(() => {
      cancelled = true;
      clearInterval(interval);
    });
  }, { strategy: 'document-idle' });

  // Add a new session
  const addSession = $(() => {
    if (state.sessions.length >= 3) return;

    // Pick a model not already in use, preferring cached
    const usedIds = state.sessions.map(s => s.modelId);
    let model = AVAILABLE_MODELS.find(m =>
      state.cachedModels.includes(m.id) && !usedIds.includes(m.id)
    );
    if (!model) {
      model = AVAILABLE_MODELS.find(m => !usedIds.includes(m.id));
    }
    if (!model) {
      model = AVAILABLE_MODELS[0];
    }

    state.sessions = [...state.sessions, {
      id: createSessionId(),
      modelId: model.id,
      modelName: model.name,
      engine: null,
      messages: [],
      status: 'idle',
      loadProgress: 0,
      loadStage: '',
      error: null,
    }];
  });

  // Remove a session
  const removeSession = $((sessionId: string) => {
    const session = state.sessions.find(s => s.id === sessionId);
    if (session?.engine) void unloadEngine(session.engine);
    state.sessions = state.sessions.filter(s => s.id !== sessionId);
  });

  // Load model for a session
  const loadModel = $(async (sessionId: string) => {
    const session = state.sessions.find(s => s.id === sessionId);
    if (!session || session.status === 'loading' || session.status === 'generating') return;

    const modelDef = getModelDefByModelId(session.modelId);

    session.status = 'loading';
    session.loadProgress = 0;
    session.loadStage = state.cachedModels.includes(session.modelId)
      ? 'Loading from cache...'
      : 'Downloading...';
    session.error = null;
    const loadStartedAt = performance.now();
    void emitBus({
      topic: 'webllm.model.load',
      kind: 'metric',
      level: 'info',
      actor: 'webllm-widget',
      data: {
        session_id: session.id,
        model_id: session.modelId,
        model_name: session.modelName,
        cached: state.cachedModels.includes(session.modelId),
      },
    });

    try {
      const webllm = await import('@mlc-ai/web-llm');
      const engine = await webllm.CreateMLCEngine(session.modelId, {
        initProgressCallback: (progress: { text: string; progress: number }) => {
          session.loadProgress = Math.round(progress.progress * 100);
          session.loadStage = progress.text;
        },
      });

      session.engine = noSerialize(engine);
      session.status = 'ready';
      session.loadStage = 'Ready';

      // Inject System Prompt if available
      if (modelDef?.systemPrompt) {
        // Clear previous system messages if any (refresh context)
        session.messages = session.messages.filter(m => m.role !== 'system');
        session.messages.unshift({
          role: 'system',
          content: modelDef.systemPrompt,
          timestamp: Date.now(),
        });
      }

      // Update cache list
      state.cachedModels = uniq([...state.cachedModels, session.modelId]);
      void computeAutoCacheTargets();
      void emitBus({
        topic: 'webllm.model.ready',
        kind: 'artifact',
        level: 'info',
        actor: 'webllm-widget',
        data: {
          session_id: session.id,
          model_id: session.modelId,
          model_name: session.modelName,
          latency_ms: Math.round(performance.now() - loadStartedAt),
        },
      });
    } catch (err) {
      session.status = 'idle';
      session.error = err instanceof Error ? err.message : 'Load failed';
      void emitBus({
        topic: 'webllm.model.error',
        kind: 'artifact',
        level: 'error',
        actor: 'webllm-widget',
        data: {
          session_id: session.id,
          model_id: session.modelId,
          model_name: session.modelName,
          error: session.error,
        },
      });
    }
  });

  // Change model for a session
  const changeModel = $((sessionId: string, modelId: string) => {
    const session = state.sessions.find(s => s.id === sessionId);
    if (!session) return;

    const model = AVAILABLE_MODELS.find(m => m.id === modelId);
    if (!model) return;

    if (session.engine) void unloadEngine(session.engine);
    session.modelId = modelId;
    session.modelName = model.name;
    session.engine = null;
    session.status = 'idle';
    session.messages = [];
    session.error = null;
  });

  // Send message in a session
  const sendMessage = $(async (
    sessionId: string,
    content: string,
    opts?: { origin?: 'user' | 'auto' | 'dialogos'; emitBus?: boolean }
  ) => {
    const session = state.sessions.find(s => s.id === sessionId);
    if (!session || !session.engine || session.status !== 'ready') return;

    // Add user message
    session.messages = [...session.messages, {
      role: 'user',
      content,
      timestamp: Date.now(),
    }];

    session.status = 'generating';
    const startedAt = performance.now();

    try {
      const engine = session.engine as unknown as {
        chat: {
          completions: {
            create: (opts: {
              messages: Array<{ role: string; content: string }>;
              temperature: number;
              max_tokens: number;
            }) => Promise<{ choices: Array<{ message: { content: string } }> }>;
          };
        };
      };

      const response = await engine.chat.completions.create({
        messages: session.messages.map(m => ({ role: m.role, content: m.content })),
        temperature: 0.7,
        max_tokens: 256,
      });

      const assistantContent = response.choices[0]?.message?.content || '';
      session.messages = [...session.messages, {
        role: 'assistant',
        content: assistantContent,
        timestamp: Date.now(),
      }];

      // Prune history to prevent memory degradation on long conversations
      if (session.messages.length > MAX_SESSION_HISTORY) {
        session.messages = session.messages.slice(-MAX_SESSION_HISTORY);
      }

      session.status = 'ready';

      if (opts?.emitBus) {
        void emitBus({
          topic: 'webllm.infer.response',
          kind: 'response',
          level: 'info',
          actor: 'webllm-widget',
          data: {
            req_id: `${opts.origin || 'local'}-${Date.now()}`,
            ok: true,
            text: assistantContent,
            latency_ms: Math.round(performance.now() - startedAt),
            session_id: session.id,
            model_id: session.modelId,
            model_name: session.modelName,
            origin: opts?.origin || 'user',
          },
        });
      }
    } catch (err) {
      session.status = 'ready';
      session.error = err instanceof Error ? err.message : 'Generation failed';
      if (opts?.emitBus) {
        void emitBus({
          topic: 'webllm.infer.response',
          kind: 'response',
          level: 'error',
          actor: 'webllm-widget',
          data: {
            req_id: `${opts.origin || 'local'}-${Date.now()}`,
            ok: false,
            error: session.error,
            session_id: session.id,
            model_id: session.modelId,
            model_name: session.modelName,
            origin: opts?.origin || 'user',
          },
        });
      }
    }
  });

  const getModelColor = (modelId: string) => {
    const model = AVAILABLE_MODELS.find(m => m.id === modelId);
    return model?.color || 'cyan';
  };

  const cachedCount = state.cachedModels.length;
  const sessionCount = state.sessions.length;

  return (
    <div class={fullScreen ? 'w-full h-full flex flex-col min-h-0' : 'w-full'}>
      {/* Header Bar */}
      <div
        class="flex items-center gap-3 p-3 bg-black/30 border border-border/50 rounded-t-lg cursor-pointer hover:bg-black/40 transition-colors flex-shrink-0"
        onClick$={() => expanded.value = !expanded.value}
      >
        <span
          class={`w-3 h-3 rounded-full ${
            !enabled.value
              ? 'bg-gray-400'
              : state.webgpuStatus === 'ready'
                ? 'bg-green-400'
                : state.webgpuStatus === 'checking'
                  ? 'bg-yellow-400 animate-pulse'
                  : 'bg-red-400'
          }`}
        />
        <span class="text-lg">🧠</span>
        <div class="flex-1">
          <div class="font-medium">WebLLM Multi-Mind Arena</div>
          <div class="text-xs text-muted-foreground">
            {!enabled.value
              ? 'Disabled (manual override)'
              : state.webgpuStatus === 'ready'
                ? `${sessionCount} session${sessionCount !== 1 ? 's' : ''} active`
                : state.error || 'Checking WebGPU...'}
          </div>
        </div>
        {cachedCount > 0 && (
          <span class="text-xs px-2 py-0.5 rounded bg-green-500/20 text-green-400">
            {cachedCount} cached
          </span>
        )}
        {state.gpuInfo && (
          <span class="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-400 max-w-[150px] truncate">
            {state.gpuInfo}
          </span>
        )}
        {enabled.value && (
          <button
            class="text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-300 hover:bg-red-500/30"
            onClick$={(event) => {
              event.stopPropagation();
              disableWebLLM();
            }}
          >
            Disable WebLLM
          </button>
        )}
        <span class={`text-muted-foreground transition-transform ${expanded.value ? 'rotate-180' : ''}`}>
          ▼
        </span>
      </div>

      {/* Main Content */}
      {expanded.value && !enabled.value && (
        <div class={`border border-t-0 border-border/50 rounded-b-lg bg-black/20 overflow-hidden flex flex-col ${fullScreen ? 'flex-1 min-h-0' : ''}`}>
          <div class="p-4 space-y-3">
            <div class="text-sm text-muted-foreground">
              WebLLM is always-on by default for edge inference. You can disable it to conserve resources.
            </div>
            <button
              class="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white font-medium w-fit"
              onClick$={enableWebLLM}
            >
              Enable WebLLM
            </button>
            <div class="text-xs text-muted-foreground">
              Requires HTTPS + WebGPU-capable browser. Models download on demand.
            </div>
          </div>
        </div>
      )}

      {expanded.value && enabled.value && state.webgpuStatus === 'ready' && (
        <div class={`border border-t-0 border-border/50 rounded-b-lg bg-black/20 overflow-hidden flex flex-col ${fullScreen ? 'flex-1 min-h-0' : ''}`}>
          {/* Controls Bar */}
          <div class="flex items-center gap-3 p-2 border-b border-border/50 bg-black/30 flex-shrink-0">
            <button
              class="px-3 py-1 rounded text-xs font-medium bg-green-600 hover:bg-green-500 text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1"
              onClick$={addSession}
              disabled={sessionCount >= 3}
            >
              <span>+</span> Add Mind
            </button>

            <label class="flex items-center gap-2 text-xs cursor-pointer">
              <input
                type="checkbox"
                checked={state.conversationMode}
                onChange$={(e) => {
                  const next = (e.target as HTMLInputElement).checked;
                  state.conversationMode = next;
                  if (!next) {
                    dialogosActive.value = false;
                    if (dialogosTimer.value) {
                      clearTimeout(dialogosTimer.value);
                      dialogosTimer.value = null;
                    }
                    dialogosPending.value = null;
                  }
                }}
                class="rounded border-border"
              />
              <span 
                id="webllm-arena-mode-label"
                class={state.conversationMode ? 'text-yellow-400' : 'text-muted-foreground'}
              >
                🔄 Arena Mode
              </span>
            </label>

            {state.conversationMode && (
              <div class="flex items-center gap-2">
                <span 
                  id="webllm-current-mode-badge"
                  class="text-[10px] px-2 py-0.5 rounded bg-purple-500/20 text-purple-300 border border-purple-500/30 uppercase tracking-widest animate-pulse"
                >
                  {state.currentMode}
                </span>
                <span class="text-[10px] px-2 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30 font-mono">
                  TURN: {state.turnCount}
                </span>
              </div>
            )}

            <label class="flex items-center gap-2 text-xs cursor-pointer">
              <input
                type="checkbox"
                checked={state.autoCacheEnabled}
                onChange$={(e) => {
                  const next = (e.target as HTMLInputElement).checked;
                  state.autoCacheEnabled = next;
                  if (!next) {
                    autoCacheStarted.value = false;
                    state.autoCacheStatus = 'idle';
                  } else if (state.webgpuStatus === 'ready') {
                    void computeAutoCacheTargets();
                    void startAutoCache();
                  }
                }}
                class="rounded border-border"
                disabled={state.autoCacheStatus === 'running'}
              />
              <span class={state.autoCacheEnabled ? 'text-green-300' : 'text-muted-foreground'}>
                ⚡ Auto-cache
              </span>
            </label>

            <div class="flex-1" />

            <span class="text-xs text-muted-foreground">
              {sessionCount}/3 sessions
            </span>
            {state.autoCacheTargets.length > 0 && (
              <span class="text-xs text-muted-foreground">
                cache {state.autoCacheCompleted.length}/{state.autoCacheTargets.length}
              </span>
            )}
            {busBound.value && (
              <span class="text-xs px-2 py-0.5 rounded bg-cyan-500/20 text-cyan-300">
                bus-bound
              </span>
            )}
          </div>

          {/* Sessions Grid */}
          {sessionCount === 0 ? (
            <div class="p-8 text-center">
              <div class="text-muted-foreground mb-4">No sessions active</div>
              <button
                class="px-4 py-2 rounded bg-cyan-600 hover:bg-cyan-500 text-white font-medium"
                onClick$={addSession}
              >
                + Create First Session
              </button>
            </div>
          ) : (
            <div
              id="webllm-sessions-grid"
              class={`grid gap-2 p-2 ${fullScreen ? 'flex-1 min-h-0' : ''}`}
              style={{
                gridTemplateColumns: `repeat(${Math.min(sessionCount, 3)}, 1fr)`,
                ...(fullScreen
                  ? { height: '100%', gridAutoRows: '1fr' }
                  : { minHeight: '35vh', maxHeight: '45vh' }),
              }}
            >
              {state.sessions.map((session) => {
                const color = getModelColor(session.modelId);
                const isCached = state.cachedModels.includes(session.modelId);

                return (
                  <div
                    key={session.id}
                    class={`flex flex-col border rounded-lg overflow-hidden ${MODEL_COLORS[color]}`}
                  >
                    {/* Session Header */}
                    <div class="flex items-center gap-2 p-2 bg-black/30 border-b border-border/30">
                      <span class={`w-2 h-2 rounded-full ${
                        session.status === 'ready' ? 'bg-green-400' :
                        session.status === 'generating' ? 'bg-yellow-400 animate-pulse' :
                        session.status === 'loading' ? 'bg-blue-400 animate-pulse' :
                        'bg-gray-400'
                      }`} />

                      <select
                        class="flex-1 bg-black/50 border-0 rounded px-2 py-1 text-xs font-medium"
                        data-testid="model-select"
                        value={session.modelId}
                        onChange$={(e) => changeModel(session.id, (e.target as HTMLSelectElement).value)}
                        disabled={session.status === 'loading' || session.status === 'generating'}
                      >
                        {AVAILABLE_MODELS.map(m => (
                          <option key={m.id} value={m.id}>
                            {`${state.cachedModels.includes(m.id) ? '⚡' : '↓'} ${m.name}`}
                          </option>
                        ))}
                      </select>

                      {!session.engine ? (
                        <button
                          class={`px-2 py-1 rounded text-xs font-medium ${
                            isCached
                              ? 'bg-green-600 hover:bg-green-500'
                              : 'bg-cyan-600 hover:bg-cyan-500'
                          } text-white disabled:opacity-50`}
                          onClick$={() => loadModel(session.id)}
                          disabled={session.status === 'loading'}
                        >
                          {session.status === 'loading'
                            ? `${session.loadProgress}%`
                            : isCached ? '⚡' : '↓'}
                        </button>
                      ) : (
                        <span class="text-xs text-green-400">✓</span>
                      )}

                      <button
                        class="px-2 py-1 rounded text-xs bg-red-500/20 hover:bg-red-500/40 text-red-400"
                        onClick$={() => removeSession(session.id)}
                        disabled={session.status === 'loading' || session.status === 'generating'}
                      >
                        ×
                      </button>
                    </div>

                    {/* Loading Progress */}
                    {session.status === 'loading' && (
                      <div class="px-2 py-1 bg-black/20">
                        <div class="h-1 bg-black/30 rounded overflow-hidden">
                          <div
                            class="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all"
                            style={{ width: `${session.loadProgress}%` }}
                          />
                        </div>
                        <div class="text-[10px] text-muted-foreground mt-1 truncate">
                          {session.loadStage}
                        </div>
                      </div>
                    )}

                    {/* Messages */}
                    <div class="flex-1 overflow-y-auto p-2 space-y-2 min-h-0">
                      {session.messages.length === 0 && session.status === 'ready' && (
                        <div class="text-xs text-muted-foreground text-center py-4">
                          {session.modelName} ready. Send a message!
                        </div>
                      )}
                      {session.messages.map((msg, i) => (
                        <div
                          key={i}
                          class={`text-xs p-2 rounded ${
                            msg.role === 'user'
                              ? 'bg-white/10 ml-4'
                              : `${MODEL_COLORS[color]} mr-4`
                          }`}
                        >
                          <div class={`text-[10px] mb-1 ${
                            msg.role === 'user' ? 'text-white/50' : MODEL_TEXT_COLORS[color]
                          }`}>
                            {msg.role === 'user' ? 'You' : session.modelName}
                          </div>
                          <div class="whitespace-pre-wrap break-words">
                            {msg.content}
                          </div>
                        </div>
                      ))}
                      {session.status === 'generating' && (
                        <div class={`text-xs p-2 rounded ${MODEL_COLORS[color]} mr-4`}>
                          <div class={`text-[10px] mb-1 ${MODEL_TEXT_COLORS[color]}`}>
                            {session.modelName}
                          </div>
                          <div class="animate-pulse">Thinking...</div>
                        </div>
                      )}
                    </div>

                    {/* Input */}
                    {session.engine && (
                      <div class="p-2 border-t border-border/30 bg-black/30">
                        <ChatInput
                          disabled={session.status !== 'ready'}
                          onSend$={(msg: string) => sendMessage(session.id, msg, { origin: 'user', emitBus: false })}
                          placeholder={`Message ${session.modelName}...`}
                        />
                      </div>
                    )}

                    {/* Error */}
                    {session.error && (
                      <div class="p-2 bg-red-500/20 text-xs text-red-400 border-t border-red-500/30">
                        {session.error}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}

          {/* Footer */}
          <div class="p-2 border-t border-border/50 bg-black/30 text-xs text-muted-foreground flex items-center justify-between flex-shrink-0">
            <div class="flex items-center gap-2">
              <span>🔒</span>
              <span>100% local inference</span>
            </div>
            {state.autoCacheStatus === 'running' && (
              <span class="text-xs text-green-300 truncate max-w-[45%]">
                ⚡ {state.autoCacheStage || 'Auto-cache running'}
              </span>
            )}
            {state.conversationMode && (
              <span 
                id="omega-protocol-footer"
                class="text-yellow-400 font-medium animate-pulse"
              >
                ⚡ Autonomous Omega Protocol Active: Dual-Mind Reciprocity Loop
              </span>
            )}
          </div>
        </div>
      )}

      {/* Checking State */}
      {expanded.value && enabled.value && state.webgpuStatus === 'checking' && (
        <div class={`border border-t-0 border-border/50 rounded-b-lg bg-black/20 overflow-hidden flex flex-col ${fullScreen ? 'flex-1 min-h-0' : ''}`}>
          <div class="p-4 text-sm text-muted-foreground">Checking WebGPU…</div>
        </div>
      )}

      {/* Unsupported State */}
      {expanded.value && enabled.value && state.webgpuStatus === 'unsupported' && (
        <div class="border border-t-0 border-red-500/30 rounded-b-lg bg-red-500/5 p-4">
          <div class="text-sm text-red-400 font-medium mb-2">{state.error}</div>
          <div class="text-xs text-muted-foreground">
            Requirements: Chrome 113+, HTTPS, GPU acceleration enabled
          </div>
        </div>
      )}
    </div>
  );
});

// Separate input component to avoid re-render issues
const ChatInput = component$<{
  disabled: boolean;
  onSend$: (msg: string) => void;
  placeholder: string;
}>(({ disabled, onSend$, placeholder }) => {
  const input = useSignal('');

  const handleSend = $(() => {
    if (input.value.trim() && !disabled) {
      onSend$(input.value.trim());
      input.value = '';
    }
  });

  return (
    <div class="flex gap-2">
      <input
        type="text"
        class="flex-1 bg-black/50 border border-border/50 rounded px-2 py-1.5 text-xs"
        placeholder={placeholder}
        value={input.value}
        onInput$={(e) => { input.value = (e.target as HTMLInputElement).value; }}
        onKeyPress$={(e) => { if (e.key === 'Enter') handleSend(); }}
        disabled={disabled}
      />
      <button
        class="px-3 py-1.5 rounded bg-green-600 hover:bg-green-500 text-white text-xs font-medium disabled:opacity-50"
        onClick$={handleSend}
        disabled={disabled || !input.value.trim()}
      >
        ▶
      </button>
    </div>
  );
});
