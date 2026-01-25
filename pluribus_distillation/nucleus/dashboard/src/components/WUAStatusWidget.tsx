/**
 * WUAStatusWidget: Real-time WUA Session Status Dashboard
 * =========================================================
 * Displays live status of Web User Agent sessions with:
 * - Real-time WebSocket subscription to wua.* events
 * - Visual confidence meter (0-100%)
 * - Status badges (active/auth_required/offline/error)
 * - Provider icons (ChatGPT, Claude, Gemini)
 * - Heartbeat indicator showing daemon liveness
 * - Message count per session
 */

import { component$, useSignal, useVisibleTask$, useStore, $ } from '@builder.io/qwik';

type WUAProvider = 'chatgpt' | 'claude' | 'gemini';
type WUAStatus = 'active' | 'auth_required' | 'offline' | 'error';

interface WUASession {
  provider: WUAProvider;
  status: WUAStatus;
  confidence: number;  // 0-1
  lastSeen: number;    // timestamp
  messagesProcessed: number;
}

interface WUAState {
  sessions: WUASession[];
  lastHeartbeat: number;
  connected: boolean;
  error: string | null;
}

// Provider Icons as inline SVG components
const ProviderIcon = component$<{ provider: WUAProvider }>(({ provider }) => {
  // ChatGPT icon (OpenAI style)
  if (provider === 'chatgpt') {
    return (
      <div class="w-6 h-6 rounded bg-[#10a37f]/20 flex items-center justify-center">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" class="text-[#10a37f]">
          <path
            d="M22.2 9.03c.5-1.5.24-3.15-.7-4.5a5.4 5.4 0 0 0-5.74-2.3 5.4 5.4 0 0 0-4.22-1.98 5.4 5.4 0 0 0-5.14 3.68 5.4 5.4 0 0 0-3.62 2.62 5.4 5.4 0 0 0 .67 6.32 5.4 5.4 0 0 0 .7 4.5 5.4 5.4 0 0 0 5.74 2.3 5.4 5.4 0 0 0 4.22 1.98 5.4 5.4 0 0 0 5.14-3.68 5.4 5.4 0 0 0 3.62-2.62 5.4 5.4 0 0 0-.67-6.32z"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <path
            d="M9 14.5l3-6 3 6M9.5 13h5"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
      </div>
    );
  }

  // Claude icon (Anthropic style - stylized "A")
  if (provider === 'claude') {
    return (
      <div class="w-6 h-6 rounded bg-[#cc785c]/20 flex items-center justify-center">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" class="text-[#cc785c]">
          <path
            d="M12 4l8 16H4l8-16z"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
          <path
            d="M8 16h8M10 12h4"
            stroke="currentColor"
            stroke-width="1.5"
            stroke-linecap="round"
          />
        </svg>
      </div>
    );
  }

  // Gemini icon (Google style - stars)
  return (
    <div class="w-6 h-6 rounded bg-[#4285f4]/20 flex items-center justify-center">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" class="text-[#4285f4]">
        <path
          d="M12 2l2 6h6l-5 4 2 6-5-4-5 4 2-6-5-4h6l2-6z"
          stroke="currentColor"
          stroke-width="1.5"
          stroke-linecap="round"
          stroke-linejoin="round"
        />
      </svg>
    </div>
  );
});

// Confidence Meter visualization
const ConfidenceMeter = component$<{ value: number }>(({ value }) => {
  const percent = Math.round(value * 100);
  const bars = 5;
  const filledBars = Math.round((value * bars));

  // Color gradient based on confidence
  const colorClass = value >= 0.8 ? 'bg-green-500' :
                     value >= 0.5 ? 'bg-yellow-500' :
                     value >= 0.3 ? 'bg-orange-500' : 'bg-red-500';

  return (
    <div class="flex items-center gap-1" title={`Confidence: ${percent}%`}>
      <div class="flex gap-0.5">
        {Array.from({ length: bars }, (_, i) => (
          <div
            key={i}
            class={`w-1 h-3 rounded-sm ${i < filledBars ? colorClass : 'bg-gray-700'}`}
          />
        ))}
      </div>
      <span class="text-[9px] text-cyan-600 w-7 text-right">{percent}%</span>
    </div>
  );
});

// Status Badge with color coding
const StatusBadge = component$<{ status: WUAStatus }>(({ status }) => {
  const config: Record<WUAStatus, { bg: string; text: string; label: string }> = {
    active: { bg: 'bg-green-500/20', text: 'text-green-400', label: 'ACTIVE' },
    auth_required: { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: 'AUTH' },
    offline: { bg: 'bg-gray-500/20', text: 'text-gray-400', label: 'OFFLINE' },
    error: { bg: 'bg-red-500/20', text: 'text-red-400', label: 'ERROR' },
  };

  const c = config[status] || config.offline;

  return (
    <span class={`text-[8px] px-1.5 py-0.5 rounded ${c.bg} ${c.text} font-bold tracking-wider`}>
      {c.label}
    </span>
  );
});

export const WUAStatusWidget = component$(() => {
  const state = useStore<WUAState>({
    sessions: [],
    lastHeartbeat: 0,
    connected: false,
    error: null,
  });

  const wsRef = useSignal<WebSocket | null>(null);

  // Helper to update or add a session
  const updateSession = $((update: Partial<WUASession> & { provider: WUAProvider }) => {
    const idx = state.sessions.findIndex(s => s.provider === update.provider);
    if (idx >= 0) {
      // Merge updates
      const existing = state.sessions[idx];
      state.sessions[idx] = { ...existing, ...update };
    } else {
      // Create new session with defaults
      const newSession: WUASession = {
        provider: update.provider,
        status: update.status || 'offline',
        confidence: update.confidence ?? 0,
        lastSeen: update.lastSeen ?? Date.now(),
        messagesProcessed: update.messagesProcessed ?? 0,
      };
      state.sessions = [...state.sessions, newSession];
    }
  });

  // Connect to WebSocket bus and subscribe to wua.* events
  useVisibleTask$(({ cleanup }) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/bus`;

    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let pingInterval: ReturnType<typeof setInterval> | null = null;

    const connect = () => {
      try {
        const ws = new WebSocket(wsUrl);
        wsRef.value = ws;

        ws.onopen = () => {
          state.connected = true;
          state.error = null;
          state.lastHeartbeat = Date.now();

          // Subscribe to wua topics
          ws.send(JSON.stringify({
            action: 'subscribe',
            topics: ['wua.*', 'wua.session.*', 'wua.heartbeat'],
          }));
        };

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            const topic = data.topic as string;

            if (!topic?.startsWith('wua.')) return;

            // Update heartbeat on any wua message
            state.lastHeartbeat = Date.now();

            // Handle specific topics
            if (topic === 'wua.heartbeat') {
              // Just updates lastHeartbeat, already done above
              return;
            }

            // Session status updates: wua.session.<provider>
            if (topic.startsWith('wua.session.')) {
              const provider = topic.split('.')[2] as WUAProvider;
              if (['chatgpt', 'claude', 'gemini'].includes(provider)) {
                const payload = data.data || data.payload || {};
                updateSession({
                  provider,
                  status: payload.status as WUAStatus,
                  confidence: typeof payload.confidence === 'number' ? payload.confidence : undefined,
                  lastSeen: Date.now(),
                  messagesProcessed: typeof payload.messagesProcessed === 'number' ? payload.messagesProcessed : undefined,
                });
              }
              return;
            }

            // Message processed events: wua.message.<provider>
            if (topic.startsWith('wua.message.')) {
              const provider = topic.split('.')[2] as WUAProvider;
              if (['chatgpt', 'claude', 'gemini'].includes(provider)) {
                const session = state.sessions.find(s => s.provider === provider);
                if (session) {
                  updateSession({
                    provider,
                    messagesProcessed: session.messagesProcessed + 1,
                    lastSeen: Date.now(),
                  });
                }
              }
              return;
            }

            // Auth required events: wua.auth.<provider>
            if (topic.startsWith('wua.auth.')) {
              const provider = topic.split('.')[2] as WUAProvider;
              if (['chatgpt', 'claude', 'gemini'].includes(provider)) {
                updateSession({
                  provider,
                  status: 'auth_required',
                  lastSeen: Date.now(),
                });
              }
              return;
            }

            // Error events: wua.error.<provider>
            if (topic.startsWith('wua.error.')) {
              const provider = topic.split('.')[2] as WUAProvider;
              if (['chatgpt', 'claude', 'gemini'].includes(provider)) {
                updateSession({
                  provider,
                  status: 'error',
                  confidence: 0,
                  lastSeen: Date.now(),
                });
              }
              return;
            }
          } catch {
            // Ignore parse errors
          }
        };

        ws.onerror = () => {
          state.error = 'WebSocket connection error';
          state.connected = false;
        };

        ws.onclose = () => {
          state.connected = false;
          wsRef.value = null;

          // Reconnect after delay
          reconnectTimer = setTimeout(connect, 5000);
        };

        // Periodic ping to detect stale connections
        pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ action: 'ping' }));
          }
        }, 30000);
      } catch (err) {
        state.error = err instanceof Error ? err.message : 'Connection failed';
        state.connected = false;
        reconnectTimer = setTimeout(connect, 5000);
      }
    };

    connect();

    cleanup(() => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (pingInterval) clearInterval(pingInterval);
      if (wsRef.value) {
        wsRef.value.close();
        wsRef.value = null;
      }
    });
  });

  // Check if heartbeat is recent (within last 60 seconds)
  const isLive = state.lastHeartbeat > 0 && (Date.now() - state.lastHeartbeat) < 60000;

  // Sort sessions by provider name for consistent display
  const sortedSessions = [...state.sessions].sort((a, b) =>
    a.provider.localeCompare(b.provider)
  );

  return (
    <div class="wua-status-widget bg-black/40 border border-cyan-900/30 rounded-xl p-4">
      {/* Header */}
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-cyan-400 text-sm font-bold uppercase tracking-wider">
          WUA Sessions
        </h3>
        <div class="flex items-center gap-2">
          {/* Connection status indicator */}
          <div
            class={`w-2 h-2 rounded-full transition-colors ${
              state.connected
                ? isLive
                  ? 'bg-green-500 animate-pulse'
                  : 'bg-yellow-500'
                : 'bg-red-500'
            }`}
            title={state.connected ? (isLive ? 'Connected & Live' : 'Connected, waiting for heartbeat') : 'Disconnected'}
          />
          <span class="text-[10px] text-cyan-700 font-mono">
            {state.connected ? (isLive ? 'LIVE' : 'WAIT') : 'OFFLINE'}
          </span>
        </div>
      </div>

      {/* Error display */}
      {state.error && (
        <div class="text-xs text-red-400 bg-red-500/10 border border-red-500/20 rounded p-2 mb-3">
          {state.error}
        </div>
      )}

      {/* Session list */}
      <div class="space-y-3">
        {sortedSessions.map(session => (
          <div
            key={session.provider}
            class="flex items-center justify-between p-2 bg-black/30 rounded border border-cyan-900/20 hover:border-cyan-700/30 transition-colors"
          >
            <div class="flex items-center gap-3">
              <ProviderIcon provider={session.provider} />
              <div>
                <div class="text-cyan-300 text-xs font-medium capitalize">
                  {session.provider}
                </div>
                <div class="text-[10px] text-cyan-700 font-mono">
                  {session.messagesProcessed.toLocaleString()} msgs
                </div>
              </div>
            </div>

            <div class="flex items-center gap-2">
              <ConfidenceMeter value={session.confidence} />
              <StatusBadge status={session.status} />
            </div>
          </div>
        ))}

        {/* Empty state */}
        {sortedSessions.length === 0 && (
          <div class="text-center text-cyan-800 text-xs py-6">
            <div class="mb-2">No active WUA sessions</div>
            <div class="text-[10px] text-cyan-900">
              Waiting for wua.* events...
            </div>
          </div>
        )}
      </div>

      {/* Footer stats */}
      {sortedSessions.length > 0 && (
        <div class="mt-4 pt-3 border-t border-cyan-900/20">
          <div class="flex items-center justify-between text-[10px] text-cyan-700">
            <span>
              Total sessions: <span class="text-cyan-500">{sortedSessions.length}</span>
            </span>
            <span>
              Active: <span class="text-green-500">
                {sortedSessions.filter(s => s.status === 'active').length}
              </span>
            </span>
            <span>
              Messages: <span class="text-cyan-500">
                {sortedSessions.reduce((sum, s) => sum + s.messagesProcessed, 0).toLocaleString()}
              </span>
            </span>
          </div>
        </div>
      )}

      {/* Last heartbeat timestamp */}
      {state.lastHeartbeat > 0 && (
        <div class="mt-2 text-[9px] text-cyan-900 text-right font-mono">
          last: {new Date(state.lastHeartbeat).toISOString().slice(11, 19)}Z
        </div>
      )}
    </div>
  );
});

export default WUAStatusWidget;
