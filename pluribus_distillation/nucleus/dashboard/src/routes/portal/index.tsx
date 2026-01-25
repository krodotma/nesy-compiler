import { component$, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import type { BusEvent } from '../../lib/state/types';
import { PortalIngressPanel, PortalInceptionPanel } from '../../components/portal';
import { PortalMetricsPanel } from '../../components/PortalMetricsPanel';
import { LoadingStage } from '../../components/LoadingStage';

const EVENTS_MAX = 200;
const RECONNECT_DELAY_MS = 5000;

export default component$(() => {
  const events = useSignal<BusEvent[]>([]);
  const connected = useSignal(false);

  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const wsHost = window.location.host;
    const isSecure = window.location.protocol === 'https:';
    const wsUrl = `${isSecure ? 'wss' : 'ws'}://${wsHost}/ws/bus`;
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const connect = () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }

      socket = new WebSocket(wsUrl);
      socket.onopen = () => {
        connected.value = true;
        socket?.send(JSON.stringify({ type: 'sync' }));
      };
      socket.onclose = () => {
        connected.value = false;
        socket = null;
        reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
      };
      socket.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if (data.type === 'sync' && Array.isArray(data.events)) {
            events.value = data.events.slice(-EVENTS_MAX);
          } else if (data.type === 'event' && data.event) {
            events.value = [...events.value, data.event].slice(-EVENTS_MAX);
          }
        } catch {
          // ignore malformed payloads
        }
      };
    };

    connect();

    cleanup(() => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
      if (socket) {
        socket.close();
        socket = null;
      }
    });
  });

  return (
    <div class="min-h-screen p-6 space-y-6">
      <header class="flex items-center justify-between">
        <div>
          <h1 class="text-xl font-semibold tracking-wide text-foreground">Portal</h1>
          <p class="text-xs text-muted-foreground">Ingress routing and destination selection.</p>
        </div>
        <span
          class={`text-[10px] px-2 py-1 rounded border ${
            connected.value
              ? 'border-emerald-500/40 text-emerald-300 bg-emerald-500/10'
              : 'border-border/60 text-muted-foreground bg-muted/20'
          }`}
        >
          {connected.value ? 'bus live' : 'bus offline'}
        </span>
      </header>

      <div class="grid gap-6 lg:grid-cols-3">
        <div class="lg:col-span-2">
          <LoadingStage id="comp:portal-ingress-route">
            <div class="glass-surface glass-surface-1 p-4 glass-hover-lift">
              <PortalIngressPanel events={events.value} />
            </div>
          </LoadingStage>
        </div>

        <div class="space-y-6">
          <LoadingStage id="comp:portal-inception">
            <PortalInceptionPanel events={events.value} />
          </LoadingStage>
          <LoadingStage id="comp:portal-metrics">
            <PortalMetricsPanel events={events.value} />
          </LoadingStage>
        </div>
      </div>
    </div>
  );
});
