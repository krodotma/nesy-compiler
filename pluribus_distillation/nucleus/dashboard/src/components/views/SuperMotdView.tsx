/**
 * SuperMOTD View - Dedicated view for system status visualization
 * Implements Karpathy-style OS boot sequence visualization for Pluribus
 */

import { component$, useSignal, useVisibleTask$, useComputed$, useTask$ } from '@builder.io/qwik';
import { type DocumentHead, useLocation } from '@builder.io/qwik-city';
import type { AgentStatus, STRpRequest, VPSSession, BusEvent } from '../../lib/state/types';
import { SuperMotd } from '../supermotd';
import { BusObservatoryView } from '../BusObservatoryView';
import { EventSearchBox } from '../EventVisualization';

interface SuperMotdViewProps {
  events: BusEvent[];
  session: VPSSession;
  connected: boolean;
  emitBus?: (topic: string, kind: string, data: Record<string, unknown>) => Promise<void>;
}

export const SuperMotdView = component$<SuperMotdViewProps>(({ events, session, connected, emitBus }) => {
  const location = useLocation();
  const eventFilter = useSignal<string | null>(null);
  const eventSearchPattern = useSignal<string>('');
  const eventSearchMode = useSignal<'glob' | 'regex' | 'ltl' | 'actor' | 'topic'>('glob');
  const showEventDetails = useSignal(false);
  const selectedEvent = useSignal<BusEvent | null>(null);
  const eventsSignal = useSignal<BusEvent[]>(events);
  const sessionSignal = useSignal<VPSSession>(session);
  const agentsSignal = useSignal<AgentStatus[]>([]);
  const requestsSignal = useSignal<STRpRequest[]>([]);

  useTask$(({ track }) => {
    track(() => events);
    eventsSignal.value = events;
  });

  useTask$(({ track }) => {
    track(() => session);
    sessionSignal.value = session;
  });

  // Auto-refresh mechanism
  const refreshInterval = useSignal<number>(0);
  const isAutoRefreshing = useComputed$(() => refreshInterval.value > 0);

  useVisibleTask$(({ track }) => {
    track(() => isAutoRefreshing.value);
    if (!isAutoRefreshing.value) return;

    const interval = setInterval(() => {
      // Trigger a refresh by updating a signal
      eventFilter.value = eventFilter.value;
    }, refreshInterval.value * 1000);

    return () => clearInterval(interval);
  });

  // Filtered events for the observatory
  const filteredEvents = useComputed$(() => {
    let filtered = eventsSignal.value;

    // Apply search pattern
    if (eventSearchPattern.value.trim()) {
      const pattern = eventSearchPattern.value.trim();
      const mode = eventSearchMode.value;

      filtered = filtered.filter(e => {
        switch (mode) {
          case 'glob': {
            // Convert glob to regex: * → .*, ? → .
            const regexPattern = pattern
              .replace(/[.+^${}()|[\]\\]/g, '\\$&')
              .replace(/\*/g, '.*')
              .replace(/\?/g, '.');
            try {
              return new RegExp(regexPattern, 'i').test(e.topic);
            } catch {
              return e.topic.includes(pattern);
            }
          }
          case 'regex': {
            // Strip /.../ wrapper if present
            const match = pattern.match(/^\/(.+)\/([gimsuy]*)$/);
            try {
              const regex = match
                ? new RegExp(match[1], match[2] || 'i')
                : new RegExp(pattern, 'i');
              return regex.test(e.topic) || regex.test(e.actor || '') || regex.test(JSON.stringify(e.data || {}));
            } catch {
              return e.topic.includes(pattern);
            }
          }
          case 'ltl': {
            // LTL mode: look for temporal patterns (simplified)
            // ◇ = eventually (exists), □ = always, ○ = next
            if (pattern.startsWith('◇') || pattern.startsWith('eventually')) {
              const target = pattern.replace(/^(◇|eventually)\s*/, '');
              return e.topic.includes(target);
            }
            if (pattern.startsWith('□') || pattern.startsWith('always')) {
              const target = pattern.replace(/^(□|always)\s*/, '');
              return e.topic.includes(target);
            }
            return e.topic.includes(pattern);
          }
          case 'actor':
            return (e.actor || '').toLowerCase().includes(pattern.toLowerCase());
          case 'topic':
            return e.topic.toLowerCase().startsWith(pattern.toLowerCase());
          default:
            return e.topic.includes(pattern);
        }
      });
    }

    return filtered;
  });

  return (
    <div class="h-full flex flex-col">
      {/* Header with controls */}
      <div class="border-b border-border p-4 bg-card">
        <div class="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
          <div>
            <h1 class="text-xl font-bold text-foreground">SuperMOTD</h1>
            <p class="text-sm text-muted-foreground mt-1">
              Karpathy-style OS boot sequence visualization for Pluribus
            </p>
          </div>
          
          <div class="flex flex-wrap gap-2 w-full lg:w-auto">
            <EventSearchBox
              onSearch$={((pattern: string, mode: 'glob' | 'regex' | 'ltl' | 'actor' | 'topic') => {
                eventSearchPattern.value = pattern;
                eventSearchMode.value = mode;
              })}
              placeholder="Filter events (e.g., 'system.boot.log', 'slou', 'oiterate')"
            />
            
            <div class="flex gap-2">
              <select
                value={refreshInterval.value}
                onChange$={(e) => {
                  const value = (e.target as HTMLSelectElement | null)?.value ?? '0';
                  refreshInterval.value = parseInt(value, 10) || 0;
                }}
                class="text-xs rounded border border-border bg-background px-2 py-1.5"
              >
                <option value={0}>Auto-refresh: Off</option>
                <option value={5}>5s</option>
                <option value={10}>10s</option>
                <option value={30}>30s</option>
                <option value={60}>60s</option>
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Main content area */}
      <div class="flex-1 overflow-hidden grid grid-cols-1 lg:grid-cols-2 gap-4 p-4">
        {/* SuperMOTD Panel */}
        <div class="flex flex-col">
          <div class="mb-2">
            <h2 class="text-lg font-semibold text-foreground">System Status (SLOU)</h2>
            <p class="text-sm text-muted-foreground">Real-time system load ontological unit visualization</p>
          </div>
          <div class="flex-1 overflow-auto">
            <SuperMotd
              connected={connected}
              events={eventsSignal.value}
              session={sessionSignal.value}
              emitBus$={emitBus}
            />

            {/* System Health Summary */}
            <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div class="bg-muted/30 p-2 rounded border border-border">
                <div class="text-muted-foreground">Providers</div>
                <div class="font-mono">
                  {Object.values(sessionSignal.value.providers).filter(p => p.available).length}/
                  {Object.keys(sessionSignal.value.providers).length}
                </div>
              </div>
              <div class="bg-muted/30 p-2 rounded border border-border">
                <div class="text-muted-foreground">Flow Mode</div>
                <div class="font-mono">{sessionSignal.value.flowMode === 'm' ? 'Monitor' : 'Auto'}</div>
              </div>
              <div class="bg-muted/30 p-2 rounded border border-border">
                <div class="text-muted-foreground">Bus Events</div>
                <div class="font-mono">{eventsSignal.value.length}</div>
              </div>
              <div class="bg-muted/30 p-2 rounded border border-border">
                <div class="text-muted-foreground">Status</div>
                <div class={`font-mono ${connected ? 'text-green-500' : 'text-red-500'}`}>
                  {connected ? 'Connected' : 'Disconnected'}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Event Observatory */}
        <div class="flex flex-col">
          <div class="mb-2">
            <h2 class="text-lg font-semibold text-foreground">Bus Observatory</h2>
            <p class="text-sm text-muted-foreground">Filtered event stream for deep analysis</p>
          </div>
          <div class="flex-1 overflow-auto">
            <BusObservatoryView
              events={filteredEvents}
              session={sessionSignal}
              agents={agentsSignal}
              requests={requestsSignal}
              connected={connected}
            />
          </div>
        </div>
      </div>

      {/* System Status Summary */}
      <div class="border-t border-border bg-card p-4">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div class="flex flex-col">
            <span class="text-muted-foreground">Total Events</span>
            <span class="font-mono">{eventsSignal.value.length}</span>
          </div>
          <div class="flex flex-col">
            <span class="text-muted-foreground">Bus Status</span>
            <span class={`font-mono ${connected ? 'text-green-500' : 'text-red-500'}`}>
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-muted-foreground">Active Providers</span>
            <span class="font-mono">
              {Object.values(session.providers).filter(p => p.available).length}/
              {Object.keys(session.providers).length}
            </span>
          </div>
          <div class="flex flex-col">
            <span class="text-muted-foreground">Flow Mode</span>
            <span class="font-mono">{session.flowMode === 'm' ? 'Monitor' : 'Auto'}</span>
          </div>
        </div>
      </div>
    </div>
  );
});

export const head: DocumentHead = {
  title: "SuperMOTD - Pluribus System Status",
  meta: [
    {
      name: "description",
      content: "Karpathy-style OS boot sequence visualization for Pluribus - Real-time system status dashboard"
    }
  ]
};
