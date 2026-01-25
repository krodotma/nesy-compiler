/**
 * NotificationSidepanel.tsx
 * =========================
 * A vertical sliding panel for real-time event notifications.
 */

import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  noSerialize,
  type NoSerialize,
  type QRL,
} from '@builder.io/qwik';
import { useTracking } from '../lib/telemetry/use-tracking';
import type { BusEvent } from '../lib/state/types';
import { createBusClient, type BusClient } from '../lib/bus/bus-client';
import { NotificationCard } from './NotificationCard';
import { NotificationDetailOverlay } from './NotificationDetailOverlay';
import { Button } from './ui/Button';
import { NeonTitle, NeonBadge } from './ui/NeonTitle';

// M3 Components - NotificationSidepanel
import '@material/web/elevation/elevation.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';

export interface NotificationSidepanelProps {
  /** Maximum number of notifications to display */
  maxNotifications?: number;
  /** Topic patterns to subscribe to (default: ['*']) */
  topics?: string[];
  /** Callback when navigating to an actor */
  onNavigateToActor$?: QRL<(actor: string) => void>;
  /** Callback when navigating to a topic */
  onNavigateToTopic$?: QRL<(topic: string) => void>;
}

/** Panel width constants */
const PANEL_WIDTH_CLOSED = 12; // px - just the handle
const PANEL_WIDTH_OPEN = 320; // px - full panel width
const PANEL_WIDTH_PEEK = 48; // px - when hovering near edge

/** Calculate fisheye scale based on item position relative to viewport center */
function calculateFisheyeScale(
  itemTop: number,
  itemHeight: number,
  viewportHeight: number,
  scrollTop: number
): number {
  const viewportCenter = scrollTop + viewportHeight / 2;
  const itemCenter = itemTop + itemHeight / 2;
  const distanceFromCenter = Math.abs(viewportCenter - itemCenter);
  const maxDistance = viewportHeight / 2;
  const normalizedDistance = Math.min(distanceFromCenter / maxDistance, 1);
  return 1 - normalizedDistance * 0.3;
}

export const NotificationSidepanel = component$<NotificationSidepanelProps>(({
  maxNotifications = 100,
  topics = ['*'],
  onNavigateToActor$,
  onNavigateToTopic$,
}) => {
  useTracking('comp:notification-sidepanel');

  const isOpen = useSignal(false);
  const isPeeking = useSignal(false);
  const scrollTop = useSignal(0);
  const viewportHeight = useSignal(0);
  const panelRef = useSignal<HTMLDivElement>();
  const listRef = useSignal<HTMLDivElement>();

  const state = useStore<{
    events: BusEvent[];
    busClient: NoSerialize<BusClient> | null;
    connected: boolean;
    selectedEvent: BusEvent | null;
    detailOpen: boolean;
    unreadCount: number;
  }>({
    events: [],
    busClient: null,
    connected: false,
    selectedEvent: null,
    detailOpen: false,
    unreadCount: 0,
  });

  const openPanel = $(() => {
    isOpen.value = true;
    isPeeking.value = false;
    state.unreadCount = 0;
  });

  const closePanel = $(() => {
    isOpen.value = false;
    isPeeking.value = false;
  });

  const handleMouseEnter = $(() => {
    if (!isOpen.value) isPeeking.value = true;
  });

  const handleMouseLeave = $(() => {
    if (!state.detailOpen) {
      isPeeking.value = false;
      isOpen.value = false;
    }
  });

  const handleScroll = $(() => {
    if (listRef.value) scrollTop.value = listRef.value.scrollTop;
  });

  const openDetail = $((event: BusEvent) => {
    state.selectedEvent = event;
    state.detailOpen = true;
  });

  const closeDetail = $(() => {
    state.detailOpen = false;
    state.selectedEvent = null;
  });

  const handleNavigateToActor = $((actor: string) => {
    onNavigateToActor$?.(actor);
    handleCloseAll();
  });

  const handleNavigateToTopic = $((topic: string) => {
    onNavigateToTopic$?.(topic);
    handleCloseAll();
  });

  const handleCloseAll = $(() => {
    state.detailOpen = false;
    state.selectedEvent = null;
    isOpen.value = false;
  });

  const clearAll = $(() => {
    state.events = [];
    state.unreadCount = 0;
  });

  useVisibleTask$(({ cleanup }) => {
    viewportHeight.value = window.innerHeight;
    const handleResize = () => { viewportHeight.value = window.innerHeight; };
    window.addEventListener('resize', handleResize);

    const client = createBusClient({ platform: 'browser' });
    state.busClient = noSerialize(client);

    client.connect().then(() => {
      state.connected = true;
      const unsubscribes: (() => void)[] = [];
      for (const topic of topics) {
        const unsub = client.subscribe(topic, (event: BusEvent) => {
          state.events = [event, ...state.events].slice(0, maxNotifications);
          if (!isOpen.value) state.unreadCount = Math.min(state.unreadCount + 1, 99);
        });
        unsubscribes.push(unsub);
      }
      client.getEvents(maxNotifications).then((events) => {
        if (events.length > 0) {
          const merged = [...state.events];
          for (const evt of events) {
            if (!merged.some(e => e.id === evt.id && e.iso === evt.iso)) merged.push(evt);
          }
          merged.sort((a, b) => {
            const aTs = Date.parse(a.iso || '') || a.ts || 0;
            const bTs = Date.parse(b.iso || '') || b.ts || 0;
            return bTs - aTs;
          });
          state.events = merged.slice(0, maxNotifications);
        }
      });
      cleanup(() => {
        for (const unsub of unsubscribes) unsub();
        client.disconnect();
      });
    }).catch(() => {
      state.connected = false;
    });
    cleanup(() => { window.removeEventListener('resize', handleResize); });
  });

  useVisibleTask$(({ cleanup }) => {
    const handleKeydown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (state.detailOpen) {
          state.detailOpen = false;
          state.selectedEvent = null;
        } else if (isOpen.value) {
          isOpen.value = false;
        }
      }
      if (e.key === 'n' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const target = e.target as HTMLElement;
        if (target.tagName !== 'INPUT' && target.tagName !== 'TEXTAREA') {
          isOpen.value = !isOpen.value;
          if (isOpen.value) state.unreadCount = 0;
        }
      }
    };
    window.addEventListener('keydown', handleKeydown);
    cleanup(() => { window.removeEventListener('keydown', handleKeydown); });
  });

  const panelWidth = isOpen.value ? PANEL_WIDTH_OPEN : isPeeking.value ? PANEL_WIDTH_PEEK : PANEL_WIDTH_CLOSED;

  return (
    <>
      <div
        ref={panelRef}
        class="fixed top-0 right-0 bottom-0 z-[60] transition-all duration-300 ease-out pointer-events-auto"
        style={{ width: `${panelWidth}px` }}
        onMouseLeave$={handleMouseLeave}
      >
        <div
          class={`absolute left-0 top-1/2 -translate-y-1/2 w-2 h-24 rounded-l-full bg-md-primary/40 backdrop-blur-md cursor-pointer transition-all duration-300 hover:w-4 hover:bg-md-primary shadow-lg ${isOpen.value ? 'opacity-0 scale-0' : 'opacity-100 scale-100'}`}
          onClick$={openPanel}
          onMouseEnter$={handleMouseEnter}
          role="button"
          aria-label="Notifications"
          data-testid="notification-trigger"
        >
          {state.unreadCount > 0 && (
            <div class="absolute -left-2 -top-2 w-5 h-5 rounded-full bg-md-error text-md-on-error text-[10px] font-bold flex items-center justify-center shadow-md border border-md-error-container">
              {state.unreadCount > 9 ? '9+' : state.unreadCount}
            </div>
          )}
        </div>

        <div
          class={`absolute inset-0 left-2 bg-md-surface-container-high/95 border-l border-[var(--glass-border)] backdrop-blur-xl flex flex-col transition-all duration-300 shadow-2xl glass-sidebar-right-glow ${isOpen.value || isPeeking.value ? 'translate-x-0' : 'translate-x-full'}`}
          data-testid="notification-sidepanel"
        >
          <md-elevation></md-elevation>
          
          <div class="p-4 border-b border-[var(--glass-border)] flex items-center justify-between bg-md-surface-container-lowest/50">
            <div class="flex items-center gap-3">
              <NeonTitle level="span" color="magenta" size="sm" animation="flicker">Pulse Feed</NeonTitle>
              <div class={`w-2 h-2 rounded-full ${state.connected ? 'bg-md-success animate-pulse' : 'bg-md-outline'}`} />
            </div>
            <div class="flex items-center gap-1">
              <Button variant="text" class="h-8 text-[10px] uppercase font-bold tracking-tighter" onClick$={clearAll}>Clear</Button>
              <Button variant="icon" icon="close" class="h-8 w-8" onClick$={closePanel}></Button>
            </div>
          </div>

          <div
            ref={listRef}
            class="flex-1 overflow-y-auto no-scrollbar p-3 space-y-3 bg-md-surface-container-low/20"
            onScroll$={handleScroll}
          >
            {state.events.length === 0 ? (
              <div class="h-full flex flex-col items-center justify-center text-md-on-surface-variant/30 space-y-2 opacity-50">
                <md-icon class="text-4xl">notifications_off</md-icon>
                <div class="text-[10px] font-bold uppercase tracking-widest">No Active Events</div>
              </div>
            ) : (
              state.events.map((event, index) => {
                const itemHeight = 90;
                const itemTop = index * (itemHeight + 12);
                const scale = calculateFisheyeScale(itemTop, itemHeight, viewportHeight.value, scrollTop.value);
                return (
                  <NotificationCard
                    key={`${event.id || ''}-${event.iso}-${index}`}
                    event={event}
                    scale={scale}
                    onViewDetails$={openDetail}
                  />
                );
              })
            )}
          </div>

          <div class="p-3 border-t border-[var(--glass-border)] text-[9px] font-bold uppercase tracking-tighter text-md-on-surface-variant/40 flex items-center justify-between bg-md-surface-container-low">
            <span>Toggle with <span class="text-md-primary">[N]</span></span>
            <span class="mono tracking-normal">{state.events.length} stored</span>
          </div>
        </div>
      </div>

      <NotificationDetailOverlay
        open={state.detailOpen}
        event={state.selectedEvent}
        onClose$={closeDetail}
        onNavigateToActor$={handleNavigateToActor}
        onNavigateToTopic$={handleNavigateToTopic}
      />
    </>
  );
});
