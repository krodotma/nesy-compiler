import { component$, type Signal, $, useSignal, useVisibleTask$ } from '@builder.io/qwik';
import type { BusEvent } from '../../lib/state/types';
import { EventSearchBox, TimelineSparkline, EventFlowmap, EnrichedEventCard, EventStatsBadges, type SearchMode } from '../EventVisualization';
import { VirtualList } from '../ui/VirtualList';

interface EventsViewProps {
  events: Signal<BusEvent[]>;
  filteredEvents: Signal<BusEvent[]>;
  eventSearchPattern: Signal<string>;
  eventSearchMode: Signal<SearchMode>;
  eventFilter: Signal<string | null>;
  showEventTimeline: Signal<boolean>;
  showEventFlowmap: Signal<boolean>;
  showNdjsonView: Signal<boolean>;
}

export const EventsView = component$<EventsViewProps>((props) => {
  const {
    events,
    filteredEvents,
    eventSearchPattern,
    eventSearchMode,
    eventFilter,
    showEventTimeline,
    showEventFlowmap,
    showNdjsonView
  } = props;

  const viewportHeight = useSignal(900);
  useVisibleTask$(({ cleanup }) => {
    const update = () => {
      viewportHeight.value = typeof window !== 'undefined' ? window.innerHeight : 900;
    };
    update();
    window.addEventListener('resize', update, { passive: true });
    cleanup(() => window.removeEventListener('resize', update));
  });

  const renderNdjsonRow = $((event: BusEvent) => (
    <div
      class={`px-2 py-0.5 border-b border-[var(--glass-border-subtle)] glass-hover-glow glass-transition-hover truncate ${
        event.level === 'error' ? 'glass-status-error' :
        event.level === 'warn' ? 'glass-status-warning' :
        event.kind === 'artifact' ? 'text-purple-400' :
        'glass-text-muted'
      }`}
    >
      {JSON.stringify(event)}
    </div>
  ));

  // Reverse items for display once to avoid expensive reverse in render loop
  const displayEvents = filteredEvents.value.slice().reverse();

  return (
    <div class="space-y-4">
      {/* Search & Visualization Controls */}
      <div class="flex flex-col lg:flex-row gap-4">
        {/* Search Box */}
        <div class="flex-1">
          <EventSearchBox
            onSearch$={((pattern: string, mode: SearchMode) => {
              eventSearchPattern.value = pattern;
              eventSearchMode.value = mode;
            })}
            placeholder="Search: strp.*, /error|warn/i, ◇response, @claude..."
          />
        </div>

        {/* Visualization Toggles */}
        <div class="flex items-center gap-2">
          <button
            onClick$={() => showEventTimeline.value = !showEventTimeline.value}
            class={`glass-chip glass-transition-hover ${
              showEventTimeline.value
                ? 'glass-chip-accent-cyan'
                : 'glass-hover-glow'
            }`}
          >
            📈 Timeline
          </button>
          <button
            onClick$={() => showEventFlowmap.value = !showEventFlowmap.value}
            class={`glass-chip glass-transition-hover ${
              showEventFlowmap.value
                ? 'glass-chip-accent-purple'
                : 'glass-hover-glow'
            }`}
          >
            🔗 Flowmap
          </button>
          <button
            onClick$={() => showNdjsonView.value = !showNdjsonView.value}
            class={`glass-chip glass-transition-hover ${
              showNdjsonView.value
                ? 'glass-chip-accent-cyan'
                : 'glass-hover-glow'
            }`}
          >
            📋 NDJSON
          </button>
        </div>
      </div>

      {/* Dimensional Event Stats Header (Clickable Filters) */}
      <div class="grid grid-cols-2 md:grid-cols-6 gap-2">
        <div
          class={`glass-surface p-3 text-center cursor-pointer glass-hover-glow glass-transition-hover ${!eventFilter.value ? 'ring-2 ring-primary' : ''}`}
          onClick$={() => eventFilter.value = null}
        >
          <div class="text-2xl font-bold text-primary">{events.value.length}</div>
          <div class="text-xs glass-text-muted">Total Events</div>
        </div>
        <div
          class={`glass-surface-subtle p-3 text-center cursor-pointer glass-hover-glow glass-transition-hover border border-green-500/30 ${eventFilter.value === 'metric' ? 'ring-2 ring-green-500' : ''}`}
          onClick$={() => eventFilter.value = 'metric'}
        >
          <div class="text-2xl font-bold text-green-400">
            {events.value.filter(e => e.kind === 'metric').length}
          </div>
          <div class="text-xs glass-text-muted">Metrics</div>
        </div>
        <div
          class={`glass-surface-subtle p-3 text-center cursor-pointer glass-hover-glow glass-transition-hover border border-blue-500/30 ${eventFilter.value === 'request' ? 'ring-2 ring-blue-500' : ''}`}
          onClick$={() => eventFilter.value = 'request'}
        >
          <div class="text-2xl font-bold text-blue-400">
            {events.value.filter(e => e.kind === 'request').length}
          </div>
          <div class="text-xs glass-text-muted">Requests</div>
        </div>
        <div
          class={`glass-surface-subtle p-3 text-center cursor-pointer glass-hover-glow glass-transition-hover border border-purple-500/30 ${eventFilter.value === 'artifact' ? 'ring-2 ring-purple-500' : ''}`}
          onClick$={() => eventFilter.value = 'artifact'}
        >
          <div class="text-2xl font-bold text-purple-400">
            {events.value.filter(e => e.kind === 'artifact').length}
          </div>
          <div class="text-xs glass-text-muted">Artifacts</div>
        </div>
        <div
          class={`glass-surface-subtle p-3 text-center cursor-pointer glass-hover-glow glass-transition-hover border border-red-500/30 ${eventFilter.value === 'error' ? 'ring-2 ring-red-500' : ''}`}
          onClick$={() => eventFilter.value = 'error'}
        >
          <div class="text-2xl font-bold text-red-400">
            {events.value.filter(e => e.level === 'error').length}
          </div>
          <div class="text-xs glass-text-muted">Errors</div>
        </div>
        <div
          class={`glass-surface-subtle p-3 text-center cursor-pointer glass-hover-glow glass-transition-hover border border-yellow-500/30 ${eventFilter.value === 'high-impact' ? 'ring-2 ring-yellow-500' : ''}`}
          onClick$={() => eventFilter.value = 'high-impact'}
        >
          <div class="text-2xl font-bold text-yellow-400">
            {events.value.filter(e => (e as any).semantic?.impact === 'high' || (e as any).semantic?.impact === 'critical').length}
          </div>
          <div class="text-xs glass-text-muted">High Impact</div>
        </div>
      </div>

      {/* Semantic Stats Badges */}
      <EventStatsBadges events={filteredEvents.value} />

      {/* Timeline & Flowmap Visualizations */}
      {(showEventTimeline.value || showEventFlowmap.value) && (
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {showEventTimeline.value && (
            <TimelineSparkline events={filteredEvents.value} buckets={60} height={60} width={400} />
          )}
          {showEventFlowmap.value && (
            <EventFlowmap events={filteredEvents.value} maxNodes={16} height={180} />
          )}
        </div>
      )}

      {/* Event Card Profiler Grid OR NDJSON Raw View */}
      <div class="glass-surface">
        <div class="p-3 border-b border-[var(--glass-border)] flex items-center justify-between">
          <div class="flex items-center gap-3">
            <h2 class="font-semibold glass-text-title">{showNdjsonView.value ? 'NDJSON Raw Feed' : 'Event Profiler'}</h2>
            <span class="glass-chip glass-chip-accent-emerald">Live Feed</span>
            {eventSearchPattern.value && (
              <span class="glass-chip glass-chip-accent-cyan font-mono">
                {eventSearchMode.value}: {eventSearchPattern.value}
              </span>
            )}
          </div>
          <span class="text-sm glass-text-muted">{filteredEvents.value.length} items</span>
        </div>

        {/* NDJSON Raw View */}
        {showNdjsonView.value ? (
          <div class="p-2 h-[calc(100vh-520px)] bg-black/50 font-mono text-[10px]">
            {filteredEvents.value.length > 0 ? (
                <VirtualList
                    items={displayEvents}
                    height={Math.max(240, viewportHeight.value - 540)}
                    itemHeight={24} // Approximate height per line
                    renderItem$={renderNdjsonRow}
                    className="custom-scrollbar"
                />
            ) : (
                <div class="p-8 text-center text-muted-foreground">
                    No events to display
                </div>
            )}
          </div>
        ) : (
          /* Event Cards View (Standard Grid) */
          <div class="p-4 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4 overflow-auto h-[calc(100vh-520px)] content-start">
            {displayEvents.map((event, i) => (
              <EnrichedEventCard
                key={i}
                event={event}
                index={i}
                showLTL={true}
                showVectors={true}
                showKG={true}
              />
            ))}

            {filteredEvents.value.length === 0 && (
              <div class="col-span-full p-12 text-center text-muted-foreground border-2 border-dashed border-[var(--glass-border-subtle)] rounded-xl">
                <div class="text-4xl mb-4">📭</div>
                No events match the active filter.
                {eventSearchPattern.value && (
                  <div class="mt-2 text-xs">
                    Search: <code class="bg-muted/30 px-1 rounded">{eventSearchPattern.value}</code>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
});
