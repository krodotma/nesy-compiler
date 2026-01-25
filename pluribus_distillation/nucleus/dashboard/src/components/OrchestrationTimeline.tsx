/**
 * OrchestrationTimeline - Multi-agent orchestration timeline view
 *
 * Phase 5, Iteration 43 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Timeline visualization of agent activities
 * - Parallel execution tracking
 * - Synchronization points
 * - Dependency visualization
 * - Playback controls
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
} from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export interface TimelineEvent {
  id: string;
  agentId: string;
  agentName: string;
  type: 'task_start' | 'task_end' | 'sync' | 'handoff' | 'error' | 'milestone';
  title: string;
  timestamp: string;
  duration?: number; // ms
  linkedEventId?: string;
  data?: Record<string, unknown>;
}

export interface TimelineAgent {
  id: string;
  name: string;
  color: string;
}

export interface OrchestrationTimelineProps {
  /** Timeline events */
  events: TimelineEvent[];
  /** Agents involved */
  agents: TimelineAgent[];
  /** Start time of timeline */
  startTime: string;
  /** End time of timeline */
  endTime?: string;
  /** Time range in minutes */
  rangeMinutes?: number;
}

// ============================================================================
// Helpers
// ============================================================================

function getEventIcon(type: string): string {
  switch (type) {
    case 'task_start': return '▶';
    case 'task_end': return '✓';
    case 'sync': return '◆';
    case 'handoff': return '→';
    case 'error': return '✕';
    case 'milestone': return '★';
    default: return '●';
  }
}

function getEventColor(type: string): string {
  switch (type) {
    case 'task_start': return '#3b82f6';
    case 'task_end': return '#22c55e';
    case 'sync': return '#a855f7';
    case 'handoff': return '#f97316';
    case 'error': return '#ef4444';
    case 'milestone': return '#eab308';
    default: return '#6b7280';
  }
}

function formatTime(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return dateStr;
  }
}

function formatDuration(ms?: number): string {
  if (!ms) return '';
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

// ============================================================================
// Component
// ============================================================================

export const OrchestrationTimeline = component$<OrchestrationTimelineProps>(({
  events,
  agents,
  startTime,
  endTime,
  rangeMinutes = 30,
}) => {
  // State
  const selectedEventId = useSignal<string | null>(null);
  const hoveredEventId = useSignal<string | null>(null);
  const zoomLevel = useSignal(1);
  const scrollOffset = useSignal(0);

  // Dimensions
  const timelineWidth = 800;
  const rowHeight = 40;
  const headerHeight = 30;
  const leftPadding = 100;
  const rightPadding = 20;

  // Computed
  const selectedEvent = useComputed$(() =>
    events.find(e => e.id === selectedEventId.value)
  );

  const timeRange = useComputed$(() => {
    const start = new Date(startTime).getTime();
    const end = endTime ? new Date(endTime).getTime() : start + rangeMinutes * 60 * 1000;
    return { start, end, duration: end - start };
  });

  // Position calculator
  const getXPosition = $((timestamp: string): number => {
    const time = new Date(timestamp).getTime();
    const range = timeRange.value;
    const ratio = (time - range.start) / range.duration;
    return leftPadding + ratio * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value;
  });

  // Generate time markers
  const timeMarkers = useComputed$(() => {
    const markers: { time: string; position: number }[] = [];
    const range = timeRange.value;
    const interval = Math.max(60000, Math.floor(range.duration / 10)); // At least 1 minute

    for (let t = range.start; t <= range.end; t += interval) {
      const ratio = (t - range.start) / range.duration;
      markers.push({
        time: formatTime(new Date(t).toISOString()),
        position: leftPadding + ratio * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value,
      });
    }

    return markers;
  });

  // Group events by agent
  const eventsByAgent = useComputed$(() => {
    const map = new Map<string, TimelineEvent[]>();
    agents.forEach(a => map.set(a.id, []));
    events.forEach(e => {
      const agentEvents = map.get(e.agentId) || [];
      agentEvents.push(e);
      map.set(e.agentId, agentEvents);
    });
    return map;
  });

  // Find linked events (for drawing connections)
  const linkedPairs = useComputed$(() => {
    const pairs: { from: TimelineEvent; to: TimelineEvent }[] = [];
    events.forEach(e => {
      if (e.linkedEventId) {
        const linked = events.find(other => other.id === e.linkedEventId);
        if (linked) {
          pairs.push({ from: e, to: linked });
        }
      }
    });
    return pairs;
  });

  const totalHeight = headerHeight + agents.length * rowHeight + 20;

  return (
    <div class="rounded-lg border border-border bg-card overflow-hidden">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">ORCHESTRATION TIMELINE</span>
          <span class="text-[9px] text-muted-foreground">
            {formatTime(startTime)} - {endTime ? formatTime(endTime) : 'now'}
          </span>
        </div>
        <div class="flex items-center gap-2">
          <button
            onClick$={() => { zoomLevel.value = Math.max(0.5, zoomLevel.value - 0.25); }}
            class="text-[10px] px-2 py-1 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50"
          >
            −
          </button>
          <span class="text-[9px] text-muted-foreground w-12 text-center">
            {Math.round(zoomLevel.value * 100)}%
          </span>
          <button
            onClick$={() => { zoomLevel.value = Math.min(3, zoomLevel.value + 0.25); }}
            class="text-[10px] px-2 py-1 rounded bg-muted/30 text-muted-foreground hover:bg-muted/50"
          >
            +
          </button>
        </div>
      </div>

      {/* Timeline SVG */}
      <div class="overflow-x-auto">
        <svg
          width={timelineWidth * zoomLevel.value}
          height={totalHeight}
          class="bg-muted/5"
        >
          {/* Time markers */}
          {timeMarkers.value.map((marker, i) => (
            <g key={i}>
              <line
                x1={marker.position}
                y1={headerHeight}
                x2={marker.position}
                y2={totalHeight}
                stroke="rgba(255,255,255,0.05)"
                stroke-dasharray="2,2"
              />
              <text
                x={marker.position}
                y={headerHeight - 8}
                text-anchor="middle"
                fill="rgba(255,255,255,0.4)"
                font-size="9"
              >
                {marker.time}
              </text>
            </g>
          ))}

          {/* Agent rows */}
          {agents.map((agent, agentIndex) => {
            const y = headerHeight + agentIndex * rowHeight;
            const agentEvents = eventsByAgent.value.get(agent.id) || [];

            return (
              <g key={agent.id}>
                {/* Row background */}
                <rect
                  x={0}
                  y={y}
                  width={timelineWidth * zoomLevel.value}
                  height={rowHeight}
                  fill={agentIndex % 2 === 0 ? 'rgba(255,255,255,0.02)' : 'transparent'}
                />

                {/* Agent label */}
                <text
                  x={10}
                  y={y + rowHeight / 2 + 4}
                  fill={agent.color}
                  font-size="10"
                  font-weight="500"
                >
                  @{agent.name}
                </text>

                {/* Agent timeline base */}
                <line
                  x1={leftPadding}
                  y1={y + rowHeight / 2}
                  x2={timelineWidth * zoomLevel.value - rightPadding}
                  y2={y + rowHeight / 2}
                  stroke="rgba(255,255,255,0.1)"
                  stroke-width="2"
                />

                {/* Task bars (for events with duration) */}
                {agentEvents.filter(e => e.duration).map(event => {
                  const startX = new Date(event.timestamp).getTime();
                  const endX = startX + (event.duration || 0);
                  const range = timeRange.value;

                  const x1 = leftPadding + ((startX - range.start) / range.duration) * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value;
                  const x2 = leftPadding + ((endX - range.start) / range.duration) * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value;

                  return (
                    <rect
                      key={`bar-${event.id}`}
                      x={x1}
                      y={y + rowHeight / 2 - 6}
                      width={Math.max(4, x2 - x1)}
                      height={12}
                      rx={2}
                      fill={agent.color}
                      opacity={0.3}
                    />
                  );
                })}

                {/* Event markers */}
                {agentEvents.map(event => {
                  const eventTime = new Date(event.timestamp).getTime();
                  const range = timeRange.value;
                  const x = leftPadding + ((eventTime - range.start) / range.duration) * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value;
                  const isSelected = selectedEventId.value === event.id;
                  const isHovered = hoveredEventId.value === event.id;

                  return (
                    <g
                      key={event.id}
                      onMouseEnter$={() => { hoveredEventId.value = event.id; }}
                      onMouseLeave$={() => { hoveredEventId.value = null; }}
                      onClick$={() => { selectedEventId.value = event.id; }}
                      class="cursor-pointer"
                    >
                      <circle
                        cx={x}
                        cy={y + rowHeight / 2}
                        r={isSelected || isHovered ? 8 : 6}
                        fill={getEventColor(event.type)}
                        stroke={isSelected ? 'white' : 'transparent'}
                        stroke-width={2}
                        class="transition-all"
                      />
                      <text
                        x={x}
                        y={y + rowHeight / 2 + 3}
                        text-anchor="middle"
                        fill="white"
                        font-size="8"
                        class="pointer-events-none"
                      >
                        {getEventIcon(event.type)}
                      </text>

                      {/* Hover tooltip */}
                      {isHovered && (
                        <g>
                          <rect
                            x={x - 60}
                            y={y - 25}
                            width={120}
                            height={20}
                            rx={4}
                            fill="rgba(0,0,0,0.9)"
                          />
                          <text
                            x={x}
                            y={y - 12}
                            text-anchor="middle"
                            fill="white"
                            font-size="9"
                          >
                            {event.title.slice(0, 20)}
                          </text>
                        </g>
                      )}
                    </g>
                  );
                })}
              </g>
            );
          })}

          {/* Connection lines for linked events */}
          {linkedPairs.value.map((pair, i) => {
            const fromAgent = agents.findIndex(a => a.id === pair.from.agentId);
            const toAgent = agents.findIndex(a => a.id === pair.to.agentId);
            if (fromAgent === -1 || toAgent === -1) return null;

            const fromTime = new Date(pair.from.timestamp).getTime();
            const toTime = new Date(pair.to.timestamp).getTime();
            const range = timeRange.value;

            const x1 = leftPadding + ((fromTime - range.start) / range.duration) * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value;
            const x2 = leftPadding + ((toTime - range.start) / range.duration) * (timelineWidth - leftPadding - rightPadding) * zoomLevel.value;
            const y1 = headerHeight + fromAgent * rowHeight + rowHeight / 2;
            const y2 = headerHeight + toAgent * rowHeight + rowHeight / 2;

            return (
              <path
                key={i}
                d={`M ${x1} ${y1} C ${(x1 + x2) / 2} ${y1}, ${(x1 + x2) / 2} ${y2}, ${x2} ${y2}`}
                fill="none"
                stroke="rgba(168, 85, 247, 0.4)"
                stroke-width="1.5"
                stroke-dasharray="4,2"
              />
            );
          })}
        </svg>
      </div>

      {/* Selected event detail */}
      {selectedEvent.value && (
        <div class="p-3 border-t border-border/30 bg-muted/5">
          <div class="flex items-center justify-between mb-2">
            <div class="flex items-center gap-2">
              <span
                class="w-3 h-3 rounded-full"
                style={{ backgroundColor: getEventColor(selectedEvent.value.type) }}
              />
              <span class="text-xs font-medium text-foreground">{selectedEvent.value.title}</span>
            </div>
            <button
              onClick$={() => { selectedEventId.value = null; }}
              class="text-muted-foreground hover:text-foreground"
            >
              ✕
            </button>
          </div>
          <div class="grid grid-cols-3 gap-4 text-[9px]">
            <div>
              <span class="text-muted-foreground">Agent:</span>
              <span class="ml-1 text-foreground">@{selectedEvent.value.agentName}</span>
            </div>
            <div>
              <span class="text-muted-foreground">Time:</span>
              <span class="ml-1 text-foreground">{formatTime(selectedEvent.value.timestamp)}</span>
            </div>
            <div>
              <span class="text-muted-foreground">Duration:</span>
              <span class="ml-1 text-foreground">{formatDuration(selectedEvent.value.duration) || 'N/A'}</span>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div class="p-2 border-t border-border/50 flex items-center justify-center gap-4 text-[9px]">
        {[
          { type: 'task_start', label: 'Start' },
          { type: 'task_end', label: 'End' },
          { type: 'sync', label: 'Sync' },
          { type: 'handoff', label: 'Handoff' },
          { type: 'milestone', label: 'Milestone' },
          { type: 'error', label: 'Error' },
        ].map(item => (
          <div key={item.type} class="flex items-center gap-1">
            <span
              class="w-2 h-2 rounded-full"
              style={{ backgroundColor: getEventColor(item.type) }}
            />
            <span class="text-muted-foreground">{item.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

export default OrchestrationTimeline;
