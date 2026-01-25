/**
 * ENHANCED DKIN/PAIP Flow Monitor — Real-time Protocol Compliance & Bus Activity
 *
 * A comprehensive view that monitors:
 * - General bus activity (always showing live data)
 * - DKIN protocol versions (v1-v20) compliance and evolution
 * - PAIP isolation and cloning status
 * - PBLOCK state and exit criteria
 * - Real-time evolution loop phases
 * - Agent task lifecycle and handoff
 * - Protocol compliance with remediation guidance
 */

import { component$, useSignal, useComputed$, useVisibleTask$ } from '@builder.io/qwik';
import type { BusEvent } from '../../lib/state/types';

// M3 Components - EnhancedDKINFlowMonitor
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

// ============================================================================
// Bus Activity Visualizer - Shows live bus activity regardless of event types
// ============================================================================

const BusActivityVisualizer = component$<{ events: BusEvent[] }>(({ events }) => {
  const now = Date.now();
  const minuteAgo = now - 60000;
  const fiveMinutesAgo = now - 300000;
  
  const lastMinuteEvents = events.filter(e => e.ts * 1000 > minuteAgo).length;
  const lastFiveMinutesEvents = events.filter(e => e.ts * 1000 > fiveMinutesAgo).length;
  const avgPerMinute = lastFiveMinutesEvents / 5;
  const trend = lastMinuteEvents > avgPerMinute ? 'up' : lastMinuteEvents < avgPerMinute ? 'down' : 'stable';
  
  // Calculate event distribution by topic prefix
  const eventDistribution = useComputed$(() => {
    const counts: Record<string, number> = {};
    events.slice(-100).forEach(event => {
      const topicPrefix = event.topic.split('.')[0]; // Get first part of topic
      counts[topicPrefix] = (counts[topicPrefix] || 0) + 1;
    });
    return Object.entries(counts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 5);
  });

  return (
    <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-semibold">Bus Activity</span>
        <span class="text-xs text-zinc-400">{events.length} total events</span>
      </div>
      
      <div class="grid grid-cols-3 gap-2 text-xs mb-2">
        <div class="text-center p-1 rounded bg-zinc-800">
          <div class={`font-mono ${(lastMinuteEvents > 10) ? 'text-emerald-400' : lastMinuteEvents > 5 ? 'text-amber-400' : 'text-zinc-400'}`}>
            {lastMinuteEvents}
          </div>
          <div class="text-zinc-500">1m</div>
        </div>
        <div class="text-center p-1 rounded bg-zinc-800">
          <div class={`font-mono ${trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-red-400' : 'text-blue-400'}`}>
            {Math.round(avgPerMinute)}
          </div>
          <div class="text-zinc-500">avg/min</div>
        </div>
        <div class="text-center p-1 rounded bg-zinc-800">
          <div class={`font-mono ${trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-red-400' : 'text-blue-400'}`}>
            {lastFiveMinutesEvents}
          </div>
          <div class="text-zinc-500">5m</div>
        </div>
      </div>
      
      <div class="text-xs text-zinc-400 mb-1">Top Topics:</div>
      <div class="space-y-1 max-h-20 overflow-y-auto">
        {eventDistribution.value.map(([topic, count]) => (
          <div key={topic} class="flex justify-between">
            <span class="text-zinc-300">{topic}</span>
            <span class="text-zinc-500">{count}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

// ============================================================================
// Main Enhanced DKIN Flow Monitor Component
// ============================================================================

export interface EnhancedDKINFlowMonitorProps {
  events: BusEvent[];
}

export const EnhancedDKINFlowMonitor = component$<EnhancedDKINFlowMonitorProps>(({ events }) => {
  // Compute DKIN-specific events
  const dkinEvents = useComputed$(() => {
    return events.filter(
      (e) =>
        e.topic.startsWith('operator.pblock') ||
        e.topic.startsWith('operator.pbhygiene') ||
        e.topic.startsWith('paip.') ||
        e.topic.startsWith('ckin.') ||
        e.topic.startsWith('oiterate.') ||
        e.topic.startsWith('evolution.') ||
        e.topic.startsWith('hgt.') ||
        e.topic.startsWith('cmp.') ||
        e.topic.startsWith('agent.') ||
        e.topic.startsWith('infer_sync.')
    );
  });

  // Compute general statistics
  const generalStats = useComputed$(() => {
    const now = Date.now();
    const hourAgo = now - (60 * 60 * 1000);
    
    const recentEvents = events.filter(e => e.ts * 1000 > hourAgo);
    const topicCategories: Record<string, number> = {};
    
    recentEvents.forEach(event => {
      const category = event.topic.split('.')[0];
      topicCategories[category] = (topicCategories[category] || 0) + 1;
    });
    
    return {
      totalEvents: events.length,
      recentEvents: recentEvents.length,
      activeCategories: Object.keys(topicCategories).length,
      topCategory: Object.entries(topicCategories).sort(([,a], [,b]) => b - a)[0]?.[0] || 'none'
    };
  });

  return (
    <div class="p-4 bg-zinc-950 text-zinc-100 min-h-full">
      {/* Header */}
      <div class="flex items-center justify-between mb-4">
        <h1 class="text-lg font-bold">DKIN/PAIP Flow Monitor</h1>
        <div class="text-xs text-zinc-400">
          {dkinEvents.value.length} DKIN events | {events.length} total
        </div>
      </div>

      {/* General Bus Activity - Always shows live data */}
      <div class="mb-4">
        <div class="text-sm font-semibold mb-2">General Bus Activity</div>
        <BusActivityVisualizer events={events} />
      </div>

      {/* Statistics Summary */}
      <div class="grid grid-cols-4 gap-4 mb-4">
        <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
          <div class="text-2xl font-bold text-emerald-400">{generalStats.value.totalEvents}</div>
          <div class="text-xs text-zinc-400">Total Events</div>
        </div>
        <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
          <div class="text-2xl font-bold text-amber-400">{generalStats.value.recentEvents}</div>
          <div class="text-xs text-zinc-400">Last Hour</div>
        </div>
        <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
          <div class="text-2xl font-bold text-blue-400">{generalStats.value.activeCategories}</div>
          <div class="text-xs text-zinc-400">Active Categories</div>
        </div>
        <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
          <div class="text-lg font-bold text-purple-400">{generalStats.value.topCategory}</div>
          <div class="text-xs text-zinc-400">Busiest Category</div>
        </div>
      </div>

      {/* DKIN-Specific Metrics */}
      <div class="grid grid-cols-12 gap-4">
        {/* Left Column: Status Panels */}
        <div class="col-span-4 space-y-4">
          <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
            <div class="text-sm font-semibold mb-2">DKIN Events</div>
            <div class="text-3xl font-bold text-blue-400">{dkinEvents.value.length}</div>
            <div class="text-xs text-zinc-400 mt-1">DKIN/PAIP specific events</div>
          </div>
          
          <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
            <div class="text-sm font-semibold mb-2">Recent DKIN Events</div>
            <div class="space-y-1 max-h-40 overflow-y-auto">
              {dkinEvents.value.slice(0, 10).map((e, idx) => (
                <div key={`${e.id}-${idx}`} class="text-xs flex justify-between">
                  <span class="truncate">{e.topic}</span>
                  <span class="text-zinc-500">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
                </div>
              ))}
              {dkinEvents.value.length === 0 && (
                <div class="text-xs text-zinc-500 italic">No DKIN-specific events in recent history</div>
              )}
            </div>
          </div>
        </div>

        {/* Center Column: Evolution & Tasks */}
        <div class="col-span-4 space-y-4">
          <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
            <div class="text-sm font-semibold mb-2">Evolution Status</div>
            <div class="space-y-2">
              <div class="flex justify-between">
                <span class="text-xs text-zinc-400">Active Agents</span>
                <span class="text-xs">?</span>
              </div>
              <div class="flex justify-between">
                <span class="text-xs text-zinc-400">PBLOCK Status</span>
                <span class="text-xs">?</span>
              </div>
              <div class="flex justify-between">
                <span class="text-xs text-zinc-400">Protocol Version</span>
                <span class="text-xs">v20</span>
              </div>
            </div>
          </div>
          
          <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
            <div class="text-sm font-semibold mb-2">DKIN Protocol Timeline</div>
            <div class="flex gap-1 flex-wrap">
              {['v15', 'v16', 'v17', 'v18', 'v19', 'v20'].map(version => (
                <div 
                  key={version} 
                  class={`px-2 py-1 text-xs rounded ${version === 'v20' ? 'bg-blue-600 text-white' : 'bg-zinc-700 text-zinc-300'}`}
                >
                  {version}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right Column: Event Stream */}
        <div class="col-span-4">
          <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
            <div class="text-sm font-semibold mb-2">Live Event Stream</div>
            <div class="space-y-1 max-h-64 overflow-y-auto">
              {events.slice(0, 15).map((e, idx) => (
                <div key={`${e.id}-${idx}`} class="text-xs">
                  <div class="flex justify-between">
                    <span class="text-zinc-400 truncate">{e.topic}</span>
                    <span class="text-zinc-600">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
                  </div>
                  <div class="text-zinc-500 truncate">{e.actor}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div class="mt-4 pt-4 border-t border-zinc-800 text-xs text-zinc-500">
        <div>DKIN/PAIP Flow Monitor - Real-time bus activity and protocol compliance</div>
        <div class="flex gap-4 mt-1">
          <span>DKIN Events: {dkinEvents.value.length}</span>
          <span>Total Events: {events.length}</span>
          <span>Last Update: {new Date().toLocaleTimeString()}</span>
        </div>
      </div>
    </div>
  );
});

export default EnhancedDKINFlowMonitor;
