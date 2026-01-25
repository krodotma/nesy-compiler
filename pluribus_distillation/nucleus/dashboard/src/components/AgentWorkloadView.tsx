/**
 * AgentWorkloadView - Agent workload distribution visualization
 *
 * Phase 5, Iteration 40 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Workload distribution chart
 * - Task queue visualization
 * - Capacity planning indicators
 * - Rebalancing suggestions
 * - Historical workload trends
 */

import {
  component$,
  useSignal,
  useComputed$,
  $,
  type QRL,
} from '@builder.io/qwik';

// M3 Components - AgentWorkloadView
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/filled-button.js';
import '@material/web/button/text-button.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/progress/circular-progress.js';

// ============================================================================
// Types
// ============================================================================

export interface WorkloadAgent {
  id: string;
  name: string;
  class: 'sagent' | 'swagent' | 'cagent';
  status: 'active' | 'idle' | 'busy' | 'offline';
  currentLoad: number; // 0-100
  maxCapacity: number;
  assignedTasks: number;
  completedToday: number;
  avgTaskDuration: number; // minutes
  queueDepth: number;
  capabilities: string[];
  history: { time: string; load: number }[];
}

export interface WorkloadTask {
  id: string;
  title: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  estimatedDuration: number;
  assignedTo?: string;
  requiredCapabilities?: string[];
  createdAt: string;
}

export interface RebalanceSuggestion {
  taskId: string;
  fromAgent?: string;
  toAgent: string;
  reason: string;
  impact: 'high' | 'medium' | 'low';
}

export interface AgentWorkloadViewProps {
  /** Agents with workload data */
  agents: WorkloadAgent[];
  /** Pending tasks */
  pendingTasks: WorkloadTask[];
  /** Rebalance suggestions */
  suggestions?: RebalanceSuggestion[];
  /** Callback when task is reassigned */
  onReassignTask$?: QRL<(taskId: string, toAgentId: string) => void>;
  /** Callback when rebalance is applied */
  onApplyRebalance$?: QRL<(suggestions: RebalanceSuggestion[]) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getLoadColor(load: number): string {
  if (load >= 90) return '#ef4444';  // red
  if (load >= 70) return '#f97316';  // orange
  if (load >= 50) return '#eab308';  // yellow
  return '#22c55e';                   // green
}

function getLoadBgColor(load: number): string {
  if (load >= 90) return 'bg-red-500/20 border-red-500/30';
  if (load >= 70) return 'bg-orange-500/20 border-orange-500/30';
  if (load >= 50) return 'bg-amber-500/20 border-amber-500/30';
  return 'bg-emerald-500/20 border-emerald-500/30';
}

function getStatusColor(status: string): string {
  switch (status) {
    case 'active': return 'text-emerald-400';
    case 'idle': return 'text-blue-400';
    case 'busy': return 'text-amber-400';
    case 'offline': return 'text-gray-400';
    default: return 'text-muted-foreground';
  }
}

function getPriorityColor(priority: string): string {
  switch (priority) {
    case 'critical': return 'bg-red-500/20 text-red-400';
    case 'high': return 'bg-orange-500/20 text-orange-400';
    case 'medium': return 'bg-amber-500/20 text-amber-400';
    case 'low': return 'bg-blue-500/20 text-blue-400';
    default: return 'bg-muted/20 text-muted-foreground';
  }
}

function formatDuration(mins: number): string {
  if (mins < 60) return `${mins}m`;
  const hours = Math.floor(mins / 60);
  const remaining = mins % 60;
  return remaining ? `${hours}h ${remaining}m` : `${hours}h`;
}

// ============================================================================
// Component
// ============================================================================

export const AgentWorkloadView = component$<AgentWorkloadViewProps>(({
  agents,
  pendingTasks,
  suggestions = [],
  onReassignTask$,
  onApplyRebalance$,
}) => {
  // State
  const selectedAgentId = useSignal<string | null>(null);
  const showAssignModal = useSignal(false);
  const selectedTaskId = useSignal<string | null>(null);
  const viewMode = useSignal<'grid' | 'chart'>('grid');

  // Computed
  const selectedAgent = useComputed$(() =>
    agents.find(a => a.id === selectedAgentId.value)
  );

  const overallStats = useComputed$(() => {
    const totalCapacity = agents.reduce((sum, a) => sum + a.maxCapacity, 0);
    const totalLoad = agents.reduce((sum, a) => sum + a.currentLoad, 0);
    const totalQueue = agents.reduce((sum, a) => sum + a.queueDepth, 0);
    const busyAgents = agents.filter(a => a.currentLoad >= 70).length;
    const idleAgents = agents.filter(a => a.currentLoad < 30 && a.status !== 'offline').length;

    return {
      avgLoad: agents.length > 0 ? Math.round(totalLoad / agents.length) : 0,
      totalCapacity,
      totalQueue,
      busyAgents,
      idleAgents,
      pendingTasks: pendingTasks.length,
    };
  });

  const sortedAgents = useComputed$(() =>
    [...agents].sort((a, b) => b.currentLoad - a.currentLoad)
  );

  // SVG chart dimensions
  const chartWidth = 400;
  const chartHeight = 150;
  const barWidth = Math.min(30, (chartWidth - 40) / agents.length - 4);

  // Actions
  const selectAgent = $((agentId: string) => {
    selectedAgentId.value = selectedAgentId.value === agentId ? null : agentId;
  });

  const reassignTask = $(async (taskId: string, toAgentId: string) => {
    if (onReassignTask$) {
      await onReassignTask$(taskId, toAgentId);
    }
    showAssignModal.value = false;
    selectedTaskId.value = null;
  });

  const applyRebalance = $(async () => {
    if (onApplyRebalance$ && suggestions.length > 0) {
      await onApplyRebalance$(suggestions);
    }
  });

  return (
    <div class="rounded-lg border border-border bg-card">
      {/* Header */}
      <div class="flex items-center justify-between p-3 border-b border-border/50">
        <div class="flex items-center gap-2">
          <span class="text-xs font-semibold text-muted-foreground">AGENT WORKLOAD</span>
          <span class={`text-[10px] px-2 py-0.5 rounded border ${getLoadBgColor(overallStats.value.avgLoad)}`}>
            {overallStats.value.avgLoad}% avg
          </span>
        </div>
        <div class="flex items-center gap-2">
          <button
            onClick$={() => { viewMode.value = 'grid'; }}
            class={`text-[10px] px-2 py-1 rounded transition-colors ${
              viewMode.value === 'grid' ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/30'
            }`}
          >
            Grid
          </button>
          <button
            onClick$={() => { viewMode.value = 'chart'; }}
            class={`text-[10px] px-2 py-1 rounded transition-colors ${
              viewMode.value === 'chart' ? 'bg-primary/20 text-primary' : 'text-muted-foreground hover:bg-muted/30'
            }`}
          >
            Chart
          </button>
        </div>
      </div>

      {/* Summary stats */}
      <div class="grid grid-cols-5 gap-2 p-3 border-b border-border/30 bg-muted/5">
        <div class="text-center">
          <div class="text-lg font-bold text-foreground">{agents.length}</div>
          <div class="text-[9px] text-muted-foreground">Agents</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-amber-400">{overallStats.value.busyAgents}</div>
          <div class="text-[9px] text-muted-foreground">Busy</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-blue-400">{overallStats.value.idleAgents}</div>
          <div class="text-[9px] text-muted-foreground">Idle</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-purple-400">{overallStats.value.totalQueue}</div>
          <div class="text-[9px] text-muted-foreground">Queued</div>
        </div>
        <div class="text-center">
          <div class="text-lg font-bold text-orange-400">{pendingTasks.length}</div>
          <div class="text-[9px] text-muted-foreground">Pending</div>
        </div>
      </div>

      {/* Rebalance suggestions */}
      {suggestions.length > 0 && (
        <div class="p-2 border-b border-border/30 bg-amber-500/5">
          <div class="flex items-center justify-between">
            <div class="flex items-center gap-2">
              <span class="text-[9px] font-semibold text-amber-400">
                {suggestions.length} REBALANCE SUGGESTIONS
              </span>
            </div>
            <button
              onClick$={applyRebalance}
              class="text-[9px] px-2 py-1 rounded bg-amber-500/20 text-amber-400 hover:bg-amber-500/30"
            >
              Apply All
            </button>
          </div>
          <div class="mt-2 space-y-1">
            {suggestions.slice(0, 3).map(s => (
              <div key={s.taskId} class="text-[9px] text-muted-foreground">
                Move task to <span class="text-amber-400">@{s.toAgent}</span>: {s.reason}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Main content */}
      {viewMode.value === 'grid' ? (
        <div class="grid grid-cols-2 gap-0 min-h-[300px]">
          {/* Agent grid */}
          <div class="border-r border-border/30 p-3 max-h-[350px] overflow-y-auto">
            <div class="grid grid-cols-2 gap-2">
              {sortedAgents.value.map(agent => (
                <div
                  key={agent.id}
                  onClick$={() => selectAgent(agent.id)}
                  class={`p-3 rounded-lg border cursor-pointer transition-all ${
                    selectedAgentId.value === agent.id
                      ? 'border-primary bg-primary/5'
                      : `border-border/30 hover:border-border/50 ${getLoadBgColor(agent.currentLoad)}`
                  }`}
                >
                  <div class="flex items-center justify-between mb-2">
                    <span class="text-[10px] font-medium text-foreground">{agent.name}</span>
                    <span class={`text-[8px] ${getStatusColor(agent.status)}`}>
                      ● {agent.status}
                    </span>
                  </div>

                  {/* Load gauge */}
                  <div class="relative h-20 flex items-end justify-center mb-2">
                    <div
                      class="w-8 rounded-t transition-all duration-500"
                      style={{
                        height: `${agent.currentLoad}%`,
                        backgroundColor: getLoadColor(agent.currentLoad),
                      }}
                    />
                    <div class="absolute inset-0 flex items-center justify-center">
                      <span class="text-lg font-bold text-foreground">{agent.currentLoad}%</span>
                    </div>
                  </div>

                  <div class="grid grid-cols-2 gap-1 text-[8px] text-muted-foreground">
                    <div>Queue: {agent.queueDepth}</div>
                    <div>Done: {agent.completedToday}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Agent detail / Task assignment */}
          <div class="p-3 max-h-[350px] overflow-y-auto">
            {selectedAgent.value ? (
              <div>
                <div class="mb-4">
                  <div class="flex items-center justify-between">
                    <span class="text-sm font-medium text-foreground">{selectedAgent.value.name}</span>
                    <span class={`text-[9px] px-2 py-0.5 rounded ${
                      selectedAgent.value.class === 'sagent' ? 'bg-purple-500/20 text-purple-400' :
                      selectedAgent.value.class === 'swagent' ? 'bg-cyan-500/20 text-cyan-400' :
                      'bg-orange-500/20 text-orange-400'
                    }`}>
                      {selectedAgent.value.class.toUpperCase()}
                    </span>
                  </div>

                  <div class="grid grid-cols-2 gap-2 mt-3 text-[9px]">
                    <div>
                      <span class="text-muted-foreground">Current Load:</span>
                      <span class="ml-1 text-foreground font-bold">{selectedAgent.value.currentLoad}%</span>
                    </div>
                    <div>
                      <span class="text-muted-foreground">Capacity:</span>
                      <span class="ml-1 text-foreground">{selectedAgent.value.maxCapacity}</span>
                    </div>
                    <div>
                      <span class="text-muted-foreground">Avg Duration:</span>
                      <span class="ml-1 text-foreground">{formatDuration(selectedAgent.value.avgTaskDuration)}</span>
                    </div>
                    <div>
                      <span class="text-muted-foreground">Completed:</span>
                      <span class="ml-1 text-emerald-400">{selectedAgent.value.completedToday} today</span>
                    </div>
                  </div>

                  <div class="mt-3">
                    <span class="text-[9px] text-muted-foreground">Capabilities:</span>
                    <div class="flex flex-wrap gap-1 mt-1">
                      {selectedAgent.value.capabilities.map(cap => (
                        <span key={cap} class="text-[8px] px-1.5 py-0.5 rounded bg-muted/30 text-muted-foreground">
                          {cap}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Load history mini-chart */}
                <div class="mb-4 p-2 rounded bg-muted/10 border border-border/30">
                  <div class="text-[9px] text-muted-foreground mb-2">Load History (24h)</div>
                  <svg width="100%" height="40" viewBox="0 0 200 40">
                    {selectedAgent.value.history.length > 1 && (
                      <path
                        d={selectedAgent.value.history.map((h, i) => {
                          const x = (i / (selectedAgent.value!.history.length - 1)) * 200;
                          const y = 40 - (h.load / 100) * 40;
                          return i === 0 ? `M ${x} ${y}` : `L ${x} ${y}`;
                        }).join(' ')}
                        fill="none"
                        stroke={getLoadColor(selectedAgent.value.currentLoad)}
                        stroke-width="2"
                      />
                    )}
                  </svg>
                </div>

                {/* Pending tasks to assign */}
                <div>
                  <div class="text-[9px] font-semibold text-muted-foreground mb-2">ASSIGN TASKS</div>
                  <div class="space-y-1">
                    {pendingTasks.filter(t => !t.assignedTo).slice(0, 5).map(task => (
                      <div
                        key={task.id}
                        class="flex items-center justify-between p-2 rounded bg-muted/10 hover:bg-muted/20 transition-colors"
                      >
                        <div>
                          <div class="text-[10px] text-foreground">{task.title}</div>
                          <div class="flex items-center gap-2 mt-0.5">
                            <span class={`text-[8px] px-1 py-0.5 rounded ${getPriorityColor(task.priority)}`}>
                              {task.priority}
                            </span>
                            <span class="text-[8px] text-muted-foreground">
                              ~{formatDuration(task.estimatedDuration)}
                            </span>
                          </div>
                        </div>
                        <button
                          onClick$={() => reassignTask(task.id, selectedAgent.value!.id)}
                          class="text-[9px] px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30"
                        >
                          Assign
                        </button>
                      </div>
                    ))}
                    {pendingTasks.filter(t => !t.assignedTo).length === 0 && (
                      <div class="text-[9px] text-muted-foreground text-center py-2">
                        No unassigned tasks
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div class="flex items-center justify-center h-full text-[10px] text-muted-foreground">
                Select an agent to view details
              </div>
            )}
          </div>
        </div>
      ) : (
        /* Chart view */
        <div class="p-4">
          <svg width="100%" height={chartHeight} viewBox={`0 0 ${chartWidth} ${chartHeight}`}>
            {/* Background grid */}
            {[0, 25, 50, 75, 100].map(pct => (
              <g key={pct}>
                <line
                  x1={30}
                  y1={chartHeight - 20 - (pct / 100) * (chartHeight - 40)}
                  x2={chartWidth - 10}
                  y2={chartHeight - 20 - (pct / 100) * (chartHeight - 40)}
                  stroke="rgba(255,255,255,0.1)"
                  stroke-dasharray="2,2"
                />
                <text
                  x={25}
                  y={chartHeight - 17 - (pct / 100) * (chartHeight - 40)}
                  text-anchor="end"
                  fill="rgba(255,255,255,0.4)"
                  font-size="8"
                >
                  {pct}%
                </text>
              </g>
            ))}

            {/* Bars */}
            {sortedAgents.value.map((agent, i) => {
              const x = 40 + i * (barWidth + 4);
              const barHeight = (agent.currentLoad / 100) * (chartHeight - 40);

              return (
                <g key={agent.id}>
                  <rect
                    x={x}
                    y={chartHeight - 20 - barHeight}
                    width={barWidth}
                    height={barHeight}
                    fill={getLoadColor(agent.currentLoad)}
                    rx={2}
                    class="cursor-pointer hover:opacity-80 transition-opacity"
                    onClick$={() => selectAgent(agent.id)}
                  />
                  <text
                    x={x + barWidth / 2}
                    y={chartHeight - 5}
                    text-anchor="middle"
                    fill="rgba(255,255,255,0.6)"
                    font-size="7"
                    class="pointer-events-none"
                  >
                    {agent.name.slice(0, 4)}
                  </text>
                  <text
                    x={x + barWidth / 2}
                    y={chartHeight - 25 - barHeight}
                    text-anchor="middle"
                    fill="white"
                    font-size="8"
                    font-weight="bold"
                    class="pointer-events-none"
                  >
                    {agent.currentLoad}%
                  </text>
                </g>
              );
            })}

            {/* Threshold line at 80% */}
            <line
              x1={30}
              y1={chartHeight - 20 - (80 / 100) * (chartHeight - 40)}
              x2={chartWidth - 10}
              y2={chartHeight - 20 - (80 / 100) * (chartHeight - 40)}
              stroke="#ef4444"
              stroke-width="1"
              stroke-dasharray="4,4"
            />
          </svg>

          {/* Legend */}
          <div class="mt-3 flex items-center justify-center gap-4 text-[9px]">
            <div class="flex items-center gap-1">
              <div class="w-3 h-3 rounded" style={{ backgroundColor: '#22c55e' }} />
              <span class="text-muted-foreground">0-50%</span>
            </div>
            <div class="flex items-center gap-1">
              <div class="w-3 h-3 rounded" style={{ backgroundColor: '#eab308' }} />
              <span class="text-muted-foreground">50-70%</span>
            </div>
            <div class="flex items-center gap-1">
              <div class="w-3 h-3 rounded" style={{ backgroundColor: '#f97316' }} />
              <span class="text-muted-foreground">70-90%</span>
            </div>
            <div class="flex items-center gap-1">
              <div class="w-3 h-3 rounded" style={{ backgroundColor: '#ef4444' }} />
              <span class="text-muted-foreground">90%+</span>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div class="p-2 border-t border-border/50 text-[9px] text-muted-foreground text-center">
        {agents.length} agents • {pendingTasks.length} pending tasks • {overallStats.value.avgLoad}% avg load
      </div>
    </div>
  );
});

export default AgentWorkloadView;
