/**
 * AgentCoordinationDashboard - Multi-agent orchestration view
 *
 * Phase 5, Iteration 36 of OITERATE lanes-widget-enhancement
 *
 * Features:
 * - Real-time agent status monitoring
 * - Task assignment interface
 * - Communication channels
 * - Health monitoring
 * - Workload distribution
 */

import {
  component$,
  useSignal,
  useStore,
  useComputed$,
  useVisibleTask$,
  $,
  noSerialize,
  type NoSerialize,
  type QRL,
} from '@builder.io/qwik';
import { Button } from './ui/Button';
import { Card } from './ui/Card';
import { Input } from './ui/Input';

// ============================================================================
// Types
// ============================================================================

export type AgentStatus = 'active' | 'idle' | 'busy' | 'offline' | 'error';
export type AgentClass = 'sagent' | 'swagent' | 'cagent';

export interface Agent {
  id: string;
  name: string;
  class: AgentClass;
  status: AgentStatus;
  currentTask?: string;
  assignedLanes: string[];
  workload: number; // 0-100
  lastHeartbeat: string;
  capabilities: string[];
  metrics: {
    tasksCompleted: number;
    avgCompletionTime: number;
    errorRate: number;
  };
}

export interface AgentTask {
  id: string;
  title: string;
  description?: string;
  assignedTo?: string;
  laneId?: string;
  priority: 'low' | 'medium' | 'high' | 'critical';
  status: 'pending' | 'assigned' | 'in_progress' | 'completed' | 'failed';
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
}

export interface AgentMessage {
  id: string;
  from: string;
  to: string | 'broadcast';
  content: string;
  timestamp: string;
  type: 'info' | 'warning' | 'error' | 'task' | 'status';
}

export interface AgentCoordinationDashboardProps {
  /** Available agents */
  agents: Agent[];
  /** Pending tasks */
  tasks: AgentTask[];
  /** Recent messages */
  messages: AgentMessage[];
  /** WebSocket URL for real-time updates */
  wsUrl?: string;
  /** Callback when task is assigned */
  onTaskAssign$?: QRL<(taskId: string, agentId: string) => void>;
  /** Callback when message is sent */
  onMessageSend$?: QRL<(message: Omit<AgentMessage, 'id' | 'timestamp'>) => void>;
}

// ============================================================================
// Helpers
// ============================================================================

function getStatusColor(status: AgentStatus): string {
  switch (status) {
    case 'active': return 'bg-md-success/10 text-md-success border-md-success/20';
    case 'idle': return 'bg-md-primary/10 text-md-primary border-md-primary/20';
    case 'busy': return 'bg-md-warning/10 text-md-warning border-md-warning/20';
    case 'offline': return 'bg-md-surface-variant text-md-on-surface-variant border-md-outline/20';
    case 'error': return 'bg-md-error/10 text-md-error border-md-error/20';
    default: return 'bg-md-surface-variant border-md-outline/20';
  }
}

function getStatusDot(status: AgentStatus): string {
  switch (status) {
    case 'active': return 'bg-md-success';
    case 'idle': return 'bg-md-primary';
    case 'busy': return 'bg-md-warning animate-pulse';
    case 'offline': return 'bg-md-outline';
    case 'error': return 'bg-md-error animate-pulse';
    default: return 'bg-md-outline';
  }
}

function getClassBadge(cls: AgentClass): string {
  switch (cls) {
    case 'sagent': return 'bg-md-tertiary/10 text-md-tertiary';
    case 'swagent': return 'bg-md-secondary/10 text-md-secondary';
    case 'cagent': return 'bg-md-primary/10 text-md-primary';
    default: return 'bg-md-surface-variant text-md-on-surface-variant';
  }
}

function getPriorityColor(priority: string): string {
  switch (priority) {
    case 'critical': return 'bg-md-error/10 text-md-error';
    case 'high': return 'bg-md-error/5 text-md-error/80';
    case 'medium': return 'bg-md-warning/10 text-md-warning';
    case 'low': return 'bg-md-primary/10 text-md-primary';
    default: return 'bg-md-surface-variant';
  }
}

function formatTime(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  } catch { return dateStr; }
}

function formatTimeSince(dateStr: string): string {
  try {
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSecs = Math.floor(diffMs / 1000);
    const diffMins = Math.floor(diffSecs / 60);
    if (diffSecs < 60) return `${diffSecs}s ago`;
    if (diffMins < 60) return `${diffMins}m ago`;
    return `${Math.floor(diffMins / 60)}h ago`;
  } catch { return dateStr; }
}

// ============================================================================
// Component
// ============================================================================

export const AgentCoordinationDashboard = component$<AgentCoordinationDashboardProps>(({
  agents: initialAgents,
  tasks: initialTasks,
  messages: initialMessages,
  wsUrl = 'wss://kroma.live/ws/bus',
  onTaskAssign$,
  onMessageSend$,
}) => {
  const agents = useSignal<Agent[]>(initialAgents);
  const tasks = useSignal<AgentTask[]>(initialTasks);
  const messages = useSignal<AgentMessage[]>(initialMessages);
  const selectedAgentId = useSignal<string | null>(null);
  const selectedTaskId = useSignal<string | null>(null);
  const activeTab = useSignal<'agents' | 'tasks' | 'messages'>('agents');
  const newMessage = useSignal('');

  const wsState = useStore<{
    connected: boolean;
    ws: NoSerialize<WebSocket> | null;
  }>({ connected: false, ws: null });

  const selectedAgent = useComputed$(() => agents.value.find(a => a.id === selectedAgentId.value));
  const pendingTasks = useComputed$(() => tasks.value.filter(t => t.status === 'pending' || t.status === 'assigned'));
  const availableAgents = useComputed$(() => agents.value.filter(a => a.status === 'active' || a.status === 'idle'));

  const stats = useComputed$(() => ({
    totalAgents: agents.value.length,
    activeAgents: agents.value.filter(a => a.status === 'active' || a.status === 'busy').length,
    pendingTasks: tasks.value.filter(t => t.status === 'pending').length,
    avgWorkload: agents.value.length > 0
      ? Math.round(agents.value.reduce((sum, a) => sum + a.workload, 0) / agents.value.length)
      : 0,
  }));

  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;
    try {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => { wsState.connected = true; ws.send(JSON.stringify({ type: 'subscribe', topic: 'agent.*' })); };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.topic?.startsWith('agent.')) {
            if (msg.data?.agentId && msg.data?.status) {
              agents.value = agents.value.map(a =>
                a.id === msg.data.agentId ? { ...a, ...msg.data, lastHeartbeat: new Date().toISOString() } : a
              );
            }
          }
        } catch (e) { console.warn('[AgentDashboard] Parse error:', e); }
      };
      ws.onclose = () => { wsState.connected = false; };
      wsState.ws = noSerialize(ws);
      cleanup(() => { ws.close(); });
    } catch (e) { console.error('[AgentDashboard] WebSocket error:', e); }
  });

  const assignTask = $(async (taskId: string, agentId: string) => {
    tasks.value = tasks.value.map(t => t.id === taskId ? { ...t, assignedTo: agentId, status: 'assigned' } : t);
    if (onTaskAssign$) await onTaskAssign$(taskId, agentId);
    selectedTaskId.value = null;
  });

  const sendMessage = $(async () => {
    if (!newMessage.value.trim()) return;
    const message: Omit<AgentMessage, 'id' | 'timestamp'> = {
      from: 'operator', to: selectedAgentId.value || 'broadcast', content: newMessage.value, type: 'info',
    };
    if (onMessageSend$) await onMessageSend$(message);
    messages.value = [{ ...message, id: `msg_${Date.now()}`, timestamp: new Date().toISOString() }, ...messages.value];
    newMessage.value = '';
  });

  return (
    <Card variant="outlined" padding="p-0" class="overflow-hidden bg-md-surface h-full flex flex-col">
      {/* Header */}
      <div class="p-4 border-b border-md-outline/10 bg-md-surface-container-low flex items-center justify-between">
        <div class="flex items-center gap-3">
          <span class="text-[11px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/60">Swarm Orchestration</span>
          <div class={`w-2.5 h-2.5 rounded-full ${wsState.connected ? 'bg-md-success shadow-[0_0_8px_var(--md-sys-color-success)]' : 'bg-md-error'}`} />
        </div>
        <div class="flex items-center gap-4 text-[10px] font-bold text-md-on-surface-variant/40 uppercase tracking-tight">
          <span>{stats.value.activeAgents}/{stats.value.totalAgents} Active</span>
          <div class="w-1 h-1 rounded-full bg-md-outline/20" />
          <span>{stats.value.pendingTasks} Pending Tasks</span>
          <div class="w-1 h-1 rounded-full bg-md-outline/20" />
          <span class="text-md-primary">{stats.value.avgWorkload}% Swarm Load</span>
        </div>
      </div>

      {/* Tabs */}
      <div class="flex bg-md-surface-container-lowest border-b border-md-outline/10">
        {(['agents', 'tasks', 'messages'] as const).map(tab => (
          <Button
            key={tab}
            variant={activeTab.value === tab ? 'tonal' : 'text'}
            onClick$={() => { activeTab.value = tab; }}
            class={`flex-1 rounded-none h-12 text-[11px] font-black uppercase tracking-widest ${activeTab.value === tab ? 'border-b-4 border-b-md-primary' : ''}`}
          >
            {tab}
            {tab === 'tasks' && pendingTasks.value.length > 0 && (
              <span class="ml-2 px-2 py-0.5 rounded-full bg-md-error text-md-on-error text-[9px]">{pendingTasks.value.length}</span>
            )}
          </Button>
        ))}
      </div>

      {/* Content Area */}
      <div class="flex-1 overflow-y-auto no-scrollbar bg-md-surface-container-lowest">
        {activeTab.value === 'agents' && (
          <div class="p-4 grid grid-cols-1 gap-3">
            {agents.value.map(agent => (
              <Card
                key={agent.id}
                variant={selectedAgentId.value === agent.id ? 'elevated' : 'outlined'}
                padding="p-4"
                interactive
                onClick$={() => { selectedAgentId.value = agent.id; }}
                class={`transition-all duration-300 ${selectedAgentId.value === agent.id ? 'ring-2 ring-md-primary bg-md-primary/5' : 'hover:bg-md-surface-container-low'}`}
              >
                <div class="flex items-start justify-between">
                  <div class="flex items-center gap-3">
                    <div class={`w-3 h-3 rounded-full ${getStatusDot(agent.status)} shadow-sm`} />
                    <div>
                      <div class="text-sm font-black text-md-on-surface uppercase tracking-tight">{agent.name}</div>
                      <span class={`text-[9px] font-bold px-2 py-0.5 rounded-full uppercase tracking-tighter ${getClassBadge(agent.class)}`}>
                        {agent.class}
                      </span>
                    </div>
                  </div>
                  <span class={`text-[9px] font-black px-3 py-1 rounded-full border uppercase tracking-widest ${getStatusColor(agent.status)}`}>
                    {agent.status}
                  </span>
                </div>

                <div class="mt-4 space-y-3">
                  <div class="space-y-1">
                    <div class="flex items-center justify-between text-[10px] font-bold uppercase tracking-tighter text-md-on-surface-variant/60">
                      <span>Workload Distribution</span>
                      <span>{agent.workload}%</span>
                    </div>
                    <div class="h-1.5 rounded-full bg-md-surface-container-highest overflow-hidden">
                      <div
                        class={`h-full rounded-full transition-all duration-500 shadow-sm ${
                          agent.workload >= 80 ? 'bg-md-error' : agent.workload >= 60 ? 'bg-md-warning' : 'bg-md-success'
                        }`}
                        style={{ width: `${agent.workload}%` }}
                      />
                    </div>
                  </div>

                  {agent.currentTask && (
                    <div class="bg-md-surface-container p-2 rounded-xl border border-md-outline/5">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase mb-1">Active Cycle</div>
                      <div class="text-[11px] font-medium text-md-on-surface truncate">{agent.currentTask}</div>
                    </div>
                  )}
                </div>

                {selectedAgentId.value === agent.id && (
                  <div class="mt-4 pt-4 border-t border-md-outline/10 grid grid-cols-3 gap-4 animate-in fade-in slide-in-from-top-2 duration-300">
                    <div class="text-center">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase mb-1">Done</div>
                      <div class="text-sm font-black text-md-on-surface">{agent.metrics.tasksCompleted}</div>
                    </div>
                    <div class="text-center">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase mb-1">Avg T</div>
                      <div class="text-sm font-black text-md-on-surface">{agent.metrics.avgCompletionTime}m</div>
                    </div>
                    <div class="text-center">
                      <div class="text-[9px] font-bold text-md-on-surface-variant/40 uppercase mb-1">Err %</div>
                      <div class={`text-sm font-black ${agent.metrics.errorRate > 5 ? 'text-md-error' : 'text-md-success'}`}>{agent.metrics.errorRate}%</div>
                    </div>
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}

        {activeTab.value === 'tasks' && (
          <div class="p-4 space-y-3">
            {tasks.value.map(task => (
              <Card
                key={task.id}
                variant="outlined"
                padding="p-4"
                class={`border-l-4 ${
                  task.status === 'completed' ? 'border-l-md-success bg-md-success/5' :
                  task.status === 'failed' ? 'border-l-md-error bg-md-error/5' :
                  'border-l-md-outline/20'
                }`}
              >
                <div class="flex items-start justify-between gap-4">
                  <div class="flex-1 min-w-0">
                    <div class="flex items-center gap-3 mb-1">
                      <h4 class="text-sm font-black text-md-on-surface uppercase tracking-tight truncate">{task.title}</h4>
                      <span class={`text-[9px] font-black px-2 py-0.5 rounded-full uppercase border ${getPriorityColor(task.priority)}`}>
                        {task.priority}
                      </span>
                    </div>
                    {task.description && <div class="text-xs text-md-on-surface-variant/80 line-clamp-2">{task.description}</div>}
                  </div>
                  <span class={`text-[9px] font-black px-3 py-1 rounded-full border uppercase tracking-widest bg-md-surface-container-high text-md-on-surface-variant`}>
                    {task.status.replace('_', ' ')}
                  </span>
                </div>

                <div class="mt-4 flex items-center justify-between text-[10px] font-bold text-md-on-surface-variant/40 uppercase tracking-tighter">
                  <div class="flex items-center gap-2">
                    {task.assignedTo ? (
                      <span class="flex items-center gap-1">Owner: <span class="text-md-primary font-black">@{task.assignedTo}</span></span>
                    ) : (
                      <span class="text-md-warning animate-pulse">Unassigned</span>
                    )}
                  </div>
                  <span>Created {formatTime(task.createdAt)}</span>
                </div>

                {task.status === 'pending' && availableAgents.value.length > 0 && (
                  <div class="mt-4 pt-4 border-t border-md-outline/10 flex items-center gap-3">
                    <select
                      class="flex-grow h-10 px-4 rounded-xl bg-md-surface-container-high border border-md-outline/20 text-xs font-black uppercase tracking-tight text-md-on-surface"
                      onChange$={(e) => {
                        const agentId = (e.target as HTMLSelectElement).value;
                        if (agentId) assignTask(task.id, agentId);
                      }}
                    >
                      <option value="">Select Target Agent</option>
                      {availableAgents.value.map(agent => (
                        <option key={agent.id} value={agent.id}>{`${agent.name} (${agent.workload}% LOAD)`}</option>
                      ))}
                    </select>
                    <Button variant="tonal" class="h-10 px-6 rounded-xl">Assign</Button>
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}

        {activeTab.value === 'messages' && (
          <div class="flex flex-col h-full">
            <div class="flex-1 overflow-y-auto p-4 space-y-3 no-scrollbar max-h-[300px]">
              {messages.value.map(msg => (
                <div
                  key={msg.id}
                  class={`p-3 rounded-2xl border ${
                    msg.type === 'error' ? 'bg-md-error/10 border-md-error/20 text-md-error' :
                    msg.type === 'warning' ? 'bg-md-warning/10 border-md-warning/20 text-md-warning' :
                    msg.type === 'task' ? 'bg-md-primary/10 border-md-primary/20 text-md-primary' :
                    'bg-md-surface-container border-md-outline/5 text-md-on-surface'
                  }`}
                >
                  <div class="flex items-center justify-between mb-1 text-[10px] font-black uppercase tracking-tighter opacity-60">
                    <div class="flex items-center gap-2">
                      <span>@{msg.from}</span>
                      <span>→</span>
                      <span class={msg.to === 'broadcast' ? 'text-md-tertiary' : ''}>{msg.to}</span>
                    </div>
                    <span>{formatTime(msg.timestamp)}</span>
                  </div>
                  <div class="text-[11px] font-medium leading-relaxed">{msg.content}</div>
                </div>
              ))}
            </div>

            <div class="p-4 bg-md-surface-container-low border-t border-md-outline/10 space-y-3">
              <div class="flex gap-2">
                <select
                  value={selectedAgentId.value || 'broadcast'}
                  onChange$={(e) => {
                    const val = (e.target as HTMLSelectElement).value;
                    selectedAgentId.value = val === 'broadcast' ? null : val;
                  }}
                  class="h-10 px-3 rounded-xl bg-md-surface border border-md-outline/20 text-[10px] font-black uppercase"
                >
                  <option value="broadcast">All Agents</option>
                  {agents.value.map(a => (
                    <option key={a.id} value={a.id}>{`@${a.name}`}</option>
                  ))}
                </select>
                <div class="flex-1">
                  <Input
                    placeholder="Broadcast swarm directive..."
                    value={newMessage.value}
                    onInput$={(_, el) => { newMessage.value = el.value; }}
                    onKeyDown$={(e) => { if (e.key === 'Enter') sendMessage(); }}
                    class="h-10"
                  />
                </div>
                <Button variant="primary" icon="send" class="h-10 w-10 min-w-0" onClick$={sendMessage} />
              </div>
            </div>
          </div>
        )}
      </div>

      <div class="p-3 border-t border-md-outline/10 bg-md-surface-container-low text-[9px] font-black uppercase tracking-[0.2em] text-md-on-surface-variant/30 text-center">
        {agents.value.length} Nodes • {tasks.value.length} Directives • {messages.value.length} Signals
      </div>
    </Card>
  );
});

export default AgentCoordinationDashboard;