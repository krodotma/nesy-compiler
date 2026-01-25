/**
 * DKIN/PAIP Flow Monitor — Protocol Compliance & Code Flow Visualization
 *
 * A unified view synthesizing all DKIN protocol versions (v1-v19) to monitor:
 * - Bus event flow and health
 * - Agent task lifecycle and handoff
 * - Code evolution (rhizome) states
 * - Protocol compliance with remediation guidance
 *
 * This component serves as the central nervous system for observing
 * and correcting agent/code behavior across the Pluribus ecosystem.
 */

import { component$, useSignal, useComputed$, useVisibleTask$, type QRL } from '@builder.io/qwik';

// M3 Components - DKINFlowMonitor
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/chips/filter-chip.js';
import '@material/web/progress/linear-progress.js';
import '@material/web/tabs/tabs.js';
import '@material/web/tabs/primary-tab.js';

import type {
  DKINObservatoryState,
  DKINEvent,
  EvoLoopPhase,
  AgentTask,
  ProtocolCompliance,
  CMPMetrics,
  PBLOCKState,
  BusHealth,
  PAIPClone,
  TaskState,
} from './types';
import {
  DKIN_PROTOCOL_VERSIONS,
  EVO_PHASE_COLORS,
  TASK_STATE_COLORS,
  SPECIES_COLORS,
  DEFAULT_DKIN_STATE,
} from './types';
import { ReconciliationPanel } from './ReconciliationPanel';

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' ? (value as Record<string, unknown>) : {};

// ============================================================================
// Sparkline Component
// ============================================================================

const Sparkline = component$<{ values: number[]; width?: number; height?: number; color?: string }>(
  ({ values, width = 120, height = 24, color = '#3b82f6' }) => {
    const path = useComputed$(() => {
      if (!values || values.length === 0) return '';
      const max = Math.max(...values, 1);
      const min = Math.min(...values, 0);
      const range = max - min || 1;
      const step = width / (values.length - 1 || 1);
      return values
        .map((v, i) => {
          const x = i * step;
          const y = height - ((v - min) / range) * height;
          return `${i === 0 ? 'M' : 'L'} ${x} ${y}`;
        })
        .join(' ');
    });

    return (
      <svg width={width} height={height} class="inline-block">
        <path d={path.value} fill="none" stroke={color} stroke-width="1.5" />
      </svg>
    );
  }
);

// ============================================================================
// Protocol Version Badge
// ============================================================================

const ProtocolBadge = component$<{ version: string; active?: boolean }>(({ version, active }) => (
  <span
    class={[
      'px-2 py-0.5 rounded text-xs font-mono',
      active ? 'bg-emerald-900/50 text-emerald-300 border border-emerald-600' : 'bg-zinc-800 text-zinc-400',
    ].join(' ')}
  >
    {version}
  </span>
));

// ============================================================================
// Evo Phase Indicator
// ============================================================================

const EvoPhaseIndicator = component$<{ phase: EvoLoopPhase }>(({ phase }) => {
  const phases: EvoLoopPhase[] = ['percolate', 'assimilate', 'mutate', 'test', 'promote'];
  const phaseLabels: Record<EvoLoopPhase, string> = {
    percolate: 'Ingest',
    assimilate: 'Map',
    mutate: 'Generate',
    test: 'Verify',
    promote: 'Transfer',
    idle: 'Idle',
  };

  return (
    <div class="flex items-center gap-1">
      {phases.map((p) => (
        <div
          key={p}
          class={[
            'px-2 py-1 text-xs rounded transition-all',
            p === phase
              ? 'bg-blue-600 text-white font-semibold scale-105'
              : 'bg-zinc-800 text-zinc-500',
          ].join(' ')}
          title={phaseLabels[p]}
        >
          {phaseLabels[p]}
        </div>
      ))}
    </div>
  );
});

// ============================================================================
// CMP Gauge
// ============================================================================

const CMPGauge = component$<{ metrics: CMPMetrics }>(({ metrics }) => {
  const score = metrics.score || 0;
  const percentage = Math.min(100, Math.max(0, score * 100));
  const color = score > 0.7 ? '#10b981' : score > 0.4 ? '#f59e0b' : '#ef4444';

  return (
    <div class="flex flex-col gap-1">
      <div class="flex justify-between text-xs text-zinc-400">
        <span>CMP Score</span>
        <span style={{ color }}>{(score * 100).toFixed(1)}%</span>
      </div>
      <div class="h-2 bg-zinc-800 rounded-full overflow-hidden">
        <div
          class="h-full rounded-full transition-all duration-500"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
      <div class="grid grid-cols-4 gap-1 text-xs text-zinc-500 mt-1">
        <div>U:{(metrics.utility * 100).toFixed(0)}%</div>
        <div>R:{(metrics.robustness * 100).toFixed(0)}%</div>
        <div>C:{metrics.complexity.toFixed(1)}</div>
        <div>$:{metrics.cost.toFixed(2)}</div>
      </div>
    </div>
  );
});

// ============================================================================
// PBLOCK Status Panel
// ============================================================================

const PBLOCKPanel = component$<{ state: PBLOCKState }>(({ state }) => (
  <div
    class={[
      'p-3 rounded border',
      state.active
        ? 'bg-amber-950/30 border-amber-600'
        : 'bg-zinc-900/50 border-zinc-700',
    ].join(' ')}
  >
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-semibold">
        {state.active ? '🔒 PBLOCK ACTIVE' : '🔓 Normal Development'}
      </span>
      {state.milestone && <span class="text-xs text-zinc-400">{state.milestone}</span>}
    </div>
    {state.active && (
      <div class="grid grid-cols-3 gap-2 text-xs">
        <div class={state.exitCriteria.allTestsPass ? 'text-emerald-400' : 'text-zinc-500'}>
          {state.exitCriteria.allTestsPass ? '✓' : '○'} Tests Pass
        </div>
        <div class={state.exitCriteria.pushedToRemotes ? 'text-emerald-400' : 'text-zinc-500'}>
          {state.exitCriteria.pushedToRemotes ? '✓' : '○'} Pushed
        </div>
        <div class={state.exitCriteria.pushedToGithub ? 'text-emerald-400' : 'text-zinc-500'}>
          {state.exitCriteria.pushedToGithub ? '✓' : '○'} GitHub
        </div>
      </div>
    )}
    {state.violations > 0 && (
      <div class="mt-2 text-xs text-red-400">
        ⚠ {state.violations} guard violation{state.violations > 1 ? 's' : ''}
      </div>
    )}
  </div>
));

// ============================================================================
// Bus Health Panel
// ============================================================================

const BusHealthPanel = component$<{ health: BusHealth }>(({ health }) => {
  const status = health.needsRotation ? 'warning' : health.sizeMb > 50 ? 'caution' : 'healthy';
  const colors = {
    healthy: 'border-emerald-600 bg-emerald-950/20',
    caution: 'border-yellow-600 bg-yellow-950/20',
    warning: 'border-red-600 bg-red-950/20',
  };

  return (
    <div class={`p-3 rounded border ${colors[status]}`}>
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-semibold">Bus Health</span>
        <span class="text-xs text-zinc-400">{health.eventCount.toLocaleString()} events</span>
      </div>
      <div class="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span class="text-zinc-500">Size: </span>
          <span class={health.sizeMb > 100 ? 'text-red-400' : 'text-zinc-300'}>
            {health.sizeMb.toFixed(1)} MB
          </span>
        </div>
        <div>
          <span class="text-zinc-500">Velocity: </span>
          <span class="text-zinc-300">{health.velocity.toFixed(0)}/hr</span>
        </div>
        <div>
          <span class="text-zinc-500">Age: </span>
          <span class={health.oldestEventAge > 168 ? 'text-amber-400' : 'text-zinc-300'}>
            {health.oldestEventAge.toFixed(0)}h
          </span>
        </div>
        <div>
          {health.needsRotation && <span class="text-amber-400">⟳ Rotation needed</span>}
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// PAIP Clones Panel
// ============================================================================

const PAIPClonesPanel = component$<{ clones: PAIPClone[] }>(({ clones }) => {
  const activeClones = clones.filter((c) => !c.isOrphan && !c.isStale);
  const problemClones = clones.filter((c) => c.isOrphan || c.isStale);

  return (
    <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-semibold">PAIP Isolation</span>
        <span class="text-xs text-zinc-400">{clones.length} clone{clones.length !== 1 ? 's' : ''}</span>
      </div>
      {activeClones.length > 0 && (
        <div class="mb-2">
          <div class="text-xs text-zinc-500 mb-1">Active:</div>
          {activeClones.slice(0, 3).map((c) => (
            <div key={c.cloneDir} class="text-xs text-emerald-400 truncate">
              {c.agentId} → {c.branch}
            </div>
          ))}
        </div>
      )}
      {problemClones.length > 0 && (
        <div>
          <div class="text-xs text-amber-500 mb-1">Cleanup needed:</div>
          {problemClones.slice(0, 2).map((c) => (
            <div key={c.cloneDir} class="text-xs text-amber-400 truncate">
              {c.isOrphan ? '👻' : '⏰'} {c.agentId}
            </div>
          ))}
        </div>
      )}
      {clones.length === 0 && <div class="text-xs text-zinc-500">No active clones</div>}
    </div>
  );
});

// ============================================================================
// Compliance Panel
// ============================================================================

const CompliancePanel = component$<{ compliance: ProtocolCompliance }>(({ compliance }) => (
  <div
    class={[
      'p-3 rounded border',
      compliance.compliant
        ? 'border-emerald-600 bg-emerald-950/20'
        : 'border-red-600 bg-red-950/20',
    ].join(' ')}
  >
    <div class="flex items-center justify-between mb-2">
      <span class="text-sm font-semibold">
        {compliance.compliant ? '✓ Compliant' : '✗ Non-Compliant'}
      </span>
      <ProtocolBadge version={compliance.version} active />
    </div>
    {compliance.violations.length > 0 && (
      <div class="mb-2">
        <div class="text-xs text-red-400 mb-1">Violations:</div>
        {compliance.violations.slice(0, 3).map((v, i) => (
          <div key={i} class="text-xs text-red-300 truncate">
            • {v}
          </div>
        ))}
      </div>
    )}
    {compliance.recommendations.length > 0 && (
      <div>
        <div class="text-xs text-amber-400 mb-1">Remediation:</div>
        {compliance.recommendations.slice(0, 3).map((r, i) => (
          <div key={i} class="text-xs text-amber-300 truncate">
            → {r}
          </div>
        ))}
      </div>
    )}
  </div>
));

// ============================================================================
// Agent Task Flow
// ============================================================================

const AgentTaskFlow = component$<{ tasks: AgentTask[] }>(({ tasks }) => {
  const running = tasks.filter((t) => t.state === 'RUNNING');
  const recent = tasks.slice(0, 8);

  return (
    <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm font-semibold">Task Flow (v18)</span>
        <span class="text-xs text-zinc-400">{running.length} active</span>
      </div>
      <div class="space-y-1">
        {recent.map((t) => (
          <div key={t.taskId} class="flex items-center gap-2 text-xs">
            <span
              class="w-2 h-2 rounded-full"
              style={{ backgroundColor: TASK_STATE_COLORS[t.state] || '#6b7280' }}
            />
            <span class="text-zinc-400 w-12 truncate">{t.species}</span>
            <div class="flex-1 bg-zinc-800 rounded-full h-1.5 overflow-hidden">
              <div
                class="h-full bg-blue-500 transition-all"
                style={{ width: `${t.progress * 100}%` }}
              />
            </div>
            <span class="text-zinc-500 w-8">{(t.progress * 100).toFixed(0)}%</span>
          </div>
        ))}
        {tasks.length === 0 && <div class="text-xs text-zinc-500">No active tasks</div>}
      </div>
    </div>
  );
});

// ============================================================================
// Event Stream
// ============================================================================

const EventStream = component$<{ events: DKINEvent[] }>(({ events }) => {
  const topicColor = (topic: string): string => {
    if (topic.startsWith('operator.pblock')) return '#f59e0b';
    if (topic.startsWith('operator.pbhygiene')) return '#10b981';
    if (topic.startsWith('paip.')) return '#8b5cf6';
    if (topic.startsWith('evolution.') || topic.startsWith('hgt.')) return '#3b82f6';
    if (topic.startsWith('ckin.') || topic.startsWith('oiterate.')) return '#06b6d4';
    return '#6b7280';
  };

  return (
    <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50 max-h-64 overflow-y-auto">
      <div class="text-sm font-semibold mb-2">DKIN Event Stream</div>
      <div class="space-y-1">
        {events.slice(0, 20).map((e) => (
          <div key={e.id} class="flex items-start gap-2 text-xs">
            <span class="text-zinc-600 w-14 shrink-0">{new Date(e.ts * 1000).toLocaleTimeString()}</span>
            <span class="w-2 h-2 rounded-full mt-1 shrink-0" style={{ backgroundColor: topicColor(e.topic) }} />
            <span class="text-zinc-400 truncate">{e.topic}</span>
            <span class="text-zinc-600 shrink-0">{e.actor}</span>
          </div>
        ))}
        {events.length === 0 && <div class="text-xs text-zinc-500">No recent DKIN events</div>}
      </div>
    </div>
  );
});

// ============================================================================
// Protocol Timeline
// ============================================================================

const ProtocolTimeline = component$<{ currentVersion: string }>(({ currentVersion }) => {
  const versions = ['v17', 'v18', 'v19', 'v20', 'v21'];
  const versionNames: Record<string, string> = {
    v17: 'Hygiene',
    v18: 'Resilience',
    v19: 'Evolution',
    v20: 'Embodied',
    v21: 'Lossless',
  };

  return (
    <div class="flex items-center gap-1">
      {versions.map((v, i) => (
        <div key={v} class="flex items-center">
          <div
            class={[
              'px-2 py-1 text-xs rounded',
              v === currentVersion
                ? 'bg-blue-600 text-white font-semibold'
                : versions.indexOf(currentVersion) >= i
                ? 'bg-zinc-700 text-zinc-300'
                : 'bg-zinc-800 text-zinc-500',
            ].join(' ')}
            title={versionNames[v]}
          >
            {v}
          </div>
          {i < versions.length - 1 && <span class="text-zinc-600 mx-0.5">→</span>}
        </div>
      ))}
    </div>
  );
});

// ============================================================================
// Main DKIN Flow Monitor Component
// ============================================================================

export interface DKINFlowMonitorProps {
  events: DKINEvent[];
  emitBus$?: QRL<(topic: string, kind: string, data: Record<string, unknown>) => Promise<void>>;
}

export const DKINFlowMonitor = component$<DKINFlowMonitorProps>(({ events, emitBus$ }) => {
  const state = useSignal<DKINObservatoryState>(DEFAULT_DKIN_STATE);
  const eventHistory = useSignal<number[]>([]);

  // Parse events to update state
  useVisibleTask$(({ track }) => {
    track(() => events);

    // Filter DKIN-related events
    const dkinEvents = events.filter(
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
        e.topic.startsWith('infer_sync.') ||
        e.topic.startsWith('reconcile.')
    );

    // Update event history for sparkline
    const now = Date.now();
    const buckets = new Array(24).fill(0);
    const bucketMs = (15 * 60 * 1000) / 24;
    dkinEvents.forEach((e) => {
      const age = now - e.ts * 1000;
      const bucket = Math.floor(age / bucketMs);
      if (bucket >= 0 && bucket < 24) {
        buckets[23 - bucket]++;
      }
    });
    eventHistory.value = buckets;

    // Check PBLOCK state
    const pblockEvents = dkinEvents.filter((e) => e.topic.startsWith('operator.pblock'));
    if (pblockEvents.length > 0) {
      const latest = asRecord(pblockEvents[0].data);
      state.value = {
        ...state.value,
        pblock: {
          active: !!latest.active,
          milestone: latest.milestone as string,
          enteredBy: latest.entered_by as string,
          exitCriteria: {
            allTestsPass: !!latest.all_tests_pass,
            pushedToRemotes: !!latest.pushed_to_remotes,
            pushedToGithub: !!latest.pushed_to_github,
          },
          violations: (latest.violations as number) || 0,
        },
      };
    }

    // Check for task lifecycle events (v18)
    const taskEvents = dkinEvents.filter((e) => e.topic.match(/^agent\.\w+\.task$/));
    const taskMap = new Map<string, AgentTask>();
    taskEvents.forEach((e) => {
      const data = asRecord(e.data);
      if (data.task_id) {
        taskMap.set(data.task_id as string, {
          taskId: data.task_id as string,
          parentId: data.parent_id as string,
          species: (data.species as AgentTask['species']) || 'unknown',
          agent: (data.agent as string) || e.actor,
          state: (data.status as TaskState) || 'PENDING',
          progress: (data.progress as number) || 0,
          context: data.context as Record<string, unknown>,
          startedAt: e.ts,
          updatedAt: e.ts,
        });
      }
    });
    state.value = { ...state.value, taskGraph: Array.from(taskMap.values()) };

    // Check bus health from hygiene events
    const hygieneEvents = dkinEvents.filter((e) => e.topic === 'operator.pbhygiene.audit');
    if (hygieneEvents.length > 0) {
      const latest = asRecord(hygieneEvents[0].data);
      state.value = {
        ...state.value,
        busHealth: {
          sizeMb: (latest.size_mb as number) || 0,
          eventCount: (latest.event_count as number) || 0,
          oldestEventAge: (latest.oldest_age_hours as number) || 0,
          velocity: (latest.velocity as number) || 0,
          needsRotation: !!latest.needs_rotation,
          lastRotation: latest.last_rotation as number,
        },
      };
    }

    // Check PAIP clones
    const paipEvents = dkinEvents.filter((e) => e.topic.startsWith('paip.'));
    const cloneMap = new Map<string, PAIPClone>();
    paipEvents.forEach((e) => {
      const data = asRecord(e.data);
      if (data.clone_dir) {
        const isOrphan = e.topic === 'paip.orphan.detected';
        const isDeleted = e.topic === 'paip.clone.deleted';
        if (!isDeleted) {
          cloneMap.set(data.clone_dir as string, {
            cloneDir: data.clone_dir as string,
            agentId: (data.agent_id as string) || 'unknown',
            branch: (data.branch as string) || '',
            createdAt: e.ts,
            uncommitted: (data.uncommitted as number) || 0,
            isOrphan,
            isStale: !!data.stale,
          });
        } else {
          cloneMap.delete(data.clone_dir as string);
        }
      }
    });
    state.value = { ...state.value, paipClones: Array.from(cloneMap.values()) };

    // Compute compliance
    const violations: string[] = [];
    const recommendations: string[] = [];

    if (state.value.busHealth.needsRotation) {
      violations.push('Bus exceeds size/age threshold');
      recommendations.push('Run: pbhygiene --rotate-bus');
    }
    if (state.value.paipClones.some((c) => c.isOrphan)) {
      violations.push('Orphan PAIP clones detected');
      recommendations.push('Clean orphan clones before PBLOCK');
    }
    if (state.value.pblock.active && state.value.pblock.violations > 0) {
      violations.push(`${state.value.pblock.violations} PBLOCK guard violations`);
      recommendations.push('Use fix:/test:/refactor: commit prefixes only');
    }

    state.value = {
      ...state.value,
      protocolVersion: 'v21',
      compliance: {
        version: 'v21',
        compliant: violations.length === 0,
        violations,
        recommendations,
        lastChecked: Date.now(),
      },
      recentEvents: dkinEvents.slice(0, 50),
      lastUpdated: Date.now(),
    };
  });

  return (
    <div class="p-4 bg-zinc-950 text-zinc-100 min-h-full">
      {/* Header */}
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center gap-3">
          <h1 class="text-lg font-bold">DKIN/PAIP Flow Monitor</h1>
          <ProtocolTimeline currentVersion={state.value.protocolVersion} />
        </div>
        <div class="flex items-center gap-2">
          <Sparkline values={eventHistory.value} color="#3b82f6" />
          <span class="text-xs text-zinc-500">{state.value.recentEvents.length} events (15m)</span>
        </div>
      </div>

      {/* Main Grid */}
      <div class="grid grid-cols-12 gap-4">
        {/* Left Column: Status Panels */}
        <div class="col-span-4 space-y-4">
          <CompliancePanel compliance={state.value.compliance} />
          <ReconciliationPanel />
          <PBLOCKPanel state={state.value.pblock} />
          <BusHealthPanel health={state.value.busHealth} />
          <PAIPClonesPanel clones={state.value.paipClones} />
        </div>

        {/* Center Column: Evolution & Tasks */}
        <div class="col-span-4 space-y-4">
          <div class="p-3 rounded border border-zinc-700 bg-zinc-900/50">
            <div class="text-sm font-semibold mb-2">Evolutionary Cycle (v19)</div>
            <EvoPhaseIndicator phase={state.value.evoPhase} />
            <div class="mt-3">
              <CMPGauge metrics={state.value.cmpMetrics} />
            </div>
          </div>
          <AgentTaskFlow tasks={state.value.taskGraph} />
        </div>

        {/* Right Column: Event Stream */}
        <div class="col-span-4">
          <EventStream events={state.value.recentEvents} />
        </div>
      </div>

      {/* Footer: Protocol Versions */}
      <div class="mt-4 pt-4 border-t border-zinc-800">
        <div class="flex items-center gap-2 text-xs text-zinc-500 flex-wrap">
          <span>Protocol Coverage:</span>
          {DKIN_PROTOCOL_VERSIONS.slice(-6).map((p) => (
            <ProtocolBadge key={p.version} version={p.version} active={p.version === state.value.protocolVersion} />
          ))}
        </div>
      </div>
    </div>
  );
});

export default DKINFlowMonitor;