/**
 * IterationFlowTimeline - Co-Bio/Mecha-Neural Iteration Flow UI
 *
 * Visualizes the interplay between human (bio) and AI (mecha) iterations in
 * the Pluribus multi-agent system. Key concepts:
 *
 * - Bio Mode: Human-in-the-loop iterations (review, approval, feedback)
 * - Mecha Mode: Autonomous agentic iterations (generation, verification)
 * - Handoff Points: Transitions between bio and mecha control
 * - Convergence/Divergence: Progress toward or away from goal state
 *
 * Reference model: "15 iterations with 4 subagents" - a typical star topology
 * orchestration where a coordinator delegates to multiple specialized agents.
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  type Signal,
} from '@builder.io/qwik';
import type { BusEvent } from '../lib/state/types';

// ============================================================================
// Type Definitions
// ============================================================================

export type IterationMode = 'bio' | 'mecha' | 'hybrid';
export type IterationPhase = 'planning' | 'execution' | 'verification' | 'handoff';
export type ConvergenceState = 'converging' | 'diverging' | 'stable' | 'unknown';

export interface SubAgent {
  id: string;
  name: string;
  species: 'claude' | 'codex' | 'gemini' | 'qwen' | 'unknown';
  status: 'idle' | 'active' | 'waiting' | 'complete' | 'error';
  iterationsCompleted: number;
  currentTask?: string;
  lastActivity: number;
}

export interface Iteration {
  id: string;
  index: number;
  mode: IterationMode;
  phase: IterationPhase;
  agentId?: string;
  agentName?: string;
  description: string;
  startTime: number;
  endTime?: number;
  duration?: number;
  isHandoff: boolean;
  handoffFrom?: 'bio' | 'mecha';
  handoffTo?: 'bio' | 'mecha';
  convergenceScore?: number;
  parentIterationId?: string;
  childIterations?: string[];
  artifacts?: string[];
  tokens?: number;
}

export interface CycleProgress {
  totalIterations: number;
  completedIterations: number;
  currentIteration: number;
  bioIterations: number;
  mechaIterations: number;
  handoffCount: number;
  convergenceScore: number;
  convergenceState: ConvergenceState;
  estimatedRemaining: number;
}

export interface IterationFlowState {
  iterations: Iteration[];
  subAgents: SubAgent[];
  progress: CycleProgress;
  currentMode: IterationMode;
  lastHandoffTime: number;
  cycleStartTime: number;
  cycleGoal: string;
}

// ============================================================================
// Constants
// ============================================================================

const MODE_COLORS: Record<IterationMode, string> = {
  bio: '#10b981',    // emerald - organic life
  mecha: '#3b82f6',  // blue - machine
  hybrid: '#8b5cf6', // purple - synthesis
};

const PHASE_COLORS: Record<IterationPhase, string> = {
  planning: '#f59e0b',     // amber
  execution: '#3b82f6',    // blue
  verification: '#10b981', // emerald
  handoff: '#ec4899',      // pink
};

const SPECIES_COLORS: Record<string, string> = {
  claude: '#d97706',   // amber/orange
  codex: '#10b981',    // emerald
  gemini: '#3b82f6',   // blue
  qwen: '#8b5cf6',     // purple
  unknown: '#6b7280',  // gray
};

const CONVERGENCE_COLORS: Record<ConvergenceState, string> = {
  converging: '#10b981',  // green
  diverging: '#ef4444',   // red
  stable: '#6b7280',      // gray
  unknown: '#f59e0b',     // amber
};

const DEFAULT_STATE: IterationFlowState = {
  iterations: [],
  subAgents: [],
  progress: {
    totalIterations: 15,
    completedIterations: 0,
    currentIteration: 0,
    bioIterations: 0,
    mechaIterations: 0,
    handoffCount: 0,
    convergenceScore: 0,
    convergenceState: 'unknown',
    estimatedRemaining: 15,
  },
  currentMode: 'mecha',
  lastHandoffTime: 0,
  cycleStartTime: Date.now(),
  cycleGoal: 'Awaiting task...',
};

// ============================================================================
// Utility Functions
// ============================================================================

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function wipMeter(pct: number, width: number = 12): string {
  const clamped = Math.max(0, Math.min(100, pct));
  const filled = Math.round((clamped / 100) * width);
  const empty = width - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
}

function parseIterationEvents(events: BusEvent[]): Partial<IterationFlowState> {
  const iterations: Iteration[] = [];
  const subAgentsMap = new Map<string, SubAgent>();
  let currentMode: IterationMode = 'mecha';
  let lastHandoffTime = 0;
  let cycleGoal = 'Awaiting task...';
  let cycleStartTime = Date.now();

  // Filter relevant events
  const relevantEvents = events.filter(
    (e) =>
      e.topic.startsWith('iteration.') ||
      e.topic.startsWith('agent.') ||
      e.topic.startsWith('handoff.') ||
      e.topic.startsWith('codex.') ||
      e.topic.startsWith('coordinator.') ||
      e.topic.startsWith('cycle.')
  );

  for (const event of relevantEvents) {
    const data = (event.data as Record<string, unknown>) || {};

    // Parse cycle events
    if (event.topic === 'cycle.started') {
      cycleGoal = (data.goal as string) || cycleGoal;
      cycleStartTime = event.ts * 1000;
    }

    // Parse iteration events
    if (event.topic.startsWith('iteration.')) {
      const iteration: Iteration = {
        id: (data.iteration_id as string) || `iter-${Date.now()}`,
        index: iterations.length,
        mode: (data.mode as IterationMode) || 'mecha',
        phase: (data.phase as IterationPhase) || 'execution',
        agentId: (data.agent_id as string) || event.actor,
        agentName: (data.agent_name as string) || event.actor,
        description: (data.description as string) || event.topic,
        startTime: event.ts * 1000,
        endTime: data.end_time ? (data.end_time as number) * 1000 : undefined,
        isHandoff: event.topic === 'iteration.handoff',
        convergenceScore: data.convergence as number,
        tokens: data.tokens as number,
      };

      if (iteration.endTime && iteration.startTime) {
        iteration.duration = iteration.endTime - iteration.startTime;
      }

      iterations.push(iteration);
    }

    // Parse handoff events
    if (event.topic.startsWith('handoff.')) {
      currentMode = (data.to as IterationMode) || currentMode;
      lastHandoffTime = event.ts * 1000;

      iterations.push({
        id: `handoff-${Date.now()}`,
        index: iterations.length,
        mode: 'hybrid',
        phase: 'handoff',
        agentId: event.actor,
        description: `Handoff: ${data.from || 'mecha'} -> ${data.to || 'bio'}`,
        startTime: event.ts * 1000,
        isHandoff: true,
        handoffFrom: data.from as 'bio' | 'mecha',
        handoffTo: data.to as 'bio' | 'mecha',
      });
    }

    // Parse agent activity
    if (event.topic.match(/^agent\.\w+\.task$/)) {
      const agentId = (data.agent_id as string) || event.actor;
      const existing = subAgentsMap.get(agentId);
      const species = (data.species as SubAgent['species']) || 'unknown';

      subAgentsMap.set(agentId, {
        id: agentId,
        name: (data.agent_name as string) || agentId,
        species,
        status: (data.status as SubAgent['status']) || 'active',
        iterationsCompleted: (existing?.iterationsCompleted || 0) + 1,
        currentTask: data.task as string,
        lastActivity: event.ts * 1000,
      });
    }

    // Parse codex events (implementation agent)
    if (event.topic.startsWith('codex.')) {
      const agentId = event.actor || 'codex';
      const existing = subAgentsMap.get(agentId);

      subAgentsMap.set(agentId, {
        id: agentId,
        name: agentId,
        species: 'codex',
        status: event.topic.includes('complete') ? 'complete' : 'active',
        iterationsCompleted: (existing?.iterationsCompleted || 0) + 1,
        currentTask: (data.goal as string) || 'implementation',
        lastActivity: event.ts * 1000,
      });
    }
  }

  // Calculate progress
  const totalIterations = 15; // Reference model
  const completedIterations = iterations.filter((i) => i.endTime).length;
  const bioIterations = iterations.filter((i) => i.mode === 'bio').length;
  const mechaIterations = iterations.filter((i) => i.mode === 'mecha').length;
  const handoffCount = iterations.filter((i) => i.isHandoff).length;

  // Calculate convergence
  const recentIterations = iterations.slice(-5);
  const convergenceScores = recentIterations
    .map((i) => i.convergenceScore)
    .filter((s): s is number => typeof s === 'number');
  const avgConvergence =
    convergenceScores.length > 0
      ? convergenceScores.reduce((a, b) => a + b, 0) / convergenceScores.length
      : 0;

  let convergenceState: ConvergenceState = 'unknown';
  if (convergenceScores.length >= 2) {
    const trend = convergenceScores[convergenceScores.length - 1] - convergenceScores[0];
    if (trend > 0.05) convergenceState = 'converging';
    else if (trend < -0.05) convergenceState = 'diverging';
    else convergenceState = 'stable';
  }

  return {
    iterations,
    subAgents: Array.from(subAgentsMap.values()),
    currentMode,
    lastHandoffTime,
    cycleStartTime,
    cycleGoal,
    progress: {
      totalIterations,
      completedIterations,
      currentIteration: iterations.length,
      bioIterations,
      mechaIterations,
      handoffCount,
      convergenceScore: avgConvergence,
      convergenceState,
      estimatedRemaining: Math.max(0, totalIterations - iterations.length),
    },
  };
}

// ============================================================================
// Subcomponents
// ============================================================================

/** Mode indicator showing bio/mecha state */
const ModeIndicator = component$<{ mode: IterationMode; size?: 'sm' | 'md' | 'lg' }>(
  ({ mode, size = 'md' }) => {
    const sizeClasses = {
      sm: 'text-[10px] px-1.5 py-0.5',
      md: 'text-xs px-2 py-1',
      lg: 'text-sm px-3 py-1.5',
    };

    const modeLabels: Record<IterationMode, string> = {
      bio: 'BIO',
      mecha: 'MECHA',
      hybrid: 'HYBRID',
    };

    const modeIcons: Record<IterationMode, string> = {
      bio: '\u2661', // heart outline
      mecha: '\u2699', // gear
      hybrid: '\u221E', // infinity
    };

    return (
      <span
        class={`rounded font-bold ${sizeClasses[size]} border`}
        style={{
          backgroundColor: `${MODE_COLORS[mode]}20`,
          borderColor: `${MODE_COLORS[mode]}50`,
          color: MODE_COLORS[mode],
        }}
      >
        {modeIcons[mode]} {modeLabels[mode]}
      </span>
    );
  }
);

/** Convergence gauge */
const ConvergenceGauge = component$<{ score: number; state: ConvergenceState }>(
  ({ score, state }) => {
    const percentage = Math.min(100, Math.max(0, score * 100));

    return (
      <div class="flex flex-col gap-1">
        <div class="flex justify-between text-xs text-zinc-400">
          <span>Convergence</span>
          <span
            style={{ color: CONVERGENCE_COLORS[state] }}
            class="font-semibold"
          >
            {percentage.toFixed(0)}%
            {state === 'converging' && ' \u2191'}
            {state === 'diverging' && ' \u2193'}
            {state === 'stable' && ' \u2192'}
          </span>
        </div>
        <div class="h-2 bg-zinc-800 rounded-full overflow-hidden">
          <div
            class="h-full rounded-full transition-all duration-500"
            style={{
              width: `${percentage}%`,
              backgroundColor: CONVERGENCE_COLORS[state],
            }}
          />
        </div>
      </div>
    );
  }
);

/** Progress ring for overall cycle */
const CycleProgressRing = component$<{
  completed: number;
  total: number;
  current: number;
}>(({ completed, total, current }) => {
  const percentage = (completed / total) * 100;
  const circumference = 2 * Math.PI * 40;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div class="relative w-24 h-24">
      <svg class="w-full h-full -rotate-90" viewBox="0 0 100 100">
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke="#27272a"
          stroke-width="8"
        />
        {/* Progress arc */}
        <circle
          cx="50"
          cy="50"
          r="40"
          fill="none"
          stroke="#3b82f6"
          stroke-width="8"
          stroke-linecap="round"
          stroke-dasharray={circumference}
          stroke-dashoffset={strokeDashoffset}
          class="transition-all duration-500"
        />
      </svg>
      <div class="absolute inset-0 flex flex-col items-center justify-center">
        <span class="text-xl font-bold text-zinc-100">{current}</span>
        <span class="text-[10px] text-zinc-500">/ {total}</span>
      </div>
    </div>
  );
});

/** Subagent card */
const SubAgentCard = component$<{ agent: SubAgent }>(({ agent }) => {
  const statusColors: Record<SubAgent['status'], string> = {
    idle: '#6b7280',
    active: '#10b981',
    waiting: '#f59e0b',
    complete: '#3b82f6',
    error: '#ef4444',
  };

  return (
    <div
      class="rounded-lg border p-2 transition-all"
      style={{
        backgroundColor: `${SPECIES_COLORS[agent.species]}10`,
        borderColor: `${SPECIES_COLORS[agent.species]}30`,
      }}
    >
      <div class="flex items-center justify-between mb-1">
        <div class="flex items-center gap-1.5">
          <span
            class="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: statusColors[agent.status] }}
          />
          <span class="text-xs font-semibold text-zinc-200 truncate max-w-[80px]">
            {agent.name}
          </span>
        </div>
        <span
          class="text-[10px] px-1 py-0.5 rounded"
          style={{
            backgroundColor: `${SPECIES_COLORS[agent.species]}20`,
            color: SPECIES_COLORS[agent.species],
          }}
        >
          {agent.species}
        </span>
      </div>
      <div class="text-[10px] text-zinc-500">
        {agent.iterationsCompleted} iterations
      </div>
      {agent.currentTask && (
        <div class="text-[9px] text-zinc-400 truncate mt-0.5">
          {agent.currentTask}
        </div>
      )}
    </div>
  );
});

/** Timeline iteration node */
const IterationNode = component$<{
  iteration: Iteration;
  isActive: boolean;
  isCurrent: boolean;
}>(({ iteration, isActive, isCurrent }) => {
  const nodeSize = isCurrent ? 'w-4 h-4' : 'w-3 h-3';
  const bgColor = iteration.isHandoff
    ? PHASE_COLORS.handoff
    : MODE_COLORS[iteration.mode];

  return (
    <div class="flex flex-col items-center gap-1">
      {/* Node */}
      <div
        class={`${nodeSize} rounded-full transition-all ${
          isCurrent ? 'ring-2 ring-offset-2 ring-offset-zinc-900' : ''
        } ${isActive ? 'animate-pulse' : ''}`}
        style={{
          backgroundColor: bgColor,
          ringColor: bgColor,
        }}
        title={`#${iteration.index + 1}: ${iteration.description}`}
      />

      {/* Connector line for handoffs */}
      {iteration.isHandoff && (
        <div class="w-px h-2 bg-gradient-to-b from-pink-500 to-transparent" />
      )}
    </div>
  );
});

/** Timeline visualization */
const TimelineTrack = component$<{
  iterations: Iteration[];
  currentIndex: number;
}>(({ iterations, currentIndex }) => {
  // Group iterations by mode segments
  const segments: { mode: IterationMode; count: number; startIndex: number }[] = [];
  let currentSegment: (typeof segments)[0] | null = null;

  iterations.forEach((iter, idx) => {
    if (!currentSegment || currentSegment.mode !== iter.mode) {
      if (currentSegment) segments.push(currentSegment);
      currentSegment = { mode: iter.mode, count: 1, startIndex: idx };
    } else {
      currentSegment.count++;
    }
  });
  if (currentSegment) segments.push(currentSegment);

  return (
    <div class="relative">
      {/* Track background */}
      <div class="absolute top-1.5 left-0 right-0 h-0.5 bg-zinc-800" />

      {/* Mode segments */}
      <div class="absolute top-1.5 left-0 right-0 h-0.5 flex">
        {segments.map((seg, idx) => (
          <div
            key={idx}
            class="h-full transition-all"
            style={{
              width: `${(seg.count / iterations.length) * 100}%`,
              backgroundColor: MODE_COLORS[seg.mode],
              opacity: seg.startIndex + seg.count <= currentIndex ? 1 : 0.3,
            }}
          />
        ))}
      </div>

      {/* Iteration nodes */}
      <div class="flex justify-between items-start relative z-10">
        {iterations.slice(0, 16).map((iter, idx) => (
          <IterationNode
            key={iter.id}
            iteration={iter}
            isActive={idx === currentIndex}
            isCurrent={idx === currentIndex}
          />
        ))}
        {iterations.length === 0 && (
          <div class="w-full text-center text-xs text-zinc-500 py-2">
            No iterations yet
          </div>
        )}
      </div>
    </div>
  );
});

/** Handoff event marker */
const HandoffMarker = component$<{
  from: 'bio' | 'mecha';
  to: 'bio' | 'mecha';
  time: number;
}>(({ from, to, time }) => {
  const timeAgo = Date.now() - time;

  return (
    <div class="flex items-center gap-2 text-xs">
      <ModeIndicator mode={from} size="sm" />
      <span class="text-zinc-400">\u2192</span>
      <ModeIndicator mode={to} size="sm" />
      <span class="text-zinc-500 text-[10px]">{formatDuration(timeAgo)} ago</span>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

export interface IterationFlowTimelineProps {
  events?: Signal<BusEvent[]>;
  maxIterations?: number;
  targetIterations?: number;
  subAgentCount?: number;
  onModeChange$?: (mode: IterationMode) => void;
}

export const IterationFlowTimeline = component$<IterationFlowTimelineProps>(
  ({
    events,
    maxIterations = 20,
    targetIterations = 15,
    subAgentCount = 4,
  }) => {
    const state = useSignal<IterationFlowState>(DEFAULT_STATE);
    const lastEventCount = useSignal(0);

    // Parse events to update state
    useVisibleTask$(({ track }) => {
      if (events) {
        track(() => events.value);

        // Only reparse if events changed
        if (events.value.length !== lastEventCount.value) {
          lastEventCount.value = events.value.length;
          const parsed = parseIterationEvents(events.value);
          state.value = {
            ...state.value,
            ...parsed,
            progress: {
              ...state.value.progress,
              ...parsed.progress,
              totalIterations: targetIterations,
            },
          };
        }
      }
    });

    // Demo mode: generate sample iterations if no events
    useVisibleTask$(({ cleanup }) => {
      if (events && events.value.length > 0) return;

      // Generate demo data for visualization
      const demoAgents: SubAgent[] = [
        {
          id: 'coordinator',
          name: 'Coordinator',
          species: 'claude',
          status: 'active',
          iterationsCompleted: 3,
          currentTask: 'Orchestrating subagents',
          lastActivity: Date.now(),
        },
        {
          id: 'codex-impl',
          name: 'Codex Impl',
          species: 'codex',
          status: 'active',
          iterationsCompleted: 5,
          currentTask: 'Code generation',
          lastActivity: Date.now() - 5000,
        },
        {
          id: 'reviewer',
          name: 'PR Reviewer',
          species: 'claude',
          status: 'waiting',
          iterationsCompleted: 2,
          currentTask: 'Awaiting changes',
          lastActivity: Date.now() - 15000,
        },
        {
          id: 'researcher',
          name: 'SOTA Researcher',
          species: 'gemini',
          status: 'idle',
          iterationsCompleted: 1,
          currentTask: undefined,
          lastActivity: Date.now() - 60000,
        },
      ];

      const demoIterations: Iteration[] = [
        {
          id: 'iter-1',
          index: 0,
          mode: 'bio',
          phase: 'planning',
          description: 'Human: Define task requirements',
          startTime: Date.now() - 300000,
          endTime: Date.now() - 280000,
          isHandoff: false,
          convergenceScore: 0.1,
        },
        {
          id: 'handoff-1',
          index: 1,
          mode: 'hybrid',
          phase: 'handoff',
          description: 'Bio -> Mecha handoff',
          startTime: Date.now() - 280000,
          isHandoff: true,
          handoffFrom: 'bio',
          handoffTo: 'mecha',
        },
        {
          id: 'iter-2',
          index: 2,
          mode: 'mecha',
          phase: 'execution',
          agentId: 'coordinator',
          description: 'Coordinator: Task decomposition',
          startTime: Date.now() - 270000,
          endTime: Date.now() - 250000,
          isHandoff: false,
          convergenceScore: 0.2,
        },
        {
          id: 'iter-3',
          index: 3,
          mode: 'mecha',
          phase: 'execution',
          agentId: 'codex-impl',
          description: 'Codex: Initial implementation',
          startTime: Date.now() - 240000,
          endTime: Date.now() - 180000,
          isHandoff: false,
          convergenceScore: 0.35,
          tokens: 4500,
        },
        {
          id: 'iter-4',
          index: 4,
          mode: 'mecha',
          phase: 'verification',
          agentId: 'reviewer',
          description: 'Reviewer: Code review',
          startTime: Date.now() - 170000,
          endTime: Date.now() - 150000,
          isHandoff: false,
          convergenceScore: 0.4,
        },
        {
          id: 'iter-5',
          index: 5,
          mode: 'mecha',
          phase: 'execution',
          agentId: 'codex-impl',
          description: 'Codex: Address review feedback',
          startTime: Date.now() - 140000,
          endTime: Date.now() - 100000,
          isHandoff: false,
          convergenceScore: 0.55,
          tokens: 2800,
        },
        {
          id: 'iter-6',
          index: 6,
          mode: 'mecha',
          phase: 'verification',
          agentId: 'codex-impl',
          description: 'Codex: Run tests',
          startTime: Date.now() - 90000,
          endTime: Date.now() - 70000,
          isHandoff: false,
          convergenceScore: 0.65,
        },
        {
          id: 'iter-7',
          index: 7,
          mode: 'mecha',
          phase: 'execution',
          agentId: 'codex-impl',
          description: 'Codex: Final refinements',
          startTime: Date.now() - 60000,
          isHandoff: false,
          convergenceScore: 0.75,
          tokens: 1200,
        },
      ];

      state.value = {
        iterations: demoIterations,
        subAgents: demoAgents.slice(0, subAgentCount),
        progress: {
          totalIterations: targetIterations,
          completedIterations: 6,
          currentIteration: 7,
          bioIterations: 1,
          mechaIterations: 6,
          handoffCount: 1,
          convergenceScore: 0.75,
          convergenceState: 'converging',
          estimatedRemaining: targetIterations - 7,
        },
        currentMode: 'mecha',
        lastHandoffTime: Date.now() - 280000,
        cycleStartTime: Date.now() - 300000,
        cycleGoal: 'Implement Co-Bio/Mecha-Neural Iteration Flow UI',
      };

      // Animate progress
      let frame = 0;
      const interval = setInterval(() => {
        frame++;
        // Subtle convergence animation
        state.value = {
          ...state.value,
          progress: {
            ...state.value.progress,
            convergenceScore:
              0.75 + Math.sin(frame * 0.1) * 0.02,
          },
        };
      }, 500);

      cleanup(() => clearInterval(interval));
    });

    // Computed stats
    const bioMechaRatio = useComputed$(() => {
      const { bioIterations, mechaIterations } = state.value.progress;
      const total = bioIterations + mechaIterations;
      if (total === 0) return { bio: 0, mecha: 0 };
      return {
        bio: (bioIterations / total) * 100,
        mecha: (mechaIterations / total) * 100,
      };
    });

    const elapsedTime = useComputed$(() => {
      return Date.now() - state.value.cycleStartTime;
    });

    return (
      <div class="rounded-lg border border-zinc-700 bg-zinc-900/80 p-4">
        {/* Header */}
        <div class="flex items-center justify-between mb-4">
          <div class="flex items-center gap-3">
            <h3 class="text-sm font-semibold text-zinc-100">
              Co-Bio/Mecha-Neural Flow
            </h3>
            <ModeIndicator mode={state.value.currentMode} />
          </div>
          <div class="flex items-center gap-2 text-xs text-zinc-500">
            <span>Elapsed: {formatDuration(elapsedTime.value)}</span>
            <span class="text-zinc-600">|</span>
            <span>{state.value.subAgents.length} agents</span>
          </div>
        </div>

        {/* Goal */}
        <div class="mb-4 px-3 py-2 rounded bg-zinc-800/50 border border-zinc-700/50">
          <div class="text-[10px] text-zinc-500 mb-0.5">CYCLE GOAL</div>
          <div class="text-xs text-zinc-300">{state.value.cycleGoal}</div>
        </div>

        {/* Main layout */}
        <div class="grid grid-cols-12 gap-4">
          {/* Left: Progress ring and stats */}
          <div class="col-span-3 flex flex-col items-center gap-3">
            <CycleProgressRing
              completed={state.value.progress.completedIterations}
              total={state.value.progress.totalIterations}
              current={state.value.progress.currentIteration}
            />
            <div class="text-center">
              <div class="text-[10px] text-zinc-500">
                {state.value.progress.estimatedRemaining} remaining
              </div>
            </div>

            {/* Bio/Mecha ratio bar */}
            <div class="w-full">
              <div class="text-[10px] text-zinc-500 mb-1 flex justify-between">
                <span>Bio/Mecha Ratio</span>
              </div>
              <div class="h-2 rounded-full overflow-hidden flex">
                <div
                  class="h-full transition-all"
                  style={{
                    width: `${bioMechaRatio.value.bio}%`,
                    backgroundColor: MODE_COLORS.bio,
                  }}
                />
                <div
                  class="h-full transition-all"
                  style={{
                    width: `${bioMechaRatio.value.mecha}%`,
                    backgroundColor: MODE_COLORS.mecha,
                  }}
                />
              </div>
              <div class="flex justify-between text-[9px] text-zinc-500 mt-0.5">
                <span>{state.value.progress.bioIterations} bio</span>
                <span>{state.value.progress.mechaIterations} mecha</span>
              </div>
            </div>
          </div>

          {/* Center: Timeline and convergence */}
          <div class="col-span-6 flex flex-col gap-3">
            {/* Timeline */}
            <div class="px-2">
              <div class="text-[10px] text-zinc-500 mb-2">ITERATION TIMELINE</div>
              <TimelineTrack
                iterations={state.value.iterations}
                currentIndex={state.value.progress.currentIteration}
              />
            </div>

            {/* Convergence gauge */}
            <div class="mt-2">
              <ConvergenceGauge
                score={state.value.progress.convergenceScore}
                state={state.value.progress.convergenceState}
              />
            </div>

            {/* Recent handoffs */}
            {state.value.lastHandoffTime > 0 && (
              <div class="mt-2">
                <div class="text-[10px] text-zinc-500 mb-1">LAST HANDOFF</div>
                <HandoffMarker
                  from={
                    state.value.iterations.find((i) => i.isHandoff)?.handoffFrom ||
                    'mecha'
                  }
                  to={
                    state.value.iterations.find((i) => i.isHandoff)?.handoffTo ||
                    'bio'
                  }
                  time={state.value.lastHandoffTime}
                />
              </div>
            )}
          </div>

          {/* Right: Subagents */}
          <div class="col-span-3">
            <div class="text-[10px] text-zinc-500 mb-2">
              SUBAGENTS ({state.value.subAgents.length}/{subAgentCount})
            </div>
            <div class="space-y-2">
              {state.value.subAgents.map((agent) => (
                <SubAgentCard key={agent.id} agent={agent} />
              ))}
              {state.value.subAgents.length === 0 && (
                <div class="text-[10px] text-zinc-500 text-center py-4">
                  No active agents
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer: Stats summary */}
        <div class="mt-4 pt-3 border-t border-zinc-800 flex items-center justify-between text-[10px]">
          <div class="flex gap-3">
            <span class="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              {state.value.progress.bioIterations} bio
            </span>
            <span class="px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
              {state.value.progress.mechaIterations} mecha
            </span>
            <span class="px-2 py-0.5 rounded bg-pink-500/10 text-pink-400 border border-pink-500/20">
              {state.value.progress.handoffCount} handoffs
            </span>
          </div>
          <div class="text-zinc-500">
            Model: {targetIterations} iterations / {subAgentCount} subagents
          </div>
        </div>
      </div>
    );
  }
);

export default IterationFlowTimeline;
