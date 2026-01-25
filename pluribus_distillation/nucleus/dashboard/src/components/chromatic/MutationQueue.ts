/**
 * Chromatic Agents Visualizer - Mutation Queue
 *
 * Step 13: Buffer bus events and apply visual mutations at 60fps.
 * Ensures smooth animations by batching and rate-limiting mutations.
 */

import type {
  AgentId,
  AgentVisualEvent,
  ChromaticState,
  VisualMutation,
  CodeGraph,
  AgentVisualData,
} from './types';
import { AgentVisualState } from './types';
import {
  getAgentColor,
  getAgentHue,
  getAgentOrbitalPosition,
} from './utils/colorMap';
import {
  lerp,
  exponentialDecay,
  lerpEased,
  easeOutQuad,
} from './utils/interpolation';

// =============================================================================
// Constants
// =============================================================================

/** Maximum mutations to process per frame */
const MAX_MUTATIONS_PER_FRAME = 10;

/** Target frame time in seconds (60fps) */
const FRAME_TIME = 1 / 60;

/** Mutation priority levels */
export enum MutationPriority {
  LOW = 0,
  NORMAL = 1,
  HIGH = 2,
  CRITICAL = 3,
}

// =============================================================================
// Bus Event Types (from the bus)
// =============================================================================

export interface ChromaticBusEvent {
  topic: string;
  kind: string;
  actor: string;
  data: Record<string, unknown>;
  timestamp_iso: string;
  trace_id?: string;
}

// =============================================================================
// Mutation Types
// =============================================================================

export type MutationType =
  | 'state_change'
  | 'intensity_change'
  | 'position_change'
  | 'spawn'
  | 'despawn'
  | 'merge'
  | 'code_graph_update'
  | 'particle_burst'
  | 'focus_change';

export interface QueuedMutation extends VisualMutation {
  priority: MutationPriority;
  createdAt: number;
  /** Optional duration for animated mutations */
  duration?: number;
  /** Progress 0-1 for ongoing mutations */
  progress?: number;
}

// =============================================================================
// Mutation Factory
// =============================================================================

/**
 * Create a state change mutation
 */
function createStateChangeMutation(
  agentId: AgentId,
  newState: AgentVisualState,
  traceId?: string
): QueuedMutation {
  const timestamp = Date.now();

  return {
    type: 'state_change',
    agent_id: agentId,
    timestamp,
    payload: { newState, traceId },
    priority: MutationPriority.HIGH,
    createdAt: timestamp,
    duration: getStateDuration(newState),
    apply: (state: ChromaticState, deltaTime: number) => {
      const agent = state.agents.get(agentId);
      if (!agent) return;

      agent.state = newState;
      agent.lastUpdate = Date.now();

      // Apply state-specific visual changes
      switch (newState) {
        case AgentVisualState.IDLE:
          agent.opacity = 0.1;
          agent.intensity = 0;
          break;
        case AgentVisualState.CLONING:
          agent.opacity = lerp(agent.opacity, 0.5, deltaTime * 3);
          agent.intensity = 0.3;
          break;
        case AgentVisualState.WORKING:
          agent.opacity = 1;
          agent.intensity = 0.8;
          break;
        case AgentVisualState.COMMITTING:
          agent.intensity = 1; // Flash
          break;
        case AgentVisualState.PUSHING:
          agent.intensity = 0.9;
          break;
        case AgentVisualState.MERGED:
          // Will transition to main color
          agent.intensity = 0.5;
          break;
        case AgentVisualState.CLEANUP:
          agent.opacity = exponentialDecay(agent.opacity, 0, 2, deltaTime);
          agent.intensity = exponentialDecay(agent.intensity, 0, 2, deltaTime);
          break;
      }
    },
  };
}

/**
 * Create an intensity change mutation
 */
function createIntensityMutation(
  agentId: AgentId,
  intensity: number
): QueuedMutation {
  const timestamp = Date.now();

  return {
    type: 'intensity_change',
    agent_id: agentId,
    timestamp,
    payload: { intensity },
    priority: MutationPriority.NORMAL,
    createdAt: timestamp,
    apply: (state: ChromaticState, deltaTime: number) => {
      const agent = state.agents.get(agentId);
      if (!agent) return;

      agent.intensity = lerpEased(
        agent.intensity,
        intensity,
        deltaTime * 4,
        easeOutQuad
      );
      agent.lastUpdate = Date.now();
    },
  };
}

/**
 * Create a spawn mutation (agent becomes active)
 */
function createSpawnMutation(
  agentId: AgentId,
  branch: string,
  clonePath: string
): QueuedMutation {
  const timestamp = Date.now();
  const position = getAgentOrbitalPosition(agentId);

  return {
    type: 'spawn',
    agent_id: agentId,
    timestamp,
    payload: { branch, clonePath, position },
    priority: MutationPriority.CRITICAL,
    createdAt: timestamp,
    duration: 1000, // 1 second spawn animation
    apply: (state: ChromaticState, deltaTime: number) => {
      let agent = state.agents.get(agentId);

      if (!agent) {
        // Create new agent data
        agent = {
          id: agentId,
          state: AgentVisualState.CLONING,
          hue: getAgentHue(agentId),
          color: getAgentColor(agentId),
          intensity: 0,
          codeGraph: null,
          branch,
          position,
          opacity: 0,
          lastUpdate: Date.now(),
        };
        state.agents.set(agentId, agent);
      }

      // Animate spawn
      agent.branch = branch;
      agent.opacity = lerp(agent.opacity, 1, deltaTime * 2);
      agent.intensity = lerp(agent.intensity, 0.5, deltaTime * 2);
      agent.state = AgentVisualState.CLONING;
      agent.lastUpdate = Date.now();

      // Update prism
      if (!state.focusedAgent) {
        state.focusedAgent = agentId;
      }
    },
  };
}

/**
 * Create a despawn mutation (agent cleanup)
 */
function createDespawnMutation(agentId: AgentId): QueuedMutation {
  const timestamp = Date.now();

  return {
    type: 'despawn',
    agent_id: agentId,
    timestamp,
    payload: {},
    priority: MutationPriority.HIGH,
    createdAt: timestamp,
    duration: 800, // 0.8 second fade out
    apply: (state: ChromaticState, deltaTime: number) => {
      const agent = state.agents.get(agentId);
      if (!agent) return;

      agent.state = AgentVisualState.CLEANUP;
      agent.opacity = exponentialDecay(agent.opacity, 0, 3, deltaTime);
      agent.intensity = exponentialDecay(agent.intensity, 0, 3, deltaTime);
      agent.lastUpdate = Date.now();

      // Remove when fully faded
      if (agent.opacity < 0.01) {
        state.agents.delete(agentId);
        if (state.focusedAgent === agentId) {
          state.focusedAgent = null;
        }
      }
    },
  };
}

/**
 * Create a merge mutation
 */
function createMergeMutation(agentId: AgentId): QueuedMutation {
  const timestamp = Date.now();

  return {
    type: 'merge',
    agent_id: agentId,
    timestamp,
    payload: {},
    priority: MutationPriority.CRITICAL,
    createdAt: timestamp,
    duration: 1500, // 1.5 second merge animation
    progress: 0,
    apply: (state: ChromaticState, deltaTime: number) => {
      const agent = state.agents.get(agentId);
      if (!agent) return;

      agent.state = AgentVisualState.MERGED;

      // Move toward center (main)
      agent.position = [
        lerp(agent.position[0], 0, deltaTime * 2),
        lerp(agent.position[1], 0, deltaTime * 2),
        lerp(agent.position[2], 0, deltaTime * 2),
      ];

      // Fade toward white (main color)
      agent.intensity = lerp(agent.intensity, 1, deltaTime * 2);
      agent.lastUpdate = Date.now();

      // Update prism intensity
      state.prismIntensity = Math.min(1, state.prismIntensity + deltaTime * 0.5);
    },
  };
}

/**
 * Create a code graph update mutation
 */
function createCodeGraphMutation(
  agentId: AgentId,
  codeGraph: CodeGraph
): QueuedMutation {
  const timestamp = Date.now();

  return {
    type: 'code_graph_update',
    agent_id: agentId,
    timestamp,
    payload: { codeGraph },
    priority: MutationPriority.NORMAL,
    createdAt: timestamp,
    apply: (state: ChromaticState, _deltaTime: number) => {
      const agent = state.agents.get(agentId);
      if (!agent) return;

      agent.codeGraph = codeGraph;
      agent.lastUpdate = Date.now();
    },
  };
}

/**
 * Create a particle burst mutation (for commits, pushes)
 */
function createParticleBurstMutation(
  agentId: AgentId,
  burstType: 'commit' | 'push' | 'merge'
): QueuedMutation {
  const timestamp = Date.now();

  return {
    type: 'particle_burst',
    agent_id: agentId,
    timestamp,
    payload: { burstType },
    priority: MutationPriority.HIGH,
    createdAt: timestamp,
    duration: 500, // 0.5 second burst
    apply: (state: ChromaticState, deltaTime: number) => {
      const agent = state.agents.get(agentId);
      if (!agent) return;

      // Brief intensity spike
      agent.intensity = Math.min(1, agent.intensity + deltaTime * 5);
      agent.lastUpdate = Date.now();
    },
  };
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get animation duration for state transitions
 */
function getStateDuration(state: AgentVisualState): number {
  switch (state) {
    case AgentVisualState.CLONING:
      return 1000;
    case AgentVisualState.COMMITTING:
      return 300;
    case AgentVisualState.PUSHING:
      return 500;
    case AgentVisualState.MERGED:
      return 1500;
    case AgentVisualState.CLEANUP:
      return 800;
    default:
      return 0;
  }
}

/**
 * Parse agent ID from bus event data
 */
function parseAgentId(data: Record<string, unknown>): AgentId | null {
  const agentStr = data.agent_id ?? data.agent ?? data.actor;
  if (typeof agentStr !== 'string') return null;

  const normalized = agentStr.toLowerCase();
  if (normalized.includes('claude')) return 'claude';
  if (normalized.includes('qwen')) return 'qwen';
  if (normalized.includes('gemini')) return 'gemini';
  if (normalized.includes('codex')) return 'codex';
  return null;
}

/**
 * Parse visual state from string
 */
function parseState(stateStr: string): AgentVisualState | null {
  const normalized = stateStr.toLowerCase();
  switch (normalized) {
    case 'idle':
      return AgentVisualState.IDLE;
    case 'cloning':
      return AgentVisualState.CLONING;
    case 'working':
      return AgentVisualState.WORKING;
    case 'committing':
      return AgentVisualState.COMMITTING;
    case 'pushing':
      return AgentVisualState.PUSHING;
    case 'merged':
      return AgentVisualState.MERGED;
    case 'cleanup':
      return AgentVisualState.CLEANUP;
    default:
      return null;
  }
}

// =============================================================================
// Mutation Queue Class
// =============================================================================

export class MutationQueue {
  private pending: QueuedMutation[] = [];
  private active: QueuedMutation[] = [];
  private lastFrameTime: number = 0;
  private frameCount: number = 0;
  private mutationsApplied: number = 0;

  /** Statistics for monitoring */
  public stats = {
    pendingCount: 0,
    activeCount: 0,
    mutationsPerSecond: 0,
    droppedCount: 0,
  };

  /**
   * Handle incoming bus event and convert to mutation
   */
  onBusEvent(evt: ChromaticBusEvent): void {
    const mutation = this.eventToMutation(evt);
    if (mutation) {
      this.enqueue(mutation);
    }
  }

  /**
   * Convert bus event to visual mutation
   */
  eventToMutation(evt: ChromaticBusEvent): QueuedMutation | null {
    const { topic, data } = evt;

    // Parse agent ID from event
    const agentId = parseAgentId(data);

    switch (topic) {
      case 'paip.clone.created': {
        if (!agentId) return null;
        const branch = (data.branch ?? data.ref ?? 'unknown') as string;
        const clonePath = (data.clone_path ?? data.path ?? '/tmp/unknown') as string;
        return createSpawnMutation(agentId, branch, clonePath);
      }

      case 'paip.clone.deleted': {
        if (!agentId) return null;
        return createDespawnMutation(agentId);
      }

      case 'agent.codegraph.update':
      case 'viz.codegraph.update': {
        if (!agentId) return null;
        const codeGraph = data.code_graph as CodeGraph | undefined;
        if (!codeGraph) return null;
        return createCodeGraphMutation(agentId, codeGraph);
      }

      case 'git.commit.created': {
        if (!agentId) return null;
        return createParticleBurstMutation(agentId, 'commit');
      }

      case 'git.push.completed': {
        if (!agentId) return null;
        return createParticleBurstMutation(agentId, 'push');
      }

      case 'viz.agent.state': {
        if (!agentId) return null;
        const stateStr = data.state as string | undefined;
        if (!stateStr) return null;
        const state = parseState(stateStr);
        if (!state) return null;
        return createStateChangeMutation(agentId, state, evt.trace_id);
      }

      case 'viz.merge.animation': {
        if (!agentId) return null;
        return createMergeMutation(agentId);
      }

      case 'operator.pbrealityfix.broadcast': {
        // Reality fix affects all agents - increase intensity
        const mutations: QueuedMutation[] = [];
        for (const id of ['claude', 'qwen', 'gemini', 'codex'] as AgentId[]) {
          mutations.push(createIntensityMutation(id, 0.9));
        }
        // Return first, queue the rest
        if (mutations.length > 1) {
          for (let i = 1; i < mutations.length; i++) {
            this.enqueue(mutations[i]);
          }
        }
        return mutations[0] ?? null;
      }

      default:
        // Unknown topic, try to extract useful info
        if (agentId && data.activity_intensity !== undefined) {
          return createIntensityMutation(
            agentId,
            data.activity_intensity as number
          );
        }
        return null;
    }
  }

  /**
   * Enqueue a mutation with priority sorting
   */
  enqueue(mutation: QueuedMutation): void {
    // Insert in priority order (higher priority first)
    const insertIndex = this.pending.findIndex(
      (m) => m.priority < mutation.priority
    );

    if (insertIndex === -1) {
      this.pending.push(mutation);
    } else {
      this.pending.splice(insertIndex, 0, mutation);
    }

    // Prevent unbounded growth
    const MAX_PENDING = 100;
    if (this.pending.length > MAX_PENDING) {
      const dropped = this.pending.splice(MAX_PENDING);
      this.stats.droppedCount += dropped.length;
    }

    this.stats.pendingCount = this.pending.length;
  }

  /**
   * Apply mutations for this frame
   * @param state The chromatic state to mutate
   * @param deltaTime Time since last frame in seconds
   */
  applyFrame(state: ChromaticState, deltaTime: number): void {
    const now = Date.now();

    // Move batch from pending to active
    const batch = this.pending.splice(0, MAX_MUTATIONS_PER_FRAME);
    this.active.push(...batch);

    // Apply all active mutations
    const completed: QueuedMutation[] = [];

    for (const mutation of this.active) {
      try {
        mutation.apply(state, deltaTime);

        // Track progress for timed mutations
        if (mutation.duration && mutation.duration > 0) {
          const elapsed = now - mutation.createdAt;
          mutation.progress = Math.min(1, elapsed / mutation.duration);

          if (mutation.progress >= 1) {
            completed.push(mutation);
          }
        } else {
          // Non-timed mutations complete immediately
          completed.push(mutation);
        }
      } catch (err) {
        console.error('[MutationQueue] Failed to apply mutation:', err);
        completed.push(mutation); // Remove failed mutations
      }
    }

    // Remove completed mutations
    for (const mutation of completed) {
      const idx = this.active.indexOf(mutation);
      if (idx !== -1) {
        this.active.splice(idx, 1);
      }
    }

    // Update stats
    this.mutationsApplied += completed.length;
    this.frameCount++;

    // Calculate mutations per second every 60 frames
    if (this.frameCount >= 60) {
      const elapsed = (now - this.lastFrameTime) / 1000;
      this.stats.mutationsPerSecond = this.mutationsApplied / elapsed;
      this.mutationsApplied = 0;
      this.frameCount = 0;
      this.lastFrameTime = now;
    }

    this.stats.pendingCount = this.pending.length;
    this.stats.activeCount = this.active.length;
  }

  /**
   * Clear all pending and active mutations
   */
  clear(): void {
    this.pending = [];
    this.active = [];
    this.stats.pendingCount = 0;
    this.stats.activeCount = 0;
  }

  /**
   * Check if queue has pending work
   */
  hasPending(): boolean {
    return this.pending.length > 0 || this.active.length > 0;
  }

  /**
   * Get queue depth for monitoring
   */
  getDepth(): number {
    return this.pending.length + this.active.length;
  }
}

// =============================================================================
// Singleton Export
// =============================================================================

/** Global mutation queue instance */
export const mutationQueue = new MutationQueue();

export default MutationQueue;
