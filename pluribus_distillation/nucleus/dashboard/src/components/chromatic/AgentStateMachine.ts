/**
 * Chromatic Agents Visualizer - Agent State Machine
 *
 * Step 12: Per-agent state transitions with visual effects.
 * States: IDLE -> CLONING -> WORKING -> COMMITTING -> PUSHING -> MERGED -> CLEANUP
 */

import type { AgentId, AgentVisualData, ChromaticState } from './types';
import { AgentVisualState } from './types';
import {
  getAgentColor,
  getAgentHue,
  getAgentOrbitalPosition,
  getAgentGhostColor,
  getAgentGlowColor,
} from './utils/colorMap';
import {
  lerp,
  lerpEased,
  exponentialDecay,
  easeOutQuad,
  easeOutElastic,
  easeOutBack,
  Spring3D,
} from './utils/interpolation';

// =============================================================================
// State Transition Rules
// =============================================================================

/**
 * Valid state transitions map
 * Key: current state, Value: array of valid next states
 */
const STATE_TRANSITIONS: Record<AgentVisualState, AgentVisualState[]> = {
  [AgentVisualState.IDLE]: [AgentVisualState.CLONING],
  [AgentVisualState.CLONING]: [AgentVisualState.WORKING, AgentVisualState.CLEANUP],
  [AgentVisualState.WORKING]: [
    AgentVisualState.COMMITTING,
    AgentVisualState.CLEANUP,
  ],
  [AgentVisualState.COMMITTING]: [
    AgentVisualState.WORKING,
    AgentVisualState.PUSHING,
    AgentVisualState.CLEANUP,
  ],
  [AgentVisualState.PUSHING]: [
    AgentVisualState.WORKING,
    AgentVisualState.MERGED,
    AgentVisualState.CLEANUP,
  ],
  [AgentVisualState.MERGED]: [AgentVisualState.CLEANUP],
  [AgentVisualState.CLEANUP]: [AgentVisualState.IDLE],
};

// =============================================================================
// Visual Effect Configs per State
// =============================================================================

interface StateVisualConfig {
  opacity: number;
  intensity: number;
  scale: number;
  glowIntensity: number;
  pulseRate: number; // Pulses per second
  particleEmissionRate: number; // Particles per second
  transitionDuration: number; // ms
}

const STATE_VISUALS: Record<AgentVisualState, StateVisualConfig> = {
  [AgentVisualState.IDLE]: {
    opacity: 0.1,
    intensity: 0,
    scale: 0.5,
    glowIntensity: 0,
    pulseRate: 0,
    particleEmissionRate: 0,
    transitionDuration: 500,
  },
  [AgentVisualState.CLONING]: {
    opacity: 0.6,
    intensity: 0.4,
    scale: 0.8,
    glowIntensity: 0.3,
    pulseRate: 2,
    particleEmissionRate: 50,
    transitionDuration: 800,
  },
  [AgentVisualState.WORKING]: {
    opacity: 1.0,
    intensity: 0.8,
    scale: 1.0,
    glowIntensity: 0.6,
    pulseRate: 1,
    particleEmissionRate: 20,
    transitionDuration: 300,
  },
  [AgentVisualState.COMMITTING]: {
    opacity: 1.0,
    intensity: 1.0,
    scale: 1.2,
    glowIntensity: 1.0,
    pulseRate: 8,
    particleEmissionRate: 200,
    transitionDuration: 200,
  },
  [AgentVisualState.PUSHING]: {
    opacity: 1.0,
    intensity: 0.9,
    scale: 1.1,
    glowIntensity: 0.8,
    pulseRate: 3,
    particleEmissionRate: 100,
    transitionDuration: 400,
  },
  [AgentVisualState.MERGED]: {
    opacity: 0.8,
    intensity: 0.5,
    scale: 0.9,
    glowIntensity: 0.4,
    pulseRate: 0.5,
    particleEmissionRate: 30,
    transitionDuration: 1000,
  },
  [AgentVisualState.CLEANUP]: {
    opacity: 0,
    intensity: 0,
    scale: 0.3,
    glowIntensity: 0,
    pulseRate: 0,
    particleEmissionRate: 0,
    transitionDuration: 600,
  },
};

// =============================================================================
// State Machine Event Types
// =============================================================================

export interface StateTransitionEvent {
  agentId: AgentId;
  fromState: AgentVisualState;
  toState: AgentVisualState;
  timestamp: number;
  traceId?: string;
}

export interface StateMachineSnapshot {
  agentId: AgentId;
  currentState: AgentVisualState;
  stateEnteredAt: number;
  transitionProgress: number;
  visualConfig: StateVisualConfig;
  animatedValues: AnimatedValues;
}

interface AnimatedValues {
  opacity: number;
  intensity: number;
  scale: number;
  glowIntensity: number;
  pulsePhase: number;
  positionSpring: Spring3D;
}

// =============================================================================
// State Transition Handlers
// =============================================================================

type TransitionHandler = (
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
) => void;

/**
 * Get transition handler for a state pair
 */
function getTransitionHandler(
  from: AgentVisualState,
  to: AgentVisualState
): TransitionHandler {
  // Special handlers for specific transitions
  const key = `${from}:${to}`;

  switch (key) {
    case `${AgentVisualState.IDLE}:${AgentVisualState.CLONING}`:
      return handleIdleToCloning;

    case `${AgentVisualState.CLONING}:${AgentVisualState.WORKING}`:
      return handleCloningToWorking;

    case `${AgentVisualState.WORKING}:${AgentVisualState.COMMITTING}`:
      return handleWorkingToCommitting;

    case `${AgentVisualState.COMMITTING}:${AgentVisualState.PUSHING}`:
      return handleCommittingToPushing;

    case `${AgentVisualState.PUSHING}:${AgentVisualState.MERGED}`:
      return handlePushingToMerged;

    case `${AgentVisualState.MERGED}:${AgentVisualState.CLEANUP}`:
      return handleMergedToCleanup;

    default:
      return handleGenericTransition;
  }
}

// Transition handler implementations

function handleIdleToCloning(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Spawn effect: fade in with elastic scale
  const targetConfig = STATE_VISUALS[AgentVisualState.CLONING];
  const eased = easeOutElastic(progress);

  agent.opacity = lerp(0.1, targetConfig.opacity, eased);
  agent.intensity = lerp(0, targetConfig.intensity, eased);
}

function handleCloningToWorking(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Clone complete: settle into working state with pop effect
  const targetConfig = STATE_VISUALS[AgentVisualState.WORKING];
  const eased = easeOutBack(progress);

  agent.opacity = lerp(agent.opacity, targetConfig.opacity, eased);
  agent.intensity = lerp(agent.intensity, targetConfig.intensity, eased);
}

function handleWorkingToCommitting(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Commit flash: brief intensity spike
  const targetConfig = STATE_VISUALS[AgentVisualState.COMMITTING];

  // Quick flash at start, then settle
  if (progress < 0.3) {
    agent.intensity = 1.0;
  } else {
    agent.intensity = lerpEased(
      1.0,
      targetConfig.intensity,
      (progress - 0.3) / 0.7,
      easeOutQuad
    );
  }
}

function handleCommittingToPushing(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Push effect: directional glow toward remote
  const targetConfig = STATE_VISUALS[AgentVisualState.PUSHING];
  const eased = easeOutQuad(progress);

  agent.intensity = lerp(agent.intensity, targetConfig.intensity, eased);
}

function handlePushingToMerged(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Merge effect: move toward center, color shifts to white
  const targetConfig = STATE_VISUALS[AgentVisualState.MERGED];
  const eased = easeOutQuad(progress);

  // Position moves toward center
  agent.position = [
    lerp(agent.position[0], 0, eased * 0.5),
    lerp(agent.position[1], 0, eased * 0.5),
    lerp(agent.position[2], 0, eased * 0.5),
  ];

  agent.intensity = lerp(agent.intensity, targetConfig.intensity, eased);
}

function handleMergedToCleanup(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Cleanup: exponential fade out
  agent.opacity = exponentialDecay(agent.opacity, 0, 3, deltaTime);
  agent.intensity = exponentialDecay(agent.intensity, 0, 4, deltaTime);
}

function handleGenericTransition(
  agent: AgentVisualData,
  deltaTime: number,
  progress: number
): void {
  // Generic smooth transition
  const targetConfig = STATE_VISUALS[agent.state];
  const eased = easeOutQuad(progress);

  agent.opacity = lerp(agent.opacity, targetConfig.opacity, eased);
  agent.intensity = lerp(agent.intensity, targetConfig.intensity, eased);
}

// =============================================================================
// Agent State Machine Class
// =============================================================================

export class AgentStateMachine {
  private agentId: AgentId;
  private currentState: AgentVisualState;
  private previousState: AgentVisualState | null = null;
  private stateEnteredAt: number;
  private transitionProgress: number = 1; // 1 = fully transitioned
  private positionSpring: Spring3D;
  private pulsePhase: number = 0;

  /** Event listeners for state changes */
  private listeners: ((event: StateTransitionEvent) => void)[] = [];

  /** Persisted state for recovery */
  private persistedState: StateMachineSnapshot | null = null;

  constructor(agentId: AgentId, initialState: AgentVisualState = AgentVisualState.IDLE) {
    this.agentId = agentId;
    this.currentState = initialState;
    this.stateEnteredAt = Date.now();
    this.positionSpring = new Spring3D(getAgentOrbitalPosition(agentId));
  }

  /**
   * Get current state
   */
  getState(): AgentVisualState {
    return this.currentState;
  }

  /**
   * Get visual config for current state
   */
  getVisualConfig(): StateVisualConfig {
    return STATE_VISUALS[this.currentState];
  }

  /**
   * Check if transition to target state is valid
   */
  canTransitionTo(targetState: AgentVisualState): boolean {
    const validTransitions = STATE_TRANSITIONS[this.currentState];
    return validTransitions.includes(targetState);
  }

  /**
   * Transition to a new state
   * @returns true if transition was successful
   */
  transitionTo(targetState: AgentVisualState, traceId?: string): boolean {
    if (!this.canTransitionTo(targetState)) {
      console.warn(
        `[AgentStateMachine] Invalid transition: ${this.currentState} -> ${targetState} for agent ${this.agentId}`
      );
      return false;
    }

    const event: StateTransitionEvent = {
      agentId: this.agentId,
      fromState: this.currentState,
      toState: targetState,
      timestamp: Date.now(),
      traceId,
    };

    this.previousState = this.currentState;
    this.currentState = targetState;
    this.stateEnteredAt = Date.now();
    this.transitionProgress = 0;

    // Notify listeners
    for (const listener of this.listeners) {
      try {
        listener(event);
      } catch (err) {
        console.error('[AgentStateMachine] Listener error:', err);
      }
    }

    return true;
  }

  /**
   * Force transition (bypasses validation)
   * Use for recovery scenarios
   */
  forceState(state: AgentVisualState): void {
    this.previousState = this.currentState;
    this.currentState = state;
    this.stateEnteredAt = Date.now();
    this.transitionProgress = 1; // Skip transition animation
  }

  /**
   * Update state machine and apply visual effects
   * @param agent The agent visual data to update
   * @param deltaTime Time since last frame in seconds
   */
  update(agent: AgentVisualData, deltaTime: number): void {
    const config = STATE_VISUALS[this.currentState];

    // Update transition progress
    if (this.transitionProgress < 1) {
      const elapsed = Date.now() - this.stateEnteredAt;
      this.transitionProgress = Math.min(1, elapsed / config.transitionDuration);

      // Apply transition handler
      if (this.previousState) {
        const handler = getTransitionHandler(this.previousState, this.currentState);
        handler(agent, deltaTime, this.transitionProgress);
      }
    } else {
      // Steady state: apply ongoing visual effects
      this.applyStateEffects(agent, deltaTime, config);
    }

    // Update pulse phase
    this.pulsePhase += config.pulseRate * deltaTime * Math.PI * 2;
    if (this.pulsePhase > Math.PI * 2) {
      this.pulsePhase -= Math.PI * 2;
    }

    // Update position spring
    const targetPosition = this.currentState === AgentVisualState.MERGED
      ? [0, 0, 0] as [number, number, number]
      : getAgentOrbitalPosition(this.agentId);

    agent.position = this.positionSpring.update(targetPosition, deltaTime);

    // Sync state to agent
    agent.state = this.currentState;
  }

  /**
   * Apply ongoing visual effects for the current state
   */
  private applyStateEffects(
    agent: AgentVisualData,
    deltaTime: number,
    config: StateVisualConfig
  ): void {
    // Smoothly approach target values
    agent.opacity = lerpEased(
      agent.opacity,
      config.opacity,
      deltaTime * 3,
      easeOutQuad
    );
    agent.intensity = lerpEased(
      agent.intensity,
      config.intensity,
      deltaTime * 3,
      easeOutQuad
    );

    // Apply pulse modulation
    if (config.pulseRate > 0) {
      const pulseAmount = Math.sin(this.pulsePhase) * 0.1;
      agent.intensity = Math.min(1, agent.intensity + pulseAmount);
    }
  }

  /**
   * Get color for current state
   */
  getColor(): string {
    switch (this.currentState) {
      case AgentVisualState.IDLE:
        return getAgentGhostColor(this.agentId);
      case AgentVisualState.MERGED:
        // Blend toward white
        return getAgentColor(this.agentId); // Will be interpolated in rendering
      default:
        return getAgentColor(this.agentId);
    }
  }

  /**
   * Get glow color for current state
   */
  getGlowColor(): string {
    const config = STATE_VISUALS[this.currentState];
    if (config.glowIntensity <= 0) {
      return 'transparent';
    }
    return getAgentGlowColor(this.agentId);
  }

  /**
   * Get particle emission rate for current state
   */
  getParticleEmissionRate(): number {
    return STATE_VISUALS[this.currentState].particleEmissionRate;
  }

  /**
   * Add state change listener
   */
  onStateChange(listener: (event: StateTransitionEvent) => void): () => void {
    this.listeners.push(listener);
    return () => {
      const idx = this.listeners.indexOf(listener);
      if (idx !== -1) {
        this.listeners.splice(idx, 1);
      }
    };
  }

  /**
   * Create snapshot for persistence
   */
  createSnapshot(): StateMachineSnapshot {
    return {
      agentId: this.agentId,
      currentState: this.currentState,
      stateEnteredAt: this.stateEnteredAt,
      transitionProgress: this.transitionProgress,
      visualConfig: STATE_VISUALS[this.currentState],
      animatedValues: {
        opacity: 1, // Will be updated from agent
        intensity: 0.5,
        scale: 1,
        glowIntensity: 0.5,
        pulsePhase: this.pulsePhase,
        positionSpring: this.positionSpring,
      },
    };
  }

  /**
   * Restore from snapshot
   */
  restoreFromSnapshot(snapshot: StateMachineSnapshot): void {
    if (snapshot.agentId !== this.agentId) {
      console.warn('[AgentStateMachine] Snapshot agent ID mismatch');
      return;
    }

    this.currentState = snapshot.currentState;
    this.stateEnteredAt = snapshot.stateEnteredAt;
    this.transitionProgress = snapshot.transitionProgress;
    this.pulsePhase = snapshot.animatedValues.pulsePhase;
  }

  /**
   * Persist current state (for recovery)
   */
  persist(): void {
    this.persistedState = this.createSnapshot();
  }

  /**
   * Recover from persisted state
   */
  recover(): boolean {
    if (!this.persistedState) {
      return false;
    }
    this.restoreFromSnapshot(this.persistedState);
    return true;
  }
}

// =============================================================================
// State Machine Manager
// =============================================================================

/**
 * Manages state machines for all agents
 */
export class StateMachineManager {
  private machines: Map<AgentId, AgentStateMachine> = new Map();
  private globalListeners: ((event: StateTransitionEvent) => void)[] = [];

  /**
   * Get or create state machine for an agent
   */
  getOrCreate(agentId: AgentId): AgentStateMachine {
    let machine = this.machines.get(agentId);
    if (!machine) {
      machine = new AgentStateMachine(agentId);
      machine.onStateChange((event) => {
        for (const listener of this.globalListeners) {
          listener(event);
        }
      });
      this.machines.set(agentId, machine);
    }
    return machine;
  }

  /**
   * Get state machine if it exists
   */
  get(agentId: AgentId): AgentStateMachine | undefined {
    return this.machines.get(agentId);
  }

  /**
   * Update all state machines
   */
  updateAll(state: ChromaticState, deltaTime: number): void {
    for (const [agentId, machine] of this.machines) {
      const agentData = state.agents.get(agentId);
      if (agentData) {
        machine.update(agentData, deltaTime);
      }
    }
  }

  /**
   * Add global state change listener
   */
  onAnyStateChange(
    listener: (event: StateTransitionEvent) => void
  ): () => void {
    this.globalListeners.push(listener);
    return () => {
      const idx = this.globalListeners.indexOf(listener);
      if (idx !== -1) {
        this.globalListeners.splice(idx, 1);
      }
    };
  }

  /**
   * Get all active agents (not IDLE or CLEANUP)
   */
  getActiveAgents(): AgentId[] {
    const active: AgentId[] = [];
    for (const [agentId, machine] of this.machines) {
      const state = machine.getState();
      if (
        state !== AgentVisualState.IDLE &&
        state !== AgentVisualState.CLEANUP
      ) {
        active.push(agentId);
      }
    }
    return active;
  }

  /**
   * Persist all machines
   */
  persistAll(): void {
    for (const machine of this.machines.values()) {
      machine.persist();
    }
  }

  /**
   * Recover all machines
   */
  recoverAll(): void {
    for (const machine of this.machines.values()) {
      machine.recover();
    }
  }

  /**
   * Clear all state machines
   */
  clear(): void {
    this.machines.clear();
  }
}

// =============================================================================
// Singleton Export
// =============================================================================

/** Global state machine manager */
export const stateMachineManager = new StateMachineManager();

export default AgentStateMachine;
