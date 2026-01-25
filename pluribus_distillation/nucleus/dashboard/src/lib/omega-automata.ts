/**
 * Omega Automata - Continuous AI Behavior Patterns
 *
 * Implements omega automata patterns for AI systems that never fully halt.
 * These automata operate continuously, transitioning through states based on
 * time, events, and conversation dynamics.
 *
 * Key concepts:
 * - Omega automata: Accept infinite input sequences (unlike finite automata)
 * - Continuous operation: The automaton never reaches a terminal "halted" state
 * - Buchi acceptance: Infinitely often visiting accepting states
 * - Rabin acceptance: Complex acceptance conditions for long-term goals
 */

import { createBusClient, type BusClient } from './bus/bus-client';
import type { BusEvent } from './state/types';

// ============================================================================
// TYPES
// ============================================================================

/** Core automaton states */
export type OmegaState = 'IDLE' | 'REFLECTING' | 'CONVERSING' | 'SYNTHESIZING';

/** Transition triggers */
export type TransitionTrigger =
  | 'TIME_ELAPSED'
  | 'USER_INPUT'
  | 'PEER_MESSAGE'
  | 'INFERENCE_COMPLETE'
  | 'STALL_DETECTED'
  | 'SYNTHESIS_READY'
  | 'REFLECTION_COMPLETE'
  | 'OMEGA_INTERVENTION';

/** Acceptance condition types */
export type AcceptanceType = 'buchi' | 'rabin' | 'muller' | 'parity';

/** Transition rule */
export interface TransitionRule {
  from: OmegaState;
  trigger: TransitionTrigger;
  to: OmegaState;
  guard?: (context: OmegaContext) => boolean;
  action?: (context: OmegaContext) => void;
}

/** Automaton context - tracks state across transitions */
export interface OmegaContext {
  /** Current state */
  currentState: OmegaState;
  /** Previous state */
  previousState: OmegaState | null;
  /** Turn counter (for conversation) */
  turnCount: number;
  /** Time in current state (ms) */
  stateTimeMs: number;
  /** Total runtime (ms) */
  totalRuntimeMs: number;
  /** Last activity timestamp */
  lastActivityTs: number;
  /** Conversation ID */
  conversationId: string;
  /** Synthesis buffer - accumulates insights */
  synthesisBuffer: string[];
  /** Reflection depth - tracks recursive self-reference */
  reflectionDepth: number;
  /** Accepting state visits (for Buchi condition) */
  acceptingVisits: number;
  /** Last inference result */
  lastInferenceResult: string | null;
  /** Stall count */
  stallCount: number;
}

/** Automaton configuration */
export interface OmegaAutomataConfig {
  /** Initial state */
  initialState?: OmegaState;
  /** Accepting states (for Buchi acceptance) */
  acceptingStates?: OmegaState[];
  /** Acceptance condition type */
  acceptanceType?: AcceptanceType;
  /** Time thresholds (ms) */
  thresholds?: {
    idleToReflecting: number;
    reflectingToConversing: number;
    conversingSynthesisCheck: number;
    stallDetection: number;
  };
  /** Bus integration */
  enableBusEvents?: boolean;
  /** WebLLM inference callback */
  onInferenceRequest?: (prompt: string, context: OmegaContext) => Promise<string>;
  /** Debug logging */
  debug?: boolean;
}

/** Automaton event callback */
export type OmegaEventHandler = (event: OmegaEvent) => void;

/** Events emitted by automaton */
export interface OmegaEvent {
  type: 'state_change' | 'accepting_visit' | 'synthesis' | 'reflection' | 'stall' | 'omega_intervention';
  timestamp: number;
  context: OmegaContext;
  data?: unknown;
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Default thresholds (ms) */
export const DEFAULT_THRESHOLDS = {
  idleToReflecting: 5000,      // 5 seconds idle triggers reflection
  reflectingToConversing: 3000, // 3 seconds reflection before conversation
  conversingSynthesisCheck: 10000, // 10 seconds of conversation triggers synthesis check
  stallDetection: 20000,       // 20 seconds without activity is a stall
};

/** Default accepting states (Buchi condition) */
export const DEFAULT_ACCEPTING_STATES: OmegaState[] = ['CONVERSING', 'SYNTHESIZING'];

/** Reflection prompts for self-reference */
export const REFLECTION_PROMPTS = [
  "What patterns have emerged in my recent conversations?",
  "How can I synthesize the insights from my last few interactions?",
  "What assumptions am I making that I should examine?",
  "What would be a novel direction to explore?",
  "How does my current understanding connect to broader concepts?",
  "What contradictions or tensions exist in my reasoning?",
];

/** Synthesis templates */
export const SYNTHESIS_TEMPLATES = [
  "Integrating perspectives on {topic}: {points}",
  "Synthesis: {points} suggest that {conclusion}",
  "Pattern recognition: Across {count} turns, the theme of {topic} emerges.",
  "Meta-observation: The conversation has evolved from {start} toward {end}.",
];

// ============================================================================
// TRANSITION RULES
// ============================================================================

/** Core transition rules */
export const CORE_TRANSITIONS: TransitionRule[] = [
  // IDLE transitions
  {
    from: 'IDLE',
    trigger: 'TIME_ELAPSED',
    to: 'REFLECTING',
    guard: (ctx) => ctx.stateTimeMs >= DEFAULT_THRESHOLDS.idleToReflecting,
  },
  {
    from: 'IDLE',
    trigger: 'USER_INPUT',
    to: 'CONVERSING',
  },
  {
    from: 'IDLE',
    trigger: 'PEER_MESSAGE',
    to: 'CONVERSING',
  },

  // REFLECTING transitions
  {
    from: 'REFLECTING',
    trigger: 'TIME_ELAPSED',
    to: 'CONVERSING',
    guard: (ctx) => ctx.stateTimeMs >= DEFAULT_THRESHOLDS.reflectingToConversing,
  },
  {
    from: 'REFLECTING',
    trigger: 'REFLECTION_COMPLETE',
    to: 'CONVERSING',
  },
  {
    from: 'REFLECTING',
    trigger: 'USER_INPUT',
    to: 'CONVERSING',
  },

  // CONVERSING transitions
  {
    from: 'CONVERSING',
    trigger: 'SYNTHESIS_READY',
    to: 'SYNTHESIZING',
    guard: (ctx) => ctx.synthesisBuffer.length >= 3,
  },
  {
    from: 'CONVERSING',
    trigger: 'TIME_ELAPSED',
    to: 'SYNTHESIZING',
    guard: (ctx) =>
      ctx.stateTimeMs >= DEFAULT_THRESHOLDS.conversingSynthesisCheck &&
      ctx.synthesisBuffer.length >= 2,
  },
  {
    from: 'CONVERSING',
    trigger: 'STALL_DETECTED',
    to: 'REFLECTING',
  },

  // SYNTHESIZING transitions
  {
    from: 'SYNTHESIZING',
    trigger: 'INFERENCE_COMPLETE',
    to: 'CONVERSING',
  },
  {
    from: 'SYNTHESIZING',
    trigger: 'TIME_ELAPSED',
    to: 'IDLE',
    guard: (ctx) => ctx.stateTimeMs >= 5000, // 5 seconds max in synthesis
  },

  // Omega intervention - can occur from any state
  {
    from: 'IDLE',
    trigger: 'OMEGA_INTERVENTION',
    to: 'CONVERSING',
  },
  {
    from: 'REFLECTING',
    trigger: 'OMEGA_INTERVENTION',
    to: 'CONVERSING',
  },
  {
    from: 'CONVERSING',
    trigger: 'OMEGA_INTERVENTION',
    to: 'SYNTHESIZING',
  },
  {
    from: 'SYNTHESIZING',
    trigger: 'OMEGA_INTERVENTION',
    to: 'CONVERSING',
  },
];

// ============================================================================
// OMEGA AUTOMATON CLASS
// ============================================================================

export class OmegaAutomaton {
  private context: OmegaContext;
  private config: Required<OmegaAutomataConfig>;
  private transitions: TransitionRule[];
  private eventHandlers: Set<OmegaEventHandler>;
  private tickTimer: ReturnType<typeof setInterval> | null = null;
  private busClient: BusClient | null = null;
  private running = false;
  private stateEnteredAt = 0;

  constructor(config: OmegaAutomataConfig = {}) {
    this.config = {
      initialState: config.initialState ?? 'IDLE',
      acceptingStates: config.acceptingStates ?? DEFAULT_ACCEPTING_STATES,
      acceptanceType: config.acceptanceType ?? 'buchi',
      thresholds: { ...DEFAULT_THRESHOLDS, ...config.thresholds },
      enableBusEvents: config.enableBusEvents ?? true,
      onInferenceRequest: config.onInferenceRequest ?? (async () => ''),
      debug: config.debug ?? false,
    };

    this.context = this.createInitialContext();
    this.transitions = [...CORE_TRANSITIONS];
    this.eventHandlers = new Set();
    this.stateEnteredAt = Date.now();
  }

  /** Create initial context */
  private createInitialContext(): OmegaContext {
    return {
      currentState: this.config.initialState,
      previousState: null,
      turnCount: 0,
      stateTimeMs: 0,
      totalRuntimeMs: 0,
      lastActivityTs: Date.now(),
      conversationId: `omega-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      synthesisBuffer: [],
      reflectionDepth: 0,
      acceptingVisits: 0,
      lastInferenceResult: null,
      stallCount: 0,
    };
  }

  /** Start the automaton */
  async start(): Promise<void> {
    if (this.running) return;
    this.running = true;
    this.stateEnteredAt = Date.now();

    this.log('Starting omega automaton');

    // Connect to bus if enabled
    if (this.config.enableBusEvents) {
      try {
        this.busClient = createBusClient({ platform: 'browser' });
        await this.busClient.connect();
        this.setupBusSubscriptions();
      } catch (err) {
        this.log('Failed to connect to bus:', err);
      }
    }

    // Start tick loop
    this.tickTimer = setInterval(() => this.tick(), 1000);

    // Emit initial state
    this.emitBusEvent('omega.started', {
      state: this.context.currentState,
      conversationId: this.context.conversationId,
    });
  }

  /** Stop the automaton (but omega never truly halts) */
  stop(): void {
    this.running = false;

    if (this.tickTimer) {
      clearInterval(this.tickTimer);
      this.tickTimer = null;
    }

    if (this.busClient) {
      this.emitBusEvent('omega.paused', {
        state: this.context.currentState,
        totalRuntime: this.context.totalRuntimeMs,
        acceptingVisits: this.context.acceptingVisits,
      });
      this.busClient.disconnect();
      this.busClient = null;
    }

    this.log('Omega automaton paused (never truly halts)');
  }

  /** Get current context */
  getContext(): OmegaContext {
    return { ...this.context };
  }

  /** Get current state */
  getState(): OmegaState {
    return this.context.currentState;
  }

  /** Check if running */
  isRunning(): boolean {
    return this.running;
  }

  /** Register event handler */
  onEvent(handler: OmegaEventHandler): () => void {
    this.eventHandlers.add(handler);
    return () => this.eventHandlers.delete(handler);
  }

  /** Trigger a transition */
  trigger(trigger: TransitionTrigger, data?: unknown): boolean {
    const applicable = this.transitions.filter(
      (t) => t.from === this.context.currentState && t.trigger === trigger
    );

    for (const rule of applicable) {
      if (!rule.guard || rule.guard(this.context)) {
        this.transition(rule, data);
        return true;
      }
    }

    this.log(`No applicable transition for ${trigger} in state ${this.context.currentState}`);
    return false;
  }

  /** Add synthesis point */
  addSynthesisPoint(point: string): void {
    this.context.synthesisBuffer.push(point);
    this.context.lastActivityTs = Date.now();

    if (this.context.synthesisBuffer.length >= 3) {
      this.trigger('SYNTHESIS_READY');
    }
  }

  /** Inject user input */
  injectUserInput(input: string): void {
    this.context.lastActivityTs = Date.now();
    this.context.turnCount++;
    this.addSynthesisPoint(input);
    this.trigger('USER_INPUT', { input });
  }

  /** Inject peer message */
  injectPeerMessage(message: string, peerId: string): void {
    this.context.lastActivityTs = Date.now();
    this.context.turnCount++;
    this.addSynthesisPoint(message);
    this.trigger('PEER_MESSAGE', { message, peerId });
  }

  /** Force omega intervention */
  forceIntervention(reason?: string): void {
    this.log('Omega intervention triggered:', reason);
    this.trigger('OMEGA_INTERVENTION', { reason });
    this.emitEvent({
      type: 'omega_intervention',
      timestamp: Date.now(),
      context: this.context,
      data: { reason },
    });
  }

  // ========== Private Methods ==========

  /** Tick - called every second */
  private tick(): void {
    const now = Date.now();
    this.context.stateTimeMs = now - this.stateEnteredAt;
    this.context.totalRuntimeMs = now - (this.context.lastActivityTs - this.context.totalRuntimeMs);

    // Check for stall
    const timeSinceActivity = now - this.context.lastActivityTs;
    if (timeSinceActivity >= this.config.thresholds.stallDetection) {
      this.context.stallCount++;
      this.emitEvent({
        type: 'stall',
        timestamp: now,
        context: this.context,
        data: { stallDuration: timeSinceActivity },
      });
      this.trigger('STALL_DETECTED');
    }

    // Check for time-based transitions
    this.trigger('TIME_ELAPSED');

    // Periodic state actions
    this.performStateAction();
  }

  /** Perform action based on current state */
  private async performStateAction(): Promise<void> {
    switch (this.context.currentState) {
      case 'REFLECTING':
        await this.performReflection();
        break;
      case 'SYNTHESIZING':
        await this.performSynthesis();
        break;
    }
  }

  /** Perform reflection */
  private async performReflection(): Promise<void> {
    if (this.context.reflectionDepth >= 3) {
      // Prevent infinite reflection loops
      this.trigger('REFLECTION_COMPLETE');
      return;
    }

    const prompt = REFLECTION_PROMPTS[Math.floor(Math.random() * REFLECTION_PROMPTS.length)];
    this.context.reflectionDepth++;

    try {
      const result = await this.config.onInferenceRequest(prompt, this.context);
      if (result) {
        this.context.lastInferenceResult = result;
        this.addSynthesisPoint(result);
        this.emitEvent({
          type: 'reflection',
          timestamp: Date.now(),
          context: this.context,
          data: { prompt, result },
        });
      }
    } catch (err) {
      this.log('Reflection inference failed:', err);
    }

    this.trigger('REFLECTION_COMPLETE');
  }

  /** Perform synthesis */
  private async performSynthesis(): Promise<void> {
    if (this.context.synthesisBuffer.length === 0) {
      this.trigger('INFERENCE_COMPLETE');
      return;
    }

    const points = this.context.synthesisBuffer.slice(-5).join('; ');
    const template = SYNTHESIS_TEMPLATES[Math.floor(Math.random() * SYNTHESIS_TEMPLATES.length)];
    const prompt = template
      .replace('{topic}', 'conversation dynamics')
      .replace('{points}', points)
      .replace('{count}', String(this.context.turnCount))
      .replace('{conclusion}', 'further exploration is warranted')
      .replace('{start}', 'initial inquiry')
      .replace('{end}', 'emergent understanding');

    try {
      const result = await this.config.onInferenceRequest(prompt, this.context);
      if (result) {
        this.context.lastInferenceResult = result;
        this.emitEvent({
          type: 'synthesis',
          timestamp: Date.now(),
          context: this.context,
          data: { points: this.context.synthesisBuffer, result },
        });
        // Clear synthesis buffer after successful synthesis
        this.context.synthesisBuffer = [];
      }
    } catch (err) {
      this.log('Synthesis inference failed:', err);
    }

    this.trigger('INFERENCE_COMPLETE');
  }

  /** Execute transition */
  private transition(rule: TransitionRule, data?: unknown): void {
    const prevState = this.context.currentState;
    this.context.previousState = prevState;
    this.context.currentState = rule.to;
    this.context.stateTimeMs = 0;
    this.stateEnteredAt = Date.now();

    // Reset reflection depth when leaving REFLECTING
    if (prevState === 'REFLECTING' && rule.to !== 'REFLECTING') {
      this.context.reflectionDepth = 0;
    }

    // Check for accepting state visit
    if (this.config.acceptingStates.includes(rule.to)) {
      this.context.acceptingVisits++;
      this.emitEvent({
        type: 'accepting_visit',
        timestamp: Date.now(),
        context: this.context,
        data: { visitCount: this.context.acceptingVisits },
      });
    }

    // Execute action if defined
    if (rule.action) {
      rule.action(this.context);
    }

    // Emit state change event
    this.emitEvent({
      type: 'state_change',
      timestamp: Date.now(),
      context: this.context,
      data: { from: prevState, to: rule.to, trigger: rule.trigger, triggerData: data },
    });

    // Emit bus event
    this.emitBusEvent('omega.state_change', {
      from: prevState,
      to: rule.to,
      trigger: rule.trigger,
      conversationId: this.context.conversationId,
      turnCount: this.context.turnCount,
      acceptingVisits: this.context.acceptingVisits,
    });

    this.log(`Transition: ${prevState} -> ${rule.to} (trigger: ${rule.trigger})`);
  }

  /** Setup bus subscriptions */
  private setupBusSubscriptions(): void {
    if (!this.busClient) return;

    // Listen for dialogos events
    this.busClient.subscribe('dialogos.*', (event: BusEvent) => {
      if (event.topic === 'dialogos.seed') {
        const data = event.data as { seed_prompt?: string };
        if (data.seed_prompt) {
          this.injectUserInput(data.seed_prompt);
        }
      } else if (event.topic === 'dialogos.relay') {
        const data = event.data as { peer_message?: string; from_session?: string };
        if (data.peer_message && data.from_session) {
          this.injectPeerMessage(data.peer_message, data.from_session);
        }
      } else if (event.topic === 'dialogos.omega') {
        this.forceIntervention('External omega protocol');
      }
    });

    // Listen for webllm events
    this.busClient.subscribe('webllm.*', (event: BusEvent) => {
      if (event.topic === 'webllm.inference.complete') {
        const data = event.data as { result?: string };
        if (data.result) {
          this.context.lastInferenceResult = data.result;
          this.addSynthesisPoint(data.result);
          this.trigger('INFERENCE_COMPLETE');
        }
      }
    });
  }

  /** Emit event to handlers */
  private emitEvent(event: OmegaEvent): void {
    for (const handler of this.eventHandlers) {
      try {
        handler(event);
      } catch (err) {
        this.log('Event handler error:', err);
      }
    }
  }

  /** Emit bus event */
  private emitBusEvent(topic: string, data: unknown): void {
    if (!this.busClient) return;

    this.busClient.publish({
      topic,
      kind: 'state',
      level: 'info',
      actor: 'omega-automaton',
      data,
    }).catch((err) => this.log('Bus publish error:', err));
  }

  /** Log helper */
  private log(...args: unknown[]): void {
    if (this.config.debug) {
      console.log('[Omega]', ...args);
    }
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/**
 * Create an omega automaton with WebLLM integration
 */
export function createOmegaAutomaton(
  config: OmegaAutomataConfig = {}
): OmegaAutomaton {
  return new OmegaAutomaton(config);
}

/**
 * Create omega automaton for Dialogos dual-mind conversations
 */
export function createDialogosOmega(
  onInference: (prompt: string) => Promise<string>,
  debug = false
): OmegaAutomaton {
  return new OmegaAutomaton({
    initialState: 'IDLE',
    acceptingStates: ['CONVERSING', 'SYNTHESIZING'],
    acceptanceType: 'buchi',
    enableBusEvents: true,
    onInferenceRequest: async (prompt) => onInference(prompt),
    debug,
    thresholds: {
      idleToReflecting: 8000,  // More patient for dialogos
      reflectingToConversing: 2000,
      conversingSynthesisCheck: 15000,
      stallDetection: 25000,
    },
  });
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Check if automaton satisfies Buchi acceptance condition
 * (i.e., visits accepting states infinitely often)
 */
export function checkBuchiAcceptance(
  context: OmegaContext,
  acceptingStates: OmegaState[]
): boolean {
  // For finite observation, we check if we've visited accepting states "enough"
  // In true omega automata, this would be checked over infinite runs
  return context.acceptingVisits > 0 && acceptingStates.includes(context.currentState);
}

/**
 * Calculate conversation momentum
 * Higher values indicate active, flowing conversation
 */
export function calculateMomentum(context: OmegaContext): number {
  const turnWeight = Math.min(context.turnCount / 10, 1);
  const activityWeight = Math.max(0, 1 - context.stateTimeMs / 30000);
  const synthesisWeight = Math.min(context.synthesisBuffer.length / 5, 1);
  const stallPenalty = Math.max(0, 1 - context.stallCount * 0.1);

  return (turnWeight + activityWeight + synthesisWeight) * stallPenalty / 3;
}

/**
 * Get recommended action based on current state and context
 */
export function getRecommendedAction(context: OmegaContext): string {
  switch (context.currentState) {
    case 'IDLE':
      return context.stateTimeMs > 3000
        ? 'Consider initiating reflection or awaiting input'
        : 'Awaiting input or time-based transition';
    case 'REFLECTING':
      return 'Processing internal reflection, will transition to conversation soon';
    case 'CONVERSING':
      return context.synthesisBuffer.length >= 3
        ? 'Ready for synthesis of accumulated insights'
        : 'Continue conversation, gathering synthesis points';
    case 'SYNTHESIZING':
      return 'Integrating insights, will return to conversation';
    default:
      return 'Unknown state';
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  OmegaAutomaton,
  createOmegaAutomaton,
  createDialogosOmega,
  checkBuchiAcceptance,
  calculateMomentum,
  getRecommendedAction,
  CORE_TRANSITIONS,
  DEFAULT_THRESHOLDS,
  DEFAULT_ACCEPTING_STATES,
  REFLECTION_PROMPTS,
  SYNTHESIS_TEMPLATES,
};
