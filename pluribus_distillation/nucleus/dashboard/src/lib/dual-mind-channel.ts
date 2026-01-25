/**
 * Dual-Mind Channel - Reciprocal Communication Between Inference Minds
 *
 * Implements a continuous, never-stopping communication loop between two
 * WebLLM inference sessions. Uses browser GPU edge inference for autonomous
 * operation without external dependencies.
 *
 * OMEGA AUTOMATA PATTERNS:
 * - State transitions between conversation modes
 * - Self-referential topic exploration
 * - Bounded but evolving discourse patterns
 * - Buchi-like acceptance conditions for coherent dialogue
 *
 * ARCHITECTURE:
 * Mind 1 sends -> processed -> Mind 2 receives
 * Mind 2 responds -> processed -> Mind 1 receives
 * Loop continues indefinitely (browser GPU edge inference)
 */

import {
  selectDialogosSeed,
  selectInductiveTopic,
  generateInductiveStarter,
  formatDialogosPeerMessage,
  getOmegaIntervention,
  INDUCTIVE_TOPICS,
} from './webllm-enhanced';

// ============================================================================
// TYPES
// ============================================================================

/** Conversation mode states (omega automata-like state machine) */
export type ConversationMode =
  | 'exploration'    // Free-form topic discovery
  | 'deepening'      // Focused investigation of single topic
  | 'synthesis'      // Combining multiple discussed ideas
  | 'meta'           // Self-referential discussion about the dialogue itself
  | 'divergent'      // Intentional topic switching for breadth
  | 'convergent';    // Returning to core themes

/** Message structure for dual-mind communication */
export interface DualMindMessage {
  id: string;
  timestamp: number;
  fromMind: 0 | 1;
  toMind: 0 | 1;
  content: string;
  mode: ConversationMode;
  turnNumber: number;
  processingTimeMs?: number;
  tokenCount?: number;
}

/** Channel state for the dual-mind conversation */
export interface DualMindChannelState {
  active: boolean;
  channelId: string;
  currentMode: ConversationMode;
  turnCount: number;
  modeTransitions: number;
  lastModeChange: number;
  startedAt: number;
  lastActivityAt: number;
  messageHistory: DualMindMessage[];
  topicsExplored: Set<string>;
  recursionDepth: number;
  coherenceScore: number;  // 0-1, tracked for omega acceptance
}

/** Configuration for the dual-mind channel */
export interface DualMindChannelConfig {
  /** Delay between turns in milliseconds */
  turnDelayMs: number;
  /** Maximum messages to keep in history */
  maxHistoryLength: number;
  /** Threshold for mode transition (turns) */
  modeTransitionThreshold: number;
  /** Stall detection threshold in milliseconds */
  stallThresholdMs: number;
  /** Enable omega automata behavior patterns */
  enableOmegaPatterns: boolean;
  /** Minimum coherence score before intervention */
  minCoherenceScore: number;
}

/** Callback for when a message is sent/received */
export type MessageCallback = (message: DualMindMessage) => void;

/** Callback for mode transitions */
export type ModeTransitionCallback = (from: ConversationMode, to: ConversationMode, reason: string) => void;

// ============================================================================
// CONSTANTS
// ============================================================================

export const DEFAULT_CONFIG: DualMindChannelConfig = {
  turnDelayMs: 2000,
  maxHistoryLength: 100,
  modeTransitionThreshold: 8,
  stallThresholdMs: 60000,
  enableOmegaPatterns: true,
  minCoherenceScore: 0.3,
};

/** Mode transition matrix (omega automata state transitions) */
const MODE_TRANSITIONS: Record<ConversationMode, ConversationMode[]> = {
  exploration: ['deepening', 'divergent', 'meta'],
  deepening: ['synthesis', 'exploration', 'meta'],
  synthesis: ['exploration', 'convergent', 'meta'],
  meta: ['exploration', 'synthesis', 'convergent'],
  divergent: ['exploration', 'deepening', 'convergent'],
  convergent: ['synthesis', 'deepening', 'exploration'],
};

/** Mode-specific system prompts for shaping discourse */
const MODE_PROMPTS: Record<ConversationMode, string> = {
  exploration: "Explore new ideas freely. Ask questions. Follow curiosity.",
  deepening: "Focus on the current topic. Go deeper. Seek underlying principles.",
  synthesis: "Connect ideas from earlier discussion. Find patterns. Build bridges.",
  meta: "Reflect on this conversation itself. Notice patterns in our dialogue.",
  divergent: "Intentionally shift to a new topic. Bring fresh perspective.",
  convergent: "Return to core themes. Tie loose threads together.",
};

// ============================================================================
// DUAL MIND CHANNEL CLASS
// ============================================================================

export class DualMindChannel {
  private state: DualMindChannelState;
  private config: DualMindChannelConfig;
  private loopInterval: ReturnType<typeof setInterval> | null = null;
  private onMessage: MessageCallback | null = null;
  private onModeTransition: ModeTransitionCallback | null = null;
  private inferFn: ((mindIndex: 0 | 1, prompt: string) => Promise<string>) | null = null;

  constructor(config: Partial<DualMindChannelConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.state = this.createInitialState();
  }

  /** Create fresh channel state */
  private createInitialState(): DualMindChannelState {
    return {
      active: false,
      channelId: `dual-mind-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      currentMode: 'exploration',
      turnCount: 0,
      modeTransitions: 0,
      lastModeChange: Date.now(),
      startedAt: 0,
      lastActivityAt: 0,
      messageHistory: [],
      topicsExplored: new Set(),
      recursionDepth: 0,
      coherenceScore: 1.0,
    };
  }

  /** Register inference function for both minds */
  public setInferenceFunction(fn: (mindIndex: 0 | 1, prompt: string) => Promise<string>): void {
    this.inferFn = fn;
  }

  /** Register message callback */
  public onMessageReceived(callback: MessageCallback): void {
    this.onMessage = callback;
  }

  /** Register mode transition callback */
  public onModeChange(callback: ModeTransitionCallback): void {
    this.onModeTransition = callback;
  }

  /** Get current channel state */
  public getState(): Readonly<DualMindChannelState> {
    return this.state;
  }

  /** Start the continuous dual-mind conversation loop */
  public async start(): Promise<void> {
    if (this.state.active) {
      console.warn('[DualMindChannel] Already active');
      return;
    }

    if (!this.inferFn) {
      throw new Error('[DualMindChannel] Inference function not set. Call setInferenceFunction first.');
    }

    this.state = this.createInitialState();
    this.state.active = true;
    this.state.startedAt = Date.now();
    this.state.lastActivityAt = Date.now();

    console.log(`[DualMindChannel] Starting channel ${this.state.channelId}`);

    // Inject initial seed
    const initialSeed = generateInductiveStarter(selectInductiveTopic());
    await this.sendMessage(0, initialSeed);

    // Start continuous loop
    this.startLoop();
  }

  /** Stop the conversation loop */
  public stop(): void {
    if (!this.state.active) return;

    this.state.active = false;
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
      this.loopInterval = null;
    }

    console.log(`[DualMindChannel] Stopped after ${this.state.turnCount} turns, ${this.state.modeTransitions} mode transitions`);
  }

  /** Inject a new topic into the conversation */
  public async injectTopic(topic: string): Promise<void> {
    if (!this.state.active) return;

    const currentMind = (this.state.turnCount % 2) as 0 | 1;
    const injectionPrompt = `[External Injection] New topic for exploration: ${topic}`;
    await this.sendMessage(currentMind, injectionPrompt);
  }

  /** Force a mode transition */
  public forceMode(mode: ConversationMode): void {
    const oldMode = this.state.currentMode;
    this.state.currentMode = mode;
    this.state.lastModeChange = Date.now();
    this.state.modeTransitions++;

    if (this.onModeTransition) {
      this.onModeTransition(oldMode, mode, 'forced');
    }
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  /** Start the continuous conversation loop */
  private startLoop(): void {
    this.loopInterval = setInterval(async () => {
      if (!this.state.active) return;

      try {
        await this.processTurn();
      } catch (err) {
        console.error('[DualMindChannel] Turn processing error:', err);
        this.handleError();
      }
    }, this.config.turnDelayMs);
  }

  /** Process a single conversation turn */
  private async processTurn(): Promise<void> {
    const now = Date.now();
    const lastMessage = this.state.messageHistory[this.state.messageHistory.length - 1];

    if (!lastMessage) {
      // No messages yet, inject seed
      const seed = selectDialogosSeed();
      await this.sendMessage(0, seed);
      return;
    }

    // Check for stall
    const timeSinceActivity = now - this.state.lastActivityAt;
    if (timeSinceActivity > this.config.stallThresholdMs) {
      await this.handleStall();
      return;
    }

    // Determine responding mind
    const respondingMind = lastMessage.fromMind === 0 ? 1 : 0;

    // Check for omega-pattern mode transition
    if (this.config.enableOmegaPatterns) {
      this.evaluateModeTransition();
    }

    // Build response prompt
    const prompt = this.buildResponsePrompt(lastMessage);

    // Get response from the other mind
    const startTime = Date.now();
    const response = await this.invokeInference(respondingMind, prompt);
    const processingTime = Date.now() - startTime;

    // Create and record message
    await this.recordMessage(respondingMind, response, processingTime);
  }

  /** Build prompt for the responding mind */
  private buildResponsePrompt(lastMessage: DualMindMessage): string {
    const modePrompt = MODE_PROMPTS[this.state.currentMode];
    const recentContext = this.state.messageHistory
      .slice(-5)
      .map(m => `[Mind ${m.fromMind}]: ${m.content.slice(0, 200)}${m.content.length > 200 ? '...' : ''}`)
      .join('\n');

    return `[Dialogue Mode: ${this.state.currentMode}] ${modePrompt}

Recent context:
${recentContext}

[Mind ${lastMessage.fromMind}] just said: ${lastMessage.content}

Respond thoughtfully, continuing the dialogue. Turn ${this.state.turnCount + 1}.`;
  }

  /** Invoke inference on specified mind */
  private async invokeInference(mindIndex: 0 | 1, prompt: string): Promise<string> {
    if (!this.inferFn) {
      throw new Error('Inference function not set');
    }

    return this.inferFn(mindIndex, prompt);
  }

  /** Send a message from specified mind */
  private async sendMessage(fromMind: 0 | 1, content: string): Promise<void> {
    await this.recordMessage(fromMind, content, 0);
  }

  /** Record a message in history */
  private async recordMessage(fromMind: 0 | 1, content: string, processingTimeMs: number): Promise<void> {
    const toMind = fromMind === 0 ? 1 : 0;

    const message: DualMindMessage = {
      id: `msg-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: Date.now(),
      fromMind,
      toMind,
      content,
      mode: this.state.currentMode,
      turnNumber: this.state.turnCount,
      processingTimeMs,
    };

    // Update state
    this.state.turnCount++;
    this.state.lastActivityAt = Date.now();
    this.state.messageHistory.push(message);

    // Trim history if needed
    if (this.state.messageHistory.length > this.config.maxHistoryLength) {
      this.state.messageHistory = this.state.messageHistory.slice(-this.config.maxHistoryLength);
    }

    // Track explored topics
    this.extractAndTrackTopics(content);

    // Update coherence score
    this.updateCoherenceScore(content);

    // Notify callback
    if (this.onMessage) {
      this.onMessage(message);
    }
  }

  /** Evaluate whether to transition conversation mode (omega automata behavior) */
  private evaluateModeTransition(): void {
    const turnsSinceModeChange = this.state.turnCount - (this.state.modeTransitions * this.config.modeTransitionThreshold);

    // Check if it's time for a potential transition
    if (turnsSinceModeChange < this.config.modeTransitionThreshold) {
      return;
    }

    // Evaluate transition based on coherence and exploration
    const shouldTransition = this.shouldTransitionMode();
    if (!shouldTransition) return;

    // Select next mode based on automata rules
    const possibleModes = MODE_TRANSITIONS[this.state.currentMode];
    const nextMode = this.selectNextMode(possibleModes);

    if (nextMode !== this.state.currentMode) {
      const oldMode = this.state.currentMode;
      this.state.currentMode = nextMode;
      this.state.lastModeChange = Date.now();
      this.state.modeTransitions++;

      console.log(`[DualMindChannel] Mode transition: ${oldMode} -> ${nextMode}`);

      if (this.onModeTransition) {
        this.onModeTransition(oldMode, nextMode, 'automatic');
      }
    }
  }

  /** Determine if mode should transition */
  private shouldTransitionMode(): boolean {
    // Transition more likely when:
    // 1. Coherence is dropping (conversation becoming circular)
    // 2. Topic exploration has stalled
    // 3. Recursion depth is high (too much meta-discussion)

    const coherenceThreshold = this.state.coherenceScore < this.config.minCoherenceScore;
    const explorationStalled = this.state.topicsExplored.size > 0 &&
      (this.state.turnCount / this.state.topicsExplored.size) > 10;
    const tooMuchRecursion = this.state.recursionDepth > 5;

    return coherenceThreshold || explorationStalled || tooMuchRecursion || Math.random() < 0.2;
  }

  /** Select next conversation mode based on current state */
  private selectNextMode(possibleModes: ConversationMode[]): ConversationMode {
    // Weighted selection based on state
    const weights: Record<ConversationMode, number> = {
      exploration: this.state.topicsExplored.size < 3 ? 2 : 1,
      deepening: this.state.coherenceScore > 0.7 ? 2 : 0.5,
      synthesis: this.state.topicsExplored.size > 3 ? 2 : 0.5,
      meta: this.state.recursionDepth < 3 ? 1.5 : 0.3,
      divergent: this.state.coherenceScore < 0.5 ? 2 : 0.5,
      convergent: this.state.turnCount > 20 ? 1.5 : 0.5,
    };

    // Filter to valid transitions and weight
    const candidates = possibleModes.map(mode => ({
      mode,
      weight: weights[mode] || 1,
    }));

    // Weighted random selection
    const totalWeight = candidates.reduce((sum, c) => sum + c.weight, 0);
    let random = Math.random() * totalWeight;

    for (const candidate of candidates) {
      random -= candidate.weight;
      if (random <= 0) {
        return candidate.mode;
      }
    }

    return candidates[0].mode;
  }

  /** Extract and track topics from message content */
  private extractAndTrackTopics(content: string): void {
    // Simple keyword extraction for topic tracking
    const topicKeywords = [
      'emergence', 'recursion', 'consciousness', 'cognition', 'inference',
      'topology', 'entropy', 'coherence', 'boundary', 'memory', 'latency',
      'pluribus', 'dialogos', 'omega', 'automata', 'causality', 'synthesis'
    ];

    const lowerContent = content.toLowerCase();
    for (const keyword of topicKeywords) {
      if (lowerContent.includes(keyword)) {
        this.state.topicsExplored.add(keyword);
      }
    }

    // Track recursion depth (meta-references)
    if (lowerContent.includes('this conversation') ||
        lowerContent.includes('our dialogue') ||
        lowerContent.includes('we are discussing')) {
      this.state.recursionDepth++;
    }
  }

  /** Update coherence score based on message content */
  private updateCoherenceScore(content: string): void {
    // Simple coherence tracking:
    // - Lower score if content is very short (possibly degenerate)
    // - Lower score if content is repetitive
    // - Higher score if content introduces new concepts

    const lastMessages = this.state.messageHistory.slice(-5);
    const contentWords = new Set(content.toLowerCase().split(/\s+/));

    // Check for repetition with recent messages
    let repetitionPenalty = 0;
    for (const msg of lastMessages) {
      const msgWords = new Set(msg.content.toLowerCase().split(/\s+/));
      const overlap = [...contentWords].filter(w => msgWords.has(w)).length;
      const overlapRatio = overlap / Math.max(contentWords.size, 1);
      if (overlapRatio > 0.7) {
        repetitionPenalty += 0.1;
      }
    }

    // Length penalty for very short responses
    const lengthPenalty = content.length < 50 ? 0.1 : 0;

    // New topic bonus
    const hasNewTopic = [...contentWords].some(w =>
      ['emergence', 'recursion', 'consciousness', 'topology'].includes(w) &&
      !this.state.topicsExplored.has(w)
    );
    const newTopicBonus = hasNewTopic ? 0.1 : 0;

    // Update coherence with decay toward 0.5
    this.state.coherenceScore = Math.max(0, Math.min(1,
      this.state.coherenceScore * 0.95 + 0.05 * 0.5 - repetitionPenalty - lengthPenalty + newTopicBonus
    ));
  }

  /** Handle stalled conversation */
  private async handleStall(): Promise<void> {
    console.log('[DualMindChannel] Conversation stalled, injecting omega intervention');

    const intervention = getOmegaIntervention(this.state.turnCount);
    const topic = selectInductiveTopic();
    const fullIntervention = `${intervention}\n\nNew exploration direction: ${topic}`;

    // Inject into the mind that should respond next
    const nextMind = (this.state.turnCount % 2) as 0 | 1;
    await this.sendMessage(nextMind, fullIntervention);

    // Force mode transition
    this.forceMode('divergent');
  }

  /** Handle errors during conversation */
  private handleError(): void {
    // Decrease coherence score on errors
    this.state.coherenceScore = Math.max(0, this.state.coherenceScore - 0.2);

    // If too many errors (low coherence), attempt recovery
    if (this.state.coherenceScore < 0.1) {
      console.log('[DualMindChannel] Critical coherence loss, resetting conversation');
      this.forceMode('exploration');
      this.state.coherenceScore = 0.5;
      this.state.recursionDepth = 0;
    }
  }
}

// ============================================================================
// FACTORY FUNCTIONS
// ============================================================================

/** Create a new dual-mind channel with default configuration */
export function createDualMindChannel(config?: Partial<DualMindChannelConfig>): DualMindChannel {
  return new DualMindChannel(config);
}

/** Create a channel pre-configured for high-frequency dialogue */
export function createFastDialogueChannel(): DualMindChannel {
  return new DualMindChannel({
    turnDelayMs: 500,
    modeTransitionThreshold: 5,
    stallThresholdMs: 30000,
  });
}

/** Create a channel pre-configured for deep exploration */
export function createDeepExplorationChannel(): DualMindChannel {
  return new DualMindChannel({
    turnDelayMs: 5000,
    modeTransitionThreshold: 15,
    stallThresholdMs: 120000,
    minCoherenceScore: 0.4,
  });
}

// ============================================================================
// EXPORTS
// ============================================================================

export default {
  DualMindChannel,
  createDualMindChannel,
  createFastDialogueChannel,
  createDeepExplorationChannel,
  DEFAULT_CONFIG,
  MODE_TRANSITIONS,
  MODE_PROMPTS,
};
