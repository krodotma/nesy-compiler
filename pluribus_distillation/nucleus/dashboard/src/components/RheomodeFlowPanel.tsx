/**
 * RheomodeFlowPanel - Conversational Flow Visualization
 *
 * Implements David Bohm's Rheomode concept for language and thought:
 * - Language as flowing movement (rheo = flow)
 * - Verbs as primary, nouns as secondary
 * - Thought as process, not static entities
 *
 * Features:
 * - Flow mode state machine: exploration -> deepening -> synthesis -> meta
 * - Coherence tracking (0-1 gauge)
 * - Turn history with fragment-style reveals
 * - Omega intervention indicators
 * - Integration with dual-mind-channel patterns
 */

import {
  component$,
  useSignal,
  useComputed$,
  useVisibleTask$,
  $,
  type QRL,
} from '@builder.io/qwik';
import type {
  ConversationMode,
  DualMindMessage,
  DualMindChannelState,
} from '../lib/dual-mind-channel';
import { MODE_TRANSITIONS } from '../lib/dual-mind-channel';

// ============================================================================
// TYPES
// ============================================================================

/** Rheomode flow event for bus emission */
export interface RheomodeEvent {
  type: 'mode_transition' | 'coherence_shift' | 'omega_intervention' | 'turn_complete';
  timestamp: number;
  mode: ConversationMode;
  coherence: number;
  turnNumber: number;
  data?: Record<string, unknown>;
}

/** Props for RheomodeFlowPanel */
export interface RheomodeFlowPanelProps {
  /** Current channel state from DualMindChannel */
  channelState?: DualMindChannelState;
  /** Message history for turn visualization */
  messages?: DualMindMessage[];
  /** Callback for mode changes */
  onModeChange$?: QRL<(mode: ConversationMode) => void>;
  /** Callback for omega intervention */
  onOmegaIntervention$?: QRL<() => void>;
  /** Panel height class */
  heightClass?: string;
  /** Enable animated flow lines */
  enableFlowAnimation?: boolean;
}

// ============================================================================
// CONSTANTS
// ============================================================================

/** Mode configuration with Rheomode semantics */
const RHEOMODE_MODES: Record<ConversationMode, {
  label: string;
  verb: string;  // Rheomode emphasizes verbs over nouns
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
}> = {
  exploration: {
    label: 'Exploring',
    verb: 'to-explore',
    color: 'text-cyan-400',
    bgColor: 'bg-cyan-950/40',
    borderColor: 'border-cyan-600/50',
    description: 'Free-flowing inquiry into new territory',
  },
  deepening: {
    label: 'Deepening',
    verb: 'to-deepen',
    color: 'text-emerald-400',
    bgColor: 'bg-emerald-950/40',
    borderColor: 'border-emerald-600/50',
    description: 'Concentrated attention on singular thread',
  },
  synthesis: {
    label: 'Synthesizing',
    verb: 'to-synthesize',
    color: 'text-purple-400',
    bgColor: 'bg-purple-950/40',
    borderColor: 'border-purple-600/50',
    description: 'Weaving disparate streams into coherent whole',
  },
  meta: {
    label: 'Meta-Reflecting',
    verb: 'to-reflect',
    color: 'text-amber-400',
    bgColor: 'bg-amber-950/40',
    borderColor: 'border-amber-600/50',
    description: 'Dialogue observing itself in motion',
  },
  divergent: {
    label: 'Diverging',
    verb: 'to-diverge',
    color: 'text-rose-400',
    bgColor: 'bg-rose-950/40',
    borderColor: 'border-rose-600/50',
    description: 'Intentional branching for breadth',
  },
  convergent: {
    label: 'Converging',
    verb: 'to-converge',
    color: 'text-blue-400',
    bgColor: 'bg-blue-950/40',
    borderColor: 'border-blue-600/50',
    description: 'Returning to core themes',
  },
};

/** Flow mode sequence for state machine visualization */
const MODE_SEQUENCE: ConversationMode[] = [
  'exploration',
  'deepening',
  'synthesis',
  'meta',
  'divergent',
  'convergent',
];

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

/**
 * Coherence Gauge - circular gauge showing conversation coherence 0-1
 */
const CoherenceGauge = component$<{ value: number; size?: number }>(
  ({ value, size = 80 }) => {
    const percentage = Math.max(0, Math.min(1, value)) * 100;
    const circumference = 2 * Math.PI * 35;
    const strokeDashoffset = circumference - (percentage / 100) * circumference;

    // Color transitions: red -> amber -> green
    const color = value < 0.3
      ? '#ef4444'
      : value < 0.6
        ? '#f59e0b'
        : '#10b981';

    const glowColor = value < 0.3
      ? 'rgba(239, 68, 68, 0.3)'
      : value < 0.6
        ? 'rgba(245, 158, 11, 0.3)'
        : 'rgba(16, 185, 129, 0.3)';

    return (
      <div class="relative flex items-center justify-center" style={{ width: `${size}px`, height: `${size}px` }}>
        <svg
          width={size}
          height={size}
          class="transform -rotate-90"
        >
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r="35"
            fill="none"
            stroke="rgba(255,255,255,0.1)"
            stroke-width="6"
          />
          {/* Progress circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r="35"
            fill="none"
            stroke={color}
            stroke-width="6"
            stroke-linecap="round"
            stroke-dasharray={circumference}
            stroke-dashoffset={strokeDashoffset}
            style={{
                                      filter: `drop-shadow(0 0 6px var(--glow-color))`,
                                      '--glow-color': glowColor,              transition: 'stroke-dashoffset 0.5s ease-out, stroke 0.5s ease-out',
            }}
          />
        </svg>
        <div class="absolute inset-0 flex flex-col items-center justify-center">
          <span class="text-lg font-bold" style={{ color }}>
            {(value * 100).toFixed(0)}
          </span>
          <span class="text-[9px] text-zinc-500 uppercase tracking-wider">
            coherence
          </span>
        </div>
      </div>
    );
  }
);

/**
 * Mode State Machine - visualizes flow mode transitions
 */
const ModeStateMachine = component$<{
  currentMode: ConversationMode;
  modeTransitions: number;
  onModeSelect$?: QRL<(mode: ConversationMode) => void>;
}>(({ currentMode, modeTransitions, onModeSelect$ }) => {
  return (
    <div class="space-y-2">
      <div class="flex items-center justify-between text-xs text-zinc-500">
        <span>Flow State</span>
        <span class="text-zinc-600">{modeTransitions} transitions</span>
      </div>
      <div class="flex flex-wrap gap-1">
        {MODE_SEQUENCE.map((mode) => {
          const config = RHEOMODE_MODES[mode];
          const isCurrent = mode === currentMode;
          const canTransition = MODE_TRANSITIONS[currentMode]?.includes(mode);

          return (
            <button
              key={mode}
              onClick$={() => onModeSelect$?.(mode)}
              disabled={!canTransition && !isCurrent}
              class={[
                'px-2 py-1 text-xs rounded-md border transition-all duration-300',
                isCurrent
                  ? `${config.bgColor} ${config.color} ${config.borderColor} font-semibold scale-105 shadow-lg`
                  : canTransition
                    ? 'bg-zinc-800/50 text-zinc-400 border-zinc-700 hover:bg-zinc-700/50 cursor-pointer'
                    : 'bg-zinc-900/30 text-zinc-600 border-zinc-800/50 cursor-not-allowed opacity-50',
              ].join(' ')}
              title={config.description}
            >
              {config.label}
            </button>
          );
        })}
      </div>
      {/* Rheomode verb display */}
      <div class="flex items-center gap-2 mt-2">
        <span class="text-[10px] text-zinc-600 uppercase tracking-wider">Rheomode:</span>
        <span class={`text-sm font-mono italic ${RHEOMODE_MODES[currentMode].color}`}>
          {RHEOMODE_MODES[currentMode].verb}
        </span>
      </div>
    </div>
  );
});

/**
 * Turn History Fragment - displays message fragments with reveal animation
 */
const TurnHistoryFragment = component$<{
  message: DualMindMessage;
  index: number;
  isLatest: boolean;
}>(({ message, index, isLatest }) => {
  const maxPreviewLength = 80;
  const preview = message.content.length > maxPreviewLength
    ? message.content.slice(0, maxPreviewLength) + '...'
    : message.content;

  const modeConfig = RHEOMODE_MODES[message.mode];
  const mindLabel = message.fromMind === 0 ? 'M0' : 'M1';
  const mindColor = message.fromMind === 0 ? 'text-cyan-400' : 'text-purple-400';

  return (
    <div
      class={[
        'p-2 rounded border transition-all duration-500',
        isLatest
          ? 'bg-zinc-800/60 border-zinc-600 shadow-md'
          : 'bg-zinc-900/40 border-zinc-800/50',
        'animate-in fade-in slide-in-from-bottom-2',
      ].join(' ')}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div class="flex items-center justify-between gap-2 mb-1">
        <div class="flex items-center gap-2">
          <span class={`text-[10px] font-mono font-bold ${mindColor}`}>
            {mindLabel}
          </span>
          <span class={`text-[9px] px-1.5 py-0.5 rounded ${modeConfig.bgColor} ${modeConfig.color}`}>
            {modeConfig.label}
          </span>
        </div>
        <div class="flex items-center gap-2 text-[9px] text-zinc-600">
          <span>T{message.turnNumber}</span>
          {message.processingTimeMs && message.processingTimeMs > 0 && (
            <span>{message.processingTimeMs}ms</span>
          )}
        </div>
      </div>
      <p class="text-xs text-zinc-300 leading-relaxed">
        {preview}
      </p>
    </div>
  );
});

/**
 * Omega Intervention Indicator
 */
const OmegaIndicator = component$<{
  active: boolean;
  stallCount: number;
  onIntervene$?: QRL<() => void>;
}>(({ active, stallCount, onIntervene$ }) => {
  return (
    <div
      class={[
        'p-3 rounded-lg border transition-all duration-300',
        active
          ? 'bg-amber-950/50 border-amber-500/50 animate-pulse'
          : 'bg-zinc-900/30 border-zinc-800',
      ].join(' ')}
    >
      <div class="flex items-center justify-between">
        <div class="flex items-center gap-2">
          <div
            class={[
              'w-2 h-2 rounded-full transition-colors',
              active ? 'bg-amber-400 shadow-amber-400/50 shadow-lg' : 'bg-zinc-600',
            ].join(' ')}
          />
          <span class="text-xs font-semibold text-zinc-300">
            Omega Protocol
          </span>
        </div>
        <button
          onClick$={() => onIntervene$?.()}
          class={[
            'px-2 py-1 text-[10px] rounded border transition-all',
            active
              ? 'bg-amber-600/20 text-amber-300 border-amber-500/50 hover:bg-amber-600/30'
              : 'bg-zinc-800 text-zinc-500 border-zinc-700 hover:bg-zinc-700',
          ].join(' ')}
        >
          Intervene
        </button>
      </div>
      <div class="mt-2 flex items-center justify-between text-[10px]">
        <span class="text-zinc-500">
          {active ? 'Intervention active' : 'Monitoring flow'}
        </span>
        {stallCount > 0 && (
          <span class="text-amber-400/80">
            {stallCount} stall{stallCount !== 1 ? 's' : ''} detected
          </span>
        )}
      </div>
    </div>
  );
});

/**
 * Flow Lines Animation - visualizes the flowing nature of dialogue
 */
const FlowLinesCanvas = component$<{
  coherence: number;
  mode: ConversationMode;
  active: boolean;
}>(({ coherence, mode, active }) => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const animationFrame = useSignal<number>(0);

  useVisibleTask$(({ track, cleanup }) => {
    track(() => coherence);
    track(() => mode);
    track(() => active);

    const canvas = canvasRef.value;
    if (!canvas || !active) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);

    const modeConfig = RHEOMODE_MODES[mode];
    const baseColor = modeConfig.color.includes('cyan') ? 'rgba(34, 211, 238, '
      : modeConfig.color.includes('emerald') ? 'rgba(52, 211, 153, '
      : modeConfig.color.includes('purple') ? 'rgba(167, 139, 250, '
      : modeConfig.color.includes('amber') ? 'rgba(251, 191, 36, '
      : modeConfig.color.includes('rose') ? 'rgba(251, 113, 133, '
      : 'rgba(96, 165, 250, ';

    let time = 0;
    const animate = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      // Draw flowing lines
      const lineCount = Math.floor(3 + coherence * 5);
      for (let i = 0; i < lineCount; i++) {
        ctx.beginPath();
        ctx.strokeStyle = `${baseColor}${0.1 + coherence * 0.3})`;
        ctx.lineWidth = 1 + coherence * 2;

        const yOffset = (height / (lineCount + 1)) * (i + 1);
        const amplitude = 10 + coherence * 20;
        const frequency = 0.02 + i * 0.005;
        const phase = time + i * 0.5;

        ctx.moveTo(0, yOffset);
        for (let x = 0; x < width; x += 4) {
          const y = yOffset + Math.sin(x * frequency + phase) * amplitude;
          ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animationFrame.value = requestAnimationFrame(animate);
    };

    animate();

    cleanup(() => {
      if (animationFrame.value) {
        cancelAnimationFrame(animationFrame.value);
      }
    });
  });

  return (
    <canvas
      ref={canvasRef}
      class="w-full h-12 rounded bg-black/20"
      style={{ opacity: active ? 1 : 0.3 }}
    />
  );
});

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const RheomodeFlowPanel = component$<RheomodeFlowPanelProps>(({
  channelState,
  messages = [],
  onModeChange$,
  onOmegaIntervention$,
  heightClass = 'h-[600px]',
  enableFlowAnimation = true,
}) => {
  const localCoherence = useSignal(1.0);
  const localMode = useSignal<ConversationMode>('exploration');
  const localTurnCount = useSignal(0);
  const localTransitions = useSignal(0);
  const omegaActive = useSignal(false);
  const stallCount = useSignal(0);

  // Sync with channel state
  useVisibleTask$(({ track }) => {
    if (channelState) {
      track(() => channelState);
      localCoherence.value = channelState.coherenceScore;
      localMode.value = channelState.currentMode;
      localTurnCount.value = channelState.turnCount;
      localTransitions.value = channelState.modeTransitions;

      // Detect omega activation from coherence drop
      if (channelState.coherenceScore < 0.3) {
        omegaActive.value = true;
      }
    }
  });

  // Compute displayed messages
  const displayMessages = useComputed$(() => {
    const source = channelState?.messageHistory || messages;
    return source.slice(-8).reverse();
  });

  // Handle mode selection
  const handleModeSelect = $((mode: ConversationMode) => {
    localMode.value = mode;
    localTransitions.value++;
    onModeChange$?.(mode);
  });

  // Handle omega intervention
  const handleOmegaIntervention = $(() => {
    omegaActive.value = true;
    stallCount.value++;
    onOmegaIntervention$?.();
  });

  const currentModeConfig = RHEOMODE_MODES[localMode.value];
  const isActive = !!channelState?.active || messages.length > 0;

  return (
    <div class={`flex flex-col ${heightClass} bg-zinc-950 rounded-lg border border-zinc-800 overflow-hidden`}>
      {/* Header */}
      <div class="px-4 py-3 border-b border-zinc-800 flex items-center justify-between">
        <div class="flex items-center gap-3">
          <h2 class="text-sm font-bold text-zinc-100">Rheomode Flow</h2>
          <span class={`text-[10px] px-2 py-0.5 rounded ${currentModeConfig.bgColor} ${currentModeConfig.color}`}>
            {currentModeConfig.verb}
          </span>
        </div>
        <div class="flex items-center gap-3 text-[10px] text-zinc-500">
          <span>T:{localTurnCount.value}</span>
          <span>M:{localTransitions.value}</span>
          <div
            class={[
              'w-2 h-2 rounded-full transition-colors',
              isActive ? 'bg-emerald-400 animate-pulse' : 'bg-zinc-600',
            ].join(' ')}
          />
        </div>
      </div>

      {/* Flow Animation */}
      {enableFlowAnimation && (
        <div class="px-4 py-2 border-b border-zinc-800/50">
          <FlowLinesCanvas
            coherence={localCoherence.value}
            mode={localMode.value}
            active={isActive}
          />
        </div>
      )}

      {/* Main Content Grid */}
      <div class="flex-1 grid grid-cols-12 gap-4 p-4 overflow-hidden">
        {/* Left Column: Coherence & Omega */}
        <div class="col-span-4 space-y-4 flex flex-col">
          {/* Coherence Gauge */}
          <div class="flex-shrink-0 p-4 rounded-lg border border-zinc-800 bg-zinc-900/30 flex flex-col items-center">
            <CoherenceGauge value={localCoherence.value} size={100} />
            <p class="mt-3 text-[10px] text-zinc-500 text-center leading-relaxed">
              {currentModeConfig.description}
            </p>
          </div>

          {/* Mode State Machine */}
          <div class="flex-shrink-0 p-3 rounded-lg border border-zinc-800 bg-zinc-900/30">
            <ModeStateMachine
              currentMode={localMode.value}
              modeTransitions={localTransitions.value}
              onModeSelect$={handleModeSelect}
            />
          </div>

          {/* Omega Indicator */}
          <div class="flex-shrink-0">
            <OmegaIndicator
              active={omegaActive.value}
              stallCount={stallCount.value}
              onIntervene$={handleOmegaIntervention}
            />
          </div>
        </div>

        {/* Right Column: Turn History */}
        <div class="col-span-8 flex flex-col min-h-0">
          <div class="flex items-center justify-between mb-2">
            <span class="text-xs font-semibold text-zinc-400">Turn History</span>
            <span class="text-[10px] text-zinc-600">
              {displayMessages.value.length} of {channelState?.messageHistory?.length || messages.length}
            </span>
          </div>
          <div class="flex-1 overflow-y-auto space-y-2 pr-2">
            {displayMessages.value.length > 0 ? (
              displayMessages.value.map((msg, i) => (
                <TurnHistoryFragment
                  key={msg.id}
                  message={msg}
                  index={i}
                  isLatest={i === 0}
                />
              ))
            ) : (
              <div class="flex items-center justify-center h-32 text-zinc-600 text-sm">
                <div class="text-center">
                  <p class="mb-1">Awaiting dialogue flow...</p>
                  <p class="text-[10px] text-zinc-700">
                    The rheomode unfolds through conversation
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Footer: Rheomode Philosophy Note */}
      <div class="px-4 py-2 border-t border-zinc-800/50 bg-zinc-900/20">
        <p class="text-[9px] text-zinc-600 italic text-center">
          "Thought is not a static entity but a flowing movement" - David Bohm
        </p>
      </div>
    </div>
  );
});

export default RheomodeFlowPanel;
