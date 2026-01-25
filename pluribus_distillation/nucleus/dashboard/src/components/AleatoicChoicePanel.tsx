/**
 * AleatoicChoicePanel.tsx
 * =======================
 *
 * Human-in-the-loop decision component for branching/stochastic choices.
 *
 * Use cases:
 * 1. System has multiple valid options and needs human input
 * 2. Random/stochastic selection could apply but HITL is preferred
 * 3. User needs to make a branching decision
 *
 * Features:
 * - Card-based option display with visual previews
 * - Confidence scores visualization
 * - "Let the system decide" surrender option
 * - Bus event emission on selection
 * - Deadline/timeout support
 * - Undo capability within grace period
 */

import { component$, useSignal, useStore, useVisibleTask$, $, type QRL } from '@builder.io/qwik';

// ============================================================================
// Types
// ============================================================================

export type OptionKind = 'action' | 'branch' | 'parameter' | 'strategy' | 'agent' | 'artifact';

export interface ChoiceOption {
  id: string;
  label: string;
  description?: string;
  kind?: OptionKind;
  confidence?: number;        // 0.0 - 1.0
  preview?: {
    type: 'text' | 'code' | 'image' | 'diff' | 'json';
    content: string;
  };
  metadata?: Record<string, unknown>;
  recommended?: boolean;      // System recommendation
  risky?: boolean;            // High-risk option warning
  reversible?: boolean;       // Can this choice be undone?
}

export interface AleatoicChoiceRequest {
  id: string;
  topic: string;
  actor: string;
  question: string;
  context?: string;
  options: ChoiceOption[];
  defaultOptionId?: string;   // Used if system decides or timeout
  deadlineIso?: string;       // When auto-selection kicks in
  gracePeriodMs?: number;     // Undo window after selection
  aleatoryMode?: 'weighted' | 'uniform' | 'best'; // How system would decide
}

export interface AleatoicChoiceResult {
  requestId: string;
  selectedOptionId: string;
  selectionMode: 'human' | 'system' | 'timeout';
  timestamp: string;
  undoAvailable: boolean;
}

export interface AleatoicChoicePanelProps {
  request: AleatoicChoiceRequest;
  onSelect$?: QRL<(result: AleatoicChoiceResult) => void>;
  onUndo$?: QRL<(requestId: string) => void>;
  compact?: boolean;
  showConfidence?: boolean;
}

// ============================================================================
// Helpers
// ============================================================================

const KIND_STYLES: Record<OptionKind, { bg: string; border: string; icon: string }> = {
  action: { bg: 'bg-blue-500/10', border: 'border-blue-500/30', icon: 'A' },
  branch: { bg: 'bg-purple-500/10', border: 'border-purple-500/30', icon: 'B' },
  parameter: { bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', icon: 'P' },
  strategy: { bg: 'bg-orange-500/10', border: 'border-orange-500/30', icon: 'S' },
  agent: { bg: 'bg-green-500/10', border: 'border-green-500/30', icon: '@' },
  artifact: { bg: 'bg-pink-500/10', border: 'border-pink-500/30', icon: '*' },
};

function formatTimeRemaining(deadlineIso: string): string {
  const now = Date.now();
  const deadline = Date.parse(deadlineIso);
  if (!Number.isFinite(deadline)) return '';

  const diffMs = deadline - now;
  if (diffMs <= 0) return 'expired';

  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);

  if (diffSec < 60) return `${diffSec}s`;
  if (diffMin < 60) return `${diffMin}m ${diffSec % 60}s`;
  return `${Math.floor(diffMin / 60)}h ${diffMin % 60}m`;
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return 'text-green-400';
  if (confidence >= 0.6) return 'text-cyan-400';
  if (confidence >= 0.4) return 'text-yellow-400';
  return 'text-orange-400';
}

function getConfidenceBar(confidence: number): string {
  const filled = Math.round(confidence * 10);
  const empty = 10 - filled;
  return '\u2588'.repeat(filled) + '\u2591'.repeat(empty);
}

// ============================================================================
// Sub-components
// ============================================================================

interface OptionCardProps {
  option: ChoiceOption;
  selected: boolean;
  showConfidence: boolean;
  onSelect$: QRL<(id: string) => void>;
  compact: boolean;
}

const OptionCard = component$<OptionCardProps>(({
  option,
  selected,
  showConfidence,
  onSelect$,
  compact,
}) => {
  const kind = option.kind || 'action';
  const kindStyle = KIND_STYLES[kind];

  const baseClasses = `
    group relative rounded-lg border transition-all duration-200 cursor-pointer
    ${selected
      ? 'border-primary bg-primary/10 shadow-lg shadow-primary/20 ring-2 ring-primary/30'
      : `${kindStyle.border} ${kindStyle.bg} hover:border-primary/50 hover:shadow-md`}
    ${compact ? 'p-3' : 'p-4'}
    ${option.risky ? 'border-l-2 border-l-red-500' : ''}
    ${option.recommended ? 'border-l-2 border-l-green-500' : ''}
  `;

  return (
    <div
      class={baseClasses}
      onClick$={() => onSelect$(option.id)}
    >
      {/* Header: Kind badge + Label */}
      <div class="flex items-center gap-2 mb-2">
        <span
          class={`w-5 h-5 rounded flex items-center justify-center text-[10px] font-bold ${kindStyle.bg} ${kindStyle.border} border`}
          title={kind}
        >
          {kindStyle.icon}
        </span>
        <span class={`font-medium flex-1 ${compact ? 'text-sm' : ''}`}>
          {option.label}
        </span>
        {option.recommended && (
          <span class="text-[9px] px-1.5 py-0.5 rounded bg-green-500/20 text-green-400 border border-green-500/30">
            REC
          </span>
        )}
        {option.risky && (
          <span class="text-[9px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30">
            RISK
          </span>
        )}
      </div>

      {/* Description */}
      {option.description && !compact && (
        <p class="text-sm text-muted-foreground mb-2 line-clamp-2">
          {option.description}
        </p>
      )}

      {/* Confidence bar */}
      {showConfidence && option.confidence !== undefined && (
        <div class="flex items-center gap-2 text-xs mb-2">
          <span class="text-muted-foreground">conf:</span>
          <span class={`font-mono ${getConfidenceColor(option.confidence)}`}>
            {getConfidenceBar(option.confidence)}
          </span>
          <span class={`${getConfidenceColor(option.confidence)}`}>
            {(option.confidence * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Preview */}
      {option.preview && !compact && (
        <div class="mt-2 p-2 rounded bg-black/20 border border-[var(--glass-border-subtle)] overflow-hidden">
          {option.preview.type === 'code' && (
            <pre class="text-[10px] font-mono text-cyan-300 whitespace-pre-wrap line-clamp-4">
              {option.preview.content}
            </pre>
          )}
          {option.preview.type === 'text' && (
            <p class="text-[11px] text-muted-foreground line-clamp-3">
              {option.preview.content}
            </p>
          )}
          {option.preview.type === 'json' && (
            <pre class="text-[10px] font-mono text-purple-300 whitespace-pre-wrap line-clamp-4">
              {option.preview.content}
            </pre>
          )}
          {option.preview.type === 'diff' && (
            <pre class="text-[10px] font-mono whitespace-pre-wrap line-clamp-4">
              {option.preview.content.split('\n').map((line, i) => (
                <span
                  key={i}
                  class={
                    line.startsWith('+') ? 'text-green-400' :
                    line.startsWith('-') ? 'text-red-400' :
                    line.startsWith('@') ? 'text-cyan-400' :
                    'text-muted-foreground'
                  }
                >
                  {line}{'\n'}
                </span>
              ))}
            </pre>
          )}
        </div>
      )}

      {/* Selection indicator */}
      {selected && (
        <div class="absolute top-2 right-2">
          <span class="text-primary text-lg">&#10003;</span>
        </div>
      )}

      {/* Reversible indicator */}
      {option.reversible === false && (
        <div class="absolute bottom-2 right-2">
          <span class="text-[9px] text-orange-400/60" title="Irreversible">
            &#9888;
          </span>
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

export const AleatoicChoicePanel = component$<AleatoicChoicePanelProps>(({
  request,
  onSelect$,
  onUndo$,
  compact = false,
  showConfidence = true,
}) => {
  const selectedId = useSignal<string | null>(null);
  const confirmed = useSignal(false);
  const undoAvailable = useSignal(false);
  const timeRemaining = useSignal<string>('');

  const state = useStore({
    selectionMode: 'human' as 'human' | 'system' | 'timeout',
    gracePeriodActive: false,
    graceEndTime: 0,
  });

  // Update countdown timer
  useVisibleTask$(({ track, cleanup }) => {
    track(() => request.deadlineIso);

    if (!request.deadlineIso) return;

    const interval = setInterval(() => {
      const remaining = formatTimeRemaining(request.deadlineIso!);
      timeRemaining.value = remaining;

      // Auto-select on timeout if not confirmed
      if (remaining === 'expired' && !confirmed.value) {
        handleSystemDecision();
      }
    }, 1000);

    cleanup(() => clearInterval(interval));
  });

  // Grace period timer
  useVisibleTask$(({ track, cleanup }) => {
    track(() => state.gracePeriodActive);

    if (!state.gracePeriodActive) return;

    const interval = setInterval(() => {
      if (Date.now() >= state.graceEndTime) {
        state.gracePeriodActive = false;
        undoAvailable.value = false;
      }
    }, 100);

    cleanup(() => clearInterval(interval));
  });

  const handleOptionSelect = $((optionId: string) => {
    if (confirmed.value) return;
    selectedId.value = optionId;
  });

  const handleConfirm = $(async () => {
    if (!selectedId.value || confirmed.value) return;

    confirmed.value = true;
    state.selectionMode = 'human';

    const gracePeriod = request.gracePeriodMs || 5000;
    state.gracePeriodActive = true;
    state.graceEndTime = Date.now() + gracePeriod;
    undoAvailable.value = true;

    const result: AleatoicChoiceResult = {
      requestId: request.id,
      selectedOptionId: selectedId.value,
      selectionMode: 'human',
      timestamp: new Date().toISOString(),
      undoAvailable: true,
    };

    // Emit bus event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('pluribus:bus:emit', {
        detail: {
          topic: 'aleatoric.choice.selected',
          kind: 'response',
          level: 'info',
          actor: 'dashboard',
          data: result,
        },
      }));
    }

    if (onSelect$) {
      await onSelect$(result);
    }
  });

  const handleSystemDecision = $(async () => {
    if (confirmed.value) return;

    // Select based on aleatory mode
    let chosenId: string;
    const opts = request.options;

    if (request.defaultOptionId) {
      chosenId = request.defaultOptionId;
    } else if (request.aleatoryMode === 'best') {
      // Pick highest confidence
      const sorted = [...opts].sort((a, b) => (b.confidence || 0) - (a.confidence || 0));
      chosenId = sorted[0]?.id || opts[0]?.id;
    } else if (request.aleatoryMode === 'weighted') {
      // Weighted random by confidence
      const total = opts.reduce((sum, o) => sum + (o.confidence || 0.5), 0);
      let rand = Math.random() * total;
      chosenId = opts[0]?.id;
      for (const opt of opts) {
        rand -= (opt.confidence || 0.5);
        if (rand <= 0) {
          chosenId = opt.id;
          break;
        }
      }
    } else {
      // Uniform random
      chosenId = opts[Math.floor(Math.random() * opts.length)]?.id;
    }

    selectedId.value = chosenId;
    confirmed.value = true;
    state.selectionMode = request.deadlineIso && formatTimeRemaining(request.deadlineIso) === 'expired'
      ? 'timeout'
      : 'system';

    const result: AleatoicChoiceResult = {
      requestId: request.id,
      selectedOptionId: chosenId,
      selectionMode: state.selectionMode,
      timestamp: new Date().toISOString(),
      undoAvailable: false,
    };

    // Emit bus event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('pluribus:bus:emit', {
        detail: {
          topic: 'aleatoric.choice.surrendered',
          kind: 'response',
          level: 'info',
          actor: 'dashboard',
          data: result,
        },
      }));
    }

    if (onSelect$) {
      await onSelect$(result);
    }
  });

  const handleUndo = $(async () => {
    if (!state.gracePeriodActive) return;

    confirmed.value = false;
    state.gracePeriodActive = false;
    undoAvailable.value = false;

    // Emit bus event
    if (typeof window !== 'undefined') {
      window.dispatchEvent(new CustomEvent('pluribus:bus:emit', {
        detail: {
          topic: 'aleatoric.choice.undone',
          kind: 'response',
          level: 'info',
          actor: 'dashboard',
          data: { requestId: request.id },
        },
      }));
    }

    if (onUndo$) {
      await onUndo$(request.id);
    }
  });

  // Grid columns based on option count
  const gridCols = request.options.length <= 2
    ? 'grid-cols-1 md:grid-cols-2'
    : request.options.length <= 4
      ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-4'
      : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5';

  return (
    <div class="glass-panel rounded-lg border border-border overflow-hidden">
      {/* Header */}
      <div class="p-4 border-b border-border bg-muted/20">
        <div class="flex items-center justify-between gap-4">
          <div class="flex items-center gap-3 min-w-0">
            <span class="text-2xl">&#9878;</span>
            <div class="min-w-0">
              <h3 class="font-semibold text-sm truncate">
                Aleatoric Choice Required
              </h3>
              <p class="text-xs text-muted-foreground truncate">
                @{request.actor} &middot; {request.topic}
              </p>
            </div>
          </div>

          {/* Status badges */}
          <div class="flex items-center gap-2 flex-shrink-0">
            {request.deadlineIso && timeRemaining.value && (
              <span
                class={`text-xs px-2 py-1 rounded border ${
                  timeRemaining.value === 'expired'
                    ? 'bg-red-500/20 text-red-400 border-red-500/30'
                    : 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30'
                }`}
              >
                {timeRemaining.value}
              </span>
            )}
            {confirmed.value && (
              <span
                class={`text-xs px-2 py-1 rounded border ${
                  state.selectionMode === 'human'
                    ? 'bg-green-500/20 text-green-400 border-green-500/30'
                    : state.selectionMode === 'timeout'
                      ? 'bg-orange-500/20 text-orange-400 border-orange-500/30'
                      : 'bg-purple-500/20 text-purple-400 border-purple-500/30'
                }`}
              >
                {state.selectionMode === 'human' ? 'SELECTED' :
                 state.selectionMode === 'timeout' ? 'TIMEOUT' : 'SYSTEM'}
              </span>
            )}
          </div>
        </div>

        {/* Question */}
        <p class="mt-3 text-sm">
          {request.question}
        </p>

        {/* Context */}
        {request.context && (
          <p class="mt-2 text-xs text-muted-foreground p-2 rounded bg-black/20 border border-[var(--glass-border-subtle)]">
            {request.context}
          </p>
        )}
      </div>

      {/* Options grid */}
      <div class={`p-4 grid ${gridCols} gap-3`}>
        {request.options.map((opt) => (
          <OptionCard
            key={opt.id}
            option={opt}
            selected={selectedId.value === opt.id}
            showConfidence={showConfidence}
            onSelect$={handleOptionSelect}
            compact={compact}
          />
        ))}
      </div>

      {/* Actions footer */}
      <div class="p-4 border-t border-border bg-muted/10 flex items-center justify-between gap-4">
        <div class="flex items-center gap-3">
          {/* System decide button */}
          <button
            class={`text-xs px-3 py-1.5 rounded border transition-colors
              ${confirmed.value
                ? 'bg-muted/30 text-muted-foreground border-border cursor-not-allowed'
                : 'bg-purple-500/10 text-purple-400 border-purple-500/30 hover:bg-purple-500/20'
              }`}
            onClick$={handleSystemDecision}
            disabled={confirmed.value}
            title={`Let system decide (${request.aleatoryMode || 'uniform'} selection)`}
          >
            &#9861; Let System Decide
          </button>

          {/* Aleatory mode indicator */}
          <span class="text-[10px] text-muted-foreground">
            mode: {request.aleatoryMode || 'uniform'}
          </span>
        </div>

        <div class="flex items-center gap-3">
          {/* Undo button */}
          {undoAvailable.value && state.gracePeriodActive && (
            <button
              class="text-xs px-3 py-1.5 rounded border bg-orange-500/10 text-orange-400 border-orange-500/30 hover:bg-orange-500/20 transition-colors"
              onClick$={handleUndo}
            >
              &#8634; Undo
            </button>
          )}

          {/* Confirm button */}
          <button
            class={`text-sm px-4 py-2 rounded font-medium transition-colors
              ${!selectedId.value || confirmed.value
                ? 'bg-muted/30 text-muted-foreground border border-border cursor-not-allowed'
                : 'bg-primary text-primary-foreground hover:bg-primary/90'
              }`}
            onClick$={handleConfirm}
            disabled={!selectedId.value || confirmed.value}
          >
            {confirmed.value ? 'Confirmed' : 'Confirm Selection'}
          </button>
        </div>
      </div>

      {/* Grace period indicator */}
      {state.gracePeriodActive && (
        <div class="h-1 bg-muted overflow-hidden">
          <div
            class="h-full bg-primary transition-all duration-100"
            style={{
              width: `${Math.max(0, ((state.graceEndTime - Date.now()) / (request.gracePeriodMs || 5000)) * 100)}%`,
            }}
          />
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Convenience wrapper for multiple pending choices
// ============================================================================

export interface AleatoicChoiceQueueProps {
  requests: AleatoicChoiceRequest[];
  onSelect$?: QRL<(result: AleatoicChoiceResult) => void>;
  onUndo$?: QRL<(requestId: string) => void>;
}

export const AleatoicChoiceQueue = component$<AleatoicChoiceQueueProps>(({
  requests,
  onSelect$,
  onUndo$,
}) => {
  if (requests.length === 0) {
    return (
      <div class="glass-panel p-6 text-center text-muted-foreground">
        <span class="text-2xl mb-2 block">&#9878;</span>
        <p class="text-sm">No pending choices</p>
      </div>
    );
  }

  return (
    <div class="space-y-4">
      <div class="flex items-center justify-between px-1">
        <h2 class="text-sm font-semibold text-muted-foreground">
          PENDING CHOICES
        </h2>
        <span class="text-xs px-2 py-0.5 rounded bg-primary/20 text-primary">
          {requests.length}
        </span>
      </div>

      {requests.map((req) => (
        <AleatoicChoicePanel
          key={req.id}
          request={req}
          onSelect$={onSelect$}
          onUndo$={onUndo$}
        />
      ))}
    </div>
  );
});

export default AleatoicChoicePanel;
