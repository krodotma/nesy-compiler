/**
 * PortalIngressSelector.tsx
 *
 * Portal Ingress Mode Selection UI Component
 *
 * When content enters PORTAL, the user must choose one of three paths:
 * 1. Actualized-Mode (AM) - Light path, direct integration into world model
 * 2. Shadowmode (SM) - Dark path, deferred processing in hysteresis buffer
 * 3. Auto - System decides based on confidence score
 *
 * Integrates with MetabolicForkView from portal_contract_v1.md.
 *
 * @see nucleus/specs/portal_contract_v1.md
 * @see nucleus/plans/portal_a2ui_implementation.md
 */

import { component$, useSignal, useComputed$, $, useVisibleTask$ } from '@builder.io/qwik';
import { createBusClient } from '../../lib/bus/bus-client';

// ============================================================================
// Types
// ============================================================================

/** Ingress mode for content entering PORTAL */
export type IngressMode = 'AM' | 'SM' | 'AUTO';

/** Confidence assessment from the system */
export interface ConfidenceScore {
  /** Overall confidence 0-1 */
  overall: number;
  /** Clarity of semantic meaning */
  semantic_clarity: number;
  /** Match with existing world model */
  model_alignment: number;
  /** Source credibility */
  source_trust: number;
  /** Noise floor assessment */
  noise_floor: number;
}

/** Destination path preview */
export interface DestinationPath {
  mode: IngressMode;
  label: string;
  description: string;
  destination: string;
  eta_ms: number;
}

/** Fragment metadata for ingress */
export interface IngressFragment {
  id: string;
  content_preview: string;
  source_type: 'url' | 'file' | 'text' | 'arxiv' | 'github';
  source_uri?: string;
  texture_density?: number;
  byte_size: number;
  created_iso: string;
  asset_id?: string;
}

/** Props for the PortalIngressSelector component */
export interface PortalIngressSelectorProps {
  /** The fragment being ingested */
  fragment: IngressFragment;
  /** Pre-computed confidence score (optional - will fetch if not provided) */
  confidence?: ConfidenceScore;
  /** Callback when mode is selected */
  onModeSelected$?: (mode: IngressMode, fragment: IngressFragment) => void;
  /** Callback when selection is cancelled */
  onCancel$?: () => void;
  /** Custom class for the container */
  class?: string;
  /** Enable auto-selection mode */
  autoSelectEnabled?: boolean;
  /** Auto-selection threshold (confidence above this auto-selects AM) */
  autoSelectThreshold?: number;
}

// ============================================================================
// Constants
// ============================================================================

const MODE_CONFIG: Record<IngressMode, {
  label: string;
  icon: string;
  color: string;
  bgColor: string;
  borderColor: string;
  description: string;
  destination: string;
}> = {
  AM: {
    label: 'Actualized-Mode',
    icon: 'sun',
    color: 'text-amber-300',
    bgColor: 'bg-amber-500/10',
    borderColor: 'border-amber-500/30',
    description: 'Direct integration. The fragment becomes part of the actualized world model immediately.',
    destination: 'World Model -> Entelexis Layer -> Knowledge Graph',
  },
  SM: {
    label: 'Shadowmode',
    icon: 'moon',
    color: 'text-purple-300',
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    description: 'Deferred processing. The fragment enters the hysteresis buffer for later consideration.',
    destination: 'Hysteresis Buffer -> Shadow Potentials -> Review Queue',
  },
  AUTO: {
    label: 'Auto',
    icon: 'zap',
    color: 'text-cyan-300',
    bgColor: 'bg-cyan-500/10',
    borderColor: 'border-cyan-500/30',
    description: 'Let the system decide based on confidence score and texture density.',
    destination: 'Confidence Analysis -> Metabolic Fork -> Dynamic Path',
  },
};

// ============================================================================
// Helper Functions
// ============================================================================

function confidenceToPercent(value: number): number {
  return Math.round(Math.max(0, Math.min(1, value)) * 100);
}

function confidenceColor(value: number): string {
  if (value >= 0.8) return 'text-green-400';
  if (value >= 0.6) return 'text-cyan-400';
  if (value >= 0.4) return 'text-yellow-400';
  return 'text-red-400';
}

function confidenceBarColor(value: number): string {
  if (value >= 0.8) return 'bg-green-400';
  if (value >= 0.6) return 'bg-cyan-400';
  if (value >= 0.4) return 'bg-yellow-400';
  return 'bg-red-400';
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function sourceTypeIcon(type: string): string {
  switch (type) {
    case 'url': return 'link';
    case 'file': return 'file';
    case 'arxiv': return 'book-open';
    case 'github': return 'github';
    default: return 'file-text';
  }
}

function generateDefaultConfidence(): ConfidenceScore {
  return {
    overall: 0.65,
    semantic_clarity: 0.7,
    model_alignment: 0.6,
    source_trust: 0.7,
    noise_floor: 0.25,
  };
}

// ============================================================================
// Sub-Components
// ============================================================================

interface ConfidenceDisplayProps {
  score: ConfidenceScore;
}

const ConfidenceDisplay = component$<ConfidenceDisplayProps>(({ score }) => {
  const metrics = [
    { key: 'semantic_clarity', label: 'Semantic Clarity', value: score.semantic_clarity },
    { key: 'model_alignment', label: 'Model Alignment', value: score.model_alignment },
    { key: 'source_trust', label: 'Source Trust', value: score.source_trust },
    { key: 'noise_floor', label: 'Noise Floor', value: score.noise_floor, invert: true },
  ];

  return (
    <div class="space-y-3">
      {/* Overall Score */}
      <div class="flex items-center justify-between">
        <span class="text-sm text-muted-foreground">Overall Confidence</span>
        <span class={`text-lg font-bold mono ${confidenceColor(score.overall)}`}>
          {confidenceToPercent(score.overall)}%
        </span>
      </div>

      {/* Progress bar for overall */}
      <div class="h-2 bg-muted/30 rounded-full overflow-hidden">
        <div
          class={`h-full transition-all duration-500 ${confidenceBarColor(score.overall)}`}
          style={{ width: `${confidenceToPercent(score.overall)}%` }}
        />
      </div>

      {/* Individual metrics */}
      <div class="grid grid-cols-2 gap-2 mt-4">
        {metrics.map((m) => {
          const displayValue = m.invert ? 1 - m.value : m.value;
          return (
            <div
              key={m.key}
              class="px-2 py-1.5 rounded border border-border/50 bg-muted/20"
            >
              <div class="flex items-center justify-between text-[10px]">
                <span class="text-muted-foreground">{m.label}</span>
                <span class={`mono ${confidenceColor(displayValue)}`}>
                  {confidenceToPercent(displayValue)}%
                </span>
              </div>
              <div class="h-1 bg-muted/30 rounded-full mt-1 overflow-hidden">
                <div
                  class={`h-full ${confidenceBarColor(displayValue)}`}
                  style={{ width: `${confidenceToPercent(displayValue)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});

interface ModeCardProps {
  mode: IngressMode;
  selected: boolean;
  recommended: boolean;
  onSelect$: () => void;
}

const ModeCard = component$<ModeCardProps>(({ mode, selected, recommended, onSelect$ }) => {
  const config = MODE_CONFIG[mode];

  return (
    <button
      type="button"
      onClick$={onSelect$}
      class={`
        relative flex flex-col p-4 rounded-lg border-2 transition-all duration-200
        hover:scale-[1.02] hover:shadow-lg
        ${selected
          ? `${config.borderColor} ${config.bgColor} ring-2 ring-offset-2 ring-offset-background ring-${config.color.replace('text-', '')}`
          : 'border-border/50 bg-muted/10 hover:border-border'
        }
      `}
      data-testid={`portal-mode-${mode.toLowerCase()}`}
    >
      {/* Recommended badge */}
      {recommended && (
        <div class="absolute -top-2 -right-2 px-2 py-0.5 text-[10px] font-bold rounded bg-green-500/20 text-green-400 border border-green-500/30">
          RECOMMENDED
        </div>
      )}

      {/* Icon and label */}
      <div class="flex items-center gap-3 mb-3">
        <div class={`w-10 h-10 rounded-full flex items-center justify-center ${config.bgColor}`}>
          {mode === 'AM' && (
            <svg class={`w-5 h-5 ${config.color}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <circle cx="12" cy="12" r="5" stroke-width="2" />
              <path stroke-width="2" d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
            </svg>
          )}
          {mode === 'SM' && (
            <svg class={`w-5 h-5 ${config.color}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-width="2" d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z" />
            </svg>
          )}
          {mode === 'AUTO' && (
            <svg class={`w-5 h-5 ${config.color}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          )}
        </div>
        <div>
          <h3 class={`font-semibold ${config.color}`}>{config.label}</h3>
          <span class="text-[10px] text-muted-foreground mono">({mode})</span>
        </div>
      </div>

      {/* Description */}
      <p class="text-xs text-muted-foreground mb-3 text-left">
        {config.description}
      </p>

      {/* Destination preview */}
      <div class="mt-auto pt-3 border-t border-border/30">
        <div class="flex items-center gap-2 text-[10px]">
          <svg class="w-3 h-3 text-muted-foreground" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-width="2" d="M13 5l7 7-7 7M5 5l7 7-7 7" />
          </svg>
          <span class="text-muted-foreground mono">{config.destination}</span>
        </div>
      </div>

      {/* Selection indicator */}
      {selected && (
        <div class="absolute top-2 left-2">
          <svg class={`w-5 h-5 ${config.color}`} fill="currentColor" viewBox="0 0 24 24">
            <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z" />
          </svg>
        </div>
      )}
    </button>
  );
});

interface MetabolicForkViewProps {
  selectedMode: IngressMode | null;
  confidence: ConfidenceScore;
}

const MetabolicForkView = component$<MetabolicForkViewProps>(({ selectedMode, confidence }) => {
  const amWidth = useComputed$(() => {
    if (selectedMode === 'AM') return 100;
    if (selectedMode === 'SM') return 0;
    return Math.round(confidence.overall * 100);
  });

  const smWidth = useComputed$(() => 100 - amWidth.value);

  return (
    <div class="p-4 rounded-lg border border-border/50 bg-muted/10">
      <h4 class="text-sm font-semibold text-muted-foreground mb-3">
        Metabolic Fork Preview
      </h4>

      {/* Visual fork diagram */}
      <div class="relative h-24">
        {/* Central input */}
        <div class="absolute left-1/2 top-0 -translate-x-1/2 w-8 h-8 rounded-full bg-white/10 border-2 border-[var(--glass-border-hover)] flex items-center justify-center">
          <svg class="w-4 h-4 text-white/70" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
          </svg>
        </div>

        {/* Fork lines */}
        <svg class="absolute inset-0 w-full h-full" viewBox="0 0 200 100" preserveAspectRatio="none">
          {/* AM path */}
          <path
            d="M100,15 Q100,50 40,85"
            fill="none"
            stroke={selectedMode === 'AM' || (selectedMode === 'AUTO' && confidence.overall >= 0.6) ? 'rgb(253,224,71)' : 'rgba(253,224,71,0.2)'}
            stroke-width="2"
            stroke-dasharray={selectedMode === 'AM' ? 'none' : '5,5'}
          />
          {/* SM path */}
          <path
            d="M100,15 Q100,50 160,85"
            fill="none"
            stroke={selectedMode === 'SM' || (selectedMode === 'AUTO' && confidence.overall < 0.6) ? 'rgb(192,132,252)' : 'rgba(192,132,252,0.2)'}
            stroke-width="2"
            stroke-dasharray={selectedMode === 'SM' ? 'none' : '5,5'}
          />
        </svg>

        {/* AM destination */}
        <div class={`
          absolute left-4 bottom-0 px-3 py-1.5 rounded border text-xs
          ${selectedMode === 'AM' || (selectedMode === 'AUTO' && confidence.overall >= 0.6)
            ? 'border-amber-500/50 bg-amber-500/20 text-amber-300'
            : 'border-border/30 bg-muted/10 text-muted-foreground'
          }
        `}>
          <span class="font-semibold">AM</span>
          <span class="text-[10px] ml-1">Entelexis</span>
        </div>

        {/* SM destination */}
        <div class={`
          absolute right-4 bottom-0 px-3 py-1.5 rounded border text-xs
          ${selectedMode === 'SM' || (selectedMode === 'AUTO' && confidence.overall < 0.6)
            ? 'border-purple-500/50 bg-purple-500/20 text-purple-300'
            : 'border-border/30 bg-muted/10 text-muted-foreground'
          }
        `}>
          <span class="font-semibold">SM</span>
          <span class="text-[10px] ml-1">Hysteresis</span>
        </div>
      </div>

      {/* Split bar */}
      <div class="mt-4">
        <div class="flex items-center justify-between text-[10px] text-muted-foreground mb-1">
          <span>Actualized</span>
          <span>Shadowed</span>
        </div>
        <div class="h-2 flex rounded-full overflow-hidden">
          <div
            class="bg-amber-400 transition-all duration-500"
            style={{ width: `${amWidth.value}%` }}
          />
          <div
            class="bg-purple-400 transition-all duration-500"
            style={{ width: `${smWidth.value}%` }}
          />
        </div>
        <div class="flex items-center justify-between text-[10px] mt-1">
          <span class="text-amber-400 mono">{amWidth.value}%</span>
          <span class="text-purple-400 mono">{smWidth.value}%</span>
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

export const PortalIngressSelector = component$<PortalIngressSelectorProps>((props) => {
  const {
    fragment,
    confidence: initialConfidence,
    autoSelectEnabled = false,
    autoSelectThreshold = 0.85,
  } = props;

  const selectedMode = useSignal<IngressMode | null>(null);
  const confidence = useSignal<ConfidenceScore>(initialConfidence || generateDefaultConfidence());
  const isSubmitting = useSignal(false);
  const busConnected = useSignal(false);

  // Recommended mode based on confidence
  const recommendedMode = useComputed$<IngressMode>(() => {
    if (confidence.value.overall >= 0.7) return 'AM';
    if (confidence.value.overall <= 0.4) return 'SM';
    return 'AUTO';
  });

  // Connect to bus for event emission
  useVisibleTask$(({ cleanup }) => {
    if (typeof window === 'undefined') return;

    const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/bus`;
    const client = createBusClient({ platform: 'browser', wsUrl });

    client.connect()
      .then(() => {
        busConnected.value = true;
      })
      .catch(() => {
        busConnected.value = false;
      });

    cleanup(() => {
      client.disconnect();
    });

    // Store client for later use
    (window as any).__portalBusClient__ = client;
  });

  // Handle mode selection
  const handleModeSelect = $((mode: IngressMode) => {
    selectedMode.value = mode;
  });

  // Handle confirmation
  const handleConfirm = $(async () => {
    if (!selectedMode.value) return;

    isSubmitting.value = true;

    try {
      // Emit bus event for mode selection
      const event = {
        topic: 'portal.ingress.mode_selected',
        kind: 'action',
        level: 'info' as const,
        actor: 'portal-ingress-selector',
        data: {
          fragment_id: fragment.id,
          mode: selectedMode.value,
          confidence: confidence.value,
          source_type: fragment.source_type,
          source_uri: fragment.source_uri,
          source_ref: fragment.asset_id ? `portal://asset/${fragment.asset_id}` : fragment.source_uri,
          asset_id: fragment.asset_id,
          byte_size: fragment.byte_size,
          selected_at: new Date().toISOString(),
        },
      };

      // Try to publish via fetch API
      await fetch('/api/emit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(event),
      }).catch(() => {/* Best effort */});

      // Call the callback if provided
      if (props.onModeSelected$) {
        props.onModeSelected$(selectedMode.value, fragment);
      }
    } finally {
      isSubmitting.value = false;
    }
  });

  // Handle cancel
  const handleCancel = $(() => {
    if (props.onCancel$) {
      props.onCancel$();
    }
  });

  return (
    <div
      class={`
        p-6 rounded-xl border border-border/50 bg-card/80 backdrop-blur-sm
        shadow-xl max-w-3xl mx-auto
        ${props.class || ''}
      `}
      data-testid="portal-ingress-selector"
    >
      {/* Header */}
      <div class="flex items-center justify-between mb-6">
        <div>
          <h2 class="text-lg font-bold text-foreground">Portal Ingress</h2>
          <p class="text-sm text-muted-foreground">
            Select integration mode for incoming content
          </p>
        </div>
        <div class="flex items-center gap-2">
          <span class={`w-2 h-2 rounded-full ${busConnected.value ? 'bg-green-400' : 'bg-red-400'}`} />
          <span class="text-[10px] text-muted-foreground">
            {busConnected.value ? 'BUS CONNECTED' : 'BUS OFFLINE'}
          </span>
        </div>
      </div>

      {/* Fragment preview */}
      <div class="mb-6 p-4 rounded-lg border border-border/30 bg-muted/10">
        <div class="flex items-start gap-3">
          <div class="w-10 h-10 rounded-lg bg-muted/30 flex items-center justify-center">
            {fragment.source_type === 'url' && (
              <svg class="w-5 h-5 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101" />
                <path stroke-width="2" d="M10.172 13.828a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
            )}
            {fragment.source_type === 'file' && (
              <svg class="w-5 h-5 text-purple-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            )}
            {fragment.source_type === 'arxiv' && (
              <svg class="w-5 h-5 text-amber-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
            )}
            {fragment.source_type === 'github' && (
              <svg class="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
            )}
            {fragment.source_type === 'text' && (
              <svg class="w-5 h-5 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-width="2" d="M4 6h16M4 12h16m-7 6h7" />
              </svg>
            )}
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-center gap-2">
              <span class="text-[10px] px-1.5 py-0.5 rounded bg-muted/40 text-muted-foreground uppercase">
                {fragment.source_type}
              </span>
              <span class="text-[10px] text-muted-foreground mono">
                {formatBytes(fragment.byte_size)}
              </span>
              {fragment.texture_density !== undefined && (
                <span class="text-[10px] text-muted-foreground">
                  density: {(fragment.texture_density * 100).toFixed(1)}%
                </span>
              )}
            </div>
            <p class="text-sm text-foreground mt-1 line-clamp-2">
              {fragment.content_preview}
            </p>
            {fragment.source_uri && (
              <p class="text-[10px] text-muted-foreground mono mt-1 truncate">
                {fragment.source_uri}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Two-column layout for confidence and fork view */}
      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Confidence display */}
        <div class="p-4 rounded-lg border border-border/50 bg-muted/10">
          <h4 class="text-sm font-semibold text-muted-foreground mb-3">
            Confidence Analysis
          </h4>
          <ConfidenceDisplay score={confidence.value} />
        </div>

        {/* Metabolic fork view */}
        <MetabolicForkView
          selectedMode={selectedMode.value}
          confidence={confidence.value}
        />
      </div>

      {/* Mode selection */}
      <div class="mb-6">
        <h4 class="text-sm font-semibold text-muted-foreground mb-3">
          Select Integration Mode
        </h4>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
          <ModeCard
            mode="AM"
            selected={selectedMode.value === 'AM'}
            recommended={recommendedMode.value === 'AM'}
            onSelect$={() => handleModeSelect('AM')}
          />
          <ModeCard
            mode="SM"
            selected={selectedMode.value === 'SM'}
            recommended={recommendedMode.value === 'SM'}
            onSelect$={() => handleModeSelect('SM')}
          />
          <ModeCard
            mode="AUTO"
            selected={selectedMode.value === 'AUTO'}
            recommended={recommendedMode.value === 'AUTO'}
            onSelect$={() => handleModeSelect('AUTO')}
          />
        </div>
      </div>

      {/* Auto-select info */}
      {autoSelectEnabled && (
        <div class="mb-6 p-3 rounded border border-cyan-500/30 bg-cyan-500/10">
          <div class="flex items-center gap-2 text-sm text-cyan-300">
            <svg class="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>
              Auto-select enabled: Confidence above {(autoSelectThreshold * 100).toFixed(0)}% will auto-route to AM
            </span>
          </div>
        </div>
      )}

      {/* Actions */}
      <div class="flex items-center justify-end gap-3">
        <button
          type="button"
          onClick$={handleCancel}
          class="px-4 py-2 text-sm rounded border border-border hover:bg-muted/30 transition-colors"
        >
          Cancel
        </button>
        <button
          type="button"
          onClick$={handleConfirm}
          disabled={!selectedMode.value || isSubmitting.value}
          class={`
            px-6 py-2 text-sm font-semibold rounded transition-all
            ${selectedMode.value
              ? 'bg-primary text-primary-foreground hover:opacity-90'
              : 'bg-muted/30 text-muted-foreground cursor-not-allowed'
            }
          `}
          data-testid="portal-confirm-button"
        >
          {isSubmitting.value ? (
            <span class="flex items-center gap-2">
              <svg class="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" />
                <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              Processing...
            </span>
          ) : (
            <>Confirm {selectedMode.value || 'Selection'}</>
          )}
        </button>
      </div>
    </div>
  );
});

export default PortalIngressSelector;
