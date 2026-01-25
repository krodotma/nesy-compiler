/**
 * STRpLeadsView - Content Curation Leads Dashboard
 *
 * Displays leads from the MOTD-Curate pipeline grouped by decision.
 * Features:
 * - Tabbed view (promote/defer/reject)
 * - Thumbnail previews
 * - Topic badges and keywords
 * - Action buttons (Watch, Ingest, Archive)
 * - Live bus updates
 */

import {
  component$,
  useSignal,
  useComputed$,
  type Signal,
  type QRL,
  $,
} from '@builder.io/qwik';
import type {
  STRpLead,
  LeadDecision,
  LeadTab,
  LeadTopic,
  LeadAction,
  TransfiguredLead,
  TransfigurationStage,
  MetabolicMode,
  SextetGate,
} from '../../lib/state/leads_types';
import {
  getDecisionColor,
  getTopicIcon,
  groupLeadsByDecision,
  groupLeadsByMetabolic,
  groupLeadsByStage,
  initializeTransfiguration,
  beginSophosDialectic,
  createTransfigurationEvent,
  completeTransfiguration,
  autoGenerateThesis,
  autoGenerateAntithesis,
  autoGenerateSynthesis,
  autoFillPentad,
  evaluateSextetGates,
  LEAD_BUS_TOPICS,
  TRANSFIGURATION_BUS_TOPICS,
} from '../../lib/state/leads_types';

// ============================================================================
// Props Interface
// ============================================================================

interface STRpLeadsViewProps {
  /** Signal containing all leads */
  leads: Signal<STRpLead[]>;
  /** Callback for dispatching actions (returns action ID or void) */
  dispatchAction: QRL<(topic: string, data: Record<string, unknown>) => void | string | Promise<void | string>>;
  /** Connection status */
  connected?: boolean;
}

// ============================================================================
// Sub-components
// ============================================================================

/** Lead thumbnail with fallback */
const LeadThumbnail = component$<{ artifacts: STRpLead['artifacts']; title: string }>(
  ({ artifacts, title }) => {
    const hasThumb = !!artifacts.thumb;
    const hasGif = !!artifacts.gif;

    return (
      <div class="relative w-20 h-20 rounded-lg overflow-hidden bg-muted/50 flex-shrink-0">
        {hasThumb || hasGif ? (
          <img
            src={hasGif ? artifacts.gif : artifacts.thumb}
            alt={title}
            class="w-full h-full object-cover"
            loading="lazy"
          />
        ) : (
          <div class="w-full h-full flex items-center justify-center text-2xl text-muted-foreground">
            📄
          </div>
        )}
        {/* Media type indicators */}
        <div class="absolute bottom-1 right-1 flex gap-0.5">
          {artifacts.clip && (
            <span class="text-[10px] bg-black/60 text-white px-1 rounded">🎬</span>
          )}
          {artifacts.mp3 && (
            <span class="text-[10px] bg-black/60 text-white px-1 rounded">🎵</span>
          )}
        </div>
      </div>
    );
  }
);

/** Keywords display */
const KeywordTags = component$<{ keywords: string[] }>(({ keywords }) => {
  const displayKeywords = keywords.slice(0, 4);
  const remaining = keywords.length - 4;

  return (
    <div class="flex flex-wrap gap-1">
      {displayKeywords.map((kw) => (
        <span
          key={kw}
          class="text-[10px] px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground"
        >
          {kw}
        </span>
      ))}
      {remaining > 0 && (
        <span class="text-[10px] px-1.5 py-0.5 rounded bg-muted/30 text-muted-foreground">
          +{remaining}
        </span>
      )}
    </div>
  );
});

/** Single lead card */
const LeadCard = component$<{
  lead: STRpLead;
  onAction$: QRL<(action: LeadAction, lead: STRpLead) => void>;
}>(({ lead, onAction$ }) => {
  const decisionColor = getDecisionColor(lead.decision);
  const topicIcon = getTopicIcon(lead.topic);

  return (
    <div class="p-4 hover:bg-muted/20 transition-colors border-b border-border/30 last:border-b-0">
      <div class="flex gap-4">
        {/* Thumbnail */}
        <LeadThumbnail artifacts={lead.artifacts} title={lead.title} />

        {/* Content */}
        <div class="flex-1 min-w-0">
          {/* Header row */}
          <div class="flex items-start justify-between gap-2 mb-1">
            <div class="flex-1 min-w-0">
              <a
                href={lead.url}
                target="_blank"
                rel="noopener noreferrer"
                class="font-medium text-primary hover:underline line-clamp-1"
                title={lead.title}
              >
                {lead.title}
              </a>
            </div>
            <div class="flex items-center gap-2 flex-shrink-0">
              {/* Decision badge */}
              <span
                class={`text-xs px-2 py-0.5 rounded border ${decisionColor}`}
              >
                {lead.decision}
              </span>
            </div>
          </div>

          {/* Meta row */}
          <div class="flex items-center gap-2 text-xs text-muted-foreground mb-2">
            <span class="flex items-center gap-1">
              {topicIcon}
              <span>{lead.topic}</span>
            </span>
            <span class="text-muted-foreground/50">|</span>
            <span class="font-mono">{lead.ts.slice(0, 10)}</span>
            {lead.priority && (
              <>
                <span class="text-muted-foreground/50">|</span>
                <span class={lead.priority === 1 ? 'text-red-400' : ''}>
                  P{lead.priority}
                </span>
              </>
            )}
            <span class="text-muted-foreground/50">|</span>
            <span class="font-mono text-[10px]">{lead.actor}</span>
          </div>

          {/* Keywords */}
          {lead.keywords.length > 0 && (
            <div class="mb-2">
              <KeywordTags keywords={lead.keywords} />
            </div>
          )}

          {/* Stage Progression Indicator (Iter 2+3+4) */}
          {(lead as TransfiguredLead).stage && (lead as TransfiguredLead).stage !== 'raw' && (
            <div class="mb-2 space-y-2">
              {/* Flow + Radar row */}
              <div class="flex items-center gap-3">
                <TransfigurationFlowIndicator
                  stage={(lead as TransfiguredLead).stage || 'raw'}
                  sophos={(lead as TransfiguredLead).sophos}
                  sextet={(lead as TransfiguredLead).sextet}
                />
                {(lead as TransfiguredLead).sextet?.compliance_vector && (
                  <SextetRadarChart
                    complianceVector={(lead as TransfiguredLead).sextet?.compliance_vector}
                    size={60}
                  />
                )}
                {/* Bohm tree indicator */}
                <BohmBranchIndicator
                  nodeId={(lead as TransfiguredLead).bohm_node_id}
                  parentId={(lead as TransfiguredLead).parent_lead_id}
                  childIds={(lead as TransfiguredLead).child_lead_ids}
                />
              </div>
              {/* Dialectic display for resolved leads */}
              {(lead as TransfiguredLead).sophos?.resolution_status === 'resolved' && (
                <DialecticDisplay sophos={(lead as TransfiguredLead).sophos} />
              )}
            </div>
          )}

          {/* Notes */}
          {lead.notes && (
            <p class="text-xs text-muted-foreground line-clamp-2 mb-2">
              {lead.notes}
            </p>
          )}

          {/* Actions row */}
          <div class="flex items-center gap-2">
            <button
              onClick$={() => onAction$('watch', lead)}
              class="text-xs px-2 py-1 rounded bg-primary/10 text-primary hover:bg-primary/20 transition-colors flex items-center gap-1"
            >
              <span>👁️</span>
              <span>Watch</span>
            </button>
            <button
              onClick$={() => onAction$('transfigure', lead)}
              class="text-xs px-2 py-1 rounded bg-violet-500/10 text-violet-400 hover:bg-violet-500/20 transition-colors flex items-center gap-1"
              disabled={(lead as TransfiguredLead).stage === 'actualized'}
            >
              <span>⚗️</span>
              <span>{(lead as TransfiguredLead).stage === 'actualized' ? 'Transfigured' : 'Transfigure'}</span>
            </button>
            <button
              onClick$={() => onAction$('archive', lead)}
              class="text-xs px-2 py-1 rounded bg-muted/50 text-muted-foreground hover:bg-muted transition-colors flex items-center gap-1"
              disabled={lead.archived}
            >
              <span>📦</span>
              <span>{lead.archived ? 'Archived' : 'Archive'}</span>
            </button>

            {/* Quick decision buttons */}
            <div class="ml-auto flex items-center gap-1">
              {lead.decision !== 'promote' && (
                <button
                  onClick$={() => onAction$('promote', lead)}
                  class="text-xs px-2 py-1 rounded bg-green-500/10 text-green-400 hover:bg-green-500/20"
                  title="Promote"
                >
                  ▲
                </button>
              )}
              {lead.decision !== 'defer' && (
                <button
                  onClick$={() => onAction$('defer', lead)}
                  class="text-xs px-2 py-1 rounded bg-yellow-500/10 text-yellow-400 hover:bg-yellow-500/20"
                  title="Defer"
                >
                  ⏸
                </button>
              )}
              {lead.decision !== 'reject' && (
                <button
                  onClick$={() => onAction$('reject', lead)}
                  class="text-xs px-2 py-1 rounded bg-red-500/10 text-red-400 hover:bg-red-500/20"
                  title="Reject"
                >
                  ✕
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

// ============================================================================
// Stage Progression Indicator (Iter 2)
// ============================================================================

/** Visual indicator of all 9 transfiguration stages */
const STAGE_CONFIG: Record<TransfigurationStage, { icon: string; color: string; label: string }> = {
  raw: { icon: '○', color: 'gray-400', label: 'Raw' },
  questioning: { icon: '?', color: 'blue-400', label: 'Questioning' },
  synthesized: { icon: '◎', color: 'cyan-400', label: 'Synthesized' },
  instantiating: { icon: '⬡', color: 'teal-400', label: 'Instantiating' },
  validating: { icon: '⚙', color: 'amber-400', label: 'Validating' },
  transfigured: { icon: '⬢', color: 'violet-400', label: 'Transfigured' },
  actualized: { icon: '▲', color: 'emerald-400', label: 'Actualized' },
  shadowed: { icon: '◐', color: 'purple-400', label: 'Shadowed' },
  decayed: { icon: '◇', color: 'red-400', label: 'Decayed' },
};

const STAGE_ORDER: TransfigurationStage[] = [
  'raw', 'questioning', 'synthesized', 'instantiating',
  'validating', 'transfigured', 'actualized', 'shadowed', 'decayed'
];

/** Stage progression indicator component */
const StageProgressionIndicator = component$<{
  currentStage: TransfigurationStage;
  compact?: boolean;
}>(({ currentStage, compact = false }) => {
  const currentIndex = STAGE_ORDER.indexOf(currentStage);

  if (compact) {
    // Compact single-line view for cards
    const config = STAGE_CONFIG[currentStage];
    return (
      <span class={`text-[10px] px-1.5 py-0.5 rounded bg-${config.color}/20 text-${config.color} font-mono flex items-center gap-1`}>
        <span>{config.icon}</span>
        <span>{currentStage}</span>
      </span>
    );
  }

  // Full progression view
  return (
    <div class="flex items-center gap-0.5">
      {STAGE_ORDER.slice(0, 7).map((stage, index) => {
        const config = STAGE_CONFIG[stage];
        const isPast = index < currentIndex;
        const isCurrent = index === currentIndex;
        const isFuture = index > currentIndex;

        return (
          <div key={stage} class="flex items-center">
            <div
              class={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] transition-all ${
                isCurrent
                  ? `bg-${config.color}/30 text-${config.color} ring-2 ring-${config.color}/50`
                  : isPast
                    ? `bg-${config.color}/20 text-${config.color}`
                    : 'bg-muted/30 text-muted-foreground/50'
              }`}
              title={config.label}
            >
              {isPast ? '✓' : config.icon}
            </div>
            {index < 6 && (
              <div
                class={`w-2 h-0.5 ${isPast ? 'bg-emerald-400/50' : 'bg-muted/30'}`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
});

/** Sextet gate status display - inline badges */
const SextetGatesDisplay = component$<{
  gates?: { P: SextetGate; E: SextetGate; L: SextetGate; R: SextetGate; Q: SextetGate; Ω: SextetGate };
}>(({ gates }) => {
  if (!gates) return null;

  const gateOrder = ['P', 'E', 'L', 'R', 'Q', 'Ω'] as const;

  return (
    <div class="flex items-center gap-0.5 font-mono text-[9px]">
      {gateOrder.map((gateKey) => {
        const gate = gates[gateKey];
        const isPassed = gate?.passed === true;
        const isFailed = gate?.passed === false && gate?.score < 0.3;
        const isWarned = gate?.passed === false && gate?.score >= 0.3;

        return (
          <span
            key={gateKey}
            class={`px-1 rounded ${
              isPassed ? 'bg-emerald-500/30 text-emerald-400' :
              isFailed ? 'bg-red-500/30 text-red-400' :
              isWarned ? 'bg-amber-500/30 text-amber-400' :
              'bg-muted/30 text-muted-foreground/50'
            }`}
            title={`${gateKey}: ${gate?.check || 'pending'} (${Math.round((gate?.score || 0) * 100)}%)`}
          >
            {gateKey}{isPassed ? '✓' : isFailed ? '✗' : isWarned ? '!' : '?'}
          </span>
        );
      })}
    </div>
  );
});

/** Sextet gate radar visualization (Iteration 3) */
const SextetRadarChart = component$<{
  complianceVector?: [number, number, number, number, number, number];
  size?: number;
}>(({ complianceVector, size = 80 }) => {
  if (!complianceVector) return null;

  const [P, E, L, R, Q, Ω] = complianceVector;
  const labels = ['P', 'E', 'L', 'R', 'Q', 'Ω'];
  const halfSize = size / 2;
  const maxRadius = halfSize - 10;

  // Calculate hexagonal points for each gate
  const getPoint = (index: number, value: number) => {
    const angle = (Math.PI * 2 * index) / 6 - Math.PI / 2; // Start from top
    const radius = value * maxRadius;
    return {
      x: halfSize + Math.cos(angle) * radius,
      y: halfSize + Math.sin(angle) * radius,
    };
  };

  const points = complianceVector.map((v, i) => getPoint(i, v));
  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`).join(' ') + ' Z';

  // Grid hexagon (at 50% and 100%)
  const gridPath50 = Array.from({ length: 6 }, (_, i) => getPoint(i, 0.5))
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`)
    .join(' ') + ' Z';
  const gridPath100 = Array.from({ length: 6 }, (_, i) => getPoint(i, 1.0))
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x},${p.y}`)
    .join(' ') + ' Z';

  // Average score for color
  const avgScore = complianceVector.reduce((a, b) => a + b, 0) / 6;
  const fillColor = avgScore >= 0.8 ? 'rgba(16, 185, 129, 0.3)' : // emerald
    avgScore >= 0.5 ? 'rgba(245, 158, 11, 0.3)' : // amber
    'rgba(239, 68, 68, 0.3)'; // red
  const strokeColor = avgScore >= 0.8 ? '#10b981' : avgScore >= 0.5 ? '#f59e0b' : '#ef4444';

  return (
    <div class="relative" style={{ width: `${size}px`, height: `${size}px` }}>
      <svg width={size} height={size} class="absolute inset-0">
        {/* Grid lines */}
        <path d={gridPath100} fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="1" />
        <path d={gridPath50} fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1" />

        {/* Axis lines */}
        {Array.from({ length: 6 }, (_, i) => {
          const p = getPoint(i, 1.0);
          return (
            <line
              key={i}
              x1={halfSize}
              y1={halfSize}
              x2={p.x}
              y2={p.y}
              stroke="rgba(255,255,255,0.1)"
              stroke-width="1"
            />
          );
        })}

        {/* Data polygon */}
        <path d={pathD} fill={fillColor} stroke={strokeColor} stroke-width="2" />

        {/* Data points */}
        {points.map((p, i) => (
          <circle
            key={i}
            cx={p.x}
            cy={p.y}
            r="3"
            fill={strokeColor}
          />
        ))}
      </svg>

      {/* Labels */}
      {labels.map((label, i) => {
        const p = getPoint(i, 1.25);
        return (
          <span
            key={i}
            class="absolute text-[8px] font-mono text-muted-foreground"
            style={{
              left: `${p.x}px`,
              top: `${p.y}px`,
              transform: 'translate(-50%, -50%)',
            }}
          >
            {label}
          </span>
        );
      })}

      {/* Center score */}
      <span
        class="absolute text-[10px] font-bold"
        style={{
          left: '50%',
          top: '50%',
          transform: 'translate(-50%, -50%)',
          color: strokeColor,
        }}
      >
        {Math.round(avgScore * 100)}
      </span>
    </div>
  );
});

/** Etymology source badge (Iteration 4) */
const EtymonBadge = component$<{ etymonRef?: string }>(({ etymonRef }) => {
  if (!etymonRef) return null;

  const isASL = etymonRef.startsWith('ASL-');
  const isEtymon = etymonRef.startsWith('etymon:');
  const label = etymonRef.replace('ASL-2025:', '').replace('etymon:', '');

  return (
    <span
      class={`text-[9px] px-1.5 py-0.5 rounded font-mono ${
        isASL ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30' :
        isEtymon ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30' :
        'bg-muted/30 text-muted-foreground'
      }`}
      title={`Grounded in: ${etymonRef}`}
    >
      {isASL ? '⚡' : '◉'} {label}
    </span>
  );
});

/** Dialectic synthesis display (Iteration 4) */
const DialecticDisplay = component$<{
  sophos?: {
    thesis?: { text: string; confidence: number; sources?: string[] };
    antithesis?: { text: string; confidence: number; sources?: string[] };
    synthesis?: { text: string; confidence: number; sources?: string[] };
    resolution_status: string;
  };
}>(({ sophos }) => {
  if (!sophos || sophos.resolution_status === 'pending') return null;

  const getEtymon = (sources?: string[]) => sources?.find(s => s.startsWith('ASL-') || s.startsWith('etymon:'));

  return (
    <div class="space-y-2 text-xs">
      {sophos.thesis && (
        <div class="flex items-start gap-2 p-2 rounded bg-blue-500/10 border-l-2 border-blue-500">
          <span class="text-blue-400 font-bold">T</span>
          <div class="flex-1">
            <p class="text-blue-300/90 line-clamp-2">{sophos.thesis.text}</p>
            <div class="flex items-center gap-2 mt-1">
              <span class="text-[10px] text-blue-400/60">{Math.round(sophos.thesis.confidence * 100)}%</span>
              <EtymonBadge etymonRef={getEtymon(sophos.thesis.sources)} />
            </div>
          </div>
        </div>
      )}
      {sophos.antithesis && (
        <div class="flex items-start gap-2 p-2 rounded bg-red-500/10 border-l-2 border-red-500">
          <span class="text-red-400 font-bold">A</span>
          <div class="flex-1">
            <p class="text-red-300/90 line-clamp-2">{sophos.antithesis.text}</p>
            <div class="flex items-center gap-2 mt-1">
              <span class="text-[10px] text-red-400/60">{Math.round(sophos.antithesis.confidence * 100)}%</span>
              <EtymonBadge etymonRef={getEtymon(sophos.antithesis.sources)} />
            </div>
          </div>
        </div>
      )}
      {sophos.synthesis && (
        <div class="flex items-start gap-2 p-2 rounded bg-emerald-500/10 border-l-2 border-emerald-500">
          <span class="text-emerald-400 font-bold">S</span>
          <div class="flex-1">
            <p class="text-emerald-300/90 line-clamp-2">{sophos.synthesis.text}</p>
            <div class="flex items-center gap-2 mt-1">
              <span class="text-[10px] text-emerald-400/60">{Math.round(sophos.synthesis.confidence * 100)}%</span>
              <EtymonBadge etymonRef={getEtymon(sophos.synthesis.sources)} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

/** Bohm Tree Branch indicator (Iteration 4) */
const BohmBranchIndicator = component$<{
  parentId?: string;
  childIds?: string[];
  nodeId?: string;
}>(({ parentId, childIds, nodeId }) => {
  if (!nodeId && !parentId && (!childIds || childIds.length === 0)) return null;

  const hasParent = !!parentId;
  const hasChildren = childIds && childIds.length > 0;

  return (
    <div class="flex items-center gap-1 text-[9px] font-mono text-muted-foreground">
      {hasParent && (
        <span class="px-1 rounded bg-muted/30" title={`Parent: ${parentId}`}>
          ↑
        </span>
      )}
      {nodeId && (
        <span class="px-1 rounded bg-violet-500/20 text-violet-400" title={`Node: ${nodeId}`}>
          ⬡{nodeId.slice(-4)}
        </span>
      )}
      {hasChildren && (
        <span class="px-1 rounded bg-muted/30" title={`Children: ${childIds.length}`}>
          ↓{childIds.length}
        </span>
      )}
    </div>
  );
});

/** Transfiguration Flow Indicator - shows the complete pipeline (Iteration 3) */
const TransfigurationFlowIndicator = component$<{
  stage: TransfigurationStage;
  sophos?: { resolution_status: string };
  sextet?: { verdict: { status: string } };
}>(({ stage, sophos, sextet }) => {
  const stages = [
    { key: 'raw', label: 'Raw', icon: '○' },
    { key: 'sophos', label: 'SOPHOS', icon: '?', phases: ['questioning', 'synthesized'] },
    { key: 'holon', label: 'Holon', icon: '⬡', phases: ['instantiating', 'validating', 'transfigured'] },
    { key: 'portal', label: 'Portal', icon: '◎', phases: ['actualized', 'shadowed'] },
  ];

  const getStageIndex = () => {
    if (stage === 'raw') return 0;
    if (['questioning', 'synthesized'].includes(stage)) return 1;
    if (['instantiating', 'validating', 'transfigured'].includes(stage)) return 2;
    if (['actualized', 'shadowed', 'decayed'].includes(stage)) return 3;
    return 0;
  };

  const currentIndex = getStageIndex();

  return (
    <div class="flex items-center gap-1 p-2 bg-muted/20 rounded-lg">
      {stages.map((s, i) => {
        const isPast = i < currentIndex;
        const isCurrent = i === currentIndex;
        const statusIcon = isPast ? '✓' :
          isCurrent && sophos?.resolution_status === 'resolved' && i === 1 ? '✓' :
          isCurrent && sextet?.verdict?.status === 'PASSED' && i === 2 ? '✓' :
          isCurrent ? '▸' : '○';

        return (
          <div key={s.key} class="flex items-center">
            <div class={`flex flex-col items-center ${isCurrent ? 'text-primary' : isPast ? 'text-emerald-400' : 'text-muted-foreground/50'}`}>
              <div class={`w-8 h-8 rounded-full flex items-center justify-center text-sm border-2 ${
                isCurrent ? 'border-primary bg-primary/20' :
                isPast ? 'border-emerald-400 bg-emerald-500/20' :
                'border-muted/30 bg-muted/10'
              }`}>
                {statusIcon}
              </div>
              <span class="text-[9px] mt-1 font-medium">{s.label}</span>
            </div>
            {i < stages.length - 1 && (
              <div class={`w-8 h-0.5 mx-1 ${isPast ? 'bg-emerald-400' : 'bg-muted/30'}`} />
            )}
          </div>
        );
      })}
    </div>
  );
});

// ============================================================================
// Metabolic Badges
// ============================================================================

/** Metabolic mode indicator badge */
const MetabolicBadge = component$<{ mode: MetabolicMode | null }>(({ mode }) => {
  if (mode === 'AM') {
    return (
      <span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 font-mono">
        AM
      </span>
    );
  }
  if (mode === 'SM') {
    return (
      <span class="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-400 border border-purple-500/30 font-mono">
        SM
      </span>
    );
  }
  return (
    <span class="text-[10px] px-1.5 py-0.5 rounded bg-red-500/20 text-red-400 border border-red-500/30 font-mono">
      ∅
    </span>
  );
});

/** Transfiguration Flow Visualization (Raw → SOPHOS → Holon → Portal) */
const TransfigurationFlowView = component$<{
  leads: STRpLead[];
  onAction$: QRL<(action: LeadAction | 'transfigure', lead: STRpLead) => void>;
}>(({ leads, onAction$ }) => {
  const metabolicGroups = groupLeadsByMetabolic(leads);
  const stageGroups = groupLeadsByStage(leads);

  // Count leads in each phase of the flow
  const sophosCount = stageGroups.questioning.length + stageGroups.synthesized.length;
  const holonCount = stageGroups.instantiating.length + stageGroups.validating.length + stageGroups.transfigured.length;

  return (
    <div class="space-y-4">
      {/* Flow Pipeline Visualization */}
      <div class="flex items-center justify-center gap-2 p-3 rounded-lg bg-muted/20 border border-border/50">
        <div class="flex items-center gap-1 text-xs">
          <span class="px-2 py-1 rounded bg-gray-500/20 text-gray-400 font-mono">
            Raw: {stageGroups.raw.length}
          </span>
          <span class="text-muted-foreground">→</span>
          <span class="px-2 py-1 rounded bg-blue-500/20 text-blue-400 font-mono">
            SOPHOS: {sophosCount}
          </span>
          <span class="text-muted-foreground">→</span>
          <span class="px-2 py-1 rounded bg-cyan-500/20 text-cyan-400 font-mono">
            Holon: {holonCount}
          </span>
          <span class="text-muted-foreground">→</span>
          <span class="px-2 py-1 rounded bg-emerald-500/20 text-emerald-400 font-mono">
            AM: {metabolicGroups.AM.length}
          </span>
          <span class="text-purple-400">/</span>
          <span class="px-2 py-1 rounded bg-purple-500/20 text-purple-400 font-mono">
            SM: {metabolicGroups.SM.length}
          </span>
        </div>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Pending Column - Not yet through Sextet */}
        <div class="rounded-lg border-2 border-blue-500/30 bg-blue-500/5">
          <div class="px-4 py-3 border-b border-blue-500/30 bg-blue-500/10">
            <div class="flex items-center justify-between">
              <h3 class="font-semibold text-blue-400 flex items-center gap-2">
                <span class="text-lg">⚗️</span>
                <span>Pending Transfiguration</span>
              </h3>
              <span class="text-sm text-blue-400/70 font-mono">
                {metabolicGroups.pending.length}
              </span>
            </div>
            <p class="text-xs text-blue-400/60 mt-1">
              Raw → SOPHOS → Holon validation
            </p>
          </div>
          <div class="max-h-[350px] overflow-y-auto">
            {metabolicGroups.pending.length === 0 ? (
              <div class="p-8 text-center text-blue-400/50">
                <div class="text-2xl mb-2">◇</div>
                <div class="text-xs">All leads transfigured</div>
              </div>
            ) : (
              metabolicGroups.pending.map((lead) => (
                <div
                  key={lead.lead_id}
                  class="p-3 border-b border-blue-500/20 last:border-b-0 hover:bg-blue-500/10"
                >
                  <div class="flex items-start gap-2">
                    <span class="text-blue-400">○</span>
                    <div class="flex-1 min-w-0">
                      <a
                        href={lead.url}
                        target="_blank"
                        class="text-sm font-medium text-foreground hover:text-blue-400 line-clamp-1"
                      >
                        {lead.title}
                      </a>
                      <div class="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                        <span>{getTopicIcon(lead.topic)}</span>
                        <span class="font-mono text-blue-400/60">{lead.stage || 'raw'}</span>
                      </div>
                    </div>
                    <button
                      onClick$={() => onAction$('transfigure', lead)}
                      class="text-xs px-2 py-1 rounded bg-violet-500/20 text-violet-400 hover:bg-violet-500/30"
                    >
                      ⚗️
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* AM Column - Actualized Mode (after Sextet passed) */}
        <div class="rounded-lg border-2 border-emerald-500/30 bg-emerald-500/5">
          <div class="px-4 py-3 border-b border-emerald-500/30 bg-emerald-500/10">
            <div class="flex items-center justify-between">
              <h3 class="font-semibold text-emerald-400 flex items-center gap-2">
                <span class="text-lg">▲</span>
                <span>AM: Actualized</span>
              </h3>
              <span class="text-sm text-emerald-400/70 font-mono">
                {metabolicGroups.AM.length}
              </span>
            </div>
            <p class="text-xs text-emerald-400/60 mt-1">
              Sextet ✓ → Knowledge Graph
            </p>
          </div>
          <div class="max-h-[350px] overflow-y-auto">
            {metabolicGroups.AM.length === 0 ? (
              <div class="p-8 text-center text-emerald-400/50">
                <div class="text-2xl mb-2">◇</div>
                <div class="text-xs">No actualized leads</div>
              </div>
            ) : (
              metabolicGroups.AM.map((lead) => (
                <div
                  key={lead.lead_id}
                  class="p-3 border-b border-emerald-500/20 last:border-b-0 hover:bg-emerald-500/10"
                >
                  <div class="flex items-start gap-2">
                    <span class="text-emerald-400">▸</span>
                    <div class="flex-1 min-w-0">
                      <a
                        href={lead.url}
                        target="_blank"
                        class="text-sm font-medium text-foreground hover:text-emerald-400 line-clamp-1"
                      >
                        {lead.title}
                      </a>
                      <div class="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                        <span>{getTopicIcon(lead.topic)}</span>
                        <span class="font-mono text-emerald-400/60">actualized</span>
                        {lead.sextet && (
                          <span class="text-[10px] px-1 rounded bg-emerald-500/30 text-emerald-300">
                            [P✓E✓L✓R✓Q✓Ω✓]
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* SM Column - Shadowmode (potential held) */}
        <div class="rounded-lg border-2 border-purple-500/30 bg-purple-500/5">
          <div class="px-4 py-3 border-b border-purple-500/30 bg-purple-500/10">
            <div class="flex items-center justify-between">
              <h3 class="font-semibold text-purple-400 flex items-center gap-2">
                <span class="text-lg">◐</span>
                <span>SM: Shadowmode</span>
              </h3>
              <span class="text-sm text-purple-400/70 font-mono">
                {metabolicGroups.SM.length}
              </span>
            </div>
            <p class="text-xs text-purple-400/60 mt-1">
              Sextet ✓ → Held Potential
            </p>
          </div>
          <div class="max-h-[350px] overflow-y-auto">
            {metabolicGroups.SM.length === 0 ? (
              <div class="p-8 text-center text-purple-400/50">
                <div class="text-2xl mb-2">◑</div>
                <div class="text-xs">No shadow potentials</div>
              </div>
            ) : (
              metabolicGroups.SM.map((lead) => (
                <div
                  key={lead.lead_id}
                  class="p-3 border-b border-purple-500/20 last:border-b-0 hover:bg-purple-500/10"
                >
                  <div class="flex items-start gap-2">
                    <span class="text-purple-400/50">◦</span>
                    <div class="flex-1 min-w-0">
                      <a
                        href={lead.url}
                        target="_blank"
                        class="text-sm font-medium text-foreground/70 hover:text-purple-400 line-clamp-1"
                      >
                        {lead.title}
                      </a>
                      <div class="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                        <span>{getTopicIcon(lead.topic)}</span>
                        <span class="font-mono text-purple-400/60">potential</span>
                      </div>
                    </div>
                    <button
                      onClick$={() => onAction$('transfigure', lead)}
                      class="text-xs px-2 py-1 rounded bg-purple-500/20 text-purple-400 hover:bg-purple-500/30"
                    >
                      →AM
                    </button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
});

// Alias for backwards compatibility
const MetabolicForkView = TransfigurationFlowView;

type ViewMode = 'list' | 'metabolic';

export const STRpLeadsView = component$<STRpLeadsViewProps>((props) => {
  const { leads, dispatchAction, connected = true } = props;

  // Local state
  const activeTab = useSignal<LeadTab>('all');
  const topicFilter = useSignal<LeadTopic | 'ALL'>('ALL');
  const searchQuery = useSignal('');
  const viewMode = useSignal<ViewMode>('list');

  // Computed: grouped leads
  const groupedLeads = useComputed$(() => {
    return groupLeadsByDecision(leads.value);
  });

  // Computed: counts per decision
  const counts = useComputed$(() => ({
    all: leads.value.length,
    promote: groupedLeads.value.promote.length,
    defer: groupedLeads.value.defer.length,
    reject: groupedLeads.value.reject.length,
  }));

  // Computed: filtered leads
  const filteredLeads = useComputed$(() => {
    let result = leads.value;

    // Filter by tab
    if (activeTab.value !== 'all') {
      result = result.filter((l) => l.decision === activeTab.value);
    }

    // Filter by topic
    if (topicFilter.value !== 'ALL') {
      result = result.filter((l) => l.topic === topicFilter.value);
    }

    // Filter by search
    if (searchQuery.value.trim()) {
      const q = searchQuery.value.toLowerCase();
      result = result.filter(
        (l) =>
          l.title.toLowerCase().includes(q) ||
          l.keywords.some((k) => k.toLowerCase().includes(q)) ||
          l.notes?.toLowerCase().includes(q)
      );
    }

    // Sort by timestamp (newest first)
    return result.slice().sort((a, b) => b.ts.localeCompare(a.ts));
  });

  // Available topics for filter
  const availableTopics = useComputed$(() => {
    const topics = new Set<string>();
    for (const lead of leads.value) {
      topics.add(lead.topic);
    }
    return Array.from(topics).sort();
  });

  // Action handler - emits transfiguration flow events
  const handleAction = $((action: LeadAction | 'transfigure', lead: STRpLead) => {
    const topic = (() => {
      switch (action) {
        case 'watch':
          return LEAD_BUS_TOPICS.LEAD_WATCH;
        case 'transfigure':
          return TRANSFIGURATION_BUS_TOPICS.SOPHOS_QUESTION_POSED;
        case 'archive':
          return LEAD_BUS_TOPICS.LEAD_ARCHIVE;
        case 'promote':
        case 'defer':
        case 'reject':
          return LEAD_BUS_TOPICS.LEAD_DECISION;
        default:
          return `strp.lead.action.${action}`;
      }
    })();

    // Emit Lead action event
    dispatchAction(topic, {
      lead_id: lead.lead_id,
      action,
      lead,
    });

    // For TRANSFIGURE action - run full transfiguration pipeline (Iter 2)
    // Uses auto-generation for thesis/antithesis/synthesis and validates through Sextet
    if (action === 'transfigure') {
      // Run the complete transfiguration pipeline
      const transfiguredLead = completeTransfiguration(lead);
      const event = createTransfigurationEvent(transfiguredLead);

      // Emit the full pipeline result
      dispatchAction(event.topic, {
        ...event.data,
        source: 'leads',
        action: 'complete_transfiguration',
        pipeline: {
          thesis: transfiguredLead.sophos?.thesis,
          antithesis: transfiguredLead.sophos?.antithesis,
          synthesis: transfiguredLead.sophos?.synthesis,
          pentad: transfiguredLead.pentad,
          sextet_verdict: transfiguredLead.sextet?.verdict,
          metabolic_mode: transfiguredLead.metabolic_mode,
        },
        // Signal to UI to show transfiguration result
        open_sophos: true,
        sophos_question_id: transfiguredLead.sophos?.question_id,
      });

      // Also emit individual stage events for observability
      if (transfiguredLead.sophos) {
        dispatchAction(TRANSFIGURATION_BUS_TOPICS.SOPHOS_SYNTHESIS_COMPLETE, {
          lead_id: lead.lead_id,
          question_id: transfiguredLead.sophos.question_id,
          resolution: transfiguredLead.sophos.resolution_status,
        });
      }
      if (transfiguredLead.sextet) {
        dispatchAction(TRANSFIGURATION_BUS_TOPICS.SEXTET_VALIDATION_COMPLETE, {
          lead_id: lead.lead_id,
          sextet_id: transfiguredLead.sextet.sextet_id,
          verdict: transfiguredLead.sextet.verdict,
          compliance_vector: transfiguredLead.sextet.compliance_vector,
        });
      }
    }

    // For decision changes, track stage progression
    if (action === 'promote' || action === 'defer' || action === 'reject') {
      const tLead = initializeTransfiguration({ ...lead, decision: action as LeadDecision });
      const event = createTransfigurationEvent(tLead);
      dispatchAction(event.topic, {
        ...event.data,
        source: 'leads',
        transition: `${lead.decision} → ${action}`,
      });
    }
  });

  // Tab config
  const tabs: { id: LeadTab; label: string; color: string }[] = [
    { id: 'all', label: 'All', color: 'text-foreground' },
    { id: 'promote', label: 'Promote', color: 'text-green-400' },
    { id: 'defer', label: 'Defer', color: 'text-yellow-400' },
    { id: 'reject', label: 'Reject', color: 'text-red-400' },
  ];

  return (
    <div class="space-y-4">
      {/* Header */}
      <div class="flex flex-col lg:flex-row gap-4 items-start lg:items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold flex items-center gap-2">
            <span>📋</span>
            <span>STRp Leads</span>
            {!connected && (
              <span class="text-xs px-2 py-0.5 rounded bg-red-500/20 text-red-400">
                Disconnected
              </span>
            )}
          </h2>
          <p class="text-sm text-muted-foreground">
            Curated content from MOTD pipeline
          </p>
        </div>

        <div class="flex items-center gap-3">
          {/* View mode toggle */}
          <div class="flex rounded-md border border-border overflow-hidden">
            <button
              onClick$={() => { viewMode.value = 'list'; }}
              class={`text-xs px-3 py-2 transition-colors ${
                viewMode.value === 'list'
                  ? 'bg-primary/20 text-primary'
                  : 'bg-background text-muted-foreground hover:bg-muted'
              }`}
            >
              ☰ List
            </button>
            <button
              onClick$={() => { viewMode.value = 'metabolic'; }}
              class={`text-xs px-3 py-2 transition-colors border-l border-border ${
                viewMode.value === 'metabolic'
                  ? 'bg-primary/20 text-primary'
                  : 'bg-background text-muted-foreground hover:bg-muted'
              }`}
            >
              ⑂ AM/SM
            </button>
          </div>

          {/* Topic filter */}
          <select
            class="text-xs px-3 py-2 rounded border border-border bg-background"
            value={topicFilter.value}
            onChange$={(e) => {
              topicFilter.value = (e.target as HTMLSelectElement)
                .value as LeadTopic | 'ALL';
            }}
          >
            <option value="ALL">All Topics</option>
            {availableTopics.value.map((t) => (
              <option key={t} value={t}>
                {`${getTopicIcon(t)} ${t}`}
              </option>
            ))}
          </select>

          {/* Search */}
          <input
            type="text"
            placeholder="Search leads..."
            class="text-xs px-3 py-2 rounded border border-border bg-background w-48"
            value={searchQuery.value}
            onInput$={(e) => {
              searchQuery.value = (e.target as HTMLInputElement).value;
            }}
          />

          <span class="text-sm text-muted-foreground">
            {filteredLeads.value.length} / {leads.value.length} leads
          </span>
        </div>
      </div>

      {/* Tabs */}
      <div class="flex border-b border-border">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick$={() => {
              activeTab.value = tab.id;
            }}
            class={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab.value === tab.id
                ? `border-primary ${tab.color}`
                : 'border-transparent text-muted-foreground hover:text-foreground'
            }`}
          >
            {tab.label}
            <span class="ml-2 text-xs opacity-75">
              ({counts.value[tab.id]})
            </span>
          </button>
        ))}
      </div>

      {/* Metabolic Fork View (AM/SM) */}
      {viewMode.value === 'metabolic' && (
        <MetabolicForkView
          leads={filteredLeads.value}
          onAction$={handleAction}
        />
      )}

      {/* List View */}
      {viewMode.value === 'list' && (
        <div class="rounded-lg border border-border bg-card">
          {filteredLeads.value.length === 0 ? (
            <div class="p-12 text-center text-muted-foreground">
              <div class="text-4xl mb-4">📭</div>
              <div class="text-sm">
                {leads.value.length === 0
                  ? 'No leads available. Waiting for MOTD pipeline...'
                  : 'No leads match the current filters.'}
              </div>
            </div>
          ) : (
            <div class="divide-y divide-border/30 max-h-[calc(100vh-320px)] overflow-y-auto">
              {filteredLeads.value.map((lead) => (
                <LeadCard
                  key={lead.lead_id}
                  lead={lead}
                  onAction$={handleAction}
                />
              ))}
            </div>
          )}
        </div>
      )}

      {/* Stats summary - now includes metabolic mode counts */}
      <div class="grid grid-cols-2 md:grid-cols-7 gap-2">
        <div class="rounded-lg border border-border bg-card p-3 text-center">
          <div class="text-2xl font-bold text-foreground">{counts.value.all}</div>
          <div class="text-xs text-muted-foreground">Total</div>
        </div>
        <div class="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-emerald-400">
            {counts.value.promote}
          </div>
          <div class="text-xs text-muted-foreground flex items-center justify-center gap-1">
            <span>AM</span>
            <span class="text-[10px] opacity-60">▲</span>
          </div>
        </div>
        <div class="rounded-lg border border-purple-500/30 bg-purple-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-purple-400">
            {counts.value.defer}
          </div>
          <div class="text-xs text-muted-foreground flex items-center justify-center gap-1">
            <span>SM</span>
            <span class="text-[10px] opacity-60">◐</span>
          </div>
        </div>
        <div class="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-red-400">{counts.value.reject}</div>
          <div class="text-xs text-muted-foreground flex items-center justify-center gap-1">
            <span>∅</span>
            <span class="text-[10px] opacity-60">decay</span>
          </div>
        </div>
        <div class="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-blue-400">
            {leads.value.filter((l) => l.ingested).length}
          </div>
          <div class="text-xs text-muted-foreground">Portal</div>
        </div>
        <div class="rounded-lg border border-green-500/30 bg-green-500/10 p-3 text-center col-span-2 md:col-span-2">
          <div class="text-xs text-muted-foreground mb-1">Metabolic Flow</div>
          <div class="flex items-center justify-center gap-2 text-sm">
            <span class="text-emerald-400">{counts.value.promote} AM</span>
            <span class="text-muted-foreground">→</span>
            <span class="text-purple-400">{counts.value.defer} SM</span>
            <span class="text-muted-foreground">→</span>
            <span class="text-blue-400">{leads.value.filter((l) => l.ingested).length} ⊕</span>
          </div>
        </div>
      </div>
    </div>
  );
});
