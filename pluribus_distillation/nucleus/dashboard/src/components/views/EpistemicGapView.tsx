/**
 * EpistemicGapView - Visualizes knowledge gaps and uncertainty in the system
 *
 * This component provides a visual interface for:
 * 1. Known unknowns - things the system knows it doesn't know
 * 2. Confidence levels in current knowledge
 * 3. Suggested research/exploration paths to fill gaps
 * 4. Integration with SOTA research pipeline
 *
 * Connects to bus events:
 * - epistemic.gap.discovered
 * - epistemic.gap.filled
 * - epistemic.gap.updated
 * - sota.research.requested
 * - sota.research.completed
 */

import { component$, useSignal, useVisibleTask$, $, type Signal, type QRL } from '@builder.io/qwik';
import type { HysteresisTrace } from '../../lib/state/types';

// ============================================================================
// Epistemic Gap Types
// ============================================================================

/** Severity/priority of an epistemic gap */
export type GapSeverity = 'critical' | 'high' | 'medium' | 'low' | 'info';

/** Category of knowledge gap */
export type GapCategory =
  | 'capability'       // Missing system capability
  | 'integration'      // Integration knowledge needed
  | 'domain'           // Domain expertise required
  | 'sota'             // State-of-the-art research needed
  | 'implementation'   // Implementation details unknown
  | 'validation'       // Testing/validation gaps
  | 'architecture'     // Architectural decisions pending
  | 'security'         // Security considerations unknown
  | 'performance';     // Performance characteristics unknown

/** Status of gap resolution */
export type GapStatus =
  | 'open'            // Identified but not addressed
  | 'researching'     // Active research in progress
  | 'proposed'        // Solution proposed, pending validation
  | 'validating'      // Validation in progress
  | 'resolved'        // Gap filled successfully
  | 'wontfix'         // Accepted as limitation
  | 'deferred';       // Deferred to future work

/** Confidence level in current knowledge */
export interface ConfidenceScore {
  level: number;           // 0-100 confidence percentage
  basis: 'empirical' | 'theoretical' | 'inferred' | 'assumed' | 'unknown';
  lastValidated?: string;  // ISO timestamp
  validationMethod?: string;
}

/** Research path to fill a gap */
export interface ResearchPath {
  id: string;
  title: string;
  description: string;
  estimatedEffort: 'trivial' | 'small' | 'medium' | 'large' | 'unknown';
  suggestedAgents: string[];  // Which agents can tackle this
  prerequisites: string[];     // Other gaps that must be filled first
  sotaRefs: string[];          // SOTA catalog references
  successCriteria: string[];
  status: 'suggested' | 'active' | 'completed' | 'blocked';
}

/** Core epistemic gap structure */
export interface EpistemicGap {
  id: string;
  title: string;
  description: string;
  category: GapCategory;
  severity: GapSeverity;
  status: GapStatus;

  // Relationship modeling
  parentId?: string;           // Hierarchical parent gap
  childIds: string[];          // Sub-gaps
  relatedGapIds: string[];     // Related but not hierarchical
  blockedByIds: string[];      // Dependencies

  // Confidence tracking
  confidence: ConfidenceScore;

  // Hysteresis - path-dependent memory from system types
  hysteresis?: HysteresisTrace;

  // Research integration
  researchPaths: ResearchPath[];
  activeResearchId?: string;

  // Provenance
  discoveredAt: string;        // ISO timestamp
  discoveredBy: string;        // Actor who identified this
  lastUpdatedAt: string;
  lastUpdatedBy: string;
  resolvedAt?: string;
  resolvedBy?: string;

  // Context
  context: Record<string, unknown>;
  tags: string[];

  // Impact estimation
  impactAreas: string[];
  impactScore: number;         // 0-100
}

/** Bus event payload for gap events */
export interface EpistemicGapEvent {
  gap: EpistemicGap;
  action: 'discover' | 'update' | 'fill' | 'defer' | 'escalate';
  actor: string;
  reason?: string;
}

// ============================================================================
// Component Props
// ============================================================================

interface EpistemicGapViewProps {
  gaps: Signal<EpistemicGap[]>;
  selectedGapId: Signal<string | null>;
  dispatchAction: QRL<(type: string, payload: Record<string, unknown>) => void>;
}

// ============================================================================
// Helper Components
// ============================================================================

const SeverityBadge = component$<{ severity: GapSeverity }>(({ severity }) => {
  const colors: Record<GapSeverity, string> = {
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
    high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
    medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
    low: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    info: 'bg-gray-500/20 text-gray-400 border-gray-500/30',
  };

  return (
    <span class={`text-xs px-2 py-0.5 rounded border ${colors[severity]}`}>
      {severity.toUpperCase()}
    </span>
  );
});

const StatusBadge = component$<{ status: GapStatus }>(({ status }) => {
  const colors: Record<GapStatus, string> = {
    open: 'bg-purple-500/20 text-purple-400',
    researching: 'bg-blue-500/20 text-blue-400',
    proposed: 'bg-cyan-500/20 text-cyan-400',
    validating: 'bg-teal-500/20 text-teal-400',
    resolved: 'bg-green-500/20 text-green-400',
    wontfix: 'bg-gray-500/20 text-gray-400',
    deferred: 'bg-yellow-500/20 text-yellow-400',
  };

  const icons: Record<GapStatus, string> = {
    open: '?',
    researching: '*',
    proposed: '>',
    validating: '~',
    resolved: '+',
    wontfix: 'x',
    deferred: '-',
  };

  return (
    <span class={`text-xs px-2 py-0.5 rounded ${colors[status]}`}>
      {icons[status]} {status}
    </span>
  );
});

const CategoryIcon = component$<{ category: GapCategory }>(({ category }) => {
  const icons: Record<GapCategory, string> = {
    capability: '[C]',
    integration: '[I]',
    domain: '[D]',
    sota: '[S]',
    implementation: '[M]',
    validation: '[V]',
    architecture: '[A]',
    security: '[X]',
    performance: '[P]',
  };

  return (
    <span class="font-mono text-muted-foreground text-xs" title={category}>
      {icons[category]}
    </span>
  );
});

const ConfidenceMeter = component$<{ confidence: ConfidenceScore }>(({ confidence }) => {
  const width = Math.max(0, Math.min(100, confidence.level));
  const color = confidence.level >= 80 ? 'bg-green-500' :
                confidence.level >= 60 ? 'bg-blue-500' :
                confidence.level >= 40 ? 'bg-yellow-500' :
                confidence.level >= 20 ? 'bg-orange-500' :
                'bg-red-500';

  return (
    <div class="flex items-center gap-2">
      <div class="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
        <div
          class={`h-full ${color} transition-all duration-300`}
          style={{ width: `${width}%` }}
        />
      </div>
      <span class="text-xs text-muted-foreground">{confidence.level}%</span>
      <span class="text-xs text-muted-foreground/60">({confidence.basis})</span>
    </div>
  );
});

const ResearchPathCard = component$<{
  path: ResearchPath;
  onActivate: QRL<(pathId: string) => void>;
}>(({ path, onActivate }) => {
  const effortColors: Record<string, string> = {
    trivial: 'text-green-400',
    small: 'text-blue-400',
    medium: 'text-yellow-400',
    large: 'text-orange-400',
    unknown: 'text-gray-400',
  };

  return (
    <div class="p-3 rounded border border-border/50 bg-muted/10 hover:bg-muted/20 transition-colors">
      <div class="flex items-start justify-between gap-2">
        <div class="flex-1">
          <div class="font-medium text-sm">{path.title}</div>
          <div class="text-xs text-muted-foreground mt-1">{path.description}</div>
        </div>
        {path.status === 'suggested' && (
          <button
            onClick$={() => onActivate(path.id)}
            class="text-xs px-2 py-1 rounded bg-primary/20 text-primary hover:bg-primary/30 transition-colors"
          >
            Start
          </button>
        )}
        {path.status === 'active' && (
          <span class="text-xs px-2 py-1 rounded bg-blue-500/20 text-blue-400 animate-pulse">
            Active
          </span>
        )}
      </div>

      <div class="mt-2 flex flex-wrap gap-2">
        <span class={`text-xs ${effortColors[path.estimatedEffort]}`}>
          Effort: {path.estimatedEffort}
        </span>
        {path.suggestedAgents.length > 0 && (
          <span class="text-xs text-muted-foreground">
            Agents: {path.suggestedAgents.join(', ')}
          </span>
        )}
      </div>

      {path.sotaRefs.length > 0 && (
        <div class="mt-2 text-xs text-muted-foreground">
          SOTA refs: {path.sotaRefs.join(', ')}
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Gap Tree Node Component
// ============================================================================

const GapTreeNode = component$<{
  gap: EpistemicGap;
  gaps: EpistemicGap[];
  depth: number;
  selectedId: string | null;
  onSelect: QRL<(id: string) => void>;
}>(({ gap, gaps, depth, selectedId, onSelect }) => {
  const isSelected = selectedId === gap.id;
  const children = gaps.filter(g => g.parentId === gap.id);
  const indent = depth * 16;

  return (
    <div>
      <div
        class={`flex items-center gap-2 px-2 py-1.5 cursor-pointer hover:bg-muted/30 transition-colors ${
          isSelected ? 'bg-primary/10 border-l-2 border-primary' : ''
        }`}
        style={{ paddingLeft: `${indent + 8}px` }}
        onClick$={() => onSelect(gap.id)}
      >
        {children.length > 0 && (
          <span class="text-xs text-muted-foreground">[+]</span>
        )}
        <CategoryIcon category={gap.category} />
        <span class={`flex-1 text-sm truncate ${
          gap.status === 'resolved' ? 'text-muted-foreground line-through' : ''
        }`}>
          {gap.title}
        </span>
        <SeverityBadge severity={gap.severity} />
        <StatusBadge status={gap.status} />
      </div>

      {children.map(child => (
        <GapTreeNode
          key={child.id}
          gap={child}
          gaps={gaps}
          depth={depth + 1}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
});

// ============================================================================
// Gap Detail Panel
// ============================================================================

const GapDetailPanel = component$<{
  gap: EpistemicGap;
  onFillGap: QRL<(gapId: string, pathId?: string) => void>;
  onActivateResearch: QRL<(gapId: string, pathId: string) => void>;
}>(({ gap, onFillGap, onActivateResearch }) => {
  return (
    <div class="space-y-4">
      {/* Header */}
      <div class="flex items-start justify-between gap-4">
        <div>
          <div class="flex items-center gap-2">
            <CategoryIcon category={gap.category} />
            <h3 class="text-lg font-semibold">{gap.title}</h3>
          </div>
          <div class="flex items-center gap-2 mt-2">
            <SeverityBadge severity={gap.severity} />
            <StatusBadge status={gap.status} />
          </div>
        </div>

        {gap.status === 'open' && (
          <button
            onClick$={() => onFillGap(gap.id)}
            class="px-3 py-1.5 rounded bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
          >
            Fill Gap
          </button>
        )}
      </div>

      {/* Description */}
      <div class="text-sm text-muted-foreground">
        {gap.description}
      </div>

      {/* Confidence */}
      <div class="p-3 rounded bg-muted/20 border border-border/50">
        <div class="text-xs font-medium mb-2">Confidence Level</div>
        <ConfidenceMeter confidence={gap.confidence} />
        {gap.confidence.lastValidated && (
          <div class="text-xs text-muted-foreground mt-1">
            Last validated: {gap.confidence.lastValidated}
          </div>
        )}
      </div>

      {/* Hysteresis trace if available */}
      {gap.hysteresis && (
        <div class="p-3 rounded bg-purple-500/10 border border-purple-500/30">
          <div class="text-xs font-medium text-purple-400 mb-2">Hysteresis Trace</div>
          <div class="grid grid-cols-2 gap-2 text-xs">
            {gap.hysteresis.decision_points !== undefined && (
              <div>Decision Points: <span class="text-purple-300">{gap.hysteresis.decision_points}</span></div>
            )}
            {gap.hysteresis.reversibility !== undefined && (
              <div>Reversibility: <span class="text-purple-300">{(gap.hysteresis.reversibility * 100).toFixed(0)}%</span></div>
            )}
            {gap.hysteresis.entropy !== undefined && (
              <div>Entropy: <span class="text-purple-300">{gap.hysteresis.entropy.toFixed(2)}</span></div>
            )}
            {gap.hysteresis.causal_depth !== undefined && (
              <div>Causal Depth: <span class="text-purple-300">{gap.hysteresis.causal_depth}</span></div>
            )}
          </div>
        </div>
      )}

      {/* Impact */}
      <div class="p-3 rounded bg-muted/20 border border-border/50">
        <div class="flex items-center justify-between mb-2">
          <span class="text-xs font-medium">Impact Score</span>
          <span class="text-sm font-mono">{gap.impactScore}/100</span>
        </div>
        <div class="h-1.5 bg-muted rounded-full overflow-hidden">
          <div
            class={`h-full transition-all ${
              gap.impactScore >= 80 ? 'bg-red-500' :
              gap.impactScore >= 60 ? 'bg-orange-500' :
              gap.impactScore >= 40 ? 'bg-yellow-500' :
              'bg-blue-500'
            }`}
            style={{ width: `${gap.impactScore}%` }}
          />
        </div>
        {gap.impactAreas.length > 0 && (
          <div class="flex flex-wrap gap-1 mt-2">
            {gap.impactAreas.map(area => (
              <span key={area} class="text-xs px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground">
                {area}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Dependencies */}
      {gap.blockedByIds.length > 0 && (
        <div class="p-3 rounded bg-red-500/10 border border-red-500/30">
          <div class="text-xs font-medium text-red-400 mb-2">Blocked By</div>
          <div class="space-y-1">
            {gap.blockedByIds.map(id => (
              <div key={id} class="text-xs text-muted-foreground font-mono">
                {id}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Research Paths */}
      {gap.researchPaths.length > 0 && (
        <div>
          <div class="text-xs font-medium mb-2">Research Paths</div>
          <div class="space-y-2">
            {gap.researchPaths.map(path => (
              <ResearchPathCard
                key={path.id}
                path={path}
                onActivate={$((pathId: string) => onActivateResearch(gap.id, pathId))}
              />
            ))}
          </div>
        </div>
      )}

      {/* Provenance */}
      <div class="text-xs text-muted-foreground border-t border-border pt-2">
        <div>Discovered: {gap.discoveredAt} by {gap.discoveredBy}</div>
        <div>Updated: {gap.lastUpdatedAt} by {gap.lastUpdatedBy}</div>
        {gap.resolvedAt && (
          <div class="text-green-400">Resolved: {gap.resolvedAt} by {gap.resolvedBy}</div>
        )}
      </div>

      {/* Tags */}
      {gap.tags.length > 0 && (
        <div class="flex flex-wrap gap-1">
          {gap.tags.map(tag => (
            <span key={tag} class="text-xs px-1.5 py-0.5 rounded bg-muted/50 text-muted-foreground">
              #{tag}
            </span>
          ))}
        </div>
      )}
    </div>
  );
});

// ============================================================================
// Main Component
// ============================================================================

export const EpistemicGapView = component$<EpistemicGapViewProps>((props) => {
  const { gaps, selectedGapId, dispatchAction } = props;

  // View modes
  const viewMode = useSignal<'tree' | 'graph' | 'matrix'>('tree');
  const filterSeverity = useSignal<GapSeverity | 'all'>('all');
  const filterStatus = useSignal<GapStatus | 'all'>('all');
  const filterCategory = useSignal<GapCategory | 'all'>('all');
  const showResolved = useSignal(false);

  // Computed filtered gaps
  const filteredGaps = gaps.value.filter(gap => {
    if (!showResolved.value && gap.status === 'resolved') return false;
    if (filterSeverity.value !== 'all' && gap.severity !== filterSeverity.value) return false;
    if (filterStatus.value !== 'all' && gap.status !== filterStatus.value) return false;
    if (filterCategory.value !== 'all' && gap.category !== filterCategory.value) return false;
    return true;
  });

  // Root-level gaps (no parent)
  const rootGaps = filteredGaps.filter(g => !g.parentId);

  // Selected gap
  const selectedGap = gaps.value.find(g => g.id === selectedGapId.value);

  // Statistics
  const stats = {
    total: gaps.value.length,
    open: gaps.value.filter(g => g.status === 'open').length,
    researching: gaps.value.filter(g => g.status === 'researching').length,
    resolved: gaps.value.filter(g => g.status === 'resolved').length,
    critical: gaps.value.filter(g => g.severity === 'critical' && g.status !== 'resolved').length,
    avgConfidence: gaps.value.length > 0
      ? Math.round(gaps.value.reduce((sum, g) => sum + g.confidence.level, 0) / gaps.value.length)
      : 0,
  };

  // Handlers
  const handleSelectGap = $((id: string) => {
    selectedGapId.value = id;
  });

  const handleFillGap = $((gapId: string, pathId?: string) => {
    dispatchAction('epistemic.gap.fill', { gapId, pathId });
  });

  const handleActivateResearch = $((gapId: string, pathId: string) => {
    dispatchAction('sota.research.request', { gapId, pathId });
  });

  // Bus subscription for live updates
  useVisibleTask$(({ cleanup }) => {
    // This would integrate with the bus client for real-time updates
    // For now, we set up the structure for bus integration
    const topics = [
      'epistemic.gap.discovered',
      'epistemic.gap.filled',
      'epistemic.gap.updated',
      'sota.research.completed',
    ];

    console.log('[EpistemicGapView] Subscribing to topics:', topics);

    cleanup(() => {
      console.log('[EpistemicGapView] Cleaning up subscriptions');
    });
  });

  return (
    <div class="space-y-4">
      {/* Header */}
      <div class="flex items-center justify-between">
        <div>
          <h2 class="text-lg font-semibold">Epistemic Gaps</h2>
          <p class="text-sm text-muted-foreground">Known unknowns and research paths</p>
        </div>
        <div class="flex items-center gap-2">
          {/* View Mode Toggle */}
          <div class="flex rounded-lg border border-border overflow-hidden">
            {(['tree', 'graph', 'matrix'] as const).map(mode => (
              <button
                key={mode}
                onClick$={() => viewMode.value = mode}
                class={`px-3 py-1.5 text-xs transition-colors ${
                  viewMode.value === mode
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted/30 text-muted-foreground hover:bg-muted/50'
                }`}
              >
                {mode === 'tree' ? 'Tree' : mode === 'graph' ? 'Graph' : 'Matrix'}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Statistics Bar */}
      <div class="grid grid-cols-2 md:grid-cols-6 gap-2">
        <div class="rounded-lg border border-border bg-card p-3 text-center">
          <div class="text-2xl font-bold text-primary">{stats.total}</div>
          <div class="text-xs text-muted-foreground">Total Gaps</div>
        </div>
        <div class="rounded-lg border border-purple-500/30 bg-purple-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-purple-400">{stats.open}</div>
          <div class="text-xs text-muted-foreground">Open</div>
        </div>
        <div class="rounded-lg border border-blue-500/30 bg-blue-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-blue-400">{stats.researching}</div>
          <div class="text-xs text-muted-foreground">Researching</div>
        </div>
        <div class="rounded-lg border border-green-500/30 bg-green-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-green-400">{stats.resolved}</div>
          <div class="text-xs text-muted-foreground">Resolved</div>
        </div>
        <div class="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-red-400">{stats.critical}</div>
          <div class="text-xs text-muted-foreground">Critical</div>
        </div>
        <div class="rounded-lg border border-cyan-500/30 bg-cyan-500/10 p-3 text-center">
          <div class="text-2xl font-bold text-cyan-400">{stats.avgConfidence}%</div>
          <div class="text-xs text-muted-foreground">Avg Confidence</div>
        </div>
      </div>

      {/* Filters */}
      <div class="flex flex-wrap gap-2 items-center">
        <select
          class="px-2 py-1 rounded bg-muted/50 border border-border text-xs"
          value={filterSeverity.value}
          onChange$={(e) => { filterSeverity.value = (e.target as HTMLSelectElement).value as GapSeverity | 'all'; }}
        >
          <option value="all">All Severities</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
          <option value="info">Info</option>
        </select>

        <select
          class="px-2 py-1 rounded bg-muted/50 border border-border text-xs"
          value={filterStatus.value}
          onChange$={(e) => { filterStatus.value = (e.target as HTMLSelectElement).value as GapStatus | 'all'; }}
        >
          <option value="all">All Statuses</option>
          <option value="open">Open</option>
          <option value="researching">Researching</option>
          <option value="proposed">Proposed</option>
          <option value="validating">Validating</option>
          <option value="resolved">Resolved</option>
          <option value="deferred">Deferred</option>
        </select>

        <select
          class="px-2 py-1 rounded bg-muted/50 border border-border text-xs"
          value={filterCategory.value}
          onChange$={(e) => { filterCategory.value = (e.target as HTMLSelectElement).value as GapCategory | 'all'; }}
        >
          <option value="all">All Categories</option>
          <option value="capability">Capability</option>
          <option value="integration">Integration</option>
          <option value="domain">Domain</option>
          <option value="sota">SOTA</option>
          <option value="implementation">Implementation</option>
          <option value="validation">Validation</option>
          <option value="architecture">Architecture</option>
          <option value="security">Security</option>
          <option value="performance">Performance</option>
        </select>

        <label class="flex items-center gap-1 text-xs text-muted-foreground cursor-pointer">
          <input
            type="checkbox"
            checked={showResolved.value}
            onChange$={() => showResolved.value = !showResolved.value}
            class="w-3 h-3"
          />
          Show Resolved
        </label>

        <span class="text-xs text-muted-foreground ml-auto">
          Showing {filteredGaps.length} of {gaps.value.length} gaps
        </span>
      </div>

      {/* Main Content Area */}
      <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Gap List/Tree */}
        <div class="lg:col-span-2 rounded-lg border border-border bg-card overflow-hidden">
          <div class="p-3 border-b border-border flex items-center justify-between">
            <h3 class="font-semibold">Gap Hierarchy</h3>
            <span class="text-xs text-muted-foreground">
              {viewMode.value === 'tree' ? 'Tree View' : viewMode.value === 'graph' ? 'Graph View' : 'Matrix View'}
            </span>
          </div>

          {viewMode.value === 'tree' && (
            <div class="divide-y divide-border/30 max-h-[calc(100vh-400px)] overflow-auto">
              {rootGaps.length === 0 ? (
                <div class="p-8 text-center text-muted-foreground">
                  <div class="text-4xl mb-4">?</div>
                  <div>No epistemic gaps found</div>
                  <div class="text-xs mt-2">Knowledge gaps will appear here as they are discovered</div>
                </div>
              ) : (
                rootGaps.map(gap => (
                  <GapTreeNode
                    key={gap.id}
                    gap={gap}
                    gaps={filteredGaps}
                    depth={0}
                    selectedId={selectedGapId.value}
                    onSelect={handleSelectGap}
                  />
                ))
              )}
            </div>
          )}

          {viewMode.value === 'graph' && (
            <div class="h-[400px] flex items-center justify-center text-muted-foreground">
              <div class="text-center">
                <div class="text-4xl mb-4">[Graph]</div>
                <div class="text-sm">Interactive graph visualization</div>
                <div class="text-xs mt-2">(D3/Force-directed coming soon)</div>
              </div>
            </div>
          )}

          {viewMode.value === 'matrix' && (
            <div class="p-4 overflow-auto">
              <table class="w-full text-xs">
                <thead>
                  <tr class="border-b border-border">
                    <th class="text-left p-2">Gap</th>
                    <th class="text-left p-2">Category</th>
                    <th class="text-left p-2">Severity</th>
                    <th class="text-left p-2">Status</th>
                    <th class="text-left p-2">Confidence</th>
                    <th class="text-left p-2">Impact</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredGaps.map(gap => (
                    <tr
                      key={gap.id}
                      class={`border-b border-border/30 hover:bg-muted/20 cursor-pointer ${
                        selectedGapId.value === gap.id ? 'bg-primary/10' : ''
                      }`}
                      onClick$={() => handleSelectGap(gap.id)}
                    >
                      <td class="p-2 font-medium">{gap.title}</td>
                      <td class="p-2"><CategoryIcon category={gap.category} /></td>
                      <td class="p-2"><SeverityBadge severity={gap.severity} /></td>
                      <td class="p-2"><StatusBadge status={gap.status} /></td>
                      <td class="p-2">{gap.confidence.level}%</td>
                      <td class="p-2">{gap.impactScore}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Detail Panel */}
        <div class="rounded-lg border border-border bg-card overflow-hidden">
          <div class="p-3 border-b border-border">
            <h3 class="font-semibold">Gap Details</h3>
          </div>
          <div class="p-4 max-h-[calc(100vh-400px)] overflow-auto">
            {selectedGap ? (
              <GapDetailPanel
                gap={selectedGap}
                onFillGap={handleFillGap}
                onActivateResearch={handleActivateResearch}
              />
            ) : (
              <div class="text-center text-muted-foreground py-8">
                <div class="text-2xl mb-2">&lt;-</div>
                <div>Select a gap to view details</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Quick Actions Footer */}
      <div class="flex items-center justify-between p-3 rounded-lg border border-border bg-card">
        <div class="flex items-center gap-4">
          <button
            onClick$={() => dispatchAction('epistemic.gap.discover', { actor: 'dashboard' })}
            class="px-3 py-1.5 rounded text-xs bg-purple-500/20 text-purple-400 hover:bg-purple-500/30 transition-colors"
          >
            + Discover Gap
          </button>
          <button
            onClick$={() => dispatchAction('sota.research.batch', { gaps: gaps.value.filter(g => g.status === 'open').map(g => g.id) })}
            class="px-3 py-1.5 rounded text-xs bg-blue-500/20 text-blue-400 hover:bg-blue-500/30 transition-colors"
          >
            Batch Research
          </button>
        </div>
        <div class="text-xs text-muted-foreground">
          Bus: epistemic.* | SOTA pipeline integrated
        </div>
      </div>
    </div>
  );
});

export default EpistemicGapView;
