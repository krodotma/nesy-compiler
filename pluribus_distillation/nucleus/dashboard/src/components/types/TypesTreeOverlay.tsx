import {
  component$,
  useSignal,
  useStore,
  useVisibleTask$,
  $,
  noSerialize,
  type NoSerialize,
} from '@builder.io/qwik';
import { createBusClient, type BusClient } from '../../lib/bus/bus-client';
import type { BusEvent } from '../../lib/state/types';
import {
  TYPES_SCHEMA,
  TYPES_LAYOUT,
  SEXTET_SEMANTICS,
  TYPE_MUTABILITY,
  TYPE_POLYMORPHISM,
  TYPE_SCOPE,
  TYPE_AGENCY,
  TYPE_HORIZON,
  TYPE_STATES,
  getTypesPath,
  resolveTypeAxes,
  type TypeAgency,
  type TypeMutability,
  type TypePolymorphism,
  type TypeScope,
  type TypeHorizon,
  type TypeState,
  type TypeTouchAxes,
  type TypesNode,
} from '../../lib/types-schema';
import {
  appendTypesProposal,
  buildTypesProposalOverrides,
  loadTypesProposals,
  updateTypesProposalStatus,
  type TypesProposal,
  type TypesProposalPatch,
} from '../../lib/types-proposals';

// M3 Components - TypesTreeOverlay
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/button/text-button.js';
import '@material/web/iconbutton/icon-button.js';
import '@material/web/textfield/outlined-text-field.js';

interface TypesTreeOverlayProps {
  onClose$?: any;
  defaultNodeId?: string;
}

interface TypesDraft {
  label: string;
  summary: string;
  mutability: TypeMutability;
  polymorphism: TypePolymorphism;
  scope: TypeScope;
  agency: TypeAgency;
  epistemic: number;
  aleatoric: number;
  teleologyPurpose: string;
  teleologyHorizon: TypeHorizon;
  teleologyFitness: string;
  state: TypeState;
  evidence: string;
  constraints: string;
  notes: string;
}

const PROPOSAL_TOPIC = 'types.proposal.request';
const PROPOSAL_ACTOR = 'types-ui';

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
const formatPercent = (value: number) => `${Math.round(value * 100)}%`;

const listFromText = (value: string): string[] =>
  value
    .split(/[\n,]/)
    .map((item) => item.trim())
    .filter((item) => item.length > 0);

const listToText = (items: string[] = []): string => items.join('\n');

const makeId = (): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `types_${Date.now()}_${Math.random().toString(16).slice(2)}`;
};

const applyNodeOverride = (
  node: TypesNode,
  patch?: TypesProposalPatch
): TypesNode => {
  if (!patch) return node;
  return {
    ...node,
    ...patch,
  };
};

const buildDraftFromNode = (node: TypesNode, axes: TypeTouchAxes): TypesDraft => ({
  label: node.label,
  summary: node.summary,
  mutability: axes.mutability,
  polymorphism: axes.polymorphism,
  scope: axes.scope,
  agency: axes.agency,
  epistemic: axes.uncertainty.epistemic,
  aleatoric: axes.uncertainty.aleatoric,
  teleologyPurpose: axes.teleology.purpose,
  teleologyHorizon: axes.teleology.horizon,
  teleologyFitness: axes.teleology.fitness,
  state: node.state ?? 'active',
  evidence: listToText(node.evidence ?? []),
  constraints: listToText(node.constraints ?? []),
  notes: '',
});

const buildPatchFromDraft = (draft: TypesDraft): TypesProposalPatch => ({
  label: draft.label.trim(),
  summary: draft.summary.trim(),
  axes: {
    mutability: draft.mutability,
    polymorphism: draft.polymorphism,
    scope: draft.scope,
    agency: draft.agency,
    uncertainty: {
      epistemic: clamp01(draft.epistemic),
      aleatoric: clamp01(draft.aleatoric),
    },
    teleology: {
      purpose: draft.teleologyPurpose.trim(),
      horizon: draft.teleologyHorizon,
      fitness: draft.teleologyFitness.trim(),
    },
  },
  state: draft.state,
  evidence: listFromText(draft.evidence),
  constraints: listFromText(draft.constraints),
});

// Recursive Tree Item Component
const TreeItem = component$<{
  node: TypesNode;
  selectedId: string;
  onSelect$: (id: string) => void;
  depth: number;
  query?: string;
  overrides?: Record<string, TypesProposalPatch>;
}>(({ node, selectedId, onSelect$, depth, query = '', overrides }) => {
  const displayNode = applyNodeOverride(node, overrides?.[node.id]);
  const isSelected = displayNode.id === selectedId;
  const hasChildren = displayNode.children && displayNode.children.length > 0;
  const hasOverride = Boolean(overrides?.[node.id]);
  
  // Filtering logic
  const matchesSelf =
    query === '' ||
    displayNode.label.toLowerCase().includes(query.toLowerCase()) ||
    displayNode.id.toLowerCase().includes(query.toLowerCase());
  const matchesChild = hasChildren && displayNode.children!.some((c) => {
    const childDisplay = applyNodeOverride(c, overrides?.[c.id]);
    // Basic flat check for immediate children to decide if we should even render this branch
    return (
      childDisplay.label.toLowerCase().includes(query.toLowerCase()) ||
      childDisplay.id.toLowerCase().includes(query.toLowerCase()) ||
      (childDisplay.children && childDisplay.children.length > 0)
    );
  });

  // Simplified: always show if query is empty, otherwise show if self matches or we're in a match path.
  // In a real robust implementation, we'd pre-calculate the visible set.
  if (query !== '' && !matchesSelf && !matchesChild) return null;

  const isExpanded =
    query !== '' ||
    depth < 1 ||
    isSelected ||
    (TYPES_LAYOUT.indexById[selectedId] && getTypesPath(selectedId).includes(node.id));

  return (
    <div class={`types-tree-item ${matchesSelf && query !== '' ? 'filter-match' : ''}`}>
      <div 
        class={`types-tree-row ${isSelected ? 'selected' : ''}`}
        onClick$={(e) => {
          e.stopPropagation();
          onSelect$(node.id);
        }}
        style={{ paddingLeft: `${depth * 16}px` }}
      >
        <span class="types-tree-icon">
          {hasChildren ? (isExpanded ? '📂' : '📁') : '📄'}
        </span>
        <span class="types-tree-label">{displayNode.label}</span>
        {hasOverride && <span class="types-tree-delta">PREVIEW</span>}
        {displayNode.semantics && displayNode.semantics.length > 0 && (
          <span class="types-tree-badges">
            {displayNode.semantics.map((s) => (
              <span key={s} class="types-badge semantic">{s[0]}</span>
            ))}
          </span>
        )}
      </div>
      {hasChildren && (
        <div class={`types-tree-children ${isExpanded ? 'open' : 'closed'}`}>
          {displayNode.children!.map((child) => (
            <TreeItem
              key={child.id}
              node={child}
              selectedId={selectedId}
              onSelect$={onSelect$}
              depth={depth + 1}
              query={query}
              overrides={overrides}
            />
          ))}
        </div>
      )}
    </div>
  );
});

export const TypesTreeOverlay = component$<TypesTreeOverlayProps>(({ onClose$, defaultNodeId }) => {
  const initialId = defaultNodeId && TYPES_LAYOUT.indexById[defaultNodeId]
    ? defaultNodeId
    : 'types-root';
  const selectedId = useSignal(
    TYPES_LAYOUT.indexById[initialId] ? initialId : TYPES_SCHEMA.id
  );
  const proposals = useSignal<TypesProposal[]>([]);
  const proposalError = useSignal<string | null>(null);
  const proposalStatus = useSignal<string | null>(null);
  const busConnected = useSignal(false);
  const busClientRef = useSignal<NoSerialize<BusClient> | null>(null);
  const searchQuery = useSignal('');
  const isEditMode = useSignal(false);

  useVisibleTask$(({ cleanup }) => {
    proposals.value = loadTypesProposals();
    const client = createBusClient({ platform: 'browser' });
    busClientRef.value = noSerialize(client);
    client.connect().then(() => {
      busConnected.value = true;
    }).catch(() => {
      busConnected.value = false;
    });
    cleanup(() => {
      client.disconnect();
    });
  });

  const previewOverrides = buildTypesProposalOverrides(proposals.value, ['preview']);
  const selectedNode =
    TYPES_LAYOUT.indexById[selectedId.value] ?? TYPES_LAYOUT.indexById[TYPES_SCHEMA.id];
  const displayNode = applyNodeOverride(selectedNode, previewOverrides[selectedNode.id]);
  const selectedAxes = resolveTypeAxes(displayNode);
  const selectedState = displayNode.state ?? 'active';
  const selectedEvidence = displayNode.evidence ?? [];
  const selectedConstraints = displayNode.constraints ?? [];
  const draft = useStore<TypesDraft>(buildDraftFromNode(displayNode, selectedAxes));

  const syncDraftForNode = $((nodeId: string) => {
    const node = TYPES_LAYOUT.indexById[nodeId];
    if (!node) return;
    const overrides = buildTypesProposalOverrides(proposals.value, ['preview']);
    const merged = applyNodeOverride(node, overrides[nodeId]);
    const axes = resolveTypeAxes(merged);
    const nextDraft = buildDraftFromNode(merged, axes);
    Object.assign(draft, nextDraft);
  });

  const selectNode = $((nodeId: string) => {
    if (TYPES_LAYOUT.indexById[nodeId]) {
      selectedId.value = nodeId;
      if (isEditMode.value) {
        syncDraftForNode(nodeId);
      }
    }
  });

  const toggleEditMode = $(() => {
    isEditMode.value = !isEditMode.value;
    proposalError.value = null;
    proposalStatus.value = null;
    if (isEditMode.value) {
      syncDraftForNode(selectedId.value);
    }
  });

  const setProposalStatus = $((proposalId: string, status: 'pending' | 'preview' | 'archived') => {
    proposals.value = updateTypesProposalStatus(proposals.value, proposalId, status);
  });

  const submitProposal = $(async () => {
    proposalError.value = null;
    proposalStatus.value = null;

    const label = draft.label.trim();
    const summary = draft.summary.trim();
    if (!label || !summary) {
      proposalError.value = 'Label and summary are required.';
      return;
    }

    const proposal: TypesProposal = {
      id: makeId(),
      nodeId: selectedNode.id,
      createdIso: new Date().toISOString(),
      actor: PROPOSAL_ACTOR,
      kind: 'update',
      status: 'pending',
      patch: buildPatchFromDraft(draft),
      notes: draft.notes.trim() || undefined,
      parentId: selectedNode.parentId ?? null,
    };

    proposals.value = appendTypesProposal(proposal);
    proposalStatus.value = 'Proposal queued locally.';

    const client = busClientRef.value;
    if (client) {
      const event: Omit<BusEvent, 'ts' | 'iso'> = {
        id: proposal.id,
        topic: PROPOSAL_TOPIC,
        kind: 'request',
        level: 'info',
        actor: PROPOSAL_ACTOR,
        data: {
          proposal_id: proposal.id,
          node_id: proposal.nodeId,
          created_iso: proposal.createdIso,
          patch: proposal.patch,
          notes: proposal.notes,
          parent_id: proposal.parentId,
        },
        semantic: `types proposal for ${proposal.nodeId}`,
      };
      try {
        await client.publish(event);
      } catch {
        proposalStatus.value = 'Proposal queued locally (bus publish failed).';
      }
    }
  });

  // Find related nodes by semantics
  const relatedNodes = TYPES_LAYOUT.nodes
    .filter(
      (n) =>
        n.id !== selectedNode.id &&
        n.semantics.some((s) => displayNode.semantics.includes(s))
    )
    .slice(0, 5);
  const selectedProposals = [...proposals.value]
    .filter((proposal) => proposal.nodeId === selectedNode.id)
    .sort((a, b) => b.createdIso.localeCompare(a.createdIso));
  const parentNode = selectedNode.parentId
    ? applyNodeOverride(
        TYPES_LAYOUT.indexById[selectedNode.parentId],
        previewOverrides[selectedNode.parentId]
      )
    : null;

  return (
    <section class="types-overlay">
      <div class="types-overlay-header">
        <div>
          <div class="types-overlay-overline">Types Atlas</div>
          <h2 class="types-overlay-title">Hierarchy & Schema Editor</h2>
          <p class="types-overlay-subtitle">
            Navigate and manage the formal schema of the Pluribus organism.
          </p>
        </div>
        <div class="flex items-center gap-2">
          <button 
            class={`types-mode-toggle ${isEditMode.value ? 'active' : ''}`}
            onClick$={toggleEditMode}
          >
            {isEditMode.value ? 'Exit Edit' : 'Draft Proposal'}
          </button>
          {onClose$ && (
            <button class="types-overlay-close" onClick$={onClose$}>
              Close
            </button>
          )}
        </div>
      </div>

      <div class="types-overlay-body split-view">
        {/* Left: Tree Navigator */}
        <div class="types-navigator">
          <div class="types-navigator-toolbar">
            <input 
              type="text" 
              placeholder="Search types..." 
              class="types-search-input"
              bind:value={searchQuery}
            />
          </div>
          <div class="types-tree-scroll">
            <TreeItem 
              node={TYPES_SCHEMA} 
              selectedId={selectedId.value} 
              onSelect$={selectNode} 
              depth={0} 
              query={searchQuery.value}
              overrides={previewOverrides}
            />
          </div>
        </div>

        {/* Right: Inspector / Editor */}
        <aside class="types-detail-panel">
          <div class="types-detail-card">
            <div class="types-detail-header">
              <div class="flex items-center gap-2">
                {isEditMode.value ? (
                  <input
                    type="text"
                    value={draft.label}
                    onInput$={(event) => {
                      draft.label = (event.target as HTMLInputElement).value;
                    }}
                    class="types-edit-input"
                  />
                ) : (
                  <h3>{displayNode.label}</h3>
                )}
                <span class="types-detail-id-badge">{selectedNode.id}</span>
              </div>
            </div>
            
            {isEditMode.value ? (
              <textarea
                class="types-edit-textarea"
                value={draft.summary}
                onInput$={(event) => {
                  draft.summary = (event.target as HTMLTextAreaElement).value;
                }}
              />
            ) : (
              <p class="types-detail-summary">{displayNode.summary}</p>
            )}

            <div class="types-detail-section">
              <div class="types-detail-label">Sextet Semantics</div>
              <div class="types-chip-group">
                {displayNode.semantics.map((item) => (
                  <span key={item} class="types-chip semantics">{item}</span>
                ))}
              </div>
              {isEditMode.value && (
                <div class="types-muted">Semantics are anchored to Sextet governance.</div>
              )}
            </div>

            <div class="types-detail-section">
              <div class="types-detail-label">TypeTouch Axes</div>
              {isEditMode.value ? (
                <div class="types-axes-editor">
                  <label class="types-axis-field">
                    <span>Mutability</span>
                    <select
                      class="types-edit-select"
                      value={draft.mutability}
                      onChange$={(event) => {
                        draft.mutability = (event.target as HTMLSelectElement).value as TypeMutability;
                      }}
                    >
                      {TYPE_MUTABILITY.map((item) => (
                        <option key={item} value={item}>{item}</option>
                      ))}
                    </select>
                  </label>
                  <label class="types-axis-field">
                    <span>Polymorphism</span>
                    <select
                      class="types-edit-select"
                      value={draft.polymorphism}
                      onChange$={(event) => {
                        draft.polymorphism = (event.target as HTMLSelectElement).value as TypePolymorphism;
                      }}
                    >
                      {TYPE_POLYMORPHISM.map((item) => (
                        <option key={item} value={item}>{item}</option>
                      ))}
                    </select>
                  </label>
                  <label class="types-axis-field">
                    <span>Scope</span>
                    <select
                      class="types-edit-select"
                      value={draft.scope}
                      onChange$={(event) => {
                        draft.scope = (event.target as HTMLSelectElement).value as TypeScope;
                      }}
                    >
                      {TYPE_SCOPE.map((item) => (
                        <option key={item} value={item}>{item}</option>
                      ))}
                    </select>
                  </label>
                  <label class="types-axis-field">
                    <span>Agency</span>
                    <select
                      class="types-edit-select"
                      value={draft.agency}
                      onChange$={(event) => {
                        draft.agency = (event.target as HTMLSelectElement).value as TypeAgency;
                      }}
                    >
                      {TYPE_AGENCY.map((item) => (
                        <option key={item} value={item}>{item}</option>
                      ))}
                    </select>
                  </label>
                </div>
              ) : (
                <div class="types-axes-grid">
                  <div class="types-axis">
                    <span class="types-axis-label">Mutability</span>
                    <span class="types-chip axis">{selectedAxes.mutability}</span>
                  </div>
                  <div class="types-axis">
                    <span class="types-axis-label">Polymorphism</span>
                    <span class="types-chip axis">{selectedAxes.polymorphism}</span>
                  </div>
                  <div class="types-axis">
                    <span class="types-axis-label">Scope</span>
                    <span class="types-chip axis">{selectedAxes.scope}</span>
                  </div>
                  <div class="types-axis">
                    <span class="types-axis-label">Agency</span>
                    <span class="types-chip axis">{selectedAxes.agency}</span>
                  </div>
                </div>
              )}
            </div>

            <div class="types-detail-section">
              <div class="types-detail-label">Uncertainty</div>
              {isEditMode.value ? (
                <div class="types-uncertainty-edit">
                  <label class="types-uncertainty-control">
                    <span>Epistemic</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={draft.epistemic}
                      class="types-range"
                      onInput$={(event) => {
                        const value = parseFloat((event.target as HTMLInputElement).value);
                        draft.epistemic = clamp01(Number.isFinite(value) ? value : 0);
                      }}
                    />
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={draft.epistemic}
                      class="types-number"
                      onInput$={(event) => {
                        const value = parseFloat((event.target as HTMLInputElement).value);
                        draft.epistemic = clamp01(Number.isFinite(value) ? value : 0);
                      }}
                    />
                  </label>
                  <label class="types-uncertainty-control">
                    <span>Aleatoric</span>
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.05"
                      value={draft.aleatoric}
                      class="types-range"
                      onInput$={(event) => {
                        const value = parseFloat((event.target as HTMLInputElement).value);
                        draft.aleatoric = clamp01(Number.isFinite(value) ? value : 0);
                      }}
                    />
                    <input
                      type="number"
                      min="0"
                      max="1"
                      step="0.01"
                      value={draft.aleatoric}
                      class="types-number"
                      onInput$={(event) => {
                        const value = parseFloat((event.target as HTMLInputElement).value);
                        draft.aleatoric = clamp01(Number.isFinite(value) ? value : 0);
                      }}
                    />
                  </label>
                </div>
              ) : (
                <div class="types-uncertainty">
                  <div class="types-uncertainty-row">
                    <span class="types-uncertainty-label">Epistemic</span>
                    <div class="types-uncertainty-bar">
                      <span style={{ width: formatPercent(selectedAxes.uncertainty.epistemic) }} />
                    </div>
                    <span class="types-uncertainty-value">{formatPercent(selectedAxes.uncertainty.epistemic)}</span>
                  </div>
                  <div class="types-uncertainty-row">
                    <span class="types-uncertainty-label">Aleatoric</span>
                    <div class="types-uncertainty-bar">
                      <span style={{ width: formatPercent(selectedAxes.uncertainty.aleatoric) }} />
                    </div>
                    <span class="types-uncertainty-value">{formatPercent(selectedAxes.uncertainty.aleatoric)}</span>
                  </div>
                </div>
              )}
            </div>

            <div class="types-detail-section">
              <div class="types-detail-label">Teleology</div>
              {isEditMode.value ? (
                <div class="types-teleology-edit">
                  <label>
                    <span>Purpose</span>
                    <input
                      class="types-edit-input"
                      value={draft.teleologyPurpose}
                      onInput$={(event) => {
                        draft.teleologyPurpose = (event.target as HTMLInputElement).value;
                      }}
                    />
                  </label>
                  <label>
                    <span>Horizon</span>
                    <select
                      class="types-edit-select"
                      value={draft.teleologyHorizon}
                      onChange$={(event) => {
                        draft.teleologyHorizon = (event.target as HTMLSelectElement).value as TypeHorizon;
                      }}
                    >
                      {TYPE_HORIZON.map((item) => (
                        <option key={item} value={item}>{item}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>Fitness</span>
                    <input
                      class="types-edit-input"
                      value={draft.teleologyFitness}
                      onInput$={(event) => {
                        draft.teleologyFitness = (event.target as HTMLInputElement).value;
                      }}
                    />
                  </label>
                </div>
              ) : (
                <div class="types-teleology">
                  <div class="types-teleology-purpose">{selectedAxes.teleology.purpose}</div>
                  <div class="types-teleology-meta">
                    <span class="types-chip axis">{selectedAxes.teleology.horizon}</span>
                    <span class="types-teleology-fitness">{selectedAxes.teleology.fitness}</span>
                  </div>
                </div>
              )}
            </div>

            {/* Relationships Section */}
            <div class="types-detail-section">
              <div class="types-detail-label">Lineage & Relationships</div>
              <div class="types-relationship-grid">
                <div class="types-rel-box">
                  <div class="types-rel-label">Parent</div>
                  <div class="types-rel-value">
                    {selectedNode.parentId ? (
                      <button class="types-rel-link" onClick$={() => selectNode(selectedNode.parentId!)}>
                        ↑ {parentNode?.label ?? TYPES_LAYOUT.indexById[selectedNode.parentId].label}
                      </button>
                    ) : 'Root'}
                  </div>
                </div>
                <div class="types-rel-box">
                  <div class="types-rel-label">Children</div>
                  <div class="types-rel-value">
                    {selectedNode.children?.length || 0} nodes
                  </div>
                </div>
              </div>
            </div>

            {relatedNodes.length > 0 && (
              <div class="types-detail-section">
                <div class="types-detail-label">Semantic Peers</div>
                <div class="flex flex-wrap gap-1 mt-1">
                  {relatedNodes.map((peer) => {
                    const peerDisplay = applyNodeOverride(peer, previewOverrides[peer.id]);
                    return (
                      <button
                        key={peer.id}
                        class="types-rel-peer"
                        onClick$={() => selectNode(peer.id)}
                      >
                        🔗 {peerDisplay.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div class="types-detail-section">
              <div class="types-detail-label">Mechanism</div>
              <div class="types-chip-group">
                {displayNode.mechanism.map((item) => (
                  <span key={item} class="types-chip mechanism">{item}</span>
                ))}
              </div>
            </div>

            {displayNode.auom && displayNode.auom.length > 0 && (
              <div class="types-detail-section">
                <div class="types-detail-label">AuOM Flows</div>
                <div class="types-chip-group">
                  {displayNode.auom.map((item) => (
                    <span key={item} class="types-chip auom">{item}</span>
                  ))}
                </div>
              </div>
            )}

            <div class="types-detail-section">
              <div class="types-detail-label">State & Evidence</div>
              {isEditMode.value ? (
                <div class="types-state-edit">
                  <label>
                    <span>State</span>
                    <select
                      class="types-edit-select"
                      value={draft.state}
                      onChange$={(event) => {
                        draft.state = (event.target as HTMLSelectElement).value as TypeState;
                      }}
                    >
                      {TYPE_STATES.map((item) => (
                        <option key={item} value={item}>{item}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    <span>Evidence</span>
                    <textarea
                      class="types-edit-textarea"
                      value={draft.evidence}
                      onInput$={(event) => {
                        draft.evidence = (event.target as HTMLTextAreaElement).value;
                      }}
                    />
                  </label>
                </div>
              ) : (
                <>
                  <div class="types-chip-group">
                    <span class="types-chip state">{selectedState}</span>
                  </div>
                  {selectedEvidence.length > 0 ? (
                    <div class="types-evidence">
                      {selectedEvidence.map((item) => (
                        <span key={item} class="types-evidence-item">{item}</span>
                      ))}
                    </div>
                  ) : (
                    <div class="types-muted">No evidence recorded yet.</div>
                  )}
                </>
              )}
            </div>

            <div class="types-detail-section">
              <div class="types-detail-label">Constraints</div>
              {isEditMode.value ? (
                <textarea
                  class="types-edit-textarea"
                  value={draft.constraints}
                  onInput$={(event) => {
                    draft.constraints = (event.target as HTMLTextAreaElement).value;
                  }}
                />
              ) : (
                <div class="types-chip-group">
                  {selectedConstraints.length > 0 ? (
                    selectedConstraints.map((item) => (
                      <span key={item} class="types-chip constraint">{item}</span>
                    ))
                  ) : (
                    <span class="types-muted">No constraints specified.</span>
                  )}
                </div>
              )}
            </div>

            {isEditMode.value && (
              <div class="types-detail-section">
                <div class="types-detail-label">Proposal Actions</div>
                <label class="types-proposal-notes">
                  <span>Notes</span>
                  <textarea
                    class="types-edit-textarea"
                    value={draft.notes}
                    onInput$={(event) => {
                      draft.notes = (event.target as HTMLTextAreaElement).value;
                    }}
                  />
                </label>
                <div class="types-proposal-actions">
                  <button class="types-proposal-submit" onClick$={submitProposal}>
                    Submit Proposal
                  </button>
                  <span class="types-proposal-bus">
                    {busConnected.value ? 'Bus online' : 'Bus offline'}
                  </span>
                </div>
                {proposalError.value && (
                  <div class="types-error">{proposalError.value}</div>
                )}
                {proposalStatus.value && (
                  <div class="types-muted">{proposalStatus.value}</div>
                )}
              </div>
            )}
            
            <div class="types-detail-section">
              <div class="types-detail-label">Definition JSON</div>
              <pre class="types-json-preview">
{JSON.stringify({
  id: selectedNode.id,
  label: displayNode.label,
  semantics: displayNode.semantics,
  mechanism: displayNode.mechanism,
  axes: selectedAxes,
  state: selectedState,
  evidence: selectedEvidence,
  constraints: selectedConstraints
}, null, 2)}
              </pre>
            </div>
          </div>

          <div class="types-detail-card">
            <div class="types-proposal-header">
              <div>
                <div class="types-detail-label">Proposal Queue</div>
                <div class="types-muted">{selectedProposals.length} proposals for this node</div>
              </div>
            </div>
            {selectedProposals.length === 0 ? (
              <div class="types-muted">No proposals yet. Draft one to begin.</div>
            ) : (
              <div class="types-proposal-list">
                {selectedProposals.map((proposal) => {
                  const patchKeys = Object.keys(proposal.patch || {}).filter(
                    (key) => (proposal.patch as Record<string, unknown>)[key] !== undefined
                  );
                  return (
                    <div key={proposal.id} class={`types-proposal-item ${proposal.status}`}>
                      <div class="types-proposal-meta">
                        <div class="types-proposal-meta-left">
                          <span class="types-proposal-id">{proposal.id.slice(0, 8)}</span>
                          <span class="types-proposal-time">{proposal.createdIso}</span>
                        </div>
                        <span class="types-chip state">{proposal.status}</span>
                      </div>
                      <div class="types-proposal-summary">
                        Patch: {patchKeys.length > 0 ? patchKeys.join(', ') : 'none'}
                      </div>
                      {proposal.notes ? (
                        <div class="types-proposal-notes-text">{proposal.notes}</div>
                      ) : (
                        <div class="types-muted">No notes attached.</div>
                      )}
                      <div class="types-proposal-actions">
                        <button
                          class="types-proposal-preview"
                          onClick$={() => setProposalStatus(
                            proposal.id,
                            proposal.status === 'preview' ? 'pending' : 'preview'
                          )}
                        >
                          {proposal.status === 'preview' ? 'Clear Preview' : 'Preview'}
                        </button>
                        <button
                          class="types-proposal-archive"
                          onClick$={() => setProposalStatus(proposal.id, 'archived')}
                        >
                          Archive
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div class="types-detail-card">
            <h4 class="types-detail-label">Legend</h4>
            <div class="types-legend-row">
              <span class="types-legend-title">Semantics</span>
              <div class="types-chip-group">
                {SEXTET_SEMANTICS.map((item) => (
                  <span key={item} class="types-chip semantics">{item}</span>
                ))}
              </div>
            </div>
            <div class="types-legend-row">
              <span class="types-legend-title">Mutability</span>
              <div class="types-chip-group">
                {TYPE_MUTABILITY.map((item) => (
                  <span key={item} class="types-chip axis">{item}</span>
                ))}
              </div>
            </div>
            <div class="types-legend-row">
              <span class="types-legend-title">Polymorphism</span>
              <div class="types-chip-group">
                {TYPE_POLYMORPHISM.map((item) => (
                  <span key={item} class="types-chip axis">{item}</span>
                ))}
              </div>
            </div>
            <div class="types-legend-row">
              <span class="types-legend-title">Scope</span>
              <div class="types-chip-group">
                {TYPE_SCOPE.map((item) => (
                  <span key={item} class="types-chip axis">{item}</span>
                ))}
              </div>
            </div>
            <div class="types-legend-row">
              <span class="types-legend-title">Agency</span>
              <div class="types-chip-group">
                {TYPE_AGENCY.map((item) => (
                  <span key={item} class="types-chip axis">{item}</span>
                ))}
              </div>
            </div>
            <div class="types-legend-row">
              <span class="types-legend-title">State</span>
              <div class="types-chip-group">
                {TYPE_STATES.map((item) => (
                  <span key={item} class="types-chip state">{item}</span>
                ))}
              </div>
            </div>
          </div>
        </aside>
      </div>
    </section>
  );
});

export default TypesTreeOverlay;
