import type {
  TypeState,
  TypeTouchOverride,
  TypesNode,
} from './types-schema';

export type TypesProposalStatus = 'pending' | 'preview' | 'archived';
export type TypesProposalKind = 'update' | 'create';

export interface TypesProposalPatch {
  label?: string;
  summary?: string;
  axes?: TypeTouchOverride;
  state?: TypeState;
  evidence?: string[];
  constraints?: string[];
  tags?: string[];
}

export interface TypesProposal {
  id: string;
  nodeId: string;
  createdIso: string;
  actor: string;
  kind: TypesProposalKind;
  status: TypesProposalStatus;
  patch: TypesProposalPatch;
  notes?: string;
  parentId?: string | null;
  proposedNode?: Partial<TypesNode>;
}

const STORAGE_KEY = 'pluribus.types.proposals.v1';

const isBrowserStorage = (): boolean =>
  typeof window !== 'undefined' && typeof localStorage !== 'undefined';

const mergeAxes = (
  base?: TypeTouchOverride,
  next?: TypeTouchOverride
): TypeTouchOverride | undefined => {
  if (!base) return next;
  if (!next) return base;
  return {
    ...base,
    ...next,
    uncertainty: {
      ...(base.uncertainty ?? {}),
      ...(next.uncertainty ?? {}),
    },
    teleology: {
      ...(base.teleology ?? {}),
      ...(next.teleology ?? {}),
    },
  };
};

const mergePatch = (
  base: TypesProposalPatch,
  next: TypesProposalPatch
): TypesProposalPatch => ({
  ...base,
  ...next,
  axes: mergeAxes(base.axes, next.axes),
});

export const loadTypesProposals = (): TypesProposal[] => {
  if (!isBrowserStorage()) return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as TypesProposal[];
    if (!Array.isArray(parsed)) return [];
    return parsed;
  } catch {
    return [];
  }
};

export const saveTypesProposals = (proposals: TypesProposal[]): void => {
  if (!isBrowserStorage()) return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(proposals));
  } catch {
    // ignore storage failures
  }
};

export const appendTypesProposal = (proposal: TypesProposal): TypesProposal[] => {
  const current = loadTypesProposals();
  const next = [...current, proposal];
  saveTypesProposals(next);
  return next;
};

export const updateTypesProposalStatus = (
  proposals: TypesProposal[],
  id: string,
  status: TypesProposalStatus
): TypesProposal[] => {
  const next = proposals.map((proposal) =>
    proposal.id === id ? { ...proposal, status } : proposal
  );
  saveTypesProposals(next);
  return next;
};

export const buildTypesProposalOverrides = (
  proposals: TypesProposal[],
  statuses: TypesProposalStatus[] = ['preview']
): Record<string, TypesProposalPatch> => {
  const overrides: Record<string, TypesProposalPatch> = {};
  const filtered = proposals.filter((proposal) => statuses.includes(proposal.status));
  const sorted = [...filtered].sort((a, b) =>
    a.createdIso.localeCompare(b.createdIso)
  );
  for (const proposal of sorted) {
    const existing = overrides[proposal.nodeId] ?? {};
    overrides[proposal.nodeId] = mergePatch(existing, proposal.patch);
  }
  return overrides;
};
