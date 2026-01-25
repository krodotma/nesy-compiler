export const SEXTET_SEMANTICS = [
  'Object',
  'Process',
  'Type',
  'Shape',
  'Symbol',
  'Observer',
] as const;

export type SextetSemantic = (typeof SEXTET_SEMANTICS)[number];

export const SEXTET_MECHANISM = [
  'source',
  'regulator',
  'transducer',
  'reference',
  'memory',
  'feedback',
] as const;

export type SextetMechanism = (typeof SEXTET_MECHANISM)[number];

export const TYPE_MUTABILITY = ['immutable', 'bounded', 'mutable'] as const;
export type TypeMutability = (typeof TYPE_MUTABILITY)[number];

export const TYPE_POLYMORPHISM = ['none', 'parametric', 'bounded', 'ad-hoc'] as const;
export type TypePolymorphism = (typeof TYPE_POLYMORPHISM)[number];

export const TYPE_SCOPE = ['global', 'local', 'session', 'artifact'] as const;
export type TypeScope = (typeof TYPE_SCOPE)[number];

export const TYPE_AGENCY = ['passive', 'assistive', 'agentic', 'autonomous'] as const;
export type TypeAgency = (typeof TYPE_AGENCY)[number];

export const TYPE_HORIZON = ['instant', 'near', 'mid', 'long', 'infinite'] as const;
export type TypeHorizon = (typeof TYPE_HORIZON)[number];

export const TYPE_STATES = ['draft', 'active', 'deprecated', 'archived'] as const;
export type TypeState = (typeof TYPE_STATES)[number];

export interface TypeUncertainty {
  epistemic: number;
  aleatoric: number;
}

export interface TypeTeleology {
  purpose: string;
  horizon: TypeHorizon;
  fitness: string;
}

export interface TypeTouchAxes {
  mutability: TypeMutability;
  polymorphism: TypePolymorphism;
  scope: TypeScope;
  agency: TypeAgency;
  uncertainty: TypeUncertainty;
  teleology: TypeTeleology;
}

export type TypeTouchOverride = Partial<Omit<TypeTouchAxes, 'uncertainty' | 'teleology'>> & {
  uncertainty?: Partial<TypeUncertainty>;
  teleology?: Partial<TypeTeleology>;
};

export interface TypesNode {
  id: string;
  label: string;
  summary: string;
  semantics: SextetSemantic[];
  mechanism: SextetMechanism[];
  auom?: string[];
  tags?: string[];
  axes?: TypeTouchOverride;
  state?: TypeState;
  evidence?: string[];
  constraints?: string[];
  children?: TypesNode[];
}

export interface TypesLayoutNode extends TypesNode {
  depth: number;
  parentId: string | null;
  x: number;
  y: number;
  z: number;
}

export interface TypesLayoutEdge {
  from: string;
  to: string;
}

export interface TypesLayout {
  nodes: TypesLayoutNode[];
  edges: TypesLayoutEdge[];
  indexById: Record<string, TypesLayoutNode>;
  roots: string[];
}

const DEFAULT_EVIDENCE = [
  'sextet alignment',
  'auom grounding',
  'append-only evidence',
];

const DEFAULT_CONSTRAINTS = [
  'append-only provenance',
  'compatibility checks',
  'reviewed deltas',
];

const DEFAULT_STATE: TypeState = 'active';

const DEFAULT_AXES_BY_SEMANTIC: Record<SextetSemantic, TypeTouchAxes> = {
  Object: {
    mutability: 'bounded',
    polymorphism: 'bounded',
    scope: 'global',
    agency: 'passive',
    uncertainty: { epistemic: 0.35, aleatoric: 0.2 },
    teleology: {
      purpose: 'Stabilize identity for entities, artifacts, and environments.',
      horizon: 'long',
      fitness: 'traceability + persistence',
    },
  },
  Process: {
    mutability: 'mutable',
    polymorphism: 'parametric',
    scope: 'global',
    agency: 'agentic',
    uncertainty: { epistemic: 0.45, aleatoric: 0.4 },
    teleology: {
      purpose: 'Transform state through controlled change and verification.',
      horizon: 'near',
      fitness: 'throughput + coherence',
    },
  },
  Type: {
    mutability: 'bounded',
    polymorphism: 'parametric',
    scope: 'global',
    agency: 'assistive',
    uncertainty: { epistemic: 0.2, aleatoric: 0.15 },
    teleology: {
      purpose: 'Govern meaning, constraints, and interoperability.',
      horizon: 'long',
      fitness: 'consistency + coverage',
    },
  },
  Shape: {
    mutability: 'immutable',
    polymorphism: 'bounded',
    scope: 'global',
    agency: 'passive',
    uncertainty: { epistemic: 0.15, aleatoric: 0.1 },
    teleology: {
      purpose: 'Preserve invariants, topology, and equivariant structure.',
      horizon: 'infinite',
      fitness: 'stability + invariance',
    },
  },
  Symbol: {
    mutability: 'mutable',
    polymorphism: 'ad-hoc',
    scope: 'global',
    agency: 'assistive',
    uncertainty: { epistemic: 0.5, aleatoric: 0.3 },
    teleology: {
      purpose: 'Encode and transmit shared meaning across planes.',
      horizon: 'mid',
      fitness: 'clarity + reach',
    },
  },
  Observer: {
    mutability: 'bounded',
    polymorphism: 'parametric',
    scope: 'global',
    agency: 'agentic',
    uncertainty: { epistemic: 0.4, aleatoric: 0.25 },
    teleology: {
      purpose: 'Sense, verify, and close feedback loops.',
      horizon: 'near',
      fitness: 'signal fidelity + accountability',
    },
  },
};

const cloneAxes = (axes: TypeTouchAxes): TypeTouchAxes => ({
  ...axes,
  uncertainty: { ...axes.uncertainty },
  teleology: { ...axes.teleology },
});

const mergeAxes = (
  semantic: SextetSemantic,
  override?: TypeTouchOverride
): TypeTouchAxes => {
  const base = DEFAULT_AXES_BY_SEMANTIC[semantic];
  if (!override) {
    return cloneAxes(base);
  }
  return {
    ...base,
    ...override,
    uncertainty: { ...base.uncertainty, ...override.uncertainty },
    teleology: { ...base.teleology, ...override.teleology },
  };
};

export const resolveTypeAxes = (node: TypesNode): TypeTouchAxes => {
  const semantic = node.semantics[0] ?? 'Type';
  return mergeAxes(semantic, node.axes);
};

const applyTypeDefaults = (node: TypesNode): TypesNode => {
  const semantic = node.semantics[0] ?? 'Type';
  return {
    ...node,
    axes: mergeAxes(semantic, node.axes),
    state: node.state ?? DEFAULT_STATE,
    evidence: node.evidence ?? DEFAULT_EVIDENCE,
    constraints: node.constraints ?? DEFAULT_CONSTRAINTS,
    children: node.children?.map(applyTypeDefaults),
  };
};

const ROOT_AUOM = ['append-only evidence', 'typed effects', 'omega checks'];

const TYPES_SCHEMA_SEED: TypesNode = {
  id: 'types-root',
  label: 'Types Atlas',
  summary:
    'Formal schema for public-domain knowledge, events, planes, things, and common intelligence data. Anchored to Sextet and AuOM flows.',
  axes: {
    teleology: {
      purpose: 'Bind Pluribus types into a composable, learnable commons.',
      horizon: 'long',
      fitness: 'coherence + leverage',
    },
  },
  constraints: [
    'append-only provenance',
    'sextet alignment',
    'auom boundedness',
  ],
  semantics: ['Type'],
  mechanism: ['reference', 'memory'],
  auom: ROOT_AUOM,
  children: [
    {
      id: 'types-object',
      label: 'Objects',
      summary: 'Things and entities: agents, artifacts, places, and systems.',
      axes: {
        teleology: {
          purpose: 'Anchor identity, ownership, and continuity of entities.',
          horizon: 'long',
          fitness: 'traceability + persistence',
        },
      },
      semantics: ['Object'],
      mechanism: ['memory', 'source'],
      auom: ['typed effects'],
      children: [
        {
          id: 'object-agents',
          label: 'Agents and selves',
          summary: 'Humans, machines, hybrids, offspring, and identity clones.',
          semantics: ['Object'],
          mechanism: ['memory', 'source'],
          auom: ['typed effects'],
        },
        {
          id: 'object-artifacts',
          label: 'Artifacts and tools',
          summary: 'Instruments, datasets, models, and infrastructure.',
          semantics: ['Object'],
          mechanism: ['memory', 'source'],
          auom: ['typed effects'],
        },
        {
          id: 'object-places',
          label: 'Places and environments',
          summary: 'Physical sites, digital spaces, and ecological contexts.',
          semantics: ['Object'],
          mechanism: ['source', 'memory'],
          auom: ['append-only evidence'],
        },
        {
          id: 'object-systems',
          label: 'Systems and institutions',
          summary: 'Economies, organizations, and socio-technical systems.',
          semantics: ['Object'],
          mechanism: ['memory', 'reference'],
          auom: ['typed effects'],
        },
      ],
    },
    {
      id: 'types-process',
      label: 'Processes',
      summary: 'Events, workflows, coordination, and evolution.',
      axes: {
        teleology: {
          purpose: 'Coordinate change, verification, and lineage.',
          horizon: 'near',
          fitness: 'throughput + coherence',
        },
      },
      semantics: ['Process'],
      mechanism: ['transducer', 'feedback'],
      auom: ['append-only evidence'],
      children: [
        {
          id: 'process-events',
          label: 'Events',
          summary: 'Natural, social, operational, and lifecycle changes.',
          semantics: ['Process'],
          mechanism: ['source', 'feedback'],
          auom: ['append-only evidence'],
        },
        {
          id: 'process-pipelines',
          label: 'SOTA to reality pipeline',
          summary:
            'Curation -> distill -> hypothesis -> applied theory -> implementation -> verification -> provenance.',
          axes: {
            uncertainty: { epistemic: 0.55, aleatoric: 0.45 },
            teleology: {
              purpose: 'Turn SOTA into verified, reproducible reality.',
              horizon: 'near',
              fitness: 'provenance + repeatability',
            },
          },
          evidence: ['append-only evidence', 'verification gates'],
          semantics: ['Process'],
          mechanism: ['reference', 'feedback'],
          auom: ['append-only evidence'],
        },
        {
          id: 'process-lineage',
          label: 'Lineage and evolution',
          summary: 'VGT inheritance, guarded HGT, replication, and selection.',
          axes: {
            teleology: {
              purpose: 'Guarded evolution with traceable fitness gains.',
              horizon: 'long',
              fitness: 'robustness + CMP',
            },
          },
          semantics: ['Process'],
          mechanism: ['transducer', 'feedback'],
          auom: ['omega checks'],
        },
        {
          id: 'process-coordination',
          label: 'Coordination and protocols',
          summary: 'Bus events, task graphs, and consensus workflows.',
          semantics: ['Process'],
          mechanism: ['regulator', 'feedback'],
          auom: ['append-only evidence'],
        },
      ],
    },
    {
      id: 'types-type',
      label: 'Types',
      summary: 'Schemas, contracts, units, and policy surfaces.',
      axes: {
        teleology: {
          purpose: 'Normalize meaning and enforce composability.',
          horizon: 'long',
          fitness: 'consistency + coverage',
        },
      },
      semantics: ['Type'],
      mechanism: ['reference', 'regulator'],
      auom: ['typed effects'],
      children: [
        {
          id: 'type-schemas',
          label: 'Schemas and protocols',
          summary: 'Contracts, JSON schemas, and governance protocols.',
          semantics: ['Type'],
          mechanism: ['reference', 'regulator'],
          auom: ['typed effects'],
        },
        {
          id: 'type-units',
          label: 'Units and budgets',
          summary: 'Latency, energy, rights, and cost envelopes.',
          semantics: ['Type'],
          mechanism: ['reference', 'regulator'],
          auom: ['typed effects'],
        },
        {
          id: 'type-rights',
          label: 'Rights and permissions',
          summary: 'Access, safety, and legal boundary definitions.',
          axes: {
            mutability: 'immutable',
            polymorphism: 'bounded',
            teleology: {
              purpose: 'Define safe, auditable access boundaries.',
              horizon: 'long',
              fitness: 'safety + compliance',
            },
          },
          constraints: ['policy gates', 'audit evidence', 'legal alignment'],
          semantics: ['Type'],
          mechanism: ['regulator', 'reference'],
          auom: ['typed effects'],
        },
        {
          id: 'type-data',
          label: 'Data classes',
          summary: 'Public-domain, restricted, and private data tiers.',
          semantics: ['Type'],
          mechanism: ['reference', 'regulator'],
          auom: ['typed effects'],
        },
      ],
    },
    {
      id: 'types-shape',
      label: 'Shapes',
      summary: 'Invariants, planes, motifs, and topology.',
      axes: {
        teleology: {
          purpose: 'Protect invariants that keep the organism coherent.',
          horizon: 'infinite',
          fitness: 'stability + invariance',
        },
      },
      semantics: ['Shape'],
      mechanism: ['reference', 'feedback'],
      auom: ['omega checks'],
      children: [
        {
          id: 'shape-planes',
          label: 'Planes of reality',
          summary: 'Physical, digital, social, cognitive, temporal, ecological.',
          semantics: ['Shape'],
          mechanism: ['reference', 'memory'],
          auom: ['typed effects'],
          children: [
            {
              id: 'plane-physical',
              label: 'Physical plane',
              summary: 'Material objects, energy, geography, and embodiment.',
              semantics: ['Shape'],
              mechanism: ['source', 'reference'],
              auom: ['typed effects'],
            },
            {
              id: 'plane-digital',
              label: 'Digital plane',
              summary: 'Networks, software, computation, and protocols.',
              semantics: ['Shape'],
              mechanism: ['reference', 'memory'],
              auom: ['typed effects'],
            },
            {
              id: 'plane-social',
              label: 'Social plane',
              summary: 'Institutions, norms, and collective behavior.',
              semantics: ['Shape'],
              mechanism: ['reference', 'feedback'],
              auom: ['append-only evidence'],
            },
            {
              id: 'plane-cognitive',
              label: 'Cognitive plane',
              summary: 'Mental models, narratives, and sensemaking.',
              semantics: ['Shape'],
              mechanism: ['transducer', 'reference'],
              auom: ['typed effects'],
            },
            {
              id: 'plane-temporal',
              label: 'Temporal plane',
              summary: 'Cadence, history, sequence, and anticipation.',
              semantics: ['Shape'],
              mechanism: ['reference', 'feedback'],
              auom: ['omega checks'],
            },
            {
              id: 'plane-ecological',
              label: 'Ecological plane',
              summary: 'Biophysical systems, supply, and regeneration.',
              semantics: ['Shape'],
              mechanism: ['source', 'memory'],
              auom: ['append-only evidence'],
            },
          ],
        },
        {
          id: 'shape-motif-superpattern',
          label: 'Motif superpattern',
          summary: 'Recurring subgraph patterns that bind rhizomes, offspring, and guarded transfer.',
          axes: {
            mutability: 'bounded',
            polymorphism: 'ad-hoc',
            agency: 'assistive',
            uncertainty: { epistemic: 0.5, aleatoric: 0.35 },
            teleology: {
              purpose: 'Bind recurring subgraph motifs across agent lineages.',
              horizon: 'long',
              fitness: 'recurrence + verification',
            },
          },
          semantics: ['Shape', 'Process'],
          mechanism: ['reference', 'feedback'],
          auom: ['omega checks', 'append-only evidence'],
          children: [
            {
              id: 'shape-motif-omega',
              label: 'Omega motifs',
              summary: 'Recurrence-bound patterns validated against liveness gates.',
              semantics: ['Shape'],
              mechanism: ['feedback', 'reference'],
              auom: ['omega checks'],
            },
            {
              id: 'shape-motif-rhizome',
              label: 'Rhizome lineages',
              summary: 'Branching inheritance across agentic subgraphs.',
              semantics: ['Shape', 'Process'],
              mechanism: ['memory', 'feedback'],
              auom: ['append-only evidence'],
            },
            {
              id: 'shape-motif-offspring',
              label: 'Offspring and clones',
              summary: 'Self-identity splits, forks, and lineage mirrors.',
              semantics: ['Object', 'Process'],
              mechanism: ['memory', 'reference'],
              auom: ['typed effects'],
            },
            {
              id: 'shape-motif-hgt',
              label: 'Guarded transfer',
              summary: 'VGT inheritance and HGT splicing under P/E/L/R/Q gates.',
              semantics: ['Process'],
              mechanism: ['regulator', 'transducer'],
              auom: ['typed effects'],
            },
            {
              id: 'shape-motif-recurrence',
              label: 'Recurrence gates',
              summary: 'Acceptance checks for infinite-horizon pattern return.',
              semantics: ['Shape'],
              mechanism: ['feedback', 'reference'],
              auom: ['omega checks'],
            },
          ],
        },
        {
          id: 'shape-invariants',
          label: 'Invariants and topology',
          summary: 'Equivariance, geometry, liveness, and topology locks.',
          semantics: ['Shape'],
          mechanism: ['reference', 'feedback'],
          auom: ['omega checks'],
        },
      ],
    },
    {
      id: 'types-symbol',
      label: 'Symbols',
      summary: 'Encodings, languages, knowledge commons, and identifiers.',
      axes: {
        teleology: {
          purpose: 'Encode and transmit shared meaning across agents.',
          horizon: 'mid',
          fitness: 'clarity + reach',
        },
      },
      semantics: ['Symbol'],
      mechanism: ['transducer', 'memory'],
      auom: ['append-only evidence'],
      children: [
        {
          id: 'symbol-encodings',
          label: 'Encodings',
          summary: 'Text, code, images, event streams, and tokens.',
          semantics: ['Symbol'],
          mechanism: ['transducer', 'memory'],
          auom: ['append-only evidence'],
        },
        {
          id: 'symbol-commons',
          label: 'Knowledge commons',
          summary: 'Public-domain science, culture, history, and law.',
          axes: {
            scope: 'global',
            teleology: {
              purpose: 'Keep public-domain knowledge composable and durable.',
              horizon: 'long',
              fitness: 'coverage + integrity',
            },
          },
          semantics: ['Symbol'],
          mechanism: ['memory', 'reference'],
          auom: ['append-only evidence'],
          children: [
            {
              id: 'commons-science',
              label: 'Science',
              summary: 'Empirical findings, reproducible methods, and data.',
              semantics: ['Symbol'],
              mechanism: ['reference', 'memory'],
              auom: ['append-only evidence'],
            },
            {
              id: 'commons-culture',
              label: 'Culture',
              summary: 'Arts, rituals, myths, and shared meaning.',
              semantics: ['Symbol'],
              mechanism: ['memory', 'reference'],
              auom: ['append-only evidence'],
            },
            {
              id: 'commons-history',
              label: 'History',
              summary: 'Chronicles, archives, and longitudinal context.',
              semantics: ['Symbol'],
              mechanism: ['memory', 'reference'],
              auom: ['append-only evidence'],
            },
            {
              id: 'commons-technology',
              label: 'Technology',
              summary: 'Tools, standards, and applied design patterns.',
              semantics: ['Symbol'],
              mechanism: ['reference', 'transducer'],
              auom: ['append-only evidence'],
            },
            {
              id: 'commons-health',
              label: 'Health',
              summary: 'Biology, medicine, and care systems knowledge.',
              semantics: ['Symbol'],
              mechanism: ['memory', 'reference'],
              auom: ['append-only evidence'],
            },
            {
              id: 'commons-law',
              label: 'Law',
              summary: 'Rights, governance, and institutional constraints.',
              semantics: ['Symbol'],
              mechanism: ['reference', 'regulator'],
              auom: ['typed effects'],
            },
          ],
        },
        {
          id: 'symbol-identifiers',
          label: 'Identifiers',
          summary: 'IDs, hashes, claims, and provenance anchors.',
          semantics: ['Symbol'],
          mechanism: ['memory', 'reference'],
          auom: ['append-only evidence'],
        },
      ],
    },
    {
      id: 'types-observer',
      label: 'Observers',
      summary: 'Sensing, models, simulators, and auditors.',
      axes: {
        teleology: {
          purpose: 'Sense and verify across the organism.',
          horizon: 'near',
          fitness: 'signal fidelity + accountability',
        },
      },
      semantics: ['Observer'],
      mechanism: ['source', 'feedback'],
      auom: ['append-only evidence', 'omega checks'],
      children: [
        {
          id: 'observer-sensors',
          label: 'Sensors and probes',
          summary: 'Telemetry, instrumentation, and embodied sensing.',
          semantics: ['Observer'],
          mechanism: ['source', 'transducer'],
          auom: ['append-only evidence'],
        },
        {
          id: 'observer-models',
          label: 'Models and planners',
          summary: 'Predictive engines, policies, and controllers.',
          semantics: ['Observer'],
          mechanism: ['transducer', 'reference'],
          auom: ['typed effects'],
        },
        {
          id: 'observer-simulators',
          label: 'Simulators and emulators',
          summary: 'Synthetic observers for counterfactual testing.',
          semantics: ['Observer'],
          mechanism: ['transducer', 'feedback'],
          auom: ['omega checks'],
        },
        {
          id: 'observer-auditors',
          label: 'Auditors and verifiers',
          summary: 'Tests, monitors, and provenance witnesses.',
          axes: {
            agency: 'autonomous',
            teleology: {
              purpose: 'Verify truth claims and surface drift.',
              horizon: 'near',
              fitness: 'trust + compliance',
            },
          },
          evidence: ['append-only evidence', 'omega checks'],
          semantics: ['Observer'],
          mechanism: ['feedback', 'reference'],
          auom: ['append-only evidence'],
        },
      ],
    },
  ],
};

export const TYPES_SCHEMA = applyTypeDefaults(TYPES_SCHEMA_SEED);

const flattenWithParents = (
  node: TypesNode,
  depth = 0,
  parentId: string | null = null,
  acc: TypesLayoutNode[] = []
): TypesLayoutNode[] => {
  acc.push({
    ...node,
    depth,
    parentId,
    x: 0,
    y: 0,
    z: 0,
  });
  for (const child of node.children ?? []) {
    flattenWithParents(child, depth + 1, node.id, acc);
  }
  return acc;
};

const buildTypesLayout = (root: TypesNode): TypesLayout => {
  const nodes = flattenWithParents(root);
  const levels = new Map<number, TypesLayoutNode[]>();

  for (const node of nodes) {
    const group = levels.get(node.depth) ?? [];
    group.push(node);
    levels.set(node.depth, group);
  }

  for (const [depth, levelNodes] of levels.entries()) {
    const count = levelNodes.length;
    const radius = depth === 0 ? 0 : 160 + (depth - 1) * 120;
    const angleStep = count > 0 ? (Math.PI * 2) / count : 0;
    const offset = depth * 0.35;

    levelNodes.forEach((node, index) => {
      if (depth === 0) {
        node.x = 0;
        node.y = 0;
        node.z = 0;
        return;
      }
      const angle = index * angleStep + offset;
      node.x = Math.cos(angle) * radius;
      node.y = Math.sin(angle) * radius * 0.7;
      node.z = -depth * 60;
    });
  }

  const edges: TypesLayoutEdge[] = [];
  for (const node of nodes) {
    if (node.parentId) {
      edges.push({ from: node.parentId, to: node.id });
    }
  }

  const indexById: Record<string, TypesLayoutNode> = {};
  for (const node of nodes) {
    indexById[node.id] = node;
  }

  return {
    nodes,
    edges,
    indexById,
    roots: [root.id],
  };
};

export const TYPES_LAYOUT = buildTypesLayout(TYPES_SCHEMA);

export const getTypesPath = (targetId: string): string[] => {
  const path: string[] = [];
  let current = TYPES_LAYOUT.indexById[targetId];
  while (current) {
    path.push(current.id);
    if (!current.parentId) break;
    current = TYPES_LAYOUT.indexById[current.parentId];
  }
  return path;
};
