/**
 * DKIN Observatory Types — Protocol v1-v19 Synthesis
 *
 * This module defines the data model for the DKIN Observatory,
 * unifying bus states, agent states, and code flow states across
 * all protocol versions.
 */

// ============================================================================
// Protocol Version Definitions (v1-v19)
// ============================================================================

export interface ProtocolVersion {
  version: string;
  name: string;
  features: string[];
  mandates: string[];
  busTopics: string[];
}

export const DKIN_PROTOCOL_VERSIONS: ProtocolVersion[] = [
  { version: 'v1', name: 'Minimal Check-in', features: ['bus snapshot'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v2', name: 'Drift Guards', features: ['BEAM/GOLDEN counts'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v3', name: 'Enhanced Dashboard', features: ['7 sections'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v4', name: 'ITERATE', features: ['coordination kick'], mandates: [], busTopics: ['infer_sync.request'] },
  { version: 'v5', name: 'Silent Monitoring', features: ['staleness indicators'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v6', name: 'Shared Ledger', features: ['compliance snapshot'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v7', name: 'Gap Analysis', features: ['aleatoric/epistemic detection'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v8', name: 'OITERATE', features: ['omega loop', 'Büchi automaton'], mandates: [], busTopics: ['oiterate.tick'] },
  { version: 'v9', name: 'DKIN Rename', features: ['dashboard kernel packaging'], mandates: [], busTopics: ['ckin.report'] },
  { version: 'v10', name: 'Membrane', features: ['PBFLUSH', 'MABSWARM/MBAD'], mandates: [], busTopics: ['operator.pbflush.request', 'mabswarm.*'] },
  { version: 'v11', name: 'MCP Interop', features: ['official SDK observability'], mandates: [], busTopics: ['mcp.official_sdk.*'] },
  { version: 'v12', name: 'PAIP', features: ['parallel agent filesystem isolation'], mandates: ['clone isolation'], busTopics: ['paip.clone.*'] },
  { version: 'v12.1', name: 'Summary Frame', features: ['TTL clarifications'], mandates: [], busTopics: ['paip.*'] },
  { version: 'v13', name: 'PBDEEP', features: ['forensics index'], mandates: [], busTopics: ['operator.pbdeep.*'] },
  { version: 'v15', name: 'Safe Context', features: ['log hygiene'], mandates: ['context guards'], busTopics: ['operator.pbhygiene.*'] },
  { version: 'v16', name: 'PBLOCK', features: ['milestone freeze'], mandates: ['feature lock'], busTopics: ['operator.pblock.*'] },
  { version: 'v17', name: 'System Hygiene', features: ['bus rotation', 'context window safety'], mandates: ['pre-PBLOCK audit'], busTopics: ['operator.pbhygiene.*'] },
  { version: 'v18', name: 'Resilience', features: ['Agentic State Graph', 'lossless handoff'], mandates: ['task lifecycle events'], busTopics: ['agent.*.task'] },
  { version: 'v19', name: 'Evolution', features: ['Percolation Loop', 'CMP metric', 'HGT'], mandates: ['evo-logging'], busTopics: ['evolution.*', 'hgt.*', 'cmp.*'] },
];

// ============================================================================
// Evolutionary Loop States (v19)
// ============================================================================

export type EvoLoopPhase = 'percolate' | 'assimilate' | 'mutate' | 'test' | 'promote' | 'idle';

export interface EvoUnit {
  id: string;
  genotype: string;  // SOTA ID or pattern
  phenotype: string; // Code path
  fitness: number;   // CMP score
  ancestor?: string;
  hypothesis?: string;
  timestamp: number;
}

export interface CMPMetrics {
  utility: number;      // 0-1
  robustness: number;   // 0-1 (test coverage)
  complexity: number;   // cyclomatic
  cost: number;         // tokens/compute
  score: number;        // CMP = (utility * robustness) / (complexity * cost)
}

// ============================================================================
// Agent State (v18)
// ============================================================================

export type TaskState = 'PENDING' | 'RUNNING' | 'PAUSED' | 'COMPLETED' | 'FAILED' | 'BLOCKED';

export interface AgentTask {
  taskId: string;
  parentId?: string;
  species: 'claude' | 'gemini' | 'codex' | 'qwen' | 'unknown';
  agent: string;
  state: TaskState;
  progress: number;  // 0-1
  context?: Record<string, unknown>;
  checkpoint?: Record<string, unknown>;
  startedAt: number;
  updatedAt: number;
}

export interface AgentState {
  agentId: string;
  species: string;
  activeTasks: AgentTask[];
  completedTasks: number;
  failedTasks: number;
  cmpScore: number;
  lastSeen: number;
}

// ============================================================================
// PBLOCK State (v16)
// ============================================================================

export interface PBLOCKState {
  active: boolean;
  milestone?: string;
  enteredAt?: number;
  enteredBy?: string;
  reason?: string;
  exitCriteria: {
    allTestsPass: boolean;
    pushedToRemotes: boolean;
    pushedToGithub: boolean;
  };
  violations: number;
}

// ============================================================================
// PAIP State (v12)
// ============================================================================

export interface PAIPClone {
  cloneDir: string;
  agentId: string;
  branch: string;
  createdAt: number;
  uncommitted: number;
  isOrphan: boolean;
  isStale: boolean;
}

// ============================================================================
// Bus Health (v17 Hygiene)
// ============================================================================

export interface BusHealth {
  sizeMb: number;
  eventCount: number;
  oldestEventAge: number;  // hours
  velocity: number;        // events/hour
  needsRotation: boolean;
  lastRotation?: number;
}

// ============================================================================
// Protocol Compliance
// ============================================================================

export interface ProtocolCompliance {
  version: string;
  compliant: boolean;
  violations: string[];
  recommendations: string[];
  lastChecked: number;
}

// ============================================================================
// DKIN Observatory State (Unified)
// ============================================================================

export interface DKINObservatoryState {
  // Current protocol level
  protocolVersion: string;

  // Evolutionary loop (v19)
  evoPhase: EvoLoopPhase;
  evoUnits: EvoUnit[];
  cmpMetrics: CMPMetrics;

  // Agent states (v18)
  agents: AgentState[];
  taskGraph: AgentTask[];

  // PBLOCK (v16)
  pblock: PBLOCKState;

  // PAIP (v12)
  paipClones: PAIPClone[];

  // Bus health (v17)
  busHealth: BusHealth;

  // Compliance
  compliance: ProtocolCompliance;

  // Activity
  recentEvents: DKINEvent[];
  lastUpdated: number;
}

// ============================================================================
// DKIN-specific Events
// ============================================================================

export interface DKINEvent {
  id?: string;
  ts: number;
  iso: string;
  topic: string;
  kind: string;
  actor: string;
  data: unknown;
  protocolVersion?: string;
}

// ============================================================================
// Visualization Helpers
// ============================================================================

export const EVO_PHASE_COLORS: Record<EvoLoopPhase, string> = {
  percolate: '#3b82f6',   // blue
  assimilate: '#8b5cf6',  // purple
  mutate: '#f59e0b',      // amber
  test: '#10b981',        // emerald
  promote: '#06b6d4',     // cyan
  idle: '#6b7280',        // gray
};

export const TASK_STATE_COLORS: Record<TaskState, string> = {
  PENDING: '#6b7280',
  RUNNING: '#3b82f6',
  PAUSED: '#f59e0b',
  COMPLETED: '#10b981',
  FAILED: '#ef4444',
  BLOCKED: '#dc2626',
};

export const SPECIES_COLORS: Record<string, string> = {
  claude: '#d97706',     // amber/orange
  gemini: '#3b82f6',     // blue
  codex: '#10b981',      // emerald/green
  qwen: '#8b5cf6',       // purple
  unknown: '#6b7280',    // gray
};

// ============================================================================
// Default State
// ============================================================================

export const DEFAULT_DKIN_STATE: DKINObservatoryState = {
  protocolVersion: 'v19',
  evoPhase: 'idle',
  evoUnits: [],
  cmpMetrics: { utility: 0, robustness: 0, complexity: 0, cost: 0, score: 0 },
  agents: [],
  taskGraph: [],
  pblock: {
    active: false,
    exitCriteria: { allTestsPass: false, pushedToRemotes: false, pushedToGithub: false },
    violations: 0,
  },
  paipClones: [],
  busHealth: { sizeMb: 0, eventCount: 0, oldestEventAge: 0, velocity: 0, needsRotation: false },
  compliance: { version: 'v19', compliant: true, violations: [], recommendations: [], lastChecked: 0 },
  recentEvents: [],
  lastUpdated: 0,
};
