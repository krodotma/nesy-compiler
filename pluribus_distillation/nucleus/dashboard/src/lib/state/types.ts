/**
 * Unified Dashboard State Types
 *
 * Isomorphic TypeScript types shared across TUI, Web, Native, and WASM.
 * These types mirror the Python dataclasses in service_registry.py and strp_monitor.py.
 */

// ============================================================================
// Service Types (mirrors service_registry.py)
// ============================================================================

export interface ServiceDef {
  id: string;
  name: string;
  kind: 'port' | 'composition' | 'process';
  entry_point: string;
  description: string;
  port?: number;
  depends_on: string[];
  env: Record<string, string>;
  args: string[];
  tags: string[];
  auto_start: boolean;
  restart_policy: 'never' | 'on_failure' | 'always';
  health_check?: string;
  created_iso: string;
  provenance?: Record<string, unknown>;
  status?: 'stopped' | 'starting' | 'running' | 'error';
}

export interface ServiceInstance {
  service_id: string;
  instance_id: string;
  pid?: number;
  port?: number;
  status: 'stopped' | 'starting' | 'running' | 'error';
  started_iso: string;
  last_health_iso: string;
  health: 'unknown' | 'healthy' | 'unhealthy';
  error?: string;
}

// ============================================================================
// Agent Types (mirrors strp_monitor.py)
// ============================================================================

export interface AgentStatus {
  actor: string;
  status: string;
  health: string;
  queue_depth: number;
  current_task: string;
  blockers: string[];
  vor_cdi?: number;
  vor_passed?: number;
  vor_failed?: number;
  last_seen_iso: string;
}

export interface WorkerInfo {
  actor: string;
  last_seen: number;
  last_topic: string;
  pid?: number;
  provider: string;
  is_spawned: boolean;
}

// ============================================================================
// VPS Session Types (from agent_fallback_modes.md)
// ============================================================================

export type FlowMode = 'm' | 'A';  // monitor vs automatic

export interface ProviderStatus {
  available: boolean;
  lastCheck: string;
  error?: string;
  quotaRemaining?: number;
  model?: string;
  note?: string;
  prompt?: string;
  response?: string;
  verifiedAt?: string;
}

export interface PBPAIRRequest {
  id: string;
  provider: string;
  role: string;
  prompt: string;
  flowMode: FlowMode;
  created_iso: string;
  status: 'pending' | 'proposed' | 'approved' | 'completed' | 'rejected';
}

export interface PBPAIRProposal {
  request_id: string;
  proposal: string;
  created_iso: string;
}

export interface VPSSession {
  // Flow mode
  flowMode: FlowMode;

  // Provider status (web-session-only default; dynamic providers allowed).
  providers: Record<string, ProviderStatus>;

  // Fallback chain
  fallbackOrder: string[];
  activeFallback: string | null;

  // PBPAIR state
  pbpair: {
    activeRequests: PBPAIRRequest[];
    pendingProposals: PBPAIRProposal[];
  };

  // Auth state
  auth: {
    gcpProject?: string;
    gcpLocation?: string;
    claudeLoggedIn: boolean;
    geminiCliLoggedIn: boolean;
  };
}

// ============================================================================
// Workflow Types (from rd_workflow.py)
// ============================================================================

export interface WorkflowStatus {
  rd_aux_path: string;
  main_rhizome_path: string;
  drop_count: number;
  discourse_count: number;
  task_files: string[];
  pending_tasks: number;
  completed_tasks: number;
  eval_cards: number;
  ocr_files: number;
  last_sync_iso: string;
}

// ============================================================================
// Bus Event Types
// ============================================================================

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';
export type ImpactLevel = 'low' | 'medium' | 'high' | 'critical';
export type OmegaClass = 'prima' | 'meta' | 'omega';
export type EntelexisPhase = 'potential' | 'actualizing' | 'actualized' | 'decaying';
export type ReentryMode = 'observation' | 'modification' | 'self_reference' | 'closure';

// ==============================================================================
// Omega-Level Event Schema (Semantic Enrichment)
// ==============================================================================

/** Autopoietic reentry tracking - events that recursively modify the system */
export interface ReentryMarker {
  mode: ReentryMode;
  references_event_id?: string;
  closure_depth?: number;
  self_modification?: boolean;
}

/** Aristotelian entelechy - actualization of potential */
export interface EntelexisState {
  phase: EntelexisPhase;
  potential_id?: string;
  progress?: number;
  form?: string;
  matter?: string;
}

/** Path-dependent memory encoding */
export interface HysteresisTrace {
  path_hash?: string;
  decision_points?: number;
  reversibility?: number;
  entropy?: number;
  causal_depth?: number;
}

/** Multi-level semiotic meaning decomposition */
export interface SemioticalysisLayer {
  syntactic?: string;
  semantic?: string;
  pragmatic?: string;
  metalinguistic?: string;
  motif?: string;
  depth?: number;
}

/** Omega-level fundamental context - the "DNA" of agent evolution */
export interface OmegaContext {
  omega_class: OmegaClass;
  branch?: string;
  latent?: number[];
  speciation?: number;
  fitness?: number;
  automaton?: string;
}

/** Lineage context for VGT/HGT evolutionary tracking */
export interface LineageContext {
  dag_id?: string;
  lineage_id?: string;
  parent_lineage_id?: string;
  transfer_type?: 'VGT' | 'HGT' | 'none';
  generation?: number;
  mutation_op?: string;
}

/** Topology context for multi-agent coordination */
export interface TopologyContext {
  topology: 'single' | 'star' | 'peer_debate';
  fanout: number;
  coordinator?: string;
  participants?: string[];
  coordination_budget_tokens?: number;
}

/** Clade Meta-Productivity signal */
export interface CMPSignal {
  productivity_delta?: number;
  quality_score?: number;
  latency_ratio?: number;
  resource_efficiency?: number;
  lineage_health?: string;
}

// ==============================================================================
// Enhanced Bus Event (Omega-Level)
// ==============================================================================

export interface BusEvent {
  id?: string;
  topic: string;
  kind: string;
  level: LogLevel;
  actor: string;
  ts: number;
  iso: string;
  data: unknown;

  // Semantic enrichment (omega-level)
  semantic?: string;          // Human-readable summary
  reasoning?: string;         // Why this happened
  actionable?: string[];      // Suggested next actions
  impact?: ImpactLevel;       // Estimated impact level

  // Evolutionary context
  lineage?: LineageContext;
  cmp?: CMPSignal;
  topology?: TopologyContext;

  // Omega-level theoretical constructs
  reentry?: ReentryMarker;    // Autopoietic self-reference
  entelexis?: EntelexisState; // Potential-actuality tracking
  hysteresis?: HysteresisTrace; // Path-dependent memory
  semioticalysis?: SemioticalysisLayer; // Multi-level meaning
  omega?: OmegaContext;       // Fundamental evolutionary context

  // Correlation and causality
  trace_id?: string;
  parent_id?: string;
  causal_parents?: string[];
}

export interface STRpRequest {
  id: string;
  kind: string;
  actor: string;
  goal: string;
  status: 'pending' | 'working' | 'completed' | 'failed';
  created_iso: string;
  completed_iso?: string;
}

// ============================================================================
// Task Ledger Types (nucleus/tools/task_ledger.py)
// ============================================================================

export interface TaskLedgerEntry {
  id?: string;
  req_id: string;
  actor: string;
  topic: string;
  status: 'planned' | 'in_progress' | 'blocked' | 'completed' | 'abandoned';
  intent?: string;
  ts?: number;
  iso?: string;
  run_id?: string;
  meta?: Record<string, unknown>;
}

// ============================================================================
// UI Types
// ============================================================================

export type Theme = 'light' | 'dark' | 'system' | 'chroma';
export type ViewMode = 'services' | 'events' | 'requests' | 'agents' | 'vps' | 'workflow' | 'leads';

export interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  timestamp: number;
  read: boolean;
}

export interface UIState {
  theme: Theme;
  sidebarOpen: boolean;
  notifications: Notification[];
  activePanel: ViewMode;
}

// ============================================================================
// Complete Dashboard State
// ============================================================================

export interface DashboardState {
  // View mode
  mode: ViewMode;

  // Services
  services: ServiceDef[];
  instances: ServiceInstance[];
  selectedService: string | null;

  // Events
  events: BusEvent[];
  eventFilter: string | null;
  maxEvents: number;

  // Requests
  requests: STRpRequest[];

  // Agents
  agents: AgentStatus[];
  workers: WorkerInfo[];
  selectedAgent: string | null;

  // VPS Session
  session: VPSSession;

  // Workflow
  workflow: WorkflowStatus;

  // UI State
  ui: UIState;

  // Connection
  connected: boolean;
  lastSync: string;
}

// ============================================================================
// Action Types
// ============================================================================

export type DashboardAction =
  | { type: 'SET_MODE'; mode: ViewMode }
  | { type: 'SET_SERVICES'; services: ServiceDef[] }
  | { type: 'SET_INSTANCES'; instances: ServiceInstance[] }
  | { type: 'SELECT_SERVICE'; id: string | null }
  | { type: 'ADD_EVENT'; event: BusEvent }
  | { type: 'SET_EVENTS'; events: BusEvent[] }
  | { type: 'SET_EVENT_FILTER'; filter: string | null }
  | { type: 'SET_REQUESTS'; requests: STRpRequest[] }
  | { type: 'SET_AGENTS'; agents: AgentStatus[] }
  | { type: 'SET_WORKERS'; workers: WorkerInfo[] }
  | { type: 'SELECT_AGENT'; actor: string | null }
  | { type: 'SET_SESSION'; session: VPSSession }
  | { type: 'UPDATE_PROVIDER'; provider: string; status: ProviderStatus }
  | { type: 'SET_FLOW_MODE'; mode: FlowMode }
  | { type: 'SET_WORKFLOW'; workflow: WorkflowStatus }
  | { type: 'SET_THEME'; theme: Theme }
  | { type: 'TOGGLE_SIDEBAR' }
  | { type: 'ADD_NOTIFICATION'; notification: Notification }
  | { type: 'DISMISS_NOTIFICATION'; id: string }
  | { type: 'SET_CONNECTED'; connected: boolean }
  | { type: 'SYNC_STATE'; state: Partial<DashboardState> };

// ============================================================================
// Default State Factory
// ============================================================================

export function createDefaultState(): DashboardState {
  return {
    mode: 'services',
    services: [],
    instances: [],
    selectedService: null,
    events: [],
    eventFilter: null,
    maxEvents: 500,
    requests: [],
    agents: [],
    workers: [],
    selectedAgent: null,
    session: {
      flowMode: 'm',
      providers: {
        'chatgpt-web': { available: false, lastCheck: '' },
        'claude-web': { available: false, lastCheck: '' },
        'gemini-web': { available: false, lastCheck: '' },
      },
      // Web-session-only default.
      fallbackOrder: [
        'chatgpt-web',
        'claude-web',
        'gemini-web',
      ],
      activeFallback: null,
      pbpair: {
        activeRequests: [],
        pendingProposals: [],
      },
      auth: {
        claudeLoggedIn: false,
        geminiCliLoggedIn: false,
      },
    },
    workflow: {
      rd_aux_path: '',
      main_rhizome_path: '',
      drop_count: 0,
      discourse_count: 0,
      task_files: [],
      pending_tasks: 0,
      completed_tasks: 0,
      eval_cards: 0,
      ocr_files: 0,
      last_sync_iso: '',
    },
    ui: {
      theme: 'system',
      sidebarOpen: true,
      notifications: [],
      activePanel: 'services',
    },
    connected: false,
    lastSync: '',
  };
}
