/**
 * FalkorDB Schema Types for NeSy Compiler
 *
 * Step 3: Reverse engineer Kernel Schema from nucleus/tools/falkordb_schema.py
 * Mirrors the Python schema definitions in TypeScript for type safety.
 *
 * Schema Version: 1.0.0
 */

export const SCHEMA_VERSION = '1.0.0';

// ============================================================================
// Node Labels (from INDEXES in falkordb_schema.py)
// ============================================================================

export type NodeLabel =
  | 'entity'
  | 'fact'
  | 'agent'
  | 'task'
  | 'session'
  | 'lineage'
  | 'event'
  | 'motif_def'
  | 'motif_instance'
  // NeSy-specific labels
  | 'holon'
  | 'concept'
  | 'symbol';

// ============================================================================
// Entity Node (Generic knowledge graph entity)
// ============================================================================

export interface EntityNode {
  id: string;
  name: string;
  entity_type: string;
  // Extended properties
  description?: string;
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

// ============================================================================
// Fact Node (Relationship metadata with temporal validity)
// ============================================================================

export interface FactNode {
  id: string;
  valid_from?: string;  // ISO timestamp
  valid_to?: string;    // ISO timestamp (null = still valid)
  confidence?: number;  // 0-1 confidence score
  source?: string;      // Provenance
}

// ============================================================================
// Agent Node (Sagent/Swagent in the system)
// ============================================================================

export type AgentStatus = 'active' | 'idle' | 'terminated' | 'error';
export type AgentClass = 'sagent' | 'swagent' | 'external';

export interface AgentNode {
  id: string;
  status: AgentStatus;
  class: AgentClass;
  model?: string;       // e.g., 'claude-opus-4-5', 'gemini-3-pro'
  session_id?: string;
  started_at?: string;
  last_active?: string;
  capabilities?: string[];
}

// ============================================================================
// Task Node (Work items for agents)
// ============================================================================

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'failed' | 'blocked';

export interface TaskNode {
  id: string;
  status: TaskStatus;
  agent_id?: string;
  description?: string;
  priority?: number;
  created_at?: string;
  completed_at?: string;
  result?: string;
}

// ============================================================================
// Session Node (Agent sessions)
// ============================================================================

export interface SessionNode {
  id: string;
  actor: string;        // Agent/user ID
  start_ts: string;     // ISO timestamp
  end_ts?: string;
  events_count?: number;
  summary?: string;
}

// ============================================================================
// Lineage Node (Code provenance tracking)
// ============================================================================

export interface LineageNode {
  id: string;
  actor: string;        // Who created/modified
  cmp_score?: number;   // Compilation/quality score
  commit_hash?: string;
  file_path?: string;
  operation?: 'create' | 'modify' | 'delete' | 'merge';
  timestamp?: string;
}

// ============================================================================
// Event Node (Bus events)
// ============================================================================

export interface EventNode {
  id: string;
  topic: string;
  actor: string;
  ts: string;           // ISO timestamp
  trace_id?: string;
  payload?: Record<string, unknown>;
  event_type?: string;
}

// ============================================================================
// Motif Nodes (Pattern definitions and instances)
// ============================================================================

export type MotifCategory = 'antipattern' | 'idiom' | 'design_pattern' | 'refactoring';

export interface MotifDefNode {
  id: string;
  category: MotifCategory;
  name: string;
  description?: string;
  detection_rule?: string;  // Query or heuristic
  severity?: 'info' | 'warning' | 'error';
}

export interface MotifInstanceNode {
  id: string;
  motif_def_id: string;
  completed_ts?: string;
  file_path?: string;
  line_start?: number;
  line_end?: number;
  confidence?: number;
}

// ============================================================================
// NeSy-Specific Nodes
// ============================================================================

export type RingLevel = 0 | 1 | 2 | 3;

export interface HolonNode {
  id: string;
  name: string;
  path: string;
  ring: RingLevel;
  // Symbolic properties
  symbols?: string[];
  dependencies?: string[];
  // Neural properties (from LSA)
  lsa_vector?: number[];
  semantic_cluster?: string;
  stability_score?: number;
  // Metadata
  language?: string;
  loc?: number;  // Lines of code
  last_modified?: string;
}

export interface ConceptNode {
  id: string;
  name: string;
  // LSA latent dimension mapping
  dimension_index?: number;
  weight?: number;
  related_terms?: string[];
}

export interface SymbolNode {
  id: string;
  name: string;
  symbol_type: 'function' | 'class' | 'variable' | 'type' | 'module';
  defined_in: string;  // Holon ID
  signature?: string;
  visibility?: 'public' | 'private' | 'internal';
}

// ============================================================================
// Relationship Types
// ============================================================================

export type RelationshipType =
  // Code structure relationships
  | 'IMPORTS'
  | 'EXPORTS'
  | 'CALLS'
  | 'EXTENDS'
  | 'IMPLEMENTS'
  | 'DEPENDS_ON'
  | 'DEFINES'
  | 'CONTAINS'
  // Semantic relationships (from LSA)
  | 'SIMILAR_TO'
  | 'RELATED_TO'
  | 'CONCEPT_OF'
  // Temporal relationships
  | 'EVOLVED_FROM'
  | 'REFACTORED_TO'
  | 'MERGED_INTO'
  // Agent relationships
  | 'ASSIGNED_TO'
  | 'CREATED_BY'
  | 'MODIFIED_BY'
  | 'EMITTED'
  | 'TRIGGERED';

// ============================================================================
// Index Definitions (mirrors Python INDEXES)
// ============================================================================

export interface IndexDefinition {
  label: NodeLabel;
  property: string;
  type: 'default' | 'fulltext' | 'vector';
  description: string;
}

export const INDEXES: IndexDefinition[] = [
  // Entity indexes
  { label: 'entity', property: 'id', type: 'default', description: 'Primary entity lookup by ID' },
  { label: 'entity', property: 'name', type: 'default', description: 'Entity lookup by name' },
  { label: 'entity', property: 'entity_type', type: 'default', description: 'Filter entities by type' },

  // Fact indexes
  { label: 'fact', property: 'id', type: 'default', description: 'Primary fact lookup' },
  { label: 'fact', property: 'valid_from', type: 'default', description: 'Temporal queries - start' },
  { label: 'fact', property: 'valid_to', type: 'default', description: 'Temporal queries - end' },

  // Agent indexes
  { label: 'agent', property: 'id', type: 'default', description: 'Primary agent lookup' },
  { label: 'agent', property: 'status', type: 'default', description: 'Filter by status' },
  { label: 'agent', property: 'class', type: 'default', description: 'Filter by class' },

  // Task indexes
  { label: 'task', property: 'id', type: 'default', description: 'Primary task lookup' },
  { label: 'task', property: 'status', type: 'default', description: 'Filter by status' },
  { label: 'task', property: 'agent_id', type: 'default', description: 'Find by agent' },

  // Session indexes
  { label: 'session', property: 'id', type: 'default', description: 'Primary session lookup' },
  { label: 'session', property: 'actor', type: 'default', description: 'Find by actor' },
  { label: 'session', property: 'start_ts', type: 'default', description: 'Temporal queries' },

  // Lineage indexes
  { label: 'lineage', property: 'id', type: 'default', description: 'Primary lineage lookup' },
  { label: 'lineage', property: 'actor', type: 'default', description: 'Find by actor' },
  { label: 'lineage', property: 'cmp_score', type: 'default', description: 'Sort by CMP score' },

  // Event indexes
  { label: 'event', property: 'id', type: 'default', description: 'Primary event lookup' },
  { label: 'event', property: 'topic', type: 'default', description: 'Filter by topic' },
  { label: 'event', property: 'actor', type: 'default', description: 'Find by actor' },
  { label: 'event', property: 'ts', type: 'default', description: 'Temporal queries' },
  { label: 'event', property: 'trace_id', type: 'default', description: 'Trace reconstruction' },

  // Motif indexes
  { label: 'motif_def', property: 'id', type: 'default', description: 'Primary motif lookup' },
  { label: 'motif_def', property: 'category', type: 'default', description: 'Filter by category' },
  { label: 'motif_instance', property: 'id', type: 'default', description: 'Primary instance lookup' },
  { label: 'motif_instance', property: 'completed_ts', type: 'default', description: 'Temporal queries' },

  // NeSy-specific indexes
  { label: 'holon', property: 'id', type: 'default', description: 'Primary holon lookup' },
  { label: 'holon', property: 'path', type: 'default', description: 'Find by file path' },
  { label: 'holon', property: 'ring', type: 'default', description: 'Filter by ring level' },
  { label: 'holon', property: 'lsa_vector', type: 'vector', description: 'Semantic similarity search' },
  { label: 'concept', property: 'id', type: 'default', description: 'Primary concept lookup' },
  { label: 'symbol', property: 'id', type: 'default', description: 'Primary symbol lookup' },
  { label: 'symbol', property: 'name', type: 'default', description: 'Symbol name lookup' },
];

// ============================================================================
// Type Guards
// ============================================================================

export function isHolonNode(node: unknown): node is HolonNode {
  return (
    typeof node === 'object' &&
    node !== null &&
    'id' in node &&
    'path' in node &&
    'ring' in node
  );
}

export function isAgentNode(node: unknown): node is AgentNode {
  return (
    typeof node === 'object' &&
    node !== null &&
    'id' in node &&
    'status' in node &&
    'class' in node
  );
}

export function isEventNode(node: unknown): node is EventNode {
  return (
    typeof node === 'object' &&
    node !== null &&
    'id' in node &&
    'topic' in node &&
    'ts' in node
  );
}
