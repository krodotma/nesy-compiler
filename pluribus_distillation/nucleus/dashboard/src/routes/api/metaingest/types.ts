/**
 * MetaIngest REST API TypeScript Interfaces
 * ==========================================
 *
 * Shared types for all MetaIngest API endpoints.
 * Import these into your route handlers.
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

// =============================================================================
// SHARED / COMMON
// =============================================================================

/**
 * Standard API response wrapper
 */
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: APIError;
  meta: {
    timestamp: string;
    duration_ms: number;
    trace_id?: string;
  };
}

/**
 * Standard error response
 */
export interface APIError {
  code: APIErrorCode;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * Error codes
 */
export type APIErrorCode =
  | 'TERM_NOT_FOUND'
  | 'PATTERN_NOT_FOUND'
  | 'MUTATION_NOT_FOUND'
  | 'CLI_ERROR'
  | 'CLI_TIMEOUT'
  | 'INVALID_QUERY'
  | 'FALKORDB_UNAVAILABLE'
  | 'VALIDATION_FAILED'
  | 'INTERNAL_ERROR';

/**
 * Pagination parameters
 */
export interface PaginationParams {
  limit?: number;
  offset?: number;
}

/**
 * Pagination metadata in response
 */
export interface PaginationMeta {
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

/**
 * CLI execution result (internal)
 */
export interface CLIResult {
  stdout: string;
  stderr: string;
  exitCode: number;
}


// =============================================================================
// GROUP 1: ONTOLOGY
// =============================================================================

/**
 * GET /api/metaingest/ontology - Request
 */
export interface ListOntologyRequest {
  status?: 'active' | 'superseded' | 'all';
  min_fitness?: number;
  limit?: number;
}

/**
 * Fitness metrics breakdown
 */
export interface FitnessMetrics {
  usage_frequency: number;
  semantic_coherence: number;
  context_breadth: number;
  evolution_stability: number;
  overall: number;
}

/**
 * Ontology term summary
 */
export interface OntologyTermSummary {
  term: string;
  fitness: number;
  status: 'active' | 'superseded';
}

/**
 * GET /api/metaingest/ontology - Response
 */
export interface ListOntologyResponse {
  terms: OntologyTermSummary[];
  total_terms: number;
  active_terms: number;
  superseded_terms: number;
}

/**
 * GET /api/metaingest/ontology/[term] - Request
 */
export interface GetOntologyTermRequest {
  term: string;
}

/**
 * GET /api/metaingest/ontology/[term] - Response
 */
export interface OntologyTermDetail {
  term: string;
  status: 'active' | 'superseded';
  created_at: string;
  updated_at: string;
  fitness: FitnessMetrics;
  usage_count: number;
  contexts: string[];
  avg_drift: number;
  lineage: string[];
  evolution_count: number;
  evolved_from?: string;
  superseded_by?: string;
  hgt_donor?: string;
  traits_from_donor?: string[];
}

/**
 * POST /api/metaingest/ontology/[term]/evolve - Request
 */
export interface EvolveTermRequest {
  term: string;
  context: string;
  force?: boolean;
}

/**
 * Evolution record
 */
export interface EvolutionRecord {
  evolution_id: string;
  evolution_type: EvolutionType;
  source_term: string;
  target_term: string;
  donor_term?: string;
  fitness_before: number;
  fitness_after: number;
  context: string;
  lineage_attestation: string;
  timestamp: string;
  metadata?: Record<string, unknown>;
}

export type EvolutionType =
  | 'mutation'
  | 'hgt'
  | 'fusion'
  | 'speciation'
  | 'extinction';

/**
 * POST /api/metaingest/ontology/[term]/evolve - Response
 */
export interface EvolveTermResponse {
  evolved: boolean;
  record?: EvolutionRecord;
  reason?: string;
}

/**
 * Lineage node in ancestry tree
 */
export interface LineageNode {
  term: string;
  evolution_type?: EvolutionType;
  fitness_at_evolution?: number;
  timestamp?: string;
}

/**
 * GET /api/metaingest/ontology/[term]/lineage - Response
 */
export interface GetLineageResponse {
  term: string;
  lineage: string[];
  lineage_details: LineageNode[];
  depth: number;
  root_term: string;
}


// =============================================================================
// GROUP 2: DRIFT
// =============================================================================

/**
 * Drift direction classification
 */
export type DriftDirection = 'specialization' | 'generalization' | 'shift' | 'none';

/**
 * GET /api/metaingest/drift - Request
 */
export interface ListDriftRequest {
  significant_only?: boolean;
  limit?: number;
  sort_by?: 'drift_magnitude' | 'updated_at';
}

/**
 * Drift summary for a term
 */
export interface DriftSummary {
  term: string;
  observation_count: number;
  created_at: string;
  updated_at: string;
  drift_magnitude: number;
  drift_direction: DriftDirection;
  is_drifting: boolean;
}

/**
 * GET /api/metaingest/drift - Response
 */
export interface ListDriftResponse {
  terms: DriftSummary[];
  summary: {
    total_terms: number;
    drifting_terms: number;
    stable_terms: number;
    avg_drift: number;
  };
}

/**
 * GET /api/metaingest/drift/[term] - Response
 */
export interface DriftReport {
  term: string;
  drift_magnitude: number;
  drift_direction: DriftDirection;
  historical_count: number;
  current_centroid: number[];
  historical_centroid: number[];
  confidence: number;
  is_significant: boolean;
  timestamp: string;
}

/**
 * Alert severity level
 */
export type AlertSeverity = 'low' | 'medium' | 'high';

/**
 * Drift alert
 */
export interface DriftAlert {
  term: string;
  drift_magnitude: number;
  drift_direction: DriftDirection;
  confidence: number;
  historical_count: number;
  detected_at: string;
  severity: AlertSeverity;
}

/**
 * GET /api/metaingest/drift/alerts - Request
 */
export interface GetDriftAlertsRequest {
  threshold?: number;
  limit?: number;
}

/**
 * GET /api/metaingest/drift/alerts - Response
 */
export interface GetDriftAlertsResponse {
  alerts: DriftAlert[];
  total_alerts: number;
  threshold: number;
}


// =============================================================================
// GROUP 3: KNOWLEDGE GRAPH
// =============================================================================

/**
 * GET /api/metaingest/knowledge/stats - Response
 */
export interface KnowledgeGraphStats {
  graph_name: string;
  falkordb_available: boolean;
  total_nodes: number | 'error';
  total_relationships: number | 'error';
  fallback_concepts: number;
  fallback_relationships: number;
  state_path: string;
}

/**
 * POST /api/metaingest/knowledge/query - Request
 */
export interface CypherQueryRequest {
  query: string;
  params?: Record<string, unknown>;
  read_only?: boolean;
}

/**
 * POST /api/metaingest/knowledge/query - Response
 */
export interface CypherQueryResponse {
  success: boolean;
  result_set: Array<Record<string, unknown>>;
  row_count: number;
  execution_time_ms: number;
  query_hash: string;
}

/**
 * Graph node
 */
export interface GraphNode {
  term: string;
  definition?: string;
  lineage_id?: string;
  concept_id?: string;
}

/**
 * Relationship types
 */
export type RelationshipType =
  | 'RELATED_TO'
  | 'EVOLVED_FROM'
  | 'DRIFTED_TO'
  | 'LEARNED'
  | 'PROPOSED';

/**
 * Graph relationship
 */
export interface GraphRelationship {
  source: string;
  target: string;
  type: RelationshipType;
  properties?: Record<string, unknown>;
}

/**
 * GET /api/metaingest/knowledge/neighbors/[node] - Request
 */
export interface GetNeighborsRequest {
  node: string;
  depth?: number;
  rel_type?: RelationshipType;
}

/**
 * Neighbor node with relationship info
 */
export interface NeighborNode {
  node: GraphNode;
  relationship: RelationshipType;
  distance: number;
}

/**
 * GET /api/metaingest/knowledge/neighbors/[node] - Response
 */
export interface GetNeighborsResponse {
  center_node: string;
  depth: number;
  neighbors: NeighborNode[];
  total_count: number;
}


// =============================================================================
// GROUP 4: SOTA
// =============================================================================

/**
 * Technique types
 */
export type TechniqueType =
  | 'algorithm'
  | 'architecture'
  | 'optimization'
  | 'technique';

/**
 * SOTA pattern
 */
export interface SOTAPattern {
  pattern_id: string;
  name: string;
  source: string;
  technique_type: TechniqueType;
  description: string;
  key_insights: string[];
  applicability: string[];
  confidence: number;
  timestamp: string;
}

/**
 * GET /api/metaingest/sota/patterns - Request
 */
export interface ListPatternsRequest {
  type?: TechniqueType;
  limit?: number;
}

/**
 * GET /api/metaingest/sota/patterns - Response
 */
export interface ListPatternsResponse {
  patterns: SOTAPattern[];
  total_count: number;
  by_type: Record<TechniqueType, number>;
}

/**
 * POST /api/metaingest/sota/ingest - Request
 */
export interface IngestPatternRequest {
  name: string;
  description: string;
  source: string;
}

/**
 * POST /api/metaingest/sota/ingest - Response
 */
export interface IngestPatternResponse {
  pattern: SOTAPattern;
  insights_extracted: number;
}

/**
 * Mutation types
 */
export type MutationType =
  | 'enhancement'
  | 'optimization'
  | 'refactor'
  | 'pattern'
  | 'algorithm'
  | 'architecture';

/**
 * Mutation status
 */
export type MutationStatus =
  | 'proposed'
  | 'validated'
  | 'applied'
  | 'rejected'
  | 'reverted';

/**
 * Mutation candidate
 */
export interface MutationCandidate {
  mutation_id: string;
  pattern_id: string;
  target_file: string;
  target_function?: string;
  mutation_type: MutationType;
  description: string;
  original_code: string;
  proposed_code: string;
  rationale: string;
  status: MutationStatus;
  confidence: number;
  lineage_attestation: string;
  validation_results?: ValidationResult;
  timestamp: string;
}

/**
 * POST /api/metaingest/sota/generate - Request
 */
export interface GenerateMutationRequest {
  pattern_id: string;
  target_file: string;
  target_function?: string;
}

/**
 * POST /api/metaingest/sota/generate - Response
 */
export interface GenerateMutationResponse {
  success: boolean;
  mutation?: MutationCandidate;
  error?: string;
}

/**
 * Validation result
 */
export interface ValidationResult {
  valid: boolean;
  syntax_ok: boolean;
  tests_pass: boolean | null;
  safety_check: boolean;
  notes: string[];
  timestamp: string;
}

/**
 * POST /api/metaingest/sota/validate - Request
 */
export interface ValidateMutationRequest {
  mutation_id: string;
}

/**
 * POST /api/metaingest/sota/validate - Response
 */
export interface ValidateMutationResponse {
  mutation_id: string;
  result: ValidationResult;
  new_status: MutationStatus;
}


// =============================================================================
// GROUP 5: PIPELINE
// =============================================================================

/**
 * Pipeline health status
 */
export interface PipelineHealth {
  healthy: boolean;
  gate_status: string;
  tracker_status: string;
  ingestor_status: string;
  falkordb_available: boolean;
  last_processed: string | null;
  total_processed: number;
  errors_last_hour: number;
}

/**
 * Overall health status
 */
export type HealthStatus = 'healthy' | 'degraded' | 'unhealthy';

/**
 * GET /api/metaingest/health - Response
 */
export interface HealthResponse {
  status: HealthStatus;
  health: PipelineHealth;
  components: {
    gate: boolean;
    tracker: boolean;
    ingestor: boolean;
    falkordb: boolean;
  };
}

/**
 * Input source types
 */
export type InputSource = 'manual' | 'voice' | 'api' | 'agent';

/**
 * POST /api/metaingest/process - Request
 */
export interface ProcessContentRequest {
  content: string;
  source: InputSource;
  metadata?: Record<string, unknown>;
}

/**
 * Pipeline processing result
 */
export interface PipelineResult {
  input_id: string;
  success: boolean;
  gate_state: string;
  terms_tracked: number;
  drift_detected: string[];
  concepts_ingested: number;
  processing_time_ms: number;
  errors: string[];
}

/**
 * POST /api/metaingest/process - Response
 */
export interface ProcessContentResponse {
  result: PipelineResult;
  summary: {
    processed: boolean;
    new_concepts: number;
    drifting_terms: number;
  };
}

/**
 * Stats time period
 */
export type StatsPeriod = 'hour' | 'day' | 'week' | 'all';

/**
 * GET /api/metaingest/stats - Request
 */
export interface GetStatsRequest {
  period?: StatsPeriod;
}

/**
 * Gate component stats
 */
export interface GateStats {
  current_state: string;
  total_gated: number;
}

/**
 * Tracker component stats
 */
export interface TrackerStats {
  total_terms: number;
  drifting_terms: number;
}

/**
 * Ingestor component stats
 */
export interface IngestorStats {
  total_concepts: number;
  total_relationships: number;
  falkordb_available: boolean;
}

/**
 * Error record
 */
export interface ErrorRecord {
  timestamp: string;
  errors: string[];
}

/**
 * GET /api/metaingest/stats - Response
 */
export interface PipelineStats {
  total_processed: number;
  last_processed: string | null;
  processed_last_hour: number;
  processed_last_day: number;
  processed_last_week: number;
  gate: GateStats;
  tracker: TrackerStats;
  ingestor: IngestorStats;
  errors_last_hour: number;
  recent_errors: ErrorRecord[];
}


// =============================================================================
// UTILITY TYPES
// =============================================================================

/**
 * Type guard for API response
 */
export function isAPIError<T>(response: APIResponse<T>): response is APIResponse<T> & { error: APIError } {
  return !response.success && response.error !== undefined;
}

/**
 * Type guard for successful response
 */
export function isAPISuccess<T>(response: APIResponse<T>): response is APIResponse<T> & { data: T } {
  return response.success && response.data !== undefined;
}

/**
 * Extract data type from API response
 */
export type ExtractData<T extends APIResponse<unknown>> = T extends APIResponse<infer D> ? D : never;
