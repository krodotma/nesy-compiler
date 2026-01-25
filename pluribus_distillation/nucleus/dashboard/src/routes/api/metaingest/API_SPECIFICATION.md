# MetaIngest REST API Specification

**Version**: 1.0.0
**Framework**: Qwik City API Routes
**Pattern**: Python CLI wrapping via child_process.spawn

## Architecture

```
Qwik City API Route -> spawn(python3, [...args]) -> Python CLI Tool
       |                       |                          |
  HTTP Request          JSON stdout              State/FalkorDB
       |                       |                          |
  HTTP Response <-------- Parse JSON <--------------- Results
```

---

## Shared Interfaces

```typescript
// /src/routes/api/metaingest/types.ts

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
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

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
```

---

## GROUP 1: Ontology

### GET /api/metaingest/ontology
List all ontology terms with fitness scores.

**Request Interface**:
```typescript
export interface ListOntologyRequest {
  status?: 'active' | 'superseded' | 'all';
  min_fitness?: number;  // 0.0 - 1.0
  limit?: number;        // default: 50
}
```

**Response Interface**:
```typescript
export interface OntologyTerm {
  term: string;
  status: 'active' | 'superseded';
  fitness: number;
  usage_count: number;
  created_at: string;
  updated_at: string;
  lineage: string[];
  evolved_from?: string;
  superseded_by?: string;
}

export interface ListOntologyResponse {
  terms: Array<{
    term: string;
    fitness: number;
    status: string;
  }>;
  total_terms: number;
  active_terms: number;
  superseded_terms: number;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/ontology_evolver.py terms --limit 50
python3 /pluribus/nucleus/tools/ontology_evolver.py status --json
```

---

### GET /api/metaingest/ontology/[term]
Get detailed information about a specific term.

**Request Interface**:
```typescript
export interface GetOntologyTermRequest {
  term: string;  // URL parameter
}
```

**Response Interface**:
```typescript
export interface OntologyTermDetail {
  term: string;
  status: 'active' | 'superseded';
  created_at: string;
  updated_at: string;

  // Fitness metrics
  fitness: {
    usage_frequency: number;
    semantic_coherence: number;
    context_breadth: number;
    evolution_stability: number;
    overall: number;
  };

  // Usage data
  usage_count: number;
  contexts: string[];
  avg_drift: number;

  // Lineage
  lineage: string[];
  evolution_count: number;
  evolved_from?: string;
  superseded_by?: string;

  // HGT data (if applicable)
  hgt_donor?: string;
  traits_from_donor?: string[];
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/ontology_evolver.py fitness "term_name" --json
```

---

### POST /api/metaingest/ontology/[term]/evolve
Trigger evolution for a term.

**Request Interface**:
```typescript
export interface EvolveTermRequest {
  term: string;     // URL parameter
  context: string;  // Usage context for evolution
  force?: boolean;  // Force evolution regardless of fitness
}
```

**Response Interface**:
```typescript
export interface EvolutionRecord {
  evolution_id: string;
  evolution_type: 'mutation' | 'hgt' | 'fusion' | 'speciation' | 'extinction';
  source_term: string;
  target_term: string;
  donor_term?: string;
  fitness_before: number;
  fitness_after: number;
  context: string;
  lineage_attestation: string;
  timestamp: string;
}

export interface EvolveTermResponse {
  evolved: boolean;
  record?: EvolutionRecord;
  reason?: string;  // If not evolved, why
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/ontology_evolver.py evolve "term_name" --context "usage context" --json
python3 /pluribus/nucleus/tools/ontology_evolver.py evolve "term_name" --context "usage context" --force --json
```

---

### GET /api/metaingest/ontology/[term]/lineage
Get the full ancestry/lineage of a term.

**Request Interface**:
```typescript
export interface GetLineageRequest {
  term: string;  // URL parameter
}
```

**Response Interface**:
```typescript
export interface LineageNode {
  term: string;
  evolution_type?: string;
  fitness_at_evolution?: number;
  timestamp?: string;
}

export interface GetLineageResponse {
  term: string;
  lineage: string[];
  lineage_details: LineageNode[];
  depth: number;
  root_term: string;
}
```

**Python CLI Command**:
```bash
# Custom extraction from state file or:
python3 /pluribus/nucleus/tools/ontology_evolver.py fitness "term_name" --json
# Parse lineage field from output
```

---

## GROUP 2: Drift

### GET /api/metaingest/drift
List all semantic drift events.

**Request Interface**:
```typescript
export interface ListDriftRequest {
  significant_only?: boolean;  // Only drifting terms
  limit?: number;              // default: 50
  sort_by?: 'drift_magnitude' | 'updated_at';
}
```

**Response Interface**:
```typescript
export interface DriftSummary {
  term: string;
  observation_count: number;
  created_at: string;
  updated_at: string;
  drift_magnitude: number;
  drift_direction: 'specialization' | 'generalization' | 'shift' | 'none';
  is_drifting: boolean;
}

export interface ListDriftResponse {
  terms: DriftSummary[];
  summary: {
    total_terms: number;
    drifting_terms: number;
    stable_terms: number;
    avg_drift: number;
  };
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/semantic_drift_tracker.py list --json
python3 /pluribus/nucleus/tools/semantic_drift_tracker.py report --format json
```

---

### GET /api/metaingest/drift/[term]
Get drift history for a specific term.

**Request Interface**:
```typescript
export interface GetDriftRequest {
  term: string;  // URL parameter
}
```

**Response Interface**:
```typescript
export interface DriftReport {
  term: string;
  drift_magnitude: number;
  drift_direction: 'specialization' | 'generalization' | 'shift';
  historical_count: number;
  current_centroid: number[];
  historical_centroid: number[];
  confidence: number;
  is_significant: boolean;
  timestamp: string;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/semantic_drift_tracker.py check "term_name" --json
```

---

### GET /api/metaingest/drift/alerts
Get active drift alerts (significant drift detected).

**Request Interface**:
```typescript
export interface GetDriftAlertsRequest {
  threshold?: number;  // Minimum drift magnitude (default: 0.15)
  limit?: number;
}
```

**Response Interface**:
```typescript
export interface DriftAlert {
  term: string;
  drift_magnitude: number;
  drift_direction: string;
  confidence: number;
  historical_count: number;
  detected_at: string;
  severity: 'low' | 'medium' | 'high';
}

export interface GetDriftAlertsResponse {
  alerts: DriftAlert[];
  total_alerts: number;
  threshold: number;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/semantic_drift_tracker.py list --json
# Filter is_drifting: true from output
```

---

## GROUP 3: Knowledge Graph

### GET /api/metaingest/knowledge/stats
Get knowledge graph statistics.

**Request Interface**:
```typescript
// No parameters
```

**Response Interface**:
```typescript
export interface KnowledgeGraphStats {
  graph_name: string;
  falkordb_available: boolean;

  // Node counts
  total_nodes: number;
  concept_count: number;
  term_count: number;

  // Relationship counts
  total_relationships: number;
  related_to_count: number;
  evolved_from_count: number;
  drifted_to_count: number;

  // Fallback store
  fallback_concepts: number;
  fallback_relationships: number;

  state_path: string;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/knowledge_graph_ingestor.py status
```

---

### POST /api/metaingest/knowledge/query
Execute a Cypher query on the knowledge graph.

**Request Interface**:
```typescript
export interface CypherQueryRequest {
  query: string;      // Cypher query
  params?: Record<string, unknown>;  // Query parameters
  read_only?: boolean;  // Safety flag (default: true)
}
```

**Response Interface**:
```typescript
export interface CypherQueryResponse {
  success: boolean;
  result_set: Array<Record<string, unknown>>;
  row_count: number;
  execution_time_ms: number;
  query_hash: string;  // For caching
}
```

**Python CLI Command**:
```bash
# Direct FalkorDB query via Python
python3 -c "
from falkordb import FalkorDB
import json
client = FalkorDB()
graph = client.select_graph('pluribus_kg')
result = graph.query('$QUERY', {'param': '$VALUE'})
print(json.dumps([dict(zip(result.header, row)) for row in result.result_set]))
"
```

---

### GET /api/metaingest/knowledge/neighbors/[node]
Get neighboring nodes in the knowledge graph.

**Request Interface**:
```typescript
export interface GetNeighborsRequest {
  node: string;   // URL parameter - node term/ID
  depth?: number; // Traversal depth 1-3 (default: 1)
  rel_type?: string;  // Filter by relationship type
}
```

**Response Interface**:
```typescript
export interface GraphNode {
  term: string;
  definition?: string;
  lineage_id?: string;
  concept_id?: string;
}

export interface GraphRelationship {
  source: string;
  target: string;
  type: 'RELATED_TO' | 'EVOLVED_FROM' | 'DRIFTED_TO' | 'LEARNED';
  properties?: Record<string, unknown>;
}

export interface GetNeighborsResponse {
  center_node: string;
  depth: number;
  neighbors: Array<{
    node: GraphNode;
    relationship: string;
    distance: number;
  }>;
  total_count: number;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/knowledge_graph_ingestor.py query "node_term" --depth 2 --json
```

---

## GROUP 4: SOTA

### GET /api/metaingest/sota/patterns
List all ingested SOTA patterns.

**Request Interface**:
```typescript
export interface ListPatternsRequest {
  type?: 'algorithm' | 'architecture' | 'optimization' | 'technique';
  limit?: number;
}
```

**Response Interface**:
```typescript
export interface SOTAPattern {
  pattern_id: string;
  name: string;
  source: string;
  technique_type: string;
  description: string;
  key_insights: string[];
  applicability: string[];
  confidence: number;
  timestamp: string;
}

export interface ListPatternsResponse {
  patterns: SOTAPattern[];
  total_count: number;
  by_type: Record<string, number>;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/sota_mutation_engine.py patterns --json
python3 /pluribus/nucleus/tools/sota_mutation_engine.py patterns --type algorithm --json
```

---

### POST /api/metaingest/sota/ingest
Ingest a new SOTA pattern.

**Request Interface**:
```typescript
export interface IngestPatternRequest {
  name: string;
  description: string;
  source: string;  // Paper, repo, doc reference
}
```

**Response Interface**:
```typescript
export interface IngestPatternResponse {
  pattern: SOTAPattern;
  insights_extracted: number;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/sota_mutation_engine.py ingest "pattern_name" \
  --description "description text" \
  --source "source reference" \
  --json
```

---

### POST /api/metaingest/sota/generate
Generate a mutation from a pattern.

**Request Interface**:
```typescript
export interface GenerateMutationRequest {
  pattern_id: string;
  target_file: string;
  target_function?: string;  // Optional: specific function
}
```

**Response Interface**:
```typescript
export interface MutationCandidate {
  mutation_id: string;
  pattern_id: string;
  target_file: string;
  target_function?: string;
  mutation_type: 'enhancement' | 'optimization' | 'refactor' | 'pattern' | 'algorithm' | 'architecture';
  description: string;
  original_code: string;
  proposed_code: string;
  rationale: string;
  status: 'proposed' | 'validated' | 'applied' | 'rejected' | 'reverted';
  confidence: number;
  lineage_attestation: string;
  timestamp: string;
}

export interface GenerateMutationResponse {
  success: boolean;
  mutation?: MutationCandidate;
  error?: string;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/sota_mutation_engine.py generate \
  --pattern "pattern_id" \
  --target "/path/to/file.py" \
  --function "function_name" \
  --json
```

---

### POST /api/metaingest/sota/validate
Validate a mutation candidate.

**Request Interface**:
```typescript
export interface ValidateMutationRequest {
  mutation_id: string;
}
```

**Response Interface**:
```typescript
export interface ValidationResult {
  valid: boolean;
  syntax_ok: boolean;
  tests_pass: boolean | null;
  safety_check: boolean;
  notes: string[];
  timestamp: string;
}

export interface ValidateMutationResponse {
  mutation_id: string;
  result: ValidationResult;
  new_status: string;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/sota_mutation_engine.py validate "mutation_id" --json
```

---

## GROUP 5: Pipeline

### GET /api/metaingest/health
Get pipeline health status.

**Request Interface**:
```typescript
// No parameters
```

**Response Interface**:
```typescript
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

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  health: PipelineHealth;
  components: {
    gate: boolean;
    tracker: boolean;
    ingestor: boolean;
    falkordb: boolean;
  };
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/metaingest_pipeline.py health --json
```

---

### POST /api/metaingest/process
Process content through the full MetaIngest pipeline.

**Request Interface**:
```typescript
export interface ProcessContentRequest {
  content: string;
  source: 'manual' | 'voice' | 'api' | 'agent';
  metadata?: Record<string, unknown>;
}
```

**Response Interface**:
```typescript
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

export interface ProcessContentResponse {
  result: PipelineResult;
  summary: {
    processed: boolean;
    new_concepts: number;
    drifting_terms: number;
  };
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/metaingest_pipeline.py process "content text" \
  --source api \
  --json
```

---

### GET /api/metaingest/stats
Get pipeline processing statistics.

**Request Interface**:
```typescript
export interface GetStatsRequest {
  period?: 'hour' | 'day' | 'week' | 'all';
}
```

**Response Interface**:
```typescript
export interface PipelineStats {
  total_processed: number;
  last_processed: string | null;

  // Time-based stats
  processed_last_hour: number;
  processed_last_day: number;
  processed_last_week: number;

  // Component stats
  gate: {
    current_state: string;
    total_gated: number;
  };
  tracker: {
    total_terms: number;
    drifting_terms: number;
  };
  ingestor: {
    total_concepts: number;
    total_relationships: number;
    falkordb_available: boolean;
  };

  // Error stats
  errors_last_hour: number;
  recent_errors: Array<{
    timestamp: string;
    errors: string[];
  }>;
}
```

**Python CLI Command**:
```bash
python3 /pluribus/nucleus/tools/metaingest_pipeline.py status
# Combine with:
python3 /pluribus/nucleus/tools/semantic_drift_tracker.py status
python3 /pluribus/nucleus/tools/knowledge_graph_ingestor.py status
```

---

## Implementation Utilities

### CLI Wrapper Function

```typescript
// /src/routes/api/metaingest/utils/cli.ts

import { spawn } from 'child_process';

export interface CLIOptions {
  timeout?: number;  // ms, default 30000
  cwd?: string;
}

export async function runPythonCLI(
  script: string,
  args: string[],
  options: CLIOptions = {}
): Promise<CLIResult> {
  const { timeout = 30000, cwd = '/pluribus' } = options;

  return new Promise((resolve, reject) => {
    const proc = spawn('python3', [script, ...args], { cwd });
    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      proc.kill('SIGTERM');
      reject(new Error(`CLI timeout after ${timeout}ms`));
    }, timeout);

    proc.stdout.on('data', (data) => { stdout += data.toString(); });
    proc.stderr.on('data', (data) => { stderr += data.toString(); });

    proc.on('close', (code) => {
      clearTimeout(timer);
      resolve({ stdout, stderr, exitCode: code ?? 0 });
    });

    proc.on('error', (err) => {
      clearTimeout(timer);
      reject(err);
    });
  });
}

export function parseJSONOutput<T>(result: CLIResult): T {
  if (result.exitCode !== 0) {
    throw new Error(`CLI error (code ${result.exitCode}): ${result.stderr}`);
  }
  return JSON.parse(result.stdout) as T;
}
```

### Response Builder

```typescript
// /src/routes/api/metaingest/utils/response.ts

export function buildResponse<T>(
  data: T,
  startTime: number,
  traceId?: string
): APIResponse<T> {
  return {
    success: true,
    data,
    meta: {
      timestamp: new Date().toISOString(),
      duration_ms: Date.now() - startTime,
      trace_id: traceId,
    },
  };
}

export function buildErrorResponse(
  code: string,
  message: string,
  details?: Record<string, unknown>
): APIResponse<never> {
  return {
    success: false,
    error: { code, message, details },
    meta: {
      timestamp: new Date().toISOString(),
      duration_ms: 0,
    },
  };
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `TERM_NOT_FOUND` | Ontology term does not exist |
| `PATTERN_NOT_FOUND` | SOTA pattern does not exist |
| `MUTATION_NOT_FOUND` | Mutation candidate does not exist |
| `CLI_ERROR` | Python CLI execution failed |
| `CLI_TIMEOUT` | Python CLI exceeded timeout |
| `INVALID_QUERY` | Cypher query validation failed |
| `FALKORDB_UNAVAILABLE` | FalkorDB connection failed |
| `VALIDATION_FAILED` | Request validation failed |
| `INTERNAL_ERROR` | Unexpected server error |

---

## Example Route Implementation

```typescript
// /src/routes/api/metaingest/ontology/index.ts

import type { RequestHandler } from '@builder.io/qwik-city';
import { runPythonCLI, parseJSONOutput } from '../utils/cli';
import { buildResponse, buildErrorResponse } from '../utils/response';
import type { ListOntologyResponse } from '../types';

export const onGet: RequestHandler = async ({ query, json }) => {
  const startTime = Date.now();

  try {
    const limit = parseInt(query.get('limit') || '50');

    // Get terms list
    const termsResult = await runPythonCLI(
      '/pluribus/nucleus/tools/ontology_evolver.py',
      ['terms', '--limit', String(limit)]
    );

    // Get status
    const statusResult = await runPythonCLI(
      '/pluribus/nucleus/tools/ontology_evolver.py',
      ['status', '--json']
    );

    const status = parseJSONOutput(statusResult);

    // Parse terms from text output
    const terms = parseTermsOutput(termsResult.stdout);

    const data: ListOntologyResponse = {
      terms,
      total_terms: status.total_terms,
      active_terms: status.active_terms,
      superseded_terms: status.superseded_terms,
    };

    json(200, buildResponse(data, startTime));

  } catch (error) {
    json(500, buildErrorResponse(
      'CLI_ERROR',
      error instanceof Error ? error.message : 'Unknown error'
    ));
  }
};
```
