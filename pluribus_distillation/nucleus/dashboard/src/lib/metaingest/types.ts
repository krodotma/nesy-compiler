// ============================================================
// API Response Wrapper
// ============================================================
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  meta?: {
    timestamp: string;
    traceId: string;
    cached?: boolean;
    total?: number;
  };
}

// ============================================================
// Ontology Types
// ============================================================
export interface OntologyTerm {
  term: string;
  fitness: number;
  status: 'active' | 'superseded' | 'extinct';
  usageCount: number;
  createdAt: string;
  updatedAt: string;
}

export interface OntologyTermDetail extends OntologyTerm {
  definition: string[];
  lineage: string[];
  sourceAgent: string;
  vectorHash: string;
  evolutionCount: number;
  avgDrift: number;
  contexts: string[];
  fitnessMetrics: {
    usageFrequency: number;
    semanticCoherence: number;
    contextBreadth: number;
    evolutionStability: number;
    overall: number;
  };
}

export interface EvolutionRecord {
  evolutionId: string;
  evolutionType: 'mutation' | 'hgt' | 'fusion' | 'speciation' | 'extinction';
  sourceTerm: string;
  targetTerm: string;
  donorTerm: string | null;
  fitnessBefore: number;
  fitnessAfter: number;
  context: string;
  lineageAttestation: string;
  timestamp: string;
}

// ============================================================
// Drift Types
// ============================================================
export interface DriftReport {
  term: string;
  observationCount: number;
  createdAt: string;
  updatedAt: string;
  driftMagnitude: number;
  driftDirection: 'specialization' | 'generalization' | 'shift' | 'stable';
  isDrifting: boolean;
  confidence: number;
}

export interface DriftAlert {
  term: string;
  magnitude: number;
  direction: string;
  detectedAt: string;
  threshold: number;
  severity: 'warning' | 'critical';
}

// ============================================================
// Knowledge Graph Types
// ============================================================
export interface KnowledgeGraphStats {
  nodeCount: number;
  edgeCount: number;
  conceptCount: number;
  relationshipTypes: Record<string, number>;
  falkordbConnected: boolean;
  lastQueryMs: number;
}

export interface GraphNode {
  id: string;
  term: string;
  definition: string;
  lineageId: string;
  nodeType: 'concept' | 'entity' | 'relationship';
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
}

export interface GraphRelationship {
  source: string;
  target: string;
  type: string;
  weight?: number;
}

// ============================================================
// SOTA Types
// ============================================================
export interface SOTAPattern {
  patternId: string;
  name: string;
  source: string;
  techniqueType: 'algorithm' | 'architecture' | 'optimization' | 'technique';
  description: string;
  keyInsights: string[];
  applicability: string[];
  confidence: number;
  timestamp: string;
}

export interface MutationCandidate {
  mutationId: string;
  patternId: string;
  targetFile: string;
  targetFunction: string | null;
  mutationType: 'enhancement' | 'optimization' | 'refactor' | 'pattern';
  description: string;
  originalCode: string;
  proposedCode: string;
  rationale: string;
  status: 'proposed' | 'validated' | 'applied' | 'rejected' | 'reverted';
  confidence: number;
  timestamp: string;
}

export interface ValidationResult {
  verdictId: string;
  mutationId: string | null;
  overallPass: boolean;
  overallScore: number;
  gates: {
    gateName: string;
    passed: boolean;
    score: number;
    details: string;
  }[];
  recommendation: string;
  timestamp: string;
}

// ============================================================
// Pipeline Types
// ============================================================
export interface PipelineHealth {
  healthy: boolean;
  gateStatus: string;
  trackerStatus: string;
  ingestorStatus: string;
  falkordbAvailable: boolean;
  lastProcessed: string | null;
  totalProcessed: number;
  errorsLastHour: number;
}

export interface PipelineResult {
  inputId: string;
  success: boolean;
  gateState: string;
  termsTracked: number;
  driftDetected: string[];
  conceptsIngested: number;
  processingTimeMs: number;
  errors: string[];
}
