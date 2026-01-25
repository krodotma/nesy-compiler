import { createContextId } from '@builder.io/qwik';
import type {
  OntologyTerm,
  OntologyTermDetail,
  DriftReport,
  DriftAlert,
  KnowledgeGraphStats,
  SOTAPattern,
  PipelineHealth,
} from './types';

export interface MetaIngestState {
  // Loading states
  loading: {
    ontology: boolean;
    drift: boolean;
    knowledge: boolean;
    sota: boolean;
    pipeline: boolean;
  };

  // Data
  ontologyTerms: OntologyTerm[];
  selectedTerm: OntologyTermDetail | null;
  driftReports: DriftReport[];
  driftAlerts: DriftAlert[];
  knowledgeStats: KnowledgeGraphStats | null;
  sotaPatterns: SOTAPattern[];
  pipelineHealth: PipelineHealth | null;

  // UI state
  activeTab: 'ontology' | 'drift' | 'knowledge' | 'sota' | 'pipeline';
  searchQuery: string;
  filterStatus: 'all' | 'active' | 'superseded' | 'drifting';

  // Errors
  errors: {
    ontology: string | null;
    drift: string | null;
    knowledge: string | null;
    sota: string | null;
    pipeline: string | null;
  };

  // Timestamps
  lastFetch: {
    ontology: number;
    drift: number;
    knowledge: number;
    sota: number;
    pipeline: number;
  };
}

export const MetaIngestContext = createContextId<MetaIngestState>('metaingest');

export function createMetaIngestStore(): MetaIngestState {
  return {
    loading: {
      ontology: false,
      drift: false,
      knowledge: false,
      sota: false,
      pipeline: false,
    },
    ontologyTerms: [],
    selectedTerm: null,
    driftReports: [],
    driftAlerts: [],
    knowledgeStats: null,
    sotaPatterns: [],
    pipelineHealth: null,
    activeTab: 'ontology',
    searchQuery: '',
    filterStatus: 'all',
    errors: {
      ontology: null,
      drift: null,
      knowledge: null,
      sota: null,
      pipeline: null,
    },
    lastFetch: {
      ontology: 0,
      drift: 0,
      knowledge: 0,
      sota: 0,
      pipeline: 0,
    },
  };
}
