/**
 * MetaIngest API - Index
 * ======================
 *
 * Main entry point for MetaIngest REST API.
 * This module provides a redirect to the API documentation.
 *
 * Endpoint Groups:
 * - /api/metaingest/ontology   - Ontology term management
 * - /api/metaingest/drift      - Semantic drift tracking
 * - /api/metaingest/knowledge  - Knowledge graph operations
 * - /api/metaingest/sota       - SOTA pattern management
 * - /api/metaingest/health     - Pipeline health & stats
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import type { RequestHandler } from '@builder.io/qwik-city';
import { buildResponse, getTraceId } from './utils/response';

/**
 * API index response
 */
interface APIIndex {
  name: string;
  version: string;
  protocol: string;
  endpoints: {
    group: string;
    base: string;
    description: string;
    routes: string[];
  }[];
  documentation: string;
}

export const onGet: RequestHandler = async ({ json, request }) => {
  const startTime = Date.now();
  const traceId = getTraceId(request.headers);

  const index: APIIndex = {
    name: 'MetaIngest API',
    version: '1.0.0',
    protocol: 'metaingest/v1',
    endpoints: [
      {
        group: 'Ontology',
        base: '/api/metaingest/ontology',
        description: 'Ontology term management with fitness-gated evolution',
        routes: [
          'GET /api/metaingest/ontology',
          'GET /api/metaingest/ontology/[term]',
          'POST /api/metaingest/ontology/[term]/evolve',
          'GET /api/metaingest/ontology/[term]/lineage',
        ],
      },
      {
        group: 'Drift',
        base: '/api/metaingest/drift',
        description: 'Semantic drift tracking and alerts',
        routes: [
          'GET /api/metaingest/drift',
          'GET /api/metaingest/drift/[term]',
          'GET /api/metaingest/drift/alerts',
        ],
      },
      {
        group: 'Knowledge Graph',
        base: '/api/metaingest/knowledge',
        description: 'FalkorDB knowledge graph operations',
        routes: [
          'GET /api/metaingest/knowledge/stats',
          'POST /api/metaingest/knowledge/query',
          'GET /api/metaingest/knowledge/neighbors/[node]',
        ],
      },
      {
        group: 'SOTA',
        base: '/api/metaingest/sota',
        description: 'SOTA pattern ingestion and mutation',
        routes: [
          'GET /api/metaingest/sota/patterns',
          'POST /api/metaingest/sota/ingest',
          'POST /api/metaingest/sota/generate',
          'POST /api/metaingest/sota/validate',
        ],
      },
      {
        group: 'Pipeline',
        base: '/api/metaingest',
        description: 'Pipeline operations and monitoring',
        routes: [
          'GET /api/metaingest/health',
          'POST /api/metaingest/process',
          'GET /api/metaingest/stats',
        ],
      },
    ],
    documentation: '/api/metaingest/API_SPECIFICATION.md',
  };

  json(200, buildResponse(index, startTime, traceId));
};

// Re-export types for convenience
export * from './types';
