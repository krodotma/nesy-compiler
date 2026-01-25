/**
 * GET /api/metaingest/ontology/[term]
 * ====================================
 *
 * Get detailed information about a specific ontology term.
 *
 * Path Parameters:
 * - term: string - The term to look up
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import type { RequestHandler } from '@builder.io/qwik-city';
import {
  runOntologyEvolver,
  parseJSONOutput,
} from '../../utils/cli';
import {
  buildResponse,
  buildErrorResponse,
  errorToResponse,
  requireString,
  getTraceId,
  isValidationError,
  validationErrorToResponse,
} from '../../utils/response';
import type {
  OntologyTermDetail,
  FitnessMetrics,
} from '../../types';

export const onGet: RequestHandler = async ({ params, json, request }) => {
  const startTime = Date.now();
  const traceId = getTraceId(request.headers);

  try {
    // Validate term parameter
    const term = requireString(params.term, 'term');

    // Get fitness metrics
    const fitnessResult = await runOntologyEvolver('fitness', [term, '--json']);
    const fitness = parseJSONOutput<FitnessMetrics>(fitnessResult);

    // To get full term details, we need to read state directly or use multiple commands
    // For now, construct response from fitness data
    const response: OntologyTermDetail = {
      term,
      status: 'active', // Fitness command only works on existing terms
      created_at: new Date().toISOString(), // Not available from fitness command
      updated_at: new Date().toISOString(),
      fitness,
      usage_count: Math.round(fitness.usage_frequency * 100), // Reverse calculation
      contexts: [],
      avg_drift: 1 - fitness.semantic_coherence, // Reverse calculation (approximation)
      lineage: [term],
      evolution_count: Math.round((1 - fitness.evolution_stability) * 5),
    };

    json(200, buildResponse(response, startTime, traceId));

  } catch (error) {
    if (isValidationError(error)) {
      json(400, validationErrorToResponse(error, startTime));
    } else {
      // Check if term not found (CLI returns error for unknown terms with low fitness)
      const errorStr = String(error);
      if (errorStr.includes('not found') || errorStr.includes('unknown')) {
        json(404, buildErrorResponse(
          'TERM_NOT_FOUND',
          `Term "${params.term}" not found in ontology`,
          { term: params.term },
          startTime
        ));
      } else {
        json(500, errorToResponse(error, startTime));
      }
    }
  }
};
