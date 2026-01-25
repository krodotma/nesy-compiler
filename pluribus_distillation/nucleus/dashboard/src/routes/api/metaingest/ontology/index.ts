/**
 * GET /api/metaingest/ontology
 * ============================
 *
 * List all ontology terms with fitness scores.
 *
 * Query Parameters:
 * - status: 'active' | 'superseded' | 'all' (default: 'all')
 * - min_fitness: number 0.0-1.0 (filter by minimum fitness)
 * - limit: number (default: 50, max: 200)
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import type { RequestHandler } from '@builder.io/qwik-city';
import {
  runOntologyEvolver,
  parseJSONOutput,
  parseTabularOutput,
} from '../utils/cli';
import {
  buildResponse,
  errorToResponse,
  optionalNumber,
  validateEnum,
  getTraceId,
  isValidationError,
  validationErrorToResponse,
} from '../utils/response';
import type {
  ListOntologyResponse,
  OntologyTermSummary,
} from '../types';

// Status values for validation
const VALID_STATUS = ['active', 'superseded', 'all'] as const;

export const onGet: RequestHandler = async ({ query, json, request }) => {
  const startTime = Date.now();
  const traceId = getTraceId(request.headers);

  try {
    // Parse and validate query parameters
    const status = validateEnum(
      query.get('status'),
      VALID_STATUS,
      'status',
      'all'
    );
    const minFitness = optionalNumber(
      query.get('min_fitness'),
      'min_fitness',
      0,
      1,
      0
    );
    const limit = optionalNumber(query.get('limit'), 'limit', 1, 200, 50);

    // Get status first for counts
    const statusResult = await runOntologyEvolver('status', ['--json']);
    const statusData = parseJSONOutput<{
      total_terms: number;
      active_terms: number;
      superseded_terms: number;
      evolution_events: number;
      fitness_threshold: number;
      hgt_threshold: number;
    }>(statusResult);

    // Get terms list
    const termsResult = await runOntologyEvolver('terms', ['--limit', String(limit * 2)]);

    // Parse tabular output
    // Format: "  [bar] 0.85 term_name"
    const terms = parseTermsList(termsResult.stdout, status, minFitness, limit);

    const response: ListOntologyResponse = {
      terms,
      total_terms: statusData.total_terms,
      active_terms: statusData.active_terms,
      superseded_terms: statusData.superseded_terms,
    };

    json(200, buildResponse(response, startTime, traceId));

  } catch (error) {
    if (isValidationError(error)) {
      json(400, validationErrorToResponse(error, startTime));
    } else {
      json(500, errorToResponse(error, startTime));
    }
  }
};

/**
 * Parse terms list from CLI output.
 *
 * Input format:
 * ```
 * Active Terms (N):
 * ----------------------------------------
 *   ██████████ 0.85 term_name
 *   ████████░░ 0.72 other_term
 * ```
 */
function parseTermsList(
  output: string,
  statusFilter: 'active' | 'superseded' | 'all',
  minFitness: number,
  limit: number
): OntologyTermSummary[] {
  const terms: OntologyTermSummary[] = [];
  const lines = output.split('\n');

  for (const line of lines) {
    // Match lines like "  ██████████ 0.85 term_name"
    const match = line.match(/^\s*[█░]{10}\s+(\d+\.\d+)\s+(\S+)/);
    if (match) {
      const fitness = parseFloat(match[1]);
      const term = match[2];

      // All terms from 'terms' command are active
      // For superseded, would need different query
      const termStatus: 'active' | 'superseded' = 'active';

      // Apply filters
      if (fitness >= minFitness) {
        if (statusFilter === 'all' || statusFilter === termStatus) {
          terms.push({
            term,
            fitness,
            status: termStatus,
          });
        }
      }

      if (terms.length >= limit) {
        break;
      }
    }
  }

  return terms;
}
