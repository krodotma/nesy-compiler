/**
 * POST /api/metaingest/ontology/[term]/evolve
 * ============================================
 *
 * Trigger evolution for a term.
 *
 * Path Parameters:
 * - term: string - The term to evolve
 *
 * Request Body:
 * - context: string (required) - Usage context for evolution
 * - force: boolean (optional) - Force evolution regardless of fitness
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import type { RequestHandler } from '@builder.io/qwik-city';
import {
  runOntologyEvolver,
  parseJSONOutput,
  tryParseJSONOutput,
} from '../../../utils/cli';
import {
  buildResponse,
  buildErrorResponse,
  errorToResponse,
  requireString,
  optionalBoolean,
  getTraceId,
  isValidationError,
  validationErrorToResponse,
} from '../../../utils/response';
import type {
  EvolveTermResponse,
  EvolutionRecord,
} from '../../../types';

interface EvolveRequestBody {
  context: string;
  force?: boolean;
}

export const onPost: RequestHandler = async ({ params, json, request }) => {
  const startTime = Date.now();
  const traceId = getTraceId(request.headers);

  try {
    // Validate term parameter
    const term = requireString(params.term, 'term');

    // Parse request body
    const body = await request.json() as EvolveRequestBody;
    const context = requireString(body.context, 'context');
    const force = optionalBoolean(body.force, false);

    // Build CLI arguments
    const args = [term, '--context', context, '--json'];
    if (force) {
      args.push('--force');
    }

    // Execute evolution
    const result = await runOntologyEvolver('evolve', args);

    // Parse result - can be either evolution record or no-evolution response
    const parsed = tryParseJSONOutput<EvolutionRecord | { evolved: false; reason: string }>(result);

    let response: EvolveTermResponse;

    if (!parsed) {
      // Parse from text output if JSON parsing failed
      if (result.stdout.includes('No evolution needed')) {
        response = {
          evolved: false,
          reason: 'fitness above threshold',
        };
      } else if (result.stdout.includes('Evolution occurred')) {
        // Try to extract evolution info from text
        response = {
          evolved: true,
          record: extractEvolutionFromText(result.stdout, term),
        };
      } else {
        response = {
          evolved: false,
          reason: result.stdout || 'Unknown result',
        };
      }
    } else if ('evolved' in parsed && parsed.evolved === false) {
      response = {
        evolved: false,
        reason: parsed.reason,
      };
    } else {
      response = {
        evolved: true,
        record: parsed as EvolutionRecord,
      };
    }

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
 * Extract evolution record from text output when JSON is not available.
 */
function extractEvolutionFromText(output: string, sourceTerm: string): EvolutionRecord | undefined {
  const lines = output.split('\n');

  let evolutionType = 'mutation';
  let targetTerm = sourceTerm;
  let fitnessBefore = 0;
  let fitnessAfter = 0;
  let attestation = '';

  for (const line of lines) {
    if (line.includes('Evolution occurred:')) {
      const match = line.match(/Evolution occurred:\s*(\w+)/);
      if (match) evolutionType = match[1].toLowerCase();
    }
    if (line.includes('Target:')) {
      const match = line.match(/Target:\s*(\S+)/);
      if (match) targetTerm = match[1];
    }
    if (line.includes('Fitness:')) {
      const match = line.match(/Fitness:\s*([\d.]+)\s*->\s*([\d.]+)/);
      if (match) {
        fitnessBefore = parseFloat(match[1]);
        fitnessAfter = parseFloat(match[2]);
      }
    }
    if (line.includes('Attestation:')) {
      const match = line.match(/Attestation:\s*(\w+)/);
      if (match) attestation = match[1];
    }
  }

  return {
    evolution_id: `evo_${Date.now().toString(36)}`,
    evolution_type: evolutionType as EvolutionRecord['evolution_type'],
    source_term: sourceTerm,
    target_term: targetTerm,
    fitness_before: fitnessBefore,
    fitness_after: fitnessAfter,
    context: 'Extracted from CLI output',
    lineage_attestation: attestation,
    timestamp: new Date().toISOString(),
  };
}
