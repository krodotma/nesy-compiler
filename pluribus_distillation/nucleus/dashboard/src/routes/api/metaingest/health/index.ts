/**
 * GET /api/metaingest/health
 * ==========================
 *
 * Get pipeline health status.
 *
 * No parameters required.
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import type { RequestHandler } from '@builder.io/qwik-city';
import {
  runPipeline,
  parseJSONOutput,
  CLIError,
} from '../utils/cli';
import {
  buildResponse,
  errorToResponse,
  getTraceId,
} from '../utils/response';
import type {
  HealthResponse,
  PipelineHealth,
  HealthStatus,
} from '../types';

export const onGet: RequestHandler = async ({ json, request }) => {
  const startTime = Date.now();
  const traceId = getTraceId(request.headers);

  try {
    // Get health status from pipeline
    const result = await runPipeline('health', ['--json']);
    const health = parseJSONOutput<PipelineHealth>(result);

    // Determine overall status
    const status = determineStatus(health);

    // Build component status
    const components = {
      gate: health.gate_status === 'idle' || health.gate_status === 'complete',
      tracker: !health.tracker_status.includes('error'),
      ingestor: health.ingestor_status === 'connected' || health.ingestor_status === 'fallback',
      falkordb: health.falkordb_available,
    };

    const response: HealthResponse = {
      status,
      health,
      components,
    };

    // Return appropriate status code based on health
    const statusCode = status === 'healthy' ? 200 : status === 'degraded' ? 200 : 503;
    json(statusCode, buildResponse(response, startTime, traceId));

  } catch (error) {
    // Even on error, try to return a degraded status
    if (error instanceof CLIError) {
      const degradedResponse: HealthResponse = {
        status: 'unhealthy',
        health: {
          healthy: false,
          gate_status: 'unknown',
          tracker_status: 'unknown',
          ingestor_status: 'unknown',
          falkordb_available: false,
          last_processed: null,
          total_processed: 0,
          errors_last_hour: 1,
        },
        components: {
          gate: false,
          tracker: false,
          ingestor: false,
          falkordb: false,
        },
      };

      json(503, buildResponse(degradedResponse, startTime, traceId));
    } else {
      json(500, errorToResponse(error, startTime));
    }
  }
};

/**
 * Determine overall health status from component health.
 */
function determineStatus(health: PipelineHealth): HealthStatus {
  if (health.healthy) {
    return 'healthy';
  }

  // Check for degraded vs unhealthy
  const criticalIssues = [
    health.gate_status === 'error',
    health.errors_last_hour > 10,
  ];

  const warnings = [
    !health.falkordb_available,
    health.errors_last_hour > 5,
    health.ingestor_status === 'fallback',
  ];

  if (criticalIssues.some(Boolean)) {
    return 'unhealthy';
  }

  if (warnings.some(Boolean)) {
    return 'degraded';
  }

  return 'healthy';
}
