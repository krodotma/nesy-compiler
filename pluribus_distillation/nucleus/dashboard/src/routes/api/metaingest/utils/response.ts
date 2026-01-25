/**
 * API Response Utilities
 * ======================
 *
 * Utilities for building standardized API responses.
 *
 * DKIN: v29 | Protocol: metaingest/v1
 */

import type { APIResponse, APIError, APIErrorCode } from '../types';
import { CLIError, isTimeoutError } from './cli';
import { randomUUID } from 'crypto';

// =============================================================================
// RESPONSE BUILDERS
// =============================================================================

/**
 * Build a successful API response.
 *
 * @param data - Response data
 * @param startTime - Request start time (Date.now())
 * @param traceId - Optional trace ID
 *
 * @example
 * const startTime = Date.now();
 * // ... do work ...
 * return buildResponse({ terms: [...] }, startTime);
 */
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

/**
 * Build an error API response.
 *
 * @param code - Error code
 * @param message - Human-readable message
 * @param details - Optional additional details
 * @param startTime - Optional request start time
 *
 * @example
 * return buildErrorResponse('TERM_NOT_FOUND', 'Term "foo" does not exist');
 */
export function buildErrorResponse(
  code: APIErrorCode,
  message: string,
  details?: Record<string, unknown>,
  startTime?: number
): APIResponse<never> {
  return {
    success: false,
    error: { code, message, details },
    meta: {
      timestamp: new Date().toISOString(),
      duration_ms: startTime ? Date.now() - startTime : 0,
    },
  };
}

/**
 * Convert an unknown error to an API error response.
 *
 * @param error - The caught error
 * @param startTime - Request start time
 */
export function errorToResponse(
  error: unknown,
  startTime: number
): APIResponse<never> {
  // CLI timeout
  if (isTimeoutError(error)) {
    return buildErrorResponse(
      'CLI_TIMEOUT',
      'Python CLI execution timed out',
      { message: (error as Error).message },
      startTime
    );
  }

  // CLI execution error
  if (error instanceof CLIError) {
    return buildErrorResponse(
      'CLI_ERROR',
      `CLI execution failed: ${error.message}`,
      {
        exitCode: error.exitCode,
        stderr: error.stderr,
      },
      startTime
    );
  }

  // Generic error
  if (error instanceof Error) {
    return buildErrorResponse(
      'INTERNAL_ERROR',
      error.message,
      { stack: process.env.NODE_ENV === 'development' ? error.stack : undefined },
      startTime
    );
  }

  // Unknown error type
  return buildErrorResponse(
    'INTERNAL_ERROR',
    'An unknown error occurred',
    { raw: String(error) },
    startTime
  );
}

// =============================================================================
// VALIDATION HELPERS
// =============================================================================

/**
 * Validate required string parameter.
 *
 * @param value - Value to validate
 * @param name - Parameter name for error message
 * @throws Error if validation fails
 */
export function requireString(value: unknown, name: string): string {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ValidationError(`${name} is required and must be a non-empty string`);
  }
  return value.trim();
}

/**
 * Validate optional number parameter.
 *
 * @param value - Value to validate
 * @param name - Parameter name
 * @param min - Minimum value (inclusive)
 * @param max - Maximum value (inclusive)
 * @param defaultValue - Default if not provided
 */
export function optionalNumber(
  value: unknown,
  name: string,
  min: number,
  max: number,
  defaultValue: number
): number {
  if (value === undefined || value === null || value === '') {
    return defaultValue;
  }

  const num = typeof value === 'number' ? value : parseInt(String(value), 10);

  if (isNaN(num)) {
    throw new ValidationError(`${name} must be a number`);
  }

  if (num < min || num > max) {
    throw new ValidationError(`${name} must be between ${min} and ${max}`);
  }

  return num;
}

/**
 * Validate optional boolean parameter.
 */
export function optionalBoolean(value: unknown, defaultValue: boolean): boolean {
  if (value === undefined || value === null || value === '') {
    return defaultValue;
  }
  if (typeof value === 'boolean') {
    return value;
  }
  if (value === 'true' || value === '1') {
    return true;
  }
  if (value === 'false' || value === '0') {
    return false;
  }
  return defaultValue;
}

/**
 * Validate enum parameter.
 */
export function validateEnum<T extends string>(
  value: unknown,
  validValues: readonly T[],
  name: string,
  defaultValue?: T
): T {
  if (value === undefined || value === null || value === '') {
    if (defaultValue !== undefined) {
      return defaultValue;
    }
    throw new ValidationError(`${name} is required`);
  }

  if (!validValues.includes(value as T)) {
    throw new ValidationError(
      `${name} must be one of: ${validValues.join(', ')}`
    );
  }

  return value as T;
}

/**
 * Validation error class
 */
export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

/**
 * Check if error is a validation error
 */
export function isValidationError(error: unknown): error is ValidationError {
  return error instanceof ValidationError;
}

/**
 * Convert validation error to response
 */
export function validationErrorToResponse(
  error: ValidationError,
  startTime: number
): APIResponse<never> {
  return buildErrorResponse(
    'VALIDATION_FAILED',
    error.message,
    undefined,
    startTime
  );
}

// =============================================================================
// TRACE ID GENERATION
// =============================================================================

/**
 * Generate a unique trace ID for request tracking.
 */
export function generateTraceId(): string {
  return randomUUID();
}

/**
 * Extract trace ID from request headers.
 */
export function getTraceId(headers: Headers): string {
  return headers.get('x-trace-id') ||
         headers.get('x-request-id') ||
         generateTraceId();
}

// =============================================================================
// PAGINATION HELPERS
// =============================================================================

/**
 * Parse pagination parameters from query string.
 */
export function parsePagination(
  query: URLSearchParams,
  defaultLimit: number = 50,
  maxLimit: number = 200
): { limit: number; offset: number } {
  const limit = Math.min(
    optionalNumber(query.get('limit'), 'limit', 1, maxLimit, defaultLimit),
    maxLimit
  );
  const offset = optionalNumber(query.get('offset'), 'offset', 0, Infinity, 0);

  return { limit, offset };
}

/**
 * Build pagination metadata for response.
 */
export function buildPaginationMeta(
  total: number,
  limit: number,
  offset: number
): { total: number; limit: number; offset: number; has_more: boolean } {
  return {
    total,
    limit,
    offset,
    has_more: offset + limit < total,
  };
}
