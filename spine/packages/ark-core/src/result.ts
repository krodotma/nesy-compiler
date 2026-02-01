/**
 * @ark/core/result - Result Type
 *
 * Rust-inspired Result type for explicit error handling.
 * Avoids exceptions for expected error cases.
 *
 * @module
 */

/**
 * Success result
 */
export interface Ok<T> {
  readonly ok: true;
  readonly value: T;
}

/**
 * Error result
 */
export interface Err<E> {
  readonly ok: false;
  readonly error: E;
}

/**
 * Result type - either Ok(value) or Err(error)
 */
export type Result<T, E = Error> = Ok<T> | Err<E>;

/**
 * Create a success result
 */
export function ok<T>(value: T): Ok<T> {
  return { ok: true, value };
}

/**
 * Create an error result
 */
export function err<E>(error: E): Err<E> {
  return { ok: false, error };
}

/**
 * Type guard for Ok
 */
export function isOk<T, E>(result: Result<T, E>): result is Ok<T> {
  return result.ok;
}

/**
 * Type guard for Err
 */
export function isErr<T, E>(result: Result<T, E>): result is Err<E> {
  return !result.ok;
}

/**
 * Unwrap a result, throwing if error
 */
export function unwrap<T, E>(result: Result<T, E>): T {
  if (result.ok) {
    return result.value;
  }
  throw result.error;
}

/**
 * Unwrap with default value
 */
export function unwrapOr<T, E>(result: Result<T, E>, defaultValue: T): T {
  return result.ok ? result.value : defaultValue;
}

/**
 * Unwrap with lazy default
 */
export function unwrapOrElse<T, E>(
  result: Result<T, E>,
  fn: (error: E) => T
): T {
  return result.ok ? result.value : fn(result.error);
}

/**
 * Map the success value
 */
export function map<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => U
): Result<U, E> {
  if (result.ok) {
    return ok(fn(result.value));
  }
  return result;
}

/**
 * Map the error value
 */
export function mapErr<T, E, F>(
  result: Result<T, E>,
  fn: (error: E) => F
): Result<T, F> {
  if (!result.ok) {
    return err(fn(result.error));
  }
  return result;
}

/**
 * Flat map (bind) - chain Result-returning operations
 */
export function flatMap<T, U, E>(
  result: Result<T, E>,
  fn: (value: T) => Result<U, E>
): Result<U, E> {
  if (result.ok) {
    return fn(result.value);
  }
  return result;
}

/**
 * Convert a Promise to a Result
 */
export async function fromPromise<T, E = Error>(
  promise: Promise<T>
): Promise<Result<T, E>> {
  try {
    const value = await promise;
    return ok(value);
  } catch (error) {
    return err(error as E);
  }
}

/**
 * Convert a throwing function to a Result
 */
export function fromThrowable<T, E = Error>(fn: () => T): Result<T, E> {
  try {
    return ok(fn());
  } catch (error) {
    return err(error as E);
  }
}

/**
 * Combine multiple Results
 * Returns Ok with array of values if all ok, otherwise first error
 */
export function all<T, E>(results: Result<T, E>[]): Result<T[], E> {
  const values: T[] = [];

  for (const result of results) {
    if (!result.ok) {
      return result;
    }
    values.push(result.value);
  }

  return ok(values);
}

/**
 * Take first Ok result, or all errors
 */
export function any<T, E>(results: Result<T, E>[]): Result<T, E[]> {
  const errors: E[] = [];

  for (const result of results) {
    if (result.ok) {
      return result;
    }
    errors.push(result.error);
  }

  return err(errors);
}

/**
 * Partition results into successes and failures
 */
export function partition<T, E>(
  results: Result<T, E>[]
): { successes: T[]; failures: E[] } {
  const successes: T[] = [];
  const failures: E[] = [];

  for (const result of results) {
    if (result.ok) {
      successes.push(result.value);
    } else {
      failures.push(result.error);
    }
  }

  return { successes, failures };
}

/**
 * Option type - value that may or may not exist
 */
export type Option<T> = T | null;

/**
 * Some value
 */
export function some<T>(value: T): T {
  return value;
}

/**
 * No value
 */
export function none(): null {
  return null;
}

/**
 * Type guard for Some
 */
export function isSome<T>(option: Option<T>): option is T {
  return option !== null;
}

/**
 * Type guard for None
 */
export function isNone<T>(option: Option<T>): option is null {
  return option === null;
}

/**
 * Convert Option to Result
 */
export function optionToResult<T, E>(option: Option<T>, error: E): Result<T, E> {
  return isSome(option) ? ok(option) : err(error);
}

/**
 * Convert Result to Option
 */
export function resultToOption<T, E>(result: Result<T, E>): Option<T> {
  return result.ok ? result.value : null;
}
