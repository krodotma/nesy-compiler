/**
 * @ark/core/result - Result Type Tests
 */
import { describe, it, expect } from 'vitest';
import {
  ok,
  err,
  isOk,
  isErr,
  unwrap,
  unwrapOr,
  unwrapOrElse,
  map,
  mapErr,
  flatMap,
  fromPromise,
  fromThrowable,
  all,
  any,
  partition,
  some,
  none,
  isSome,
  isNone,
  optionToResult,
  resultToOption,
  type Result,
  type Option,
} from './result.js';

describe('Result', () => {
  describe('ok()', () => {
    it('should create a success result', () => {
      const result = ok(42);
      expect(result.ok).toBe(true);
      expect(result.value).toBe(42);
    });

    it('should work with objects', () => {
      const data = { name: 'test', value: 123 };
      const result = ok(data);
      expect(result.ok).toBe(true);
      expect(result.value).toEqual(data);
    });

    it('should work with null values', () => {
      const result = ok(null);
      expect(result.ok).toBe(true);
      expect(result.value).toBe(null);
    });
  });

  describe('err()', () => {
    it('should create an error result', () => {
      const result = err(new Error('test error'));
      expect(result.ok).toBe(false);
      expect(result.error.message).toBe('test error');
    });

    it('should work with string errors', () => {
      const result = err('something went wrong');
      expect(result.ok).toBe(false);
      expect(result.error).toBe('something went wrong');
    });
  });

  describe('isOk()', () => {
    it('should return true for Ok results', () => {
      const result = ok(42);
      expect(isOk(result)).toBe(true);
    });

    it('should return false for Err results', () => {
      const result = err('error');
      expect(isOk(result)).toBe(false);
    });
  });

  describe('isErr()', () => {
    it('should return true for Err results', () => {
      const result = err('error');
      expect(isErr(result)).toBe(true);
    });

    it('should return false for Ok results', () => {
      const result = ok(42);
      expect(isErr(result)).toBe(false);
    });
  });

  describe('unwrap()', () => {
    it('should return value for Ok results', () => {
      const result = ok(42);
      expect(unwrap(result)).toBe(42);
    });

    it('should throw for Err results', () => {
      const result = err(new Error('test error'));
      expect(() => unwrap(result)).toThrow('test error');
    });
  });

  describe('unwrapOr()', () => {
    it('should return value for Ok results', () => {
      const result = ok(42);
      expect(unwrapOr(result, 0)).toBe(42);
    });

    it('should return default for Err results', () => {
      const result = err('error');
      expect(unwrapOr(result, 0)).toBe(0);
    });
  });

  describe('unwrapOrElse()', () => {
    it('should return value for Ok results', () => {
      const result = ok(42);
      expect(unwrapOrElse(result, () => 0)).toBe(42);
    });

    it('should call fn with error for Err results', () => {
      const result = err('oops');
      expect(unwrapOrElse(result, (e) => `fallback: ${e}`)).toBe('fallback: oops');
    });
  });

  describe('map()', () => {
    it('should transform Ok values', () => {
      const result = ok(10);
      const mapped = map(result, (x) => x * 2);
      expect(isOk(mapped) && mapped.value).toBe(20);
    });

    it('should pass through Err', () => {
      const result: Result<number, string> = err('error');
      const mapped = map(result, (x) => x * 2);
      expect(isErr(mapped) && mapped.error).toBe('error');
    });
  });

  describe('mapErr()', () => {
    it('should transform Err values', () => {
      const result: Result<number, string> = err('error');
      const mapped = mapErr(result, (e) => new Error(e));
      expect(isErr(mapped) && mapped.error.message).toBe('error');
    });

    it('should pass through Ok', () => {
      const result: Result<number, string> = ok(42);
      const mapped = mapErr(result, (e) => new Error(e));
      expect(isOk(mapped) && mapped.value).toBe(42);
    });
  });

  describe('flatMap()', () => {
    it('should chain Ok values', () => {
      const result = ok(10);
      const chained = flatMap(result, (x) => ok(x * 2));
      expect(isOk(chained) && chained.value).toBe(20);
    });

    it('should short-circuit on Err', () => {
      const result: Result<number, string> = err('first error');
      const chained = flatMap(result, (x) => ok(x * 2));
      expect(isErr(chained) && chained.error).toBe('first error');
    });

    it('should propagate inner Err', () => {
      const result = ok(10);
      const chained = flatMap(result, () => err('inner error'));
      expect(isErr(chained) && chained.error).toBe('inner error');
    });
  });

  describe('fromPromise()', () => {
    it('should convert resolved promise to Ok', async () => {
      const result = await fromPromise(Promise.resolve(42));
      expect(isOk(result) && result.value).toBe(42);
    });

    it('should convert rejected promise to Err', async () => {
      const result = await fromPromise(Promise.reject(new Error('async error')));
      expect(isErr(result) && result.error.message).toBe('async error');
    });
  });

  describe('fromThrowable()', () => {
    it('should convert successful function to Ok', () => {
      const result = fromThrowable(() => 42);
      expect(isOk(result) && result.value).toBe(42);
    });

    it('should convert throwing function to Err', () => {
      const result = fromThrowable(() => {
        throw new Error('sync error');
      });
      expect(isErr(result) && result.error.message).toBe('sync error');
    });
  });

  describe('all()', () => {
    it('should return Ok with all values if all Ok', () => {
      const results = [ok(1), ok(2), ok(3)];
      const combined = all(results);
      expect(isOk(combined) && combined.value).toEqual([1, 2, 3]);
    });

    it('should return first Err if any Err', () => {
      const results: Result<number, string>[] = [ok(1), err('first'), ok(3), err('second')];
      const combined = all(results);
      expect(isErr(combined) && combined.error).toBe('first');
    });

    it('should work with empty array', () => {
      const combined = all([]);
      expect(isOk(combined) && combined.value).toEqual([]);
    });
  });

  describe('any()', () => {
    it('should return first Ok if any Ok', () => {
      const results: Result<number, string>[] = [err('a'), ok(42), err('b')];
      const combined = any(results);
      expect(isOk(combined) && combined.value).toBe(42);
    });

    it('should return all errors if none Ok', () => {
      const results: Result<number, string>[] = [err('a'), err('b'), err('c')];
      const combined = any(results);
      expect(isErr(combined) && combined.error).toEqual(['a', 'b', 'c']);
    });
  });

  describe('partition()', () => {
    it('should separate successes and failures', () => {
      const results: Result<number, string>[] = [ok(1), err('a'), ok(2), err('b'), ok(3)];
      const { successes, failures } = partition(results);
      expect(successes).toEqual([1, 2, 3]);
      expect(failures).toEqual(['a', 'b']);
    });

    it('should work with all Ok', () => {
      const results = [ok(1), ok(2), ok(3)];
      const { successes, failures } = partition(results);
      expect(successes).toEqual([1, 2, 3]);
      expect(failures).toEqual([]);
    });

    it('should work with all Err', () => {
      const results: Result<number, string>[] = [err('a'), err('b')];
      const { successes, failures } = partition(results);
      expect(successes).toEqual([]);
      expect(failures).toEqual(['a', 'b']);
    });
  });
});

describe('Option', () => {
  describe('some()', () => {
    it('should return the value', () => {
      expect(some(42)).toBe(42);
    });
  });

  describe('none()', () => {
    it('should return null', () => {
      expect(none()).toBe(null);
    });
  });

  describe('isSome()', () => {
    it('should return true for values', () => {
      expect(isSome(42)).toBe(true);
      expect(isSome('hello')).toBe(true);
      expect(isSome(0)).toBe(true);
      expect(isSome(false)).toBe(true);
    });

    it('should return false for null', () => {
      expect(isSome(null)).toBe(false);
    });
  });

  describe('isNone()', () => {
    it('should return true for null', () => {
      expect(isNone(null)).toBe(true);
    });

    it('should return false for values', () => {
      expect(isNone(42)).toBe(false);
    });
  });

  describe('optionToResult()', () => {
    it('should convert Some to Ok', () => {
      const result = optionToResult(42, 'error');
      expect(isOk(result) && result.value).toBe(42);
    });

    it('should convert None to Err', () => {
      const result = optionToResult(null, 'not found');
      expect(isErr(result) && result.error).toBe('not found');
    });
  });

  describe('resultToOption()', () => {
    it('should convert Ok to Some', () => {
      const option = resultToOption(ok(42));
      expect(option).toBe(42);
    });

    it('should convert Err to None', () => {
      const option = resultToOption(err('error'));
      expect(option).toBe(null);
    });
  });
});
