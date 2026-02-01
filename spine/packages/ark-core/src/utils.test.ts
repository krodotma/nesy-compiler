/**
 * @ark/core/utils - Utility Functions Tests
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  uuid,
  now,
  sleep,
  retry,
  debounce,
  throttle,
  deepClone,
  deepMerge,
  pick,
  omit,
  groupBy,
  chunk,
  unique,
  uniqueBy,
  flatten,
  clamp,
  lerp,
  mapRange,
  formatBytes,
  formatDuration,
  parseDuration,
  hash,
  deferred,
  withTimeout,
  safeJsonParse,
  assert,
  assertNever,
  isDefined,
  isObject,
  isArray,
  isString,
  isNumber,
  isFunction,
  memoize,
  LRUCache,
} from './utils.js';

describe('Utils', () => {
  describe('uuid()', () => {
    it('should generate a valid UUID v4', () => {
      const id = uuid();
      expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i);
    });

    it('should generate unique IDs', () => {
      const ids = new Set(Array.from({ length: 100 }, uuid));
      expect(ids.size).toBe(100);
    });
  });

  describe('now()', () => {
    it('should return current timestamp', () => {
      const before = Date.now();
      const result = now();
      const after = Date.now();
      expect(result).toBeGreaterThanOrEqual(before);
      expect(result).toBeLessThanOrEqual(after);
    });
  });

  describe('sleep()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should resolve after specified ms', async () => {
      const promise = sleep(1000);
      vi.advanceTimersByTime(1000);
      await expect(promise).resolves.toBeUndefined();
    });
  });

  describe('retry()', () => {
    it('should succeed on first attempt', async () => {
      const fn = vi.fn().mockResolvedValue('success');
      const result = await retry(fn);
      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should retry on failure', async () => {
      const fn = vi.fn()
        .mockRejectedValueOnce(new Error('fail 1'))
        .mockRejectedValueOnce(new Error('fail 2'))
        .mockResolvedValue('success');

      const result = await retry(fn, { maxAttempts: 3, initialDelay: 1 });
      expect(result).toBe('success');
      expect(fn).toHaveBeenCalledTimes(3);
    });

    it('should throw after max attempts', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('always fails'));

      await expect(retry(fn, { maxAttempts: 3, initialDelay: 1 }))
        .rejects.toThrow('always fails');
      expect(fn).toHaveBeenCalledTimes(3);
    });

    it('should respect retryIf condition', async () => {
      const fn = vi.fn().mockRejectedValue(new Error('fatal'));

      await expect(retry(fn, {
        maxAttempts: 3,
        initialDelay: 1,
        retryIf: () => false,
      })).rejects.toThrow('fatal');
      expect(fn).toHaveBeenCalledTimes(1);
    });
  });

  describe('debounce()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should debounce function calls', () => {
      const fn = vi.fn();
      const debounced = debounce(fn, 100);

      debounced();
      debounced();
      debounced();

      expect(fn).not.toHaveBeenCalled();

      vi.advanceTimersByTime(100);
      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should reset timer on subsequent calls', () => {
      const fn = vi.fn();
      const debounced = debounce(fn, 100);

      debounced();
      vi.advanceTimersByTime(50);
      debounced();
      vi.advanceTimersByTime(50);

      expect(fn).not.toHaveBeenCalled();

      vi.advanceTimersByTime(50);
      expect(fn).toHaveBeenCalledTimes(1);
    });
  });

  describe('throttle()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should throttle function calls', () => {
      const fn = vi.fn();
      const throttled = throttle(fn, 100);

      throttled();
      throttled();
      throttled();

      expect(fn).toHaveBeenCalledTimes(1);
    });

    it('should allow calls after limit expires', () => {
      const fn = vi.fn();
      const throttled = throttle(fn, 100);

      throttled();
      expect(fn).toHaveBeenCalledTimes(1);

      vi.advanceTimersByTime(100);
      throttled();
      expect(fn).toHaveBeenCalledTimes(2);
    });
  });

  describe('deepClone()', () => {
    it('should clone objects', () => {
      const obj = { a: 1, b: { c: 2 } };
      const clone = deepClone(obj);
      expect(clone).toEqual(obj);
      expect(clone).not.toBe(obj);
      expect(clone.b).not.toBe(obj.b);
    });

    it('should clone arrays', () => {
      const arr = [1, [2, 3], { a: 4 }];
      const clone = deepClone(arr);
      expect(clone).toEqual(arr);
      expect(clone).not.toBe(arr);
    });
  });

  describe('deepMerge()', () => {
    it('should merge objects', () => {
      const a = { x: 1, y: { z: 2 } };
      const b = { y: { w: 3 }, v: 4 };
      const merged = deepMerge(a, b);
      expect(merged).toEqual({ x: 1, y: { z: 2, w: 3 }, v: 4 });
    });

    it('should handle multiple objects', () => {
      const result = deepMerge({ a: 1 }, { b: 2 }, { c: 3 });
      expect(result).toEqual({ a: 1, b: 2, c: 3 });
    });

    it('should override with later values', () => {
      const result = deepMerge({ a: 1 }, { a: 2 });
      expect(result).toEqual({ a: 2 });
    });
  });

  describe('pick()', () => {
    it('should pick specified keys', () => {
      const obj = { a: 1, b: 2, c: 3 };
      expect(pick(obj, ['a', 'c'])).toEqual({ a: 1, c: 3 });
    });

    it('should ignore missing keys', () => {
      const obj = { a: 1 };
      expect(pick(obj, ['a', 'b' as 'a'])).toEqual({ a: 1 });
    });
  });

  describe('omit()', () => {
    it('should omit specified keys', () => {
      const obj = { a: 1, b: 2, c: 3 };
      expect(omit(obj, ['b'])).toEqual({ a: 1, c: 3 });
    });
  });

  describe('groupBy()', () => {
    it('should group by key function', () => {
      const arr = [
        { type: 'a', value: 1 },
        { type: 'b', value: 2 },
        { type: 'a', value: 3 },
      ];
      const grouped = groupBy(arr, (item) => item.type);
      expect(grouped).toEqual({
        a: [{ type: 'a', value: 1 }, { type: 'a', value: 3 }],
        b: [{ type: 'b', value: 2 }],
      });
    });
  });

  describe('chunk()', () => {
    it('should chunk array', () => {
      expect(chunk([1, 2, 3, 4, 5], 2)).toEqual([[1, 2], [3, 4], [5]]);
    });

    it('should handle empty array', () => {
      expect(chunk([], 2)).toEqual([]);
    });
  });

  describe('unique()', () => {
    it('should remove duplicates', () => {
      expect(unique([1, 2, 2, 3, 3, 3])).toEqual([1, 2, 3]);
    });
  });

  describe('uniqueBy()', () => {
    it('should remove duplicates by key', () => {
      const arr = [{ id: 1 }, { id: 2 }, { id: 1 }];
      expect(uniqueBy(arr, (x) => x.id)).toEqual([{ id: 1 }, { id: 2 }]);
    });
  });

  describe('flatten()', () => {
    it('should flatten nested arrays', () => {
      expect(flatten([1, [2, 3], 4])).toEqual([1, 2, 3, 4]);
    });
  });

  describe('clamp()', () => {
    it('should clamp value to range', () => {
      expect(clamp(5, 0, 10)).toBe(5);
      expect(clamp(-5, 0, 10)).toBe(0);
      expect(clamp(15, 0, 10)).toBe(10);
    });
  });

  describe('lerp()', () => {
    it('should interpolate linearly', () => {
      expect(lerp(0, 10, 0)).toBe(0);
      expect(lerp(0, 10, 1)).toBe(10);
      expect(lerp(0, 10, 0.5)).toBe(5);
    });
  });

  describe('mapRange()', () => {
    it('should map value between ranges', () => {
      expect(mapRange(5, 0, 10, 0, 100)).toBe(50);
      expect(mapRange(0, 0, 10, 0, 100)).toBe(0);
      expect(mapRange(10, 0, 10, 0, 100)).toBe(100);
    });
  });

  describe('formatBytes()', () => {
    it('should format bytes', () => {
      expect(formatBytes(0)).toBe('0 B');
      expect(formatBytes(1024)).toBe('1 KB');
      expect(formatBytes(1048576)).toBe('1 MB');
      expect(formatBytes(1536)).toBe('1.5 KB');
    });
  });

  describe('formatDuration()', () => {
    it('should format durations', () => {
      expect(formatDuration(500)).toBe('500ms');
      expect(formatDuration(1500)).toBe('1.5s');
      expect(formatDuration(90000)).toBe('1.5m');
      expect(formatDuration(5400000)).toBe('1.5h');
    });
  });

  describe('parseDuration()', () => {
    it('should parse duration strings', () => {
      expect(parseDuration('100ms')).toBe(100);
      expect(parseDuration('5s')).toBe(5000);
      expect(parseDuration('2m')).toBe(120000);
      expect(parseDuration('1h')).toBe(3600000);
      expect(parseDuration('1d')).toBe(86400000);
    });

    it('should handle decimals', () => {
      expect(parseDuration('1.5s')).toBe(1500);
    });

    it('should throw for invalid format', () => {
      expect(() => parseDuration('invalid')).toThrow('Invalid duration');
    });
  });

  describe('hash()', () => {
    it('should generate consistent hash', () => {
      expect(hash('test')).toBe(hash('test'));
    });

    it('should produce different hashes for different strings', () => {
      expect(hash('foo')).not.toBe(hash('bar'));
    });
  });

  describe('deferred()', () => {
    it('should create resolvable promise', async () => {
      const d = deferred<number>();
      d.resolve(42);
      expect(await d.promise).toBe(42);
    });

    it('should create rejectable promise', async () => {
      const d = deferred<number>();
      d.reject(new Error('test'));
      await expect(d.promise).rejects.toThrow('test');
    });
  });

  describe('withTimeout()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
    });

    afterEach(() => {
      vi.useRealTimers();
    });

    it('should resolve if within timeout', async () => {
      const promise = withTimeout(Promise.resolve(42), 1000);
      await expect(promise).resolves.toBe(42);
    });

    it('should reject on timeout', async () => {
      const slowPromise = new Promise(() => {});
      const promise = withTimeout(slowPromise, 100);
      vi.advanceTimersByTime(100);
      await expect(promise).rejects.toThrow('Operation timed out');
    });

    it('should use custom message', async () => {
      const slowPromise = new Promise(() => {});
      const promise = withTimeout(slowPromise, 100, 'Custom timeout');
      vi.advanceTimersByTime(100);
      await expect(promise).rejects.toThrow('Custom timeout');
    });
  });

  describe('safeJsonParse()', () => {
    it('should parse valid JSON', () => {
      expect(safeJsonParse('{"a":1}', {})).toEqual({ a: 1 });
    });

    it('should return fallback for invalid JSON', () => {
      expect(safeJsonParse('not json', { default: true })).toEqual({ default: true });
    });
  });

  describe('assert()', () => {
    it('should not throw for truthy conditions', () => {
      expect(() => assert(true, 'message')).not.toThrow();
    });

    it('should throw for falsy conditions', () => {
      expect(() => assert(false, 'failed')).toThrow('Assertion failed: failed');
    });
  });

  describe('assertNever()', () => {
    it('should throw with message', () => {
      expect(() => assertNever('x' as never, 'custom message')).toThrow('custom message');
    });

    it('should throw with default message', () => {
      expect(() => assertNever('val' as never)).toThrow('Unexpected value: val');
    });
  });

  describe('isDefined()', () => {
    it('should return true for defined values', () => {
      expect(isDefined(0)).toBe(true);
      expect(isDefined('')).toBe(true);
      expect(isDefined(false)).toBe(true);
    });

    it('should return false for null/undefined', () => {
      expect(isDefined(null)).toBe(false);
      expect(isDefined(undefined)).toBe(false);
    });
  });

  describe('isObject()', () => {
    it('should return true for objects', () => {
      expect(isObject({})).toBe(true);
      expect(isObject({ a: 1 })).toBe(true);
    });

    it('should return false for non-objects', () => {
      expect(isObject(null)).toBe(false);
      expect(isObject([])).toBe(false);
      expect(isObject('string')).toBe(false);
    });
  });

  describe('isArray()', () => {
    it('should return true for arrays', () => {
      expect(isArray([])).toBe(true);
      expect(isArray([1, 2])).toBe(true);
    });

    it('should return false for non-arrays', () => {
      expect(isArray({})).toBe(false);
      expect(isArray('string')).toBe(false);
    });
  });

  describe('isString()', () => {
    it('should return true for strings', () => {
      expect(isString('')).toBe(true);
      expect(isString('hello')).toBe(true);
    });

    it('should return false for non-strings', () => {
      expect(isString(123)).toBe(false);
      expect(isString(null)).toBe(false);
    });
  });

  describe('isNumber()', () => {
    it('should return true for numbers', () => {
      expect(isNumber(0)).toBe(true);
      expect(isNumber(42)).toBe(true);
      expect(isNumber(-1.5)).toBe(true);
    });

    it('should return false for NaN', () => {
      expect(isNumber(NaN)).toBe(false);
    });

    it('should return false for non-numbers', () => {
      expect(isNumber('42')).toBe(false);
    });
  });

  describe('isFunction()', () => {
    it('should return true for functions', () => {
      expect(isFunction(() => {})).toBe(true);
      expect(isFunction(function() {})).toBe(true);
    });

    it('should return false for non-functions', () => {
      expect(isFunction({})).toBe(false);
      expect(isFunction(null)).toBe(false);
    });
  });

  describe('memoize()', () => {
    it('should cache results', () => {
      let callCount = 0;
      const fn = memoize((x: number) => {
        callCount++;
        return x * 2;
      });

      expect(fn(5)).toBe(10);
      expect(fn(5)).toBe(10);
      expect(callCount).toBe(1);
    });

    it('should use custom key function', () => {
      const fn = memoize(
        (a: number, b: number) => a + b,
        (a, b) => `${a}-${b}`
      );

      expect(fn(1, 2)).toBe(3);
      expect(fn(1, 2)).toBe(3);
    });
  });

  describe('LRUCache', () => {
    it('should store and retrieve values', () => {
      const cache = new LRUCache<string, number>(3);
      cache.set('a', 1);
      expect(cache.get('a')).toBe(1);
    });

    it('should evict oldest entries', () => {
      const cache = new LRUCache<string, number>(2);
      cache.set('a', 1);
      cache.set('b', 2);
      cache.set('c', 3);

      expect(cache.get('a')).toBeUndefined();
      expect(cache.get('b')).toBe(2);
      expect(cache.get('c')).toBe(3);
    });

    it('should update LRU order on get', () => {
      const cache = new LRUCache<string, number>(2);
      cache.set('a', 1);
      cache.set('b', 2);
      cache.get('a'); // a is now most recently used
      cache.set('c', 3); // should evict b

      expect(cache.get('a')).toBe(1);
      expect(cache.get('b')).toBeUndefined();
      expect(cache.get('c')).toBe(3);
    });

    it('should report correct size', () => {
      const cache = new LRUCache<string, number>(3);
      expect(cache.size).toBe(0);
      cache.set('a', 1);
      expect(cache.size).toBe(1);
    });

    it('should check existence with has()', () => {
      const cache = new LRUCache<string, number>(3);
      cache.set('a', 1);
      expect(cache.has('a')).toBe(true);
      expect(cache.has('b')).toBe(false);
    });

    it('should delete entries', () => {
      const cache = new LRUCache<string, number>(3);
      cache.set('a', 1);
      expect(cache.delete('a')).toBe(true);
      expect(cache.get('a')).toBeUndefined();
    });

    it('should clear all entries', () => {
      const cache = new LRUCache<string, number>(3);
      cache.set('a', 1);
      cache.set('b', 2);
      cache.clear();
      expect(cache.size).toBe(0);
    });
  });
});
