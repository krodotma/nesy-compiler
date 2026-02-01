/**
 * @ark/core/invariants - Conservation Invariants Tests
 */
import { describe, it, expect } from 'vitest';
import {
  PHI,
  INV_PHI,
  PHI_SQUARED,
  INV_PHI_SQUARED,
  INV_PHI_CUBED,
  CMP_DISCOUNT,
  GLOBAL_CMP_FLOOR,
  CONSERVATION_INVARIANTS,
  validateInvariant,
  getInvariant,
  checkInvariants,
  fibonacci,
  fib,
  goldenDivide,
} from './invariants.js';

describe('Invariants', () => {
  describe('Golden ratio constants', () => {
    it('should have correct PHI value', () => {
      expect(PHI).toBeCloseTo(1.618033988749895, 10);
    });

    it('should have correct INV_PHI value', () => {
      expect(INV_PHI).toBeCloseTo(0.618033988749895, 10);
    });

    it('should satisfy PHI * INV_PHI = 1', () => {
      expect(PHI * INV_PHI).toBeCloseTo(1, 10);
    });

    it('should have correct PHI_SQUARED value', () => {
      expect(PHI_SQUARED).toBeCloseTo(2.618033988749895, 10);
    });

    it('should satisfy PHI_SQUARED = PHI + 1', () => {
      expect(PHI_SQUARED).toBeCloseTo(PHI + 1, 10);
    });

    it('should have correct INV_PHI_SQUARED value', () => {
      expect(INV_PHI_SQUARED).toBeCloseTo(0.381966011250105, 10);
    });

    it('should have correct INV_PHI_CUBED value', () => {
      expect(INV_PHI_CUBED).toBeCloseTo(0.236067977499790, 10);
    });
  });

  describe('CMP constants', () => {
    it('should have CMP_DISCOUNT equal to INV_PHI', () => {
      expect(CMP_DISCOUNT).toBe(INV_PHI);
    });

    it('should have GLOBAL_CMP_FLOOR equal to INV_PHI_CUBED', () => {
      expect(GLOBAL_CMP_FLOOR).toBe(INV_PHI_CUBED);
    });
  });

  describe('CONSERVATION_INVARIANTS', () => {
    it('should contain all expected values', () => {
      expect(CONSERVATION_INVARIANTS.PHI).toBe(PHI);
      expect(CONSERVATION_INVARIANTS.INV_PHI).toBe(INV_PHI);
      expect(CONSERVATION_INVARIANTS.CMP_DISCOUNT).toBe(INV_PHI);
      expect(CONSERVATION_INVARIANTS.GLOBAL_CMP_FLOOR).toBe(INV_PHI_CUBED);
      expect(CONSERVATION_INVARIANTS.CMP_CEILING).toBe(1.0);
    });

    it('should have EWC constants', () => {
      expect(CONSERVATION_INVARIANTS.EWC_LAMBDA).toBe(1000);
      expect(CONSERVATION_INVARIANTS.FISHER_SAMPLES).toBe(200);
    });

    it('should have replay buffer constants', () => {
      expect(CONSERVATION_INVARIANTS.REPLAY_ALPHA).toBe(0.6);
      expect(CONSERVATION_INVARIANTS.REPLAY_BETA).toBe(0.4);
      expect(CONSERVATION_INVARIANTS.MIN_EXPERIENCES).toBe(64);
    });

    it('should have neural network constants', () => {
      expect(CONSERVATION_INVARIANTS.LEARNING_RATE).toBe(3e-4);
      expect(CONSERVATION_INVARIANTS.GRADIENT_CLIP).toBe(1.0);
      expect(CONSERVATION_INVARIANTS.WEIGHT_DECAY).toBe(1e-4);
    });

    it('should have temporal constants', () => {
      expect(CONSERVATION_INVARIANTS.DEFAULT_TTL_S).toBe(3600);
      expect(CONSERVATION_INVARIANTS.EVENT_RETENTION_DAYS).toBe(30);
      expect(CONSERVATION_INVARIANTS.CHECKPOINT_INTERVAL).toBe(100);
    });

    it('should have bus constants', () => {
      expect(CONSERVATION_INVARIANTS.MAX_EVENT_SIZE).toBe(1_000_000);
      expect(CONSERVATION_INVARIANTS.BATCH_SIZE).toBe(100);
      expect(CONSERVATION_INVARIANTS.PUBLISH_TIMEOUT_MS).toBe(5000);
    });

    it('should have safety thresholds', () => {
      expect(CONSERVATION_INVARIANTS.THRASH_THRESHOLD).toBe(0.6);
      expect(CONSERVATION_INVARIANTS.UNCERTAINTY_THRESHOLD).toBe(0.5);
      expect(CONSERVATION_INVARIANTS.MIN_PATTERN_CONFIDENCE).toBe(0.1);
    });
  });

  describe('validateInvariant()', () => {
    it('should return true for matching values', () => {
      expect(validateInvariant('PHI', PHI)).toBe(true);
      expect(validateInvariant('CMP_DISCOUNT', INV_PHI)).toBe(true);
    });

    it('should return false for non-matching values', () => {
      expect(validateInvariant('PHI', 1.5)).toBe(false);
      expect(validateInvariant('CMP_DISCOUNT', 0.5)).toBe(false);
    });

    it('should respect tolerance', () => {
      expect(validateInvariant('PHI', PHI + 1e-7, 1e-6)).toBe(true);
      expect(validateInvariant('PHI', PHI + 1e-5, 1e-6)).toBe(false);
    });
  });

  describe('getInvariant()', () => {
    it('should return correct invariant values', () => {
      expect(getInvariant('PHI')).toBe(PHI);
      expect(getInvariant('INV_PHI')).toBe(INV_PHI);
      expect(getInvariant('EWC_LAMBDA')).toBe(1000);
      expect(getInvariant('BATCH_SIZE')).toBe(100);
    });
  });

  describe('checkInvariants()', () => {
    it('should pass for matching config', () => {
      const config = {
        PHI: PHI,
        CMP_DISCOUNT: INV_PHI,
        EWC_LAMBDA: 1000,
      };

      const result = checkInvariants(config);

      expect(result.valid).toBe(true);
      expect(result.violations).toHaveLength(0);
    });

    it('should detect violations', () => {
      const config = {
        PHI: 1.5, // Wrong!
        CMP_DISCOUNT: 0.5, // Wrong!
      };

      const result = checkInvariants(config);

      expect(result.valid).toBe(false);
      expect(result.violations).toHaveLength(2);
    });

    it('should include expected and actual in violations', () => {
      const config = { PHI: 1.5 };
      const result = checkInvariants(config);

      expect(result.violations[0].key).toBe('PHI');
      expect(result.violations[0].expected).toBe(PHI);
      expect(result.violations[0].actual).toBe(1.5);
    });

    it('should pass for config with no matching keys', () => {
      const config = { unknown: 123 };
      const result = checkInvariants(config);

      expect(result.valid).toBe(true);
    });
  });

  describe('fibonacci()', () => {
    it('should generate fibonacci sequence', () => {
      const seq = [...fibonacci(10)];
      expect(seq).toEqual([0, 1, 1, 2, 3, 5, 8, 13, 21, 34]);
    });

    it('should handle zero', () => {
      const seq = [...fibonacci(0)];
      expect(seq).toEqual([]);
    });

    it('should handle one', () => {
      const seq = [...fibonacci(1)];
      expect(seq).toEqual([0]);
    });
  });

  describe('fib()', () => {
    it('should return correct fibonacci numbers', () => {
      expect(fib(0)).toBe(0);
      expect(fib(1)).toBe(1);
      expect(fib(2)).toBe(1);
      expect(fib(3)).toBe(2);
      expect(fib(4)).toBe(3);
      expect(fib(5)).toBe(5);
      expect(fib(10)).toBe(55);
    });

    it('should handle larger numbers', () => {
      expect(fib(20)).toBe(6765);
      expect(fib(30)).toBe(832040);
    });
  });

  describe('goldenDivide()', () => {
    it('should divide value in golden ratio', () => {
      const [larger, smaller] = goldenDivide(100);

      expect(larger).toBeCloseTo(61.8033988749895, 5);
      expect(smaller).toBeCloseTo(38.1966011250105, 5);
      expect(larger + smaller).toBeCloseTo(100, 10);
    });

    it('should maintain golden ratio between parts', () => {
      const [larger, smaller] = goldenDivide(100);

      // larger/smaller should approximately equal phi
      expect(larger / smaller).toBeCloseTo(PHI, 5);
    });

    it('should work with different values', () => {
      const [larger, smaller] = goldenDivide(1);
      expect(larger).toBeCloseTo(INV_PHI, 10);
      expect(smaller).toBeCloseTo(1 - INV_PHI, 10);
    });
  });
});
