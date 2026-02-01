/**
 * @ark/core/sextet - Sextet (Six Gates) Tests
 */
import { describe, it, expect } from 'vitest';
import {
  DEFAULT_SEXTET,
  validateSextet,
  isEffectAllowed,
  type Sextet,
  type ProvenanceGate,
  type EffectsGate,
  type LivenessGate,
  type RecoveryGate,
  type QualityGate,
  type OmegaGate,
} from './sextet.js';

describe('Sextet', () => {
  describe('DEFAULT_SEXTET', () => {
    it('should have safe provenance defaults', () => {
      expect(DEFAULT_SEXTET.provenance.appendOnly).toBe(true);
      expect(DEFAULT_SEXTET.provenance.nonDpi).toBe(true);
      expect(DEFAULT_SEXTET.provenance.replayable).toBe(true);
    });

    it('should have restrictive effects defaults', () => {
      expect(DEFAULT_SEXTET.effects.allowed.file).toBe(true);
      expect(DEFAULT_SEXTET.effects.denied.payment).toBe(true);
      expect(DEFAULT_SEXTET.effects.denied.actuation).toBe(true);
      expect(DEFAULT_SEXTET.effects.tier).toBe('low');
    });

    it('should have liveness guardrails enabled', () => {
      expect(DEFAULT_SEXTET.liveness.omegaGuardrail).toBe(true);
      expect(DEFAULT_SEXTET.liveness.deterministic).toBe(true);
      expect(DEFAULT_SEXTET.liveness.wcetBudget).toBe(60000);
    });

    it('should have conservative recovery defaults', () => {
      expect(DEFAULT_SEXTET.recovery.strategy).toBe('rollback');
      expect(DEFAULT_SEXTET.recovery.verified).toBe(false);
    });

    it('should have omega aligned by default', () => {
      expect(DEFAULT_SEXTET.omega.aligned).toBe(true);
      expect(DEFAULT_SEXTET.omega.vetoable).toBe(true);
    });
  });

  describe('validateSextet()', () => {
    it('should validate default sextet', () => {
      const result = validateSextet(DEFAULT_SEXTET);

      expect(result.valid).toBe(true);
      expect(result.blockedBy).toHaveLength(0);
    });

    it('should block if appendOnly is false (P-Gate)', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        provenance: {
          ...DEFAULT_SEXTET.provenance,
          appendOnly: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy).toContain('P-Gate: appendOnly must be true');
    });

    it('should block high-tier effects without omega approval (E-Gate)', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          ...DEFAULT_SEXTET.effects,
          tier: 'high',
        },
        omega: {
          ...DEFAULT_SEXTET.omega,
          approved: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy).toContain('E-Gate: high-tier effects require omega approval');
    });

    it('should allow high-tier effects with omega approval', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          ...DEFAULT_SEXTET.effects,
          tier: 'high',
        },
        omega: {
          ...DEFAULT_SEXTET.omega,
          approved: true,
        },
      };

      const result = validateSextet(sextet);

      expect(result.blockedBy).not.toContain('E-Gate: high-tier effects require omega approval');
    });

    it('should block non-deterministic actions without checkpoint (L-Gate)', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        liveness: {
          ...DEFAULT_SEXTET.liveness,
          deterministic: false,
        },
        recovery: {
          ...DEFAULT_SEXTET.recovery,
          checkpointAvailable: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy).toContain('L-Gate: non-deterministic actions require checkpoint');
    });

    it('should allow non-deterministic with checkpoint', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        liveness: {
          ...DEFAULT_SEXTET.liveness,
          deterministic: false,
        },
        recovery: {
          ...DEFAULT_SEXTET.recovery,
          checkpointAvailable: true,
        },
      };

      const result = validateSextet(sextet);

      expect(result.blockedBy).not.toContain('L-Gate: non-deterministic actions require checkpoint');
    });

    it('should block unverified recovery for non-low tier (R-Gate)', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          ...DEFAULT_SEXTET.effects,
          tier: 'medium',
        },
        recovery: {
          ...DEFAULT_SEXTET.recovery,
          verified: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy).toContain('R-Gate: recovery must be verified for non-low tier effects');
    });

    it('should allow unverified recovery for low tier', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          ...DEFAULT_SEXTET.effects,
          tier: 'low',
        },
        recovery: {
          ...DEFAULT_SEXTET.recovery,
          verified: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.blockedBy).not.toContain('R-Gate: recovery must be verified for non-low tier effects');
    });

    it('should block coverage floor without VOR (Q-Gate)', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        quality: {
          ...DEFAULT_SEXTET.quality,
          coverageFloor: 80,
          vorPassed: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy).toContain('Q-Gate: VOR must pass when coverage floor is set');
    });

    it('should allow zero coverage floor without VOR', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        quality: {
          ...DEFAULT_SEXTET.quality,
          coverageFloor: 0,
          vorPassed: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.blockedBy).not.toContain('Q-Gate: VOR must pass when coverage floor is set');
    });

    it('should block misaligned vetoable actions (Omega-Gate)', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        omega: {
          aligned: false,
          vetoable: true,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy).toContain('Omega-Gate: action is not aligned with system goals');
    });

    it('should allow misaligned non-vetoable actions', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        omega: {
          aligned: false,
          vetoable: false,
        },
      };

      const result = validateSextet(sextet);

      expect(result.blockedBy).not.toContain('Omega-Gate: action is not aligned with system goals');
    });

    it('should collect multiple blocks', () => {
      const sextet: Sextet = {
        provenance: {
          appendOnly: false,
          nonDpi: true,
          replayable: true,
        },
        effects: {
          allowed: {},
          denied: {},
          tier: 'high',
        },
        liveness: {
          omegaGuardrail: true,
          deterministic: false,
          wcetBudget: 60000,
        },
        recovery: {
          canary: false,
          shadow: false,
          verified: false,
          checkpointAvailable: false,
          strategy: 'rollback',
        },
        quality: {
          testsFirst: false,
          coverageFloor: 80,
          propertyTests: false,
          fuzzTests: false,
          vorPassed: false,
        },
        omega: {
          aligned: false,
          vetoable: true,
        },
      };

      const result = validateSextet(sextet);

      expect(result.valid).toBe(false);
      expect(result.blockedBy.length).toBeGreaterThan(1);
    });
  });

  describe('isEffectAllowed()', () => {
    it('should return true for explicitly allowed effects', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          allowed: { file: true, network: true },
          denied: {},
          tier: 'low',
        },
      };

      expect(isEffectAllowed(sextet, 'file')).toBe(true);
      expect(isEffectAllowed(sextet, 'network')).toBe(true);
    });

    it('should return false for not explicitly allowed effects', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          allowed: { file: true },
          denied: {},
          tier: 'low',
        },
      };

      expect(isEffectAllowed(sextet, 'network')).toBe(false);
      expect(isEffectAllowed(sextet, 'payment')).toBe(false);
    });

    it('should deny takes precedence over allowed', () => {
      const sextet: Sextet = {
        ...DEFAULT_SEXTET,
        effects: {
          allowed: { file: true, network: true },
          denied: { network: true },
          tier: 'low',
        },
      };

      expect(isEffectAllowed(sextet, 'file')).toBe(true);
      expect(isEffectAllowed(sextet, 'network')).toBe(false);
    });
  });

  describe('Type definitions', () => {
    it('should support all effect types', () => {
      const effects: EffectsGate = {
        allowed: {
          network: true,
          file: true,
          payment: false,
          actuation: false,
          spawn: true,
          env: true,
        },
        denied: {},
        tier: 'medium',
      };

      expect(effects.allowed.network).toBe(true);
    });

    it('should support all recovery strategies', () => {
      const strategies = ['rollback', 'compensate', 'retry', 'escalate'];
      const recovery: RecoveryGate = {
        canary: true,
        shadow: true,
        verified: true,
        checkpointAvailable: true,
        strategy: 'compensate',
      };
      expect(strategies).toContain(recovery.strategy);
    });

    it('should support optional quality fields', () => {
      const quality: QualityGate = {
        testsFirst: true,
        coverageFloor: 90,
        propertyTests: true,
        mutationScore: 0.8,
        fuzzTests: true,
        vorPassed: true,
      };
      expect(quality.mutationScore).toBe(0.8);
    });

    it('should support optional omega fields', () => {
      const omega: OmegaGate = {
        aligned: true,
        vetoable: true,
        approved: true,
        rule: 'no-destructive-ops',
        alignmentScore: 0.95,
      };
      expect(omega.alignmentScore).toBe(0.95);
    });
  });
});
