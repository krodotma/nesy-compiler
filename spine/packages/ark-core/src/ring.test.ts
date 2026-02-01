/**
 * @ark/core/ring - Ring Security Hierarchy Tests
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  Ring,
  RING_POLICIES,
  parseURN,
  formatURN,
  canAccess,
  isEffectAllowedAtRing,
  getRequiredApproval,
  createRingContext,
  descendRing,
  requestEscalation,
  type URN,
} from './ring.js';

describe('Ring', () => {
  describe('Ring enum', () => {
    it('should have correct values', () => {
      expect(Ring.Kernel).toBe(0);
      expect(Ring.Service).toBe(1);
      expect(Ring.User).toBe(2);
      expect(Ring.Ephemeral).toBe(3);
    });
  });

  describe('RING_POLICIES', () => {
    it('should have policies for all rings', () => {
      expect(RING_POLICIES[Ring.Kernel]).toBeDefined();
      expect(RING_POLICIES[Ring.Service]).toBeDefined();
      expect(RING_POLICIES[Ring.User]).toBeDefined();
      expect(RING_POLICIES[Ring.Ephemeral]).toBeDefined();
    });

    it('should have kernel as immutable', () => {
      expect(RING_POLICIES[Ring.Kernel].canModifyLower).toBe(false);
      expect(RING_POLICIES[Ring.Kernel].requiresApproval).toBe('formal_proof');
    });

    it('should have service with omega approval', () => {
      expect(RING_POLICIES[Ring.Service].requiresApproval).toBe('omega');
    });

    it('should have user with witness approval', () => {
      expect(RING_POLICIES[Ring.User].requiresApproval).toBe('witness');
    });

    it('should have ephemeral with no approval', () => {
      expect(RING_POLICIES[Ring.Ephemeral].requiresApproval).toBe('none');
    });
  });

  describe('parseURN()', () => {
    it('should parse valid URN strings', () => {
      const urn = parseURN('ring0/axioms/core/append_only');

      expect(urn).toEqual({
        ring: Ring.Kernel,
        domain: 'axioms',
        type: 'core',
        id: 'append_only',
      });
    });

    it('should parse all ring levels', () => {
      expect(parseURN('ring0/a/b/c')?.ring).toBe(Ring.Kernel);
      expect(parseURN('ring1/a/b/c')?.ring).toBe(Ring.Service);
      expect(parseURN('ring2/a/b/c')?.ring).toBe(Ring.User);
      expect(parseURN('ring3/a/b/c')?.ring).toBe(Ring.Ephemeral);
    });

    it('should handle complex IDs', () => {
      const urn = parseURN('ring2/tools/cli/pli-v2.0');
      expect(urn?.id).toBe('pli-v2.0');
    });

    it('should handle IDs with slashes', () => {
      const urn = parseURN('ring1/mcp/server/path/to/resource');
      expect(urn?.id).toBe('path/to/resource');
    });

    it('should return null for invalid URNs', () => {
      expect(parseURN('invalid')).toBeNull();
      expect(parseURN('ring4/a/b/c')).toBeNull();
      expect(parseURN('ring0/a/b')).toBeNull();
      expect(parseURN('')).toBeNull();
    });
  });

  describe('formatURN()', () => {
    it('should format URN object to string', () => {
      const urn: URN = {
        ring: Ring.Service,
        domain: 'mcp',
        type: 'server',
        id: 'rhizome',
      };

      expect(formatURN(urn)).toBe('ring1/mcp/server/rhizome');
    });

    it('should roundtrip correctly', () => {
      const original = 'ring2/tools/cli/pli';
      const parsed = parseURN(original)!;
      expect(formatURN(parsed)).toBe(original);
    });
  });

  describe('canAccess()', () => {
    it('should allow access to same ring', () => {
      expect(canAccess(Ring.User, Ring.User, 'read')).toBe(true);
      expect(canAccess(Ring.User, Ring.User, 'write')).toBe(true);
    });

    it('should allow access to less privileged rings', () => {
      expect(canAccess(Ring.Service, Ring.User, 'read')).toBe(true);
      expect(canAccess(Ring.Service, Ring.Ephemeral, 'read')).toBe(true);
      expect(canAccess(Ring.Kernel, Ring.Ephemeral, 'read')).toBe(true);
    });

    it('should allow read access to more privileged rings', () => {
      expect(canAccess(Ring.User, Ring.Service, 'read')).toBe(true);
      expect(canAccess(Ring.Ephemeral, Ring.Kernel, 'read')).toBe(true);
    });

    it('should deny write access to more privileged rings', () => {
      expect(canAccess(Ring.User, Ring.Service, 'write')).toBe(false);
      expect(canAccess(Ring.Ephemeral, Ring.Kernel, 'write')).toBe(false);
    });
  });

  describe('isEffectAllowedAtRing()', () => {
    it('should check allowed effects for kernel', () => {
      expect(isEffectAllowedAtRing(Ring.Kernel, 'read')).toBe(true);
      expect(isEffectAllowedAtRing(Ring.Kernel, 'write')).toBe(false);
    });

    it('should check allowed effects for service', () => {
      expect(isEffectAllowedAtRing(Ring.Service, 'read')).toBe(true);
      expect(isEffectAllowedAtRing(Ring.Service, 'write')).toBe(true);
      expect(isEffectAllowedAtRing(Ring.Service, 'network')).toBe(true);
    });

    it('should check allowed effects for user', () => {
      expect(isEffectAllowedAtRing(Ring.User, 'spawn')).toBe(true);
    });

    it('should check allowed effects for ephemeral', () => {
      expect(isEffectAllowedAtRing(Ring.Ephemeral, 'read')).toBe(true);
      expect(isEffectAllowedAtRing(Ring.Ephemeral, 'write')).toBe(true);
      expect(isEffectAllowedAtRing(Ring.Ephemeral, 'network')).toBe(false);
    });
  });

  describe('getRequiredApproval()', () => {
    it('should return correct approval for each ring', () => {
      expect(getRequiredApproval(Ring.Kernel)).toBe('formal_proof');
      expect(getRequiredApproval(Ring.Service)).toBe('omega');
      expect(getRequiredApproval(Ring.User)).toBe('witness');
      expect(getRequiredApproval(Ring.Ephemeral)).toBe('none');
    });
  });

  describe('createRingContext()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
    });

    it('should create ring context', () => {
      const ctx = createRingContext(Ring.User, 'agent-1', 'session-123');

      expect(ctx.current).toBe(Ring.User);
      expect(ctx.agent).toBe('agent-1');
      expect(ctx.sessionId).toBe('session-123');
      expect(ctx.enteredAt).toBe(Date.now());
      expect(ctx.parent).toBeUndefined();
    });

    vi.useRealTimers();
  });

  describe('descendRing()', () => {
    beforeEach(() => {
      vi.useFakeTimers();
      vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
    });

    it('should descend to less privileged ring', () => {
      const parent = createRingContext(Ring.Service, 'agent-1', 'session-123');
      const child = descendRing(parent, Ring.User);

      expect(child).not.toBeNull();
      expect(child!.current).toBe(Ring.User);
      expect(child!.parent).toBe(parent);
    });

    it('should not allow ascending (to more privileged ring)', () => {
      const ctx = createRingContext(Ring.User, 'agent-1', 'session-123');
      const result = descendRing(ctx, Ring.Service);

      expect(result).toBeNull();
    });

    it('should not allow staying at same ring', () => {
      const ctx = createRingContext(Ring.User, 'agent-1', 'session-123');
      const result = descendRing(ctx, Ring.User);

      expect(result).toBeNull();
    });

    vi.useRealTimers();
  });

  describe('requestEscalation()', () => {
    it('should allow escalation to more privileged ring', () => {
      const ctx = createRingContext(Ring.User, 'agent-1', 'session-123');
      const result = requestEscalation(ctx, Ring.Service);

      expect(result.allowed).toBe(true);
      expect(result.requiresApproval).toBe('omega');
    });

    it('should return formal_proof for kernel escalation', () => {
      const ctx = createRingContext(Ring.Service, 'agent-1', 'session-123');
      const result = requestEscalation(ctx, Ring.Kernel);

      expect(result.allowed).toBe(true);
      expect(result.requiresApproval).toBe('formal_proof');
    });

    it('should not allow descending via escalation', () => {
      const ctx = createRingContext(Ring.Service, 'agent-1', 'session-123');
      const result = requestEscalation(ctx, Ring.User);

      expect(result.allowed).toBe(false);
    });

    it('should not allow same ring escalation', () => {
      const ctx = createRingContext(Ring.User, 'agent-1', 'session-123');
      const result = requestEscalation(ctx, Ring.User);

      expect(result.allowed).toBe(false);
    });
  });
});
