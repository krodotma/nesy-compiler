/**
 * @ark/core/holon - Holon (Lock & Key) Tests
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  validateHolon,
  createHolon,
  deriveHolon,
  canExecute,
  serializeHolon,
  deserializeHolon,
  type Holon,
  type HolonStatus,
} from './holon.js';
import { DEFAULT_SEXTET } from './sextet.js';
import { Ring } from './ring.js';

describe('Holon', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
  });

  describe('createHolon()', () => {
    it('should create valid holon with defaults', () => {
      const holon = createHolon(
        'Fix authentication bug',
        '/src/auth/login.ts',
        'claude-opus-4.5',
        { diff: '+const fix = true;' }
      );

      expect(holon.id).toMatch(/^[0-9a-f-]{36}$/);
      expect(holon.pentad.etymon.justification).toBe('Fix authentication bug');
      expect(holon.pentad.locus.path).toBe('/src/auth/login.ts');
      expect(holon.pentad.lineage.agent).toBe('claude-opus-4.5');
      expect(holon.pentad.artifact.data).toEqual({ diff: '+const fix = true;' });
      expect(holon.status).toBe('valid');
      expect(holon.createdAt).toBe(Date.now());
      expect(holon.validatedAt).toBeDefined();
    });

    it('should apply sextet overrides', () => {
      const holon = createHolon(
        'Test action',
        '/path',
        'agent',
        {},
        {
          effects: {
            allowed: { network: true },
            denied: {},
            tier: 'medium',
          },
        }
      );

      expect(holon.sextet.effects.tier).toBe('medium');
      expect(holon.sextet.effects.allowed.network).toBe(true);
    });

    it('should validate on creation', () => {
      const holon = createHolon(
        'Valid justification',
        '/valid/path',
        'valid-agent',
        { data: true }
      );

      expect(holon.status).toBe('valid');
    });
  });

  describe('validateHolon()', () => {
    it('should validate complete holon', () => {
      const holon = createHolon(
        'Test action',
        '/path',
        'agent',
        { data: true }
      );

      const result = validateHolon(holon);

      expect(result.valid).toBe(true);
      expect(result.status).toBe('valid');
      expect(result.pentadErrors).toHaveLength(0);
      expect(result.sextetBlocks).toHaveLength(0);
    });

    it('should detect expired holon', () => {
      const holon = createHolon(
        'Test action',
        '/path',
        'agent',
        { data: true }
      );

      // Set TTL that has already expired
      holon.pentad.kairos.ttl = 60; // 60 seconds
      holon.pentad.kairos.timestamp = Date.now() - 120000; // 2 minutes ago

      const result = validateHolon(holon);

      expect(result.valid).toBe(false);
      expect(result.status).toBe('expired');
      expect(result.sextetBlocks).toContain('TTL exceeded');
    });

    it('should detect incomplete pentad', () => {
      const holon: Holon = {
        id: 'test-id',
        pentad: {
          etymon: { id: 'e1', name: '', justification: '' }, // Missing
          locus: { path: '/path' },
          lineage: { agent: 'agent', authority: 'service' },
          kairos: { timestamp: Date.now(), budget: {} },
          artifact: { type: 'structured', data: {} },
        },
        sextet: DEFAULT_SEXTET,
        status: 'valid',
        createdAt: Date.now(),
      };

      const result = validateHolon(holon);

      expect(result.valid).toBe(false);
      expect(result.status).toBe('incomplete');
      expect(result.pentadErrors.length).toBeGreaterThan(0);
    });

    it('should detect blocked by sextet', () => {
      const holon = createHolon(
        'Test action',
        '/path',
        'agent',
        { data: true },
        {
          provenance: {
            appendOnly: false, // Violates P-Gate
            nonDpi: true,
            replayable: true,
          },
        }
      );

      const result = validateHolon(holon);

      expect(result.valid).toBe(false);
      expect(result.status).toBe('blocked');
      expect(result.sextetBlocks.length).toBeGreaterThan(0);
    });

    it('should include validation timestamp', () => {
      const holon = createHolon('Test', '/path', 'agent', {});
      const result = validateHolon(holon);

      expect(result.validatedAt).toBe(Date.now());
    });
  });

  describe('deriveHolon()', () => {
    it('should create child holon from parent', () => {
      const parent = createHolon(
        'Parent action',
        '/parent/path',
        'parent-agent',
        { parentData: true }
      );

      const child = deriveHolon(
        parent,
        'Child action derived from parent',
        '/child/path',
        { childData: true }
      );

      expect(child.parentId).toBe(parent.id);
      expect(child.pentad.lineage.parent).toBe('parent-agent');
      expect(child.pentad.etymon.justification).toBe('Child action derived from parent');
      expect(child.pentad.locus.path).toBe('/child/path');
    });

    it('should inherit provenance chain', () => {
      const parent = createHolon(
        'Parent action',
        '/parent/path',
        'parent-agent',
        { parentData: true }
      );

      const child = deriveHolon(parent, 'Child', '/child', {});

      expect(child.sextet.provenance.evidenceHash).toBe(parent.id);
    });

    it('should inherit effect restrictions', () => {
      const parent = createHolon(
        'Parent action',
        '/parent/path',
        'parent-agent',
        { parentData: true },
        {
          effects: {
            allowed: { file: true },
            denied: { network: true },
            tier: 'low',
          },
        }
      );

      const child = deriveHolon(parent, 'Child', '/child', {});

      expect(child.sextet.effects.denied.network).toBe(true);
    });
  });

  describe('canExecute()', () => {
    it('should allow valid holon', () => {
      const holon = createHolon(
        'Test action',
        '/path',
        'agent',
        { data: true }
      );

      const result = canExecute(holon);

      expect(result.allowed).toBe(true);
      expect(result.reason).toBeUndefined();
    });

    it('should reject expired holon', () => {
      const holon = createHolon('Test', '/path', 'agent', {});
      holon.pentad.kairos.ttl = 60;
      holon.pentad.kairos.timestamp = Date.now() - 120000;

      const result = canExecute(holon);

      expect(result.allowed).toBe(false);
      expect(result.reason).toBe('Holon has expired');
    });

    it('should reject incomplete holon', () => {
      const holon: Holon = {
        id: 'test-id',
        pentad: {
          etymon: { id: 'e1', name: '', justification: '' },
          locus: { path: '/path' },
          lineage: { agent: 'agent', authority: 'service' },
          kairos: { timestamp: Date.now(), budget: {} },
          artifact: { type: 'structured', data: {} },
        },
        sextet: DEFAULT_SEXTET,
        status: 'valid',
        createdAt: Date.now(),
      };

      const result = canExecute(holon);

      expect(result.allowed).toBe(false);
      expect(result.reason).toContain('Missing:');
    });

    it('should reject blocked holon', () => {
      const holon = createHolon('Test', '/path', 'agent', {}, {
        provenance: { appendOnly: false, nonDpi: true, replayable: true },
      });

      const result = canExecute(holon);

      expect(result.allowed).toBe(false);
      expect(result.reason).toContain('Blocked by:');
    });

    it('should reject omega veto', () => {
      const holon = createHolon('Test', '/path', 'agent', {}, {
        // Create holon with all gates passing except omega
        recovery: {
          canary: false,
          shadow: false,
          verified: true, // Make R-Gate pass
          checkpointAvailable: true,
          strategy: 'rollback',
        },
        omega: {
          vetoable: true,
          aligned: false, // But omega is misaligned
        },
      });

      const result = canExecute(holon);

      expect(result.allowed).toBe(false);
      // The omega misalignment is caught during validation and shows as Omega-Gate block
      expect(result.reason).toContain('Omega-Gate');
    });

    it('should reject exhausted time budget', () => {
      const holon = createHolon('Test', '/path', 'agent', {}, {
        // Create holon with all gates passing
        recovery: {
          canary: false,
          shadow: false,
          verified: true,
          checkpointAvailable: true,
          strategy: 'rollback',
        },
      });
      holon.pentad.kairos.budget.timeMs = 0;

      const result = canExecute(holon);

      expect(result.allowed).toBe(false);
      expect(result.reason).toBe('Time budget exhausted');
    });
  });

  describe('serializeHolon()', () => {
    it('should serialize holon to JSON string', () => {
      const holon = createHolon('Test', '/path', 'agent', { value: 42 });
      const json = serializeHolon(holon);

      expect(typeof json).toBe('string');
      expect(json).toContain('"justification":"Test"');
      expect(json).toContain('"value":42');
    });
  });

  describe('deserializeHolon()', () => {
    it('should deserialize JSON string to holon', () => {
      const original = createHolon('Test', '/path', 'agent', { value: 42 });
      const json = serializeHolon(original);
      const restored = deserializeHolon(json);

      expect(restored.id).toBe(original.id);
      expect(restored.pentad.etymon.justification).toBe('Test');
      expect(restored.pentad.artifact.data).toEqual({ value: 42 });
    });

    it('should roundtrip correctly', () => {
      const holon = createHolon(
        'Complex holon',
        '/src/file.ts',
        'claude-opus-4.5',
        { complex: { nested: { data: [1, 2, 3] } } }
      );

      const roundtripped = deserializeHolon(serializeHolon(holon));

      expect(roundtripped).toEqual(holon);
    });
  });

  describe('HolonStatus', () => {
    it('should support all status values', () => {
      const statuses: HolonStatus[] = ['valid', 'blocked', 'incomplete', 'expired'];
      expect(statuses).toHaveLength(4);
    });
  });

  vi.useRealTimers();
});
