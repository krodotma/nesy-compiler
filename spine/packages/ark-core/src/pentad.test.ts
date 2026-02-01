/**
 * @ark/core/pentad - Pentad (Five Coordinates) Tests
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  createPentad,
  validatePentad,
  type Pentad,
  type Etymon,
  type Locus,
  type Lineage,
  type Kairos,
  type Artifact,
  type Budget,
} from './pentad.js';

describe('Pentad', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
  });

  describe('createPentad()', () => {
    it('should create pentad with all coordinates', () => {
      const pentad = createPentad(
        'Fix authentication bug',
        '/src/auth/login.ts',
        'claude-opus-4.5',
        { type: 'code-change', lines: [42, 67] }
      );

      // Check etymon (WHY)
      expect(pentad.etymon.justification).toBe('Fix authentication bug');
      expect(pentad.etymon.name).toBe('Fix authentication bug');
      expect(pentad.etymon.id).toMatch(/^[0-9a-f-]{36}$/);

      // Check locus (WHERE)
      expect(pentad.locus.path).toBe('/src/auth/login.ts');

      // Check lineage (WHO)
      expect(pentad.lineage.agent).toBe('claude-opus-4.5');
      expect(pentad.lineage.authority).toBe('service');

      // Check kairos (WHEN)
      expect(pentad.kairos.timestamp).toBe(Date.now());
      expect(pentad.kairos.protocolVersion).toBe('DKIN v29');

      // Check artifact (WHAT)
      expect(pentad.artifact.type).toBe('structured');
      expect(pentad.artifact.data).toEqual({ type: 'code-change', lines: [42, 67] });
    });

    it('should truncate long justifications for name', () => {
      const longJustification = 'A'.repeat(100);
      const pentad = createPentad(longJustification, '/path', 'agent', {});

      expect(pentad.etymon.name).toBe('A'.repeat(50));
      expect(pentad.etymon.justification).toBe(longJustification);
    });
  });

  describe('validatePentad()', () => {
    it('should validate complete pentad', () => {
      const pentad = createPentad(
        'Valid justification',
        '/valid/path',
        'valid-agent',
        { data: true }
      );

      const result = validatePentad(pentad);

      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('should detect missing justification (WHY)', () => {
      const pentad: Pentad = {
        etymon: {
          id: 'e1',
          name: 'name',
          justification: '', // Empty
        },
        locus: { path: '/path' },
        lineage: { agent: 'agent', authority: 'service' },
        kairos: { timestamp: Date.now(), budget: {} },
        artifact: { type: 'structured', data: {} },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Etymon must have justification (WHY)');
    });

    it('should detect missing path (WHERE)', () => {
      const pentad: Pentad = {
        etymon: { id: 'e1', name: 'name', justification: 'why' },
        locus: { path: '' },
        lineage: { agent: 'agent', authority: 'service' },
        kairos: { timestamp: Date.now(), budget: {} },
        artifact: { type: 'structured', data: {} },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Locus must have path (WHERE)');
    });

    it('should detect missing agent (WHO)', () => {
      const pentad: Pentad = {
        etymon: { id: 'e1', name: 'name', justification: 'why' },
        locus: { path: '/path' },
        lineage: { agent: '', authority: 'service' },
        kairos: { timestamp: Date.now(), budget: {} },
        artifact: { type: 'structured', data: {} },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Lineage must have agent (WHO)');
    });

    it('should detect missing timestamp (WHEN)', () => {
      const pentad: Pentad = {
        etymon: { id: 'e1', name: 'name', justification: 'why' },
        locus: { path: '/path' },
        lineage: { agent: 'agent', authority: 'service' },
        kairos: { timestamp: 0, budget: {} }, // 0 is falsy
        artifact: { type: 'structured', data: {} },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Kairos must have timestamp (WHEN)');
    });

    it('should detect missing data (WHAT)', () => {
      const pentad: Pentad = {
        etymon: { id: 'e1', name: 'name', justification: 'why' },
        locus: { path: '/path' },
        lineage: { agent: 'agent', authority: 'service' },
        kairos: { timestamp: Date.now(), budget: {} },
        artifact: { type: 'structured', data: undefined as unknown as unknown },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(false);
      expect(result.errors).toContain('Artifact must have data (WHAT)');
    });

    it('should allow null as valid data', () => {
      const pentad: Pentad = {
        etymon: { id: 'e1', name: 'name', justification: 'why' },
        locus: { path: '/path' },
        lineage: { agent: 'agent', authority: 'service' },
        kairos: { timestamp: Date.now(), budget: {} },
        artifact: { type: 'structured', data: null },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(true);
    });

    it('should collect multiple errors', () => {
      const pentad: Pentad = {
        etymon: { id: 'e1', name: 'name', justification: '' },
        locus: { path: '' },
        lineage: { agent: '', authority: 'service' },
        kairos: { timestamp: 0, budget: {} },
        artifact: { type: 'structured', data: undefined as unknown as unknown },
      };

      const result = validatePentad(pentad);

      expect(result.valid).toBe(false);
      expect(result.errors).toHaveLength(5);
    });
  });

  describe('Type definitions', () => {
    it('should support all authority levels', () => {
      const levels = ['kernel', 'service', 'user', 'ephemeral'];
      const lineage: Lineage = {
        agent: 'test',
        authority: 'kernel',
      };
      expect(levels).toContain(lineage.authority);
    });

    it('should support all artifact types', () => {
      const types = ['code', 'document', 'event', 'binary', 'structured'];
      const artifact: Artifact = {
        type: 'code',
        data: 'console.log("hello")',
      };
      expect(types).toContain(artifact.type);
    });

    it('should support optional budget fields', () => {
      const budget: Budget = {
        tokens: 10000,
        timeMs: 60000,
        toolCalls: 50,
        fileOps: 100,
      };
      expect(budget.tokens).toBe(10000);
    });

    it('should support optional locus fields', () => {
      const locus: Locus = {
        path: '/src/file.ts',
        hash: 'sha256:abc123',
        lineRange: [10, 20],
        rhizomeId: 'rh-123',
      };
      expect(locus.lineRange).toEqual([10, 20]);
    });
  });

  vi.useRealTimers();
});
