import { describe, it, expect } from 'vitest';
import { NegativeSampler } from '../negative-sampler.js';

describe('NegativeSampler', () => {
  describe('constructor', () => {
    it('uses default config when no config provided', () => {
      const sampler = new NegativeSampler();
      expect(sampler).toBeInstanceOf(NegativeSampler);
    });

    it('merges partial config with defaults', () => {
      const sampler = new NegativeSampler({ maxSamples: 50 });
      expect(sampler).toBeInstanceOf(NegativeSampler);
    });
  });

  describe('sampleFromRejections', () => {
    const rejectionLog = `
REJECTION: Dead Code
REASON: Contains unused variables and unreachable code
\`\`\`
const unused = 42;
function neverCalled() { return null; }
\`\`\`
CORRECTION: Remove unused code

REJECTION: Phantom Import
REASON: Import for non-existent module
\`\`\`
import { fake } from './does-not-exist';
\`\`\`

REJECTION: Type Mismatch
REASON: Incompatible types
\`\`\`
const x: string = 123;
\`\`\`
    `.trim();

    it('parses rejection log and returns samples', async () => {
      const sampler = new NegativeSampler();
      const samples = await sampler.sampleFromRejections(rejectionLog);

      expect(samples).toHaveLength(3);
      expect(samples[0].antipatternType).toBe('Dead Code');
      expect(samples[0].reason).toBe('Contains unused variables and unreachable code');
      expect(samples[0].badCode).toContain('const unused = 42');
    });

    it('filters by antipattern types', async () => {
      const sampler = new NegativeSampler({
        antipatternTypes: ['Dead Code', 'Type Mismatch'],
      });
      const samples = await sampler.sampleFromRejections(rejectionLog);

      expect(samples).toHaveLength(2);
      expect(samples.map((s) => s.antipatternType)).toEqual([
        'Dead Code',
        'Type Mismatch',
      ]);
    });

    it('respects maxSamples limit', async () => {
      const sampler = new NegativeSampler({ maxSamples: 2 });
      const samples = await sampler.sampleFromRejections(rejectionLog);

      expect(samples).toHaveLength(2);
    });

    it('includes corrections when configured', async () => {
      const sampler = new NegativeSampler({ includeCorrections: true });
      const samples = await sampler.sampleFromRejections(rejectionLog);

      expect(samples[0].correction).toBe('Remove unused code');
    });

    it('excludes corrections when configured', async () => {
      const sampler = new NegativeSampler({ includeCorrections: false });
      const samples = await sampler.sampleFromRejections(rejectionLog);

      expect(samples[0].correction).toBeUndefined();
    });
  });

  describe('generateSyntheticBadExample', () => {
    const goodCode = `const result: string = "hello";`;

    it('generates synthetic Dead Code example', () => {
      const sampler = new NegativeSampler();
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Dead Code');

      expect(sample.antipatternType).toBe('Dead Code');
      expect(sample.badCode).toContain('unusedVar');
      expect(sample.badCode).toContain('deadFunction');
      expect(sample.reason).toContain('unreachable');
    });

    it('generates synthetic Phantom Import example', () => {
      const sampler = new NegativeSampler();
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Phantom Import');

      expect(sample.antipatternType).toBe('Phantom Import');
      expect(sample.badCode).toContain('fake-module');
      expect(sample.reason).toContain('not exist');
    });

    it('generates synthetic Deep Nesting example', () => {
      const sampler = new NegativeSampler();
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Deep Nesting');

      expect(sample.antipatternType).toBe('Deep Nesting');
      expect(sample.badCode.match(/if \(true\)/g)?.length).toBeGreaterThanOrEqual(5);
    });

    it('generates synthetic Hallucinated Import example', () => {
      const sampler = new NegativeSampler();
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Hallucinated Import');

      expect(sample.antipatternType).toBe('Hallucinated Import');
      expect(sample.badCode).toContain('AIGeneratedMagic');
      expect(sample.badCode).toContain('hallucinated-library');
    });

    it('generates fallback for unknown antipattern', () => {
      const sampler = new NegativeSampler();
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Unknown Pattern');

      expect(sample.antipatternType).toBe('Unknown Pattern');
      expect(sample.badCode).toContain('INTENTIONALLY BAD');
      expect(sample.correction).toBe(goodCode);
    });

    it('includes correction when configured', () => {
      const sampler = new NegativeSampler({ includeCorrections: true });
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Dead Code');

      expect(sample.correction).toBe(goodCode);
    });

    it('excludes correction when configured', () => {
      const sampler = new NegativeSampler({ includeCorrections: false });
      const sample = sampler.generateSyntheticBadExample(goodCode, 'Dead Code');

      expect(sample.correction).toBeUndefined();
    });
  });

  describe('formatAsTrainingNegative', () => {
    it('formats sample as training negative', () => {
      const sampler = new NegativeSampler();
      const sample = {
        badCode: 'const unused = 42;',
        reason: 'Unused variable',
        antipatternType: 'Dead Code',
        correction: 'Remove the variable',
      };

      const formatted = sampler.formatAsTrainingNegative(sample);

      expect(formatted.label).toBe('bad');
      expect(formatted.reason).toBe('Unused variable');
      expect(formatted.context).toContain('[Antipattern: Dead Code]');
      expect(formatted.context).toContain('const unused = 42');
      expect(formatted.context).toContain('Correction available');
    });

    it('handles sample without correction', () => {
      const sampler = new NegativeSampler();
      const sample = {
        badCode: 'const unused = 42;',
        reason: 'Unused variable',
        antipatternType: 'Dead Code',
      };

      const formatted = sampler.formatAsTrainingNegative(sample);

      expect(formatted.label).toBe('bad');
      expect(formatted.context).not.toContain('Correction available');
    });
  });
});
