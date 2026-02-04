import { describe, it, expect } from 'vitest';
import { ContextSerializer, type GraphNeighborhood } from '../context-serializer.js';

describe('ContextSerializer', () => {
  describe('constructor', () => {
    it('uses default maxTokens of 4096', () => {
      const serializer = new ContextSerializer();
      const text = 'a'.repeat(4096 * 4);
      expect(serializer.estimateTokens(text)).toBe(4096);
    });

    it('accepts custom maxTokens', () => {
      const serializer = new ContextSerializer(2048);
      const neighborhood: GraphNeighborhood = {
        imports: [],
        types: [],
        dependencies: [],
        callers: [],
        callees: [],
      };
      const result = serializer.serializeNeighborhood(neighborhood);
      expect(result.truncated).toBe(false);
    });
  });

  describe('serializeNeighborhood', () => {
    it('serializes empty neighborhood', () => {
      const serializer = new ContextSerializer();
      const neighborhood: GraphNeighborhood = {
        imports: [],
        types: [],
        dependencies: [],
        callers: [],
        callees: [],
      };
      const result = serializer.serializeNeighborhood(neighborhood);
      expect(result.promptText).toBe('');
      expect(result.tokenCount).toBe(0);
      expect(result.truncated).toBe(false);
    });

    it('serializes full neighborhood with all sections', () => {
      const serializer = new ContextSerializer();
      const neighborhood: GraphNeighborhood = {
        imports: ['lodash', 'react'],
        types: ['User', 'Product'],
        dependencies: ['database', 'cache'],
        callers: ['main()', 'handler()'],
        callees: ['validate()', 'save()'],
      };
      const result = serializer.serializeNeighborhood(neighborhood);

      expect(result.promptText).toContain('## Imports');
      expect(result.promptText).toContain('`lodash`');
      expect(result.promptText).toContain('## Types');
      expect(result.promptText).toContain('`User`');
      expect(result.promptText).toContain('## Dependencies');
      expect(result.promptText).toContain('## Callers');
      expect(result.promptText).toContain('## Callees');
      expect(result.truncated).toBe(false);
    });

    it('truncates when exceeding maxTokens', () => {
      const serializer = new ContextSerializer(10);
      const neighborhood: GraphNeighborhood = {
        imports: Array(100).fill('very-long-import-name'),
        types: Array(100).fill('VeryLongTypeName'),
        dependencies: [],
        callers: [],
        callees: [],
      };
      const result = serializer.serializeNeighborhood(neighborhood);

      expect(result.truncated).toBe(true);
      expect(result.promptText).toContain('[truncated]');
    });
  });

  describe('formatImports', () => {
    it('formats empty imports', () => {
      const serializer = new ContextSerializer();
      expect(serializer.formatImports([])).toBe('');
    });

    it('formats imports with backticks', () => {
      const serializer = new ContextSerializer();
      const result = serializer.formatImports(['fs', 'path']);
      expect(result).toBe('## Imports\n- `fs`\n- `path`');
    });
  });

  describe('formatTypes', () => {
    it('formats empty types', () => {
      const serializer = new ContextSerializer();
      expect(serializer.formatTypes([])).toBe('');
    });

    it('formats types with backticks', () => {
      const serializer = new ContextSerializer();
      const result = serializer.formatTypes(['User', 'Order']);
      expect(result).toBe('## Types\n- `User`\n- `Order`');
    });
  });

  describe('estimateTokens', () => {
    it('estimates ~4 chars per token', () => {
      const serializer = new ContextSerializer();
      expect(serializer.estimateTokens('abcd')).toBe(1);
      expect(serializer.estimateTokens('abcdefgh')).toBe(2);
      expect(serializer.estimateTokens('abc')).toBe(1);
    });

    it('handles empty string', () => {
      const serializer = new ContextSerializer();
      expect(serializer.estimateTokens('')).toBe(0);
    });
  });
});
