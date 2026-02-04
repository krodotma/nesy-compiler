import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { HolonExporter, type Holon, type ExportConfig, type TrainingPair } from '../holon-exporter.js';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';

describe('HolonExporter', () => {
  let exporter: HolonExporter;
  let tempDir: string;

  beforeEach(async () => {
    exporter = new HolonExporter(6379);
    tempDir = await fs.mkdtemp(path.join(os.tmpdir(), 'holon-exporter-test-'));
  });

  afterEach(async () => {
    await fs.rm(tempDir, { recursive: true, force: true });
  });

  describe('constructor', () => {
    it('creates exporter with default port', () => {
      const defaultExporter = new HolonExporter();
      expect(defaultExporter).toBeInstanceOf(HolonExporter);
    });

    it('creates exporter with custom port', () => {
      const customExporter = new HolonExporter(6380);
      expect(customExporter).toBeInstanceOf(HolonExporter);
    });
  });

  describe('formatAsTrainingPair', () => {
    it('formats a holon into a training pair', () => {
      const holon: Holon = {
        id: 'holon-001',
        name: 'TestModule',
        path: '/src/test.ts',
        ring: 0,
        content: 'export function test() { return 42; }',
        confidence: 0.95,
        symbols: ['test'],
        language: 'typescript',
        semantic_cluster: 'utilities',
      };

      const pair = exporter.formatAsTrainingPair(holon);

      expect(pair.context).toContain('File: /src/test.ts');
      expect(pair.context).toContain('Name: TestModule');
      expect(pair.context).toContain('Language: typescript');
      expect(pair.context).toContain('Symbols: test');
      expect(pair.code).toBe('export function test() { return 42; }');
      expect(pair.metadata.holonId).toBe('holon-001');
      expect(pair.metadata.ring).toBe(0);
      expect(pair.metadata.confidence).toBe(0.95);
    });

    it('handles holon with minimal properties', () => {
      const holon: Holon = {
        id: 'holon-002',
        name: 'Minimal',
        path: '/src/minimal.ts',
        ring: 1,
      };

      const pair = exporter.formatAsTrainingPair(holon);

      expect(pair.context).toContain('File: /src/minimal.ts');
      expect(pair.code).toBe('');
      expect(pair.metadata.confidence).toBe(1.0);
    });

    it('uses stability_score as fallback for confidence', () => {
      const holon: Holon = {
        id: 'holon-003',
        name: 'WithStability',
        path: '/src/stable.ts',
        ring: 0,
        stability_score: 0.88,
      };

      const pair = exporter.formatAsTrainingPair(holon);
      expect(pair.metadata.confidence).toBe(0.88);
    });
  });

  describe('exportToJSONL', () => {
    it('exports holons to JSONL file', async () => {
      const holons: Holon[] = [
        {
          id: 'h1',
          name: 'First',
          path: '/src/first.ts',
          ring: 0,
          content: 'const a = 1;',
          confidence: 0.9,
        },
        {
          id: 'h2',
          name: 'Second',
          path: '/src/second.ts',
          ring: 1,
          content: 'const b = 2;',
          confidence: 0.85,
        },
      ];

      const outputPath = path.join(tempDir, 'training.jsonl');
      const count = await exporter.exportToJSONL(holons, outputPath);

      expect(count).toBe(2);

      const content = await fs.readFile(outputPath, 'utf-8');
      const lines = content.trim().split('\n');
      expect(lines).toHaveLength(2);

      const first = JSON.parse(lines[0]) as TrainingPair;
      expect(first.metadata.holonId).toBe('h1');
      expect(first.code).toBe('const a = 1;');

      const second = JSON.parse(lines[1]) as TrainingPair;
      expect(second.metadata.holonId).toBe('h2');
    });

    it('creates nested directories if needed', async () => {
      const outputPath = path.join(tempDir, 'nested', 'deep', 'training.jsonl');
      const count = await exporter.exportToJSONL([], outputPath);

      expect(count).toBe(0);
      const exists = await fs.access(path.dirname(outputPath)).then(() => true).catch(() => false);
      expect(exists).toBe(true);
    });

    it('handles empty holon array', async () => {
      const outputPath = path.join(tempDir, 'empty.jsonl');
      const count = await exporter.exportToJSONL([], outputPath);

      expect(count).toBe(0);
      const content = await fs.readFile(outputPath, 'utf-8');
      expect(content).toBe('');
    });
  });

  describe('queryGoldHolons', () => {
    it('returns empty array (requires FalkorDB connection)', async () => {
      const config: ExportConfig = {
        ringLevels: [0, 1],
        minConfidence: 0.8,
        outputPath: '/tmp/out.jsonl',
      };

      const holons = await exporter.queryGoldHolons(config);
      expect(holons).toEqual([]);
    });
  });
});
