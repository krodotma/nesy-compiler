import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { DatasetPipeline } from '../dataset-pipeline.js';
import type { TrainingPair } from '../holon-exporter.js';
import type { NegativeSample } from '../negative-sampler.js';
import type { EditTriplet } from '../training-loop.js';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import * as os from 'node:os';

describe('DatasetPipeline', () => {
  let tmpDir: string;

  beforeEach(async () => {
    tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), 'dataset-pipeline-test-'));
  });

  afterEach(async () => {
    await fs.rm(tmpDir, { recursive: true, force: true });
  });

  it('initializes with default config', () => {
    const pipeline = new DatasetPipeline();
    expect(pipeline).toBeDefined();
  });

  it('initializes with custom config', () => {
    const pipeline = new DatasetPipeline({
      outputDir: tmpDir,
      splitRatio: { train: 0.7, val: 0.15, test: 0.15 },
    });
    expect(pipeline).toBeDefined();
  });

  it('addPositive/addNegative/addTriplet update stats', () => {
    const pipeline = new DatasetPipeline({ outputDir: tmpDir });

    const positive: TrainingPair = {
      context: 'Test context',
      code: 'const x = 1;',
      metadata: { holonId: 'h1', ring: 0, confidence: 0.9, path: '/test.ts' },
    };

    const negative: NegativeSample = {
      badCode: 'var x = undefined',
      reason: 'Uses var instead of const',
      antipatternType: 'Style Inconsistency',
    };

    const triplet: EditTriplet = {
      preState: 'const x = 1;',
      diff: '-const x = 1;\n+const x = 2;',
      postState: 'const x = 2;',
      isGood: true,
      source: 'test',
    };

    pipeline.addPositive(positive);
    pipeline.addNegative(negative);
    pipeline.addTriplet(triplet);

    const stats = pipeline.getStats();
    expect(stats.positives).toBe(1);
    expect(stats.negatives).toBe(1);
    expect(stats.triplets).toBe(1);
  });

  it('extractPositives returns empty when no holons in FalkorDB', async () => {
    const pipeline = new DatasetPipeline({ outputDir: tmpDir });
    const positives = await pipeline.extractPositives();
    expect(positives).toEqual([]);
  });

  it('extractNegatives returns empty when no rejection log exists', async () => {
    const pipeline = new DatasetPipeline({ outputDir: tmpDir });
    const negatives = await pipeline.extractNegatives();
    expect(negatives).toEqual([]);
  });

  it('extractTriplets returns empty when no triplet file exists', async () => {
    const pipeline = new DatasetPipeline({ outputDir: tmpDir });
    const triplets = await pipeline.extractTriplets();
    expect(triplets).toEqual([]);
  });

  it('buildDataset combines all sources and splits correctly', async () => {
    const pipeline = new DatasetPipeline({
      outputDir: tmpDir,
      splitRatio: { train: 0.6, val: 0.2, test: 0.2 },
    });

    for (let i = 0; i < 10; i++) {
      pipeline.addPositive({
        context: `Context ${i}`,
        code: `const x${i} = ${i};`,
        metadata: { holonId: `h${i}`, ring: 0, confidence: 0.9, path: `/test${i}.ts` },
      });
    }

    const dataset = await pipeline.buildDataset();

    expect(dataset.train).toBeDefined();
    expect(dataset.val).toBeDefined();
    expect(dataset.test).toBeDefined();
    expect(dataset.train.length + dataset.val.length + dataset.test.length).toBe(10);
  });

  it('saveToJSONL writes data correctly', async () => {
    const pipeline = new DatasetPipeline({ outputDir: tmpDir });

    const data = [
      { id: 1, value: 'test1' },
      { id: 2, value: 'test2' },
    ];

    await pipeline.saveToJSONL(data, 'output.jsonl');

    const outputPath = path.join(tmpDir, 'output.jsonl');
    const content = await fs.readFile(outputPath, 'utf-8');
    const lines = content.trim().split('\n');

    expect(lines).toHaveLength(2);
    expect(JSON.parse(lines[0])).toEqual({ id: 1, value: 'test1' });
    expect(JSON.parse(lines[1])).toEqual({ id: 2, value: 'test2' });
  });

  describe('integration', () => {
    it('full pipeline: add data, build dataset, save to JSONL', async () => {
      const pipeline = new DatasetPipeline({
        outputDir: tmpDir,
        splitRatio: { train: 0.8, val: 0.1, test: 0.1 },
      });

      for (let i = 0; i < 20; i++) {
        pipeline.addPositive({
          context: `Context ${i}`,
          code: `function f${i}() { return ${i}; }`,
          metadata: { holonId: `h${i}`, ring: i % 2 as 0 | 1, confidence: 0.85 + i * 0.01, path: `/src/f${i}.ts` },
        });
      }

      for (let i = 0; i < 5; i++) {
        pipeline.addNegative({
          badCode: `var broken${i} = undefined`,
          reason: `Test reason ${i}`,
          antipatternType: 'Dead Code',
        });
      }

      for (let i = 0; i < 3; i++) {
        pipeline.addTriplet({
          preState: `const v = ${i};`,
          diff: `-const v = ${i};\n+const v = ${i + 1};`,
          postState: `const v = ${i + 1};`,
          isGood: true,
          source: 'integration-test',
        });
      }

      const dataset = await pipeline.buildDataset();
      const totalEntries = 20 + 5 + 3;

      expect(dataset.train.length + dataset.val.length + dataset.test.length).toBe(totalEntries);
      expect(dataset.train.length).toBeGreaterThan(0);

      await pipeline.saveToJSONL(dataset.train, 'train.jsonl');
      await pipeline.saveToJSONL(dataset.val, 'val.jsonl');
      await pipeline.saveToJSONL(dataset.test, 'test.jsonl');

      const trainContent = await fs.readFile(path.join(tmpDir, 'train.jsonl'), 'utf-8');
      expect(trainContent.trim().split('\n').length).toBe(dataset.train.length);
    });
  });
});
