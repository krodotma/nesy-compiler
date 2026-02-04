/**
 * DatasetPipeline - Orchestrate Full Training Data Extraction Pipeline
 *
 * Steps 56-57 of NeSy Evolution: Combines HolonExporter, ContextSerializer,
 * EditTripletExtractor (from TrainingLoop), and NegativeSampler into a
 * unified dataset generation pipeline.
 */

import { HolonExporter, type TrainingPair, type Holon } from './holon-exporter.js';
import { ContextSerializer, type GraphNeighborhood } from './context-serializer.js';
import { NegativeSampler, type NegativeSample } from './negative-sampler.js';
import type { EditTriplet } from './training-loop.js';

export interface DatasetConfig {
  outputDir: string;
  splitRatio: {
    train: number;
    val: number;
    test: number;
  };
}

export interface DatasetSplit {
  train: DatasetEntry[];
  val: DatasetEntry[];
  test: DatasetEntry[];
}

export interface DatasetEntry {
  type: 'positive' | 'negative' | 'triplet';
  data: TrainingPair | NegativeSample | EditTriplet;
}

const DEFAULT_CONFIG: DatasetConfig = {
  outputDir: './dataset',
  splitRatio: { train: 0.8, val: 0.1, test: 0.1 },
};

export class DatasetPipeline {
  private config: DatasetConfig;
  private holonExporter: HolonExporter;
  private contextSerializer: ContextSerializer;
  private negativeSampler: NegativeSampler;
  private positives: TrainingPair[] = [];
  private negatives: NegativeSample[] = [];
  private triplets: EditTriplet[] = [];

  constructor(config?: Partial<DatasetConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.holonExporter = new HolonExporter();
    this.contextSerializer = new ContextSerializer();
    this.negativeSampler = new NegativeSampler();
  }

  async extractPositives(): Promise<TrainingPair[]> {
    const holons = await this.holonExporter.queryGoldHolons({
      ringLevels: [0, 1],
      minConfidence: 0.8,
      outputPath: `${this.config.outputDir}/positives.jsonl`,
    });

    this.positives = holons.map((holon) => {
      const neighborhood = this.extractNeighborhood(holon);
      const serialized = this.contextSerializer.serializeNeighborhood(neighborhood);
      const pair = this.holonExporter.formatAsTrainingPair(holon);

      return {
        ...pair,
        context: `${serialized.promptText}\n\n${pair.context}`,
      };
    });

    return this.positives;
  }

  async extractNegatives(): Promise<NegativeSample[]> {
    const rejectionLogPath = `${this.config.outputDir}/rejections.log`;

    try {
      const fs = await import('node:fs/promises');
      const log = await fs.readFile(rejectionLogPath, 'utf-8');
      this.negatives = await this.negativeSampler.sampleFromRejections(log);
    } catch {
      this.negatives = [];
    }

    if (this.positives.length > 0 && this.negatives.length === 0) {
      const antipatterns = ['Dead Code', 'Phantom Import', 'Type Mismatch'];
      for (const positive of this.positives.slice(0, 10)) {
        for (const antipattern of antipatterns) {
          this.negatives.push(
            this.negativeSampler.generateSyntheticBadExample(positive.code, antipattern)
          );
        }
      }
    }

    return this.negatives;
  }

  async extractTriplets(): Promise<EditTriplet[]> {
    const tripletPath = `${this.config.outputDir}/triplets.jsonl`;

    try {
      const fs = await import('node:fs/promises');
      const content = await fs.readFile(tripletPath, 'utf-8');
      const lines = content.split('\n').filter((l) => l.trim());
      this.triplets = lines.map((line) => JSON.parse(line) as EditTriplet);
    } catch {
      this.triplets = [];
    }

    if (this.triplets.length === 0 && this.positives.length >= 2) {
      for (let i = 0; i < Math.min(this.positives.length - 1, 5); i++) {
        const pre = this.positives[i];
        const post = this.positives[i + 1];
        this.triplets.push({
          preState: pre.code,
          diff: this.computeSimpleDiff(pre.code, post.code),
          postState: post.code,
          isGood: true,
          source: 'synthetic',
        });
      }
    }

    return this.triplets;
  }

  async buildDataset(): Promise<DatasetSplit> {
    await this.extractPositives();
    await this.extractNegatives();
    await this.extractTriplets();

    const allEntries: DatasetEntry[] = [
      ...this.positives.map((p) => ({ type: 'positive' as const, data: p })),
      ...this.negatives.map((n) => ({ type: 'negative' as const, data: n })),
      ...this.triplets.map((t) => ({ type: 'triplet' as const, data: t })),
    ];

    this.shuffleArray(allEntries);

    const { train, val, test } = this.config.splitRatio;
    const total = allEntries.length;
    const trainEnd = Math.floor(total * train);
    const valEnd = Math.floor(total * (train + val));

    return {
      train: allEntries.slice(0, trainEnd),
      val: allEntries.slice(trainEnd, valEnd),
      test: allEntries.slice(valEnd),
    };
  }

  async saveToJSONL(data: unknown[], filename: string): Promise<void> {
    const fs = await import('node:fs/promises');
    const path = await import('node:path');

    const outputPath = path.join(this.config.outputDir, filename);
    const dir = path.dirname(outputPath);
    await fs.mkdir(dir, { recursive: true });

    const lines = data.map((entry) => JSON.stringify(entry));
    await fs.writeFile(outputPath, lines.join('\n') + (lines.length > 0 ? '\n' : ''));
  }

  addPositive(pair: TrainingPair): void {
    this.positives.push(pair);
  }

  addNegative(sample: NegativeSample): void {
    this.negatives.push(sample);
  }

  addTriplet(triplet: EditTriplet): void {
    this.triplets.push(triplet);
  }

  getStats(): { positives: number; negatives: number; triplets: number } {
    return {
      positives: this.positives.length,
      negatives: this.negatives.length,
      triplets: this.triplets.length,
    };
  }

  private extractNeighborhood(holon: Holon): GraphNeighborhood {
    return {
      imports: holon.dependencies ?? [],
      types: holon.symbols?.filter((s) => s.startsWith('type:') || s.startsWith('interface:')) ?? [],
      dependencies: holon.dependencies ?? [],
      callers: [],
      callees: [],
    };
  }

  private computeSimpleDiff(pre: string, post: string): string {
    const preLines = pre.split('\n');
    const postLines = post.split('\n');
    const diffs: string[] = [];

    const maxLen = Math.max(preLines.length, postLines.length);
    for (let i = 0; i < maxLen; i++) {
      if (preLines[i] !== postLines[i]) {
        if (preLines[i]) diffs.push(`-${preLines[i]}`);
        if (postLines[i]) diffs.push(`+${postLines[i]}`);
      }
    }

    return diffs.join('\n') || '(no changes)';
  }

  private shuffleArray<T>(array: T[]): void {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
}
