/**
 * NegativeSampler: Create "Bad Examples" from Rejected Thrash
 *
 * Step 54 of NeSy Evolution Phase 2 (Learning Infrastructure)
 *
 * Generates negative training samples from antipattern rejections
 * to teach models "what NOT to do".
 */

export interface NegativeSample {
  badCode: string;
  reason: string;
  antipatternType: string;
  correction?: string;
}

export interface SamplingConfig {
  antipatternTypes: string[];
  maxSamples: number;
  includeCorrections: boolean;
}

interface RejectionEntry {
  code: string;
  antipatternType: string;
  reason: string;
  correction?: string;
  timestamp?: string;
}

const DEFAULT_CONFIG: SamplingConfig = {
  antipatternTypes: [],
  maxSamples: 100,
  includeCorrections: true,
};

const SYNTHETIC_TRANSFORMS: Record<string, (code: string) => { badCode: string; reason: string }> = {
  'Dead Code': (code) => ({
    badCode: `${code}\n\n// Unreachable code\nconst unusedVar = "never used";\nfunction deadFunction() { return null; }`,
    reason: 'Contains unreachable and unused code that should be removed',
  }),
  'Phantom Import': (code) => ({
    badCode: `import { nonExistent } from './fake-module';\nimport { unusedImport } from 'unused-package';\n\n${code}`,
    reason: 'Imports modules that do not exist or are not used',
  }),
  'Type Mismatch': (code) => ({
    badCode: code.replace(/: string/g, ': number').replace(/: number/g, ': string'),
    reason: 'Contains incompatible type assignments',
  }),
  'Magic Number': (code) => ({
    badCode: code.replace(/const \w+ = (\d+)/g, 'const result = 42 * 1337 + 256'),
    reason: 'Uses hardcoded numeric literals without explanation',
  }),
  'Deep Nesting': (code) => ({
    badCode: `if (true) {\n  if (true) {\n    if (true) {\n      if (true) {\n        if (true) {\n          ${code}\n        }\n      }\n    }\n  }\n}`,
    reason: 'Excessive indentation levels indicating complex control flow',
  }),
  'Long Function': (code) => ({
    badCode: `function doEverything() {\n  ${code}\n  // ... many more lines\n  ${'  console.log("line");\n'.repeat(50)}}`,
    reason: 'Function exceeds reasonable length threshold',
  }),
  'Hallucinated Import': (code) => ({
    badCode: `import { AIGeneratedMagic } from '@nonexistent/ai-magic';\nimport { FakeAPI } from 'hallucinated-library';\n\n${code}`,
    reason: 'AI generated imports for non-existent modules',
  }),
  'Incomplete Implementation': (code) => ({
    badCode: `${code}\n\n// TODO: implement this\n// FIXME: this is broken\nfunction placeholder() {\n  throw new Error('Not implemented');\n}`,
    reason: 'Contains TODO/FIXME comments or placeholder implementations',
  }),
  'Style Inconsistency': (code) => ({
    badCode: code
      .replace(/const /g, 'var ')
      .replace(/'/g, '"')
      .replace(/;$/gm, ''),
    reason: 'Code style differs from project conventions',
  }),
  'Copy-Paste Error': (code) => ({
    badCode: `${code}\n\n// Duplicated with inconsistent changes\n${code.replace(/result/g, 'resutl').replace(/data/g, 'data2')}`,
    reason: 'Duplicated code with inconsistent modifications',
  }),
};

export class NegativeSampler {
  private config: SamplingConfig;

  constructor(config?: Partial<SamplingConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async sampleFromRejections(rejectionLog: string): Promise<NegativeSample[]> {
    const entries = this.parseRejectionLog(rejectionLog);
    let filtered = entries;

    if (this.config.antipatternTypes.length > 0) {
      filtered = entries.filter((e) =>
        this.config.antipatternTypes.includes(e.antipatternType)
      );
    }

    const limited = filtered.slice(0, this.config.maxSamples);

    return limited.map((entry) => ({
      badCode: entry.code,
      reason: entry.reason,
      antipatternType: entry.antipatternType,
      correction: this.config.includeCorrections ? entry.correction : undefined,
    }));
  }

  generateSyntheticBadExample(
    goodCode: string,
    antipatternType: string
  ): NegativeSample {
    const transform = SYNTHETIC_TRANSFORMS[antipatternType];

    if (transform) {
      const { badCode, reason } = transform(goodCode);
      return {
        badCode,
        reason,
        antipatternType,
        correction: this.config.includeCorrections ? goodCode : undefined,
      };
    }

    return {
      badCode: `/* INTENTIONALLY BAD: ${antipatternType} */\n${goodCode}`,
      reason: `Synthetic example of ${antipatternType} antipattern`,
      antipatternType,
      correction: this.config.includeCorrections ? goodCode : undefined,
    };
  }

  formatAsTrainingNegative(sample: NegativeSample): {
    context: string;
    label: 'bad';
    reason: string;
  } {
    const correctionNote = sample.correction
      ? `\n\nCorrection available: ${sample.correction.slice(0, 100)}...`
      : '';

    return {
      context: `[Antipattern: ${sample.antipatternType}]\n\n${sample.badCode}${correctionNote}`,
      label: 'bad',
      reason: sample.reason,
    };
  }

  private parseRejectionLog(log: string): RejectionEntry[] {
    const entries: RejectionEntry[] = [];
    const lines = log.split('\n');

    let currentEntry: Partial<RejectionEntry> | null = null;
    let codeBuffer: string[] = [];
    let inCodeBlock = false;

    for (const line of lines) {
      if (line.startsWith('REJECTION:')) {
        if (currentEntry && currentEntry.code) {
          entries.push(currentEntry as RejectionEntry);
        }
        currentEntry = {
          antipatternType: line.replace('REJECTION:', '').trim(),
          code: '',
          reason: '',
        };
        codeBuffer = [];
        inCodeBlock = false;
      } else if (line.startsWith('REASON:') && currentEntry) {
        currentEntry.reason = line.replace('REASON:', '').trim();
      } else if (line.startsWith('CORRECTION:') && currentEntry) {
        currentEntry.correction = line.replace('CORRECTION:', '').trim();
      } else if (line.startsWith('```') && currentEntry) {
        if (inCodeBlock) {
          currentEntry.code = codeBuffer.join('\n');
          codeBuffer = [];
        }
        inCodeBlock = !inCodeBlock;
      } else if (inCodeBlock) {
        codeBuffer.push(line);
      }
    }

    if (currentEntry && currentEntry.code) {
      entries.push(currentEntry as RejectionEntry);
    }

    return entries;
  }
}
