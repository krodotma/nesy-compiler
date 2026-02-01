import { describe, it, expect } from 'vitest';
import { HolonCompiler } from '../holon-compiler.js';
import { PipelineRunner } from '../pipeline-runner.js';
import { BatchCompiler } from '../batch.js';

describe('HolonCompiler', () => {
  it('compiles', async () => {
    const r = await new HolonCompiler({ autoRetry: false }).compile({ mode: 'atom', intent: 'x' });
    expect(r.ir.kind).toBe('compiled');
  });
});

describe('PipelineRunner', () => {
  it('calls stage callbacks', async () => {
    const stages: string[] = [];
    await new PipelineRunner({ onStageComplete: s => stages.push(s) }).run({ mode: 'atom', intent: 'x' });
    expect(stages).toContain('perceive');
  });
});

describe('BatchCompiler', () => {
  it('compiles batch', async () => {
    const results = await new BatchCompiler({ autoRetry: false }).compileBatch([
      { mode: 'atom', intent: 'a' }, { mode: 'atom', intent: 'b' },
    ]);
    expect(results).toHaveLength(2);
  });
});
