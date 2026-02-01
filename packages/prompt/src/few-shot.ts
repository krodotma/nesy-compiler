import type { SymbolicStructure } from '@nesy/core';

export interface FewShotExample {
  input: string;
  output: string;
  structure?: SymbolicStructure;
  tags?: string[];
}

/**
 * Manages few-shot examples for prompt construction
 */
export class FewShotManager {
  private examples: FewShotExample[] = [];

  add(example: FewShotExample): void {
    this.examples.push(example);
  }

  select(tags?: string[], maxCount: number = 3): FewShotExample[] {
    let pool = this.examples;
    if (tags && tags.length > 0) {
      pool = pool.filter(ex => ex.tags?.some(t => tags.includes(t)));
    }
    return pool.slice(0, maxCount);
  }

  format(examples: FewShotExample[]): string {
    return examples.map((ex, i) =>
      `Example ${i + 1}:\nInput: ${ex.input}\nOutput: ${ex.output}`
    ).join('\n\n');
  }

  getAll(): FewShotExample[] {
    return [...this.examples];
  }
}
