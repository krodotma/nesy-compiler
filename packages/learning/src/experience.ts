import type { CompilationResult } from '@nesy/pipeline';
import type { Feedback } from './feedback.js';

export interface Experience {
  result: CompilationResult;
  feedback?: Feedback;
  timestamp: number;
}

/**
 * Buffer of compilation experiences for replay-based learning
 */
export class ExperienceBuffer {
  private buffer: Experience[] = [];
  private maxSize: number;

  constructor(maxSize: number = 1000) {
    this.maxSize = maxSize;
  }

  add(result: CompilationResult, feedback?: Feedback): void {
    this.buffer.push({ result, feedback, timestamp: Date.now() });
    if (this.buffer.length > this.maxSize) {
      this.buffer.shift();
    }
  }

  sample(count: number): Experience[] {
    const shuffled = [...this.buffer].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count);
  }

  getSuccessful(): Experience[] {
    return this.buffer.filter(e => e.result.stages.verify.passed);
  }

  size(): number {
    return this.buffer.length;
  }

  clear(): void {
    this.buffer = [];
  }
}
