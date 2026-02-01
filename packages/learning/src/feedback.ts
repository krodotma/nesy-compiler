import type { CompilationResult } from '@nesy/pipeline';

export interface Feedback {
  resultId: string;
  quality: number;      // 0-1
  correctness: boolean;
  notes?: string;
  timestamp: number;
}

/**
 * Collects feedback on compilation results for learning
 */
export class FeedbackCollector {
  private feedback: Feedback[] = [];

  record(result: CompilationResult, quality: number, correctness: boolean, notes?: string): Feedback {
    const fb: Feedback = {
      resultId: result.ir.id,
      quality,
      correctness,
      notes,
      timestamp: Date.now(),
    };
    this.feedback.push(fb);
    return fb;
  }

  getAll(): Feedback[] {
    return [...this.feedback];
  }

  getPositive(): Feedback[] {
    return this.feedback.filter(f => f.correctness && f.quality >= 0.7);
  }

  getNegative(): Feedback[] {
    return this.feedback.filter(f => !f.correctness || f.quality < 0.3);
  }

  averageQuality(): number {
    if (this.feedback.length === 0) return 0;
    return this.feedback.reduce((sum, f) => sum + f.quality, 0) / this.feedback.length;
  }
}
