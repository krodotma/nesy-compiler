/**
 * Token budget management for prompt construction
 */
export class TokenBudget {
  private maxTokens: number;

  constructor(maxTokens: number = 4096) {
    this.maxTokens = maxTokens;
  }

  estimateTokens(text: string): number {
    // Rough estimate: ~4 chars per token
    return Math.ceil(text.length / 4);
  }

  truncate(text: string): string {
    const estimated = this.estimateTokens(text);
    if (estimated <= this.maxTokens) return text;

    const maxChars = this.maxTokens * 4;
    return text.slice(0, maxChars) + '\n[truncated]';
  }

  remaining(usedText: string): number {
    return Math.max(0, this.maxTokens - this.estimateTokens(usedText));
  }
}
