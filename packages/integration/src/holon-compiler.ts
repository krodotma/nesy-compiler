import { type CompilationRequest, type CompiledHolon } from '@nesy/core';
import { NeSyCompiler, type CompilationResult, type CompilerConfig } from '@nesy/pipeline';
import type { HolonCompilerConfig } from './types.js';

/**
 * High-level holon compiler with retry logic and validation
 */
export class HolonCompiler {
  private compiler: NeSyCompiler;
  private config: HolonCompilerConfig;

  constructor(config: HolonCompilerConfig = {}) {
    this.config = { autoRetry: true, maxRetries: 3, ...config };
    this.compiler = new NeSyCompiler(config);
  }

  async compile(request: CompilationRequest): Promise<CompilationResult> {
    let lastError: Error | undefined;
    const maxAttempts = this.config.autoRetry ? (this.config.maxRetries ?? 3) : 1;

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        const result = await this.compiler.compile(request);
        if (result.stages.verify.passed) {
          return result;
        }
        lastError = new Error(`Verification failed on attempt ${attempt + 1}`);
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));
      }
    }

    throw lastError ?? new Error('Compilation failed');
  }

  reset(): void {
    this.compiler.reset();
  }
}
