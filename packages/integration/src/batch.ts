import { type CompilationRequest } from '@nesy/core';
import { type CompilationResult } from '@nesy/pipeline';
import { HolonCompiler } from './holon-compiler.js';
import type { HolonCompilerConfig, BatchOptions } from './types.js';

/**
 * Batch compiler for processing multiple requests
 */
export class BatchCompiler {
  private compiler: HolonCompiler;

  constructor(config: HolonCompilerConfig = {}) {
    this.compiler = new HolonCompiler(config);
  }

  async compileBatch(
    requests: CompilationRequest[],
    options: BatchOptions = {}
  ): Promise<CompilationResult[]> {
    const { concurrency = 1, stopOnError = false, progressCallback } = options;
    const results: CompilationResult[] = [];

    for (let i = 0; i < requests.length; i += concurrency) {
      const batch = requests.slice(i, i + concurrency);
      const promises = batch.map(req => this.compiler.compile(req));

      try {
        const batchResults = await Promise.all(promises);
        results.push(...batchResults);
      } catch (err) {
        if (stopOnError) throw err;
      }

      if (progressCallback) {
        progressCallback(Math.min(i + concurrency, requests.length), requests.length);
      }
    }

    return results;
  }
}
