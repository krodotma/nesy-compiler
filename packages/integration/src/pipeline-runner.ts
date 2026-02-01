import { type CompilationRequest } from '@nesy/core';
import { NeSyCompiler, type CompilationResult, type CompilerConfig } from '@nesy/pipeline';

export interface PipelineRunnerOptions extends CompilerConfig {
  onStageComplete?: (stage: string, durationMs: number) => void;
}

/**
 * Pipeline runner with stage-level hooks
 */
export class PipelineRunner {
  private compiler: NeSyCompiler;
  private options: PipelineRunnerOptions;

  constructor(options: PipelineRunnerOptions = {}) {
    this.options = options;
    this.compiler = new NeSyCompiler(options);
  }

  async run(request: CompilationRequest): Promise<CompilationResult> {
    const result = await this.compiler.compile(request);

    if (this.options.onStageComplete) {
      for (const [stage, duration] of Object.entries(result.metrics.stageDurations)) {
        this.options.onStageComplete(stage, duration);
      }
    }

    return result;
  }
}
