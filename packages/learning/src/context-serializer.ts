/**
 * ContextSerializer - Convert Graph Neighborhoods into LLM Prompt Context
 *
 * Step 52 of NeSy Evolution: Serializes code graph neighborhoods
 * (imports, types, dependencies, call relationships) into structured
 * prompt text for training data generation.
 */

export interface GraphNeighborhood {
  imports: string[];
  types: string[];
  dependencies: string[];
  callers: string[];
  callees: string[];
}

export interface SerializedContext {
  promptText: string;
  tokenCount: number;
  truncated: boolean;
}

export class ContextSerializer {
  private maxTokens: number;

  constructor(maxTokens: number = 4096) {
    this.maxTokens = maxTokens;
  }

  serializeNeighborhood(neighborhood: GraphNeighborhood): SerializedContext {
    const sections: string[] = [];

    if (neighborhood.imports.length > 0) {
      sections.push(this.formatImports(neighborhood.imports));
    }

    if (neighborhood.types.length > 0) {
      sections.push(this.formatTypes(neighborhood.types));
    }

    if (neighborhood.dependencies.length > 0) {
      sections.push(this.formatDependencies(neighborhood.dependencies));
    }

    if (neighborhood.callers.length > 0) {
      sections.push(this.formatCallers(neighborhood.callers));
    }

    if (neighborhood.callees.length > 0) {
      sections.push(this.formatCallees(neighborhood.callees));
    }

    const fullText = sections.join('\n\n');
    const tokenCount = this.estimateTokens(fullText);

    if (tokenCount <= this.maxTokens) {
      return { promptText: fullText, tokenCount, truncated: false };
    }

    const truncatedText = this.truncateToFit(fullText);
    return {
      promptText: truncatedText,
      tokenCount: this.estimateTokens(truncatedText),
      truncated: true,
    };
  }

  formatImports(imports: string[]): string {
    if (imports.length === 0) return '';
    const items = imports.map(i => `- \`${i}\``).join('\n');
    return `## Imports\n${items}`;
  }

  formatTypes(types: string[]): string {
    if (types.length === 0) return '';
    const items = types.map(t => `- \`${t}\``).join('\n');
    return `## Types\n${items}`;
  }

  estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  private formatDependencies(deps: string[]): string {
    if (deps.length === 0) return '';
    const items = deps.map(d => `- ${d}`).join('\n');
    return `## Dependencies\n${items}`;
  }

  private formatCallers(callers: string[]): string {
    if (callers.length === 0) return '';
    const items = callers.map(c => `- ${c}`).join('\n');
    return `## Callers\n${items}`;
  }

  private formatCallees(callees: string[]): string {
    if (callees.length === 0) return '';
    const items = callees.map(c => `- ${c}`).join('\n');
    return `## Callees\n${items}`;
  }

  private truncateToFit(text: string): string {
    const maxChars = this.maxTokens * 4;
    return text.slice(0, maxChars) + '\n[truncated]';
  }
}
