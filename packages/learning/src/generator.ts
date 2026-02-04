/**
 * Generator - Neuro-Symbolic Code Generation
 *
 * Steps 81-82 of NeSy Evolution Phase 8
 *
 * The Draft/Refine Architecture:
 * 1. RETRIEVAL (50ms)  - FalkorDB context
 * 2. DRAFT (200ms)     - PCFG skeleton
 * 3. REFINE (3-5s)     - SLM fills logic
 * 4. VERIFY (compile)  - Compiler feedback
 *
 * Key insight: "syntactically valid skeleton in <500ms;
 * semantically valid code in <10s with 2 refinement passes"
 */

import type { CompilerDiagnostic, RLCFLoop } from './rlcf.js';

export interface GeneratorConfig {
  /** Maximum draft attempts */
  maxDraftAttempts: number;
  /** Maximum refine iterations */
  maxRefineIterations: number;
  /** Temperature for SLM generation */
  temperature: number;
  /** Use grammar constraints */
  useGrammarConstraints: boolean;
  /** Timeout for SLM inference (ms) */
  slmTimeout: number;
}

const DEFAULT_CONFIG: GeneratorConfig = {
  maxDraftAttempts: 3,
  maxRefineIterations: 2,
  temperature: 0.7,
  useGrammarConstraints: true,
  slmTimeout: 5000,
};

export interface GenerationRequest {
  /** What to generate (function, class, module, etc.) */
  type: 'function' | 'class' | 'module' | 'statement' | 'expression';
  /** Natural language description */
  description: string;
  /** Target language */
  language: 'typescript' | 'python' | 'javascript';
  /** Type signature (if known) */
  signature?: string;
  /** Context from FalkorDB (similar code) */
  context?: ContextItem[];
  /** Constraints (must use these imports, etc.) */
  constraints?: GenerationConstraint[];
}

export interface ContextItem {
  /** Code snippet */
  code: string;
  /** Similarity score */
  similarity: number;
  /** Source path */
  source: string;
  /** Why this is relevant */
  reason: string;
}

export interface GenerationConstraint {
  type: 'must-use' | 'must-not-use' | 'must-import' | 'must-return';
  value: string;
  reason?: string;
}

export interface DraftResult {
  /** Generated skeleton code */
  skeleton: string;
  /** Holes to fill (marked with __HOLE_N__) */
  holes: HoleSpec[];
  /** PCFG derivation used */
  derivation?: PCFGDerivation;
  /** Time taken (ms) */
  draftTime: number;
}

export interface HoleSpec {
  id: string;
  /** What should go here */
  description: string;
  /** Expected type */
  expectedType?: string;
  /** Context around the hole */
  surrounding: string;
  /** Position in skeleton */
  position: { start: number; end: number };
}

export interface PCFGDerivation {
  rule: string;
  children: PCFGDerivation[];
  terminal?: string;
}

export interface RefineResult {
  /** Final generated code */
  code: string;
  /** Filled holes */
  filledHoles: Array<{ id: string; code: string }>;
  /** Compilation result */
  compiles: boolean;
  /** Diagnostics */
  diagnostics: CompilerDiagnostic[];
  /** Number of iterations */
  iterations: number;
  /** Total time (ms) */
  totalTime: number;
}

export interface GenerationResult {
  /** Whether generation succeeded */
  success: boolean;
  /** Final code (if successful) */
  code?: string;
  /** Draft result */
  draft: DraftResult;
  /** Refine result */
  refine?: RefineResult;
  /** Error (if failed) */
  error?: string;
  /** Total time (ms) */
  totalTime: number;
}

/**
 * PCFG Grammar for code structure.
 * Maps non-terminals to production rules.
 */
export interface PCFGGrammar {
  name: string;
  language: 'typescript' | 'python' | 'javascript';
  rules: Map<string, PCFGRule[]>;
  startSymbol: string;
}

export interface PCFGRule {
  /** Production (list of symbols) */
  production: string[];
  /** Probability */
  probability: number;
  /** Template for terminal generation */
  template?: string;
}

/**
 * SLM (Small Language Model) Interface.
 * Can be implemented with node-llama-cpp, transformers.js, etc.
 */
export interface SLMProvider {
  /** Generate text completion */
  complete(
    prompt: string,
    options: {
      maxTokens: number;
      temperature: number;
      stopSequences?: string[];
      grammar?: string; // GBNF grammar for constrained decoding
    }
  ): Promise<string>;

  /** Check if model is ready */
  isReady(): boolean;

  /** Get model info */
  getInfo(): { name: string; size: string; quantization: string };
}

/**
 * Generator: Draft/Refine code generation.
 */
export class Generator {
  private config: GeneratorConfig;
  private grammars: Map<string, PCFGGrammar>;
  private slm: SLMProvider | null;
  private rlcf: RLCFLoop | null;

  constructor(config?: Partial<GeneratorConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.grammars = new Map();
    this.slm = null;
    this.rlcf = null;

    // Initialize default grammars
    this.initializeDefaultGrammars();
  }

  /**
   * Set SLM provider.
   */
  setSLM(slm: SLMProvider): void {
    this.slm = slm;
  }

  /**
   * Set RLCF loop for refinement.
   */
  setRLCF(rlcf: RLCFLoop): void {
    this.rlcf = rlcf;
  }

  /**
   * Generate code from request.
   */
  async generate(
    request: GenerationRequest,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<GenerationResult> {
    const startTime = Date.now();

    // Step 1: Draft (PCFG skeleton)
    const draft = await this.draft(request);

    // If no holes, we're done
    if (draft.holes.length === 0) {
      const result = await compile(draft.skeleton);
      return {
        success: result.passed,
        code: result.passed ? draft.skeleton : undefined,
        draft,
        totalTime: Date.now() - startTime,
      };
    }

    // Step 2: Refine (SLM fills logic)
    const refine = await this.refine(draft, request, compile);

    return {
      success: refine.compiles,
      code: refine.compiles ? refine.code : undefined,
      draft,
      refine,
      totalTime: Date.now() - startTime,
    };
  }

  /**
   * Draft: Generate PCFG skeleton with holes.
   */
  async draft(request: GenerationRequest): Promise<DraftResult> {
    const startTime = Date.now();
    const grammar = this.grammars.get(request.language);

    if (!grammar) {
      // Fallback: template-based drafting
      return this.templateDraft(request, startTime);
    }

    // PCFG derivation
    const derivation = this.derivePCFG(grammar, request);
    const { skeleton, holes } = this.derivationToCode(derivation, request);

    return {
      skeleton,
      holes,
      derivation,
      draftTime: Date.now() - startTime,
    };
  }

  /**
   * Refine: Fill holes with SLM.
   */
  async refine(
    draft: DraftResult,
    request: GenerationRequest,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RefineResult> {
    const startTime = Date.now();
    let code = draft.skeleton;
    const filledHoles: Array<{ id: string; code: string }> = [];

    // Fill each hole
    for (const hole of draft.holes) {
      const filled = await this.fillHole(hole, request, code);
      filledHoles.push({ id: hole.id, code: filled });
      code = code.replace(hole.id, filled);
    }

    // Compile and check
    let result = await compile(code);
    let iterations = 1;

    // RLCF refinement loop
    if (!result.passed && this.rlcf) {
      const rlcfResult = await this.rlcf.refine(code, compile);
      code = rlcfResult.code;
      result = await compile(code);
      iterations += rlcfResult.iteration;
    }

    // Simple refinement if no RLCF
    while (!result.passed && iterations < this.config.maxRefineIterations) {
      code = await this.attemptFix(code, result.diagnostics, request);
      result = await compile(code);
      iterations++;
    }

    return {
      code,
      filledHoles,
      compiles: result.passed,
      diagnostics: result.diagnostics,
      iterations,
      totalTime: Date.now() - startTime,
    };
  }

  /**
   * Fill a hole using SLM.
   */
  private async fillHole(
    hole: HoleSpec,
    request: GenerationRequest,
    currentCode: string
  ): Promise<string> {
    if (!this.slm) {
      // Fallback: generate placeholder
      return this.placeholderForHole(hole);
    }

    const prompt = this.buildHolePrompt(hole, request, currentCode);

    try {
      const completion = await Promise.race([
        this.slm.complete(prompt, {
          maxTokens: 256,
          temperature: this.config.temperature,
          stopSequences: ['\n\n', '```', hole.expectedType ? ';' : undefined].filter(Boolean) as string[],
        }),
        new Promise<string>((_, reject) =>
          setTimeout(() => reject(new Error('SLM timeout')), this.config.slmTimeout)
        ),
      ]);

      return completion.trim();
    } catch (error) {
      // Fallback on error
      return this.placeholderForHole(hole);
    }
  }

  /**
   * Build prompt for filling a hole.
   */
  private buildHolePrompt(
    hole: HoleSpec,
    request: GenerationRequest,
    currentCode: string
  ): string {
    const contextSnippets = request.context?.slice(0, 3).map(c =>
      `// Similar code from ${c.source}:\n${c.code}`
    ).join('\n\n') || '';

    return `
// Task: ${request.description}
// Fill in the code for: ${hole.description}
${hole.expectedType ? `// Expected type: ${hole.expectedType}` : ''}

${contextSnippets}

// Current code:
${currentCode}

// Generate code to replace ${hole.id}:
`.trim();
  }

  /**
   * Generate placeholder for hole (when SLM not available).
   */
  private placeholderForHole(hole: HoleSpec): string {
    if (hole.expectedType) {
      switch (hole.expectedType) {
        case 'string':
          return '""';
        case 'number':
          return '0';
        case 'boolean':
          return 'false';
        case 'void':
          return '';
        case 'any':
          return 'undefined';
        default:
          if (hole.expectedType.endsWith('[]')) {
            return '[]';
          }
          return `{} as ${hole.expectedType}`;
      }
    }
    return '/* TODO: implement */';
  }

  /**
   * Attempt to fix code based on diagnostics.
   */
  private async attemptFix(
    code: string,
    diagnostics: CompilerDiagnostic[],
    request: GenerationRequest
  ): Promise<string> {
    if (!this.slm) {
      return code;
    }

    const errorMessages = diagnostics
      .filter(d => d.type === 'error')
      .slice(0, 3)
      .map(d => `- ${d.code}: ${d.message}`)
      .join('\n');

    const prompt = `
// Fix the following errors in this ${request.language} code:
${errorMessages}

// Current code:
${code}

// Fixed code:
`.trim();

    try {
      const fixed = await this.slm.complete(prompt, {
        maxTokens: 1024,
        temperature: 0.3, // Lower temperature for fixes
      });
      return fixed.trim();
    } catch {
      return code;
    }
  }

  /**
   * Template-based drafting (fallback when no PCFG).
   */
  private templateDraft(
    request: GenerationRequest,
    startTime: number
  ): DraftResult {
    const templates: Record<GenerationRequest['type'], Record<string, string>> = {
      function: {
        typescript: `function __HOLE_NAME__(__HOLE_PARAMS__): __HOLE_RETURN__ {\n  __HOLE_BODY__\n}`,
        javascript: `function __HOLE_NAME__(__HOLE_PARAMS__) {\n  __HOLE_BODY__\n}`,
        python: `def __HOLE_NAME__(__HOLE_PARAMS__):\n    __HOLE_BODY__`,
      },
      class: {
        typescript: `class __HOLE_NAME__ {\n  __HOLE_BODY__\n}`,
        javascript: `class __HOLE_NAME__ {\n  __HOLE_BODY__\n}`,
        python: `class __HOLE_NAME__:\n    __HOLE_BODY__`,
      },
      module: {
        typescript: `__HOLE_IMPORTS__\n\n__HOLE_BODY__\n\nexport { __HOLE_EXPORTS__ };`,
        javascript: `__HOLE_IMPORTS__\n\n__HOLE_BODY__\n\nmodule.exports = { __HOLE_EXPORTS__ };`,
        python: `__HOLE_IMPORTS__\n\n__HOLE_BODY__`,
      },
      statement: {
        typescript: `__HOLE_BODY__;`,
        javascript: `__HOLE_BODY__;`,
        python: `__HOLE_BODY__`,
      },
      expression: {
        typescript: `__HOLE_BODY__`,
        javascript: `__HOLE_BODY__`,
        python: `__HOLE_BODY__`,
      },
    };

    const skeleton = templates[request.type]?.[request.language] || '__HOLE_BODY__';
    const holes = this.extractHoles(skeleton, request);

    return {
      skeleton,
      holes,
      draftTime: Date.now() - startTime,
    };
  }

  /**
   * Extract holes from template.
   */
  private extractHoles(skeleton: string, request: GenerationRequest): HoleSpec[] {
    const holes: HoleSpec[] = [];
    const regex = /__HOLE_(\w+)__/g;
    let match;

    while ((match = regex.exec(skeleton)) !== null) {
      const id = match[0];
      const name = match[1].toLowerCase();

      holes.push({
        id,
        description: this.holeDescription(name, request),
        expectedType: this.holeType(name, request),
        surrounding: skeleton.slice(
          Math.max(0, match.index - 20),
          Math.min(skeleton.length, match.index + id.length + 20)
        ),
        position: { start: match.index, end: match.index + id.length },
      });
    }

    return holes;
  }

  /**
   * Get description for hole type.
   */
  private holeDescription(name: string, request: GenerationRequest): string {
    const descriptions: Record<string, string> = {
      name: `Name for the ${request.type}`,
      params: 'Function parameters',
      return: 'Return type',
      body: `Implementation for: ${request.description}`,
      imports: 'Required imports',
      exports: 'Exported symbols',
    };
    return descriptions[name] || `Fill in ${name}`;
  }

  /**
   * Get expected type for hole.
   */
  private holeType(name: string, request: GenerationRequest): string | undefined {
    const types: Record<string, string> = {
      name: 'string',
      params: 'string',
      return: 'string',
    };
    return types[name];
  }

  /**
   * Derive PCFG to generate structure.
   */
  private derivePCFG(
    grammar: PCFGGrammar,
    request: GenerationRequest
  ): PCFGDerivation {
    return this.deriveSymbol(grammar, grammar.startSymbol, request, 0);
  }

  /**
   * Derive a single symbol.
   */
  private deriveSymbol(
    grammar: PCFGGrammar,
    symbol: string,
    request: GenerationRequest,
    depth: number
  ): PCFGDerivation {
    const rules = grammar.rules.get(symbol);

    if (!rules || rules.length === 0 || depth > 10) {
      // Terminal or max depth
      return { rule: symbol, children: [], terminal: symbol };
    }

    // Sample rule by probability
    const rule = this.sampleRule(rules);

    const children = rule.production.map(s =>
      this.deriveSymbol(grammar, s, request, depth + 1)
    );

    return {
      rule: `${symbol} -> ${rule.production.join(' ')}`,
      children,
      terminal: rule.template,
    };
  }

  /**
   * Sample a production rule by probability.
   */
  private sampleRule(rules: PCFGRule[]): PCFGRule {
    const total = rules.reduce((sum, r) => sum + r.probability, 0);
    let r = Math.random() * total;

    for (const rule of rules) {
      r -= rule.probability;
      if (r <= 0) {
        return rule;
      }
    }

    return rules[rules.length - 1];
  }

  /**
   * Convert derivation to code with holes.
   */
  private derivationToCode(
    derivation: PCFGDerivation,
    request: GenerationRequest
  ): { skeleton: string; holes: HoleSpec[] } {
    const holes: HoleSpec[] = [];
    const code = this.derivationToString(derivation, holes, 0);

    return { skeleton: code, holes };
  }

  /**
   * Convert derivation tree to string.
   */
  private derivationToString(
    derivation: PCFGDerivation,
    holes: HoleSpec[],
    position: number
  ): string {
    if (derivation.terminal) {
      if (derivation.terminal.startsWith('__HOLE_')) {
        holes.push({
          id: derivation.terminal,
          description: `Fill: ${derivation.rule}`,
          surrounding: '',
          position: { start: position, end: position + derivation.terminal.length },
        });
      }
      return derivation.terminal;
    }

    return derivation.children.map(c =>
      this.derivationToString(c, holes, position)
    ).join('');
  }

  /**
   * Initialize default PCFG grammars.
   */
  private initializeDefaultGrammars(): void {
    // TypeScript function grammar
    const tsGrammar: PCFGGrammar = {
      name: 'typescript-function',
      language: 'typescript',
      startSymbol: 'Function',
      rules: new Map([
        ['Function', [
          { production: ['FunctionDecl'], probability: 0.6 },
          { production: ['ArrowFunction'], probability: 0.4 },
        ]],
        ['FunctionDecl', [
          { production: [], probability: 1.0, template: 'function __HOLE_NAME__(__HOLE_PARAMS__): __HOLE_RETURN__ {\n  __HOLE_BODY__\n}' },
        ]],
        ['ArrowFunction', [
          { production: [], probability: 1.0, template: 'const __HOLE_NAME__ = (__HOLE_PARAMS__): __HOLE_RETURN__ => {\n  __HOLE_BODY__\n};' },
        ]],
      ]),
    };

    this.grammars.set('typescript', tsGrammar);

    // Python function grammar
    const pyGrammar: PCFGGrammar = {
      name: 'python-function',
      language: 'python',
      startSymbol: 'Function',
      rules: new Map([
        ['Function', [
          { production: ['FunctionDef'], probability: 1.0 },
        ]],
        ['FunctionDef', [
          { production: [], probability: 1.0, template: 'def __HOLE_NAME__(__HOLE_PARAMS__) -> __HOLE_RETURN__:\n    __HOLE_BODY__' },
        ]],
      ]),
    };

    this.grammars.set('python', pyGrammar);
  }

  /**
   * Add custom grammar.
   */
  addGrammar(grammar: PCFGGrammar): void {
    this.grammars.set(grammar.language, grammar);
  }
}

/**
 * Create generator with default config.
 */
export function createGenerator(config?: Partial<GeneratorConfig>): Generator {
  return new Generator(config);
}
