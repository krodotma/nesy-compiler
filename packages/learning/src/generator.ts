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
  maxDraftAttempts: number;
  maxRefineIterations: number;
  temperature: number;
  useGrammarConstraints: boolean;
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
  type: 'function' | 'class' | 'module' | 'statement' | 'expression';
  description: string;
  language: 'typescript' | 'python' | 'javascript';
  signature?: string;
  context?: ContextItem[];
  constraints?: GenerationConstraint[];
}

export interface ContextItem {
  code: string;
  similarity: number;
  source: string;
  reason: string;
}

export interface GenerationConstraint {
  type: 'must-use' | 'must-not-use' | 'must-import' | 'must-return';
  value: string;
  reason?: string;
}

export interface DraftResult {
  skeleton: string;
  holes: HoleSpec[];
  derivation?: PCFGDerivation;
  draftTime: number;
}

export interface HoleSpec {
  id: string;
  description: string;
  expectedType?: string;
  surrounding: string;
  position: { start: number; end: number };
}

export interface PCFGDerivation {
  rule: string;
  children: PCFGDerivation[];
  terminal?: string;
}

export interface RefineResult {
  code: string;
  filledHoles: Array<{ id: string; code: string }>;
  compiles: boolean;
  diagnostics: CompilerDiagnostic[];
  iterations: number;
  totalTime: number;
}

export interface GenerationResult {
  success: boolean;
  code?: string;
  draft: DraftResult;
  refine?: RefineResult;
  error?: string;
  totalTime: number;
}

export interface SLMProvider {
  complete(prompt: string, options: {
    maxTokens: number;
    temperature: number;
    stopSequences?: string[];
    grammar?: string;
  }): Promise<string>;
  isReady(): boolean;
  getInfo(): { name: string; size: string; quantization: string };
}

interface PCFGGrammar {
  name: string;
  language: 'typescript' | 'python' | 'javascript';
  rules: Map<string, PCFGRule[]>;
  startSymbol: string;
}

interface PCFGRule {
  production: string[];
  probability: number;
  template?: string;
}

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
    this.initializeDefaultGrammars();
  }

  setSLM(slm: SLMProvider): void { this.slm = slm; }
  setRLCF(rlcf: RLCFLoop): void { this.rlcf = rlcf; }

  async generate(
    request: GenerationRequest,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<GenerationResult> {
    const startTime = Date.now();
    const draft = await this.draft(request);

    if (draft.holes.length === 0) {
      const result = await compile(draft.skeleton);
      return { success: result.passed, code: result.passed ? draft.skeleton : undefined, draft, totalTime: Date.now() - startTime };
    }

    const refine = await this.refine(draft, request, compile);
    return { success: refine.compiles, code: refine.compiles ? refine.code : undefined, draft, refine, totalTime: Date.now() - startTime };
  }

  async draft(request: GenerationRequest): Promise<DraftResult> {
    const startTime = Date.now();
    const grammar = this.grammars.get(request.language);
    if (!grammar) return this.templateDraft(request, startTime);

    const derivation = this.derivePCFG(grammar, request);
    const { skeleton, holes } = this.derivationToCode(derivation, request);
    return { skeleton, holes, derivation, draftTime: Date.now() - startTime };
  }

  async refine(
    draft: DraftResult,
    request: GenerationRequest,
    compile: (code: string) => Promise<{ passed: boolean; diagnostics: CompilerDiagnostic[] }>
  ): Promise<RefineResult> {
    const startTime = Date.now();
    let code = draft.skeleton;
    const filledHoles: Array<{ id: string; code: string }> = [];

    for (const hole of draft.holes) {
      const filled = await this.fillHole(hole, request, code);
      filledHoles.push({ id: hole.id, code: filled });
      code = code.replace(hole.id, filled);
    }

    let result = await compile(code);
    let iterations = 1;

    if (!result.passed && this.rlcf) {
      const rlcfResult = await this.rlcf.refine(code, compile);
      code = rlcfResult.code;
      result = await compile(code);
      iterations += rlcfResult.iteration;
    }

    while (!result.passed && iterations < this.config.maxRefineIterations) {
      code = await this.attemptFix(code, result.diagnostics, request);
      result = await compile(code);
      iterations++;
    }

    return { code, filledHoles, compiles: result.passed, diagnostics: result.diagnostics, iterations, totalTime: Date.now() - startTime };
  }

  private async fillHole(hole: HoleSpec, request: GenerationRequest, currentCode: string): Promise<string> {
    if (!this.slm) return this.placeholderForHole(hole);

    const prompt = this.buildHolePrompt(hole, request, currentCode);
    try {
      const completion = await Promise.race([
        this.slm.complete(prompt, {
          maxTokens: 256,
          temperature: this.config.temperature,
          stopSequences: ['\n\n', '```', hole.expectedType ? ';' : undefined].filter(Boolean) as string[],
        }),
        new Promise<string>((_, reject) => setTimeout(() => reject(new Error('SLM timeout')), this.config.slmTimeout)),
      ]);
      return completion.trim();
    } catch {
      return this.placeholderForHole(hole);
    }
  }

  private buildHolePrompt(hole: HoleSpec, request: GenerationRequest, currentCode: string): string {
    const contextSnippets = request.context?.slice(0, 3).map(c => `// Similar: ${c.source}\n${c.code}`).join('\n\n') || '';
    return `// Task: ${request.description}\n// Fill: ${hole.description}\n${hole.expectedType ? `// Type: ${hole.expectedType}` : ''}\n${contextSnippets}\n\n${currentCode}\n\n// Generate code for ${hole.id}:`.trim();
  }

  private placeholderForHole(hole: HoleSpec): string {
    if (!hole.expectedType) return '/* TODO */';
    switch (hole.expectedType) {
      case 'string': return '""';
      case 'number': return '0';
      case 'boolean': return 'false';
      case 'void': return '';
      default: return hole.expectedType.endsWith('[]') ? '[]' : `{} as ${hole.expectedType}`;
    }
  }

  private async attemptFix(code: string, diagnostics: CompilerDiagnostic[], request: GenerationRequest): Promise<string> {
    if (!this.slm) return code;
    const errorMessages = diagnostics.filter(d => d.type === 'error').slice(0, 3).map(d => `- ${d.code}: ${d.message}`).join('\n');
    const prompt = `// Fix errors:\n${errorMessages}\n\n// Code:\n${code}\n\n// Fixed:`.trim();
    try {
      return (await this.slm.complete(prompt, { maxTokens: 1024, temperature: 0.3 })).trim();
    } catch { return code; }
  }

  private templateDraft(request: GenerationRequest, startTime: number): DraftResult {
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
      statement: { typescript: `__HOLE_BODY__;`, javascript: `__HOLE_BODY__;`, python: `__HOLE_BODY__` },
      expression: { typescript: `__HOLE_BODY__`, javascript: `__HOLE_BODY__`, python: `__HOLE_BODY__` },
    };
    const skeleton = templates[request.type]?.[request.language] || '__HOLE_BODY__';
    return { skeleton, holes: this.extractHoles(skeleton, request), draftTime: Date.now() - startTime };
  }

  private extractHoles(skeleton: string, request: GenerationRequest): HoleSpec[] {
    const holes: HoleSpec[] = [];
    const regex = /__HOLE_(\w+)__/g;
    let match;
    while ((match = regex.exec(skeleton)) !== null) {
      const id = match[0];
      const name = match[1].toLowerCase();
      holes.push({
        id, description: this.holeDescription(name, request), expectedType: this.holeType(name),
        surrounding: skeleton.slice(Math.max(0, match.index - 20), Math.min(skeleton.length, match.index + id.length + 20)),
        position: { start: match.index, end: match.index + id.length },
      });
    }
    return holes;
  }

  private holeDescription(name: string, request: GenerationRequest): string {
    const desc: Record<string, string> = { name: `Name for ${request.type}`, params: 'Parameters', return: 'Return type', body: `Implement: ${request.description}`, imports: 'Imports', exports: 'Exports' };
    return desc[name] || `Fill ${name}`;
  }

  private holeType(name: string): string | undefined {
    return { name: 'string', params: 'string', return: 'string' }[name];
  }

  private derivePCFG(grammar: PCFGGrammar, request: GenerationRequest): PCFGDerivation {
    return this.deriveSymbol(grammar, grammar.startSymbol, 0);
  }

  private deriveSymbol(grammar: PCFGGrammar, symbol: string, depth: number): PCFGDerivation {
    const rules = grammar.rules.get(symbol);
    if (!rules || rules.length === 0 || depth > 10) {
      return { rule: symbol, children: [], terminal: symbol };
    }
    const rule = this.sampleRule(rules);
    return { rule: `${symbol} -> ${rule.production.join(' ')}`, children: rule.production.map(s => this.deriveSymbol(grammar, s, depth + 1)), terminal: rule.template };
  }

  private sampleRule(rules: PCFGRule[]): PCFGRule {
    const total = rules.reduce((sum, r) => sum + r.probability, 0);
    let r = Math.random() * total;
    for (const rule of rules) { r -= rule.probability; if (r <= 0) return rule; }
    return rules[rules.length - 1];
  }

  private derivationToCode(derivation: PCFGDerivation, request: GenerationRequest): { skeleton: string; holes: HoleSpec[] } {
    const holes: HoleSpec[] = [];
    const code = this.derivationToString(derivation, holes, 0);
    return { skeleton: code, holes };
  }

  private derivationToString(derivation: PCFGDerivation, holes: HoleSpec[], position: number): string {
    if (derivation.terminal) {
      if (derivation.terminal.startsWith('__HOLE_')) {
        holes.push({ id: derivation.terminal, description: `Fill: ${derivation.rule}`, surrounding: '', position: { start: position, end: position + derivation.terminal.length } });
      }
      return derivation.terminal;
    }
    return derivation.children.map(c => this.derivationToString(c, holes, position)).join('');
  }

  private initializeDefaultGrammars(): void {
    const tsGrammar: PCFGGrammar = {
      name: 'typescript-function', language: 'typescript', startSymbol: 'Function',
      rules: new Map([
        ['Function', [{ production: ['FunctionDecl'], probability: 0.6 }, { production: ['ArrowFunction'], probability: 0.4 }]],
        ['FunctionDecl', [{ production: [], probability: 1.0, template: 'function __HOLE_NAME__(__HOLE_PARAMS__): __HOLE_RETURN__ {\n  __HOLE_BODY__\n}' }]],
        ['ArrowFunction', [{ production: [], probability: 1.0, template: 'const __HOLE_NAME__ = (__HOLE_PARAMS__): __HOLE_RETURN__ => {\n  __HOLE_BODY__\n};' }]],
      ]),
    };
    this.grammars.set('typescript', tsGrammar);

    const pyGrammar: PCFGGrammar = {
      name: 'python-function', language: 'python', startSymbol: 'Function',
      rules: new Map([['Function', [{ production: ['FunctionDef'], probability: 1.0 }]], ['FunctionDef', [{ production: [], probability: 1.0, template: 'def __HOLE_NAME__(__HOLE_PARAMS__) -> __HOLE_RETURN__:\n    __HOLE_BODY__' }]]]),
    };
    this.grammars.set('python', pyGrammar);
  }

  addGrammar(grammar: PCFGGrammar): void { this.grammars.set(grammar.language, grammar); }
}

export function createGenerator(config?: Partial<GeneratorConfig>): Generator {
  return new Generator(config);
}
