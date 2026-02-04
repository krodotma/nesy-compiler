/**
 * PCFG Grammar - Probabilistic Context-Free Grammar for Code
 *
 * Step 55 of NeSy Evolution Phase 6 (Idiolect Dataset)
 *
 * PCFG learns code structure probabilistically:
 * - Parse existing code to learn production rules
 * - Use probabilities to generate likely structures
 * - Convert to GBNF for constrained decoding
 *
 * The grammar guides the SLM to generate syntactically valid code.
 */

export interface PCFGRule {
  /** Left-hand side non-terminal */
  lhs: string;
  /** Right-hand side symbols */
  rhs: string[];
  /** Probability (0-1) */
  probability: number;
  /** Count of times this rule was observed */
  count: number;
  /** Template for terminal generation */
  template?: string;
}

export interface PCFGGrammar {
  name: string;
  language: 'typescript' | 'python' | 'javascript' | 'rust' | 'go';
  rules: PCFGRule[];
  terminals: Set<string>;
  nonTerminals: Set<string>;
  startSymbol: string;
}

export interface GBNFGrammar {
  /** GBNF grammar string */
  grammar: string;
  /** Rule name -> GBNF rule mapping */
  ruleMap: Map<string, string>;
}

export interface ParseTreeNode {
  type: string;
  text?: string;
  children: ParseTreeNode[];
  start: number;
  end: number;
}

/**
 * PCFG Learner: Learn grammar from code examples.
 */
export class PCFGLearner {
  private ruleCounts: Map<string, Map<string, number>>;
  private terminals: Set<string>;
  private nonTerminals: Set<string>;

  constructor() {
    this.ruleCounts = new Map();
    this.terminals = new Set();
    this.nonTerminals = new Set();
  }

  /**
   * Learn from a parse tree.
   */
  learnFromTree(tree: ParseTreeNode): void {
    this.visitNode(tree);
  }

  /**
   * Visit a node and record production rules.
   */
  private visitNode(node: ParseTreeNode): void {
    this.nonTerminals.add(node.type);

    if (node.children.length === 0) {
      // Terminal
      if (node.text) {
        this.terminals.add(node.text);
      }
      return;
    }

    // Record production rule: node.type -> [child1.type, child2.type, ...]
    const rhs = node.children.map(c => c.type).join(' ');

    if (!this.ruleCounts.has(node.type)) {
      this.ruleCounts.set(node.type, new Map());
    }

    const counts = this.ruleCounts.get(node.type)!;
    counts.set(rhs, (counts.get(rhs) || 0) + 1);

    // Recurse
    for (const child of node.children) {
      this.visitNode(child);
    }
  }

  /**
   * Build PCFG from learned rules.
   */
  buildGrammar(name: string, language: PCFGGrammar['language'], startSymbol: string): PCFGGrammar {
    const rules: PCFGRule[] = [];

    for (const [lhs, rhsCounts] of this.ruleCounts) {
      // Calculate total count for normalization
      let total = 0;
      for (const count of rhsCounts.values()) {
        total += count;
      }

      // Create rules with probabilities
      for (const [rhsStr, count] of rhsCounts) {
        const rhs = rhsStr.split(' ').filter(s => s.length > 0);
        rules.push({
          lhs,
          rhs,
          probability: count / total,
          count,
        });
      }
    }

    return {
      name,
      language,
      rules,
      terminals: this.terminals,
      nonTerminals: this.nonTerminals,
      startSymbol,
    };
  }

  /**
   * Clear learned data.
   */
  clear(): void {
    this.ruleCounts.clear();
    this.terminals.clear();
    this.nonTerminals.clear();
  }
}

/**
 * PCFG Generator: Generate code from grammar.
 */
export class PCFGGenerator {
  private grammar: PCFGGrammar;
  private rulesByLHS: Map<string, PCFGRule[]>;
  private maxDepth: number;

  constructor(grammar: PCFGGrammar, maxDepth: number = 20) {
    this.grammar = grammar;
    this.maxDepth = maxDepth;

    // Index rules by LHS
    this.rulesByLHS = new Map();
    for (const rule of grammar.rules) {
      if (!this.rulesByLHS.has(rule.lhs)) {
        this.rulesByLHS.set(rule.lhs, []);
      }
      this.rulesByLHS.get(rule.lhs)!.push(rule);
    }
  }

  /**
   * Generate a derivation tree.
   */
  generate(): ParseTreeNode {
    return this.deriveSymbol(this.grammar.startSymbol, 0);
  }

  /**
   * Derive a symbol.
   */
  private deriveSymbol(symbol: string, depth: number): ParseTreeNode {
    const rules = this.rulesByLHS.get(symbol);

    // Terminal or max depth
    if (!rules || rules.length === 0 || depth >= this.maxDepth) {
      return {
        type: symbol,
        text: this.grammar.terminals.has(symbol) ? symbol : undefined,
        children: [],
        start: 0,
        end: 0,
      };
    }

    // Sample rule by probability
    const rule = this.sampleRule(rules);

    // Derive children
    const children = rule.rhs.map(s => this.deriveSymbol(s, depth + 1));

    return {
      type: symbol,
      children,
      start: 0,
      end: 0,
    };
  }

  /**
   * Sample a rule by probability.
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
   * Convert derivation to string.
   */
  treeToString(node: ParseTreeNode): string {
    if (node.text !== undefined) {
      return node.text;
    }

    return node.children.map(c => this.treeToString(c)).join('');
  }
}

/**
 * Convert PCFG to GBNF for llama.cpp constrained decoding.
 *
 * GBNF (GGML BNF) format:
 * rule-name ::= expression
 */
export function pcfgToGBNF(grammar: PCFGGrammar): GBNFGrammar {
  const ruleMap = new Map<string, string>();
  const lines: string[] = [];

  // Convert non-terminal names to GBNF-safe names
  const safeName = (name: string): string =>
    name.toLowerCase().replace(/[^a-z0-9]/g, '-');

  // Group rules by LHS
  const rulesByLHS = new Map<string, PCFGRule[]>();
  for (const rule of grammar.rules) {
    if (!rulesByLHS.has(rule.lhs)) {
      rulesByLHS.set(rule.lhs, []);
    }
    rulesByLHS.get(rule.lhs)!.push(rule);
  }

  // Generate GBNF rules
  for (const [lhs, rules] of rulesByLHS) {
    const gbnfLHS = safeName(lhs);

    // Multiple rules become alternatives
    const alternatives = rules.map(rule => {
      if (rule.template) {
        // Use template as literal
        return `"${escapeGBNF(rule.template)}"`;
      }

      // Convert RHS symbols
      return rule.rhs.map(s => {
        if (grammar.terminals.has(s)) {
          return `"${escapeGBNF(s)}"`;
        }
        return safeName(s);
      }).join(' ');
    });

    const gbnfRule = `${gbnfLHS} ::= ${alternatives.join(' | ')}`;
    lines.push(gbnfRule);
    ruleMap.set(lhs, gbnfRule);
  }

  // Add root rule
  const rootRule = `root ::= ${safeName(grammar.startSymbol)}`;
  lines.unshift(rootRule);

  return {
    grammar: lines.join('\n'),
    ruleMap,
  };
}

/**
 * Escape string for GBNF.
 */
function escapeGBNF(s: string): string {
  return s
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/\n/g, '\\n')
    .replace(/\r/g, '\\r')
    .replace(/\t/g, '\\t');
}

/**
 * Create common TypeScript PCFG rules.
 */
export function createTypeScriptPCFG(): PCFGGrammar {
  const rules: PCFGRule[] = [
    // Program structure
    { lhs: 'Program', rhs: ['Statement*'], probability: 1.0, count: 1 },

    // Statements
    { lhs: 'Statement', rhs: ['FunctionDeclaration'], probability: 0.3, count: 1 },
    { lhs: 'Statement', rhs: ['VariableDeclaration'], probability: 0.25, count: 1 },
    { lhs: 'Statement', rhs: ['ExpressionStatement'], probability: 0.2, count: 1 },
    { lhs: 'Statement', rhs: ['ReturnStatement'], probability: 0.15, count: 1 },
    { lhs: 'Statement', rhs: ['IfStatement'], probability: 0.1, count: 1 },

    // Function declaration
    { lhs: 'FunctionDeclaration', rhs: ['function', 'Identifier', 'ParameterList', 'TypeAnnotation', 'Block'], probability: 0.6, count: 1 },
    { lhs: 'FunctionDeclaration', rhs: ['const', 'Identifier', '=', 'ArrowFunction'], probability: 0.4, count: 1 },

    // Arrow function
    { lhs: 'ArrowFunction', rhs: ['ParameterList', 'TypeAnnotation', '=>', 'Block'], probability: 0.6, count: 1 },
    { lhs: 'ArrowFunction', rhs: ['ParameterList', 'TypeAnnotation', '=>', 'Expression'], probability: 0.4, count: 1 },

    // Parameters
    { lhs: 'ParameterList', rhs: ['(', 'Parameters?', ')'], probability: 1.0, count: 1 },
    { lhs: 'Parameters', rhs: ['Parameter'], probability: 0.4, count: 1 },
    { lhs: 'Parameters', rhs: ['Parameter', ',', 'Parameters'], probability: 0.6, count: 1 },
    { lhs: 'Parameter', rhs: ['Identifier', 'TypeAnnotation?'], probability: 1.0, count: 1 },

    // Type annotation
    { lhs: 'TypeAnnotation', rhs: [':', 'Type'], probability: 1.0, count: 1 },
    { lhs: 'Type', rhs: ['string'], probability: 0.2, count: 1 },
    { lhs: 'Type', rhs: ['number'], probability: 0.2, count: 1 },
    { lhs: 'Type', rhs: ['boolean'], probability: 0.15, count: 1 },
    { lhs: 'Type', rhs: ['void'], probability: 0.15, count: 1 },
    { lhs: 'Type', rhs: ['any'], probability: 0.1, count: 1 },
    { lhs: 'Type', rhs: ['Identifier'], probability: 0.2, count: 1 },

    // Block
    { lhs: 'Block', rhs: ['{', 'Statement*', '}'], probability: 1.0, count: 1 },

    // Variable declaration
    { lhs: 'VariableDeclaration', rhs: ['const', 'Identifier', 'TypeAnnotation?', '=', 'Expression', ';'], probability: 0.5, count: 1 },
    { lhs: 'VariableDeclaration', rhs: ['let', 'Identifier', 'TypeAnnotation?', '=', 'Expression', ';'], probability: 0.4, count: 1 },
    { lhs: 'VariableDeclaration', rhs: ['let', 'Identifier', 'TypeAnnotation?', ';'], probability: 0.1, count: 1 },

    // Expression statement
    { lhs: 'ExpressionStatement', rhs: ['Expression', ';'], probability: 1.0, count: 1 },

    // Return statement
    { lhs: 'ReturnStatement', rhs: ['return', 'Expression?', ';'], probability: 1.0, count: 1 },

    // If statement
    { lhs: 'IfStatement', rhs: ['if', '(', 'Expression', ')', 'Block'], probability: 0.6, count: 1 },
    { lhs: 'IfStatement', rhs: ['if', '(', 'Expression', ')', 'Block', 'else', 'Block'], probability: 0.4, count: 1 },

    // Expressions
    { lhs: 'Expression', rhs: ['Identifier'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['Literal'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['CallExpression'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['BinaryExpression'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['MemberExpression'], probability: 0.2, count: 1 },

    // Call expression
    { lhs: 'CallExpression', rhs: ['Expression', '(', 'Arguments?', ')'], probability: 1.0, count: 1 },
    { lhs: 'Arguments', rhs: ['Expression'], probability: 0.4, count: 1 },
    { lhs: 'Arguments', rhs: ['Expression', ',', 'Arguments'], probability: 0.6, count: 1 },

    // Binary expression
    { lhs: 'BinaryExpression', rhs: ['Expression', 'BinaryOperator', 'Expression'], probability: 1.0, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['+'], probability: 0.15, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['-'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['*'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['/'], probability: 0.05, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['==='], probability: 0.15, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['!=='], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['<'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['>'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['&&'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['||'], probability: 0.05, count: 1 },

    // Member expression
    { lhs: 'MemberExpression', rhs: ['Expression', '.', 'Identifier'], probability: 0.7, count: 1 },
    { lhs: 'MemberExpression', rhs: ['Expression', '[', 'Expression', ']'], probability: 0.3, count: 1 },

    // Literals
    { lhs: 'Literal', rhs: ['StringLiteral'], probability: 0.3, count: 1 },
    { lhs: 'Literal', rhs: ['NumberLiteral'], probability: 0.3, count: 1 },
    { lhs: 'Literal', rhs: ['BooleanLiteral'], probability: 0.2, count: 1 },
    { lhs: 'Literal', rhs: ['null'], probability: 0.1, count: 1 },
    { lhs: 'Literal', rhs: ['undefined'], probability: 0.1, count: 1 },

    // Terminals (patterns)
    { lhs: 'Identifier', rhs: [], probability: 1.0, count: 1, template: '__HOLE_IDENTIFIER__' },
    { lhs: 'StringLiteral', rhs: [], probability: 1.0, count: 1, template: '__HOLE_STRING__' },
    { lhs: 'NumberLiteral', rhs: [], probability: 1.0, count: 1, template: '__HOLE_NUMBER__' },
    { lhs: 'BooleanLiteral', rhs: ['true'], probability: 0.5, count: 1 },
    { lhs: 'BooleanLiteral', rhs: ['false'], probability: 0.5, count: 1 },
  ];

  const terminals = new Set([
    'function', 'const', 'let', 'return', 'if', 'else',
    '(', ')', '{', '}', '[', ']', ',', ';', ':', '=', '=>', '.',
    '+', '-', '*', '/', '===', '!==', '<', '>', '&&', '||',
    'string', 'number', 'boolean', 'void', 'any',
    'true', 'false', 'null', 'undefined',
  ]);

  const nonTerminals = new Set(rules.map(r => r.lhs));

  return {
    name: 'typescript-base',
    language: 'typescript',
    rules,
    terminals,
    nonTerminals,
    startSymbol: 'Program',
  };
}

/**
 * Create common Python PCFG rules.
 */
export function createPythonPCFG(): PCFGGrammar {
  const rules: PCFGRule[] = [
    // Program structure
    { lhs: 'Program', rhs: ['Statement*'], probability: 1.0, count: 1 },

    // Statements
    { lhs: 'Statement', rhs: ['FunctionDef'], probability: 0.3, count: 1 },
    { lhs: 'Statement', rhs: ['Assignment'], probability: 0.25, count: 1 },
    { lhs: 'Statement', rhs: ['ExpressionStatement'], probability: 0.2, count: 1 },
    { lhs: 'Statement', rhs: ['ReturnStatement'], probability: 0.15, count: 1 },
    { lhs: 'Statement', rhs: ['IfStatement'], probability: 0.1, count: 1 },

    // Function definition
    { lhs: 'FunctionDef', rhs: ['def', 'Identifier', 'ParameterList', 'ReturnType?', ':', 'Suite'], probability: 1.0, count: 1 },

    // Parameters
    { lhs: 'ParameterList', rhs: ['(', 'Parameters?', ')'], probability: 1.0, count: 1 },
    { lhs: 'Parameters', rhs: ['Parameter'], probability: 0.4, count: 1 },
    { lhs: 'Parameters', rhs: ['Parameter', ',', 'Parameters'], probability: 0.6, count: 1 },
    { lhs: 'Parameter', rhs: ['Identifier', 'TypeHint?'], probability: 1.0, count: 1 },

    // Type hints
    { lhs: 'TypeHint', rhs: [':', 'Type'], probability: 1.0, count: 1 },
    { lhs: 'ReturnType', rhs: ['->', 'Type'], probability: 1.0, count: 1 },
    { lhs: 'Type', rhs: ['str'], probability: 0.2, count: 1 },
    { lhs: 'Type', rhs: ['int'], probability: 0.2, count: 1 },
    { lhs: 'Type', rhs: ['float'], probability: 0.1, count: 1 },
    { lhs: 'Type', rhs: ['bool'], probability: 0.15, count: 1 },
    { lhs: 'Type', rhs: ['None'], probability: 0.15, count: 1 },
    { lhs: 'Type', rhs: ['Identifier'], probability: 0.2, count: 1 },

    // Suite (indented block)
    { lhs: 'Suite', rhs: ['NEWLINE', 'INDENT', 'Statement+', 'DEDENT'], probability: 1.0, count: 1 },

    // Assignment
    { lhs: 'Assignment', rhs: ['Identifier', 'TypeHint?', '=', 'Expression'], probability: 1.0, count: 1 },

    // Return statement
    { lhs: 'ReturnStatement', rhs: ['return', 'Expression?'], probability: 1.0, count: 1 },

    // If statement
    { lhs: 'IfStatement', rhs: ['if', 'Expression', ':', 'Suite'], probability: 0.6, count: 1 },
    { lhs: 'IfStatement', rhs: ['if', 'Expression', ':', 'Suite', 'else', ':', 'Suite'], probability: 0.4, count: 1 },

    // Expressions
    { lhs: 'Expression', rhs: ['Identifier'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['Literal'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['CallExpression'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['BinaryExpression'], probability: 0.2, count: 1 },
    { lhs: 'Expression', rhs: ['AttributeAccess'], probability: 0.2, count: 1 },

    // Call expression
    { lhs: 'CallExpression', rhs: ['Expression', '(', 'Arguments?', ')'], probability: 1.0, count: 1 },

    // Binary expression
    { lhs: 'BinaryExpression', rhs: ['Expression', 'BinaryOperator', 'Expression'], probability: 1.0, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['+'], probability: 0.15, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['-'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['*'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['/'], probability: 0.05, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['=='], probability: 0.15, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['!='], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['<'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['>'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['and'], probability: 0.1, count: 1 },
    { lhs: 'BinaryOperator', rhs: ['or'], probability: 0.05, count: 1 },

    // Attribute access
    { lhs: 'AttributeAccess', rhs: ['Expression', '.', 'Identifier'], probability: 1.0, count: 1 },

    // Literals
    { lhs: 'Literal', rhs: ['StringLiteral'], probability: 0.3, count: 1 },
    { lhs: 'Literal', rhs: ['NumberLiteral'], probability: 0.3, count: 1 },
    { lhs: 'Literal', rhs: ['True'], probability: 0.1, count: 1 },
    { lhs: 'Literal', rhs: ['False'], probability: 0.1, count: 1 },
    { lhs: 'Literal', rhs: ['None'], probability: 0.2, count: 1 },

    // Terminals
    { lhs: 'Identifier', rhs: [], probability: 1.0, count: 1, template: '__HOLE_IDENTIFIER__' },
    { lhs: 'StringLiteral', rhs: [], probability: 1.0, count: 1, template: '__HOLE_STRING__' },
    { lhs: 'NumberLiteral', rhs: [], probability: 1.0, count: 1, template: '__HOLE_NUMBER__' },
  ];

  const terminals = new Set([
    'def', 'return', 'if', 'else', 'and', 'or', 'not',
    '(', ')', '[', ']', '{', '}', ',', ':', '=', '->', '.',
    '+', '-', '*', '/', '==', '!=', '<', '>',
    'str', 'int', 'float', 'bool', 'None', 'True', 'False',
    'NEWLINE', 'INDENT', 'DEDENT',
  ]);

  const nonTerminals = new Set(rules.map(r => r.lhs));

  return {
    name: 'python-base',
    language: 'python',
    rules,
    terminals,
    nonTerminals,
    startSymbol: 'Program',
  };
}
