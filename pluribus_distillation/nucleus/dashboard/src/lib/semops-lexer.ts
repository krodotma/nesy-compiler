/**
 * Semantic Operators Lexer for Dashboard WebUI
 * =============================================
 * Client-side lexer for syntax highlighting and autocomplete in terminal views.
 */

export enum TokenType {
  OPERATOR = 'OPERATOR',
  COMMAND = 'COMMAND',
  FLAG = 'FLAG',
  VALUE = 'VALUE',
  PROMPT = 'PROMPT',
  WHITESPACE = 'WHITESPACE',
  USER_OP = 'USER_OP',
  UNKNOWN = 'UNKNOWN',
}

export interface Token {
  type: TokenType;
  value: string;
  start: number;
  end: number;
  metadata?: Record<string, unknown>;
}

export interface OperatorDef {
  id: string;
  name: string;
  domain: string;
  category: string;
  description: string;
  aliases: string[];
  options?: Record<string, string>;
  user_defined?: boolean;
}

export interface SemopsSchema {
  operators: Record<string, OperatorDef>;
  commands: string[];
  alias_map: Record<string, string>;
}

// Token colors for syntax highlighting (CSS classes)
export const TOKEN_COLORS: Record<TokenType, string> = {
  [TokenType.OPERATOR]: 'text-green-400 font-bold',
  [TokenType.COMMAND]: 'text-cyan-400 font-bold',
  [TokenType.FLAG]: 'text-yellow-400',
  [TokenType.VALUE]: 'text-gray-300',
  [TokenType.PROMPT]: 'text-gray-100',
  [TokenType.WHITESPACE]: '',
  [TokenType.USER_OP]: 'text-purple-400 font-bold',
  [TokenType.UNKNOWN]: 'text-red-400',
};

/**
 * Semantic Operators Lexer for client-side token recognition.
 */
export class SemopsLexer {
  private operators: Map<string, OperatorDef> = new Map();
  private aliasMap: Map<string, string> = new Map();
  private commands: Set<string> = new Set();

  constructor(schema?: SemopsSchema) {
    if (schema) {
      this.loadSchema(schema);
    }
  }

  /**
   * Load operator schema from JSON.
   */
  loadSchema(schema: SemopsSchema): void {
    this.operators.clear();
    this.aliasMap.clear();
    this.commands.clear();

    for (const [id, op] of Object.entries(schema.operators)) {
      this.operators.set(id, op);
      for (const alias of op.aliases) {
        this.aliasMap.set(alias.toLowerCase(), id);
      }
    }

    for (const cmd of schema.commands) {
      this.commands.add(cmd);
    }
  }

  /**
   * Tokenize input text.
   */
  tokenize(text: string): Token[] {
    const tokens: Token[] = [];
    let pos = 0;

    while (pos < text.length) {
      // Whitespace
      if (/\s/.test(text[pos])) {
        const start = pos;
        while (pos < text.length && /\s/.test(text[pos])) pos++;
        tokens.push({ type: TokenType.WHITESPACE, value: text.slice(start, pos), start, end: pos });
        continue;
      }

      // Slash command
      if (text[pos] === '/') {
        const token = this.matchCommand(text, pos);
        if (token) {
          tokens.push(token);
          pos = token.end;
          continue;
        }
      }

      // Flag
      if (text[pos] === '-') {
        const token = this.matchFlag(text, pos);
        if (token) {
          tokens.push(token);
          pos = token.end;
          continue;
        }
      }

      // Operator (at start or after whitespace)
      if (pos === 0 || (tokens.length > 0 && tokens[tokens.length - 1].type === TokenType.WHITESPACE)) {
        const token = this.matchOperator(text, pos);
        if (token) {
          tokens.push(token);
          pos = token.end;
          continue;
        }
      }

      // Default: value/prompt word
      const start = pos;
      while (pos < text.length && !/\s/.test(text[pos]) && text[pos] !== '-') pos++;
      const word = text.slice(start, pos);

      const lastToken = tokens[tokens.length - 1];
      if (lastToken && [TokenType.FLAG, TokenType.COMMAND, TokenType.OPERATOR].includes(lastToken.type)) {
        tokens.push({ type: TokenType.VALUE, value: word, start, end: pos });
      } else {
        tokens.push({ type: TokenType.PROMPT, value: word, start, end: pos });
      }
    }

    return tokens;
  }

  private matchCommand(text: string, pos: number): Token | null {
    if (text[pos] !== '/') return null;

    let end = pos + 1;
    while (end < text.length && /[\w\-_]/.test(text[end])) end++;

    const cmd = text.slice(pos + 1, end).toLowerCase();
    return {
      type: TokenType.COMMAND,
      value: text.slice(pos, end),
      start: pos,
      end,
      metadata: { command: cmd, operator_id: this.aliasMap.get(cmd) },
    };
  }

  private matchFlag(text: string, pos: number): Token | null {
    if (text[pos] !== '-') return null;

    let end = pos + 1;
    if (end < text.length && text[end] === '-') end++;
    while (end < text.length && /[\w\-_]/.test(text[end])) end++;

    if (end > pos + 1) {
      return { type: TokenType.FLAG, value: text.slice(pos, end), start: pos, end };
    }
    return null;
  }

  private matchOperator(text: string, pos: number): Token | null {
    // Try to match operators including multi-word ones like "checking in"
    let end = pos;
    let bestMatch: { end: number; opId: string } | null = null;

    while (end < text.length) {
      if (text[end] === ' ') {
        // Check if there's a continuation
        let nextEnd = end + 1;
        while (nextEnd < text.length && /\w/.test(text[nextEnd])) nextEnd++;
        const candidate = text.slice(pos, nextEnd).toLowerCase().trim();
        if (this.aliasMap.has(candidate)) {
          bestMatch = { end: nextEnd, opId: this.aliasMap.get(candidate)! };
          end = nextEnd;
          continue;
        }
        break;
      }
      if (!/[\w\-_]/.test(text[end])) break;
      end++;

      const word = text.slice(pos, end).toLowerCase();
      if (this.aliasMap.has(word)) {
        bestMatch = { end, opId: this.aliasMap.get(word)! };
      }
    }

    if (bestMatch) {
      const opDef = this.operators.get(bestMatch.opId);
      return {
        type: opDef?.user_defined ? TokenType.USER_OP : TokenType.OPERATOR,
        value: text.slice(pos, bestMatch.end),
        start: pos,
        end: bestMatch.end,
        metadata: { operator_id: bestMatch.opId, operator: opDef },
      };
    }

    return null;
  }

  /**
   * Get completions for a prefix.
   */
  complete(prefix: string): Array<{ value: string; description: string }> {
    const prefixLower = prefix.toLowerCase();
    const completions: Array<{ value: string; description: string }> = [];

    if (prefix.startsWith('/')) {
      const cmdPrefix = prefix.slice(1).toLowerCase();
      for (const cmd of this.commands) {
        if (cmd.startsWith(cmdPrefix)) {
          completions.push({ value: `/${cmd}`, description: `Command: ${cmd}` });
        }
      }
      for (const [alias, opId] of this.aliasMap) {
        if (alias.startsWith(cmdPrefix)) {
          const op = this.operators.get(opId);
          completions.push({ value: `/${alias}`, description: op?.description.slice(0, 50) || '' });
        }
      }
    } else {
      for (const [alias, opId] of this.aliasMap) {
        if (alias.startsWith(prefixLower)) {
          const op = this.operators.get(opId);
          completions.push({ value: alias, description: op?.description.slice(0, 50) || '' });
        }
      }
    }

    return completions.sort((a, b) => a.value.localeCompare(b.value));
  }

  /**
   * Get HTML-highlighted version of text.
   */
  highlight(text: string): string {
    const tokens = this.tokenize(text);
    return tokens
      .map((token) => {
        const className = TOKEN_COLORS[token.type];
        if (!className) return escapeHtml(token.value);
        return `<span class="${className}">${escapeHtml(token.value)}</span>`;
      })
      .join('');
  }

  /**
   * Get operator by name or alias.
   */
  getOperator(nameOrAlias: string): OperatorDef | undefined {
    const opId = this.aliasMap.get(nameOrAlias.toLowerCase());
    if (opId) return this.operators.get(opId);
    return this.operators.get(nameOrAlias);
  }

  /**
   * List all operators.
   */
  listOperators(): OperatorDef[] {
    return Array.from(this.operators.values()).sort((a, b) => a.name.localeCompare(b.name));
  }
}

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Default semops schema (will be loaded from server)
let defaultLexer: SemopsLexer | null = null;

/**
 * Get or create the default lexer instance.
 */
export async function getDefaultLexer(fetchSchema?: () => Promise<SemopsSchema>): Promise<SemopsLexer> {
  if (!defaultLexer) {
    defaultLexer = new SemopsLexer();
    if (fetchSchema) {
      try {
        const schema = await fetchSchema();
        defaultLexer.loadSchema(schema);
      } catch (e) {
        console.warn('Failed to load semops schema:', e);
      }
    }
  }
  return defaultLexer;
}
