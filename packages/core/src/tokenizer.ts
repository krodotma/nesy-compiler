export interface TokenizerConfig {
  splitCamelCase: boolean;
  splitSnakeCase: boolean;
  stripSyntax: boolean;
  lowercase: boolean;
}

const DEFAULT_CONFIG: TokenizerConfig = {
  splitCamelCase: true,
  splitSnakeCase: true,
  stripSyntax: true,
  lowercase: true,
};

/**
 * Tokenizes a code string based on the provided configuration.
 *
 * @param code The source code string to tokenize.
 * @param config Configuration options for the tokenizer.
 * @returns An array of string tokens.
 */
export function tokenize(code: string, config?: Partial<TokenizerConfig>): string[] {
  const cfg = { ...DEFAULT_CONFIG, ...config };
  let processed = code;

  // 1. Strip syntax: Replace non-alphanumeric characters with spaces.
  // We temporarily preserve underscores if snake_case splitting is enabled or if we haven't processed them yet,
  // but the requirement "Strips syntax tokens" usually targets punctuation like brackets, dots, etc.
  // To ensure safely handling snake_case in step 2, we allow underscores here.
  if (cfg.stripSyntax) {
    processed = processed.replace(/[^a-zA-Z0-9_]+/g, ' ');
  }

  // 2. Split snake_case: Replace underscores with spaces.
  if (cfg.splitSnakeCase) {
    processed = processed.replace(/_/g, ' ');
  }

  // 3. Split CamelCase: Insert spaces at boundaries.
  if (cfg.splitCamelCase) {
    processed = processed
      // Handle "camelCase" -> "camel Case"
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      // Handle "XMLHttp" -> "XML Http"
      .replace(/([A-Z]+)([A-Z][a-z])/g, '$1 $2');
  }

  // 4. Normalize to lowercase.
  if (cfg.lowercase) {
    processed = processed.toLowerCase();
  }

  // Final whitespace split and filter empty tokens.
  return processed
    .split(/\s+/)
    .filter((token) => token.length > 0);
}
