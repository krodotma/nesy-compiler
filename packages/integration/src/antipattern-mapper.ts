/**
 * AntipatternMapper: Map Linter Rules to NeSy Antipatterns
 *
 * Step 12 of NeSy Evolution Phase 1.5 (Sensor Fusion)
 *
 * Auto-tags graph nodes with antipattern labels based on linter violations.
 * Maps ESLint/Ruff/TSC rules to the 50 Agentic Thrash Taxonomy antipatterns.
 *
 * Reference: AGENTIC_THRASH_TAXONOMY.md
 */

import type { LintViolation, Severity } from './linter-bridge';

export interface Antipattern {
  id: number;
  name: string;
  category: AntipatternCategory;
  description: string;
  severity: Severity;
  linkedRules: string[];
}

export type AntipatternCategory =
  | 'code-quality'
  | 'architecture'
  | 'performance'
  | 'security'
  | 'maintainability'
  | 'testing'
  | 'documentation'
  | 'agent-specific';

export interface AntipatternMatch {
  antipatternId: number;
  antipatternName: string;
  ruleId: string;
  confidence: number;  // 0-1 match confidence
  violation: LintViolation;
}

/**
 * The Agentic Thrash Taxonomy - 50 Antipatterns
 * Derived from AGENTIC_THRASH_TAXONOMY.md
 */
export const ANTIPATTERNS: Antipattern[] = [
  // Code Quality (1-10)
  {
    id: 1,
    name: 'Dead Code',
    category: 'code-quality',
    description: 'Unreachable or unused code that should be removed',
    severity: 'warning',
    linkedRules: ['no-unreachable', 'no-unused-vars', 'no-unused-expressions', 'F401', 'F841'],
  },
  {
    id: 2,
    name: 'Phantom Import',
    category: 'code-quality',
    description: 'Import statement for module that does not exist or is not used',
    severity: 'error',
    linkedRules: ['no-unused-vars', 'import/no-unresolved', 'F401', 'F811', 'TS2307'],
  },
  {
    id: 3,
    name: 'Type Mismatch',
    category: 'code-quality',
    description: 'Incompatible types in assignment or function call',
    severity: 'error',
    linkedRules: ['TS2322', 'TS2345', 'TS2339', 'TS2551', 'E1101'],
  },
  {
    id: 4,
    name: 'Implicit Any',
    category: 'code-quality',
    description: 'Missing type annotations leading to implicit any',
    severity: 'warning',
    linkedRules: ['@typescript-eslint/no-explicit-any', 'TS7006', 'TS7005'],
  },
  {
    id: 5,
    name: 'Magic Number',
    category: 'code-quality',
    description: 'Hardcoded numeric literals without explanation',
    severity: 'info',
    linkedRules: ['no-magic-numbers', '@typescript-eslint/no-magic-numbers'],
  },
  {
    id: 6,
    name: 'Complex Expression',
    category: 'code-quality',
    description: 'Overly complex conditional or nested expression',
    severity: 'warning',
    linkedRules: ['complexity', 'max-depth', 'C901'],
  },
  {
    id: 7,
    name: 'Long Function',
    category: 'code-quality',
    description: 'Function exceeds reasonable length threshold',
    severity: 'warning',
    linkedRules: ['max-lines-per-function', 'max-statements'],
  },
  {
    id: 8,
    name: 'Deep Nesting',
    category: 'code-quality',
    description: 'Excessive indentation levels indicating complex control flow',
    severity: 'warning',
    linkedRules: ['max-depth', 'max-nested-callbacks'],
  },
  {
    id: 9,
    name: 'Duplicate Code',
    category: 'code-quality',
    description: 'Similar code blocks that should be extracted',
    severity: 'info',
    linkedRules: ['no-duplicate-case', 'no-dupe-keys', 'no-dupe-args'],
  },
  {
    id: 10,
    name: 'Inconsistent Naming',
    category: 'code-quality',
    description: 'Variable/function names that do not follow conventions',
    severity: 'info',
    linkedRules: ['camelcase', '@typescript-eslint/naming-convention', 'N801', 'N802'],
  },

  // Architecture (11-20)
  {
    id: 11,
    name: 'Circular Dependency',
    category: 'architecture',
    description: 'Modules that import each other creating a cycle',
    severity: 'error',
    linkedRules: ['import/no-cycle', 'import/no-self-import'],
  },
  {
    id: 12,
    name: 'God Object',
    category: 'architecture',
    description: 'Class or module that knows too much or does too much',
    severity: 'warning',
    linkedRules: ['max-classes-per-file', 'max-lines'],
  },
  {
    id: 13,
    name: 'Tight Coupling',
    category: 'architecture',
    description: 'Components with excessive dependencies on each other',
    severity: 'warning',
    linkedRules: ['import/max-dependencies'],
  },
  {
    id: 14,
    name: 'Leaky Abstraction',
    category: 'architecture',
    description: 'Internal implementation details exposed through interface',
    severity: 'warning',
    linkedRules: ['@typescript-eslint/explicit-module-boundary-types'],
  },
  {
    id: 15,
    name: 'Wrong Layer',
    category: 'architecture',
    description: 'Business logic in presentation layer or vice versa',
    severity: 'warning',
    linkedRules: ['import/no-restricted-paths'],
  },
  {
    id: 16,
    name: 'Missing Interface',
    category: 'architecture',
    description: 'Concrete implementation without interface abstraction',
    severity: 'info',
    linkedRules: ['@typescript-eslint/consistent-type-definitions'],
  },
  {
    id: 17,
    name: 'Over-Engineering',
    category: 'architecture',
    description: 'Unnecessary abstraction or complexity for simple task',
    severity: 'info',
    linkedRules: [],  // Hard to detect via linting
  },
  {
    id: 18,
    name: 'Under-Engineering',
    category: 'architecture',
    description: 'Missing necessary abstraction leading to duplication',
    severity: 'info',
    linkedRules: [],
  },
  {
    id: 19,
    name: 'Broken Encapsulation',
    category: 'architecture',
    description: 'Direct access to internal state instead of through methods',
    severity: 'warning',
    linkedRules: ['@typescript-eslint/member-ordering', 'no-underscore-dangle'],
  },
  {
    id: 20,
    name: 'Inappropriate Intimacy',
    category: 'architecture',
    description: 'Classes that know too much about each other internals',
    severity: 'warning',
    linkedRules: [],
  },

  // Performance (21-25)
  {
    id: 21,
    name: 'N+1 Query',
    category: 'performance',
    description: 'Database query inside a loop causing performance issues',
    severity: 'error',
    linkedRules: [],  // Requires semantic analysis
  },
  {
    id: 22,
    name: 'Premature Optimization',
    category: 'performance',
    description: 'Optimizations that hurt readability without measured benefit',
    severity: 'info',
    linkedRules: [],
  },
  {
    id: 23,
    name: 'Memory Leak',
    category: 'performance',
    description: 'Resource not properly released leading to memory growth',
    severity: 'error',
    linkedRules: ['no-async-promise-executor'],
  },
  {
    id: 24,
    name: 'Blocking Call',
    category: 'performance',
    description: 'Synchronous I/O or heavy computation blocking event loop',
    severity: 'warning',
    linkedRules: ['no-sync', 'require-await'],
  },
  {
    id: 25,
    name: 'Unbounded Growth',
    category: 'performance',
    description: 'Data structure that grows without limits (caches, logs)',
    severity: 'warning',
    linkedRules: [],
  },

  // Security (26-30)
  {
    id: 26,
    name: 'Injection Vulnerability',
    category: 'security',
    description: 'User input used in query/command without sanitization',
    severity: 'error',
    linkedRules: ['no-eval', 'no-new-func', 'security/detect-eval-with-expression', 'S501'],
  },
  {
    id: 27,
    name: 'Hardcoded Secret',
    category: 'security',
    description: 'API keys, passwords, or tokens in source code',
    severity: 'error',
    linkedRules: ['no-secrets/no-secrets', 'security/detect-hardcoded-credentials'],
  },
  {
    id: 28,
    name: 'Insecure Randomness',
    category: 'security',
    description: 'Using Math.random() for security-sensitive operations',
    severity: 'error',
    linkedRules: ['security/detect-pseudoRandomBytes'],
  },
  {
    id: 29,
    name: 'Path Traversal',
    category: 'security',
    description: 'User-controlled path without validation',
    severity: 'error',
    linkedRules: ['security/detect-non-literal-fs-filename'],
  },
  {
    id: 30,
    name: 'Missing Auth Check',
    category: 'security',
    description: 'Sensitive operation without authentication/authorization',
    severity: 'error',
    linkedRules: [],  // Requires semantic analysis
  },

  // Maintainability (31-40)
  {
    id: 31,
    name: 'Missing Error Handling',
    category: 'maintainability',
    description: 'Operations that can fail without try/catch or error check',
    severity: 'warning',
    linkedRules: ['no-empty', '@typescript-eslint/no-floating-promises', 'E722'],
  },
  {
    id: 32,
    name: 'Silent Failure',
    category: 'maintainability',
    description: 'Catching error but not logging or handling it',
    severity: 'warning',
    linkedRules: ['no-empty', 'no-useless-catch'],
  },
  {
    id: 33,
    name: 'Commented Code',
    category: 'maintainability',
    description: 'Dead code left as comments instead of being deleted',
    severity: 'info',
    linkedRules: ['no-warning-comments', 'spaced-comment'],
  },
  {
    id: 34,
    name: 'Missing Null Check',
    category: 'maintainability',
    description: 'Accessing property on potentially null/undefined value',
    severity: 'error',
    linkedRules: ['@typescript-eslint/no-non-null-assertion', 'TS2532', 'TS2531'],
  },
  {
    id: 35,
    name: 'Inconsistent Return',
    category: 'maintainability',
    description: 'Function with inconsistent return statements',
    severity: 'warning',
    linkedRules: ['consistent-return', '@typescript-eslint/explicit-function-return-type'],
  },
  {
    id: 36,
    name: 'Boolean Trap',
    category: 'maintainability',
    description: 'Function with boolean parameter that changes behavior',
    severity: 'info',
    linkedRules: ['no-boolean-trap'],
  },
  {
    id: 37,
    name: 'Mutation of Input',
    category: 'maintainability',
    description: 'Function that modifies its input parameters',
    severity: 'warning',
    linkedRules: ['no-param-reassign'],
  },
  {
    id: 38,
    name: 'Global State',
    category: 'maintainability',
    description: 'Using global variables or singletons for state',
    severity: 'warning',
    linkedRules: ['no-global-assign', 'no-implicit-globals'],
  },
  {
    id: 39,
    name: 'Temporal Coupling',
    category: 'maintainability',
    description: 'Methods that must be called in specific order',
    severity: 'info',
    linkedRules: [],
  },
  {
    id: 40,
    name: 'Feature Envy',
    category: 'maintainability',
    description: 'Method that uses more features of another class',
    severity: 'info',
    linkedRules: [],
  },

  // Agent-Specific (41-50)
  {
    id: 41,
    name: 'Hallucinated Import',
    category: 'agent-specific',
    description: 'AI generated import for non-existent module',
    severity: 'error',
    linkedRules: ['import/no-unresolved', 'TS2307', 'F401'],
  },
  {
    id: 42,
    name: 'Incomplete Implementation',
    category: 'agent-specific',
    description: 'TODO/FIXME comments or placeholder implementations',
    severity: 'warning',
    linkedRules: ['no-warning-comments'],
  },
  {
    id: 43,
    name: 'Copy-Paste Error',
    category: 'agent-specific',
    description: 'Duplicated code with inconsistent modifications',
    severity: 'warning',
    linkedRules: [],
  },
  {
    id: 44,
    name: 'Context Confusion',
    category: 'agent-specific',
    description: 'Code that mixes concerns from different files/modules',
    severity: 'warning',
    linkedRules: [],
  },
  {
    id: 45,
    name: 'Stale Reference',
    category: 'agent-specific',
    description: 'Reference to API or pattern that has changed',
    severity: 'error',
    linkedRules: ['deprecation/deprecation', '@typescript-eslint/no-deprecated'],
  },
  {
    id: 46,
    name: 'Over-Commenting',
    category: 'agent-specific',
    description: 'Excessive or redundant comments that add no value',
    severity: 'info',
    linkedRules: [],
  },
  {
    id: 47,
    name: 'Under-Commenting',
    category: 'agent-specific',
    description: 'Complex logic without explanatory comments',
    severity: 'info',
    linkedRules: [],
  },
  {
    id: 48,
    name: 'Style Inconsistency',
    category: 'agent-specific',
    description: 'Code style that differs from project conventions',
    severity: 'info',
    linkedRules: ['prettier/prettier', 'indent', 'quotes', 'semi'],
  },
  {
    id: 49,
    name: 'Regression Introduction',
    category: 'agent-specific',
    description: 'Change that breaks existing functionality',
    severity: 'error',
    linkedRules: [],  // Requires test analysis
  },
  {
    id: 50,
    name: 'Scope Creep',
    category: 'agent-specific',
    description: 'Changes beyond what was requested',
    severity: 'warning',
    linkedRules: [],  // Requires intent analysis
  },
];

// Build reverse index: rule -> antipatterns
const ruleToAntipatterns = new Map<string, number[]>();
for (const ap of ANTIPATTERNS) {
  for (const rule of ap.linkedRules) {
    const existing = ruleToAntipatterns.get(rule) || [];
    existing.push(ap.id);
    ruleToAntipatterns.set(rule, existing);
  }
}

/**
 * Map a lint violation to antipatterns.
 */
export function mapViolationToAntipatterns(
  violation: LintViolation
): AntipatternMatch[] {
  const matches: AntipatternMatch[] = [];
  const antipatternIds = ruleToAntipatterns.get(violation.ruleId);

  if (antipatternIds) {
    for (const id of antipatternIds) {
      const ap = ANTIPATTERNS.find(a => a.id === id);
      if (ap) {
        matches.push({
          antipatternId: ap.id,
          antipatternName: ap.name,
          ruleId: violation.ruleId,
          confidence: calculateConfidence(violation, ap),
          violation,
        });
      }
    }
  }

  return matches;
}

/**
 * Map all violations in a file to antipatterns.
 */
export function mapFileViolations(
  violations: LintViolation[]
): Map<number, AntipatternMatch[]> {
  const byAntipattern = new Map<number, AntipatternMatch[]>();

  for (const v of violations) {
    const matches = mapViolationToAntipatterns(v);
    for (const match of matches) {
      const existing = byAntipattern.get(match.antipatternId) || [];
      existing.push(match);
      byAntipattern.set(match.antipatternId, existing);
    }
  }

  return byAntipattern;
}

/**
 * Get antipattern summary for a file (for graph node properties).
 */
export function getAntipatternSummary(
  violations: LintViolation[]
): {
  antipatternIds: number[];
  antipatternNames: string[];
  totalMatches: number;
  bySeverity: Record<Severity, number>;
  byCategory: Record<AntipatternCategory, number>;
} {
  const byAntipattern = mapFileViolations(violations);
  const antipatternIds = Array.from(byAntipattern.keys());
  const antipatternNames = antipatternIds.map(id =>
    ANTIPATTERNS.find(a => a.id === id)?.name || `Unknown-${id}`
  );

  const bySeverity: Record<Severity, number> = {
    error: 0,
    warning: 0,
    info: 0,
    hint: 0,
  };

  const byCategory: Record<AntipatternCategory, number> = {
    'code-quality': 0,
    'architecture': 0,
    'performance': 0,
    'security': 0,
    'maintainability': 0,
    'testing': 0,
    'documentation': 0,
    'agent-specific': 0,
  };

  let totalMatches = 0;

  for (const [id, matches] of byAntipattern) {
    const ap = ANTIPATTERNS.find(a => a.id === id);
    if (ap) {
      bySeverity[ap.severity] += matches.length;
      byCategory[ap.category] += matches.length;
      totalMatches += matches.length;
    }
  }

  return {
    antipatternIds,
    antipatternNames,
    totalMatches,
    bySeverity,
    byCategory,
  };
}

/**
 * Calculate confidence that a violation matches an antipattern.
 */
function calculateConfidence(
  violation: LintViolation,
  antipattern: Antipattern
): number {
  let confidence = 0.7;  // Base confidence for rule match

  // Higher confidence if severity matches
  if (violation.severity === antipattern.severity) {
    confidence += 0.2;
  }

  // Higher confidence for exact rule match (vs prefix match)
  if (antipattern.linkedRules.includes(violation.ruleId)) {
    confidence += 0.1;
  }

  return Math.min(1, confidence);
}

/**
 * Get antipattern by ID.
 */
export function getAntipattern(id: number): Antipattern | undefined {
  return ANTIPATTERNS.find(a => a.id === id);
}

/**
 * Get all antipatterns for a category.
 */
export function getAntipatternsByCategory(
  category: AntipatternCategory
): Antipattern[] {
  return ANTIPATTERNS.filter(a => a.category === category);
}
