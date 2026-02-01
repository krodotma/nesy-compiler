/**
 * @ark/core/pentad - The Five Coordinates (Positive Intent / The Key)
 *
 * The Pentad represents the 5 positive coordinates that justify any action:
 * - WHY (Etymon): Semantic root/law justifying action
 * - WHERE (Locus): Physical binding site (file/hash)
 * - WHO (Lineage): Agent identity and authority chain
 * - WHEN (Kairos): Opportune moment and budget constraints
 * - WHAT (Artifact): Concrete output/phenomenon
 *
 * @module
 */

/**
 * Authority levels in the Ring hierarchy
 */
export type AuthorityLevel = 'kernel' | 'service' | 'user' | 'ephemeral';

/**
 * Budget constraints for actions
 */
export interface Budget {
  /** Maximum tokens for LLM calls */
  tokens?: number;
  /** Maximum time in milliseconds */
  timeMs?: number;
  /** Maximum number of tool calls */
  toolCalls?: number;
  /** Maximum file operations */
  fileOps?: number;
}

/**
 * WHY - Etymon: The semantic root/law justifying the action
 */
export interface Etymon {
  /** Unique identifier for the etymon */
  id: string;
  /** Human-readable name */
  name: string;
  /** Reference to spec or axiom (URN format) */
  specRef?: string;
  /** The justification text */
  justification: string;
  /** Parent etymon for derivation chains */
  parent?: string;
}

/**
 * WHERE - Locus: The physical binding site
 */
export interface Locus {
  /** File path or resource path */
  path: string;
  /** SHA256 hash for content-addressed storage */
  hash?: string;
  /** Line number range if applicable */
  lineRange?: [number, number];
  /** Rhizome artifact ID */
  rhizomeId?: string;
}

/**
 * WHO - Lineage: Agent identity and authority chain
 */
export interface Lineage {
  /** Agent identifier */
  agent: string;
  /** Authority level in ring hierarchy */
  authority: AuthorityLevel;
  /** Parent agent that spawned this one */
  parent?: string;
  /** Session identifier */
  sessionId?: string;
  /** Cryptographic signature (PQC ML-DSA-65) */
  signature?: string;
}

/**
 * WHEN - Kairos: The opportune moment and constraints
 */
export interface Kairos {
  /** Unix timestamp in milliseconds */
  timestamp: number;
  /** Budget constraints */
  budget: Budget;
  /** TTL in seconds (0 = immortal) */
  ttl?: number;
  /** Iteration/turn number in conversation */
  iteration?: number;
  /** Protocol version (DKIN v29) */
  protocolVersion?: string;
}

/**
 * WHAT - Artifact: The concrete output/phenomenon
 */
export interface Artifact {
  /** Type discriminator */
  type: 'code' | 'document' | 'event' | 'binary' | 'structured';
  /** MIME type */
  mimeType?: string;
  /** Size in bytes */
  size?: number;
  /** The actual data (typed by type field) */
  data: unknown;
  /** Metadata key-value pairs */
  metadata?: Record<string, unknown>;
}

/**
 * The Pentad - Five coordinates that justify an action
 *
 * @example
 * ```typescript
 * const pentad: Pentad = {
 *   etymon: {
 *     id: 'fix-auth-bug',
 *     name: 'Authentication Fix',
 *     justification: 'User cannot login with valid credentials'
 *   },
 *   locus: {
 *     path: '/src/auth/login.ts',
 *     hash: 'sha256:abc123...',
 *     lineRange: [42, 67]
 *   },
 *   lineage: {
 *     agent: 'claude-opus-4.5',
 *     authority: 'service',
 *     sessionId: 'sess-12345'
 *   },
 *   kairos: {
 *     timestamp: Date.now(),
 *     budget: { tokens: 10000, timeMs: 60000 },
 *     protocolVersion: 'DKIN v29'
 *   },
 *   artifact: {
 *     type: 'code',
 *     mimeType: 'text/typescript',
 *     data: '// Fixed authentication logic...'
 *   }
 * };
 * ```
 */
export interface Pentad {
  /** WHY - The justification */
  etymon: Etymon;
  /** WHERE - The binding site */
  locus: Locus;
  /** WHO - The agent identity */
  lineage: Lineage;
  /** WHEN - The timing and constraints */
  kairos: Kairos;
  /** WHAT - The output */
  artifact: Artifact;

  // Compatibility aliases for simpler access
  /** @deprecated Use etymon - WHY alias */
  why?: Etymon;
  /** @deprecated Use locus - WHERE alias */
  where?: Locus;
  /** @deprecated Use lineage - WHO alias */
  who?: Lineage;
  /** @deprecated Use kairos - WHEN alias */
  when?: Kairos;
  /** @deprecated Use artifact - WHAT alias */
  what?: Artifact;
}

/**
 * Create a minimal Pentad with required fields
 */
export function createPentad(
  justification: string,
  path: string,
  agent: string,
  data: unknown
): Pentad {
  return {
    etymon: {
      id: crypto.randomUUID(),
      name: justification.slice(0, 50),
      justification,
    },
    locus: {
      path,
    },
    lineage: {
      agent,
      authority: 'service',
    },
    kairos: {
      timestamp: Date.now(),
      budget: {},
      protocolVersion: 'DKIN v29',
    },
    artifact: {
      type: 'structured',
      data,
    },
  };
}

/**
 * Validate that a Pentad has all required fields
 */
export function validatePentad(pentad: Pentad): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  if (!pentad.etymon?.justification) {
    errors.push('Etymon must have justification (WHY)');
  }
  if (!pentad.locus?.path) {
    errors.push('Locus must have path (WHERE)');
  }
  if (!pentad.lineage?.agent) {
    errors.push('Lineage must have agent (WHO)');
  }
  if (!pentad.kairos?.timestamp) {
    errors.push('Kairos must have timestamp (WHEN)');
  }
  if (pentad.artifact?.data === undefined) {
    errors.push('Artifact must have data (WHAT)');
  }

  return { valid: errors.length === 0, errors };
}
