/**
 * @ark/core/holon - The Holon (Lock & Key Mechanism)
 *
 * The Holon is the foundational unit of Pluribus - a complete justification
 * for any action. It combines:
 * - Pentad (The Key): 5 positive coordinates that justify action
 * - Sextet (The Lock): 6 negative constraints that guard against unsafe actions
 *
 * An action can only proceed when the Key fits the Lock:
 * - All Pentad coordinates must be valid (justification exists)
 * - All Sextet gates must pass (no constraints violated)
 *
 * @module
 */

import { Pentad, validatePentad, createPentad } from './pentad.js';
import { Sextet, validateSextet, DEFAULT_SEXTET } from './sextet.js';
import { Ring } from './ring.js';

/**
 * Holon status after validation
 */
export type HolonStatus =
  | 'valid'      // Key fits lock, action may proceed
  | 'blocked'    // Lock blocks the key
  | 'incomplete' // Key is missing coordinates
  | 'expired';   // Kairos TTL exceeded

/**
 * The Holon - Complete justification for an action
 *
 * @example
 * ```typescript
 * const holon: Holon = {
 *   id: 'holon-12345',
 *   pentad: { ... },  // The Key
 *   sextet: { ... },  // The Lock
 *   status: 'valid',
 *   createdAt: Date.now(),
 * };
 *
 * const result = validateHolon(holon);
 * if (result.valid) {
 *   // Action may proceed
 * }
 * ```
 */
export interface Holon {
  /** Unique identifier for this holon */
  id: string;
  /** The Pentad (Key) - 5 positive coordinates */
  pentad: Pentad;
  /** The Sextet (Lock) - 6 negative constraints */
  sextet: Sextet;
  /** Current validation status */
  status: HolonStatus;
  /** Creation timestamp in milliseconds */
  createdAt: number;
  /** Last validation timestamp */
  validatedAt?: number;
  /** Parent holon if derived */
  parentId?: string;
  /** Child holons spawned from this one */
  childIds?: string[];
  /** Ring level for security context */
  ring?: Ring;
}

/**
 * Validation result for a Holon
 */
export interface HolonValidation {
  /** Overall validity */
  valid: boolean;
  /** Resulting status */
  status: HolonStatus;
  /** Errors from Pentad validation */
  pentadErrors: string[];
  /** Gates blocked by Sextet */
  sextetBlocks: string[];
  /** Timestamp of validation */
  validatedAt: number;
}

/**
 * Validate a Holon - check if the Key fits the Lock
 */
export function validateHolon(holon: Holon): HolonValidation {
  const now = Date.now();

  // Check expiration first
  if (holon.pentad.kairos.ttl) {
    const expiresAt = holon.pentad.kairos.timestamp + (holon.pentad.kairos.ttl * 1000);
    if (now > expiresAt) {
      return {
        valid: false,
        status: 'expired',
        pentadErrors: [],
        sextetBlocks: ['TTL exceeded'],
        validatedAt: now,
      };
    }
  }

  // Validate Pentad (The Key)
  const pentadResult = validatePentad(holon.pentad);

  // Validate Sextet (The Lock)
  const sextetResult = validateSextet(holon.sextet);

  // Determine overall status
  let status: HolonStatus;
  if (pentadResult.errors.length > 0) {
    status = 'incomplete';
  } else if (sextetResult.blockedBy.length > 0) {
    status = 'blocked';
  } else {
    status = 'valid';
  }

  return {
    valid: status === 'valid',
    status,
    pentadErrors: pentadResult.errors,
    sextetBlocks: sextetResult.blockedBy,
    validatedAt: now,
  };
}

/**
 * Create a new Holon with default Sextet
 */
export function createHolon(
  justification: string,
  path: string,
  agent: string,
  data: unknown,
  sextetOverrides?: Partial<Sextet>
): Holon {
  const pentad = createPentad(justification, path, agent, data);

  const sextet: Sextet = {
    ...DEFAULT_SEXTET,
    ...sextetOverrides,
  };

  const holon: Holon = {
    id: crypto.randomUUID(),
    pentad,
    sextet,
    status: 'valid', // Will be validated
    createdAt: Date.now(),
  };

  // Validate and update status
  const validation = validateHolon(holon);
  holon.status = validation.status;
  holon.validatedAt = validation.validatedAt;

  return holon;
}

/**
 * Derive a child Holon from a parent
 * Inherits lineage and some sextet constraints
 */
export function deriveHolon(
  parent: Holon,
  justification: string,
  path: string,
  data: unknown
): Holon {
  const child = createHolon(
    justification,
    path,
    parent.pentad.lineage.agent,
    data,
    {
      // Inherit provenance chain
      provenance: {
        ...parent.sextet.provenance,
        evidenceHash: parent.id, // Link to parent
      },
      // Inherit effect restrictions (can only be more restrictive)
      effects: parent.sextet.effects,
      // Inherit liveness constraints
      liveness: parent.sextet.liveness,
    }
  );

  // Link parent-child relationship
  child.parentId = parent.id;
  child.pentad.lineage.parent = parent.pentad.lineage.agent;

  return child;
}

/**
 * Check if a Holon can be executed right now
 * Combines validation with additional runtime checks
 */
export function canExecute(holon: Holon): { allowed: boolean; reason?: string } {
  // First validate the holon structure
  const validation = validateHolon(holon);
  if (!validation.valid) {
    return {
      allowed: false,
      reason: validation.status === 'expired'
        ? 'Holon has expired'
        : validation.status === 'incomplete'
        ? `Missing: ${validation.pentadErrors.join(', ')}`
        : `Blocked by: ${validation.sextetBlocks.join(', ')}`,
    };
  }

  // Check omega veto
  if (holon.sextet.omega.vetoable && holon.sextet.omega.aligned === false) {
    return {
      allowed: false,
      reason: 'Omega veto: action not aligned with system goals',
    };
  }

  // Check budget constraints
  const budget = holon.pentad.kairos.budget;
  if (budget.timeMs !== undefined && budget.timeMs <= 0) {
    return {
      allowed: false,
      reason: 'Time budget exhausted',
    };
  }

  return { allowed: true };
}

/**
 * Serialize Holon to JSON-compatible format
 */
export function serializeHolon(holon: Holon): string {
  return JSON.stringify(holon);
}

/**
 * Deserialize Holon from JSON string
 */
export function deserializeHolon(json: string): Holon {
  return JSON.parse(json) as Holon;
}
