/**
 * @ark/core/ring - Ring Security Hierarchy
 *
 * The Ring system enforces access control and authority levels:
 * - Ring 0 (Kernel): Immutable core, highest authority
 * - Ring 1 (Service): Infrastructure services
 * - Ring 2 (User): User-level operations
 * - Ring 3 (Ephemeral): Temporary/sandboxed operations
 *
 * Lower ring numbers = higher privilege
 * Code at Ring N can only access Ring N or higher (less privileged)
 *
 * @module
 */

/**
 * Ring levels in the security hierarchy
 */
export enum Ring {
  /** Ring 0: Kernel - Immutable core, DNA, Constitution */
  Kernel = 0,
  /** Ring 1: Service - Infrastructure, daemons, MCP servers */
  Service = 1,
  /** Ring 2: User - User-level operations, agents */
  User = 2,
  /** Ring 3: Ephemeral - Temporary clones, sandboxed execution */
  Ephemeral = 3,
}

/**
 * Ring metadata and policy
 */
export interface RingPolicy {
  /** Ring level */
  ring: Ring;
  /** Human-readable name */
  name: string;
  /** Description of this ring's purpose */
  description: string;
  /** Can code at this ring modify lower rings? */
  canModifyLower: boolean;
  /** Can code at this ring read lower rings? */
  canReadLower: boolean;
  /** Allowed effect types at this ring */
  allowedEffects: string[];
  /** Required approval for mutations */
  requiresApproval: 'none' | 'witness' | 'omega' | 'formal_proof';
}

/**
 * Default ring policies
 */
export const RING_POLICIES: Record<Ring, RingPolicy> = {
  [Ring.Kernel]: {
    ring: Ring.Kernel,
    name: 'Kernel',
    description: 'Immutable core - DNA, Constitution, Axioms',
    canModifyLower: false, // Ring 0 is immutable
    canReadLower: true,
    allowedEffects: ['read'],
    requiresApproval: 'formal_proof',
  },
  [Ring.Service]: {
    ring: Ring.Service,
    name: 'Service',
    description: 'Infrastructure services - daemons, MCP servers, bus',
    canModifyLower: false,
    canReadLower: true,
    allowedEffects: ['read', 'write', 'network'],
    requiresApproval: 'omega',
  },
  [Ring.User]: {
    ring: Ring.User,
    name: 'User',
    description: 'User-level operations - agents, tools',
    canModifyLower: false,
    canReadLower: true,
    allowedEffects: ['read', 'write', 'network', 'spawn'],
    requiresApproval: 'witness',
  },
  [Ring.Ephemeral]: {
    ring: Ring.Ephemeral,
    name: 'Ephemeral',
    description: 'Temporary sandboxed execution - PAIP clones',
    canModifyLower: false,
    canReadLower: true,
    allowedEffects: ['read', 'write'],
    requiresApproval: 'none',
  },
};

/**
 * URN (Uniform Resource Name) for ring-aware addressing
 * Format: ring{0-3}/{domain}/{type}/{id}
 *
 * @example
 * "ring0/axioms/core/append_only"
 * "ring1/mcp/server/rhizome"
 * "ring2/tools/cli/pli"
 * "ring3/clone/paip/agent-12345"
 */
export interface URN {
  /** Ring level (0-3) */
  ring: Ring;
  /** Domain (axioms, mcp, tools, clone, etc.) */
  domain: string;
  /** Type within domain */
  type: string;
  /** Unique identifier */
  id: string;
}

/**
 * Parse a URN string into components
 */
export function parseURN(urn: string): URN | null {
  const match = urn.match(/^ring([0-3])\/([^/]+)\/([^/]+)\/(.+)$/);
  if (!match) return null;

  return {
    ring: parseInt(match[1], 10) as Ring,
    domain: match[2],
    type: match[3],
    id: match[4],
  };
}

/**
 * Format URN components into a string
 */
export function formatURN(urn: URN): string {
  return `ring${urn.ring}/${urn.domain}/${urn.type}/${urn.id}`;
}

/**
 * Check if an actor at sourceRing can access a resource at targetRing
 */
export function canAccess(sourceRing: Ring, targetRing: Ring, operation: 'read' | 'write'): boolean {
  // Can always access same ring
  if (sourceRing === targetRing) {
    return true;
  }

  // Can access higher rings (less privileged)
  if (sourceRing < targetRing) {
    return true;
  }

  const policy = RING_POLICIES[sourceRing];

  // Check if lower ring access is allowed for this operation
  if (operation === 'read') {
    return policy.canReadLower;
  }

  return policy.canModifyLower;
}

/**
 * Check if an effect is allowed at a given ring
 */
export function isEffectAllowedAtRing(ring: Ring, effect: string): boolean {
  const policy = RING_POLICIES[ring];
  return policy.allowedEffects.includes(effect);
}

/**
 * Get the required approval level for mutations at a ring
 */
export function getRequiredApproval(ring: Ring): RingPolicy['requiresApproval'] {
  return RING_POLICIES[ring].requiresApproval;
}

/**
 * Ring context for tracking current execution ring
 */
export interface RingContext {
  /** Current ring level */
  current: Ring;
  /** Agent operating at this ring */
  agent: string;
  /** Session identifier */
  sessionId: string;
  /** Entry timestamp */
  enteredAt: number;
  /** Parent context if escalated/descended */
  parent?: RingContext;
}

/**
 * Create a new ring context
 */
export function createRingContext(
  ring: Ring,
  agent: string,
  sessionId: string
): RingContext {
  return {
    current: ring,
    agent,
    sessionId,
    enteredAt: Date.now(),
  };
}

/**
 * Descend to a lower-privilege ring (higher number)
 */
export function descendRing(context: RingContext, targetRing: Ring): RingContext | null {
  // Can only descend to higher ring numbers (less privilege)
  if (targetRing <= context.current) {
    return null;
  }

  return {
    current: targetRing,
    agent: context.agent,
    sessionId: context.sessionId,
    enteredAt: Date.now(),
    parent: context,
  };
}

/**
 * Attempt to escalate to a higher-privilege ring (lower number)
 * Requires approval based on target ring's policy
 */
export function requestEscalation(
  context: RingContext,
  targetRing: Ring
): { allowed: boolean; requiresApproval: RingPolicy['requiresApproval'] } {
  // Can only escalate to lower ring numbers (more privilege)
  if (targetRing >= context.current) {
    return { allowed: false, requiresApproval: 'none' };
  }

  const approval = getRequiredApproval(targetRing);
  return { allowed: true, requiresApproval: approval };
}
