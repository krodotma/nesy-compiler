/**
 * @ark/core/sextet - The Six Gates (Negative Constraints / The Lock)
 *
 * The Sextet represents the 6 negative constraints that guard against unsafe actions:
 * - P (Provenance): Append-only evidence, non-DPI, replayable
 * - E (Effects): Typed effects (network/file/payment/actuation restrictions)
 * - L (Liveness): Omega-guardrails, determinism, WCET/tick budgets
 * - R (Recovery): Verifiable recovery, canary/shadow promotion
 * - Q (Quality): Tests-first, coverage floors, property/mutation tests
 * - OMEGA: Teleological alignment and veto power
 *
 * @module
 */

/**
 * P-Gate: Provenance constraints
 * Ensures append-only evidence trail and replayability
 */
export interface ProvenanceGate {
  /** All evidence is append-only (no retroactive edits) */
  appendOnly: boolean;
  /** Non-DPI: No personally identifiable info in traces */
  nonDpi: boolean;
  /** Events can be replayed to reconstruct state */
  replayable: boolean;
  /** SHA256 hash of the evidence chain */
  evidenceHash?: string;
  /** Bus topic where evidence is emitted */
  busTopic?: string;
}

/**
 * Effect types that can be restricted
 */
export interface EffectTypes {
  /** Network access (HTTP, WebSocket, etc.) */
  network: boolean;
  /** File system access (read/write/delete) */
  file: boolean;
  /** Payment/financial transactions */
  payment: boolean;
  /** Physical actuation (IoT, robotics) */
  actuation: boolean;
  /** Process spawning */
  spawn: boolean;
  /** Environment variable access */
  env: boolean;
}

/**
 * E-Gate: Effects constraints
 * Typed effect discipline restricting side effects
 */
export interface EffectsGate {
  /** Allowed effect types */
  allowed: Partial<EffectTypes>;
  /** Denied effect types (takes precedence) */
  denied: Partial<EffectTypes>;
  /** Effect tier: low (read-only), medium (state changes), high (destructive) */
  tier: 'low' | 'medium' | 'high';
  /** Maximum number of effects per action */
  maxEffects?: number;
}

/**
 * L-Gate: Liveness constraints
 * Ensures termination and resource bounds
 */
export interface LivenessGate {
  /** Omega automaton guardrails active */
  omegaGuardrail: boolean;
  /** Action is deterministic (same input -> same output) */
  deterministic: boolean;
  /** Worst-Case Execution Time budget in milliseconds */
  wcetBudget: number;
  /** Maximum number of ticks/iterations */
  tickBudget?: number;
  /** Timeout for async operations in milliseconds */
  asyncTimeout?: number;
  /** Memory limit in bytes */
  memoryLimit?: number;
}

/**
 * R-Gate: Recovery constraints
 * Ensures verifiable recovery and safe promotion
 */
export interface RecoveryGate {
  /** Canary deployment enabled */
  canary: boolean;
  /** Shadow/parallel execution for verification */
  shadow: boolean;
  /** Recovery path verified */
  verified: boolean;
  /** Rollback checkpoint available */
  checkpointAvailable: boolean;
  /** Recovery strategy */
  strategy: 'rollback' | 'compensate' | 'retry' | 'escalate';
}

/**
 * Q-Gate: Quality constraints
 * Ensures test coverage and quality standards
 */
export interface QualityGate {
  /** Tests written before code */
  testsFirst: boolean;
  /** Minimum coverage percentage (0-100) */
  coverageFloor: number;
  /** Property-based tests included */
  propertyTests: boolean;
  /** Mutation testing score minimum */
  mutationScore?: number;
  /** Fuzz testing enabled */
  fuzzTests: boolean;
  /** VOR (Verification of Reality) checks pass */
  vorPassed: boolean;
}

/**
 * Omega-Gate: Teleological alignment
 * The final arbiter with veto power
 */
export interface OmegaGate {
  /** Action aligns with system goals */
  aligned: boolean;
  /** Omega has veto power over this action */
  vetoable: boolean;
  /** Explicit omega approval obtained */
  approved?: boolean;
  /** Omega guardian rule that applies */
  rule?: string;
  /** Alignment score (0-1) */
  alignmentScore?: number;
}

/**
 * The Sextet - Six gates that constrain actions
 *
 * @example
 * ```typescript
 * const sextet: Sextet = {
 *   provenance: {
 *     appendOnly: true,
 *     nonDpi: true,
 *     replayable: true,
 *     busTopic: 'agent.action'
 *   },
 *   effects: {
 *     allowed: { file: true, network: false },
 *     denied: { payment: true, actuation: true },
 *     tier: 'medium'
 *   },
 *   liveness: {
 *     omegaGuardrail: true,
 *     deterministic: true,
 *     wcetBudget: 30000
 *   },
 *   recovery: {
 *     canary: false,
 *     shadow: false,
 *     verified: true,
 *     checkpointAvailable: true,
 *     strategy: 'rollback'
 *   },
 *   quality: {
 *     testsFirst: true,
 *     coverageFloor: 80,
 *     propertyTests: true,
 *     fuzzTests: false,
 *     vorPassed: true
 *   },
 *   omega: {
 *     aligned: true,
 *     vetoable: true,
 *     alignmentScore: 0.95
 *   }
 * };
 * ```
 */
export interface Sextet {
  /** P-Gate: Provenance constraints */
  provenance: ProvenanceGate;
  /** E-Gate: Effects constraints */
  effects: EffectsGate;
  /** L-Gate: Liveness constraints */
  liveness: LivenessGate;
  /** R-Gate: Recovery constraints */
  recovery: RecoveryGate;
  /** Q-Gate: Quality constraints */
  quality: QualityGate;
  /** Omega-Gate: Teleological alignment */
  omega: OmegaGate;
}

/**
 * Default Sextet with safe defaults
 */
export const DEFAULT_SEXTET: Sextet = {
  provenance: {
    appendOnly: true,
    nonDpi: true,
    replayable: true,
  },
  effects: {
    allowed: { file: true },
    denied: { payment: true, actuation: true },
    tier: 'low',
  },
  liveness: {
    omegaGuardrail: true,
    deterministic: true,
    wcetBudget: 60000,
  },
  recovery: {
    canary: false,
    shadow: false,
    verified: false,
    checkpointAvailable: false,
    strategy: 'rollback',
  },
  quality: {
    testsFirst: false,
    coverageFloor: 0,
    propertyTests: false,
    fuzzTests: false,
    vorPassed: false,
  },
  omega: {
    aligned: true,
    vetoable: true,
  },
};

/**
 * Validate that all Sextet gates pass
 * Returns false if any gate blocks the action
 */
export function validateSextet(sextet: Sextet): { valid: boolean; blockedBy: string[] } {
  const blockedBy: string[] = [];

  // P-Gate validation
  if (!sextet.provenance.appendOnly) {
    blockedBy.push('P-Gate: appendOnly must be true');
  }

  // E-Gate validation
  if (sextet.effects.tier === 'high' && !sextet.omega.approved) {
    blockedBy.push('E-Gate: high-tier effects require omega approval');
  }

  // L-Gate validation
  if (!sextet.liveness.deterministic && !sextet.recovery.checkpointAvailable) {
    blockedBy.push('L-Gate: non-deterministic actions require checkpoint');
  }

  // R-Gate validation
  if (!sextet.recovery.verified && sextet.effects.tier !== 'low') {
    blockedBy.push('R-Gate: recovery must be verified for non-low tier effects');
  }

  // Q-Gate validation
  if (sextet.quality.coverageFloor > 0 && !sextet.quality.vorPassed) {
    blockedBy.push('Q-Gate: VOR must pass when coverage floor is set');
  }

  // Omega-Gate validation
  if (sextet.omega.vetoable && sextet.omega.aligned === false) {
    blockedBy.push('Omega-Gate: action is not aligned with system goals');
  }

  return { valid: blockedBy.length === 0, blockedBy };
}

/**
 * Check if a specific effect is allowed by the E-Gate
 */
export function isEffectAllowed(sextet: Sextet, effect: keyof EffectTypes): boolean {
  // Denied takes precedence
  if (sextet.effects.denied[effect]) {
    return false;
  }
  // Check if explicitly allowed
  return sextet.effects.allowed[effect] === true;
}
