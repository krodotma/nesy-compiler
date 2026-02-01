/**
 * @ark/core/invariants - Conservation Invariants
 *
 * Mathematical constants and invariants that must be preserved
 * throughout the system. These are based on the golden ratio
 * and provide stable defaults for various subsystems.
 *
 * The golden ratio phi (φ) appears throughout:
 * - CMP discount factor (1/φ)
 * - Global floors (1/φ³)
 * - Replay buffer parameters
 *
 * @module
 */

/**
 * Golden ratio φ = (1 + √5) / 2 ≈ 1.618033988749895
 */
export const PHI = (1 + Math.sqrt(5)) / 2;

/**
 * Inverse golden ratio 1/φ ≈ 0.618033988749895
 */
export const INV_PHI = 1 / PHI;

/**
 * φ² ≈ 2.618033988749895
 */
export const PHI_SQUARED = PHI * PHI;

/**
 * 1/φ² ≈ 0.381966011250105
 */
export const INV_PHI_SQUARED = 1 / PHI_SQUARED;

/**
 * 1/φ³ ≈ 0.236067977499790
 */
export const INV_PHI_CUBED = 1 / (PHI * PHI * PHI);

/**
 * CMP discount factor = 1/φ ≈ 0.618
 * Explicitly exported for convenience
 */
export const CMP_DISCOUNT = INV_PHI;

/**
 * Global CMP floor = 1/φ³ ≈ 0.236
 * Explicitly exported for convenience
 */
export const GLOBAL_CMP_FLOOR = INV_PHI_CUBED;

/**
 * Conservation invariants
 *
 * These values MUST be preserved during migration and should
 * not be changed without careful consideration of system-wide effects.
 */
export const CONSERVATION_INVARIANTS = {
  // Golden ratio constants
  /** Golden ratio φ */
  PHI,
  /** Inverse golden ratio 1/φ */
  INV_PHI,

  // CMP (Cognitive Metric Priority) constants
  /** CMP discount factor = 1/φ ≈ 0.618 */
  CMP_DISCOUNT: INV_PHI,
  /** Global CMP floor = 1/φ³ ≈ 0.236 */
  GLOBAL_CMP_FLOOR: INV_PHI_CUBED,
  /** CMP ceiling */
  CMP_CEILING: 1.0,

  // Continual Learning constants
  /** EWC regularization strength */
  EWC_LAMBDA: 1000,
  /** Fisher information samples */
  FISHER_SAMPLES: 200,
  /** Replay buffer priority exponent α */
  REPLAY_ALPHA: 0.6,
  /** Replay buffer IS correction β */
  REPLAY_BETA: 0.4,
  /** Minimum experiences before replay */
  MIN_EXPERIENCES: 64,

  // Neural network constants
  /** Default learning rate */
  LEARNING_RATE: 3e-4,
  /** Gradient clipping threshold */
  GRADIENT_CLIP: 1.0,
  /** Weight decay */
  WEIGHT_DECAY: 1e-4,

  // Temporal constants
  /** Default TTL in seconds */
  DEFAULT_TTL_S: 3600,
  /** Event retention in days */
  EVENT_RETENTION_DAYS: 30,
  /** Checkpoint interval */
  CHECKPOINT_INTERVAL: 100,

  // Bus constants
  /** Maximum event size (bytes) */
  MAX_EVENT_SIZE: 1_000_000,
  /** Event batch size */
  BATCH_SIZE: 100,
  /** Publish timeout (ms) */
  PUBLISH_TIMEOUT_MS: 5000,

  // Safety thresholds
  /** Thrash detection threshold */
  THRASH_THRESHOLD: 0.6,
  /** Uncertainty threshold for exploration */
  UNCERTAINTY_THRESHOLD: 0.5,
  /** Minimum confidence for patterns */
  MIN_PATTERN_CONFIDENCE: 0.1,
} as const;

/**
 * Type for conservation invariant keys
 */
export type InvariantKey = keyof typeof CONSERVATION_INVARIANTS;

/**
 * Validate that a value respects an invariant
 */
export function validateInvariant(
  key: InvariantKey,
  value: number,
  tolerance: number = 1e-6
): boolean {
  const expected = CONSERVATION_INVARIANTS[key];
  return Math.abs(value - expected) < tolerance;
}

/**
 * Get invariant value
 */
export function getInvariant(key: InvariantKey): number {
  return CONSERVATION_INVARIANTS[key];
}

/**
 * Check if all critical invariants are preserved in a configuration
 */
export function checkInvariants(config: Record<string, number>): {
  valid: boolean;
  violations: { key: string; expected: number; actual: number }[];
} {
  const violations: { key: string; expected: number; actual: number }[] = [];

  for (const [key, expected] of Object.entries(CONSERVATION_INVARIANTS)) {
    if (key in config) {
      const actual = config[key];
      if (Math.abs(actual - expected) > 1e-6) {
        violations.push({ key, expected, actual });
      }
    }
  }

  return {
    valid: violations.length === 0,
    violations,
  };
}

/**
 * Fibonacci sequence generator
 */
export function* fibonacci(n: number): Generator<number> {
  let a = 0;
  let b = 1;

  for (let i = 0; i < n; i++) {
    yield a;
    [a, b] = [b, a + b];
  }
}

/**
 * Get nth Fibonacci number
 */
export function fib(n: number): number {
  // Binet's formula for efficiency
  return Math.round(
    (Math.pow(PHI, n) - Math.pow(-INV_PHI, n)) / Math.sqrt(5)
  );
}

/**
 * Golden ratio division
 * Divides a value into two parts in the golden ratio
 */
export function goldenDivide(value: number): [number, number] {
  const larger = value * INV_PHI;
  const smaller = value - larger;
  return [larger, smaller];
}
