/**
 * Chromatic Agents Visualizer - Interpolation & Easing Functions
 *
 * Step 14: Smooth visual transitions
 * - Position: lerp with easeOutQuad
 * - Color: HSL interpolation for hue shifts
 * - Scale: spring physics for "pop" effects
 * - Opacity: exponential decay for fades
 */

// =============================================================================
// Easing Functions
// =============================================================================

/**
 * Linear interpolation (no easing)
 */
export function linear(t: number): number {
  return t;
}

/**
 * Ease out quadratic - decelerating to zero velocity
 */
export function easeOutQuad(t: number): number {
  return t * (2 - t);
}

/**
 * Ease in quadratic - accelerating from zero velocity
 */
export function easeInQuad(t: number): number {
  return t * t;
}

/**
 * Ease in-out quadratic - acceleration until halfway, then deceleration
 */
export function easeInOutQuad(t: number): number {
  return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t;
}

/**
 * Ease out cubic - stronger deceleration
 */
export function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

/**
 * Ease out elastic - overshoot with bounce
 */
export function easeOutElastic(t: number): number {
  const p = 0.3;
  return Math.pow(2, -10 * t) * Math.sin((t - p / 4) * (2 * Math.PI) / p) + 1;
}

/**
 * Ease out back - slight overshoot
 */
export function easeOutBack(t: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

// =============================================================================
// Interpolation Functions
// =============================================================================

/**
 * Linear interpolation between two values
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Linear interpolation with easing function
 */
export function lerpEased(
  a: number,
  b: number,
  t: number,
  easeFn: (t: number) => number = easeOutQuad
): number {
  return lerp(a, b, easeFn(t));
}

/**
 * Interpolate 3D positions
 */
export function lerp3D(
  a: [number, number, number],
  b: [number, number, number],
  t: number,
  easeFn: (t: number) => number = easeOutQuad
): [number, number, number] {
  const eased = easeFn(t);
  return [
    lerp(a[0], b[0], eased),
    lerp(a[1], b[1], eased),
    lerp(a[2], b[2], eased),
  ];
}

/**
 * Interpolate HSL colors (handles hue wrapping)
 */
export function lerpHSL(
  h1: number, s1: number, l1: number,
  h2: number, s2: number, l2: number,
  t: number
): [number, number, number] {
  // Handle hue wrapping (e.g., 350 to 10 should go through 0)
  let dh = h2 - h1;
  if (dh > 180) dh -= 360;
  if (dh < -180) dh += 360;

  const h = ((h1 + dh * t) % 360 + 360) % 360;
  const s = lerp(s1, s2, t);
  const l = lerp(l1, l2, t);

  return [h, s, l];
}

// =============================================================================
// Spring Physics
// =============================================================================

interface SpringState {
  position: number;
  velocity: number;
}

interface SpringConfig {
  stiffness: number;   // Spring stiffness (k)
  damping: number;     // Damping coefficient (c)
  mass: number;        // Mass (m)
}

const DEFAULT_SPRING_CONFIG: SpringConfig = {
  stiffness: 170,
  damping: 26,
  mass: 1,
};

/**
 * Update spring state for one frame
 */
export function updateSpring(
  state: SpringState,
  target: number,
  deltaTime: number,
  config: Partial<SpringConfig> = {}
): SpringState {
  const { stiffness, damping, mass } = { ...DEFAULT_SPRING_CONFIG, ...config };

  const displacement = state.position - target;
  const springForce = -stiffness * displacement;
  const dampingForce = -damping * state.velocity;
  const acceleration = (springForce + dampingForce) / mass;

  const newVelocity = state.velocity + acceleration * deltaTime;
  const newPosition = state.position + newVelocity * deltaTime;

  return {
    position: newPosition,
    velocity: newVelocity,
  };
}

/**
 * Check if spring has settled (close enough to target)
 */
export function isSpringSettled(
  state: SpringState,
  target: number,
  threshold: number = 0.001
): boolean {
  return Math.abs(state.position - target) < threshold &&
         Math.abs(state.velocity) < threshold;
}

/**
 * Create a spring animator for 3D positions
 */
export class Spring3D {
  private x: SpringState = { position: 0, velocity: 0 };
  private y: SpringState = { position: 0, velocity: 0 };
  private z: SpringState = { position: 0, velocity: 0 };
  private config: SpringConfig;

  constructor(
    initial: [number, number, number] = [0, 0, 0],
    config: Partial<SpringConfig> = {}
  ) {
    this.x.position = initial[0];
    this.y.position = initial[1];
    this.z.position = initial[2];
    this.config = { ...DEFAULT_SPRING_CONFIG, ...config };
  }

  update(target: [number, number, number], deltaTime: number): [number, number, number] {
    this.x = updateSpring(this.x, target[0], deltaTime, this.config);
    this.y = updateSpring(this.y, target[1], deltaTime, this.config);
    this.z = updateSpring(this.z, target[2], deltaTime, this.config);

    return [this.x.position, this.y.position, this.z.position];
  }

  get position(): [number, number, number] {
    return [this.x.position, this.y.position, this.z.position];
  }

  isSettled(target: [number, number, number], threshold: number = 0.001): boolean {
    return isSpringSettled(this.x, target[0], threshold) &&
           isSpringSettled(this.y, target[1], threshold) &&
           isSpringSettled(this.z, target[2], threshold);
  }
}

// =============================================================================
// Exponential Decay
// =============================================================================

/**
 * Exponential decay for fade effects
 * @param current Current value
 * @param target Target value (usually 0 for fading out)
 * @param rate Decay rate (higher = faster)
 * @param deltaTime Time since last update
 */
export function exponentialDecay(
  current: number,
  target: number,
  rate: number,
  deltaTime: number
): number {
  return target + (current - target) * Math.exp(-rate * deltaTime);
}

/**
 * Smooth damp - critically damped spring for smooth transitions
 */
export function smoothDamp(
  current: number,
  target: number,
  velocity: { value: number },
  smoothTime: number,
  maxSpeed: number = Infinity,
  deltaTime: number = 1/60
): number {
  smoothTime = Math.max(0.0001, smoothTime);
  const omega = 2 / smoothTime;
  const x = omega * deltaTime;
  const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);

  let change = current - target;
  const maxChange = maxSpeed * smoothTime;
  change = Math.max(-maxChange, Math.min(maxChange, change));

  const temp = (velocity.value + omega * change) * deltaTime;
  velocity.value = (velocity.value - omega * temp) * exp;

  let result = target + (change + temp) * exp;

  // Prevent overshooting
  if (target - current > 0 === result > target) {
    result = target;
    velocity.value = 0;
  }

  return result;
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

/**
 * Map a value from one range to another
 */
export function mapRange(
  value: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  return outMin + (outMax - outMin) * ((value - inMin) / (inMax - inMin));
}

/**
 * Smoothstep interpolation (cubic Hermite interpolation)
 */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/**
 * Ping-pong value between 0 and 1
 */
export function pingPong(t: number): number {
  const mod = t % 2;
  return mod <= 1 ? mod : 2 - mod;
}
