/**
 * Chromatic Agents Visualizer - Particle System
 *
 * Step 15: GPU-accelerated particles for data flow visualization.
 * Uses Three.js Points with BufferGeometry for performance.
 */

import type { AgentId, ParticleConfig, ParticleStream } from './types';
import {
  getAgentColor,
  getAgentHue,
  hexToThreeColor,
  getAgentRGBNormalized,
} from './utils/colorMap';
import { lerp, lerp3D, easeOutQuad } from './utils/interpolation';

// =============================================================================
// Constants
// =============================================================================

/** Default particle pool size per stream */
const DEFAULT_POOL_SIZE = 1000;

/** Maximum number of active particle streams */
const MAX_STREAMS = 16;

/** Particle vertex shader */
export const PARTICLE_VERTEX_SHADER = `
  attribute float size;
  attribute float alpha;
  attribute vec3 customColor;

  varying float vAlpha;
  varying vec3 vColor;

  void main() {
    vAlpha = alpha;
    vColor = customColor;

    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = size * (300.0 / -mvPosition.z);
    gl_Position = projectionMatrix * mvPosition;
  }
`;

/** Particle fragment shader */
export const PARTICLE_FRAGMENT_SHADER = `
  varying float vAlpha;
  varying vec3 vColor;

  void main() {
    // Circular particle with soft edges
    vec2 center = gl_PointCoord - vec2(0.5);
    float dist = length(center);
    if (dist > 0.5) discard;

    float softness = 1.0 - smoothstep(0.3, 0.5, dist);
    gl_FragColor = vec4(vColor, vAlpha * softness);
  }
`;

// =============================================================================
// Particle Types
// =============================================================================

export interface Particle {
  /** Position [x, y, z] */
  position: [number, number, number];
  /** Velocity [vx, vy, vz] */
  velocity: [number, number, number];
  /** Current lifetime (counts down) */
  lifetime: number;
  /** Initial lifetime (for ratio calculations) */
  initialLifetime: number;
  /** Particle size */
  size: number;
  /** Color as RGB normalized [0-1] */
  color: [number, number, number];
  /** Alpha (opacity) */
  alpha: number;
  /** Is particle active */
  active: boolean;
}

export type StreamType =
  | 'bus_event'
  | 'commit_burst'
  | 'push_stream'
  | 'merge_spiral'
  | 'clone_spawn'
  | 'ambient';

// =============================================================================
// Stream Configurations per Event Type
// =============================================================================

const STREAM_CONFIGS: Record<StreamType, Partial<ParticleConfig>> = {
  bus_event: {
    count: 5,
    size: 2,
    speed: 3,
    lifetime: 1,
    emissionRate: 10,
  },
  commit_burst: {
    count: 50,
    size: 4,
    speed: 8,
    lifetime: 0.8,
    emissionRate: 200,
  },
  push_stream: {
    count: 100,
    size: 3,
    speed: 5,
    lifetime: 1.5,
    emissionRate: 80,
  },
  merge_spiral: {
    count: 200,
    size: 3.5,
    speed: 2,
    lifetime: 2,
    emissionRate: 100,
  },
  clone_spawn: {
    count: 80,
    size: 2.5,
    speed: 4,
    lifetime: 1.2,
    emissionRate: 60,
  },
  ambient: {
    count: 20,
    size: 1.5,
    speed: 0.5,
    lifetime: 3,
    emissionRate: 5,
  },
};

// =============================================================================
// Particle Pool (Object Pooling for Performance)
// =============================================================================

export class ParticlePool {
  private particles: Particle[] = [];
  private activeCount: number = 0;
  private poolSize: number;

  constructor(poolSize: number = DEFAULT_POOL_SIZE) {
    this.poolSize = poolSize;
    this.initializePool();
  }

  private initializePool(): void {
    for (let i = 0; i < this.poolSize; i++) {
      this.particles.push({
        position: [0, 0, 0],
        velocity: [0, 0, 0],
        lifetime: 0,
        initialLifetime: 1,
        size: 1,
        color: [1, 1, 1],
        alpha: 0,
        active: false,
      });
    }
  }

  /**
   * Get an inactive particle from the pool
   */
  acquire(): Particle | null {
    for (const particle of this.particles) {
      if (!particle.active) {
        particle.active = true;
        this.activeCount++;
        return particle;
      }
    }
    return null; // Pool exhausted
  }

  /**
   * Return a particle to the pool
   */
  release(particle: Particle): void {
    if (particle.active) {
      particle.active = false;
      particle.alpha = 0;
      this.activeCount--;
    }
  }

  /**
   * Get all active particles
   */
  getActive(): Particle[] {
    return this.particles.filter((p) => p.active);
  }

  /**
   * Get all particles (for buffer updates)
   */
  getAll(): Particle[] {
    return this.particles;
  }

  /**
   * Get active count
   */
  getActiveCount(): number {
    return this.activeCount;
  }

  /**
   * Reset all particles
   */
  reset(): void {
    for (const particle of this.particles) {
      particle.active = false;
      particle.alpha = 0;
    }
    this.activeCount = 0;
  }
}

// =============================================================================
// Particle Emitter
// =============================================================================

export interface EmitterConfig {
  /** Source position */
  position: [number, number, number];
  /** Optional target position (for directional streams) */
  target?: [number, number, number];
  /** Stream type */
  streamType: StreamType;
  /** Agent ID (for color) */
  agentId: AgentId;
  /** Override particle config */
  configOverride?: Partial<ParticleConfig>;
  /** Is emitter active */
  active: boolean;
  /** Emission accumulator (for fractional emissions) */
  emissionAccumulator: number;
}

export class ParticleEmitter {
  private pool: ParticlePool;
  private config: ParticleConfig;
  private emitterConfig: EmitterConfig;

  constructor(
    pool: ParticlePool,
    streamType: StreamType,
    agentId: AgentId,
    position: [number, number, number],
    target?: [number, number, number]
  ) {
    this.pool = pool;

    // Merge default config with stream-specific config
    const streamConfig = STREAM_CONFIGS[streamType];
    this.config = {
      count: streamConfig.count ?? 50,
      color: getAgentColor(agentId),
      speed: streamConfig.speed ?? 3,
      size: streamConfig.size ?? 2,
      lifetime: streamConfig.lifetime ?? 1,
      emissionRate: streamConfig.emissionRate ?? 20,
    };

    this.emitterConfig = {
      position,
      target,
      streamType,
      agentId,
      active: true,
      emissionAccumulator: 0,
    };
  }

  /**
   * Update emitter and emit particles
   */
  update(deltaTime: number): void {
    if (!this.emitterConfig.active) return;

    // Accumulate emission time
    this.emitterConfig.emissionAccumulator += deltaTime * this.config.emissionRate;

    // Emit particles
    while (this.emitterConfig.emissionAccumulator >= 1) {
      this.emitParticle();
      this.emitterConfig.emissionAccumulator -= 1;
    }
  }

  /**
   * Emit a single particle
   */
  private emitParticle(): void {
    const particle = this.pool.acquire();
    if (!particle) return; // Pool exhausted

    const { position, target, agentId, streamType } = this.emitterConfig;
    const color = getAgentRGBNormalized(agentId);

    // Initialize particle
    particle.position = [...position];
    particle.lifetime = this.config.lifetime;
    particle.initialLifetime = this.config.lifetime;
    particle.size = this.config.size * (0.8 + Math.random() * 0.4); // Size variation
    particle.color = color;
    particle.alpha = 1;

    // Calculate velocity based on stream type
    particle.velocity = this.calculateVelocity(streamType, position, target);
  }

  /**
   * Calculate initial velocity based on stream type
   */
  private calculateVelocity(
    streamType: StreamType,
    from: [number, number, number],
    to?: [number, number, number]
  ): [number, number, number] {
    const speed = this.config.speed;

    switch (streamType) {
      case 'bus_event':
        // Random radial outward
        return this.randomSpherical(speed * 0.5);

      case 'commit_burst':
        // Explosive burst in all directions
        return this.randomSpherical(speed);

      case 'push_stream':
        // Directional toward target (or upward to "remote")
        if (to) {
          return this.directionToward(from, to, speed);
        }
        return [0, speed, 0]; // Default upward

      case 'merge_spiral':
        // Spiral toward center
        return this.spiralVelocity(from, speed);

      case 'clone_spawn':
        // Outward expansion from spawn point
        return this.randomSpherical(speed * 0.7);

      case 'ambient':
        // Slow random drift
        return this.randomSpherical(speed * 0.3);

      default:
        return [0, 0, 0];
    }
  }

  /**
   * Generate random spherical velocity
   */
  private randomSpherical(magnitude: number): [number, number, number] {
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    return [
      Math.sin(phi) * Math.cos(theta) * magnitude,
      Math.sin(phi) * Math.sin(theta) * magnitude,
      Math.cos(phi) * magnitude,
    ];
  }

  /**
   * Calculate velocity toward target
   */
  private directionToward(
    from: [number, number, number],
    to: [number, number, number],
    speed: number
  ): [number, number, number] {
    const dx = to[0] - from[0];
    const dy = to[1] - from[1];
    const dz = to[2] - from[2];
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (dist < 0.001) return [0, speed, 0];

    // Add some spread
    const spread = 0.3;
    return [
      (dx / dist + (Math.random() - 0.5) * spread) * speed,
      (dy / dist + (Math.random() - 0.5) * spread) * speed,
      (dz / dist + (Math.random() - 0.5) * spread) * speed,
    ];
  }

  /**
   * Calculate spiral velocity toward center
   */
  private spiralVelocity(
    pos: [number, number, number],
    speed: number
  ): [number, number, number] {
    const dist = Math.sqrt(pos[0] * pos[0] + pos[2] * pos[2]);
    if (dist < 0.001) return [0, speed * 0.5, 0];

    // Inward + tangential components
    const inward = 0.6;
    const tangent = 0.4;

    const nx = -pos[0] / dist;
    const nz = -pos[2] / dist;
    const tx = -nz; // Perpendicular
    const tz = nx;

    return [
      (nx * inward + tx * tangent) * speed,
      (Math.random() - 0.5) * speed * 0.2, // Slight vertical variation
      (nz * inward + tz * tangent) * speed,
    ];
  }

  /**
   * Set emitter active state
   */
  setActive(active: boolean): void {
    this.emitterConfig.active = active;
  }

  /**
   * Update emitter position
   */
  setPosition(position: [number, number, number]): void {
    this.emitterConfig.position = position;
  }

  /**
   * Update target position
   */
  setTarget(target: [number, number, number]): void {
    this.emitterConfig.target = target;
  }

  /**
   * Trigger a burst of particles
   */
  burst(count: number): void {
    for (let i = 0; i < count; i++) {
      this.emitParticle();
    }
  }
}

// =============================================================================
// Main Particle System
// =============================================================================

export interface ParticleSystemStats {
  activeParticles: number;
  activeEmitters: number;
  poolUtilization: number;
}

export class ParticleSystem {
  private pools: Map<AgentId, ParticlePool> = new Map();
  private emitters: Map<string, ParticleEmitter> = new Map();
  private gravity: [number, number, number] = [0, -0.5, 0];
  private drag: number = 0.98;

  /** Typed array buffers for GPU (Three.js compatibility) */
  private positionBuffer: Float32Array | null = null;
  private colorBuffer: Float32Array | null = null;
  private sizeBuffer: Float32Array | null = null;
  private alphaBuffer: Float32Array | null = null;

  /** Buffer size */
  private bufferSize: number = 0;

  constructor() {
    // Initialize pools for each agent
    for (const agentId of ['claude', 'qwen', 'gemini', 'codex', 'main'] as AgentId[]) {
      this.pools.set(agentId, new ParticlePool(DEFAULT_POOL_SIZE));
    }

    // Initialize buffers
    this.initializeBuffers();
  }

  /**
   * Initialize GPU buffers
   */
  private initializeBuffers(): void {
    // Total particles across all pools
    this.bufferSize = DEFAULT_POOL_SIZE * 5; // 5 agents

    this.positionBuffer = new Float32Array(this.bufferSize * 3);
    this.colorBuffer = new Float32Array(this.bufferSize * 3);
    this.sizeBuffer = new Float32Array(this.bufferSize);
    this.alphaBuffer = new Float32Array(this.bufferSize);
  }

  /**
   * Create an emitter for an agent
   */
  createEmitter(
    id: string,
    agentId: AgentId,
    streamType: StreamType,
    position: [number, number, number],
    target?: [number, number, number]
  ): ParticleEmitter {
    const pool = this.pools.get(agentId);
    if (!pool) {
      throw new Error(`No particle pool for agent: ${agentId}`);
    }

    const emitter = new ParticleEmitter(pool, streamType, agentId, position, target);
    this.emitters.set(id, emitter);

    // Enforce max streams
    if (this.emitters.size > MAX_STREAMS) {
      // Remove oldest emitter
      const firstKey = this.emitters.keys().next().value;
      if (firstKey) {
        this.emitters.delete(firstKey);
      }
    }

    return emitter;
  }

  /**
   * Get emitter by ID
   */
  getEmitter(id: string): ParticleEmitter | undefined {
    return this.emitters.get(id);
  }

  /**
   * Remove emitter
   */
  removeEmitter(id: string): void {
    this.emitters.delete(id);
  }

  /**
   * Trigger a burst on an agent
   */
  triggerBurst(
    agentId: AgentId,
    streamType: StreamType,
    position: [number, number, number],
    count: number = 50
  ): void {
    const pool = this.pools.get(agentId);
    if (!pool) return;

    const emitter = new ParticleEmitter(pool, streamType, agentId, position);
    emitter.burst(count);
  }

  /**
   * Update all particles
   * @param deltaTime Time since last frame in seconds
   */
  update(deltaTime: number): void {
    // Update emitters
    for (const emitter of this.emitters.values()) {
      emitter.update(deltaTime);
    }

    // Update all particles in all pools
    for (const pool of this.pools.values()) {
      for (const particle of pool.getActive()) {
        this.updateParticle(particle, deltaTime);

        // Check lifetime
        if (particle.lifetime <= 0) {
          pool.release(particle);
        }
      }
    }

    // Update GPU buffers
    this.updateBuffers();
  }

  /**
   * Update a single particle
   */
  private updateParticle(particle: Particle, deltaTime: number): void {
    // Apply gravity
    particle.velocity[0] += this.gravity[0] * deltaTime;
    particle.velocity[1] += this.gravity[1] * deltaTime;
    particle.velocity[2] += this.gravity[2] * deltaTime;

    // Apply drag
    particle.velocity[0] *= this.drag;
    particle.velocity[1] *= this.drag;
    particle.velocity[2] *= this.drag;

    // Update position
    particle.position[0] += particle.velocity[0] * deltaTime;
    particle.position[1] += particle.velocity[1] * deltaTime;
    particle.position[2] += particle.velocity[2] * deltaTime;

    // Update lifetime
    particle.lifetime -= deltaTime;

    // Fade based on lifetime ratio
    const lifeRatio = particle.lifetime / particle.initialLifetime;
    particle.alpha = easeOutQuad(lifeRatio);

    // Shrink as it fades
    particle.size *= 0.995;
  }

  /**
   * Update typed array buffers for GPU rendering
   */
  private updateBuffers(): void {
    if (!this.positionBuffer || !this.colorBuffer || !this.sizeBuffer || !this.alphaBuffer) {
      return;
    }

    let bufferIndex = 0;

    for (const pool of this.pools.values()) {
      for (const particle of pool.getAll()) {
        if (bufferIndex >= this.bufferSize) break;

        // Position
        this.positionBuffer[bufferIndex * 3] = particle.position[0];
        this.positionBuffer[bufferIndex * 3 + 1] = particle.position[1];
        this.positionBuffer[bufferIndex * 3 + 2] = particle.position[2];

        // Color
        this.colorBuffer[bufferIndex * 3] = particle.color[0];
        this.colorBuffer[bufferIndex * 3 + 1] = particle.color[1];
        this.colorBuffer[bufferIndex * 3 + 2] = particle.color[2];

        // Size
        this.sizeBuffer[bufferIndex] = particle.active ? particle.size : 0;

        // Alpha
        this.alphaBuffer[bufferIndex] = particle.active ? particle.alpha : 0;

        bufferIndex++;
      }
    }
  }

  /**
   * Get position buffer for Three.js
   */
  getPositionBuffer(): Float32Array | null {
    return this.positionBuffer;
  }

  /**
   * Get color buffer for Three.js
   */
  getColorBuffer(): Float32Array | null {
    return this.colorBuffer;
  }

  /**
   * Get size buffer for Three.js
   */
  getSizeBuffer(): Float32Array | null {
    return this.sizeBuffer;
  }

  /**
   * Get alpha buffer for Three.js
   */
  getAlphaBuffer(): Float32Array | null {
    return this.alphaBuffer;
  }

  /**
   * Get buffer size (number of particles)
   */
  getBufferSize(): number {
    return this.bufferSize;
  }

  /**
   * Get system statistics
   */
  getStats(): ParticleSystemStats {
    let activeParticles = 0;
    for (const pool of this.pools.values()) {
      activeParticles += pool.getActiveCount();
    }

    return {
      activeParticles,
      activeEmitters: this.emitters.size,
      poolUtilization: activeParticles / this.bufferSize,
    };
  }

  /**
   * Set gravity
   */
  setGravity(gravity: [number, number, number]): void {
    this.gravity = gravity;
  }

  /**
   * Set drag coefficient
   */
  setDrag(drag: number): void {
    this.drag = Math.max(0, Math.min(1, drag));
  }

  /**
   * Reset all particles
   */
  reset(): void {
    for (const pool of this.pools.values()) {
      pool.reset();
    }
    this.emitters.clear();
  }

  /**
   * Dispose resources
   */
  dispose(): void {
    this.reset();
    this.positionBuffer = null;
    this.colorBuffer = null;
    this.sizeBuffer = null;
    this.alphaBuffer = null;
  }
}

// =============================================================================
// Particle Stream Factory
// =============================================================================

/**
 * Create a particle stream configuration
 */
export function createParticleStream(
  agentId: AgentId,
  from: [number, number, number],
  to: [number, number, number],
  streamType: StreamType
): ParticleStream {
  const config = STREAM_CONFIGS[streamType];

  return {
    from,
    to,
    color: getAgentColor(agentId),
    active: true,
    config: {
      count: config.count,
      speed: config.speed,
      size: config.size,
      lifetime: config.lifetime,
      emissionRate: config.emissionRate,
    },
  };
}

// =============================================================================
// Singleton Export
// =============================================================================

/** Global particle system instance */
export const particleSystem = new ParticleSystem();

export default ParticleSystem;
