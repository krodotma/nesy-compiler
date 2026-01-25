/**
 * MergeAnimation - Merge Beam Effect for Chromatic Agents Visualizer
 *
 * Step 11: Merge Animation Trigger
 *
 * When agent pushes successfully:
 * - Colored particles stream toward white main center
 * - Flash effect on completion
 * - Agent color fades to ghost state
 *
 * Uses P5.js for particle rendering.
 */

import type { AgentId } from './types';
import { AGENT_COLORS } from './utils/colorMap';
import { easeOutCubic, easeOutQuad } from './utils/interpolation';

// =============================================================================
// Types
// =============================================================================

interface Particle {
  /** X position */
  x: number;
  /** Y position */
  y: number;
  /** X velocity */
  vx: number;
  /** Y velocity */
  vy: number;
  /** Current size */
  size: number;
  /** Initial size */
  initialSize: number;
  /** Current alpha (0-255) */
  alpha: number;
  /** Lifetime remaining (seconds) */
  lifetime: number;
  /** Maximum lifetime */
  maxLifetime: number;
  /** Color components [r, g, b] */
  color: [number, number, number];
  /** Particle type */
  type: 'stream' | 'burst' | 'trail';
}

interface MergeEvent {
  /** Agent triggering the merge */
  agentId: AgentId;
  /** Agent color */
  color: string;
  /** Start Y position (agent row) */
  startY: number;
  /** Animation progress (0-1) */
  progress: number;
  /** Animation duration in seconds */
  duration: number;
  /** Elapsed time */
  elapsed: number;
  /** Associated particles */
  particles: Particle[];
  /** Flash phase (0-1, for final flash) */
  flashPhase: number;
  /** Is complete */
  complete: boolean;
}

// =============================================================================
// MergeAnimation Class
// =============================================================================

export class MergeAnimation {
  private width: number;
  private height: number;
  private activeEvents: MergeEvent[] = [];
  private frameCount: number = 0;

  // Target point (white main center)
  private targetX: number;
  private targetY: number;

  // Animation constants
  private readonly ANIMATION_DURATION = 1.5; // seconds
  private readonly PARTICLE_COUNT = 40;
  private readonly FLASH_DURATION = 0.3;
  private readonly MAX_CONCURRENT_EVENTS = 4;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;

    // Center target point (representing main branch)
    this.targetX = width - 60;
    this.targetY = height / 2;
  }

  /**
   * Trigger a new merge animation for an agent
   */
  trigger(agentId: AgentId, color: string, startY: number): void {
    // Limit concurrent events
    if (this.activeEvents.length >= this.MAX_CONCURRENT_EVENTS) {
      // Remove oldest event
      this.activeEvents.shift();
    }

    // Parse color to RGB
    const rgb = this.hexToRgb(color);

    // Create new merge event
    const event: MergeEvent = {
      agentId,
      color,
      startY,
      progress: 0,
      duration: this.ANIMATION_DURATION,
      elapsed: 0,
      particles: this.createParticles(startY, rgb),
      flashPhase: 0,
      complete: false,
    };

    this.activeEvents.push(event);
  }

  /**
   * Create initial particles for merge stream
   */
  private createParticles(
    startY: number,
    color: [number, number, number]
  ): Particle[] {
    const particles: Particle[] = [];
    const startX = 30;

    // Stream particles - main beam
    for (let i = 0; i < this.PARTICLE_COUNT; i++) {
      const delay = i / this.PARTICLE_COUNT;
      const lifetime = 0.8 + Math.random() * 0.4;

      particles.push({
        x: startX + Math.random() * 20,
        y: startY + (Math.random() - 0.5) * 20,
        vx: 0,
        vy: 0,
        size: 3 + Math.random() * 3,
        initialSize: 3 + Math.random() * 3,
        alpha: 0, // Will fade in based on delay
        lifetime: lifetime + delay,
        maxLifetime: lifetime + delay,
        color,
        type: 'stream',
      });
    }

    // Burst particles - spawn when beam hits target
    for (let i = 0; i < 15; i++) {
      const angle = (i / 15) * Math.PI * 2;
      const speed = 50 + Math.random() * 30;

      particles.push({
        x: this.targetX,
        y: this.targetY,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        size: 2 + Math.random() * 2,
        initialSize: 2 + Math.random() * 2,
        alpha: 0, // Will activate at completion
        lifetime: 0.5,
        maxLifetime: 0.5,
        color,
        type: 'burst',
      });
    }

    // Trail particles - leave a fading trail
    for (let i = 0; i < 10; i++) {
      particles.push({
        x: startX,
        y: startY + (Math.random() - 0.5) * 10,
        vx: 0,
        vy: 0,
        size: 1 + Math.random() * 2,
        initialSize: 1 + Math.random() * 2,
        alpha: 0,
        lifetime: 1.0,
        maxLifetime: 1.0,
        color,
        type: 'trail',
      });
    }

    return particles;
  }

  /**
   * Update all active merge animations
   */
  update(
    p: typeof import('p5').default.prototype,
    deltaTime: number
  ): void {
    this.frameCount++;

    for (const event of this.activeEvents) {
      if (event.complete) continue;

      event.elapsed += deltaTime;
      event.progress = Math.min(1, event.elapsed / event.duration);

      // Update particles
      this.updateParticles(event, deltaTime);

      // Start flash when animation nears completion
      if (event.progress > 0.8 && event.flashPhase < 1) {
        event.flashPhase = (event.progress - 0.8) / 0.2;
      }

      // Mark complete when animation finishes
      if (event.progress >= 1 && event.flashPhase >= 1) {
        event.complete = true;
      }
    }

    // Remove completed events
    this.activeEvents = this.activeEvents.filter(e => !e.complete);
  }

  /**
   * Update particles for a merge event
   */
  private updateParticles(event: MergeEvent, deltaTime: number): void {
    const startX = 30;
    const progress = event.progress;

    for (const particle of event.particles) {
      // Decrease lifetime
      particle.lifetime -= deltaTime;

      if (particle.type === 'stream') {
        // Stream particles: move toward target with easing
        const particleProgress = easeOutCubic(progress);
        const targetProgress = 1 - (particle.maxLifetime - (particle.maxLifetime - particle.lifetime)) / particle.maxLifetime;

        if (targetProgress > 0) {
          // Interpolate position toward target
          const t = Math.min(1, targetProgress * particleProgress * 2);
          particle.x = startX + (this.targetX - startX) * t;
          particle.y = event.startY + (this.targetY - event.startY) * t * easeOutQuad(t);

          // Add some waviness
          particle.y += Math.sin(this.frameCount * 0.1 + particle.x * 0.05) * 5 * (1 - t);

          // Fade in then out
          if (t < 0.2) {
            particle.alpha = (t / 0.2) * 255;
          } else if (t > 0.8) {
            particle.alpha = ((1 - t) / 0.2) * 255;
          } else {
            particle.alpha = 255;
          }

          // Size pulsing
          particle.size = particle.initialSize * (0.8 + Math.sin(this.frameCount * 0.2) * 0.2);
        }
      } else if (particle.type === 'burst') {
        // Burst particles: activate when progress > 0.9
        if (progress > 0.9) {
          // Update position based on velocity
          particle.x += particle.vx * deltaTime;
          particle.y += particle.vy * deltaTime;

          // Decay velocity
          particle.vx *= 0.95;
          particle.vy *= 0.95;

          // Fade out
          const lifetimeRatio = particle.lifetime / particle.maxLifetime;
          particle.alpha = lifetimeRatio * 255;
          particle.size = particle.initialSize * lifetimeRatio;
        }
      } else if (particle.type === 'trail') {
        // Trail particles: stationary, fade out slowly
        const lifetimeRatio = particle.lifetime / particle.maxLifetime;
        particle.alpha = lifetimeRatio * 150;
      }
    }
  }

  /**
   * Render all active merge animations
   */
  render(p: typeof import('p5').default.prototype): void {
    for (const event of this.activeEvents) {
      if (event.complete) continue;

      // Render beam line
      this.renderBeam(p, event);

      // Render particles
      this.renderParticles(p, event);

      // Render flash effect
      if (event.flashPhase > 0) {
        this.renderFlash(p, event);
      }
    }
  }

  /**
   * Render the main beam line
   */
  private renderBeam(
    p: typeof import('p5').default.prototype,
    event: MergeEvent
  ): void {
    const startX = 30;
    const beamProgress = easeOutCubic(event.progress);
    const beamEndX = startX + (this.targetX - startX) * beamProgress;
    const beamEndY = event.startY + (this.targetY - event.startY) * beamProgress * easeOutQuad(beamProgress);

    const color = p.color(event.color);

    // Outer glow
    p.stroke(
      p.red(color),
      p.green(color),
      p.blue(color),
      60
    );
    p.strokeWeight(8);
    p.line(startX, event.startY, beamEndX, beamEndY);

    // Middle layer
    p.stroke(
      p.red(color),
      p.green(color),
      p.blue(color),
      120
    );
    p.strokeWeight(4);
    p.line(startX, event.startY, beamEndX, beamEndY);

    // Core line
    p.stroke(255, 255, 255, 200);
    p.strokeWeight(2);
    p.line(startX, event.startY, beamEndX, beamEndY);
  }

  /**
   * Render particles
   */
  private renderParticles(
    p: typeof import('p5').default.prototype,
    event: MergeEvent
  ): void {
    p.noStroke();

    for (const particle of event.particles) {
      if (particle.alpha <= 0 || particle.lifetime <= 0) continue;

      const [r, g, b] = particle.color;

      if (particle.type === 'stream') {
        // Glowing stream particle
        p.fill(r, g, b, particle.alpha * 0.5);
        p.ellipse(particle.x, particle.y, particle.size * 2, particle.size * 2);

        p.fill(255, 255, 255, particle.alpha * 0.8);
        p.ellipse(particle.x, particle.y, particle.size * 0.8, particle.size * 0.8);
      } else if (particle.type === 'burst') {
        // Burst particle (outer glow + core)
        p.fill(r, g, b, particle.alpha * 0.3);
        p.ellipse(particle.x, particle.y, particle.size * 3, particle.size * 3);

        p.fill(r, g, b, particle.alpha);
        p.ellipse(particle.x, particle.y, particle.size, particle.size);

        p.fill(255, 255, 255, particle.alpha);
        p.ellipse(particle.x, particle.y, particle.size * 0.5, particle.size * 0.5);
      } else if (particle.type === 'trail') {
        // Simple trail dot
        p.fill(r, g, b, particle.alpha);
        p.ellipse(particle.x, particle.y, particle.size, particle.size);
      }
    }
  }

  /**
   * Render flash effect at target
   */
  private renderFlash(
    p: typeof import('p5').default.prototype,
    event: MergeEvent
  ): void {
    const flashAlpha = (1 - event.flashPhase) * 200;
    const flashSize = 30 + event.flashPhase * 50;
    const color = p.color(event.color);

    // Outer glow
    p.fill(
      p.red(color),
      p.green(color),
      p.blue(color),
      flashAlpha * 0.3
    );
    p.noStroke();
    p.ellipse(this.targetX, this.targetY, flashSize * 2, flashSize * 2);

    // Middle ring
    p.fill(
      p.red(color),
      p.green(color),
      p.blue(color),
      flashAlpha * 0.5
    );
    p.ellipse(this.targetX, this.targetY, flashSize * 1.2, flashSize * 1.2);

    // White core flash
    p.fill(255, 255, 255, flashAlpha);
    p.ellipse(this.targetX, this.targetY, flashSize * 0.6, flashSize * 0.6);

    // Draw rays
    const rayCount = 8;
    for (let i = 0; i < rayCount; i++) {
      const angle = (i / rayCount) * Math.PI * 2 + this.frameCount * 0.02;
      const rayLength = flashSize * (1 - event.flashPhase);

      const x1 = this.targetX + Math.cos(angle) * 10;
      const y1 = this.targetY + Math.sin(angle) * 10;
      const x2 = this.targetX + Math.cos(angle) * rayLength;
      const y2 = this.targetY + Math.sin(angle) * rayLength;

      p.stroke(255, 255, 255, flashAlpha * 0.5);
      p.strokeWeight(2);
      p.line(x1, y1, x2, y2);
    }
  }

  /**
   * Convert hex color to RGB tuple
   */
  private hexToRgb(hex: string): [number, number, number] {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return [r, g, b];
  }

  /**
   * Resize the animation canvas
   */
  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.targetX = width - 60;
    this.targetY = height / 2;
  }

  /**
   * Check if any animations are active
   */
  isActive(): boolean {
    return this.activeEvents.length > 0;
  }

  /**
   * Clear all active animations
   */
  clear(): void {
    this.activeEvents = [];
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create a merge animation handler
 */
export function createMergeAnimation(
  width: number,
  height: number
): MergeAnimation {
  return new MergeAnimation(width, height);
}

/**
 * Helper to trigger merge on successful push event
 */
export function onPushSuccess(
  mergeAnimation: MergeAnimation,
  agentId: AgentId,
  agentRowY: number
): void {
  const color = AGENT_COLORS[agentId]?.hex ?? '#888888';
  mergeAnimation.trigger(agentId, color, agentRowY);
}
