/**
 * PrismCore - Central Prism Object for Chromatic Visualizer
 *
 * Step 6: Central prism geometry that:
 * - Refracts "white" main into colored agent rays when PAIP activates
 * - Pulses with bus activity intensity
 * - Rotates slowly (aesthetic)
 * - Emits particle streams toward active agent trees
 *
 * This is a vanilla Three.js implementation for Qwik compatibility.
 */

import * as THREE from 'three';
import { AGENT_COLORS, getAgentOrder, getAgentOrbitalPosition } from './utils/colorMap';
import type { AgentId, PrismCoreState } from './types';
// @ts-ignore
import smokedGlassFrag from './shaders/smoked-glass.frag?raw';

// =============================================================================
// Prism Geometry Creation
// =============================================================================

/**
 * Create a triangular prism geometry
 */
function createPrismGeometry(size: number = 1, height: number = 2): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();

  // Triangular prism vertices
  const halfSize = size / 2;
  const halfHeight = height / 2;

  // Triangle at y = -halfHeight
  const bottomTriangle = [
    [0, -halfHeight, -halfSize * Math.sqrt(3) / 3],           // Back vertex
    [-halfSize, -halfHeight, halfSize * Math.sqrt(3) / 6],     // Front left
    [halfSize, -halfHeight, halfSize * Math.sqrt(3) / 6],      // Front right
  ];

  // Triangle at y = halfHeight
  const topTriangle = [
    [0, halfHeight, -halfSize * Math.sqrt(3) / 3],
    [-halfSize, halfHeight, halfSize * Math.sqrt(3) / 6],
    [halfSize, halfHeight, halfSize * Math.sqrt(3) / 6],
  ];

  // Build faces (each face is two triangles)
  const vertices: number[] = [];
  const normals: number[] = [];
  const uvs: number[] = [];

  // Bottom face (pointing down)
  addFace(vertices, normals, uvs,
    bottomTriangle[0], bottomTriangle[2], bottomTriangle[1],
    [0, -1, 0]
  );

  // Top face (pointing up)
  addFace(vertices, normals, uvs,
    topTriangle[0], topTriangle[1], topTriangle[2],
    [0, 1, 0]
  );

  // Three side faces
  for (let i = 0; i < 3; i++) {
    const next = (i + 1) % 3;
    const normal = calculateNormal(
      bottomTriangle[i] as [number, number, number],
      bottomTriangle[next] as [number, number, number],
      topTriangle[i] as [number, number, number]
    );

    // Two triangles per side
    addFace(vertices, normals, uvs,
      bottomTriangle[i], bottomTriangle[next], topTriangle[i],
      normal
    );
    addFace(vertices, normals, uvs,
      bottomTriangle[next], topTriangle[next], topTriangle[i],
      normal
    );
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

  return geometry;
}

function addFace(
  vertices: number[],
  normals: number[],
  uvs: number[],
  v0: number[],
  v1: number[],
  v2: number[],
  normal: number[]
): void {
  vertices.push(...v0, ...v1, ...v2);
  normals.push(...normal, ...normal, ...normal);
  uvs.push(0, 0, 1, 0, 0.5, 1);
}

function calculateNormal(
  v0: [number, number, number],
  v1: [number, number, number],
  v2: [number, number, number]
): number[] {
  const ax = v1[0] - v0[0], ay = v1[1] - v0[1], az = v1[2] - v0[2];
  const bx = v2[0] - v0[0], by = v2[1] - v0[1], bz = v2[2] - v0[2];
  const nx = ay * bz - az * by;
  const ny = az * bx - ax * bz;
  const nz = ax * by - ay * bx;
  const len = Math.sqrt(nx * nx + ny * ny + nz * nz);
  return [nx / len, ny / len, nz / len];
}

// =============================================================================
// Refraction Beam Class
// =============================================================================

class RefractionBeam {
  public readonly agentId: AgentId;
  public readonly mesh: THREE.Mesh;
  public intensity: number = 0;
  private targetIntensity: number = 0;

  constructor(agentId: AgentId, scene: THREE.Scene) {
    this.agentId = agentId;

    const targetPos = getAgentOrbitalPosition(agentId, 5);
    const color = new THREE.Color(AGENT_COLORS[agentId].hex);

    // Create beam geometry (thin cylinder from center to agent position)
    const direction = new THREE.Vector3(...targetPos).normalize();
    const length = Math.sqrt(
      targetPos[0] ** 2 + targetPos[1] ** 2 + targetPos[2] ** 2
    );

    const geometry = new THREE.CylinderGeometry(0.02, 0.05, length, 8);
    geometry.translate(0, length / 2, 0);
    geometry.rotateX(Math.PI / 2);

    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.lookAt(direction);
    this.mesh.visible = false;

    scene.add(this.mesh);
  }

  setActive(active: boolean): void {
    this.targetIntensity = active ? 1 : 0;
    this.mesh.visible = active || this.intensity > 0.01;
  }

  update(deltaTime: number): void {
    // Smooth intensity transition
    const speed = 3;
    this.intensity += (this.targetIntensity - this.intensity) * speed * deltaTime;

    const material = this.mesh.material as THREE.MeshBasicMaterial;
    material.opacity = this.intensity * 0.6;

    if (this.intensity < 0.01 && this.targetIntensity === 0) {
      this.mesh.visible = false;
    }
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

// =============================================================================
// PrismCore Class
// =============================================================================

export class PrismCore {
  public readonly group: THREE.Group;
  public state: PrismCoreState;

  private prismMesh: THREE.Mesh;
  private glowMesh: THREE.Mesh;
  private beams: Map<AgentId, RefractionBeam>;
  private particleSystems: Map<AgentId, THREE.Points>;
  private material: THREE.ShaderMaterial;

  constructor(scene: THREE.Scene) {
    this.group = new THREE.Group();
    this.state = {
      rotation: 0,
      pulsePhase: 0,
      activeBeams: [],
      intensity: 0.5,
    };

    // Create main prism with custom ShaderMaterial (Phase 1 Step 4)
    const prismGeometry = createPrismGeometry(1.5, 2.5);
    
    this.material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uColorStart: { value: new THREE.Color(0x111111) },
        uColorEnd: { value: new THREE.Color(0x050505) },
        uColorPrimary: { value: new THREE.Color(0xffffff) },
        uNoiseScale: { value: 2.0 },
        uNoiseSpeed: { value: 0.2 },
        uIntensity: { value: 0.5 },
      },
      vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal;
        varying vec3 vViewPosition;
        void main() {
          vUv = uv;
          vNormal = normalize(normalMatrix * normal);
          vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
          vViewPosition = -mvPosition.xyz;
          gl_Position = projectionMatrix * mvPosition;
        }
      `,
      fragmentShader: smokedGlassFrag,
      transparent: true,
    });

    this.prismMesh = new THREE.Mesh(prismGeometry, this.material);
    this.group.add(this.prismMesh);

    // Create glow mesh (slightly larger, additive)
    const glowGeometry = createPrismGeometry(1.6, 2.6);
    const glowMaterial = new THREE.MeshBasicMaterial({
      color: 0xffffff,
      transparent: true,
      opacity: 0.15,
      blending: THREE.AdditiveBlending,
      side: THREE.BackSide,
    });

    this.glowMesh = new THREE.Mesh(glowGeometry, glowMaterial);
    this.group.add(this.glowMesh);

    // Create refraction beams for each agent
    this.beams = new Map();
    this.particleSystems = new Map();

    for (const agentId of getAgentOrder()) {
      const beam = new RefractionBeam(agentId, scene);
      this.beams.set(agentId, beam);

      // Create particle system for this agent
      const particles = this.createParticleSystem(agentId);
      this.particleSystems.set(agentId, particles);
      scene.add(particles);
    }

    scene.add(this.group);
  }

  private createParticleSystem(agentId: AgentId): THREE.Points {
    const particleCount = 50;
    const positions = new Float32Array(particleCount * 3);
    const velocities = new Float32Array(particleCount * 3);
    const lifetimes = new Float32Array(particleCount);

    // Initialize particles at center with random velocities toward agent
    const targetPos = getAgentOrbitalPosition(agentId, 5);

    for (let i = 0; i < particleCount; i++) {
      const i3 = i * 3;
      positions[i3] = 0;
      positions[i3 + 1] = 0;
      positions[i3 + 2] = 0;

      velocities[i3] = targetPos[0] * (0.5 + Math.random() * 0.5);
      velocities[i3 + 1] = targetPos[1] * (0.5 + Math.random() * 0.5) + (Math.random() - 0.5);
      velocities[i3 + 2] = targetPos[2] * (0.5 + Math.random() * 0.5);

      lifetimes[i] = Math.random();
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.userData.velocities = velocities;
    geometry.userData.lifetimes = lifetimes;

    const material = new THREE.PointsMaterial({
      color: new THREE.Color(AGENT_COLORS[agentId].hex),
      size: 0.08,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });

    const points = new THREE.Points(geometry, material);
    points.visible = false;
    points.userData.agentId = agentId;

    return points;
  }

  /**
   * Activate refraction beam for an agent
   */
  activateBeam(agentId: AgentId): void {
    const beam = this.beams.get(agentId);
    if (beam) {
      beam.setActive(true);
      if (!this.state.activeBeams.includes(agentId)) {
        this.state.activeBeams.push(agentId);
      }
    }

    const particles = this.particleSystems.get(agentId);
    if (particles) {
      particles.visible = true;
      (particles.material as THREE.PointsMaterial).opacity = 0.8;
    }
  }

  /**
   * Deactivate refraction beam for an agent
   */
  deactivateBeam(agentId: AgentId): void {
    const beam = this.beams.get(agentId);
    if (beam) {
      beam.setActive(false);
    }

    const idx = this.state.activeBeams.indexOf(agentId);
    if (idx !== -1) {
      this.state.activeBeams.splice(idx, 1);
    }

    const particles = this.particleSystems.get(agentId);
    if (particles) {
      (particles.material as THREE.PointsMaterial).opacity = 0;
    }
  }

  /**
   * Set overall activity intensity (0-1)
   */
  setIntensity(intensity: number): void {
    this.state.intensity = Math.max(0, Math.min(1, intensity));
  }

  /**
   * Update prism animation state
   */
  update(deltaTime: number): void {
    // Update uniforms
    this.material.uniforms.uTime.value += deltaTime;
    this.material.uniforms.uIntensity.value = this.state.intensity;

    // Slow rotation
    this.state.rotation += deltaTime * 0.2;
    this.group.rotation.y = this.state.rotation;

    // Pulse effect
    this.state.pulsePhase += deltaTime * 2;
    const pulse = 0.5 + 0.5 * Math.sin(this.state.pulsePhase);
    const pulseIntensity = this.state.intensity * pulse;

    // Update glow opacity based on intensity
    const glowMaterial = this.glowMesh.material as THREE.MeshBasicMaterial;
    glowMaterial.opacity = 0.1 + pulseIntensity * 0.2;

    // Scale pulse
    const scale = 1 + pulseIntensity * 0.05;
    this.glowMesh.scale.setScalar(scale);

    // Update all beams
    for (const beam of this.beams.values()) {
      beam.update(deltaTime);
    }

    // Update particle systems
    for (const particles of this.particleSystems.values()) {
      this.updateParticles(particles, deltaTime);
    }
  }

  private updateParticles(particles: THREE.Points, deltaTime: number): void {
    if (!particles.visible) return;

    const positions = particles.geometry.attributes.position.array as Float32Array;
    const velocities = particles.geometry.userData.velocities as Float32Array;
    const lifetimes = particles.geometry.userData.lifetimes as Float32Array;
    const agentId = particles.userData.agentId as AgentId;
    const targetPos = getAgentOrbitalPosition(agentId, 5);

    for (let i = 0; i < lifetimes.length; i++) {
      const i3 = i * 3;

      // Update lifetime
      lifetimes[i] -= deltaTime * 0.5;

      if (lifetimes[i] <= 0) {
        // Reset particle
        positions[i3] = 0;
        positions[i3 + 1] = 0;
        positions[i3 + 2] = 0;
        lifetimes[i] = 1;

        // New random velocity toward target
        velocities[i3] = targetPos[0] * (0.3 + Math.random() * 0.7);
        velocities[i3 + 1] = targetPos[1] * (0.3 + Math.random() * 0.7) + (Math.random() - 0.5) * 0.5;
        velocities[i3 + 2] = targetPos[2] * (0.3 + Math.random() * 0.7);
      } else {
        // Move particle
        positions[i3] += velocities[i3] * deltaTime;
        positions[i3 + 1] += velocities[i3 + 1] * deltaTime;
        positions[i3 + 2] += velocities[i3 + 2] * deltaTime;
      }
    }

    particles.geometry.attributes.position.needsUpdate = true;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.prismMesh.geometry.dispose();
    (this.prismMesh.material as THREE.Material).dispose();
    this.glowMesh.geometry.dispose();
    (this.glowMesh.material as THREE.Material).dispose();

    for (const beam of this.beams.values()) {
      beam.dispose();
    }

    for (const particles of this.particleSystems.values()) {
      particles.geometry.dispose();
      (particles.material as THREE.Material).dispose();
    }
  }
}

export default PrismCore;
