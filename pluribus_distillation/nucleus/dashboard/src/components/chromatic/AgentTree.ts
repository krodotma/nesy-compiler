/**
 * AgentTree - Per-Agent Tree Renderer
 *
 * Step 7: Tree Node Representation
 * Each code node (file/module) as:
 * - Sphere with emissive material (color = agent hue)
 * - Size = sqrt(lines_changed)
 * - Opacity = recency (fades over time)
 * - Connected by glowing lines (imports/deps)
 * - Bloom post-processing for neon effect
 *
 * This is a vanilla Three.js implementation for Qwik compatibility.
 */

import * as THREE from 'three';
import {
  AGENT_COLORS,
  getAgentColor,
  getAgentOrbitalPosition,
  getAgentRGBNormalized,
} from './utils/colorMap';
import { exponentialDecay, lerp3D, easeOutQuad } from './utils/interpolation';
import { AgentVisualState } from './types';
import type { AgentId, CodeGraph, CodeNode, TreeNodeVisual } from './types';

// =============================================================================
// Constants
// =============================================================================

const NODE_BASE_RADIUS = 0.08;
const NODE_MAX_RADIUS = 0.4;
const EDGE_LINE_WIDTH = 2;
const FADE_RATE = 0.5; // Opacity decay rate
const MAX_NODES = 100;

// =============================================================================
// TreeNode Class
// =============================================================================

class TreeNode {
  public readonly id: string;
  public readonly path: string;
  public mesh: THREE.Mesh;
  public visual: TreeNodeVisual;

  private targetPosition: [number, number, number];
  private targetRadius: number;
  private targetOpacity: number;

  constructor(
    codeNode: CodeNode,
    agentId: AgentId,
    basePosition: [number, number, number],
    index: number
  ) {
    this.id = codeNode.path;
    this.path = codeNode.path;

    // Calculate position relative to agent base position
    // Arrange nodes in a spiral pattern
    const angle = (index * 137.5 * Math.PI) / 180; // Golden angle
    const radius = 0.5 + Math.sqrt(index) * 0.3;
    const height = (Math.random() - 0.5) * 2;

    this.targetPosition = [
      basePosition[0] + Math.cos(angle) * radius,
      basePosition[1] + height,
      basePosition[2] + Math.sin(angle) * radius,
    ];

    // Calculate visual size based on lines changed
    this.targetRadius = Math.min(
      NODE_MAX_RADIUS,
      NODE_BASE_RADIUS + Math.sqrt(codeNode.lines_changed) * 0.02
    );

    this.targetOpacity = 1;

    // Initialize visual state
    this.visual = {
      id: this.id,
      path: this.path,
      position: [...this.targetPosition] as [number, number, number],
      radius: this.targetRadius,
      opacity: 0, // Start invisible
      emissive: 0.5,
      age: 0,
    };

    // Create mesh
    const geometry = new THREE.SphereGeometry(1, 16, 12);
    const color = new THREE.Color(getAgentColor(agentId));

    const material = new THREE.MeshStandardMaterial({
      color,
      emissive: color,
      emissiveIntensity: 0.5,
      transparent: true,
      opacity: 0,
      metalness: 0.3,
      roughness: 0.4,
    });

    this.mesh = new THREE.Mesh(geometry, material);
    this.mesh.scale.setScalar(this.targetRadius);
    this.mesh.position.set(...this.targetPosition);
    this.mesh.userData.treeNode = this;
  }

  update(deltaTime: number): void {
    this.visual.age += deltaTime;

    // Fade in animation
    if (this.visual.opacity < this.targetOpacity) {
      this.visual.opacity = Math.min(
        this.targetOpacity,
        this.visual.opacity + deltaTime * 2
      );
    }

    // Recency fade (after 30 seconds, start fading)
    if (this.visual.age > 30) {
      this.targetOpacity = Math.max(0.2, 1 - (this.visual.age - 30) / 60);
      this.visual.opacity = exponentialDecay(
        this.visual.opacity,
        this.targetOpacity,
        FADE_RATE,
        deltaTime
      );
    }

    // Smooth position transition
    this.visual.position = lerp3D(
      this.visual.position,
      this.targetPosition,
      deltaTime * 2,
      easeOutQuad
    );

    // Update mesh
    this.mesh.position.set(...this.visual.position);
    this.mesh.scale.setScalar(this.visual.radius);

    const material = this.mesh.material as THREE.MeshStandardMaterial;
    material.opacity = this.visual.opacity;
    material.emissiveIntensity = this.visual.emissive * this.visual.opacity;
  }

  setHighlighted(highlighted: boolean): void {
    this.visual.emissive = highlighted ? 1 : 0.5;
    const material = this.mesh.material as THREE.MeshStandardMaterial;
    material.emissiveIntensity = this.visual.emissive;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    (this.mesh.material as THREE.Material).dispose();
  }
}

// =============================================================================
// AgentTree Class
// =============================================================================

export class AgentTree {
  public readonly agentId: AgentId;
  public readonly group: THREE.Group;
  public state: AgentVisualState;
  public basePosition: [number, number, number];

  private nodes: Map<string, TreeNode>;
  private edges: THREE.Line[];
  private edgeGroup: THREE.Group;
  private intensity: number = 0;
  private targetIntensity: number = 0;
  private visible: boolean = false;

  constructor(agentId: AgentId, scene: THREE.Scene) {
    this.agentId = agentId;
    this.group = new THREE.Group();
    this.edgeGroup = new THREE.Group();
    this.nodes = new Map();
    this.edges = [];
    this.state = AgentVisualState.IDLE;
    this.basePosition = getAgentOrbitalPosition(agentId, 5);

    this.group.position.set(...this.basePosition);
    this.group.add(this.edgeGroup);

    // Initially hidden
    this.group.visible = false;

    scene.add(this.group);
  }

  /**
   * Update the tree with a new code graph
   */
  updateCodeGraph(codeGraph: CodeGraph | null): void {
    if (!codeGraph) {
      this.clear();
      return;
    }

    // Add or update nodes
    const seenPaths = new Set<string>();

    for (let i = 0; i < Math.min(codeGraph.nodes.length, MAX_NODES); i++) {
      const codeNode = codeGraph.nodes[i];
      seenPaths.add(codeNode.path);

      if (!this.nodes.has(codeNode.path)) {
        // Create new node
        const treeNode = new TreeNode(
          codeNode,
          this.agentId,
          [0, 0, 0], // Relative to group
          i
        );
        this.nodes.set(codeNode.path, treeNode);
        this.group.add(treeNode.mesh);
      } else {
        // Update existing node (reset age for activity)
        const treeNode = this.nodes.get(codeNode.path)!;
        treeNode.visual.age = 0;
      }
    }

    // Remove old nodes not in current graph
    for (const [path, node] of this.nodes) {
      if (!seenPaths.has(path)) {
        this.group.remove(node.mesh);
        node.dispose();
        this.nodes.delete(path);
      }
    }

    // Rebuild edges
    this.rebuildEdges(codeGraph.edges);
  }

  private rebuildEdges(edgeList: [string, string][]): void {
    // Clear old edges
    for (const edge of this.edges) {
      this.edgeGroup.remove(edge);
      edge.geometry.dispose();
      (edge.material as THREE.Material).dispose();
    }
    this.edges = [];

    const color = new THREE.Color(getAgentColor(this.agentId));

    for (const [fromPath, toPath] of edgeList) {
      const fromNode = this.nodes.get(fromPath);
      const toNode = this.nodes.get(toPath);

      if (!fromNode || !toNode) continue;

      const points = [
        new THREE.Vector3(...fromNode.visual.position),
        new THREE.Vector3(...toNode.visual.position),
      ];

      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color,
        transparent: true,
        opacity: 0.3,
        linewidth: EDGE_LINE_WIDTH,
      });

      const line = new THREE.Line(geometry, material);
      this.edges.push(line);
      this.edgeGroup.add(line);
    }
  }

  /**
   * Set the visual state
   */
  setState(state: AgentVisualState): void {
    this.state = state;

    switch (state) {
      case AgentVisualState.IDLE:
        this.targetIntensity = 0;
        break;
      case AgentVisualState.CLONING:
        this.targetIntensity = 0.3;
        this.show();
        break;
      case AgentVisualState.WORKING:
        this.targetIntensity = 1;
        this.show();
        break;
      case AgentVisualState.COMMITTING:
        this.targetIntensity = 1.2; // Extra bright flash
        break;
      case AgentVisualState.PUSHING:
        this.targetIntensity = 0.8;
        break;
      case AgentVisualState.MERGED:
        this.targetIntensity = 0;
        break;
      case AgentVisualState.CLEANUP:
        this.targetIntensity = 0;
        break;
    }
  }

  /**
   * Set activity intensity (0-1)
   */
  setIntensity(intensity: number): void {
    this.targetIntensity = Math.max(0, Math.min(1.2, intensity));
  }

  /**
   * Show the tree
   */
  show(): void {
    this.visible = true;
    this.group.visible = true;
  }

  /**
   * Hide the tree
   */
  hide(): void {
    this.visible = false;
    // Don't immediately hide - let fade out
  }

  /**
   * Clear all nodes
   */
  clear(): void {
    for (const node of this.nodes.values()) {
      this.group.remove(node.mesh);
      node.dispose();
    }
    this.nodes.clear();

    for (const edge of this.edges) {
      this.edgeGroup.remove(edge);
      edge.geometry.dispose();
      (edge.material as THREE.Material).dispose();
    }
    this.edges = [];
  }

  /**
   * Update animation state
   */
  update(deltaTime: number): void {
    // Smooth intensity transition
    this.intensity += (this.targetIntensity - this.intensity) * deltaTime * 3;

    // Update all nodes
    for (const node of this.nodes.values()) {
      node.update(deltaTime);

      // Apply intensity to emissive
      const material = node.mesh.material as THREE.MeshStandardMaterial;
      material.emissiveIntensity = node.visual.emissive * this.intensity;
    }

    // Update edge opacity
    for (const edge of this.edges) {
      const material = edge.material as THREE.LineBasicMaterial;
      material.opacity = 0.3 * this.intensity;
    }

    // Update edge positions (in case nodes moved)
    this.updateEdgePositions();

    // Hide group if fully faded out
    if (this.intensity < 0.01 && !this.visible) {
      this.group.visible = false;
    }

    // Gentle rotation for visual interest
    this.group.rotation.y += deltaTime * 0.1;
  }

  private updateEdgePositions(): void {
    const nodesList = Array.from(this.nodes.values());
    let edgeIndex = 0;

    for (const edge of this.edges) {
      if (edgeIndex * 2 + 1 >= nodesList.length) break;

      const positions = edge.geometry.attributes.position.array as Float32Array;
      const fromNode = nodesList[edgeIndex * 2];
      const toNode = nodesList[edgeIndex * 2 + 1];

      if (fromNode && toNode) {
        positions[0] = fromNode.visual.position[0];
        positions[1] = fromNode.visual.position[1];
        positions[2] = fromNode.visual.position[2];
        positions[3] = toNode.visual.position[0];
        positions[4] = toNode.visual.position[1];
        positions[5] = toNode.visual.position[2];
        edge.geometry.attributes.position.needsUpdate = true;
      }

      edgeIndex++;
    }
  }

  /**
   * Get the center of mass of all nodes
   */
  getCenterOfMass(): [number, number, number] {
    if (this.nodes.size === 0) {
      return [...this.basePosition] as [number, number, number];
    }

    let x = 0, y = 0, z = 0;
    for (const node of this.nodes.values()) {
      x += node.visual.position[0];
      y += node.visual.position[1];
      z += node.visual.position[2];
    }

    const count = this.nodes.size;
    return [
      this.basePosition[0] + x / count,
      this.basePosition[1] + y / count,
      this.basePosition[2] + z / count,
    ];
  }

  /**
   * Get node count
   */
  getNodeCount(): number {
    return this.nodes.size;
  }

  /**
   * Clean up resources
   */
  dispose(): void {
    this.clear();
  }
}

export default AgentTree;
