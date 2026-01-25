/**
 * Chromatic Agents Visualizer - Type Definitions
 *
 * Bus event schemas and data structures for the prism metaphor visualization.
 * Main (white) refracts into chromatic agent branches, merging back on completion.
 */

// =============================================================================
// Agent Visual States (Step 12 from plan)
// =============================================================================

export enum AgentVisualState {
  IDLE = 'idle',           // No clone, invisible/ghosted
  CLONING = 'cloning',     // Particles spawning, tree growing
  WORKING = 'working',     // Full color, pulsing with activity
  COMMITTING = 'committing', // Brief flash, commit particle burst
  PUSHING = 'pushing',     // Beam toward remote (off-screen glow)
  MERGED = 'merged',       // Color drains to white, merges with main
  CLEANUP = 'cleanup',     // Fade out, particles dissipate
}

// =============================================================================
// Code Graph Structures (Step 3)
// =============================================================================

export interface CodeNode {
  /** File or module path */
  path: string;
  /** Number of lines changed */
  lines_changed: number;
  /** Node type for visual differentiation */
  node_type: 'file' | 'module' | 'directory';
  /** Import/dependency connections */
  dependencies: string[];
  /** Last modification timestamp (ISO) */
  last_modified_iso: string;
  /** Weight for visual sizing: sqrt(lines_changed) */
  visual_weight: number;
}

export interface CodeGraph {
  /** Root path of the graph */
  root: string;
  /** All nodes in the graph */
  nodes: CodeNode[];
  /** Edges as [source_path, target_path] tuples */
  edges: [string, string][];
  /** Graph timestamp */
  timestamp_iso: string;
}

// =============================================================================
// Agent Visual Event (Step 1 - main schema)
// =============================================================================

export interface AgentVisualEvent {
  [key: string]: unknown;
  /** Agent identifier: claude, qwen, gemini, codex */
  agent_id: AgentId;
  /** Git branch name: e.g., CLAUDE_FINAL_ASSERTION */
  branch: string;
  /** Clone path on filesystem: e.g., /tmp/pluribus_claude_1734740000 */
  clone_path: string;
  /** Current visual state */
  state: AgentVisualState;
  /** HSL hue value 0-360 (assigned per agent) */
  color_hue: number;
  /** Current code graph snapshot */
  code_graph: CodeGraph | null;
  /** Activity intensity 0-1 (maps to saturation/glow) */
  activity_intensity: number;
  /** Event timestamp */
  timestamp_iso: string;
  /** Optional trace ID for correlation */
  trace_id?: string;
}

// =============================================================================
// Agent Identifiers (Step 2)
// =============================================================================

export type AgentId = 'claude' | 'qwen' | 'gemini' | 'codex' | 'main';

// =============================================================================
// Bus Topics (Step 4)
// =============================================================================

export const VIZ_TOPICS = [
  'paip.clone.created',
  'paip.clone.deleted',
  'agent.codegraph.update',
  'git.commit.created',
  'git.push.completed',
  'operator.pbrealityfix.broadcast',
  'viz.agent.state',
  'viz.codegraph.update',
  'viz.merge.animation',
  'viz.user.interaction',
] as const;

export type VizTopic = typeof VIZ_TOPICS[number];

// =============================================================================
// Visualization State
// =============================================================================

export interface AgentVisualData {
  /** Agent identifier */
  id: AgentId;
  /** Current state */
  state: AgentVisualState;
  /** HSL hue for this agent */
  hue: number;
  /** Hex color string */
  color: string;
  /** Activity level 0-1 */
  intensity: number;
  /** Current code graph */
  codeGraph: CodeGraph | null;
  /** Branch name if active */
  branch: string | null;
  /** Position in 3D space [x, y, z] */
  position: [number, number, number];
  /** Visibility (for transitions) */
  opacity: number;
  /** Last update timestamp */
  lastUpdate: number;
}

export interface ChromaticState {
  /** All agent visual states */
  agents: Map<AgentId, AgentVisualData>;
  /** Central prism activity (0-1) */
  prismIntensity: number;
  /** Currently focused agent (for camera) */
  focusedAgent: AgentId | null;
  /** Bus connection status */
  busConnected: boolean;
  /** Events per minute metric */
  eventsPerMinute: number;
  /** Main branch ahead count */
  mainAhead: number;
}

// =============================================================================
// Visual Mutations (Step 13)
// =============================================================================

export interface VisualMutation {
  type:
    | 'state_change'
    | 'intensity_change'
    | 'position_change'
    | 'spawn'
    | 'despawn'
    | 'merge'
    | 'code_graph_update'
    | 'particle_burst'
    | 'focus_change';
  agent_id: AgentId;
  timestamp: number;
  payload: Record<string, unknown>;
  /** Apply this mutation to the scene */
  apply: (state: ChromaticState, deltaTime: number) => void;
}

// =============================================================================
// Prism Core State
// =============================================================================

export interface PrismCoreState {
  /** Rotation angle (radians) */
  rotation: number;
  /** Pulse phase (0-2PI) */
  pulsePhase: number;
  /** Active refraction beams */
  activeBeams: AgentId[];
  /** Overall intensity (aggregate of all agents) */
  intensity: number;
}

// =============================================================================
// Tree Node Visual State
// =============================================================================

export interface TreeNodeVisual {
  /** Unique node ID */
  id: string;
  /** File/module path */
  path: string;
  /** 3D position */
  position: [number, number, number];
  /** Sphere radius (based on lines_changed) */
  radius: number;
  /** Current opacity (for fade effects) */
  opacity: number;
  /** Emissive intensity */
  emissive: number;
  /** Age in seconds (for recency fade) */
  age: number;
}

// =============================================================================
// Particle System Types
// =============================================================================

export interface ParticleConfig {
  /** Particle count */
  count: number;
  /** Base color (hex) */
  color: string;
  /** Particle speed */
  speed: number;
  /** Particle size */
  size: number;
  /** Lifetime in seconds */
  lifetime: number;
  /** Emission rate per second */
  emissionRate: number;
}

export interface ParticleStream {
  /** Source position */
  from: [number, number, number];
  /** Target position */
  to: [number, number, number];
  /** Stream color */
  color: string;
  /** Active state */
  active: boolean;
  /** Particle config override */
  config: Partial<ParticleConfig>;
}

// =============================================================================
// Camera Controls State
// =============================================================================

export interface CameraState {
  /** Current target position */
  target: [number, number, number];
  /** Camera position */
  position: [number, number, number];
  /** Zoom level */
  zoom: number;
  /** Is transitioning */
  transitioning: boolean;
  /** Auto-follow most active agent */
  autoFollow: boolean;
}

// =============================================================================
// HUD Types (Step 9)
// =============================================================================

export interface AgentHUDData {
  id: AgentId;
  name: string;
  color: string;
  progress: number; // 0-100
  state: AgentVisualState;
  linesChanged: number;
  commitsCount: number;
}

export interface HUDState {
  agents: AgentHUDData[];
  eventsPerMinute: number;
  mainAhead: number;
  busStatus: 'connected' | 'disconnected' | 'reconnecting';
}
