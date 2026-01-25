/**
 * Chromatic Agents Visualizer - Agent Color Mapping
 *
 * Step 2: Map each pillar agent to a distinct chromatic hue.
 *
 * Claude  -> Magenta (300 deg) - #FF00FF neon
 * Qwen    -> Cyan (180 deg)    - #00FFFF neon
 * Gemini  -> Yellow (60 deg)   - #FFFF00 neon
 * Codex   -> Green (120 deg)   - #00FF00 neon
 * Main    -> White (achromatic) - #FFFFFF
 */

import type { AgentId } from '../types';

// =============================================================================
// Color Constants
// =============================================================================

export const AGENT_COLORS: Record<AgentId, {
  hex: string;
  hue: number;
  rgb: [number, number, number];
  hsl: [number, number, number];
  name: string;
}> = {
  claude: {
    hex: '#FF00FF',
    hue: 300,
    rgb: [255, 0, 255],
    hsl: [300, 100, 50],
    name: 'Magenta',
  },
  qwen: {
    hex: '#00FFFF',
    hue: 180,
    rgb: [0, 255, 255],
    hsl: [180, 100, 50],
    name: 'Cyan',
  },
  gemini: {
    hex: '#FFFF00',
    hue: 60,
    rgb: [255, 255, 0],
    hsl: [60, 100, 50],
    name: 'Yellow',
  },
  codex: {
    hex: '#00FF00',
    hue: 120,
    rgb: [0, 255, 0],
    hsl: [120, 100, 50],
    name: 'Green',
  },
  main: {
    hex: '#FFFFFF',
    hue: 0,
    rgb: [255, 255, 255],
    hsl: [0, 0, 100],
    name: 'White',
  },
};

// =============================================================================
// Color Utility Functions
// =============================================================================

/**
 * Get the hex color for an agent
 */
export function getAgentColor(agentId: AgentId): string {
  return AGENT_COLORS[agentId]?.hex ?? '#888888';
}

/**
 * Get the HSL hue for an agent (0-360)
 */
export function getAgentHue(agentId: AgentId): number {
  return AGENT_COLORS[agentId]?.hue ?? 0;
}

/**
 * Get RGB values for an agent (0-255 each)
 */
export function getAgentRGB(agentId: AgentId): [number, number, number] {
  return AGENT_COLORS[agentId]?.rgb ?? [128, 128, 128];
}

/**
 * Get normalized RGB values for Three.js (0-1 each)
 */
export function getAgentRGBNormalized(agentId: AgentId): [number, number, number] {
  const rgb = getAgentRGB(agentId);
  return [rgb[0] / 255, rgb[1] / 255, rgb[2] / 255];
}

/**
 * Apply intensity (0-1) to agent color, returning modified hex
 */
export function getAgentColorWithIntensity(agentId: AgentId, intensity: number): string {
  if (agentId === 'main') {
    // White with intensity = adjust lightness
    const l = Math.round(50 + intensity * 50);
    return `hsl(0, 0%, ${l}%)`;
  }

  const { hue } = AGENT_COLORS[agentId];
  const saturation = 100;
  const lightness = Math.round(30 + intensity * 40); // 30-70%

  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

/**
 * Get a glow color variant (higher lightness for bloom effect)
 */
export function getAgentGlowColor(agentId: AgentId): string {
  if (agentId === 'main') {
    return '#FFFFFF';
  }

  const { hue } = AGENT_COLORS[agentId];
  return `hsl(${hue}, 100%, 70%)`;
}

/**
 * Get a dim/ghost color variant (for inactive states)
 */
export function getAgentGhostColor(agentId: AgentId): string {
  if (agentId === 'main') {
    return '#333333';
  }

  const { hue } = AGENT_COLORS[agentId];
  return `hsl(${hue}, 30%, 20%)`;
}

/**
 * Interpolate between two agent colors by ratio (0-1)
 * Used for merge animations
 */
export function interpolateAgentColors(
  fromAgent: AgentId,
  toAgent: AgentId,
  ratio: number
): string {
  const fromRGB = getAgentRGB(fromAgent);
  const toRGB = getAgentRGB(toAgent);

  const r = Math.round(fromRGB[0] + (toRGB[0] - fromRGB[0]) * ratio);
  const g = Math.round(fromRGB[1] + (toRGB[1] - fromRGB[1]) * ratio);
  const b = Math.round(fromRGB[2] + (toRGB[2] - fromRGB[2]) * ratio);

  return `rgb(${r}, ${g}, ${b})`;
}

/**
 * Convert hex to Three.js compatible integer color
 */
export function hexToThreeColor(hex: string): number {
  return parseInt(hex.replace('#', ''), 16);
}

/**
 * Get Three.js integer color for an agent
 */
export function getAgentThreeColor(agentId: AgentId): number {
  return hexToThreeColor(getAgentColor(agentId));
}

// =============================================================================
// Position Mapping
// =============================================================================

/**
 * Get the orbital position for an agent around the prism
 * Returns [x, y, z] position in 3D space
 */
export function getAgentOrbitalPosition(agentId: AgentId, radius: number = 5): [number, number, number] {
  // Main is at center
  if (agentId === 'main') {
    return [0, 0, 0];
  }

  // Distribute agents evenly around the prism
  const positions: Record<Exclude<AgentId, 'main'>, number> = {
    claude: 0,    // Front
    qwen: 90,     // Right
    gemini: 180,  // Back
    codex: 270,   // Left
  };

  const angleDeg = positions[agentId];
  const angleRad = (angleDeg * Math.PI) / 180;

  const x = Math.cos(angleRad) * radius;
  const z = Math.sin(angleRad) * radius;
  const y = 0; // All on same plane initially

  return [x, y, z];
}

// =============================================================================
// CSS Variable Generation
// =============================================================================

/**
 * Generate CSS custom properties for agent colors
 * Useful for 2D HUD overlay
 */
export function generateAgentCSSVariables(): string {
  const lines: string[] = [':root {'];

  for (const [agentId, colors] of Object.entries(AGENT_COLORS)) {
    lines.push(`  --agent-${agentId}-color: ${colors.hex};`);
    lines.push(`  --agent-${agentId}-hue: ${colors.hue};`);
    lines.push(`  --agent-${agentId}-glow: ${getAgentGlowColor(agentId as AgentId)};`);
    lines.push(`  --agent-${agentId}-ghost: ${getAgentGhostColor(agentId as AgentId)};`);
  }

  lines.push('}');
  return lines.join('\n');
}

// =============================================================================
// Agent Ordering
// =============================================================================

/**
 * Get ordered list of agent IDs (excluding main)
 */
export function getAgentOrder(): Exclude<AgentId, 'main'>[] {
  return ['claude', 'qwen', 'gemini', 'codex'];
}

/**
 * Get all agent IDs including main
 */
export function getAllAgentIds(): AgentId[] {
  return ['main', 'claude', 'qwen', 'gemini', 'codex'];
}
