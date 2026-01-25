/**
 * Chromatic Agents Visualizer - Reduced Motion & Accessibility Fallback
 *
 * Step 18: Accessibility & Fallback
 * - Reduced motion mode (static tree, no particles)
 * - Color-blind friendly secondary indicators (shapes)
 * - 2D fallback for low-end devices
 * - Screen reader descriptions of agent states
 */

import type { AgentId, ChromaticState, AgentVisualData } from './types';
import { AgentVisualState } from './types';
import { AGENT_COLORS, getAgentOrder } from './utils/colorMap';

// =============================================================================
// Types
// =============================================================================

export interface A11yConfig {
  /** User prefers reduced motion (from media query) */
  reducedMotion: boolean;
  /** User needs colorblind-friendly indicators */
  colorBlindMode: boolean;
  /** Force 2D mode for low-end devices */
  force2D: boolean;
  /** Enable screen reader announcements */
  screenReaderEnabled: boolean;
  /** High contrast mode */
  highContrast: boolean;
}

export interface ShapeIndicator {
  /** Shape type for colorblind differentiation */
  shape: 'circle' | 'square' | 'triangle' | 'diamond' | 'pentagon';
  /** Pattern for additional differentiation */
  pattern: 'solid' | 'striped' | 'dotted' | 'crosshatch';
  /** SVG path data for the shape */
  svgPath: string;
  /** Unicode symbol for text fallback */
  unicode: string;
}

export interface ScreenReaderDescription {
  /** Short label for the agent */
  label: string;
  /** Full description of current state */
  description: string;
  /** ARIA live region priority */
  priority: 'polite' | 'assertive';
}

// =============================================================================
// Shape Indicators for Colorblind Users (Step 18)
// =============================================================================

export const AGENT_SHAPES: Record<AgentId, ShapeIndicator> = {
  claude: {
    shape: 'circle',
    pattern: 'solid',
    svgPath: 'M50,5 A45,45 0 1,1 49.99,5 Z',
    unicode: '\u25CF', // Black circle
  },
  qwen: {
    shape: 'square',
    pattern: 'solid',
    svgPath: 'M5,5 H95 V95 H5 Z',
    unicode: '\u25A0', // Black square
  },
  gemini: {
    shape: 'triangle',
    pattern: 'solid',
    svgPath: 'M50,5 L95,95 L5,95 Z',
    unicode: '\u25B2', // Black up-pointing triangle
  },
  codex: {
    shape: 'diamond',
    pattern: 'solid',
    svgPath: 'M50,5 L95,50 L50,95 L5,50 Z',
    unicode: '\u25C6', // Black diamond
  },
  main: {
    shape: 'pentagon',
    pattern: 'solid',
    svgPath: 'M50,5 L95,40 L77,95 L23,95 L5,40 Z',
    unicode: '\u2B1F', // Pentagon
  },
};

// =============================================================================
// State Descriptions for Screen Readers
// =============================================================================

const STATE_DESCRIPTIONS: Record<AgentVisualState, string> = {
  idle: 'is currently idle with no active work',
  cloning: 'is creating a new working copy of the repository',
  working: 'is actively making code changes',
  committing: 'is saving changes with a commit',
  pushing: 'is uploading changes to the remote repository',
  merged: 'has successfully merged changes into main',
  cleanup: 'is cleaning up temporary files',
};

// =============================================================================
// Reduced Motion Manager
// =============================================================================

export class ReducedMotionManager {
  private config: A11yConfig;
  private mediaQuery: MediaQueryList | null = null;
  private listeners: Set<(config: A11yConfig) => void> = new Set();
  private ariaLiveRegion: HTMLElement | null = null;
  private lastAnnouncedState: Map<AgentId, AgentVisualState> = new Map();

  constructor() {
    this.config = {
      reducedMotion: false,
      colorBlindMode: false,
      force2D: false,
      screenReaderEnabled: false,
      highContrast: false,
    };

    this.initialize();
  }

  /**
   * Initialize accessibility detection
   */
  private initialize(): void {
    // Check if we're in a browser environment
    if (typeof window === 'undefined' || typeof document === 'undefined') {
      return;
    }

    // Detect prefers-reduced-motion
    this.mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    this.config.reducedMotion = this.mediaQuery.matches;

    this.mediaQuery.addEventListener('change', (e) => {
      this.config.reducedMotion = e.matches;
      this.notifyListeners();
    });

    // Detect prefers-contrast
    const contrastQuery = window.matchMedia('(prefers-contrast: more)');
    this.config.highContrast = contrastQuery.matches;

    contrastQuery.addEventListener('change', (e) => {
      this.config.highContrast = e.matches;
      this.notifyListeners();
    });

    // Detect low-end device via performance API
    this.detectLowEndDevice();

    // Create ARIA live region for announcements
    this.createAriaLiveRegion();
  }

  /**
   * Detect if device should use 2D fallback
   */
  private detectLowEndDevice(): void {
    if (typeof navigator === 'undefined') return;

    // Check for hardware concurrency (CPU cores)
    const cores = navigator.hardwareConcurrency || 4;
    if (cores <= 2) {
      this.config.force2D = true;
    }

    // Check for device memory API (Chrome only)
    const deviceMemory = (navigator as { deviceMemory?: number }).deviceMemory;
    if (deviceMemory !== undefined && deviceMemory < 4) {
      this.config.force2D = true;
    }

    // Check WebGL support
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) {
        this.config.force2D = true;
      }
    } catch {
      this.config.force2D = true;
    }
  }

  /**
   * Create hidden ARIA live region for screen reader announcements
   */
  private createAriaLiveRegion(): void {
    if (typeof document === 'undefined') return;

    this.ariaLiveRegion = document.createElement('div');
    this.ariaLiveRegion.setAttribute('role', 'status');
    this.ariaLiveRegion.setAttribute('aria-live', 'polite');
    this.ariaLiveRegion.setAttribute('aria-atomic', 'true');
    this.ariaLiveRegion.className = 'chromatic-sr-only';

    // Visually hidden but accessible to screen readers
    Object.assign(this.ariaLiveRegion.style, {
      position: 'absolute',
      width: '1px',
      height: '1px',
      padding: '0',
      margin: '-1px',
      overflow: 'hidden',
      clip: 'rect(0, 0, 0, 0)',
      whiteSpace: 'nowrap',
      border: '0',
    });

    document.body.appendChild(this.ariaLiveRegion);
  }

  /**
   * Get current accessibility configuration
   */
  getConfig(): A11yConfig {
    return { ...this.config };
  }

  /**
   * Manually enable colorblind mode
   */
  setColorBlindMode(enabled: boolean): void {
    this.config.colorBlindMode = enabled;
    this.notifyListeners();
  }

  /**
   * Manually enable screen reader support
   */
  setScreenReaderEnabled(enabled: boolean): void {
    this.config.screenReaderEnabled = enabled;
    this.notifyListeners();
  }

  /**
   * Force reduced motion mode
   */
  setReducedMotion(enabled: boolean): void {
    this.config.reducedMotion = enabled;
    this.notifyListeners();
  }

  /**
   * Subscribe to config changes
   */
  subscribe(callback: (config: A11yConfig) => void): () => void {
    this.listeners.add(callback);
    return () => this.listeners.delete(callback);
  }

  private notifyListeners(): void {
    for (const listener of this.listeners) {
      listener(this.getConfig());
    }
  }

  /**
   * Check if animations should be disabled
   */
  shouldDisableAnimations(): boolean {
    return this.config.reducedMotion;
  }

  /**
   * Check if particles should be disabled
   */
  shouldDisableParticles(): boolean {
    return this.config.reducedMotion || this.config.force2D;
  }

  /**
   * Check if 3D should be disabled
   */
  shouldUse2DFallback(): boolean {
    return this.config.force2D;
  }

  /**
   * Get shape indicator for an agent (colorblind support)
   */
  getAgentShape(agentId: AgentId): ShapeIndicator {
    return AGENT_SHAPES[agentId];
  }

  /**
   * Generate screen reader description for agent state
   */
  getScreenReaderDescription(agent: AgentVisualData): ScreenReaderDescription {
    const colorInfo = AGENT_COLORS[agent.id];
    const stateDesc = STATE_DESCRIPTIONS[agent.state];
    const intensityDesc = agent.intensity > 0.7 ? 'highly active' :
                          agent.intensity > 0.3 ? 'moderately active' : 'low activity';

    return {
      label: `${colorInfo.name} Agent (${agent.id})`,
      description: `The ${colorInfo.name} agent ${stateDesc}. Current activity level: ${intensityDesc}.` +
                   (agent.branch ? ` Working on branch: ${agent.branch}.` : ''),
      priority: agent.state === 'merged' || agent.state === 'pushing' ? 'assertive' : 'polite',
    };
  }

  /**
   * Announce agent state change to screen readers
   */
  announceStateChange(agent: AgentVisualData): void {
    if (!this.config.screenReaderEnabled || !this.ariaLiveRegion) return;

    // Only announce if state actually changed
    const lastState = this.lastAnnouncedState.get(agent.id);
    if (lastState === agent.state) return;

    this.lastAnnouncedState.set(agent.id, agent.state);

    const desc = this.getScreenReaderDescription(agent);
    this.ariaLiveRegion.setAttribute('aria-live', desc.priority);
    this.ariaLiveRegion.textContent = desc.description;
  }

  /**
   * Generate static tree representation for reduced motion mode
   */
  generateStaticTreeSVG(state: ChromaticState): string {
    const agents = getAgentOrder();
    const width = 400;
    const height = 400;
    const centerX = width / 2;
    const centerY = height / 2;
    const radius = 120;

    let svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Chromatic Agents Static View">`;

    // Add title for screen readers
    svg += `<title>Chromatic Agents Visualization - Static View</title>`;
    svg += `<desc>A static representation of agent activity around a central prism. `;
    svg += `Shapes indicate different agents: circle for Claude, square for Qwen, triangle for Gemini, diamond for Codex.</desc>`;

    // Draw central prism (pentagon for main)
    svg += `<polygon points="${this.getPrismPoints(centerX, centerY, 40)}" fill="#FFFFFF" stroke="#CCCCCC" stroke-width="2"/>`;

    // Draw agents
    agents.forEach((agentId, index) => {
      const angle = (index * 90 - 45) * (Math.PI / 180);
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;

      const agentData = state.agents.get(agentId);
      const color = AGENT_COLORS[agentId].hex;
      const shape = AGENT_SHAPES[agentId];
      const opacity = agentData?.opacity ?? 0.3;

      // Draw connecting line to center
      svg += `<line x1="${centerX}" y1="${centerY}" x2="${x}" y2="${y}" stroke="${color}" stroke-opacity="${opacity * 0.5}" stroke-width="2"/>`;

      // Draw agent shape
      svg += this.renderAgentShape(x, y, shape, color, opacity, agentId, agentData?.state ?? AgentVisualState.IDLE);
    });

    svg += '</svg>';
    return svg;
  }

  private getPrismPoints(cx: number, cy: number, size: number): string {
    const points: string[] = [];
    for (let i = 0; i < 5; i++) {
      const angle = (i * 72 - 90) * (Math.PI / 180);
      points.push(`${cx + Math.cos(angle) * size},${cy + Math.sin(angle) * size}`);
    }
    return points.join(' ');
  }

  private renderAgentShape(
    x: number,
    y: number,
    shape: ShapeIndicator,
    color: string,
    opacity: number,
    agentId: AgentId,
    state: AgentVisualState
  ): string {
    const size = 25;
    const stateDesc = STATE_DESCRIPTIONS[state];

    let shapeSvg = `<g role="img" aria-label="${agentId} agent ${stateDesc}" opacity="${opacity}">`;

    switch (shape.shape) {
      case 'circle':
        shapeSvg += `<circle cx="${x}" cy="${y}" r="${size}" fill="${color}" stroke="#000" stroke-width="2"/>`;
        break;
      case 'square':
        shapeSvg += `<rect x="${x - size}" y="${y - size}" width="${size * 2}" height="${size * 2}" fill="${color}" stroke="#000" stroke-width="2"/>`;
        break;
      case 'triangle':
        shapeSvg += `<polygon points="${x},${y - size} ${x + size},${y + size} ${x - size},${y + size}" fill="${color}" stroke="#000" stroke-width="2"/>`;
        break;
      case 'diamond':
        shapeSvg += `<polygon points="${x},${y - size} ${x + size},${y} ${x},${y + size} ${x - size},${y}" fill="${color}" stroke="#000" stroke-width="2"/>`;
        break;
      case 'pentagon':
        shapeSvg += `<polygon points="${this.getPrismPoints(x, y, size)}" fill="${color}" stroke="#000" stroke-width="2"/>`;
        break;
    }

    // Add state indicator (border style)
    if (state === 'working') {
      // Dashed border for working
      shapeSvg += `<circle cx="${x}" cy="${y}" r="${size + 5}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="5,5"/>`;
    } else if (state === 'pushing' || state === 'committing') {
      // Solid outer ring for important states
      shapeSvg += `<circle cx="${x}" cy="${y}" r="${size + 5}" fill="none" stroke="${color}" stroke-width="3"/>`;
    }

    shapeSvg += '</g>';
    return shapeSvg;
  }

  /**
   * Generate CSS for high contrast mode
   */
  getHighContrastStyles(): string {
    if (!this.config.highContrast) return '';

    return `
      .chromatic-visualizer {
        background: #000 !important;
      }
      .chromatic-agent {
        border: 3px solid #FFF !important;
      }
      .chromatic-agent-label {
        color: #FFF !important;
        background: #000 !important;
        font-weight: bold !important;
      }
      .chromatic-hud {
        background: #000 !important;
        border: 2px solid #FFF !important;
      }
      .chromatic-progress-bar {
        border: 2px solid #FFF !important;
      }
    `;
  }

  /**
   * Cleanup resources
   */
  destroy(): void {
    if (this.ariaLiveRegion && this.ariaLiveRegion.parentNode) {
      this.ariaLiveRegion.parentNode.removeChild(this.ariaLiveRegion);
    }
    this.listeners.clear();
    this.lastAnnouncedState.clear();
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let instance: ReducedMotionManager | null = null;

export function getReducedMotionManager(): ReducedMotionManager {
  if (!instance) {
    instance = new ReducedMotionManager();
  }
  return instance;
}

// =============================================================================
// React Hook (for Qwik/React integration)
// =============================================================================

/**
 * Hook signature for framework integration
 * Usage in component:
 *   const a11y = useReducedMotion();
 *   if (a11y.shouldDisableParticles()) { ... }
 */
export interface UseReducedMotionReturn {
  config: A11yConfig;
  shouldDisableAnimations: () => boolean;
  shouldDisableParticles: () => boolean;
  shouldUse2DFallback: () => boolean;
  getAgentShape: (agentId: AgentId) => ShapeIndicator;
  getScreenReaderDescription: (agent: AgentVisualData) => ScreenReaderDescription;
  announceStateChange: (agent: AgentVisualData) => void;
  generateStaticTreeSVG: (state: ChromaticState) => string;
  setColorBlindMode: (enabled: boolean) => void;
  setScreenReaderEnabled: (enabled: boolean) => void;
}

export function createReducedMotionHook(): UseReducedMotionReturn {
  const manager = getReducedMotionManager();

  return {
    config: manager.getConfig(),
    shouldDisableAnimations: () => manager.shouldDisableAnimations(),
    shouldDisableParticles: () => manager.shouldDisableParticles(),
    shouldUse2DFallback: () => manager.shouldUse2DFallback(),
    getAgentShape: (agentId) => manager.getAgentShape(agentId),
    getScreenReaderDescription: (agent) => manager.getScreenReaderDescription(agent),
    announceStateChange: (agent) => manager.announceStateChange(agent),
    generateStaticTreeSVG: (state) => manager.generateStaticTreeSVG(state),
    setColorBlindMode: (enabled) => manager.setColorBlindMode(enabled),
    setScreenReaderEnabled: (enabled) => manager.setScreenReaderEnabled(enabled),
  };
}
