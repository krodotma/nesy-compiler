/**
 * AgentHUD - P5.js 2D Canvas Overlay for Chromatic Agents Visualizer
 *
 * Step 9: HUD Layer Design
 *
 * 2D canvas overlay showing:
 * - Agent progress bars with chromatic colors
 * - State labels (CLONING, WORKING, PUSHING, etc)
 * - Bus event rate counter
 * - Main branch ahead count
 *
 * Uses P5.js for canvas rendering with Qwik component wrapper.
 */

import { component$, useSignal, useVisibleTask$, type Signal } from '@builder.io/qwik';
import type { AgentId, AgentVisualState, HUDState, AgentHUDData } from './types';
import { AGENT_COLORS, getAgentOrder } from './utils/colorMap';
import { Sparkline, type SparklineData } from './Sparkline';
import { MergeAnimation } from './MergeAnimation';

// M3 Components - AgentHUD
import '@material/web/elevation/elevation.js';
import '@material/web/ripple/ripple.js';
import '@material/web/progress/linear-progress.js';

// =============================================================================
// Types
// =============================================================================

interface AgentHUDProps {
  /** HUD state signal from parent visualizer */
  state?: Signal<HUDState>;
  /** Sparkline data for each agent (15-min rolling window) */
  sparklineData?: Signal<Record<AgentId, SparklineData>>;
  /** Merge events signal */
  mergeEvents?: Signal<{ agentId: AgentId; timestamp: number }[]>;
  /** Width of the HUD canvas */
  width?: number;
  /** Height of the HUD canvas */
  height?: number;
  /** Show sparklines (default: true) */
  showSparklines?: boolean;
  /** Show merge animations (default: true) */
  showMergeAnimations?: boolean;
}

// Default HUD state for demo mode
const DEFAULT_HUD_STATE: HUDState = {
  agents: [
    { id: 'claude', name: 'Claude', color: AGENT_COLORS.claude.hex, progress: 80, state: 'pushing' as AgentVisualState, linesChanged: 245, commitsCount: 3 },
    { id: 'qwen', name: 'Qwen', color: AGENT_COLORS.qwen.hex, progress: 60, state: 'working' as AgentVisualState, linesChanged: 128, commitsCount: 2 },
    { id: 'gemini', name: 'Gemini', color: AGENT_COLORS.gemini.hex, progress: 40, state: 'committing' as AgentVisualState, linesChanged: 67, commitsCount: 1 },
    { id: 'codex', name: 'Codex', color: AGENT_COLORS.codex.hex, progress: 20, state: 'cloning' as AgentVisualState, linesChanged: 0, commitsCount: 0 },
  ],
  eventsPerMinute: 142,
  mainAhead: 3,
  busStatus: 'connected',
};

// =============================================================================
// P5.js HUD Renderer Class
// =============================================================================

class HUDRenderer {
  private p5: typeof import('p5').default.prototype | null = null;
  private canvas: HTMLCanvasElement;
  private width: number;
  private height: number;
  private state: HUDState;
  private sparklines: Map<AgentId, Sparkline> = new Map();
  private mergeAnimation: MergeAnimation;
  private frameCount: number = 0;
  private showSparklines: boolean;
  private showMergeAnimations: boolean;

  constructor(
    canvas: HTMLCanvasElement,
    width: number,
    height: number,
    showSparklines: boolean = true,
    showMergeAnimations: boolean = true
  ) {
    this.canvas = canvas;
    this.width = width;
    this.height = height;
    this.state = DEFAULT_HUD_STATE;
    this.showSparklines = showSparklines;
    this.showMergeAnimations = showMergeAnimations;

    // Initialize sparklines for each agent
    for (const agentId of getAgentOrder()) {
      this.sparklines.set(agentId, new Sparkline(agentId, 120)); // 15 min at 1 sample/7.5s
    }

    // Initialize merge animation handler
    this.mergeAnimation = new MergeAnimation(width, height);
  }

  async init(): Promise<void> {
    const p5Module = await import('p5');
    const P5 = p5Module.default;

    // Create P5 instance in instance mode
    this.p5 = new P5((p: typeof import('p5').default.prototype) => {
      p.setup = () => {
        p.createCanvas(this.width, this.height, this.canvas);
        p.textFont('JetBrains Mono, Monaco, monospace');
        p.noSmooth();
      };

      p.draw = () => {
        this.render(p);
      };
    }, this.canvas.parentElement!);
  }

  updateState(state: HUDState): void {
    this.state = state;

    // Update sparklines with new data
    for (const agent of state.agents) {
      const sparkline = this.sparklines.get(agent.id);
      if (sparkline) {
        sparkline.addDataPoint({
          timestamp: Date.now(),
          commits: agent.commitsCount,
          linesChanged: agent.linesChanged,
          eventRate: state.eventsPerMinute / state.agents.length,
        });
      }
    }
  }

  updateSparklineData(data: Record<AgentId, SparklineData>): void {
    for (const [agentId, sparklineData] of Object.entries(data) as [AgentId, SparklineData][]) {
      const sparkline = this.sparklines.get(agentId);
      if (sparkline) {
        sparkline.setData(sparklineData);
      }
    }
  }

  triggerMerge(agentId: AgentId): void {
    const agent = this.state.agents.find(a => a.id === agentId);
    if (agent) {
      const yOffset = this.getAgentYOffset(agentId);
      this.mergeAnimation.trigger(agentId, agent.color, yOffset);
    }
  }

  private getAgentYOffset(agentId: AgentId): number {
    const agents = getAgentOrder();
    const index = agents.indexOf(agentId as Exclude<AgentId, 'main'>);
    return 60 + index * 55;
  }

  private render(p: typeof import('p5').default.prototype): void {
    this.frameCount++;

    // Clear with transparency
    p.clear();

    // Draw background panel
    this.drawBackground(p);

    // Draw header
    this.drawHeader(p);

    // Draw agent rows
    this.drawAgentRows(p);

    // Draw footer stats
    this.drawFooter(p);

    // Draw sparklines if enabled
    if (this.showSparklines) {
      this.drawSparklines(p);
    }

    // Draw merge animations if enabled
    if (this.showMergeAnimations) {
      this.mergeAnimation.update(p, 1 / 60);
      this.mergeAnimation.render(p);
    }
  }

  private drawBackground(p: typeof import('p5').default.prototype): void {
    // Main panel background
    p.noStroke();
    p.fill(0, 0, 0, 200);
    p.rect(8, 8, 260, 250, 8);

    // Subtle border glow
    p.noFill();
    p.stroke(100, 100, 120, 80);
    p.strokeWeight(1);
    p.rect(8, 8, 260, 250, 8);

    // Inner highlight
    p.stroke(255, 255, 255, 20);
    p.rect(9, 9, 258, 248, 7);
  }

  private drawHeader(p: typeof import('p5').default.prototype): void {
    // Title
    p.fill(255, 255, 255, 230);
    p.noStroke();
    p.textSize(11);
    p.textAlign(p.LEFT, p.TOP);
    p.text('AGENTS CHROMATIC VIEW', 20, 20);

    // Status indicator
    const statusColor = this.state.busStatus === 'connected' ? p.color(0, 255, 100)
      : this.state.busStatus === 'reconnecting' ? p.color(255, 200, 0)
      : p.color(255, 80, 80);

    p.fill(statusColor);
    p.noStroke();
    p.ellipse(248, 26, 8, 8);

    // Divider line
    p.stroke(100, 100, 120, 100);
    p.strokeWeight(1);
    p.line(16, 38, 260, 38);
  }

  private drawAgentRows(p: typeof import('p5').default.prototype): void {
    const agents = this.state.agents;
    const startY = 50;
    const rowHeight = 55;

    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i];
      const y = startY + i * rowHeight;

      this.drawAgentRow(p, agent, y);
    }
  }

  private drawAgentRow(
    p: typeof import('p5').default.prototype,
    agent: AgentHUDData,
    y: number
  ): void {
    const agentColor = p.color(agent.color);

    // Agent dot (pulsing based on activity)
    const pulse = Math.sin(this.frameCount * 0.1 + agent.progress * 0.1) * 0.2 + 0.8;
    p.fill(p.red(agentColor), p.green(agentColor), p.blue(agentColor), 255 * pulse);
    p.noStroke();
    p.ellipse(28, y + 10, 10, 10);

    // Agent name
    p.fill(255, 255, 255, 200);
    p.textSize(10);
    p.textAlign(p.LEFT, p.CENTER);
    p.text(agent.id.toUpperCase(), 40, y + 10);

    // Progress bar background
    p.fill(40, 40, 50);
    p.noStroke();
    p.rect(100, y + 4, 100, 12, 2);

    // Progress bar fill
    const barWidth = (agent.progress / 100) * 100;
    if (barWidth > 0) {
      // Gradient effect via multiple rects
      for (let i = 0; i < barWidth; i += 2) {
        const intensity = 0.6 + (i / barWidth) * 0.4;
        p.fill(
          p.red(agentColor) * intensity,
          p.green(agentColor) * intensity,
          p.blue(agentColor) * intensity,
          220
        );
        p.rect(100 + i, y + 4, 2, 12);
      }

      // Glow effect at end of bar
      if (agent.state === 'working' || agent.state === 'pushing') {
        const glowIntensity = (Math.sin(this.frameCount * 0.2) + 1) * 0.5;
        p.fill(p.red(agentColor), p.green(agentColor), p.blue(agentColor), 100 * glowIntensity);
        p.ellipse(100 + barWidth, y + 10, 16, 16);
      }
    }

    // Progress percentage
    p.fill(180, 180, 200);
    p.textSize(9);
    p.textAlign(p.RIGHT, p.CENTER);
    p.text(`${agent.progress}%`, 222, y + 10);

    // State label with color coding
    const stateColor = this.getStateColor(p, agent.state);
    p.fill(stateColor);
    p.textSize(8);
    p.textAlign(p.LEFT, p.CENTER);
    p.text(agent.state.toUpperCase(), 100, y + 26);

    // Lines changed indicator
    if (agent.linesChanged > 0) {
      p.fill(100, 200, 100, 180);
      p.textSize(8);
      p.textAlign(p.RIGHT, p.CENTER);
      p.text(`+${agent.linesChanged}`, 252, y + 26);
    }
  }

  private getStateColor(
    p: typeof import('p5').default.prototype,
    state: AgentVisualState
  ): import('p5').Color {
    switch (state) {
      case 'idle':
        return p.color(100, 100, 100);
      case 'cloning':
        return p.color(100, 180, 255);
      case 'working':
        return p.color(100, 255, 150);
      case 'committing':
        return p.color(255, 200, 100);
      case 'pushing':
        return p.color(200, 100, 255);
      case 'merged':
        return p.color(255, 255, 255);
      case 'cleanup':
        return p.color(150, 150, 150);
      default:
        return p.color(128, 128, 128);
    }
  }

  private drawFooter(p: typeof import('p5').default.prototype): void {
    const y = 232;

    // Divider line
    p.stroke(100, 100, 120, 100);
    p.strokeWeight(1);
    p.line(16, y - 10, 260, y - 10);

    // Bus events/min
    p.fill(150, 200, 255);
    p.noStroke();
    p.textSize(9);
    p.textAlign(p.LEFT, p.CENTER);
    p.text(`Bus: ${this.state.eventsPerMinute} events/min`, 20, y);

    // Main ahead count
    p.fill(200, 200, 200);
    p.textAlign(p.RIGHT, p.CENTER);
    p.text(`Main: ${this.state.mainAhead} ahead`, 252, y);
  }

  private drawSparklines(p: typeof import('p5').default.prototype): void {
    // Draw sparklines panel on the right side
    const panelX = 280;
    const panelY = 8;
    const panelWidth = 180;
    const panelHeight = 250;

    // Background
    p.noStroke();
    p.fill(0, 0, 0, 180);
    p.rect(panelX, panelY, panelWidth, panelHeight, 8);

    // Title
    p.fill(255, 255, 255, 200);
    p.textSize(10);
    p.textAlign(p.LEFT, p.TOP);
    p.text('ACTIVITY (15 MIN)', panelX + 12, panelY + 12);

    // Draw each agent's sparkline
    const agents = getAgentOrder();
    const sparklineHeight = 45;
    const startY = panelY + 35;

    for (let i = 0; i < agents.length; i++) {
      const agentId = agents[i];
      const sparkline = this.sparklines.get(agentId);
      const y = startY + i * (sparklineHeight + 8);

      if (sparkline) {
        sparkline.render(p, panelX + 12, y, panelWidth - 24, sparklineHeight);
      }
    }
  }

  resize(width: number, height: number): void {
    this.width = width;
    this.height = height;
    this.mergeAnimation.resize(width, height);
    if (this.p5) {
      this.p5.resizeCanvas(width, height);
    }
  }

  dispose(): void {
    if (this.p5) {
      this.p5.remove();
      this.p5 = null;
    }
    this.sparklines.clear();
  }
}

// =============================================================================
// Qwik Component
// =============================================================================

export const AgentHUD = component$<AgentHUDProps>((props) => {
  const canvasRef = useSignal<HTMLCanvasElement>();
  const isInitialized = useSignal(false);

  useVisibleTask$(async ({ cleanup, track }) => {
    const canvas = canvasRef.value;
    if (!canvas) return;

    // Track state changes
    track(() => props.state?.value);
    track(() => props.sparklineData?.value);
    track(() => props.mergeEvents?.value);

    const width = props.width ?? (props.showSparklines !== false ? 470 : 276);
    const height = props.height ?? 266;

    // Initialize renderer
    const renderer = new HUDRenderer(
      canvas,
      width,
      height,
      props.showSparklines !== false,
      props.showMergeAnimations !== false
    );

    await renderer.init();
    isInitialized.value = true;

    // Set initial state
    if (props.state?.value) {
      renderer.updateState(props.state.value);
    }

    // Watch for state updates
    let lastStateUpdate = 0;
    const stateUpdateInterval = setInterval(() => {
      if (props.state?.value && Date.now() - lastStateUpdate > 100) {
        renderer.updateState(props.state.value);
        lastStateUpdate = Date.now();
      }
    }, 100);

    // Watch for sparkline data updates
    let lastSparklineUpdate = 0;
    const sparklineUpdateInterval = setInterval(() => {
      if (props.sparklineData?.value && Date.now() - lastSparklineUpdate > 500) {
        renderer.updateSparklineData(props.sparklineData.value);
        lastSparklineUpdate = Date.now();
      }
    }, 500);

    // Watch for merge events
    let processedMerges = new Set<number>();
    const mergeEventInterval = setInterval(() => {
      if (props.mergeEvents?.value) {
        for (const event of props.mergeEvents.value) {
          if (!processedMerges.has(event.timestamp)) {
            renderer.triggerMerge(event.agentId);
            processedMerges.add(event.timestamp);
          }
        }
        // Clean up old processed merges
        if (processedMerges.size > 100) {
          processedMerges = new Set(
            [...processedMerges].slice(-50)
          );
        }
      }
    }, 50);

    // Demo mode if no state provided
    let demoInterval: ReturnType<typeof setInterval> | null = null;
    if (!props.state) {
      let demoProgress = 0;
      demoInterval = setInterval(() => {
        demoProgress = (demoProgress + 1) % 100;
        const demoState: HUDState = {
          ...DEFAULT_HUD_STATE,
          agents: DEFAULT_HUD_STATE.agents.map((agent, i) => ({
            ...agent,
            progress: (demoProgress + i * 20) % 100,
            linesChanged: Math.floor(Math.random() * 50) + agent.linesChanged,
          })),
          eventsPerMinute: 100 + Math.floor(Math.random() * 100),
        };
        renderer.updateState(demoState);

        // Occasionally trigger merge animation
        if (Math.random() < 0.02) {
          const agents = getAgentOrder();
          const randomAgent = agents[Math.floor(Math.random() * agents.length)];
          renderer.triggerMerge(randomAgent);
        }
      }, 200);
    }

    cleanup(() => {
      clearInterval(stateUpdateInterval);
      clearInterval(sparklineUpdateInterval);
      clearInterval(mergeEventInterval);
      if (demoInterval) {
        clearInterval(demoInterval);
      }
      renderer.dispose();
    });
  });

  const canvasWidth = props.width ?? (props.showSparklines !== false ? 470 : 276);
  const canvasHeight = props.height ?? 266;

  return (
    <div
      class="absolute pointer-events-none"
      style={{
        left: '16px',
        bottom: '16px',
        width: `${canvasWidth}px`,
        height: `${canvasHeight}px`,
      }}
    >
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        class="w-full h-full"
        style={{ display: 'block' }}
      />
      {!isInitialized.value && (
        <div class="absolute inset-0 flex items-center justify-center glass-surface-overlay rounded-lg">
          <span class="text-xs text-glass-text-muted glass-pulse">Loading HUD...</span>
        </div>
      )}
    </div>
  );
});

export default AgentHUD;
