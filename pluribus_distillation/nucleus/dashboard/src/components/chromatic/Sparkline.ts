/**
 * Sparkline - Activity Sparklines for Chromatic Agents Visualizer
 *
 * Step 10: Activity Sparklines
 *
 * Per-agent sparkline showing:
 * - Commit frequency (last 15min)
 * - Lines changed over time (area chart)
 * - Bus event rate
 * - Rendered as glowing colored lines
 *
 * Uses P5.js for rendering with a rolling window buffer.
 */

import type { AgentId } from './types';
import { AGENT_COLORS } from './utils/colorMap';

// =============================================================================
// Types
// =============================================================================

export interface SparklineDataPoint {
  /** Timestamp in milliseconds */
  timestamp: number;
  /** Cumulative commits at this point */
  commits: number;
  /** Cumulative lines changed at this point */
  linesChanged: number;
  /** Event rate at this point (events/min) */
  eventRate: number;
}

export interface SparklineData {
  /** Data points in the rolling window */
  points: SparklineDataPoint[];
  /** Maximum values for normalization */
  maxCommits: number;
  maxLines: number;
  maxEventRate: number;
}

// =============================================================================
// Sparkline Class
// =============================================================================

export class Sparkline {
  private agentId: AgentId;
  private color: string;
  private glowColor: string;
  private data: SparklineData;
  private maxPoints: number;
  private frameCount: number = 0;

  // Rolling window: 15 minutes at 1 sample per 7.5 seconds = 120 points
  private readonly WINDOW_DURATION_MS = 15 * 60 * 1000; // 15 minutes

  constructor(agentId: AgentId, maxPoints: number = 120) {
    this.agentId = agentId;
    this.maxPoints = maxPoints;
    this.color = AGENT_COLORS[agentId]?.hex ?? '#888888';
    this.glowColor = this.lightenColor(this.color, 0.3);

    this.data = {
      points: [],
      maxCommits: 1,
      maxLines: 100,
      maxEventRate: 100,
    };
  }

  /**
   * Add a new data point to the rolling window
   */
  addDataPoint(point: SparklineDataPoint): void {
    this.data.points.push(point);

    // Remove old points outside the window
    const cutoffTime = Date.now() - this.WINDOW_DURATION_MS;
    this.data.points = this.data.points.filter(p => p.timestamp >= cutoffTime);

    // Limit to max points
    if (this.data.points.length > this.maxPoints) {
      this.data.points = this.data.points.slice(-this.maxPoints);
    }

    // Update max values for normalization
    this.updateMaxValues();
  }

  /**
   * Set data directly (for external data source)
   */
  setData(data: SparklineData): void {
    this.data = data;
  }

  /**
   * Get current data
   */
  getData(): SparklineData {
    return this.data;
  }

  /**
   * Update max values for normalization
   */
  private updateMaxValues(): void {
    if (this.data.points.length === 0) return;

    this.data.maxCommits = Math.max(
      1,
      Math.max(...this.data.points.map(p => p.commits))
    );
    this.data.maxLines = Math.max(
      100,
      Math.max(...this.data.points.map(p => p.linesChanged))
    );
    this.data.maxEventRate = Math.max(
      100,
      Math.max(...this.data.points.map(p => p.eventRate))
    );
  }

  /**
   * Render the sparkline using P5.js
   */
  render(
    p: typeof import('p5').default.prototype,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    this.frameCount++;

    const points = this.data.points;
    if (points.length < 2) {
      this.renderEmptyState(p, x, y, width, height);
      return;
    }

    // Draw agent label
    this.drawLabel(p, x, y);

    // Render area for lines changed
    this.renderLinesArea(p, x, y + 12, width, height - 14);

    // Render commit frequency line (on top)
    this.renderCommitLine(p, x, y + 12, width, height - 14);

    // Render event rate dots
    this.renderEventRateDots(p, x, y + 12, width, height - 14);
  }

  /**
   * Draw empty state when no data
   */
  private renderEmptyState(
    p: typeof import('p5').default.prototype,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    // Agent label
    this.drawLabel(p, x, y);

    // Empty line
    p.stroke(100, 100, 100, 50);
    p.strokeWeight(1);
    p.line(x, y + height / 2, x + width, y + height / 2);

    // "No data" text
    p.fill(100, 100, 100, 100);
    p.textSize(8);
    p.textAlign(p.CENTER, p.CENTER);
    p.text('awaiting data...', x + width / 2, y + height / 2);
  }

  /**
   * Draw agent label
   */
  private drawLabel(
    p: typeof import('p5').default.prototype,
    x: number,
    y: number
  ): void {
    const agentColor = p.color(this.color);

    // Agent dot
    p.fill(agentColor);
    p.noStroke();
    p.ellipse(x + 4, y + 4, 6, 6);

    // Agent name
    p.fill(180, 180, 200);
    p.textSize(8);
    p.textAlign(p.LEFT, p.CENTER);
    p.text(this.agentId.toUpperCase(), x + 12, y + 4);
  }

  /**
   * Render lines changed as area chart
   */
  private renderLinesArea(
    p: typeof import('p5').default.prototype,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    const points = this.data.points;
    const agentColor = p.color(this.color);

    // Calculate x scale
    const xScale = width / (points.length - 1);

    // Build area shape
    p.beginShape();
    p.noStroke();

    // Bottom left corner
    p.vertex(x, y + height);

    // Data points
    for (let i = 0; i < points.length; i++) {
      const px = x + i * xScale;
      const normalizedLines = points[i].linesChanged / this.data.maxLines;
      const py = y + height - normalizedLines * height * 0.8;
      p.vertex(px, py);
    }

    // Bottom right corner
    p.vertex(x + width, y + height);

    // Fill with gradient effect (approximated)
    const alpha = 60 + Math.sin(this.frameCount * 0.05) * 20;
    p.fill(
      p.red(agentColor),
      p.green(agentColor),
      p.blue(agentColor),
      alpha
    );
    p.endShape(p.CLOSE);
  }

  /**
   * Render commit frequency as glowing line
   */
  private renderCommitLine(
    p: typeof import('p5').default.prototype,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    const points = this.data.points;
    const agentColor = p.color(this.color);
    const glowColor = p.color(this.glowColor);

    // Calculate x scale
    const xScale = width / (points.length - 1);

    // Draw glow layer (wider stroke, lower opacity)
    p.stroke(
      p.red(glowColor),
      p.green(glowColor),
      p.blue(glowColor),
      80
    );
    p.strokeWeight(4);
    p.noFill();
    p.beginShape();

    for (let i = 0; i < points.length; i++) {
      const px = x + i * xScale;
      const normalizedCommits = points[i].commits / this.data.maxCommits;
      const py = y + height - normalizedCommits * height * 0.9;
      p.vertex(px, py);
    }
    p.endShape();

    // Draw main line (thinner, full opacity)
    p.stroke(agentColor);
    p.strokeWeight(2);
    p.beginShape();

    for (let i = 0; i < points.length; i++) {
      const px = x + i * xScale;
      const normalizedCommits = points[i].commits / this.data.maxCommits;
      const py = y + height - normalizedCommits * height * 0.9;
      p.vertex(px, py);
    }
    p.endShape();

    // Draw commit dots at peaks
    const commitDeltas = this.getCommitDeltas();
    for (let i = 0; i < points.length; i++) {
      if (commitDeltas[i] > 0) {
        const px = x + i * xScale;
        const normalizedCommits = points[i].commits / this.data.maxCommits;
        const py = y + height - normalizedCommits * height * 0.9;

        // Pulsing dot
        const pulse = Math.sin(this.frameCount * 0.2 + i) * 0.3 + 0.7;
        const dotSize = 4 + commitDeltas[i] * 2;

        p.fill(
          p.red(glowColor),
          p.green(glowColor),
          p.blue(glowColor),
          200 * pulse
        );
        p.noStroke();
        p.ellipse(px, py, dotSize, dotSize);
      }
    }
  }

  /**
   * Render event rate as background dots
   */
  private renderEventRateDots(
    p: typeof import('p5').default.prototype,
    x: number,
    y: number,
    width: number,
    height: number
  ): void {
    const points = this.data.points;
    const agentColor = p.color(this.color);

    // Calculate x scale
    const xScale = width / (points.length - 1);

    // Draw sparse dots representing event rate
    for (let i = 0; i < points.length; i += 4) {
      const px = x + i * xScale;
      const normalizedRate = points[i].eventRate / this.data.maxEventRate;
      const py = y + height - normalizedRate * height * 0.5;

      // Very subtle dots
      const alpha = 30 + normalizedRate * 50;
      p.fill(
        p.red(agentColor),
        p.green(agentColor),
        p.blue(agentColor),
        alpha
      );
      p.noStroke();
      p.ellipse(px, py, 3, 3);
    }
  }

  /**
   * Calculate commit deltas between points
   */
  private getCommitDeltas(): number[] {
    const points = this.data.points;
    const deltas: number[] = [0];

    for (let i = 1; i < points.length; i++) {
      deltas.push(Math.max(0, points[i].commits - points[i - 1].commits));
    }

    return deltas;
  }

  /**
   * Lighten a hex color
   */
  private lightenColor(hex: string, factor: number): string {
    // Parse hex
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    // Lighten
    const newR = Math.min(255, Math.round(r + (255 - r) * factor));
    const newG = Math.min(255, Math.round(g + (255 - g) * factor));
    const newB = Math.min(255, Math.round(b + (255 - b) * factor));

    // Convert back to hex
    return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`;
  }

  /**
   * Clear all data
   */
  clear(): void {
    this.data = {
      points: [],
      maxCommits: 1,
      maxLines: 100,
      maxEventRate: 100,
    };
  }
}

// =============================================================================
// Factory Function
// =============================================================================

/**
 * Create sparklines for all agents
 */
export function createAgentSparklines(
  maxPoints: number = 120
): Map<AgentId, Sparkline> {
  const sparklines = new Map<AgentId, Sparkline>();

  const agentIds: Exclude<AgentId, 'main'>[] = ['claude', 'qwen', 'gemini', 'codex'];

  for (const agentId of agentIds) {
    sparklines.set(agentId, new Sparkline(agentId, maxPoints));
  }

  return sparklines;
}

/**
 * Generate mock sparkline data for demo mode
 */
export function generateMockSparklineData(
  duration: number = 15 * 60 * 1000,
  interval: number = 7500
): SparklineData {
  const points: SparklineDataPoint[] = [];
  const now = Date.now();
  const numPoints = Math.floor(duration / interval);

  let cumulativeCommits = 0;
  let cumulativeLines = 0;

  for (let i = 0; i < numPoints; i++) {
    // Simulate occasional commits
    if (Math.random() < 0.15) {
      cumulativeCommits++;
      cumulativeLines += Math.floor(Math.random() * 50) + 10;
    }

    points.push({
      timestamp: now - duration + i * interval,
      commits: cumulativeCommits,
      linesChanged: cumulativeLines,
      eventRate: 50 + Math.random() * 100 + Math.sin(i * 0.2) * 30,
    });
  }

  return {
    points,
    maxCommits: Math.max(1, cumulativeCommits),
    maxLines: Math.max(100, cumulativeLines),
    maxEventRate: 200,
  };
}
