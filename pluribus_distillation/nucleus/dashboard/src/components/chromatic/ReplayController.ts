/**
 * Chromatic Agents Visualizer - Replay Controller
 *
 * Step 19: Persistence & Replay
 * - Record visualization state to bus
 * - "Replay last hour" scrubber
 * - Screenshot/video export
 * - Shareable state URLs
 */

import type {
  ChromaticState,
  AgentId,
  AgentVisualData,
  AgentVisualState,
  CodeGraph,
} from './types';

// =============================================================================
// Types
// =============================================================================

export interface StateSnapshot {
  /** Unique snapshot ID */
  id: string;
  /** ISO timestamp of this snapshot */
  timestamp_iso: string;
  /** Unix timestamp in milliseconds */
  timestamp_ms: number;
  /** Serialized chromatic state */
  state: SerializedChromaticState;
  /** Optional associated bus event topic */
  trigger_topic?: string;
  /** Compressed size in bytes (if stored compressed) */
  compressed_size?: number;
}

export interface SerializedChromaticState {
  agents: SerializedAgentData[];
  prismIntensity: number;
  focusedAgent: AgentId | null;
  busConnected: boolean;
  eventsPerMinute: number;
  mainAhead: number;
}

export interface SerializedAgentData {
  id: AgentId;
  state: AgentVisualState;
  hue: number;
  color: string;
  intensity: number;
  codeGraph: CodeGraph | null;
  branch: string | null;
  position: [number, number, number];
  opacity: number;
  lastUpdate: number;
}

export interface ReplayConfig {
  /** Maximum duration to store in memory (ms) - default 1 hour */
  maxDuration: number;
  /** Snapshot interval (ms) - default 1000ms */
  snapshotInterval: number;
  /** Maximum snapshots to keep */
  maxSnapshots: number;
  /** Enable compression for stored snapshots */
  enableCompression: boolean;
  /** Auto-emit snapshots to bus */
  emitToBus: boolean;
}

export interface PlaybackState {
  /** Is currently playing */
  isPlaying: boolean;
  /** Is paused */
  isPaused: boolean;
  /** Current playback position (0-1) */
  position: number;
  /** Current timestamp being played */
  currentTimestamp: number;
  /** Playback speed multiplier */
  speed: number;
  /** Is in loop mode */
  loop: boolean;
}

export interface ScrubberState {
  /** Start timestamp of recorded range */
  startTime: number;
  /** End timestamp of recorded range */
  endTime: number;
  /** Total duration in ms */
  duration: number;
  /** Number of snapshots available */
  snapshotCount: number;
  /** Current scrubber position (0-1) */
  position: number;
}

export interface ExportOptions {
  /** Export format */
  format: 'screenshot' | 'video' | 'json';
  /** For video: target duration in seconds */
  videoDuration?: number;
  /** For video: frames per second */
  fps?: number;
  /** Include state data in export */
  includeState?: boolean;
  /** Canvas element to capture (for screenshot/video) */
  canvas?: HTMLCanvasElement;
}

// =============================================================================
// Default Configuration
// =============================================================================

const DEFAULT_CONFIG: ReplayConfig = {
  maxDuration: 60 * 60 * 1000, // 1 hour
  snapshotInterval: 1000, // 1 second
  maxSnapshots: 3600, // 1 hour at 1/sec
  enableCompression: true,
  emitToBus: true,
};

// =============================================================================
// Replay Controller
// =============================================================================

export class ReplayController {
  private config: ReplayConfig;
  private snapshots: StateSnapshot[] = [];
  private playback: PlaybackState;
  private recordingActive: boolean = false;
  private recordingInterval: ReturnType<typeof setInterval> | null = null;
  private playbackInterval: ReturnType<typeof setInterval> | null = null;
  private currentStateGetter: (() => ChromaticState) | null = null;
  private onStateRestore: ((state: ChromaticState) => void) | null = null;
  private listeners: Map<string, Set<(data: unknown) => void>> = new Map();

  constructor(config: Partial<ReplayConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.playback = {
      isPlaying: false,
      isPaused: false,
      position: 0,
      currentTimestamp: 0,
      speed: 1,
      loop: false,
    };
  }

  // ===========================================================================
  // Recording
  // ===========================================================================

  /**
   * Start recording visualization state
   */
  startRecording(
    stateGetter: () => ChromaticState,
    onRestore?: (state: ChromaticState) => void
  ): void {
    if (this.recordingActive) return;

    this.currentStateGetter = stateGetter;
    this.onStateRestore = onRestore ?? null;
    this.recordingActive = true;

    // Take initial snapshot
    this.takeSnapshot();

    // Start periodic recording
    this.recordingInterval = setInterval(() => {
      this.takeSnapshot();
    }, this.config.snapshotInterval);

    this.emit('recording.started', { timestamp: Date.now() });
  }

  /**
   * Stop recording
   */
  stopRecording(): void {
    if (!this.recordingActive) return;

    this.recordingActive = false;
    if (this.recordingInterval) {
      clearInterval(this.recordingInterval);
      this.recordingInterval = null;
    }

    this.emit('recording.stopped', {
      timestamp: Date.now(),
      snapshotCount: this.snapshots.length,
    });
  }

  /**
   * Take a single snapshot of current state
   */
  takeSnapshot(triggerTopic?: string): StateSnapshot | null {
    if (!this.currentStateGetter) return null;

    const state = this.currentStateGetter();
    const now = Date.now();

    const snapshot: StateSnapshot = {
      id: this.generateSnapshotId(),
      timestamp_iso: new Date(now).toISOString(),
      timestamp_ms: now,
      state: this.serializeState(state),
      trigger_topic: triggerTopic,
    };

    // Apply compression if enabled
    if (this.config.enableCompression) {
      const compressed = this.compressSnapshot(snapshot);
      snapshot.compressed_size = compressed.length;
    }

    this.snapshots.push(snapshot);

    // Prune old snapshots if over limit
    this.pruneSnapshots();

    this.emit('snapshot.created', { id: snapshot.id, timestamp: now });

    return snapshot;
  }

  /**
   * Serialize ChromaticState for storage
   */
  private serializeState(state: ChromaticState): SerializedChromaticState {
    const agents: SerializedAgentData[] = [];

    state.agents.forEach((agent, id) => {
      agents.push({
        id,
        state: agent.state,
        hue: agent.hue,
        color: agent.color,
        intensity: agent.intensity,
        codeGraph: agent.codeGraph,
        branch: agent.branch,
        position: agent.position,
        opacity: agent.opacity,
        lastUpdate: agent.lastUpdate,
      });
    });

    return {
      agents,
      prismIntensity: state.prismIntensity,
      focusedAgent: state.focusedAgent,
      busConnected: state.busConnected,
      eventsPerMinute: state.eventsPerMinute,
      mainAhead: state.mainAhead,
    };
  }

  /**
   * Deserialize stored state back to ChromaticState
   */
  private deserializeState(serialized: SerializedChromaticState): ChromaticState {
    const agents = new Map<AgentId, AgentVisualData>();

    for (const agent of serialized.agents) {
      agents.set(agent.id, {
        id: agent.id,
        state: agent.state,
        hue: agent.hue,
        color: agent.color,
        intensity: agent.intensity,
        codeGraph: agent.codeGraph,
        branch: agent.branch,
        position: agent.position,
        opacity: agent.opacity,
        lastUpdate: agent.lastUpdate,
      });
    }

    return {
      agents,
      prismIntensity: serialized.prismIntensity,
      focusedAgent: serialized.focusedAgent,
      busConnected: serialized.busConnected,
      eventsPerMinute: serialized.eventsPerMinute,
      mainAhead: serialized.mainAhead,
    };
  }

  /**
   * Simple compression (JSON stringify with minimal whitespace)
   * In production, could use lz-string or similar
   */
  private compressSnapshot(snapshot: StateSnapshot): string {
    return JSON.stringify(snapshot);
  }

  /**
   * Prune snapshots exceeding max duration or count
   */
  private pruneSnapshots(): void {
    const now = Date.now();
    const cutoff = now - this.config.maxDuration;

    // Remove old snapshots
    this.snapshots = this.snapshots.filter(
      (s) => s.timestamp_ms >= cutoff
    );

    // Trim to max count
    while (this.snapshots.length > this.config.maxSnapshots) {
      this.snapshots.shift();
    }
  }

  // ===========================================================================
  // Playback Controls
  // ===========================================================================

  /**
   * Start playback from current position
   */
  play(): void {
    if (this.snapshots.length === 0) return;
    if (this.playback.isPlaying && !this.playback.isPaused) return;

    this.playback.isPlaying = true;
    this.playback.isPaused = false;

    const startTime = this.getTimestampAtPosition(this.playback.position);
    this.playback.currentTimestamp = startTime;

    this.playbackInterval = setInterval(() => {
      this.advancePlayback();
    }, 1000 / 60); // 60 FPS playback

    this.emit('playback.started', { position: this.playback.position });
  }

  /**
   * Pause playback
   */
  pause(): void {
    if (!this.playback.isPlaying) return;

    this.playback.isPaused = true;
    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }

    this.emit('playback.paused', { position: this.playback.position });
  }

  /**
   * Stop playback and reset to start
   */
  stop(): void {
    this.playback.isPlaying = false;
    this.playback.isPaused = false;
    this.playback.position = 0;
    this.playback.currentTimestamp = this.getStartTime();

    if (this.playbackInterval) {
      clearInterval(this.playbackInterval);
      this.playbackInterval = null;
    }

    this.emit('playback.stopped', {});
  }

  /**
   * Seek to specific position (0-1)
   */
  seek(position: number): void {
    position = Math.max(0, Math.min(1, position));
    this.playback.position = position;
    this.playback.currentTimestamp = this.getTimestampAtPosition(position);

    const snapshot = this.getSnapshotAtPosition(position);
    if (snapshot && this.onStateRestore) {
      const state = this.deserializeState(snapshot.state);
      this.onStateRestore(state);
    }

    this.emit('playback.seeked', { position });
  }

  /**
   * Seek to specific timestamp
   */
  seekToTimestamp(timestamp: number): void {
    const position = this.getPositionAtTimestamp(timestamp);
    this.seek(position);
  }

  /**
   * Set playback speed
   */
  setSpeed(speed: number): void {
    this.playback.speed = Math.max(0.1, Math.min(10, speed));
    this.emit('playback.speed', { speed: this.playback.speed });
  }

  /**
   * Toggle loop mode
   */
  setLoop(enabled: boolean): void {
    this.playback.loop = enabled;
  }

  /**
   * Advance playback by one frame
   */
  private advancePlayback(): void {
    const frameDelta = (1000 / 60) * this.playback.speed;
    this.playback.currentTimestamp += frameDelta;

    const endTime = this.getEndTime();

    if (this.playback.currentTimestamp >= endTime) {
      if (this.playback.loop) {
        this.playback.currentTimestamp = this.getStartTime();
        this.playback.position = 0;
      } else {
        this.stop();
        return;
      }
    }

    this.playback.position = this.getPositionAtTimestamp(this.playback.currentTimestamp);

    // Find and apply the snapshot for current time
    const snapshot = this.getSnapshotAtTimestamp(this.playback.currentTimestamp);
    if (snapshot && this.onStateRestore) {
      const state = this.deserializeState(snapshot.state);
      this.onStateRestore(state);
    }

    this.emit('playback.frame', {
      position: this.playback.position,
      timestamp: this.playback.currentTimestamp,
    });
  }

  // ===========================================================================
  // Scrubber UI Data
  // ===========================================================================

  /**
   * Get scrubber state for UI
   */
  getScrubberState(): ScrubberState {
    return {
      startTime: this.getStartTime(),
      endTime: this.getEndTime(),
      duration: this.getDuration(),
      snapshotCount: this.snapshots.length,
      position: this.playback.position,
    };
  }

  /**
   * Get playback state
   */
  getPlaybackState(): PlaybackState {
    return { ...this.playback };
  }

  /**
   * Get all snapshots (for timeline visualization)
   */
  getSnapshots(): StateSnapshot[] {
    return [...this.snapshots];
  }

  /**
   * Get snapshot at index
   */
  getSnapshotAt(index: number): StateSnapshot | null {
    return this.snapshots[index] ?? null;
  }

  // ===========================================================================
  // Time Helpers
  // ===========================================================================

  private getStartTime(): number {
    return this.snapshots[0]?.timestamp_ms ?? Date.now();
  }

  private getEndTime(): number {
    return this.snapshots[this.snapshots.length - 1]?.timestamp_ms ?? Date.now();
  }

  private getDuration(): number {
    return this.getEndTime() - this.getStartTime();
  }

  private getTimestampAtPosition(position: number): number {
    return this.getStartTime() + this.getDuration() * position;
  }

  private getPositionAtTimestamp(timestamp: number): number {
    const duration = this.getDuration();
    if (duration === 0) return 0;
    return (timestamp - this.getStartTime()) / duration;
  }

  private getSnapshotAtPosition(position: number): StateSnapshot | null {
    const timestamp = this.getTimestampAtPosition(position);
    return this.getSnapshotAtTimestamp(timestamp);
  }

  private getSnapshotAtTimestamp(timestamp: number): StateSnapshot | null {
    // Binary search for closest snapshot at or before timestamp
    let left = 0;
    let right = this.snapshots.length - 1;
    let result: StateSnapshot | null = null;

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      const snapshot = this.snapshots[mid];

      if (snapshot.timestamp_ms <= timestamp) {
        result = snapshot;
        left = mid + 1;
      } else {
        right = mid - 1;
      }
    }

    return result;
  }

  // ===========================================================================
  // Export Functions
  // ===========================================================================

  /**
   * Export screenshot of current state
   */
  async exportScreenshot(canvas: HTMLCanvasElement): Promise<Blob> {
    return new Promise((resolve, reject) => {
      try {
        canvas.toBlob((blob) => {
          if (blob) {
            resolve(blob);
          } else {
            reject(new Error('Failed to create screenshot blob'));
          }
        }, 'image/png');
      } catch (err) {
        reject(err);
      }
    });
  }

  /**
   * Export state as JSON
   */
  exportJSON(): string {
    return JSON.stringify({
      version: 1,
      exportedAt: new Date().toISOString(),
      config: this.config,
      snapshots: this.snapshots,
    }, null, 2);
  }

  /**
   * Import state from JSON
   */
  importJSON(json: string): boolean {
    try {
      const data = JSON.parse(json);
      if (data.version !== 1) {
        console.warn('Unsupported replay format version:', data.version);
        return false;
      }

      this.snapshots = data.snapshots || [];
      this.emit('import.completed', { snapshotCount: this.snapshots.length });
      return true;
    } catch (err) {
      console.error('Failed to import replay data:', err);
      return false;
    }
  }

  /**
   * Generate shareable URL with encoded state
   */
  generateShareableURL(baseUrl: string): string {
    const scrubberState = this.getScrubberState();
    const params = new URLSearchParams({
      replay: 'true',
      start: scrubberState.startTime.toString(),
      end: scrubberState.endTime.toString(),
      pos: scrubberState.position.toFixed(4),
    });

    return `${baseUrl}?${params.toString()}`;
  }

  /**
   * Parse shareable URL parameters
   */
  parseShareableURL(url: string): { start: number; end: number; position: number } | null {
    try {
      const urlObj = new URL(url);
      const replay = urlObj.searchParams.get('replay');

      if (replay !== 'true') return null;

      return {
        start: parseInt(urlObj.searchParams.get('start') ?? '0', 10),
        end: parseInt(urlObj.searchParams.get('end') ?? '0', 10),
        position: parseFloat(urlObj.searchParams.get('pos') ?? '0'),
      };
    } catch {
      return null;
    }
  }

  /**
   * Hook for video export - returns frames iterator
   * Caller is responsible for encoding (e.g., using WebCodecs or ffmpeg.wasm)
   */
  async *generateVideoFrames(
    options: ExportOptions
  ): AsyncGenerator<{ frame: ImageBitmap | null; timestamp: number; position: number }> {
    const { videoDuration = 30, fps = 30 } = options;
    const totalFrames = videoDuration * fps;
    const duration = this.getDuration();

    for (let i = 0; i < totalFrames; i++) {
      const position = i / totalFrames;
      const timestamp = this.getTimestampAtPosition(position);

      // Apply state for this frame
      const snapshot = this.getSnapshotAtTimestamp(timestamp);
      if (snapshot && this.onStateRestore) {
        const state = this.deserializeState(snapshot.state);
        this.onStateRestore(state);
      }

      // Yield frame placeholder - caller captures canvas
      yield {
        frame: null, // Caller captures canvas after state restoration
        timestamp,
        position,
      };

      // Small delay to allow rendering
      await new Promise((resolve) => setTimeout(resolve, 1));
    }
  }

  // ===========================================================================
  // Event System
  // ===========================================================================

  private generateSnapshotId(): string {
    return `snap_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  on(event: string, callback: (data: unknown) => void): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);

    return () => {
      this.listeners.get(event)?.delete(callback);
    };
  }

  private emit(event: string, data: unknown): void {
    const callbacks = this.listeners.get(event);
    if (callbacks) {
      for (const cb of callbacks) {
        try {
          cb(data);
        } catch (err) {
          console.error(`Error in replay event listener for ${event}:`, err);
        }
      }
    }
  }

  // ===========================================================================
  // Cleanup
  // ===========================================================================

  destroy(): void {
    this.stopRecording();
    this.stop();
    this.snapshots = [];
    this.listeners.clear();
    this.currentStateGetter = null;
    this.onStateRestore = null;
  }
}

// =============================================================================
// Singleton Instance
// =============================================================================

let replayInstance: ReplayController | null = null;

export function getReplayController(config?: Partial<ReplayConfig>): ReplayController {
  if (!replayInstance) {
    replayInstance = new ReplayController(config);
  }
  return replayInstance;
}

// =============================================================================
// Scrubber UI Component Types (for framework integration)
// =============================================================================

export interface ScrubberProps {
  /** Current scrubber state */
  state: ScrubberState;
  /** Current playback state */
  playback: PlaybackState;
  /** Callback when user seeks */
  onSeek: (position: number) => void;
  /** Callback for play/pause toggle */
  onPlayPause: () => void;
  /** Callback for stop */
  onStop: () => void;
  /** Callback for speed change */
  onSpeedChange: (speed: number) => void;
  /** Theme: light or dark */
  theme?: 'light' | 'dark';
}

/**
 * Generate scrubber HTML for framework-agnostic rendering
 */
export function renderScrubberHTML(props: ScrubberProps): string {
  const { state, playback, theme = 'dark' } = props;
  const bgColor = theme === 'dark' ? '#1a1a2e' : '#f5f5f5';
  const fgColor = theme === 'dark' ? '#fff' : '#000';
  const accentColor = '#00ffff'; // Cyan accent

  const formatTime = (ms: number): string => {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const elapsed = state.duration * state.position;
  const remaining = state.duration - elapsed;

  return `
    <div class="chromatic-scrubber" style="
      background: ${bgColor};
      color: ${fgColor};
      padding: 12px 16px;
      border-radius: 8px;
      font-family: system-ui, sans-serif;
      font-size: 12px;
    ">
      <!-- Time Display -->
      <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span class="elapsed">${formatTime(elapsed)}</span>
        <span class="remaining">-${formatTime(remaining)}</span>
      </div>

      <!-- Scrubber Track -->
      <div class="scrubber-track" style="
        height: 8px;
        background: rgba(255,255,255,0.2);
        border-radius: 4px;
        cursor: pointer;
        position: relative;
        margin-bottom: 12px;
      ">
        <div class="scrubber-progress" style="
          width: ${state.position * 100}%;
          height: 100%;
          background: ${accentColor};
          border-radius: 4px;
          position: relative;
        ">
          <div class="scrubber-thumb" style="
            position: absolute;
            right: -6px;
            top: -4px;
            width: 16px;
            height: 16px;
            background: ${accentColor};
            border: 2px solid ${fgColor};
            border-radius: 50%;
          "></div>
        </div>

        <!-- Snapshot Markers -->
        ${state.snapshotCount > 0 ? `
          <div class="snapshot-markers" style="
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
          ">
            <!-- Markers rendered dynamically -->
          </div>
        ` : ''}
      </div>

      <!-- Controls -->
      <div class="controls" style="display: flex; align-items: center; gap: 8px;">
        <button class="play-pause" style="
          background: none;
          border: none;
          color: ${fgColor};
          font-size: 18px;
          cursor: pointer;
          padding: 4px 8px;
        ">
          ${playback.isPlaying && !playback.isPaused ? '\u23F8' : '\u25B6'}
        </button>

        <button class="stop" style="
          background: none;
          border: none;
          color: ${fgColor};
          font-size: 18px;
          cursor: pointer;
          padding: 4px 8px;
        ">
          \u23F9
        </button>

        <select class="speed" style="
          background: rgba(255,255,255,0.1);
          border: 1px solid rgba(255,255,255,0.3);
          color: ${fgColor};
          padding: 4px 8px;
          border-radius: 4px;
          font-size: 11px;
        ">
          <option value="0.25" ${playback.speed === 0.25 ? 'selected' : ''}>0.25x</option>
          <option value="0.5" ${playback.speed === 0.5 ? 'selected' : ''}>0.5x</option>
          <option value="1" ${playback.speed === 1 ? 'selected' : ''}>1x</option>
          <option value="2" ${playback.speed === 2 ? 'selected' : ''}>2x</option>
          <option value="4" ${playback.speed === 4 ? 'selected' : ''}>4x</option>
        </select>

        <label style="display: flex; align-items: center; gap: 4px; margin-left: auto;">
          <input type="checkbox" class="loop" ${playback.loop ? 'checked' : ''} />
          Loop
        </label>

        <span class="snapshot-count" style="opacity: 0.7;">
          ${state.snapshotCount} snapshots
        </span>
      </div>
    </div>
  `;
}
