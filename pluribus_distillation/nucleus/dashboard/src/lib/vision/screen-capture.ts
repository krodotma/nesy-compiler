/**
 * WebRTC Screen Capture Module for SUPERWORKERS
 *
 * Full implementation of screen/window/tab capture using getDisplayMedia API.
 * Integrates with VLM providers for visual context injection.
 *
 * @see https://www.metered.ca/blog/webrtc-screen-sharing/
 * @see https://webrtc.github.io/samples/src/content/getusermedia/getdisplaymedia/
 * @see https://w3c.github.io/mediacapture-screen-share/
 *
 * @module vision/screen-capture
 */

// =============================================================================
// CONSTANTS - Golden Ratio Optimization
// =============================================================================

/** Golden ratio for geometric optimization */
export const PHI = 1.618033988749895;

/** Fibonacci sequence for token/quality budgets */
export const FIBONACCI = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584] as const;

/** Golden-ratio scaled quality tiers (0-1 normalized) */
export const GOLDEN_QUALITY_TIERS = {
  minimal: 1 / (PHI * PHI * PHI),    // ~0.236
  low: 1 / (PHI * PHI),               // ~0.382
  medium: 1 / PHI,                    // ~0.618
  high: 1.0,                          // 1.0
  ultra: PHI,                         // ~1.618 (oversampling)
} as const;

/** Default capture constraints with golden-ratio aspect */
export const DEFAULT_CONSTRAINTS = {
  width: { ideal: 1920 },
  height: { ideal: Math.round(1920 / PHI) },  // ~1187 (golden aspect)
  frameRate: { ideal: 30, max: 60 },
} as const;

// =============================================================================
// TYPES
// =============================================================================

export interface ScreenCaptureOptions {
  /** Prefer browser tab capture */
  preferTab?: boolean;
  /** Prefer application window capture */
  preferWindow?: boolean;
  /** Prefer entire screen/monitor capture */
  preferScreen?: boolean;
  /** Cursor visibility: always show, only on motion, or never */
  includeCursor?: 'always' | 'motion' | 'never';
  /** Include audio (Windows-only for desktop, cross-platform for tab) */
  includeAudio?: boolean;
  /** Target resolution width */
  width?: number;
  /** Target resolution height */
  height?: number;
  /** Target frame rate */
  frameRate?: number;
  /** Quality tier (affects compression) */
  qualityTier?: keyof typeof GOLDEN_QUALITY_TIERS;
}

export interface CapturedFrame {
  /** Base64-encoded PNG data URL */
  dataUrl: string;
  /** Frame width in pixels */
  width: number;
  /** Frame height in pixels */
  height: number;
  /** Capture timestamp (performance.now()) */
  timestamp: number;
  /** Display surface type */
  displaySurface: 'monitor' | 'window' | 'browser' | 'unknown';
  /** Golden score (0-1) based on quality metrics */
  goldenScore: number;
}

export interface ScreenCaptureStats {
  width: number;
  height: number;
  frameRate: number;
  displaySurface: string;
  bitrate?: number;
  packetsLost?: number;
  /** Golden quality score (0-PHI range) */
  goldenScore: number;
  /** Fibonacci tier index (0-17) */
  fibonacciTier: number;
}

export interface StreamController {
  /** Stop the stream and cleanup */
  stop: () => void;
  /** Pause frame capture (stream continues) */
  pause: () => void;
  /** Resume frame capture */
  resume: () => void;
  /** Get current stream */
  getStream: () => MediaStream | null;
  /** Check if actively capturing */
  isActive: () => boolean;
}

// =============================================================================
// GOLDEN RATIO UTILITIES
// =============================================================================

/**
 * Calculate golden score from quality metrics.
 * Uses geometric mean with phi-weighted factors.
 */
export function calculateGoldenScore(metrics: {
  resolution: number;      // 0-1 (actual/target)
  frameRate: number;       // 0-1 (actual/target)
  stability: number;       // 0-1 (1 - packet_loss_ratio)
  latency?: number;        // 0-1 (1 - normalized_latency)
}): number {
  const weights = {
    resolution: PHI,           // Most important
    frameRate: 1.0,
    stability: 1 / PHI,        // Least variable
    latency: 1 / (PHI * PHI),  // Often unavailable
  };

  const factors = [
    Math.pow(metrics.resolution, weights.resolution),
    Math.pow(metrics.frameRate, weights.frameRate),
    Math.pow(metrics.stability, weights.stability),
  ];

  if (metrics.latency !== undefined) {
    factors.push(Math.pow(metrics.latency, weights.latency));
  }

  // Geometric mean
  const product = factors.reduce((a, b) => a * b, 1);
  return Math.pow(product, 1 / factors.length);
}

/**
 * Find nearest Fibonacci tier for a given value.
 * Returns index into FIBONACCI array.
 */
export function fibonacciTier(value: number, scale: number = 1000): number {
  const scaled = value * scale;
  for (let i = 0; i < FIBONACCI.length; i++) {
    if (FIBONACCI[i] >= scaled) {
      return i;
    }
  }
  return FIBONACCI.length - 1;
}

/**
 * Calculate optimal dimensions using golden ratio.
 */
export function goldenDimensions(
  targetWidth: number,
  aspectMode: 'golden' | 'wide' | 'square' = 'golden'
): { width: number; height: number } {
  switch (aspectMode) {
    case 'golden':
      return { width: targetWidth, height: Math.round(targetWidth / PHI) };
    case 'wide':
      return { width: targetWidth, height: Math.round(targetWidth / (PHI * PHI)) };
    case 'square':
      return { width: targetWidth, height: targetWidth };
  }
}

// =============================================================================
// CORE CAPTURE FUNCTIONS
// =============================================================================

/**
 * Check if screen capture is supported in this browser.
 */
export function isScreenCaptureSupported(): boolean {
  return !!(
    navigator.mediaDevices &&
    typeof navigator.mediaDevices.getDisplayMedia === 'function'
  );
}

/**
 * Capture screen/window/tab using WebRTC getDisplayMedia API.
 *
 * User will be prompted to select what to share:
 * - Entire screen (all monitors if multiple displays)
 * - Specific application window
 * - Browser tab
 *
 * @throws {Error} If permission denied or no source available
 */
export async function captureScreen(
  options: ScreenCaptureOptions = {}
): Promise<MediaStream> {
  if (!isScreenCaptureSupported()) {
    throw new Error('Screen capture not supported in this browser');
  }

  // Build display surface hint
  let displaySurface: 'browser' | 'window' | 'monitor' | undefined;
  if (options.preferTab) displaySurface = 'browser';
  else if (options.preferWindow) displaySurface = 'window';
  else if (options.preferScreen) displaySurface = 'monitor';

  // Apply golden-ratio scaling to dimensions
  const qualityMultiplier = options.qualityTier
    ? GOLDEN_QUALITY_TIERS[options.qualityTier]
    : 1.0;

  const constraints: DisplayMediaStreamOptions = {
    video: {
      cursor: options.includeCursor || 'always',
      displaySurface,
      width: options.width
        ? { ideal: Math.round(options.width * qualityMultiplier) }
        : DEFAULT_CONSTRAINTS.width,
      height: options.height
        ? { ideal: Math.round(options.height * qualityMultiplier) }
        : DEFAULT_CONSTRAINTS.height,
      frameRate: options.frameRate
        ? { ideal: options.frameRate }
        : DEFAULT_CONSTRAINTS.frameRate,
    },
    audio: options.includeAudio || false,
  };

  try {
    const stream = await navigator.mediaDevices.getDisplayMedia(constraints);

    // Emit capture start event to bus (if available)
    emitCaptureEvent('screen_capture.started', {
      displaySurface: stream.getVideoTracks()[0]?.getSettings().displaySurface,
      hasAudio: stream.getAudioTracks().length > 0,
    });

    return stream;
  } catch (err) {
    if (err instanceof DOMException) {
      switch (err.name) {
        case 'NotAllowedError':
          throw new Error('Screen capture permission denied by user');
        case 'NotFoundError':
          throw new Error('No screen capture source available');
        case 'NotReadableError':
          throw new Error('Screen capture source is not readable (possibly in use)');
        case 'OverconstrainedError':
          throw new Error(`Screen capture constraints cannot be satisfied: ${err.message}`);
        case 'AbortError':
          throw new Error('Screen capture was aborted');
      }
    }
    throw err;
  }
}

/**
 * Capture a single frame from screen as base64 PNG for VLM analysis.
 * Automatically stops the stream after capture.
 */
export async function captureFrameForVLM(
  options: ScreenCaptureOptions = {}
): Promise<CapturedFrame> {
  const stream = await captureScreen(options);
  const startTime = performance.now();

  try {
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();

    // Use ImageCapture API if available (Chrome, Edge)
    if ('ImageCapture' in window) {
      const ImageCaptureClass = (window as unknown as { ImageCapture: new (track: MediaStreamTrack) => { grabFrame(): Promise<ImageBitmap> } }).ImageCapture;
      const imageCapture = new ImageCaptureClass(track);
      const bitmap = await imageCapture.grabFrame();

      const canvas = document.createElement('canvas');
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Canvas 2D context unavailable');
      ctx.drawImage(bitmap, 0, 0);

      const dataUrl = canvas.toDataURL('image/png');

      // Calculate golden score
      const targetWidth = options.width || DEFAULT_CONSTRAINTS.width.ideal;
      const targetHeight = options.height || DEFAULT_CONSTRAINTS.height.ideal!;
      const goldenScore = calculateGoldenScore({
        resolution: Math.min(bitmap.width / targetWidth, bitmap.height / targetHeight, 1),
        frameRate: 1.0,  // Single frame, assume good
        stability: 1.0,   // No packet loss for local capture
      });

      return {
        dataUrl,
        width: bitmap.width,
        height: bitmap.height,
        timestamp: performance.now(),
        displaySurface: (settings.displaySurface as CapturedFrame['displaySurface']) || 'unknown',
        goldenScore,
      };
    }

    // Fallback: Use video element + canvas
    const video = document.createElement('video');
    video.srcObject = stream;
    video.muted = true;
    await video.play();

    // Wait for video to have dimensions
    await new Promise<void>((resolve) => {
      if (video.videoWidth > 0) {
        resolve();
      } else {
        video.onloadedmetadata = () => resolve();
      }
    });

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Canvas 2D context unavailable');
    ctx.drawImage(video, 0, 0);

    const dataUrl = canvas.toDataURL('image/png');

    const targetWidth = options.width || DEFAULT_CONSTRAINTS.width.ideal;
    const targetHeight = options.height || DEFAULT_CONSTRAINTS.height.ideal!;
    const goldenScore = calculateGoldenScore({
      resolution: Math.min(video.videoWidth / targetWidth, video.videoHeight / targetHeight, 1),
      frameRate: 1.0,
      stability: 1.0,
    });

    return {
      dataUrl,
      width: video.videoWidth,
      height: video.videoHeight,
      timestamp: performance.now(),
      displaySurface: (settings.displaySurface as CapturedFrame['displaySurface']) || 'unknown',
      goldenScore,
    };
  } finally {
    // Always stop tracks to release screen capture
    stream.getTracks().forEach((track) => track.stop());
    emitCaptureEvent('screen_capture.stopped', {
      duration: performance.now() - startTime,
    });
  }
}

/**
 * Stream screen capture with continuous frame callbacks and stats monitoring.
 * Returns a controller for managing the stream lifecycle.
 */
export async function streamScreenWithStats(
  onFrame: (frame: CapturedFrame) => void,
  onStats: (stats: ScreenCaptureStats) => void,
  options: ScreenCaptureOptions & { intervalMs?: number } = {}
): Promise<StreamController> {
  const intervalMs = options.intervalMs || 1000;
  const stream = await captureScreen(options);
  const track = stream.getVideoTracks()[0];

  let frameInterval: ReturnType<typeof setInterval> | null = null;
  let isPaused = false;
  let frameCount = 0;
  let lastFrameTime = performance.now();

  // Calculate target metrics for golden scoring
  const targetWidth = options.width || DEFAULT_CONSTRAINTS.width.ideal;
  const targetHeight = options.height || DEFAULT_CONSTRAINTS.height.ideal!;
  const targetFrameRate = options.frameRate || DEFAULT_CONSTRAINTS.frameRate.ideal;

  const captureFrame = async () => {
    if (isPaused) return;

    try {
      const settings = track.getSettings();
      const now = performance.now();
      const actualFrameRate = 1000 / (now - lastFrameTime);
      lastFrameTime = now;
      frameCount++;

      // Capture frame
      let dataUrl: string;
      let width: number;
      let height: number;

      if ('ImageCapture' in window) {
        const ImageCaptureClass = (window as unknown as { ImageCapture: new (track: MediaStreamTrack) => { grabFrame(): Promise<ImageBitmap> } }).ImageCapture;
        const imageCapture = new ImageCaptureClass(track);
        const bitmap = await imageCapture.grabFrame();

        const canvas = document.createElement('canvas');
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        const ctx = canvas.getContext('2d');
        if (ctx) {
          ctx.drawImage(bitmap, 0, 0);
          dataUrl = canvas.toDataURL('image/png');
        } else {
          return;
        }
        width = bitmap.width;
        height = bitmap.height;
      } else {
        // Fallback implementation would go here
        return;
      }

      // Calculate golden score
      const goldenScore = calculateGoldenScore({
        resolution: Math.min(width / targetWidth, height / targetHeight, 1),
        frameRate: Math.min(actualFrameRate / targetFrameRate, 1),
        stability: 1.0,  // Local capture, no packet loss
      });

      const frame: CapturedFrame = {
        dataUrl,
        width,
        height,
        timestamp: now,
        displaySurface: (settings.displaySurface as CapturedFrame['displaySurface']) || 'unknown',
        goldenScore,
      };

      onFrame(frame);

      // Emit stats
      const stats: ScreenCaptureStats = {
        width,
        height,
        frameRate: actualFrameRate,
        displaySurface: settings.displaySurface || 'unknown',
        goldenScore,
        fibonacciTier: fibonacciTier(goldenScore),
      };

      onStats(stats);
    } catch (err) {
      console.error('[screen-capture] Frame capture failed:', err);
    }
  };

  // Start frame capture interval
  frameInterval = setInterval(captureFrame, intervalMs);

  // Capture first frame immediately
  captureFrame();

  // Return controller
  return {
    stop: () => {
      if (frameInterval) {
        clearInterval(frameInterval);
        frameInterval = null;
      }
      stream.getTracks().forEach((t) => t.stop());
      emitCaptureEvent('screen_capture.stopped', { frameCount });
    },
    pause: () => {
      isPaused = true;
      emitCaptureEvent('screen_capture.paused', { frameCount });
    },
    resume: () => {
      isPaused = false;
      emitCaptureEvent('screen_capture.resumed', { frameCount });
    },
    getStream: () => stream,
    isActive: () => !isPaused && frameInterval !== null,
  };
}

// =============================================================================
// BUS EVENT EMISSION
// =============================================================================

/**
 * Emit screen capture event to Pluribus bus (if available).
 */
function emitCaptureEvent(topic: string, data: Record<string, unknown>): void {
  try {
    // Check if bus bridge is available
    if (typeof window !== 'undefined' && (window as any).__PLURIBUS_BUS__) {
      (window as any).__PLURIBUS_BUS__.emit({
        topic: `vision.${topic}`,
        kind: 'metric',
        level: 'debug',
        data: {
          ...data,
          timestamp: Date.now(),
          goldenPhi: PHI,
        },
      });
    }
  } catch {
    // Silently ignore bus errors
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export default {
  captureScreen,
  captureFrameForVLM,
  streamScreenWithStats,
  isScreenCaptureSupported,
  calculateGoldenScore,
  fibonacciTier,
  goldenDimensions,
  PHI,
  FIBONACCI,
  GOLDEN_QUALITY_TIERS,
};
