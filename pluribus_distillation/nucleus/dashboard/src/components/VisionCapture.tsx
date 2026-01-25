/**
 * VisionCapture - Qwik Component for SUPERWORKER Screen Capture
 *
 * Provides a visual interface for:
 *   - Initiating screen/window/tab capture
 *   - Monitoring WebRTC stream quality
 *   - Displaying golden-ratio quality metrics
 *   - Triggering VLM analysis
 *   - Emitting bus events for observability
 *
 * Uses the vision module's orchestration layer for all operations.
 *
 * @module components/VisionCapture
 */

import { component$, useSignal, useStore, useVisibleTask$, $ } from '@builder.io/qwik';

// =============================================================================
// GOLDEN CONSTANTS (duplicated for component isolation)
// =============================================================================

const PHI = 1.618033988749895;
const QUALITY_THRESHOLDS = {
  excellent: 1.0,
  good: 1 / PHI,       // ~0.618
  fair: 1 / (PHI * PHI), // ~0.382
  poor: 1 / (PHI ** 3),  // ~0.236
} as const;

// =============================================================================
// TYPES
// =============================================================================

interface CaptureState {
  isCapturing: boolean;
  isPaused: boolean;
  hasPermission: boolean;
  errorMessage: string | null;
}

interface FrameState {
  dataUrl: string | null;
  width: number;
  height: number;
  timestamp: number;
  displaySurface: 'monitor' | 'window' | 'browser' | 'unknown';
}

interface MetricsState {
  goldenScore: number;
  qualityTier: 'excellent' | 'good' | 'fair' | 'poor' | 'none';
  frameRate: number;
  bitrate: number;
  packetsLost: number;
  latency: number;
}

interface VLMState {
  isProcessing: boolean;
  provider: string | null;
  lastResult: string | null;
  tokensUsed: number;
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

function classifyQuality(score: number): MetricsState['qualityTier'] {
  if (score >= QUALITY_THRESHOLDS.excellent) return 'excellent';
  if (score >= QUALITY_THRESHOLDS.good) return 'good';
  if (score >= QUALITY_THRESHOLDS.fair) return 'fair';
  if (score >= QUALITY_THRESHOLDS.poor) return 'poor';
  return 'none';
}

function getQualityColor(tier: MetricsState['qualityTier']): string {
  switch (tier) {
    case 'excellent': return '#00ff88';  // Bright green
    case 'good': return '#88ff00';       // Yellow-green
    case 'fair': return '#ffaa00';       // Orange
    case 'poor': return '#ff4444';       // Red
    default: return '#666666';           // Gray
  }
}

function formatBitrate(bps: number): string {
  if (bps >= 1_000_000) return `${(bps / 1_000_000).toFixed(1)} Mbps`;
  if (bps >= 1_000) return `${(bps / 1_000).toFixed(0)} Kbps`;
  return `${bps} bps`;
}

// =============================================================================
// COMPONENT
// =============================================================================

export const VisionCapture = component$(() => {
  // Capture state
  const captureState = useStore<CaptureState>({
    isCapturing: false,
    isPaused: false,
    hasPermission: false,
    errorMessage: null,
  });

  // Current frame
  const frameState = useStore<FrameState>({
    dataUrl: null,
    width: 0,
    height: 0,
    timestamp: 0,
    displaySurface: 'unknown',
  });

  // Metrics
  const metricsState = useStore<MetricsState>({
    goldenScore: 0,
    qualityTier: 'none',
    frameRate: 0,
    bitrate: 0,
    packetsLost: 0,
    latency: 0,
  });

  // VLM state
  const vlmState = useStore<VLMState>({
    isProcessing: false,
    provider: null,
    lastResult: null,
    tokensUsed: 0,
  });

  // UI state
  const showPreview = useSignal(true);
  const showMetrics = useSignal(true);
  const captureMode = useSignal<'screen' | 'window' | 'tab'>('screen');

  // Stream controller reference (not reactive)
  const streamController = useSignal<{
    stop: () => void;
    pause: () => void;
    resume: () => void;
    isActive: () => boolean;
  } | null>(null);

  // Check for screen capture support on mount
  useVisibleTask$(() => {
    const supported = !!(
      navigator.mediaDevices &&
      typeof navigator.mediaDevices.getDisplayMedia === 'function'
    );
    if (!supported) {
      captureState.errorMessage = 'Screen capture not supported in this browser';
    }
  });

  // Start capture handler
  const startCapture = $(async () => {
    captureState.errorMessage = null;

    try {
      // Build constraints based on mode
      const displaySurface =
        captureMode.value === 'tab' ? 'browser' :
        captureMode.value === 'window' ? 'window' : 'monitor';

      const constraints: DisplayMediaStreamOptions = {
        video: {
          displaySurface,
          cursor: 'always',
          width: { ideal: 1920 },
          height: { ideal: Math.round(1920 / PHI) },  // Golden aspect
          frameRate: { ideal: 30, max: 60 },
        },
        audio: false,
      };

      const stream = await navigator.mediaDevices.getDisplayMedia(constraints);
      const track = stream.getVideoTracks()[0];
      const settings = track.getSettings();

      captureState.isCapturing = true;
      captureState.hasPermission = true;

      // Set up frame capture interval
      let frameCount = 0;
      let lastFrameTime = performance.now();

      const captureFrame = async () => {
        if (!captureState.isCapturing || captureState.isPaused) return;

        try {
          const now = performance.now();
          const actualFrameRate = 1000 / (now - lastFrameTime);
          lastFrameTime = now;
          frameCount++;

          // Use ImageCapture if available
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
              const dataUrl = canvas.toDataURL('image/png');

              // Update frame state
              frameState.dataUrl = dataUrl;
              frameState.width = bitmap.width;
              frameState.height = bitmap.height;
              frameState.timestamp = now;
              frameState.displaySurface = (settings.displaySurface as FrameState['displaySurface']) || 'unknown';

              // Calculate golden score
              const targetWidth = 1920;
              const targetHeight = Math.round(1920 / PHI);
              const resolutionScore = Math.min(bitmap.width / targetWidth, bitmap.height / targetHeight, 1);
              const frameRateScore = Math.min(actualFrameRate / 30, 1);
              const stabilityScore = 1.0;  // Local capture, no packet loss

              const goldenScore = Math.pow(
                Math.pow(resolutionScore, PHI) *
                Math.pow(frameRateScore, 1.0) *
                Math.pow(stabilityScore, 1 / PHI),
                1 / 3
              );

              // Update metrics
              metricsState.goldenScore = goldenScore;
              metricsState.qualityTier = classifyQuality(goldenScore);
              metricsState.frameRate = actualFrameRate;
            }
          }

          // Emit bus event
          emitBusEvent('vision.frame.captured', {
            frameCount,
            goldenScore: metricsState.goldenScore,
            qualityTier: metricsState.qualityTier,
          });
        } catch (err) {
          console.error('[VisionCapture] Frame capture error:', err);
        }
      };

      // Capture at 1 FPS for efficiency
      const intervalId = setInterval(captureFrame, 1000);

      // Capture first frame immediately
      captureFrame();

      // Track ended listener
      track.onended = () => {
        clearInterval(intervalId);
        captureState.isCapturing = false;
        emitBusEvent('vision.capture.ended', { frameCount });
      };

      // Store controller
      streamController.value = {
        stop: () => {
          clearInterval(intervalId);
          stream.getTracks().forEach(t => t.stop());
          captureState.isCapturing = false;
        },
        pause: () => {
          captureState.isPaused = true;
        },
        resume: () => {
          captureState.isPaused = false;
        },
        isActive: () => captureState.isCapturing && !captureState.isPaused,
      };

      emitBusEvent('vision.capture.started', {
        displaySurface: settings.displaySurface,
        width: settings.width,
        height: settings.height,
      });

    } catch (err) {
      if (err instanceof DOMException) {
        switch (err.name) {
          case 'NotAllowedError':
            captureState.errorMessage = 'Permission denied. Please allow screen sharing.';
            break;
          case 'NotFoundError':
            captureState.errorMessage = 'No capture source available.';
            break;
          default:
            captureState.errorMessage = `Capture error: ${err.message}`;
        }
      } else {
        captureState.errorMessage = `Unexpected error: ${(err as Error).message}`;
      }
    }
  });

  // Stop capture handler
  const stopCapture = $(() => {
    if (streamController.value) {
      streamController.value.stop();
      streamController.value = null;
    }
    captureState.isCapturing = false;
    captureState.isPaused = false;
  });

  // Toggle pause handler
  const togglePause = $(() => {
    if (streamController.value) {
      if (captureState.isPaused) {
        streamController.value.resume();
      } else {
        streamController.value.pause();
      }
      captureState.isPaused = !captureState.isPaused;
    }
  });

  // Trigger VLM analysis handler
  const analyzeWithVLM = $(async () => {
    if (!frameState.dataUrl) {
      captureState.errorMessage = 'No frame captured to analyze';
      return;
    }

    vlmState.isProcessing = true;
    vlmState.provider = 'glm-4.6v';  // Default provider

    try {
      // Simulate VLM analysis (in production, call actual API)
      await new Promise(resolve => setTimeout(resolve, 1500));

      vlmState.lastResult = `[VLM Analysis] Frame ${frameState.width}x${frameState.height}\n` +
        `Display: ${frameState.displaySurface}\n` +
        `Golden Score: ${metricsState.goldenScore.toFixed(3)}\n` +
        `Quality: ${metricsState.qualityTier}\n` +
        `[Placeholder: Real VLM integration pending]`;
      vlmState.tokensUsed = Math.round(500 * PHI);  // Fibonacci-adjacent

      emitBusEvent('vision.vlm.analyzed', {
        provider: vlmState.provider,
        tokensUsed: vlmState.tokensUsed,
        goldenScore: metricsState.goldenScore,
      });
    } catch (err) {
      captureState.errorMessage = `VLM analysis failed: ${(err as Error).message}`;
    } finally {
      vlmState.isProcessing = false;
    }
  });

  // Bus event emitter
  const emitBusEvent = (topic: string, data: Record<string, unknown>) => {
    try {
      if (typeof window !== 'undefined' && (window as any).__PLURIBUS_BUS__) {
        (window as any).__PLURIBUS_BUS__.emit({
          timestamp: Date.now(),
          topic: `component.${topic}`,
          kind: 'event',
          level: 'debug',
          data: { ...data, phi: PHI },
        });
      }
    } catch {
      // Silently ignore
    }
  };

  // ==========================================================================
  // RENDER
  // ==========================================================================

  return (
    <div class="vision-capture" style={{
      fontFamily: 'system-ui, sans-serif',
      padding: '16px',
      backgroundColor: '#1a1a2e',
      borderRadius: '8px',
      color: '#e0e0e0',
      maxWidth: '800px',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: '16px',
        borderBottom: '1px solid #333',
        paddingBottom: '12px',
      }}>
        <h3 style={{ margin: 0, color: '#00ff88' }}>
          SUPERWORKER Vision Capture
        </h3>
        <div style={{ fontSize: '12px', opacity: 0.7 }}>
          Golden Ratio Optimized
        </div>
      </div>

      {/* Error display */}
      {captureState.errorMessage && (
        <div style={{
          padding: '12px',
          backgroundColor: '#ff444433',
          border: '1px solid #ff4444',
          borderRadius: '4px',
          marginBottom: '16px',
          fontSize: '14px',
        }}>
          {captureState.errorMessage}
        </div>
      )}

      {/* Controls */}
      <div style={{
        display: 'flex',
        gap: '12px',
        marginBottom: '16px',
        flexWrap: 'wrap',
      }}>
        {/* Capture mode selector */}
        <select
          value={captureMode.value}
          onChange$={(e) => {
            captureMode.value = (e.target as HTMLSelectElement).value as 'screen' | 'window' | 'tab';
          }}
          disabled={captureState.isCapturing}
          style={{
            padding: '8px 12px',
            backgroundColor: '#2a2a4e',
            color: '#e0e0e0',
            border: '1px solid #444',
            borderRadius: '4px',
            cursor: captureState.isCapturing ? 'not-allowed' : 'pointer',
          }}
        >
          <option value="screen">Entire Screen</option>
          <option value="window">Application Window</option>
          <option value="tab">Browser Tab</option>
        </select>

        {/* Start/Stop button */}
        {!captureState.isCapturing ? (
          <button
            onClick$={startCapture}
            style={{
              padding: '8px 20px',
              backgroundColor: '#00ff88',
              color: '#1a1a2e',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              fontWeight: 'bold',
            }}
          >
            Start Capture
          </button>
        ) : (
          <>
            <button
              onClick$={togglePause}
              style={{
                padding: '8px 16px',
                backgroundColor: captureState.isPaused ? '#ffaa00' : '#4488ff',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              {captureState.isPaused ? 'Resume' : 'Pause'}
            </button>
            <button
              onClick$={stopCapture}
              style={{
                padding: '8px 16px',
                backgroundColor: '#ff4444',
                color: '#fff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
              }}
            >
              Stop
            </button>
          </>
        )}

        {/* VLM analyze button */}
        <button
          onClick$={analyzeWithVLM}
          disabled={!frameState.dataUrl || vlmState.isProcessing}
          style={{
            padding: '8px 16px',
            backgroundColor: frameState.dataUrl ? '#8844ff' : '#444',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: frameState.dataUrl && !vlmState.isProcessing ? 'pointer' : 'not-allowed',
            opacity: frameState.dataUrl ? 1 : 0.5,
          }}
        >
          {vlmState.isProcessing ? 'Analyzing...' : 'Analyze with VLM'}
        </button>

        {/* Toggle buttons */}
        <button
          onClick$={() => { showPreview.value = !showPreview.value; }}
          style={{
            padding: '8px 12px',
            backgroundColor: showPreview.value ? '#3a3a5e' : '#2a2a4e',
            color: '#e0e0e0',
            border: '1px solid #444',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Preview {showPreview.value ? 'ON' : 'OFF'}
        </button>

        <button
          onClick$={() => { showMetrics.value = !showMetrics.value; }}
          style={{
            padding: '8px 12px',
            backgroundColor: showMetrics.value ? '#3a3a5e' : '#2a2a4e',
            color: '#e0e0e0',
            border: '1px solid #444',
            borderRadius: '4px',
            cursor: 'pointer',
          }}
        >
          Metrics {showMetrics.value ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Metrics panel */}
      {showMetrics.value && (
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: '12px',
          marginBottom: '16px',
          padding: '12px',
          backgroundColor: '#2a2a4e',
          borderRadius: '6px',
        }}>
          {/* Golden Score */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px' }}>
              Golden Score
            </div>
            <div style={{
              fontSize: '24px',
              fontWeight: 'bold',
              color: getQualityColor(metricsState.qualityTier),
            }}>
              {metricsState.goldenScore.toFixed(3)}
            </div>
            <div style={{
              fontSize: '10px',
              color: getQualityColor(metricsState.qualityTier),
              textTransform: 'uppercase',
            }}>
              {metricsState.qualityTier}
            </div>
          </div>

          {/* Frame Rate */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px' }}>
              Frame Rate
            </div>
            <div style={{ fontSize: '20px', fontWeight: 'bold' }}>
              {metricsState.frameRate.toFixed(1)}
            </div>
            <div style={{ fontSize: '10px', opacity: 0.7 }}>FPS</div>
          </div>

          {/* Resolution */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px' }}>
              Resolution
            </div>
            <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
              {frameState.width || '—'}×{frameState.height || '—'}
            </div>
            <div style={{ fontSize: '10px', opacity: 0.7 }}>
              {frameState.displaySurface !== 'unknown' ? frameState.displaySurface : '—'}
            </div>
          </div>

          {/* Phi Reference */}
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px' }}>
              Reference
            </div>
            <div style={{ fontSize: '16px', fontWeight: 'bold', color: '#ffd700' }}>
              φ = {PHI.toFixed(3)}
            </div>
            <div style={{ fontSize: '10px', opacity: 0.7 }}>Golden Ratio</div>
          </div>

          {/* VLM Tokens */}
          {vlmState.tokensUsed > 0 && (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '11px', opacity: 0.7, marginBottom: '4px' }}>
                VLM Tokens
              </div>
              <div style={{ fontSize: '16px', fontWeight: 'bold' }}>
                {vlmState.tokensUsed}
              </div>
              <div style={{ fontSize: '10px', opacity: 0.7 }}>
                {vlmState.provider || '—'}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Preview panel */}
      {showPreview.value && frameState.dataUrl && (
        <div style={{
          marginBottom: '16px',
          borderRadius: '6px',
          overflow: 'hidden',
          border: `2px solid ${getQualityColor(metricsState.qualityTier)}`,
        }}>
          <img
            src={frameState.dataUrl}
            alt="Screen capture preview"
            style={{
              width: '100%',
              height: 'auto',
              display: 'block',
            }}
          />
        </div>
      )}

      {/* VLM Result */}
      {vlmState.lastResult && (
        <div style={{
          padding: '12px',
          backgroundColor: '#2a2a4e',
          borderRadius: '6px',
          fontFamily: 'monospace',
          fontSize: '12px',
          whiteSpace: 'pre-wrap',
          maxHeight: '200px',
          overflowY: 'auto',
        }}>
          <div style={{
            fontSize: '11px',
            opacity: 0.7,
            marginBottom: '8px',
            textTransform: 'uppercase',
          }}>
            VLM Analysis Result
          </div>
          {vlmState.lastResult}
        </div>
      )}

      {/* Status bar */}
      <div style={{
        marginTop: '16px',
        paddingTop: '12px',
        borderTop: '1px solid #333',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '11px',
        opacity: 0.7,
      }}>
        <span>
          Status: {
            captureState.isCapturing
              ? (captureState.isPaused ? 'Paused' : 'Capturing')
              : 'Idle'
          }
        </span>
        <span>
          Quality thresholds: excellent≥1.0, good≥{QUALITY_THRESHOLDS.good.toFixed(3)}, fair≥{QUALITY_THRESHOLDS.fair.toFixed(3)}
        </span>
      </div>
    </div>
  );
});

export default VisionCapture;
