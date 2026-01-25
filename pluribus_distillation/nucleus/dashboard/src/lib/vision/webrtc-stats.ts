/**
 * WebRTC Stats Collector for SUPERWORKERS
 *
 * Production-grade monitoring of WebRTC connections with golden-ratio optimization.
 * Uses RTCPeerConnection.getStats() for bitrate, framerate, latency metrics.
 *
 * Best Practices (from research):
 * - Call getStats() once per second (recommended)
 * - Keep 3-5 second window for averaging jittery metrics
 * - Wait 2+ seconds after connect for RTCRemoteInboundRTP reports
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/RTCPeerConnection/getStats
 * @see https://webrtchacks.com/power-up-getstats-for-client-monitoring/
 * @see https://github.com/muaz-khan/getStats
 *
 * @module vision/webrtc-stats
 */

import { PHI, FIBONACCI, calculateGoldenScore, fibonacciTier } from './screen-capture';

// =============================================================================
// TYPES
// =============================================================================

export interface VideoMetrics {
  /** Bitrate in kbps */
  bitrate: number;
  /** Frames per second */
  frameRate: number;
  /** Packets lost (cumulative) */
  packetsLost: number;
  /** Packet loss ratio (0-1) */
  packetLossRatio: number;
  /** Jitter in milliseconds */
  jitter: number;
  /** Frame width */
  width: number;
  /** Frame height */
  height: number;
  /** Frames decoded (cumulative) */
  framesDecoded: number;
  /** Frames dropped (cumulative) */
  framesDropped: number;
  /** Frame drop ratio (0-1) */
  frameDropRatio: number;
}

export interface AudioMetrics {
  /** Bitrate in kbps */
  bitrate: number;
  /** Packets lost (cumulative) */
  packetsLost: number;
  /** Packet loss ratio (0-1) */
  packetLossRatio: number;
  /** Jitter in milliseconds */
  jitter: number;
  /** Audio level (0-1) */
  audioLevel: number;
}

export interface ConnectionMetrics {
  /** Round-trip time in milliseconds */
  rtt: number;
  /** Available outbound bandwidth estimate in kbps */
  availableBandwidth: number;
  /** ICE candidate pair state */
  candidatePairState: string;
  /** Local candidate type (host, srflx, relay) */
  localCandidateType: string;
  /** Remote candidate type */
  remoteCandidateType: string;
  /** Transport protocol (udp, tcp) */
  protocol: string;
}

export interface WebRTCMetrics {
  /** Collection timestamp */
  timestamp: number;
  /** Time since last collection (ms) */
  elapsed: number;
  /** Inbound video metrics */
  videoInbound: VideoMetrics | null;
  /** Outbound video metrics */
  videoOutbound: VideoMetrics | null;
  /** Inbound audio metrics */
  audioInbound: AudioMetrics | null;
  /** Outbound audio metrics */
  audioOutbound: AudioMetrics | null;
  /** Connection-level metrics */
  connection: ConnectionMetrics | null;
  /** Golden quality score (0-PHI) */
  goldenScore: number;
  /** Fibonacci tier (0-17) */
  fibonacciTier: number;
  /** Quality classification */
  qualityClass: 'excellent' | 'good' | 'fair' | 'poor' | 'critical';
}

export interface StatsCollectorOptions {
  /** Collection interval in milliseconds (default: 1000) */
  intervalMs?: number;
  /** Averaging window size (default: 5) */
  windowSize?: number;
  /** Emit to Pluribus bus */
  emitToBus?: boolean;
  /** Target video bitrate for scoring (kbps) */
  targetVideoBitrate?: number;
  /** Target frame rate for scoring */
  targetFrameRate?: number;
}

// =============================================================================
// GOLDEN RATIO QUALITY THRESHOLDS
// =============================================================================

/** Quality classification thresholds using golden ratio */
const QUALITY_THRESHOLDS = {
  excellent: 1 / PHI + 0.2,      // ~0.818 (high bar)
  good: 1 / PHI,                  // ~0.618
  fair: 1 / (PHI * PHI),          // ~0.382
  poor: 1 / (PHI * PHI * PHI),    // ~0.236
  // Below poor is critical
} as const;

/** Default targets for quality scoring */
const DEFAULT_TARGETS = {
  videoBitrate: 2500,   // 2.5 Mbps
  frameRate: 30,
  rtt: 100,             // 100ms max acceptable
  packetLoss: 0.01,     // 1% max acceptable
} as const;

// =============================================================================
// STATS COLLECTOR CLASS
// =============================================================================

/**
 * WebRTC Stats Collector with golden-ratio optimization.
 *
 * Usage:
 * ```typescript
 * const collector = new WebRTCStatsCollector(peerConnection, {
 *   intervalMs: 1000,
 *   emitToBus: true,
 * });
 *
 * collector.onMetrics((metrics) => {
 *   console.log('Quality:', metrics.qualityClass, 'Score:', metrics.goldenScore);
 * });
 *
 * collector.start();
 * // ... later
 * collector.stop();
 * ```
 */
export class WebRTCStatsCollector {
  private pc: RTCPeerConnection;
  private options: Required<StatsCollectorOptions>;
  private intervalId: ReturnType<typeof setInterval> | null = null;
  private callbacks: Array<(metrics: WebRTCMetrics) => void> = [];

  // Previous stats for delta calculation
  private prevStats: Map<string, RTCStats> = new Map();
  private prevTimestamp: number = 0;

  // Sliding window for averaging
  private metricsWindow: WebRTCMetrics[] = [];

  constructor(
    peerConnection: RTCPeerConnection,
    options: StatsCollectorOptions = {}
  ) {
    this.pc = peerConnection;
    this.options = {
      intervalMs: options.intervalMs ?? 1000,
      windowSize: options.windowSize ?? 5,
      emitToBus: options.emitToBus ?? true,
      targetVideoBitrate: options.targetVideoBitrate ?? DEFAULT_TARGETS.videoBitrate,
      targetFrameRate: options.targetFrameRate ?? DEFAULT_TARGETS.frameRate,
    };
  }

  /**
   * Register callback for metrics updates.
   */
  onMetrics(callback: (metrics: WebRTCMetrics) => void): void {
    this.callbacks.push(callback);
  }

  /**
   * Start collecting stats at configured interval.
   */
  start(): void {
    if (this.intervalId) return;

    // Initial delay of 2 seconds for RTCRemoteInboundRTP to be available
    setTimeout(() => {
      this.collect();
      this.intervalId = setInterval(() => this.collect(), this.options.intervalMs);
    }, 2000);
  }

  /**
   * Stop collecting stats.
   */
  stop(): void {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Collect stats once (manual trigger).
   */
  async collect(): Promise<WebRTCMetrics> {
    const report = await this.pc.getStats();
    const now = performance.now();
    const elapsed = this.prevTimestamp ? now - this.prevTimestamp : this.options.intervalMs;

    const metrics: WebRTCMetrics = {
      timestamp: now,
      elapsed,
      videoInbound: null,
      videoOutbound: null,
      audioInbound: null,
      audioOutbound: null,
      connection: null,
      goldenScore: 0,
      fibonacciTier: 0,
      qualityClass: 'critical',
    };

    // Process each stats object
    report.forEach((stats) => {
      this.processStats(stats, metrics, elapsed);
    });

    // Calculate golden score
    metrics.goldenScore = this.calculateOverallScore(metrics);
    metrics.fibonacciTier = fibonacciTier(metrics.goldenScore);
    metrics.qualityClass = this.classifyQuality(metrics.goldenScore);

    // Update sliding window
    this.metricsWindow.push(metrics);
    if (this.metricsWindow.length > this.options.windowSize) {
      this.metricsWindow.shift();
    }

    // Update previous state
    this.prevTimestamp = now;

    // Notify callbacks
    this.callbacks.forEach((cb) => cb(metrics));

    // Emit to bus
    if (this.options.emitToBus) {
      this.emitToBus(metrics);
    }

    return metrics;
  }

  /**
   * Get averaged metrics over the sliding window.
   */
  getAveraged(): WebRTCMetrics | null {
    if (this.metricsWindow.length === 0) return null;

    const window = this.metricsWindow;
    const count = window.length;

    // Average numeric fields
    const avgGoldenScore = window.reduce((sum, m) => sum + m.goldenScore, 0) / count;

    // Return most recent with averaged score
    const latest = window[window.length - 1];
    return {
      ...latest,
      goldenScore: avgGoldenScore,
      fibonacciTier: fibonacciTier(avgGoldenScore),
      qualityClass: this.classifyQuality(avgGoldenScore),
    };
  }

  // ===========================================================================
  // PRIVATE METHODS
  // ===========================================================================

  private processStats(
    stats: RTCStats,
    metrics: WebRTCMetrics,
    elapsed: number
  ): void {
    const prev = this.prevStats.get(stats.id);
    this.prevStats.set(stats.id, stats);

    switch (stats.type) {
      case 'inbound-rtp':
        this.processInboundRtp(stats as RTCInboundRtpStreamStats, prev, metrics, elapsed);
        break;
      case 'outbound-rtp':
        this.processOutboundRtp(stats as RTCOutboundRtpStreamStats, prev, metrics, elapsed);
        break;
      case 'candidate-pair':
        this.processCandidatePair(stats as RTCIceCandidatePairStats, metrics);
        break;
      case 'remote-inbound-rtp':
        this.processRemoteInboundRtp(stats as RTCRemoteInboundRtpStreamStats, metrics);
        break;
    }
  }

  private processInboundRtp(
    stats: RTCInboundRtpStreamStats,
    prev: RTCStats | undefined,
    metrics: WebRTCMetrics,
    elapsed: number
  ): void {
    const prevStats = prev as RTCInboundRtpStreamStats | undefined;
    const elapsedSec = elapsed / 1000;

    if (stats.kind === 'video') {
      const bytesReceived = stats.bytesReceived || 0;
      const prevBytes = prevStats?.bytesReceived || 0;
      const bitrate = prevStats ? ((bytesReceived - prevBytes) * 8) / elapsedSec / 1000 : 0;

      const packetsReceived = stats.packetsReceived || 0;
      const packetsLost = stats.packetsLost || 0;
      const totalPackets = packetsReceived + packetsLost;
      const packetLossRatio = totalPackets > 0 ? packetsLost / totalPackets : 0;

      const framesDecoded = stats.framesDecoded || 0;
      const framesDropped = stats.framesDropped || 0;
      const frameDropRatio = framesDecoded > 0 ? framesDropped / framesDecoded : 0;

      metrics.videoInbound = {
        bitrate,
        frameRate: stats.framesPerSecond || 0,
        packetsLost,
        packetLossRatio,
        jitter: (stats.jitter || 0) * 1000,
        width: stats.frameWidth || 0,
        height: stats.frameHeight || 0,
        framesDecoded,
        framesDropped,
        frameDropRatio,
      };
    } else if (stats.kind === 'audio') {
      const bytesReceived = stats.bytesReceived || 0;
      const prevBytes = prevStats?.bytesReceived || 0;
      const bitrate = prevStats ? ((bytesReceived - prevBytes) * 8) / elapsedSec / 1000 : 0;

      const packetsReceived = stats.packetsReceived || 0;
      const packetsLost = stats.packetsLost || 0;
      const totalPackets = packetsReceived + packetsLost;
      const packetLossRatio = totalPackets > 0 ? packetsLost / totalPackets : 0;

      metrics.audioInbound = {
        bitrate,
        packetsLost,
        packetLossRatio,
        jitter: (stats.jitter || 0) * 1000,
        audioLevel: (stats as any).audioLevel || 0,
      };
    }
  }

  private processOutboundRtp(
    stats: RTCOutboundRtpStreamStats,
    prev: RTCStats | undefined,
    metrics: WebRTCMetrics,
    elapsed: number
  ): void {
    const prevStats = prev as RTCOutboundRtpStreamStats | undefined;
    const elapsedSec = elapsed / 1000;

    if (stats.kind === 'video') {
      const bytesSent = stats.bytesSent || 0;
      const prevBytes = prevStats?.bytesSent || 0;
      const bitrate = prevStats ? ((bytesSent - prevBytes) * 8) / elapsedSec / 1000 : 0;

      metrics.videoOutbound = {
        bitrate,
        frameRate: stats.framesPerSecond || 0,
        packetsLost: 0,  // Outbound doesn't have this directly
        packetLossRatio: 0,
        jitter: 0,
        width: stats.frameWidth || 0,
        height: stats.frameHeight || 0,
        framesDecoded: 0,
        framesDropped: 0,
        frameDropRatio: 0,
      };
    } else if (stats.kind === 'audio') {
      const bytesSent = stats.bytesSent || 0;
      const prevBytes = prevStats?.bytesSent || 0;
      const bitrate = prevStats ? ((bytesSent - prevBytes) * 8) / elapsedSec / 1000 : 0;

      metrics.audioOutbound = {
        bitrate,
        packetsLost: 0,
        packetLossRatio: 0,
        jitter: 0,
        audioLevel: 0,
      };
    }
  }

  private processCandidatePair(
    stats: RTCIceCandidatePairStats,
    metrics: WebRTCMetrics
  ): void {
    if (stats.state !== 'succeeded') return;

    metrics.connection = {
      rtt: (stats.currentRoundTripTime || 0) * 1000,
      availableBandwidth: (stats.availableOutgoingBitrate || 0) / 1000,
      candidatePairState: stats.state,
      localCandidateType: (stats as any).localCandidateType || 'unknown',
      remoteCandidateType: (stats as any).remoteCandidateType || 'unknown',
      protocol: (stats as any).protocol || 'unknown',
    };
  }

  private processRemoteInboundRtp(
    stats: RTCRemoteInboundRtpStreamStats,
    metrics: WebRTCMetrics
  ): void {
    // RTT from remote reports (more accurate than candidate-pair for media)
    if (stats.roundTripTime !== undefined && metrics.connection) {
      metrics.connection.rtt = stats.roundTripTime * 1000;
    }
  }

  private calculateOverallScore(metrics: WebRTCMetrics): number {
    const factors: number[] = [];

    // Video quality factor
    if (metrics.videoInbound || metrics.videoOutbound) {
      const video = metrics.videoInbound || metrics.videoOutbound!;
      const bitrateScore = Math.min(video.bitrate / this.options.targetVideoBitrate, 1);
      const fpsScore = Math.min(video.frameRate / this.options.targetFrameRate, 1);
      const stabilityScore = 1 - Math.min(video.packetLossRatio * 10, 1);  // 10% loss = 0

      factors.push(
        calculateGoldenScore({
          resolution: bitrateScore,
          frameRate: fpsScore,
          stability: stabilityScore,
        })
      );
    }

    // Connection quality factor
    if (metrics.connection) {
      const rttScore = 1 - Math.min(metrics.connection.rtt / (DEFAULT_TARGETS.rtt * 5), 1);
      factors.push(rttScore);
    }

    // Geometric mean of all factors
    if (factors.length === 0) return 0;
    const product = factors.reduce((a, b) => a * b, 1);
    return Math.pow(product, 1 / factors.length);
  }

  private classifyQuality(score: number): WebRTCMetrics['qualityClass'] {
    if (score >= QUALITY_THRESHOLDS.excellent) return 'excellent';
    if (score >= QUALITY_THRESHOLDS.good) return 'good';
    if (score >= QUALITY_THRESHOLDS.fair) return 'fair';
    if (score >= QUALITY_THRESHOLDS.poor) return 'poor';
    return 'critical';
  }

  private emitToBus(metrics: WebRTCMetrics): void {
    try {
      if (typeof window !== 'undefined' && (window as any).__PLURIBUS_BUS__) {
        (window as any).__PLURIBUS_BUS__.emit({
          topic: 'vision.webrtc.stats',
          kind: 'metric',
          level: metrics.qualityClass === 'critical' ? 'warn' : 'debug',
          data: {
            goldenScore: metrics.goldenScore,
            fibonacciTier: metrics.fibonacciTier,
            qualityClass: metrics.qualityClass,
            videoBitrate: metrics.videoInbound?.bitrate || metrics.videoOutbound?.bitrate,
            frameRate: metrics.videoInbound?.frameRate || metrics.videoOutbound?.frameRate,
            rtt: metrics.connection?.rtt,
            packetLoss: metrics.videoInbound?.packetLossRatio,
            timestamp: metrics.timestamp,
          },
        });
      }
    } catch {
      // Silently ignore bus errors
    }
  }
}

// =============================================================================
// STANDALONE FUNCTIONS
// =============================================================================

/**
 * One-shot stats collection from a peer connection.
 */
export async function collectStats(
  pc: RTCPeerConnection,
  options?: StatsCollectorOptions
): Promise<WebRTCMetrics> {
  const collector = new WebRTCStatsCollector(pc, options);
  return collector.collect();
}

/**
 * Create a stats monitor that emits quality alerts.
 */
export function createQualityMonitor(
  pc: RTCPeerConnection,
  onQualityChange: (quality: WebRTCMetrics['qualityClass'], metrics: WebRTCMetrics) => void,
  options?: StatsCollectorOptions
): WebRTCStatsCollector {
  const collector = new WebRTCStatsCollector(pc, options);
  let lastQuality: WebRTCMetrics['qualityClass'] | null = null;

  collector.onMetrics((metrics) => {
    if (metrics.qualityClass !== lastQuality) {
      lastQuality = metrics.qualityClass;
      onQualityChange(metrics.qualityClass, metrics);
    }
  });

  return collector;
}

// =============================================================================
// EXPORTS
// =============================================================================

export {
  PHI,
  FIBONACCI,
  QUALITY_THRESHOLDS,
  DEFAULT_TARGETS,
};

export default WebRTCStatsCollector;
