/**
 * Browser API Type Declarations for SUPERWORKERS Vision Module
 *
 * These declarations extend the standard DOM types with APIs that exist
 * in modern browsers but aren't yet in TypeScript's lib.dom.d.ts.
 *
 * @see https://developer.mozilla.org/en-US/docs/Web/API/ImageCapture
 * @see https://w3c.github.io/mediacapture-screen-share/
 */

// =============================================================================
// ImageCapture API
// =============================================================================

/**
 * ImageCapture interface for grabbing frames from video tracks.
 * @see https://w3c.github.io/mediacapture-image/#imagecaptureapi
 */
declare class ImageCapture {
  constructor(videoTrack: MediaStreamTrack);

  /** The MediaStreamTrack passed to the constructor */
  readonly track: MediaStreamTrack;

  /**
   * Takes a single exposure using the video capture device and returns an ImageBitmap.
   * This is the primary method for capturing frames for VLM analysis.
   */
  grabFrame(): Promise<ImageBitmap>;

  /**
   * Takes a single exposure and returns a Blob containing the image data.
   * @param photoSettings - Optional settings for the photo
   */
  takePhoto(photoSettings?: PhotoSettings): Promise<Blob>;

  /**
   * Returns the current photo capabilities.
   */
  getPhotoCapabilities(): Promise<PhotoCapabilities>;

  /**
   * Returns the current photo settings.
   */
  getPhotoSettings(): Promise<PhotoSettings>;
}

interface PhotoSettings {
  fillLightMode?: 'auto' | 'off' | 'flash';
  imageHeight?: number;
  imageWidth?: number;
  redEyeReduction?: boolean;
}

interface PhotoCapabilities {
  fillLightMode?: string[];
  imageHeight?: MediaSettingsRange;
  imageWidth?: MediaSettingsRange;
  redEyeReduction?: boolean;
}

interface MediaSettingsRange {
  max: number;
  min: number;
  step: number;
}

// =============================================================================
// Screen Capture Extensions
// =============================================================================

/**
 * Extended MediaTrackConstraints for screen capture.
 * These properties are part of the Screen Capture API but not in standard DOM types.
 */
interface MediaTrackConstraintSet {
  /** Cursor visibility preference */
  cursor?: 'always' | 'motion' | 'never';

  /** Preferred display surface type */
  displaySurface?: 'browser' | 'monitor' | 'window';

  /** Whether to capture system audio (Windows-only for screen capture) */
  suppressLocalAudioPlayback?: boolean;

  /** Self-browser surface exclusion */
  selfBrowserSurface?: 'include' | 'exclude';

  /** Surface switching behavior */
  surfaceSwitching?: 'include' | 'exclude';

  /** System audio capture */
  systemAudio?: 'include' | 'exclude';
}

/**
 * Extended MediaTrackSettings for screen capture.
 */
interface MediaTrackSettings {
  /** The type of display surface being captured */
  displaySurface?: 'browser' | 'monitor' | 'window';

  /** Cursor capture mode */
  cursor?: 'always' | 'motion' | 'never';

  /** Whether the captured surface is a logical surface */
  logicalSurface?: boolean;
}

// =============================================================================
// DisplayMediaStreamOptions Extensions
// =============================================================================

/**
 * Extended options for getDisplayMedia with screen capture specific settings.
 */
interface DisplayMediaStreamOptions {
  video?: boolean | (MediaTrackConstraints & {
    cursor?: 'always' | 'motion' | 'never';
    displaySurface?: 'browser' | 'monitor' | 'window';
  });
  audio?: boolean | MediaTrackConstraints;
  selfBrowserSurface?: 'include' | 'exclude';
  surfaceSwitching?: 'include' | 'exclude';
  systemAudio?: 'include' | 'exclude';
  preferCurrentTab?: boolean;
}

// =============================================================================
// RTCRemoteInboundRtpStreamStats (for WebRTC stats)
// =============================================================================

/**
 * Statistics for RTP packets received from a remote peer.
 * Not in all TypeScript DOM libs.
 */
interface RTCRemoteInboundRtpStreamStats extends RTCReceivedRtpStreamStats {
  /** ID of the local outbound RTP stream this is associated with */
  localId?: string;

  /** Round trip time (seconds) */
  roundTripTime?: number;

  /** Total round trip time (seconds) */
  totalRoundTripTime?: number;

  /** Fraction of packets lost */
  fractionLost?: number;

  /** Number of round trip time measurements */
  roundTripTimeMeasurements?: number;
}

// Make ImageCapture available on window
interface Window {
  ImageCapture: typeof ImageCapture;
}
