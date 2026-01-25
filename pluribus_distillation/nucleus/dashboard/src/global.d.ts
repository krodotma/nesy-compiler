/**
 * Global Type Declarations for Pluribus Dashboard
 *
 * Extends standard DOM types with browser APIs that exist but aren't
 * in TypeScript's lib.dom.d.ts yet.
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
  readonly track: MediaStreamTrack;
  grabFrame(): Promise<ImageBitmap>;
  takePhoto(photoSettings?: PhotoSettings): Promise<Blob>;
  getPhotoCapabilities(): Promise<PhotoCapabilities>;
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

// =============================================================================
// Screen Capture Extensions
// =============================================================================

interface MediaTrackConstraintSet {
  cursor?: ConstrainDOMString;
  displaySurface?: ConstrainDOMString;
  suppressLocalAudioPlayback?: ConstrainBoolean;
  selfBrowserSurface?: ConstrainDOMString;
  surfaceSwitching?: ConstrainDOMString;
  systemAudio?: ConstrainDOMString;
}

interface MediaTrackSettings {
  displaySurface?: string;
  cursor?: string;
  logicalSurface?: boolean;
}

// =============================================================================
// RTCRemoteInboundRtpStreamStats
// =============================================================================

interface RTCRemoteInboundRtpStreamStats extends RTCReceivedRtpStreamStats {
  localId?: string;
  roundTripTime?: number;
  totalRoundTripTime?: number;
  fractionLost?: number;
  roundTripTimeMeasurements?: number;
}

// =============================================================================
// Window Extensions
// =============================================================================

interface Window {
  ImageCapture: typeof ImageCapture;
  __PLURIBUS_BUS__?: {
    emit: (event: {
      topic: string;
      kind: string;
      level: string;
      data: Record<string, unknown>;
    }) => void;
  };
}
