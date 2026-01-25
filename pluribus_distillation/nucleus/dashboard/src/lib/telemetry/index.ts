/**
 * Telemetry Module
 *
 * Client-side error and performance telemetry for Pluribus Dashboard.
 * Streams all warnings, errors, and issues to the backend bus for
 * rapid pre-detection and debugging.
 */

export {
  initErrorCollector,
  destroyErrorCollector,
  trackWebSocket,
  reportComponentError,
  logError,
  logWarn,
  logInfo,
  type TelemetryEvent,
  type ErrorSeverity,
  type ErrorCategory,
} from './error-collector';
