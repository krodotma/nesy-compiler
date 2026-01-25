/**
 * Bus Module Exports
 *
 * Browser-safe exports only. Node-specific functionality (startBridge)
 * is in bus-bridge.ts and should be imported directly when needed on server.
 */

export * from './bus-client';

// Note: startBridge is intentionally NOT exported here to avoid pulling
// Node.js-only dependencies (path, fs) into browser bundles.
// Import directly from './bus-bridge' when running on Node.js.
