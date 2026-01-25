/**
 * Lanes Library - Main Entry Point
 *
 * Phases 7-8 of OITERATE lanes-widget-enhancement
 *
 * Re-exports all lanes modules for convenient importing.
 */

// ============================================================================
// Phase 7: State & Integration
// ============================================================================

// Core store
export * from './store';

// Event bridge (store-bus sync)
export * from './event-bridge';

// IndexedDB persistence
export * from './indexeddb';

// Undo/Redo system
export * from './undo-redo';

// Component bus (inter-component communication)
export * from './component-bus';

// Export utilities
export * from './export';

// Pagination utilities
export * from './pagination';

// Performance utilities
export * from './performance';

// ============================================================================
// Phase 8: Workflows & Automation
// ============================================================================

// Batch operations
export * from './batch-ops';

// Cascade engine for dependencies
export * from './cascade';

// Automation API
export * from './automation-api';
