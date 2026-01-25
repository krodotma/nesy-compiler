/**
 * Actions Module - Unified action system for server interactions
 *
 * Provides notebook-style output cells and uniform action handling
 * throughout the dashboard.
 */

// Types
export type {
  ActionStatus,
  ActionRequest,
  ActionOutput,
  ActionResult,
  ActionCell,
  ServiceActionType,
  ServiceActionPayload,
  SystemActionType,
  CommandActionPayload,
} from './types';

// Service Actions (non-JSX, safe for lib build)
export {
  createServiceActionHandlers,
  createMockBusEmitter,
  type ServiceActionHandler,
  type BusEmitter,
} from './serviceActions';

// Note: Component exports (OutputCell, ActionWrapper, ActionProvider)
// should be imported directly from their respective .tsx files
// when used in Qwik components. They are not exported here to
// avoid JSX compilation issues in the lib build.
