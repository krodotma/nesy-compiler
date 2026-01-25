/**
 * IPE Context Schema
 *
 * Type definitions for the In-Place Editor context capture system.
 * Re-exports types from context-capture.ts for external consumption.
 */

export type {
  ElementType,
  UniformValue,
  ShaderContext,
  IPEContext,
  DistortionReport,
} from './context-capture';

// ============================================================================
// Bus Event Types
// ============================================================================

import type { IPEContext } from './context-capture';

/** IPE mode states */
export type IPEMode = 'off' | 'inspect' | 'edit';

/** Edit scope - global affects theme.ts, instance affects single element */
export type IPEScope = 'global' | 'instance';

/** Sextet validation verdict */
export type SextetVerdict = 'PASSED' | 'WARNED' | 'FAILED';

/** IPE mode change event */
export interface IPEModeChangeEvent {
  topic: 'ipe.mode.change';
  data: {
    mode: IPEMode;
    actor: string;
  };
}

/** IPE element selection event */
export interface IPEElementSelectEvent {
  topic: 'ipe.element.select';
  data: {
    context: IPEContext;
    position: { x: number; y: number };
  };
}

/** IPE token edit event */
export interface IPETokenEditEvent {
  topic: 'ipe.token.edit';
  data: {
    scope: IPEScope;
    instanceId?: string;
    token: string;
    oldValue: string;
    newValue: string;
  };
}

/** IPE shader edit event */
export interface IPEShaderEditEvent {
  topic: 'ipe.shader.edit';
  data: {
    shaderId: string;
    field: 'fragment' | 'vertex' | 'uniform';
    uniformName?: string;
    oldValue: string | number | number[];
    newValue: string | number | number[];
  };
}

/** IPE change record */
export interface IPEChange {
  type: 'token' | 'shader' | 'style';
  target: string;
  before: unknown;
  after: unknown;
}

/** IPE save event */
export interface IPESaveEvent {
  topic: 'ipe.save';
  data: {
    verdict: SextetVerdict;
    scope: IPEScope;
    changes: IPEChange[];
    sextetVector: [number, number, number, number, number, number];
  };
}

/** Union of all IPE events */
export type IPEBusEvent =
  | IPEModeChangeEvent
  | IPEElementSelectEvent
  | IPETokenEditEvent
  | IPEShaderEditEvent
  | IPESaveEvent;

// ============================================================================
// Persistence Types
// ============================================================================

/** Instance style override stored in localStorage */
export interface IPEInstanceOverride {
  instanceId: string;
  styles: Record<string, string>;
  shaderOverride?: string;
  purpose?: string;
  updatedAt: string;
}

/** IPE localStorage schema */
export interface IPELocalStorage {
  version: number;
  instances: Record<string, IPEInstanceOverride>;
  undoStack: IPEContext[];
  preferences: IPEPreferences;
}

/** IPE panel preferences */
export interface IPEPreferences {
  panelPosition: { x: number; y: number };
  panelSize: { width: number; height: number };
  activeTab: 'tokens' | 'artdept' | 'raw';
  togglePosition: { x: number; y: number };
}

// ============================================================================
// Validation Types
// ============================================================================

/** Sextet gate result */
export interface SextetGateResult {
  gate: 'P' | 'E' | 'L' | 'R' | 'Q' | 'Ω';
  passed: boolean;
  message?: string;
}

/** Full Sextet validation result */
export interface SextetValidationResult {
  verdict: SextetVerdict;
  gates: SextetGateResult[];
  vector: [number, number, number, number, number, number];
}

// ============================================================================
// Default Values
// ============================================================================

export const DEFAULT_IPE_PREFERENCES: IPEPreferences = {
  panelPosition: { x: -1, y: -1 }, // -1 means auto-position
  panelSize: { width: 380, height: 520 },
  activeTab: 'tokens',
  togglePosition: { x: -1, y: -1 }, // -1 means bottom-right default
};

export const DEFAULT_IPE_STORAGE: IPELocalStorage = {
  version: 1,
  instances: {},
  undoStack: [],
  preferences: DEFAULT_IPE_PREFERENCES,
};

// ============================================================================
// Allowed CSS Properties (E-Gate)
// ============================================================================

export const ALLOWED_CSS_PROPERTIES = [
  'color', 'background-color', 'border-color', 'fill', 'stroke',
  'font-size', 'font-weight', 'line-height', 'letter-spacing',
  'padding', 'margin', 'border-radius', 'box-shadow',
  'opacity', 'filter', 'transform',
] as const;

export type AllowedCSSProperty = typeof ALLOWED_CSS_PROPERTIES[number];
