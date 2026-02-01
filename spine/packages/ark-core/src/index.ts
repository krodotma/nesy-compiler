/**
 * @ark/core - Core types for Neo Pluribus
 *
 * This package provides the foundational types for the Neo Pluribus system:
 * - Holon: The lock-and-key mechanism for justified actions
 * - Pentad: The 5 positive coordinates (WHY/WHERE/WHO/WHEN/WHAT)
 * - Sextet: The 6 negative constraints (P/E/L/R/Q/Omega gates)
 * - Ring: Security hierarchy (Ring 0-3)
 * - Result: Rust-inspired error handling
 * - Types: Core entity types (Task, Agent, etc.)
 * - Events: Event bus abstractions
 * - Identity: Authentication and authorization
 * - Invariants: Conservation constants (PHI, CMP, etc.)
 * - Utils: Common utility functions
 *
 * @module
 * @example
 * ```typescript
 * import {
 *   Holon,
 *   Pentad,
 *   Sextet,
 *   Ring,
 *   createHolon,
 *   validateHolon,
 *   ok, err, Result,
 *   PHI, CONSERVATION_INVARIANTS,
 * } from '@ark/core';
 *
 * // Create a holon for a file modification
 * const holon = createHolon(
 *   'Fix authentication bug in login handler',
 *   '/src/auth/login.ts',
 *   'claude-opus-4.5',
 *   { type: 'code-change', diff: '...' }
 * );
 *
 * // Validate before proceeding
 * const result = validateHolon(holon);
 * if (result.valid) {
 *   // Proceed with action
 * }
 * ```
 */

// Re-export everything from submodules
export * from './pentad.js';
export * from './sextet.js';
export * from './holon.js';
export * from './ring.js';
export * from './result.js';
export * from './types.js';
export * from './events.js';
export * from './identity.js';
export * from './invariants.js';
export * from './utils.js';

// Version and metadata
export const VERSION = '0.1.0';
export const PROTOCOL_VERSION = 'DKIN v29';
