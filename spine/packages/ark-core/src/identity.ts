/**
 * @ark/core/identity - Identity and Authentication
 *
 * Defines identity types, authentication, and authorization primitives.
 * Based on the DKIN (Distributed Knowledge Identity Network) protocol.
 *
 * @module
 */

import type { Id, Timestamp, Metadata } from './types.js';
import type { Ring } from './ring.js';

/**
 * Identity types in the system
 */
export type IdentityType =
  | 'human'
  | 'agent'
  | 'service'
  | 'system'
  | 'anonymous';

/**
 * Authentication method
 */
export type AuthMethod =
  | 'api_key'
  | 'jwt'
  | 'certificate'
  | 'oauth2'
  | 'passkey'
  | 'internal';

/**
 * Identity principal
 */
export interface Principal {
  /** Unique identity ID */
  id: Id;
  /** Identity type */
  type: IdentityType;
  /** Display name */
  name: string;
  /** Email (for humans) */
  email?: string;
  /** Security ring */
  ring: Ring;
  /** Roles assigned */
  roles: string[];
  /** Capabilities granted */
  capabilities: string[];
  /** Parent principal (for delegation) */
  parent?: Id;
  /** Creation timestamp */
  createdAt: Timestamp;
  /** Expiration timestamp */
  expiresAt?: Timestamp;
  /** Additional metadata */
  metadata?: Metadata;
}

/**
 * Authentication credentials
 */
export interface Credentials {
  /** Principal ID */
  principalId: Id;
  /** Authentication method */
  method: AuthMethod;
  /** Credential data (method-specific) */
  data: Metadata;
  /** When credentials were issued */
  issuedAt: Timestamp;
  /** When credentials expire */
  expiresAt?: Timestamp;
  /** Credential fingerprint for revocation */
  fingerprint: string;
}

/**
 * Authentication token
 */
export interface AuthToken {
  /** Token ID */
  id: Id;
  /** Associated principal */
  principal: Principal;
  /** Token value */
  token: string;
  /** Token type (e.g., 'Bearer') */
  tokenType: string;
  /** When issued */
  issuedAt: Timestamp;
  /** When expires */
  expiresAt: Timestamp;
  /** Refresh token (if applicable) */
  refreshToken?: string;
  /** Scopes granted */
  scopes: string[];
}

/**
 * Session information
 */
export interface Session {
  /** Session ID */
  id: Id;
  /** Associated principal */
  principal: Principal;
  /** Session start time */
  startedAt: Timestamp;
  /** Last activity time */
  lastActiveAt: Timestamp;
  /** Session expiration */
  expiresAt: Timestamp;
  /** IP address (if applicable) */
  ipAddress?: string;
  /** User agent (if applicable) */
  userAgent?: string;
  /** Session metadata */
  metadata?: Metadata;
}

/**
 * Permission definition
 */
export interface Permission {
  /** Permission name */
  name: string;
  /** Resource pattern (glob) */
  resource: string;
  /** Allowed actions */
  actions: string[];
  /** Conditions for permission */
  conditions?: PermissionCondition[];
}

/**
 * Permission condition
 */
export interface PermissionCondition {
  /** Condition type */
  type: 'time' | 'location' | 'rate_limit' | 'custom';
  /** Condition parameters */
  params: Metadata;
}

/**
 * Role definition
 */
export interface Role {
  /** Role name */
  name: string;
  /** Role description */
  description: string;
  /** Permissions granted */
  permissions: Permission[];
  /** Parent roles (for inheritance) */
  inherits?: string[];
  /** Security ring required */
  minRing?: Ring;
}

/**
 * Access decision
 */
export type AccessDecision = 'allow' | 'deny' | 'challenge';

/**
 * Authorization request
 */
export interface AuthzRequest {
  /** Requesting principal */
  principal: Principal;
  /** Resource being accessed */
  resource: string;
  /** Action being performed */
  action: string;
  /** Request context */
  context?: Metadata;
}

/**
 * Authorization response
 */
export interface AuthzResponse {
  /** Access decision */
  decision: AccessDecision;
  /** Reason for decision */
  reason?: string;
  /** Permissions matched */
  matchedPermissions?: Permission[];
  /** Challenge data (if decision is 'challenge') */
  challenge?: Metadata;
}

/**
 * Create an anonymous principal
 */
export function createAnonymousPrincipal(): Principal {
  return {
    id: crypto.randomUUID(),
    type: 'anonymous',
    name: 'anonymous',
    ring: 3 as Ring,
    roles: [],
    capabilities: [],
    createdAt: Date.now(),
  };
}

/**
 * Create an agent principal
 */
export function createAgentPrincipal(
  name: string,
  role: string,
  capabilities: string[],
  ring: Ring = 2 as Ring,
  parent?: Id
): Principal {
  return {
    id: crypto.randomUUID(),
    type: 'agent',
    name,
    ring,
    roles: [role],
    capabilities,
    parent,
    createdAt: Date.now(),
  };
}

/**
 * Create a service principal
 */
export function createServicePrincipal(
  name: string,
  capabilities: string[],
  ring: Ring = 1 as Ring
): Principal {
  return {
    id: crypto.randomUUID(),
    type: 'service',
    name,
    ring,
    roles: ['service'],
    capabilities,
    createdAt: Date.now(),
  };
}

/**
 * Check if principal has capability
 */
export function hasCapability(principal: Principal, capability: string): boolean {
  // Wildcards
  if (principal.capabilities.includes('*')) return true;
  if (principal.capabilities.includes(capability)) return true;

  // Prefix wildcards (e.g., 'file.*' matches 'file.read')
  const prefix = capability.split('.').slice(0, -1).join('.') + '.*';
  return principal.capabilities.includes(prefix);
}

/**
 * Check if principal has role
 */
export function hasRole(principal: Principal, role: string): boolean {
  return principal.roles.includes(role);
}

/**
 * Check if principal has minimum ring access
 */
export function hasRingAccess(principal: Principal, requiredRing: Ring): boolean {
  return principal.ring <= requiredRing;
}

/**
 * Check if principal is expired
 */
export function isPrincipalExpired(principal: Principal): boolean {
  if (!principal.expiresAt) return false;
  return Date.now() > principal.expiresAt;
}

/**
 * Check if session is expired
 */
export function isSessionExpired(session: Session): boolean {
  return Date.now() > session.expiresAt;
}

/**
 * Check if token is expired
 */
export function isTokenExpired(token: AuthToken): boolean {
  return Date.now() > token.expiresAt;
}

/**
 * Generate a secure fingerprint
 */
export function generateFingerprint(data: string): string {
  // Simple hash - in production use proper crypto
  let hash = 0;
  for (let i = 0; i < data.length; i++) {
    const char = data.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return Math.abs(hash).toString(16).padStart(8, '0');
}

/**
 * Simple authorize function
 */
export function authorize(request: AuthzRequest, roles: Role[]): AuthzResponse {
  // Check if principal is expired
  if (isPrincipalExpired(request.principal)) {
    return {
      decision: 'deny',
      reason: 'Principal expired',
    };
  }

  // Get all permissions from principal's roles
  const permissions: Permission[] = [];
  for (const roleName of request.principal.roles) {
    const role = roles.find(r => r.name === roleName);
    if (role) {
      permissions.push(...role.permissions);
    }
  }

  // Check permissions
  const matchedPermissions: Permission[] = [];
  for (const permission of permissions) {
    // Check resource pattern
    if (!matchResource(permission.resource, request.resource)) continue;

    // Check action
    if (!permission.actions.includes(request.action) && !permission.actions.includes('*')) {
      continue;
    }

    matchedPermissions.push(permission);
  }

  if (matchedPermissions.length > 0) {
    return {
      decision: 'allow',
      matchedPermissions,
    };
  }

  return {
    decision: 'deny',
    reason: 'No matching permissions',
  };
}

/**
 * Simple resource pattern matching (glob-like)
 */
function matchResource(pattern: string, resource: string): boolean {
  if (pattern === '*') return true;
  if (pattern === resource) return true;

  // Convert glob to regex
  const regex = new RegExp(
    '^' + pattern
      .replace(/\./g, '\\.')
      .replace(/\*/g, '.*')
      .replace(/\?/g, '.') + '$'
  );

  return regex.test(resource);
}
