/**
 * @ark/core/identity - Identity and Authentication Tests
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import {
  createAnonymousPrincipal,
  createAgentPrincipal,
  createServicePrincipal,
  hasCapability,
  hasRole,
  hasRingAccess,
  isPrincipalExpired,
  isSessionExpired,
  isTokenExpired,
  generateFingerprint,
  authorize,
  type Principal,
  type Session,
  type AuthToken,
  type Role,
  type Permission,
} from './identity.js';
import { Ring } from './ring.js';

describe('Identity', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date('2025-01-28T12:00:00Z'));
  });

  describe('createAnonymousPrincipal()', () => {
    it('should create anonymous principal with correct defaults', () => {
      const principal = createAnonymousPrincipal();

      expect(principal.type).toBe('anonymous');
      expect(principal.name).toBe('anonymous');
      expect(principal.ring).toBe(Ring.Ephemeral);
      expect(principal.roles).toEqual([]);
      expect(principal.capabilities).toEqual([]);
      expect(principal.id).toMatch(/^[0-9a-f-]{36}$/);
      expect(principal.createdAt).toBe(Date.now());
    });
  });

  describe('createAgentPrincipal()', () => {
    it('should create agent principal with provided values', () => {
      const principal = createAgentPrincipal(
        'test-agent',
        'coder',
        ['file.read', 'file.write']
      );

      expect(principal.type).toBe('agent');
      expect(principal.name).toBe('test-agent');
      expect(principal.ring).toBe(Ring.User);
      expect(principal.roles).toEqual(['coder']);
      expect(principal.capabilities).toEqual(['file.read', 'file.write']);
    });

    it('should accept custom ring', () => {
      const principal = createAgentPrincipal(
        'service-agent',
        'service',
        ['*'],
        Ring.Service
      );

      expect(principal.ring).toBe(Ring.Service);
    });

    it('should accept parent ID', () => {
      const principal = createAgentPrincipal(
        'child-agent',
        'worker',
        [],
        Ring.User,
        'parent-agent-id'
      );

      expect(principal.parent).toBe('parent-agent-id');
    });
  });

  describe('createServicePrincipal()', () => {
    it('should create service principal', () => {
      const principal = createServicePrincipal(
        'mcp-server',
        ['bus.publish', 'bus.subscribe']
      );

      expect(principal.type).toBe('service');
      expect(principal.name).toBe('mcp-server');
      expect(principal.ring).toBe(Ring.Service);
      expect(principal.roles).toEqual(['service']);
      expect(principal.capabilities).toEqual(['bus.publish', 'bus.subscribe']);
    });
  });

  describe('hasCapability()', () => {
    it('should return true for exact capability match', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: [],
        capabilities: ['file.read', 'file.write'],
        createdAt: Date.now(),
      };

      expect(hasCapability(principal, 'file.read')).toBe(true);
      expect(hasCapability(principal, 'file.write')).toBe(true);
      expect(hasCapability(principal, 'file.delete')).toBe(false);
    });

    it('should return true for wildcard capability', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: [],
        capabilities: ['*'],
        createdAt: Date.now(),
      };

      expect(hasCapability(principal, 'anything')).toBe(true);
      expect(hasCapability(principal, 'file.read')).toBe(true);
    });

    it('should support prefix wildcards', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: [],
        capabilities: ['file.*'],
        createdAt: Date.now(),
      };

      expect(hasCapability(principal, 'file.read')).toBe(true);
      expect(hasCapability(principal, 'file.write')).toBe(true);
      expect(hasCapability(principal, 'network.connect')).toBe(false);
    });
  });

  describe('hasRole()', () => {
    it('should check if principal has role', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: ['admin', 'developer'],
        capabilities: [],
        createdAt: Date.now(),
      };

      expect(hasRole(principal, 'admin')).toBe(true);
      expect(hasRole(principal, 'developer')).toBe(true);
      expect(hasRole(principal, 'viewer')).toBe(false);
    });
  });

  describe('hasRingAccess()', () => {
    it('should allow access to same or higher ring numbers', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User, // Ring 2
        roles: [],
        capabilities: [],
        createdAt: Date.now(),
      };

      expect(hasRingAccess(principal, Ring.User)).toBe(true);
      expect(hasRingAccess(principal, Ring.Ephemeral)).toBe(true);
      expect(hasRingAccess(principal, Ring.Service)).toBe(false);
      expect(hasRingAccess(principal, Ring.Kernel)).toBe(false);
    });

    it('should allow kernel to access all rings', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'system',
        name: 'kernel',
        ring: Ring.Kernel,
        roles: [],
        capabilities: [],
        createdAt: Date.now(),
      };

      expect(hasRingAccess(principal, Ring.Kernel)).toBe(true);
      expect(hasRingAccess(principal, Ring.Service)).toBe(true);
      expect(hasRingAccess(principal, Ring.User)).toBe(true);
      expect(hasRingAccess(principal, Ring.Ephemeral)).toBe(true);
    });
  });

  describe('isPrincipalExpired()', () => {
    it('should return false if no expiration', () => {
      const principal = createAnonymousPrincipal();
      expect(isPrincipalExpired(principal)).toBe(false);
    });

    it('should return false if not expired', () => {
      const principal: Principal = {
        ...createAnonymousPrincipal(),
        expiresAt: Date.now() + 60000,
      };
      expect(isPrincipalExpired(principal)).toBe(false);
    });

    it('should return true if expired', () => {
      const principal: Principal = {
        ...createAnonymousPrincipal(),
        expiresAt: Date.now() - 1000,
      };
      expect(isPrincipalExpired(principal)).toBe(true);
    });
  });

  describe('isSessionExpired()', () => {
    it('should return true for expired sessions', () => {
      const session: Session = {
        id: 's1',
        principal: createAnonymousPrincipal(),
        startedAt: Date.now() - 7200000,
        lastActiveAt: Date.now() - 3600000,
        expiresAt: Date.now() - 1000,
      };
      expect(isSessionExpired(session)).toBe(true);
    });

    it('should return false for active sessions', () => {
      const session: Session = {
        id: 's1',
        principal: createAnonymousPrincipal(),
        startedAt: Date.now(),
        lastActiveAt: Date.now(),
        expiresAt: Date.now() + 3600000,
      };
      expect(isSessionExpired(session)).toBe(false);
    });
  });

  describe('isTokenExpired()', () => {
    it('should check token expiration', () => {
      const validToken: AuthToken = {
        id: 't1',
        principal: createAnonymousPrincipal(),
        token: 'abc123',
        tokenType: 'Bearer',
        issuedAt: Date.now(),
        expiresAt: Date.now() + 3600000,
        scopes: [],
      };
      expect(isTokenExpired(validToken)).toBe(false);

      const expiredToken: AuthToken = {
        ...validToken,
        expiresAt: Date.now() - 1000,
      };
      expect(isTokenExpired(expiredToken)).toBe(true);
    });
  });

  describe('generateFingerprint()', () => {
    it('should generate consistent fingerprints', () => {
      const fp1 = generateFingerprint('test-data');
      const fp2 = generateFingerprint('test-data');
      expect(fp1).toBe(fp2);
    });

    it('should generate different fingerprints for different data', () => {
      const fp1 = generateFingerprint('data-1');
      const fp2 = generateFingerprint('data-2');
      expect(fp1).not.toBe(fp2);
    });

    it('should return 8 character hex string', () => {
      const fp = generateFingerprint('test');
      expect(fp).toMatch(/^[0-9a-f]{8}$/);
    });
  });

  describe('authorize()', () => {
    const roles: Role[] = [
      {
        name: 'reader',
        description: 'Read-only access',
        permissions: [
          { name: 'read-files', resource: '/files/*', actions: ['read'] },
        ],
      },
      {
        name: 'writer',
        description: 'Read-write access',
        permissions: [
          { name: 'all-files', resource: '/files/*', actions: ['read', 'write', 'delete'] },
        ],
      },
      {
        name: 'admin',
        description: 'Full access',
        permissions: [
          { name: 'everything', resource: '*', actions: ['*'] },
        ],
      },
    ];

    it('should allow authorized actions', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: ['reader'],
        capabilities: [],
        createdAt: Date.now(),
      };

      const result = authorize({
        principal,
        resource: '/files/test.txt',
        action: 'read',
      }, roles);

      expect(result.decision).toBe('allow');
      expect(result.matchedPermissions).toHaveLength(1);
    });

    it('should deny unauthorized actions', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: ['reader'],
        capabilities: [],
        createdAt: Date.now(),
      };

      const result = authorize({
        principal,
        resource: '/files/test.txt',
        action: 'delete',
      }, roles);

      expect(result.decision).toBe('deny');
      expect(result.reason).toBe('No matching permissions');
    });

    it('should deny expired principals', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: ['admin'],
        capabilities: [],
        createdAt: Date.now() - 7200000,
        expiresAt: Date.now() - 1000,
      };

      const result = authorize({
        principal,
        resource: '/anything',
        action: 'any',
      }, roles);

      expect(result.decision).toBe('deny');
      expect(result.reason).toBe('Principal expired');
    });

    it('should handle wildcard resource patterns', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'admin',
        ring: Ring.User,
        roles: ['admin'],
        capabilities: [],
        createdAt: Date.now(),
      };

      const result = authorize({
        principal,
        resource: '/any/path/here',
        action: 'anything',
      }, roles);

      expect(result.decision).toBe('allow');
    });

    it('should handle principals with no matching roles', () => {
      const principal: Principal = {
        id: 'p1',
        type: 'agent',
        name: 'test',
        ring: Ring.User,
        roles: ['unknown-role'],
        capabilities: [],
        createdAt: Date.now(),
      };

      const result = authorize({
        principal,
        resource: '/files/test.txt',
        action: 'read',
      }, roles);

      expect(result.decision).toBe('deny');
    });
  });

  vi.useRealTimers();
});
