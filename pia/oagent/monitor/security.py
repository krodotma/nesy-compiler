#!/usr/bin/env python3
"""
Monitor Security Module - Step 291

Authentication and authorization for the Monitor Agent.

PBTSO Phase: SEQUESTER

Bus Topics:
- monitor.security.auth.request (subscribed)
- monitor.security.auth.success (emitted)
- monitor.security.auth.failure (emitted)
- monitor.security.authz.check (emitted)
- monitor.security.authz.denied (emitted)

Protocol: DKIN v30, PAIP v16, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class AuthMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    AGENT_IDENTITY = "agent_identity"
    NONE = "none"


class Permission(Enum):
    """Monitor permissions."""
    METRICS_READ = "metrics:read"
    METRICS_WRITE = "metrics:write"
    ALERTS_READ = "alerts:read"
    ALERTS_WRITE = "alerts:write"
    ALERTS_ACKNOWLEDGE = "alerts:acknowledge"
    DASHBOARDS_READ = "dashboards:read"
    DASHBOARDS_WRITE = "dashboards:write"
    REPORTS_READ = "reports:read"
    REPORTS_WRITE = "reports:write"
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    ADMIN = "admin"


class RingLevel(Enum):
    """Security ring levels."""
    RING_0 = 0  # Kernel/Constitutional - Full access
    RING_1 = 1  # System agents - Monitor, Deploy
    RING_2 = 2  # Privileged agents - Code, Test
    RING_3 = 3  # User agents - Research, Review


@dataclass
class AuthToken:
    """Authentication token.

    Attributes:
        token_id: Unique token ID
        subject: Token subject (user/agent ID)
        issued_at: Token issue time
        expires_at: Token expiration time
        permissions: Granted permissions
        ring_level: Security ring level
        metadata: Additional metadata
    """
    token_id: str
    subject: str
    issued_at: float
    expires_at: float
    permissions: Set[Permission]
    ring_level: RingLevel
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at

    def has_permission(self, permission: Permission) -> bool:
        """Check if token has permission."""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "subject": self.subject,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "permissions": [p.value for p in self.permissions],
            "ring_level": self.ring_level.value,
            "expired": self.is_expired(),
        }


@dataclass
class AuthPrincipal:
    """Authenticated principal.

    Attributes:
        principal_id: Unique principal ID
        name: Principal name
        auth_method: Authentication method used
        ring_level: Security ring level
        permissions: Granted permissions
        api_key_hash: Hash of API key (if used)
        created_at: Creation timestamp
        last_auth: Last authentication timestamp
    """
    principal_id: str
    name: str
    auth_method: AuthMethod
    ring_level: RingLevel
    permissions: Set[Permission]
    api_key_hash: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    last_auth: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "principal_id": self.principal_id,
            "name": self.name,
            "auth_method": self.auth_method.value,
            "ring_level": self.ring_level.value,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at,
            "last_auth": self.last_auth,
        }


@dataclass
class AuthResult:
    """Authentication result.

    Attributes:
        success: Whether authentication succeeded
        principal: Authenticated principal
        token: Auth token (if generated)
        error: Error message (if failed)
        duration_ms: Auth duration
    """
    success: bool
    principal: Optional[AuthPrincipal] = None
    token: Optional[AuthToken] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "principal": self.principal.to_dict() if self.principal else None,
            "token": self.token.to_dict() if self.token else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AuthzResult:
    """Authorization result.

    Attributes:
        allowed: Whether action is allowed
        permission: Requested permission
        reason: Reason for decision
        checked_at: Check timestamp
    """
    allowed: bool
    permission: Permission
    reason: str = ""
    checked_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "permission": self.permission.value,
            "reason": self.reason,
            "checked_at": self.checked_at,
        }


class MonitorSecurityModule:
    """
    Security module for the Monitor Agent.

    Provides:
    - API key authentication
    - Token-based authentication
    - Agent identity verification
    - Permission-based authorization
    - Ring-level access control

    Example:
        security = MonitorSecurityModule()

        # Create principal
        principal, api_key = security.create_principal(
            name="monitor-client",
            permissions={Permission.METRICS_READ},
        )

        # Authenticate
        result = await security.authenticate(api_key=api_key)

        # Authorize
        authz = security.authorize(
            result.principal,
            Permission.METRICS_READ,
        )
    """

    BUS_TOPICS = {
        "auth_request": "monitor.security.auth.request",
        "auth_success": "monitor.security.auth.success",
        "auth_failure": "monitor.security.auth.failure",
        "authz_check": "monitor.security.authz.check",
        "authz_denied": "monitor.security.authz.denied",
    }

    # A2A heartbeat settings
    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    # Default token lifetime (1 hour)
    DEFAULT_TOKEN_LIFETIME = 3600

    # Ring level permissions
    RING_PERMISSIONS = {
        RingLevel.RING_0: set(Permission),
        RingLevel.RING_1: {
            Permission.METRICS_READ, Permission.METRICS_WRITE,
            Permission.ALERTS_READ, Permission.ALERTS_WRITE, Permission.ALERTS_ACKNOWLEDGE,
            Permission.DASHBOARDS_READ, Permission.DASHBOARDS_WRITE,
            Permission.REPORTS_READ, Permission.REPORTS_WRITE,
            Permission.CONFIG_READ, Permission.CONFIG_WRITE,
        },
        RingLevel.RING_2: {
            Permission.METRICS_READ, Permission.METRICS_WRITE,
            Permission.ALERTS_READ, Permission.ALERTS_ACKNOWLEDGE,
            Permission.DASHBOARDS_READ,
            Permission.REPORTS_READ,
            Permission.CONFIG_READ,
        },
        RingLevel.RING_3: {
            Permission.METRICS_READ,
            Permission.ALERTS_READ,
            Permission.DASHBOARDS_READ,
            Permission.REPORTS_READ,
        },
    }

    def __init__(
        self,
        token_lifetime: int = DEFAULT_TOKEN_LIFETIME,
        enable_audit: bool = True,
        bus_dir: Optional[str] = None,
    ):
        """Initialize security module.

        Args:
            token_lifetime: Token lifetime in seconds
            enable_audit: Enable audit logging
            bus_dir: Bus directory
        """
        self._token_lifetime = token_lifetime
        self._enable_audit = enable_audit
        self._last_heartbeat = time.time()

        # Storage
        self._principals: Dict[str, AuthPrincipal] = {}
        self._tokens: Dict[str, AuthToken] = {}
        self._api_key_index: Dict[str, str] = {}  # hash -> principal_id
        self._revoked_tokens: Set[str] = set()
        self._lock = threading.RLock()

        # Audit log
        self._audit_log: List[Dict[str, Any]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

        # Create default system principal
        self._create_system_principal()

    def create_principal(
        self,
        name: str,
        permissions: Optional[Set[Permission]] = None,
        ring_level: RingLevel = RingLevel.RING_3,
        auth_method: AuthMethod = AuthMethod.API_KEY,
    ) -> tuple[AuthPrincipal, Optional[str]]:
        """Create a new principal.

        Args:
            name: Principal name
            permissions: Granted permissions
            ring_level: Security ring level
            auth_method: Authentication method

        Returns:
            Tuple of (principal, api_key if applicable)
        """
        principal_id = f"principal-{uuid.uuid4().hex[:12]}"

        # Filter permissions by ring level
        max_permissions = self.RING_PERMISSIONS.get(ring_level, set())
        if permissions:
            effective_permissions = permissions & max_permissions
        else:
            effective_permissions = max_permissions.copy()

        api_key = None
        api_key_hash = None

        if auth_method == AuthMethod.API_KEY:
            api_key = f"mk_{secrets.token_urlsafe(32)}"
            api_key_hash = self._hash_api_key(api_key)

        principal = AuthPrincipal(
            principal_id=principal_id,
            name=name,
            auth_method=auth_method,
            ring_level=ring_level,
            permissions=effective_permissions,
            api_key_hash=api_key_hash,
        )

        with self._lock:
            self._principals[principal_id] = principal
            if api_key_hash:
                self._api_key_index[api_key_hash] = principal_id

        self._audit("principal_created", {"principal_id": principal_id, "name": name})

        return principal, api_key

    def delete_principal(self, principal_id: str) -> bool:
        """Delete a principal.

        Args:
            principal_id: Principal to delete

        Returns:
            True if deleted
        """
        with self._lock:
            if principal_id not in self._principals:
                return False

            principal = self._principals[principal_id]

            # Remove API key index
            if principal.api_key_hash:
                self._api_key_index.pop(principal.api_key_hash, None)

            # Revoke all tokens
            for token_id, token in list(self._tokens.items()):
                if token.subject == principal_id:
                    self._revoked_tokens.add(token_id)
                    del self._tokens[token_id]

            del self._principals[principal_id]

        self._audit("principal_deleted", {"principal_id": principal_id})
        return True

    async def authenticate(
        self,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        agent_id: Optional[str] = None,
        generate_token: bool = True,
    ) -> AuthResult:
        """Authenticate a request.

        Args:
            api_key: API key
            token: Existing token
            agent_id: Agent identity
            generate_token: Generate new token on success

        Returns:
            Authentication result
        """
        start_time = time.time()

        self._emit_bus_event(
            self.BUS_TOPICS["auth_request"],
            {"method": "api_key" if api_key else "token" if token else "agent"},
        )

        # Token authentication
        if token:
            result = self._authenticate_token(token)
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        # API key authentication
        if api_key:
            result = await self._authenticate_api_key(api_key, generate_token)
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        # Agent identity authentication
        if agent_id:
            result = self._authenticate_agent(agent_id, generate_token)
            result.duration_ms = (time.time() - start_time) * 1000
            return result

        result = AuthResult(
            success=False,
            error="No credentials provided",
        )
        result.duration_ms = (time.time() - start_time) * 1000

        self._emit_bus_event(
            self.BUS_TOPICS["auth_failure"],
            {"error": result.error},
            level="warning",
        )

        return result

    def authorize(
        self,
        principal: AuthPrincipal,
        permission: Permission,
        resource: Optional[str] = None,
    ) -> AuthzResult:
        """Authorize an action.

        Args:
            principal: Authenticated principal
            permission: Required permission
            resource: Target resource

        Returns:
            Authorization result
        """
        self._emit_bus_event(
            self.BUS_TOPICS["authz_check"],
            {
                "principal_id": principal.principal_id,
                "permission": permission.value,
                "resource": resource,
            },
        )

        # Check ring level permissions
        ring_permissions = self.RING_PERMISSIONS.get(principal.ring_level, set())
        if permission not in ring_permissions:
            result = AuthzResult(
                allowed=False,
                permission=permission,
                reason=f"Permission not allowed for ring level {principal.ring_level.value}",
            )
            self._emit_bus_event(
                self.BUS_TOPICS["authz_denied"],
                result.to_dict(),
                level="warning",
            )
            return result

        # Check principal permissions
        if principal.has_permission(permission):
            return AuthzResult(
                allowed=True,
                permission=permission,
                reason="Permission granted",
            )

        result = AuthzResult(
            allowed=False,
            permission=permission,
            reason="Principal does not have required permission",
        )

        self._emit_bus_event(
            self.BUS_TOPICS["authz_denied"],
            result.to_dict(),
            level="warning",
        )

        return result

    def issue_token(
        self,
        principal: AuthPrincipal,
        permissions: Optional[Set[Permission]] = None,
        lifetime: Optional[int] = None,
    ) -> AuthToken:
        """Issue a new token.

        Args:
            principal: Principal to issue token for
            permissions: Token permissions (subset of principal permissions)
            lifetime: Token lifetime

        Returns:
            New token
        """
        token_id = f"mtk_{secrets.token_urlsafe(32)}"
        now = time.time()

        # Filter permissions
        if permissions:
            token_permissions = permissions & principal.permissions
        else:
            token_permissions = principal.permissions.copy()

        token = AuthToken(
            token_id=token_id,
            subject=principal.principal_id,
            issued_at=now,
            expires_at=now + (lifetime or self._token_lifetime),
            permissions=token_permissions,
            ring_level=principal.ring_level,
            metadata={"principal_name": principal.name},
        )

        with self._lock:
            self._tokens[token_id] = token

        self._audit("token_issued", {
            "token_id": token_id,
            "principal_id": principal.principal_id,
        })

        return token

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token.

        Args:
            token_id: Token to revoke

        Returns:
            True if revoked
        """
        with self._lock:
            if token_id in self._tokens:
                del self._tokens[token_id]
            self._revoked_tokens.add(token_id)

        self._audit("token_revoked", {"token_id": token_id})
        return True

    def validate_token(self, token_id: str) -> Optional[AuthToken]:
        """Validate a token.

        Args:
            token_id: Token to validate

        Returns:
            Token if valid, None otherwise
        """
        with self._lock:
            if token_id in self._revoked_tokens:
                return None

            token = self._tokens.get(token_id)
            if not token:
                return None

            if token.is_expired():
                del self._tokens[token_id]
                return None

            return token

    def rotate_api_key(self, principal_id: str) -> Optional[str]:
        """Rotate a principal's API key.

        Args:
            principal_id: Principal ID

        Returns:
            New API key or None
        """
        with self._lock:
            principal = self._principals.get(principal_id)
            if not principal or principal.auth_method != AuthMethod.API_KEY:
                return None

            # Remove old key
            if principal.api_key_hash:
                self._api_key_index.pop(principal.api_key_hash, None)

            # Generate new key
            new_key = f"mk_{secrets.token_urlsafe(32)}"
            new_hash = self._hash_api_key(new_key)

            principal.api_key_hash = new_hash
            self._api_key_index[new_hash] = principal_id

        self._audit("api_key_rotated", {"principal_id": principal_id})
        return new_key

    def get_principal(self, principal_id: str) -> Optional[AuthPrincipal]:
        """Get a principal by ID.

        Args:
            principal_id: Principal ID

        Returns:
            Principal or None
        """
        with self._lock:
            return self._principals.get(principal_id)

    def list_principals(self) -> List[Dict[str, Any]]:
        """List all principals.

        Returns:
            List of principal info
        """
        with self._lock:
            return [p.to_dict() for p in self._principals.values()]

    def get_audit_log(
        self,
        limit: int = 100,
        action_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get audit log entries.

        Args:
            limit: Maximum entries
            action_filter: Filter by action

        Returns:
            Audit log entries
        """
        entries = self._audit_log
        if action_filter:
            entries = [e for e in entries if e.get("action") == action_filter]
        return list(reversed(entries[-limit:]))

    def cleanup_expired(self) -> int:
        """Clean up expired tokens.

        Returns:
            Number of tokens cleaned
        """
        cleaned = 0
        with self._lock:
            for token_id in list(self._tokens.keys()):
                token = self._tokens[token_id]
                if token.is_expired():
                    del self._tokens[token_id]
                    cleaned += 1
        return cleaned

    def get_statistics(self) -> Dict[str, Any]:
        """Get security statistics.

        Returns:
            Statistics
        """
        with self._lock:
            return {
                "principals_count": len(self._principals),
                "active_tokens": len(self._tokens),
                "revoked_tokens": len(self._revoked_tokens),
                "audit_entries": len(self._audit_log),
                "by_ring_level": {
                    level.name: sum(
                        1 for p in self._principals.values()
                        if p.ring_level == level
                    )
                    for level in RingLevel
                },
            }

    def _authenticate_token(self, token_id: str) -> AuthResult:
        """Authenticate using token."""
        token = self.validate_token(token_id)
        if not token:
            self._emit_bus_event(
                self.BUS_TOPICS["auth_failure"],
                {"error": "Invalid or expired token"},
                level="warning",
            )
            return AuthResult(success=False, error="Invalid or expired token")

        with self._lock:
            principal = self._principals.get(token.subject)
            if not principal:
                return AuthResult(success=False, error="Principal not found")

            principal.last_auth = time.time()

        self._emit_bus_event(
            self.BUS_TOPICS["auth_success"],
            {"principal_id": principal.principal_id, "method": "token"},
        )

        return AuthResult(success=True, principal=principal, token=token)

    async def _authenticate_api_key(
        self,
        api_key: str,
        generate_token: bool,
    ) -> AuthResult:
        """Authenticate using API key."""
        key_hash = self._hash_api_key(api_key)

        with self._lock:
            principal_id = self._api_key_index.get(key_hash)
            if not principal_id:
                self._emit_bus_event(
                    self.BUS_TOPICS["auth_failure"],
                    {"error": "Invalid API key"},
                    level="warning",
                )
                return AuthResult(success=False, error="Invalid API key")

            principal = self._principals.get(principal_id)
            if not principal:
                return AuthResult(success=False, error="Principal not found")

            principal.last_auth = time.time()

        token = None
        if generate_token:
            token = self.issue_token(principal)

        self._emit_bus_event(
            self.BUS_TOPICS["auth_success"],
            {"principal_id": principal.principal_id, "method": "api_key"},
        )

        return AuthResult(success=True, principal=principal, token=token)

    def _authenticate_agent(
        self,
        agent_id: str,
        generate_token: bool,
    ) -> AuthResult:
        """Authenticate using agent identity."""
        # Look up or create agent principal
        with self._lock:
            for principal in self._principals.values():
                if principal.name == agent_id and principal.auth_method == AuthMethod.AGENT_IDENTITY:
                    principal.last_auth = time.time()

                    token = None
                    if generate_token:
                        token = self.issue_token(principal)

                    self._emit_bus_event(
                        self.BUS_TOPICS["auth_success"],
                        {"principal_id": principal.principal_id, "method": "agent_identity"},
                    )

                    return AuthResult(success=True, principal=principal, token=token)

        self._emit_bus_event(
            self.BUS_TOPICS["auth_failure"],
            {"error": "Unknown agent"},
            level="warning",
        )

        return AuthResult(success=False, error="Unknown agent")

    def _create_system_principal(self) -> None:
        """Create default system principal."""
        principal = AuthPrincipal(
            principal_id="principal-system",
            name="monitor-agent",
            auth_method=AuthMethod.AGENT_IDENTITY,
            ring_level=RingLevel.RING_1,
            permissions=self.RING_PERMISSIONS[RingLevel.RING_1].copy(),
        )
        self._principals[principal.principal_id] = principal

    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        """Add audit log entry."""
        if not self._enable_audit:
            return

        entry = {
            "timestamp": time.time(),
            "action": action,
            "details": details,
        }

        self._audit_log.append(entry)
        if len(self._audit_log) > 10000:
            self._audit_log = self._audit_log[-10000:]

    def emit_heartbeat(self) -> bool:
        """Emit heartbeat for A2A protocol.

        Returns:
            True if heartbeat was emitted
        """
        now = time.time()
        if now - self._last_heartbeat < self.HEARTBEAT_INTERVAL - 30:
            return False

        self._last_heartbeat = now

        self._emit_bus_event(
            "a2a.heartbeat",
            {
                "component": "monitor_security",
                "status": "healthy",
                "principals": len(self._principals),
                "active_tokens": len(self._tokens),
            },
        )

        return True

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event",
    ) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-security",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass

        return event_id


# Singleton instance
_security: Optional[MonitorSecurityModule] = None


def get_security() -> MonitorSecurityModule:
    """Get or create the security module singleton.

    Returns:
        MonitorSecurityModule instance
    """
    global _security
    if _security is None:
        _security = MonitorSecurityModule()
    return _security


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Monitor Security Module (Step 291)")
    parser.add_argument("--create-principal", metavar="NAME", help="Create a principal")
    parser.add_argument("--ring", type=int, default=3, help="Ring level (0-3)")
    parser.add_argument("--list-principals", action="store_true", help="List all principals")
    parser.add_argument("--authenticate", metavar="API_KEY", help="Authenticate with API key")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--audit", action="store_true", help="Show audit log")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    security = get_security()

    if args.create_principal:
        ring = RingLevel(args.ring)
        principal, api_key = security.create_principal(
            name=args.create_principal,
            ring_level=ring,
        )
        if args.json:
            print(json.dumps({
                "principal": principal.to_dict(),
                "api_key": api_key,
            }, indent=2))
        else:
            print(f"Created principal: {principal.principal_id}")
            print(f"  Name: {principal.name}")
            print(f"  Ring: {principal.ring_level.value}")
            if api_key:
                print(f"  API Key: {api_key}")

    if args.list_principals:
        principals = security.list_principals()
        if args.json:
            print(json.dumps(principals, indent=2))
        else:
            print("Principals:")
            for p in principals:
                print(f"  {p['principal_id']}: {p['name']} (Ring {p['ring_level']})")

    if args.authenticate:
        async def auth():
            return await security.authenticate(api_key=args.authenticate)
        result = asyncio.run(auth())
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "success" if result.success else "failed"
            print(f"Authentication: {status}")
            if result.principal:
                print(f"  Principal: {result.principal.name}")
            if result.token:
                print(f"  Token: {result.token.token_id}")
            if result.error:
                print(f"  Error: {result.error}")

    if args.stats:
        stats = security.get_statistics()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Security Statistics:")
            for k, v in stats.items():
                print(f"  {k}: {v}")

    if args.audit:
        audit = security.get_audit_log(limit=20)
        if args.json:
            print(json.dumps(audit, indent=2))
        else:
            print("Audit Log:")
            for entry in audit:
                print(f"  [{entry['action']}] {entry['details']}")
