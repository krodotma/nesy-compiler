#!/usr/bin/env python3
"""
security_module.py - Security Module (Step 41)

Authentication, authorization, and security features for Research Agent.
Supports JWT tokens, API keys, RBAC, and audit logging.

PBTSO Phase: PROTECT

Bus Topics:
- a2a.research.security.auth
- a2a.research.security.authz
- a2a.research.security.audit
- research.security.token.issue
- research.security.token.revoke

Protocol: DKIN v30, PAIP v16, CITIZEN v2
"""
from __future__ import annotations

import base64
import fcntl
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from ..bootstrap import AgentBus


# ============================================================================
# Configuration
# ============================================================================


class AuthMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    BASIC = "basic"
    BEARER = "bearer"
    CERTIFICATE = "certificate"


class Permission(Enum):
    """Research agent permissions."""
    # Read permissions
    READ_INDEX = "read:index"
    READ_SEARCH = "read:search"
    READ_ANALYSIS = "read:analysis"
    READ_DOCS = "read:docs"
    READ_METRICS = "read:metrics"

    # Write permissions
    WRITE_INDEX = "write:index"
    WRITE_CONFIG = "write:config"
    WRITE_PLUGIN = "write:plugin"

    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_CONFIG = "admin:config"
    ADMIN_SYSTEM = "admin:system"

    # All permissions
    ALL = "*"


class Role(Enum):
    """Predefined roles."""
    ANONYMOUS = "anonymous"
    READER = "reader"
    WRITER = "writer"
    ADMIN = "admin"
    SYSTEM = "system"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ANONYMOUS: {Permission.READ_SEARCH, Permission.READ_DOCS},
    Role.READER: {
        Permission.READ_INDEX, Permission.READ_SEARCH,
        Permission.READ_ANALYSIS, Permission.READ_DOCS, Permission.READ_METRICS
    },
    Role.WRITER: {
        Permission.READ_INDEX, Permission.READ_SEARCH,
        Permission.READ_ANALYSIS, Permission.READ_DOCS, Permission.READ_METRICS,
        Permission.WRITE_INDEX, Permission.WRITE_CONFIG
    },
    Role.ADMIN: {
        Permission.READ_INDEX, Permission.READ_SEARCH,
        Permission.READ_ANALYSIS, Permission.READ_DOCS, Permission.READ_METRICS,
        Permission.WRITE_INDEX, Permission.WRITE_CONFIG, Permission.WRITE_PLUGIN,
        Permission.ADMIN_USERS, Permission.ADMIN_CONFIG
    },
    Role.SYSTEM: {Permission.ALL},
}


@dataclass
class SecurityConfig:
    """Configuration for security module."""

    enable_auth: bool = True
    enable_authz: bool = True
    enable_audit: bool = True
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiry_seconds: int = 3600
    api_key_prefix: str = "research_"
    api_key_length: int = 32
    max_failed_attempts: int = 5
    lockout_duration_seconds: int = 300
    session_timeout_seconds: int = 1800
    enable_rate_limiting: bool = True
    audit_retention_days: int = 90
    bus_path: Optional[str] = None

    def __post_init__(self):
        if not self.jwt_secret:
            # Generate a default secret (should be overridden in production)
            self.jwt_secret = secrets.token_hex(32)
        if self.bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.bus_path = f"{pluribus_root}/.pluribus/bus/events.ndjson"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class Principal:
    """A security principal (user, service, etc.)."""

    id: str
    name: str
    type: str = "user"  # user, service, system
    roles: List[Role] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def has_permission(self, permission: Permission) -> bool:
        """Check if principal has a permission."""
        if Permission.ALL in self.permissions:
            return True
        if permission in self.permissions:
            return True
        # Check role permissions
        for role in self.roles:
            role_perms = ROLE_PERMISSIONS.get(role, set())
            if Permission.ALL in role_perms or permission in role_perms:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "roles": [r.value for r in self.roles],
            "permissions": [p.value for p in self.permissions],
            "metadata": self.metadata,
        }


@dataclass
class AuthToken:
    """An authentication token."""

    token: str
    principal_id: str
    type: AuthMethod
    issued_at: float
    expires_at: float
    scopes: List[str] = field(default_factory=list)
    revoked: bool = False

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() > self.expires_at

    @property
    def is_valid(self) -> bool:
        """Check if token is valid."""
        return not self.revoked and not self.is_expired

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "principal_id": self.principal_id,
            "type": self.type.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "scopes": self.scopes,
            "revoked": self.revoked,
        }


@dataclass
class AuthResult:
    """Result of an authentication attempt."""

    success: bool
    principal: Optional[Principal] = None
    token: Optional[AuthToken] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "principal": self.principal.to_dict() if self.principal else None,
            "error": self.error,
        }


@dataclass
class AuthzResult:
    """Result of an authorization check."""

    allowed: bool
    principal: Optional[Principal] = None
    permission: Optional[Permission] = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "permission": self.permission.value if self.permission else None,
            "reason": self.reason,
        }


@dataclass
class AuditEntry:
    """An audit log entry."""

    id: str
    timestamp: float
    principal_id: str
    action: str
    resource: str
    result: str  # success, denied, error
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "principal_id": self.principal_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "metadata": self.metadata,
        }


# ============================================================================
# Authentication Providers
# ============================================================================


class AuthProvider(ABC):
    """Abstract base for authentication providers."""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate with credentials."""
        pass

    @abstractmethod
    def validate_token(self, token: str) -> AuthResult:
        """Validate a token."""
        pass


class APIKeyAuthProvider(AuthProvider):
    """API key authentication provider."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._keys: Dict[str, Principal] = {}
        self._lock = threading.Lock()

    def generate_key(self, principal: Principal) -> str:
        """Generate an API key for a principal."""
        key = f"{self.config.api_key_prefix}{secrets.token_hex(self.config.api_key_length)}"
        key_hash = self._hash_key(key)

        with self._lock:
            self._keys[key_hash] = principal

        return key

    def revoke_key(self, key: str) -> bool:
        """Revoke an API key."""
        key_hash = self._hash_key(key)
        with self._lock:
            if key_hash in self._keys:
                del self._keys[key_hash]
                return True
        return False

    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate with API key."""
        api_key = credentials.get("api_key", "")

        if not api_key:
            return AuthResult(success=False, error="Missing API key")

        key_hash = self._hash_key(api_key)

        with self._lock:
            principal = self._keys.get(key_hash)

        if principal:
            return AuthResult(success=True, principal=principal)

        return AuthResult(success=False, error="Invalid API key")

    def validate_token(self, token: str) -> AuthResult:
        """Validate an API key as token."""
        return self.authenticate({"api_key": token})

    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()


class JWTAuthProvider(AuthProvider):
    """JWT authentication provider."""

    def __init__(self, config: SecurityConfig):
        self.config = config
        self._principals: Dict[str, Principal] = {}
        self._revoked_tokens: Set[str] = set()
        self._lock = threading.Lock()

    def issue_token(self, principal: Principal, scopes: Optional[List[str]] = None) -> AuthToken:
        """Issue a JWT token."""
        now = time.time()
        expires_at = now + self.config.jwt_expiry_seconds
        token_id = str(uuid.uuid4())

        # Create JWT payload
        payload = {
            "jti": token_id,
            "sub": principal.id,
            "iat": int(now),
            "exp": int(expires_at),
            "scopes": scopes or [],
            "roles": [r.value for r in principal.roles],
        }

        # Encode token (simplified - use pyjwt in production)
        token = self._encode_jwt(payload)

        with self._lock:
            self._principals[principal.id] = principal

        return AuthToken(
            token=token,
            principal_id=principal.id,
            type=AuthMethod.JWT,
            issued_at=now,
            expires_at=expires_at,
            scopes=scopes or [],
        )

    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token."""
        try:
            payload = self._decode_jwt(token)
            token_id = payload.get("jti")
            if token_id:
                with self._lock:
                    self._revoked_tokens.add(token_id)
                return True
        except Exception:
            pass
        return False

    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate with JWT."""
        token = credentials.get("token", "")
        return self.validate_token(token)

    def validate_token(self, token: str) -> AuthResult:
        """Validate a JWT token."""
        try:
            payload = self._decode_jwt(token)

            # Check expiration
            exp = payload.get("exp", 0)
            if time.time() > exp:
                return AuthResult(success=False, error="Token expired")

            # Check revocation
            token_id = payload.get("jti")
            with self._lock:
                if token_id in self._revoked_tokens:
                    return AuthResult(success=False, error="Token revoked")

            # Get principal
            principal_id = payload.get("sub")
            with self._lock:
                principal = self._principals.get(principal_id)

            if not principal:
                # Reconstruct principal from token
                principal = Principal(
                    id=principal_id,
                    name=principal_id,
                    roles=[Role(r) for r in payload.get("roles", [])],
                )

            return AuthResult(success=True, principal=principal)

        except Exception as e:
            return AuthResult(success=False, error=f"Invalid token: {e}")

    def _encode_jwt(self, payload: Dict[str, Any]) -> str:
        """Encode a JWT token (simplified implementation)."""
        header = {"alg": self.config.jwt_algorithm, "typ": "JWT"}

        header_b64 = base64.urlsafe_b64encode(
            json.dumps(header).encode()
        ).rstrip(b"=").decode()

        payload_b64 = base64.urlsafe_b64encode(
            json.dumps(payload).encode()
        ).rstrip(b"=").decode()

        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.config.jwt_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        signature_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

        return f"{message}.{signature_b64}"

    def _decode_jwt(self, token: str) -> Dict[str, Any]:
        """Decode a JWT token (simplified implementation)."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header_b64, payload_b64, signature_b64 = parts

        # Verify signature
        message = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            self.config.jwt_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).digest()
        expected_sig_b64 = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode()

        if not hmac.compare_digest(signature_b64, expected_sig_b64):
            raise ValueError("Invalid signature")

        # Decode payload
        padding = 4 - (len(payload_b64) % 4)
        if padding != 4:
            payload_b64 += "=" * padding

        payload = json.loads(base64.urlsafe_b64decode(payload_b64))
        return payload


# ============================================================================
# Security Manager
# ============================================================================


class SecurityManager:
    """
    Comprehensive security manager for Research Agent.

    Features:
    - Multiple authentication methods
    - Role-based access control (RBAC)
    - Audit logging
    - Session management
    - Rate limiting

    PBTSO Phase: PROTECT

    Example:
        security = SecurityManager()

        # Authenticate
        result = security.authenticate({
            "method": "api_key",
            "api_key": "research_abc123..."
        })

        # Check authorization
        if security.authorize(result.principal, Permission.WRITE_INDEX):
            # Perform operation
            pass

        # Audit
        security.audit(
            principal=result.principal,
            action="index_update",
            resource="codebase/main",
        )
    """

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the security manager.

        Args:
            config: Security configuration
            bus: AgentBus for event emission
        """
        self.config = config or SecurityConfig()
        self.bus = bus or AgentBus()

        # Auth providers
        self._api_key_provider = APIKeyAuthProvider(self.config)
        self._jwt_provider = JWTAuthProvider(self.config)

        # Failed attempt tracking
        self._failed_attempts: Dict[str, List[float]] = {}
        self._locked_principals: Dict[str, float] = {}

        # Sessions
        self._sessions: Dict[str, Dict[str, Any]] = {}

        # Audit log (in-memory, should use persistent storage in production)
        self._audit_log: List[AuditEntry] = []

        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "auth_attempts": 0,
            "auth_successes": 0,
            "auth_failures": 0,
            "authz_checks": 0,
            "authz_allowed": 0,
            "authz_denied": 0,
            "audit_entries": 0,
        }

    def authenticate(self, credentials: Dict[str, Any]) -> AuthResult:
        """
        Authenticate with provided credentials.

        Args:
            credentials: Authentication credentials

        Returns:
            AuthResult with authentication outcome
        """
        self._stats["auth_attempts"] += 1

        method = credentials.get("method", AuthMethod.API_KEY.value)
        identifier = credentials.get("identifier", credentials.get("api_key", "unknown"))

        # Check lockout
        if self._is_locked_out(identifier):
            self._stats["auth_failures"] += 1
            self._emit_event("a2a.research.security.auth", {
                "success": False,
                "method": method,
                "error": "Account locked",
            }, level="warning")
            return AuthResult(success=False, error="Account temporarily locked")

        # Authenticate based on method
        if method == AuthMethod.API_KEY.value:
            result = self._api_key_provider.authenticate(credentials)
        elif method == AuthMethod.JWT.value:
            result = self._jwt_provider.authenticate(credentials)
        elif method == AuthMethod.BEARER.value:
            token = credentials.get("token", "")
            result = self._jwt_provider.validate_token(token)
        else:
            result = AuthResult(success=False, error=f"Unsupported auth method: {method}")

        # Track failures
        if not result.success:
            self._record_failed_attempt(identifier)
            self._stats["auth_failures"] += 1
        else:
            self._clear_failed_attempts(identifier)
            self._stats["auth_successes"] += 1

        # Emit event
        self._emit_event("a2a.research.security.auth", {
            "success": result.success,
            "method": method,
            "principal_id": result.principal.id if result.principal else None,
            "error": result.error,
        }, level="info" if result.success else "warning")

        return result

    def authorize(
        self,
        principal: Optional[Principal],
        permission: Permission,
        resource: Optional[str] = None,
    ) -> AuthzResult:
        """
        Check if principal is authorized for an operation.

        Args:
            principal: The principal requesting access
            permission: Required permission
            resource: Optional resource identifier

        Returns:
            AuthzResult with authorization outcome
        """
        self._stats["authz_checks"] += 1

        if not self.config.enable_authz:
            self._stats["authz_allowed"] += 1
            return AuthzResult(allowed=True, principal=principal, permission=permission)

        if not principal:
            self._stats["authz_denied"] += 1
            result = AuthzResult(
                allowed=False,
                permission=permission,
                reason="No principal"
            )
            self._emit_authz_event(result, resource)
            return result

        allowed = principal.has_permission(permission)

        if allowed:
            self._stats["authz_allowed"] += 1
        else:
            self._stats["authz_denied"] += 1

        result = AuthzResult(
            allowed=allowed,
            principal=principal,
            permission=permission,
            reason=None if allowed else "Permission denied"
        )

        self._emit_authz_event(result, resource)
        return result

    def audit(
        self,
        principal: Optional[Principal],
        action: str,
        resource: str,
        result: str = "success",
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Record an audit log entry.

        Args:
            principal: Principal performing action
            action: Action being performed
            resource: Resource being accessed
            result: Result of the action
            ip_address: Client IP address
            user_agent: Client user agent
            metadata: Additional metadata

        Returns:
            The audit entry
        """
        if not self.config.enable_audit:
            return AuditEntry(
                id="disabled",
                timestamp=time.time(),
                principal_id=principal.id if principal else "anonymous",
                action=action,
                resource=resource,
                result=result,
            )

        entry = AuditEntry(
            id=str(uuid.uuid4()),
            timestamp=time.time(),
            principal_id=principal.id if principal else "anonymous",
            action=action,
            resource=resource,
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata or {},
        )

        with self._lock:
            self._audit_log.append(entry)
            self._stats["audit_entries"] += 1

        # Emit audit event
        self._emit_event("a2a.research.security.audit", entry.to_dict())

        return entry

    def create_api_key(self, principal: Principal) -> str:
        """Create an API key for a principal."""
        key = self._api_key_provider.generate_key(principal)

        self._emit_event("research.security.token.issue", {
            "type": "api_key",
            "principal_id": principal.id,
        })

        return key

    def revoke_api_key(self, key: str) -> bool:
        """Revoke an API key."""
        success = self._api_key_provider.revoke_key(key)

        if success:
            self._emit_event("research.security.token.revoke", {
                "type": "api_key",
            })

        return success

    def issue_jwt(
        self,
        principal: Principal,
        scopes: Optional[List[str]] = None,
    ) -> AuthToken:
        """Issue a JWT token for a principal."""
        token = self._jwt_provider.issue_token(principal, scopes)

        self._emit_event("research.security.token.issue", {
            "type": "jwt",
            "principal_id": principal.id,
            "expires_at": token.expires_at,
        })

        return token

    def revoke_jwt(self, token: str) -> bool:
        """Revoke a JWT token."""
        success = self._jwt_provider.revoke_token(token)

        if success:
            self._emit_event("research.security.token.revoke", {
                "type": "jwt",
            })

        return success

    def create_session(self, principal: Principal) -> str:
        """Create a session for a principal."""
        session_id = secrets.token_urlsafe(32)

        with self._lock:
            self._sessions[session_id] = {
                "principal": principal,
                "created_at": time.time(),
                "last_activity": time.time(),
            }

        return session_id

    def validate_session(self, session_id: str) -> Optional[Principal]:
        """Validate a session and return the principal."""
        with self._lock:
            session = self._sessions.get(session_id)

            if not session:
                return None

            # Check timeout
            if time.time() - session["last_activity"] > self.config.session_timeout_seconds:
                del self._sessions[session_id]
                return None

            # Update activity
            session["last_activity"] = time.time()
            return session["principal"]

    def destroy_session(self, session_id: str) -> bool:
        """Destroy a session."""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
        return False

    def get_audit_log(
        self,
        principal_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit log entries."""
        with self._lock:
            entries = self._audit_log.copy()

        if principal_id:
            entries = [e for e in entries if e.principal_id == principal_id]

        if action:
            entries = [e for e in entries if e.action == action]

        return entries[-limit:]

    def require_auth(
        self,
        permission: Optional[Permission] = None,
    ) -> Callable:
        """
        Decorator requiring authentication.

        Args:
            permission: Optional required permission

        Example:
            @security.require_auth(Permission.WRITE_INDEX)
            def update_index(principal: Principal, data: Dict):
                ...
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(principal: Optional[Principal] = None, *args, **kwargs):
                if not principal:
                    raise PermissionError("Authentication required")

                if permission:
                    result = self.authorize(principal, permission)
                    if not result.allowed:
                        raise PermissionError(f"Permission denied: {permission.value}")

                return func(principal, *args, **kwargs)

            return wrapper
        return decorator

    def get_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        return {
            **self._stats,
            "active_sessions": len(self._sessions),
            "locked_accounts": len(self._locked_principals),
        }

    def _is_locked_out(self, identifier: str) -> bool:
        """Check if identifier is locked out."""
        with self._lock:
            if identifier in self._locked_principals:
                locked_until = self._locked_principals[identifier]
                if time.time() < locked_until:
                    return True
                del self._locked_principals[identifier]
        return False

    def _record_failed_attempt(self, identifier: str) -> None:
        """Record a failed authentication attempt."""
        with self._lock:
            now = time.time()

            if identifier not in self._failed_attempts:
                self._failed_attempts[identifier] = []

            # Clean old attempts
            window_start = now - self.config.lockout_duration_seconds
            self._failed_attempts[identifier] = [
                ts for ts in self._failed_attempts[identifier]
                if ts > window_start
            ]

            self._failed_attempts[identifier].append(now)

            # Check for lockout
            if len(self._failed_attempts[identifier]) >= self.config.max_failed_attempts:
                self._locked_principals[identifier] = now + self.config.lockout_duration_seconds

    def _clear_failed_attempts(self, identifier: str) -> None:
        """Clear failed attempts for identifier."""
        with self._lock:
            if identifier in self._failed_attempts:
                del self._failed_attempts[identifier]

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "security",
            "level": level,
            "actor": "research-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _emit_authz_event(self, result: AuthzResult, resource: Optional[str]) -> None:
        """Emit authorization event."""
        self._emit_event("a2a.research.security.authz", {
            "allowed": result.allowed,
            "principal_id": result.principal.id if result.principal else None,
            "permission": result.permission.value if result.permission else None,
            "resource": resource,
            "reason": result.reason,
        }, level="info" if result.allowed else "warning")


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Security Module."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Security Module (Step 41)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate API key
    gen_parser = subparsers.add_parser("generate-key", help="Generate API key")
    gen_parser.add_argument("--name", required=True, help="Principal name")
    gen_parser.add_argument("--role", default="reader", help="Role (reader, writer, admin)")

    # Issue JWT
    jwt_parser = subparsers.add_parser("issue-jwt", help="Issue JWT token")
    jwt_parser.add_argument("--name", required=True, help="Principal name")
    jwt_parser.add_argument("--role", default="reader", help="Role")
    jwt_parser.add_argument("--expiry", type=int, default=3600, help="Expiry in seconds")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run security demo")

    args = parser.parse_args()

    security = SecurityManager()

    if args.command == "generate-key":
        role = Role(args.role)
        principal = Principal(
            id=str(uuid.uuid4())[:8],
            name=args.name,
            roles=[role],
        )
        key = security.create_api_key(principal)
        print(f"API Key: {key}")
        print(f"Principal ID: {principal.id}")
        print(f"Role: {role.value}")

    elif args.command == "issue-jwt":
        role = Role(args.role)
        principal = Principal(
            id=str(uuid.uuid4())[:8],
            name=args.name,
            roles=[role],
        )

        config = SecurityConfig(jwt_expiry_seconds=args.expiry)
        security = SecurityManager(config)

        token = security.issue_jwt(principal)
        print(f"JWT Token: {token.token}")
        print(f"Principal ID: {principal.id}")
        print(f"Expires: {datetime.fromtimestamp(token.expires_at)}")

    elif args.command == "stats":
        stats = security.get_stats()
        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Security Statistics:")
            print(f"  Auth Attempts: {stats['auth_attempts']}")
            print(f"  Auth Successes: {stats['auth_successes']}")
            print(f"  Auth Failures: {stats['auth_failures']}")
            print(f"  Authz Checks: {stats['authz_checks']}")
            print(f"  Authz Allowed: {stats['authz_allowed']}")
            print(f"  Authz Denied: {stats['authz_denied']}")

    elif args.command == "demo":
        print("Running security demo...\n")

        # Create a principal
        principal = Principal(
            id="user-123",
            name="Demo User",
            roles=[Role.WRITER],
        )
        print(f"Created principal: {principal.name} with role {Role.WRITER.value}")

        # Generate API key
        key = security.create_api_key(principal)
        print(f"Generated API key: {key[:20]}...")

        # Authenticate
        result = security.authenticate({
            "method": "api_key",
            "api_key": key,
        })
        print(f"Authentication: {'SUCCESS' if result.success else 'FAILED'}")

        # Check permissions
        for perm in [Permission.READ_SEARCH, Permission.WRITE_INDEX, Permission.ADMIN_SYSTEM]:
            authz = security.authorize(result.principal, perm)
            status = "ALLOWED" if authz.allowed else "DENIED"
            print(f"Permission {perm.value}: {status}")

        # Audit
        security.audit(
            principal=result.principal,
            action="demo_action",
            resource="demo/resource",
            result="success",
        )
        print("\nAudit entry recorded")

        print("\nDemo complete.")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
