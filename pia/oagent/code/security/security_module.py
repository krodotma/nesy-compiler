#!/usr/bin/env python3
"""
security_module.py - Security Module (Step 91)

PBTSO Phase: SKILL, VERIFY

Provides:
- Authentication via tokens/API keys
- Role-based access control (RBAC)
- Permission management
- Audit logging
- Session management

Bus Topics:
- code.security.auth
- code.security.access
- code.security.audit
- code.security.session

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Set, Union

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore


# =============================================================================
# Configuration
# =============================================================================

class AuthMethod(Enum):
    """Authentication methods."""
    TOKEN = "token"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    INTERNAL = "internal"


class PermissionAction(Enum):
    """Permission actions."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    DELETE = "delete"
    ADMIN = "admin"


@dataclass
class SecurityConfig:
    """Configuration for security module."""
    enable_auth: bool = True
    enable_audit: bool = True
    token_expiry_s: int = 3600
    max_sessions: int = 100
    rate_limit_requests: int = 1000
    rate_limit_window_s: int = 60
    secret_key: Optional[str] = None
    allowed_origins: List[str] = field(default_factory=list)
    require_encryption: bool = False
    heartbeat_interval_s: int = 300
    heartbeat_timeout_s: int = 900

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enable_auth": self.enable_auth,
            "enable_audit": self.enable_audit,
            "token_expiry_s": self.token_expiry_s,
            "max_sessions": self.max_sessions,
            "rate_limit_requests": self.rate_limit_requests,
        }


# =============================================================================
# Agent Bus with File Locking
# =============================================================================

class LockedAgentBus:
    """Agent bus with file locking for safe concurrent writes."""

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self._ensure_bus_dir()

    def _default_bus_path(self) -> Path:
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _ensure_bus_dir(self) -> None:
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: Dict[str, Any]) -> str:
        """Emit event to bus with file locking."""
        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            **event
        }

        line = json.dumps(full_event, ensure_ascii=False, separators=(",", ":")) + "\n"

        fd = os.open(str(self.bus_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, line.encode("utf-8"))
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

        return event_id


# =============================================================================
# Security Types
# =============================================================================

@dataclass
class Permission:
    """A permission for a resource."""
    resource: str
    action: PermissionAction
    conditions: Dict[str, Any] = field(default_factory=dict)

    def matches(self, resource: str, action: PermissionAction) -> bool:
        """Check if permission matches request."""
        # Wildcard support
        if self.resource == "*":
            resource_match = True
        elif self.resource.endswith("/*"):
            resource_match = resource.startswith(self.resource[:-2])
        else:
            resource_match = self.resource == resource

        action_match = self.action == PermissionAction.ADMIN or self.action == action

        return resource_match and action_match

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resource": self.resource,
            "action": self.action.value,
            "conditions": self.conditions,
        }


@dataclass
class Role:
    """A role with permissions."""
    name: str
    permissions: List[Permission]
    description: str = ""
    parent_roles: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "permissions": [p.to_dict() for p in self.permissions],
            "description": self.description,
            "parent_roles": self.parent_roles,
        }


@dataclass
class Principal:
    """A security principal (user/service)."""
    id: str
    name: str
    roles: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "roles": self.roles,
            "attributes": self.attributes,
            "created_at": self.created_at,
        }


@dataclass
class AuthToken:
    """An authentication token."""
    id: str
    principal_id: str
    method: AuthMethod
    issued_at: float
    expires_at: float
    scope: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "principal_id": self.principal_id,
            "method": self.method.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "scope": self.scope,
            "is_expired": self.is_expired,
        }


@dataclass
class Session:
    """An active session."""
    id: str
    token_id: str
    principal_id: str
    created_at: float
    last_active: float
    ip_address: str = ""
    user_agent: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "token_id": self.token_id,
            "principal_id": self.principal_id,
            "created_at": self.created_at,
            "last_active": self.last_active,
        }


@dataclass
class AuthResult:
    """Result of authentication."""
    success: bool
    principal: Optional[Principal] = None
    token: Optional[AuthToken] = None
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "principal": self.principal.to_dict() if self.principal else None,
            "token": self.token.to_dict() if self.token else None,
            "error": self.error,
        }


@dataclass
class AccessDecision:
    """Result of access check."""
    allowed: bool
    resource: str
    action: PermissionAction
    principal_id: str
    reason: str = ""
    matched_permission: Optional[Permission] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "resource": self.resource,
            "action": self.action.value,
            "principal_id": self.principal_id,
            "reason": self.reason,
        }


@dataclass
class AuditEntry:
    """Audit log entry."""
    id: str
    timestamp: float
    principal_id: str
    action: str
    resource: str
    result: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "principal_id": self.principal_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
        }


# =============================================================================
# Built-in Roles
# =============================================================================

BUILTIN_ROLES = {
    "admin": Role(
        name="admin",
        description="Full administrative access",
        permissions=[
            Permission("*", PermissionAction.ADMIN),
        ],
    ),
    "developer": Role(
        name="developer",
        description="Code read/write access",
        permissions=[
            Permission("code/*", PermissionAction.READ),
            Permission("code/*", PermissionAction.WRITE),
            Permission("code/*", PermissionAction.EXECUTE),
        ],
    ),
    "viewer": Role(
        name="viewer",
        description="Read-only access",
        permissions=[
            Permission("*", PermissionAction.READ),
        ],
    ),
    "service": Role(
        name="service",
        description="Service account access",
        permissions=[
            Permission("api/*", PermissionAction.READ),
            Permission("api/*", PermissionAction.EXECUTE),
        ],
    ),
}


# =============================================================================
# Security Module
# =============================================================================

class SecurityModule:
    """
    Security module for authentication and authorization.

    PBTSO Phase: SKILL, VERIFY

    Features:
    - Token-based authentication
    - Role-based access control (RBAC)
    - Permission management
    - Session management
    - Audit logging

    Usage:
        security = SecurityModule(config)
        auth = security.authenticate(token_string)
        if auth.success:
            access = security.authorize(auth.principal, "code/file.py", PermissionAction.WRITE)
            if access.allowed:
                # proceed with operation
    """

    BUS_TOPICS = {
        "auth": "code.security.auth",
        "access": "code.security.access",
        "audit": "code.security.audit",
        "session": "code.security.session",
    }

    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        bus: Optional[LockedAgentBus] = None,
    ):
        self.config = config or SecurityConfig()
        self.bus = bus or LockedAgentBus()

        # Secret key for token signing
        self._secret_key = (
            self.config.secret_key or
            os.environ.get("CODE_SECRET_KEY") or
            secrets.token_hex(32)
        )

        # Storage
        self._principals: Dict[str, Principal] = {}
        self._roles: Dict[str, Role] = dict(BUILTIN_ROLES)
        self._tokens: Dict[str, AuthToken] = {}
        self._sessions: Dict[str, Session] = {}
        self._api_keys: Dict[str, str] = {}  # key -> principal_id
        self._audit_log: List[AuditEntry] = []

        self._lock = Lock()

        # Create default service principal
        self._create_default_principals()

    def _create_default_principals(self) -> None:
        """Create default principals."""
        # Internal service principal
        self._principals["internal"] = Principal(
            id="internal",
            name="Internal Service",
            roles=["admin"],
            attributes={"type": "service"},
        )

    # =========================================================================
    # Authentication
    # =========================================================================

    def authenticate(
        self,
        credential: str,
        method: AuthMethod = AuthMethod.TOKEN,
    ) -> AuthResult:
        """
        Authenticate a credential.

        Args:
            credential: Token string or API key
            method: Authentication method

        Returns:
            AuthResult with principal and token if successful
        """
        if not self.config.enable_auth:
            # Auth disabled, return internal principal
            return AuthResult(
                success=True,
                principal=self._principals.get("internal"),
            )

        if method == AuthMethod.TOKEN:
            return self._authenticate_token(credential)
        elif method == AuthMethod.API_KEY:
            return self._authenticate_api_key(credential)
        elif method == AuthMethod.INTERNAL:
            return self._authenticate_internal(credential)
        else:
            return AuthResult(success=False, error=f"Unsupported method: {method}")

    def _authenticate_token(self, token_string: str) -> AuthResult:
        """Authenticate via token."""
        # Verify token signature
        try:
            parts = token_string.split(".")
            if len(parts) != 2:
                return AuthResult(success=False, error="Invalid token format")

            token_id, signature = parts
            expected_sig = self._sign(token_id)

            if not hmac.compare_digest(signature, expected_sig):
                return AuthResult(success=False, error="Invalid token signature")

            with self._lock:
                token = self._tokens.get(token_id)
                if not token:
                    return AuthResult(success=False, error="Token not found")

                if token.is_expired:
                    return AuthResult(success=False, error="Token expired")

                principal = self._principals.get(token.principal_id)
                if not principal:
                    return AuthResult(success=False, error="Principal not found")

            self._audit("authenticate", "token", "success", {"principal": principal.id})

            self.bus.emit({
                "topic": self.BUS_TOPICS["auth"],
                "kind": "auth",
                "actor": "security-module",
                "data": {
                    "method": "token",
                    "principal_id": principal.id,
                    "success": True,
                },
            })

            return AuthResult(success=True, principal=principal, token=token)

        except Exception as e:
            self._audit("authenticate", "token", "failure", {"error": str(e)})
            return AuthResult(success=False, error=str(e))

    def _authenticate_api_key(self, api_key: str) -> AuthResult:
        """Authenticate via API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        with self._lock:
            principal_id = self._api_keys.get(key_hash)
            if not principal_id:
                return AuthResult(success=False, error="Invalid API key")

            principal = self._principals.get(principal_id)
            if not principal:
                return AuthResult(success=False, error="Principal not found")

        self._audit("authenticate", "api_key", "success", {"principal": principal.id})
        return AuthResult(success=True, principal=principal)

    def _authenticate_internal(self, agent_id: str) -> AuthResult:
        """Authenticate internal service."""
        # For internal A2A communication
        with self._lock:
            principal = self._principals.get(agent_id)
            if not principal:
                # Create service principal on the fly
                principal = Principal(
                    id=agent_id,
                    name=f"Service: {agent_id}",
                    roles=["service"],
                    attributes={"type": "service"},
                )
                self._principals[agent_id] = principal

        return AuthResult(success=True, principal=principal)

    def _sign(self, data: str) -> str:
        """Sign data with secret key."""
        return hmac.new(
            self._secret_key.encode(),
            data.encode(),
            hashlib.sha256,
        ).hexdigest()[:32]

    # =========================================================================
    # Token Management
    # =========================================================================

    def create_token(
        self,
        principal_id: str,
        scope: Optional[List[str]] = None,
        expiry_s: Optional[int] = None,
    ) -> Optional[AuthToken]:
        """Create a new authentication token."""
        with self._lock:
            if principal_id not in self._principals:
                return None

            token_id = f"tok-{secrets.token_hex(16)}"
            now = time.time()

            token = AuthToken(
                id=token_id,
                principal_id=principal_id,
                method=AuthMethod.TOKEN,
                issued_at=now,
                expires_at=now + (expiry_s or self.config.token_expiry_s),
                scope=scope or [],
            )

            self._tokens[token_id] = token

        self._audit("create_token", f"principal:{principal_id}", "success", {"token_id": token_id})
        return token

    def get_token_string(self, token: AuthToken) -> str:
        """Get signed token string."""
        signature = self._sign(token.id)
        return f"{token.id}.{signature}"

    def revoke_token(self, token_id: str) -> bool:
        """Revoke a token."""
        with self._lock:
            if token_id in self._tokens:
                del self._tokens[token_id]
                self._audit("revoke_token", f"token:{token_id}", "success", {})
                return True
        return False

    # =========================================================================
    # API Key Management
    # =========================================================================

    def create_api_key(self, principal_id: str) -> Optional[str]:
        """Create a new API key for a principal."""
        with self._lock:
            if principal_id not in self._principals:
                return None

            api_key = f"pk_{secrets.token_urlsafe(32)}"
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            self._api_keys[key_hash] = principal_id

        self._audit("create_api_key", f"principal:{principal_id}", "success", {})
        return api_key

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        with self._lock:
            if key_hash in self._api_keys:
                del self._api_keys[key_hash]
                self._audit("revoke_api_key", "api_key", "success", {})
                return True
        return False

    # =========================================================================
    # Authorization
    # =========================================================================

    def authorize(
        self,
        principal: Principal,
        resource: str,
        action: PermissionAction,
    ) -> AccessDecision:
        """
        Check if principal has access to resource.

        Args:
            principal: The security principal
            resource: Resource path/identifier
            action: Action to perform

        Returns:
            AccessDecision with result
        """
        if not self.config.enable_auth:
            return AccessDecision(
                allowed=True,
                resource=resource,
                action=action,
                principal_id=principal.id,
                reason="Auth disabled",
            )

        # Collect all permissions from roles
        permissions = self._get_principal_permissions(principal)

        # Check for matching permission
        for perm in permissions:
            if perm.matches(resource, action):
                decision = AccessDecision(
                    allowed=True,
                    resource=resource,
                    action=action,
                    principal_id=principal.id,
                    reason=f"Granted by permission: {perm.resource}:{perm.action.value}",
                    matched_permission=perm,
                )
                self._emit_access_event(decision)
                return decision

        # No matching permission
        decision = AccessDecision(
            allowed=False,
            resource=resource,
            action=action,
            principal_id=principal.id,
            reason="No matching permission",
        )
        self._emit_access_event(decision)
        return decision

    def _get_principal_permissions(self, principal: Principal) -> List[Permission]:
        """Get all permissions for a principal including inherited."""
        permissions: List[Permission] = []
        visited_roles: Set[str] = set()

        def collect_role_permissions(role_name: str) -> None:
            if role_name in visited_roles:
                return
            visited_roles.add(role_name)

            role = self._roles.get(role_name)
            if not role:
                return

            permissions.extend(role.permissions)

            # Collect parent role permissions
            for parent in role.parent_roles:
                collect_role_permissions(parent)

        for role_name in principal.roles:
            collect_role_permissions(role_name)

        return permissions

    def _emit_access_event(self, decision: AccessDecision) -> None:
        """Emit access decision event."""
        self.bus.emit({
            "topic": self.BUS_TOPICS["access"],
            "kind": "access",
            "actor": "security-module",
            "data": decision.to_dict(),
        })

        self._audit(
            f"access:{decision.action.value}",
            decision.resource,
            "granted" if decision.allowed else "denied",
            {"principal": decision.principal_id},
        )

    # =========================================================================
    # Principal Management
    # =========================================================================

    def create_principal(
        self,
        name: str,
        roles: List[str],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Principal:
        """Create a new principal."""
        principal = Principal(
            id=f"prin-{uuid.uuid4().hex[:12]}",
            name=name,
            roles=roles,
            attributes=attributes or {},
        )

        with self._lock:
            self._principals[principal.id] = principal

        self._audit("create_principal", f"principal:{principal.id}", "success", {"name": name})
        return principal

    def get_principal(self, principal_id: str) -> Optional[Principal]:
        """Get a principal by ID."""
        return self._principals.get(principal_id)

    def update_principal_roles(self, principal_id: str, roles: List[str]) -> bool:
        """Update principal's roles."""
        with self._lock:
            if principal_id not in self._principals:
                return False
            self._principals[principal_id].roles = roles

        self._audit("update_roles", f"principal:{principal_id}", "success", {"roles": roles})
        return True

    def delete_principal(self, principal_id: str) -> bool:
        """Delete a principal."""
        with self._lock:
            if principal_id in self._principals:
                del self._principals[principal_id]
                self._audit("delete_principal", f"principal:{principal_id}", "success", {})
                return True
        return False

    # =========================================================================
    # Role Management
    # =========================================================================

    def create_role(
        self,
        name: str,
        permissions: List[Permission],
        description: str = "",
        parent_roles: Optional[List[str]] = None,
    ) -> Role:
        """Create a new role."""
        role = Role(
            name=name,
            permissions=permissions,
            description=description,
            parent_roles=parent_roles or [],
        )

        with self._lock:
            self._roles[name] = role

        self._audit("create_role", f"role:{name}", "success", {})
        return role

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def delete_role(self, name: str) -> bool:
        """Delete a role."""
        if name in BUILTIN_ROLES:
            return False  # Cannot delete builtin roles

        with self._lock:
            if name in self._roles:
                del self._roles[name]
                self._audit("delete_role", f"role:{name}", "success", {})
                return True
        return False

    # =========================================================================
    # Session Management
    # =========================================================================

    def create_session(
        self,
        token: AuthToken,
        ip_address: str = "",
        user_agent: str = "",
    ) -> Session:
        """Create a new session."""
        session = Session(
            id=f"sess-{uuid.uuid4().hex[:12]}",
            token_id=token.id,
            principal_id=token.principal_id,
            created_at=time.time(),
            last_active=time.time(),
            ip_address=ip_address,
            user_agent=user_agent,
        )

        with self._lock:
            # Enforce max sessions
            if len(self._sessions) >= self.config.max_sessions:
                # Remove oldest session
                oldest = min(self._sessions.values(), key=lambda s: s.last_active)
                del self._sessions[oldest.id]

            self._sessions[session.id] = session

        self.bus.emit({
            "topic": self.BUS_TOPICS["session"],
            "kind": "session",
            "actor": "security-module",
            "data": {
                "event": "created",
                "session_id": session.id,
                "principal_id": session.principal_id,
            },
        })

        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session."""
        return self._sessions.get(session_id)

    def update_session_activity(self, session_id: str) -> bool:
        """Update session last active time."""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id].last_active = time.time()
                return True
        return False

    def end_session(self, session_id: str) -> bool:
        """End a session."""
        with self._lock:
            if session_id in self._sessions:
                session = self._sessions[session_id]
                del self._sessions[session_id]

                self.bus.emit({
                    "topic": self.BUS_TOPICS["session"],
                    "kind": "session",
                    "actor": "security-module",
                    "data": {
                        "event": "ended",
                        "session_id": session_id,
                        "principal_id": session.principal_id,
                    },
                })
                return True
        return False

    def list_sessions(self, principal_id: Optional[str] = None) -> List[Session]:
        """List active sessions."""
        with self._lock:
            sessions = list(self._sessions.values())

        if principal_id:
            sessions = [s for s in sessions if s.principal_id == principal_id]

        return sessions

    # =========================================================================
    # Audit
    # =========================================================================

    def _audit(
        self,
        action: str,
        resource: str,
        result: str,
        details: Dict[str, Any],
    ) -> None:
        """Record audit entry."""
        if not self.config.enable_audit:
            return

        entry = AuditEntry(
            id=f"aud-{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            principal_id=details.get("principal", "system"),
            action=action,
            resource=resource,
            result=result,
            details=details,
        )

        with self._lock:
            self._audit_log.append(entry)
            # Keep last 10000 entries
            if len(self._audit_log) > 10000:
                self._audit_log = self._audit_log[-10000:]

        self.bus.emit({
            "topic": self.BUS_TOPICS["audit"],
            "kind": "audit",
            "actor": "security-module",
            "data": entry.to_dict(),
        })

    def get_audit_log(
        self,
        principal_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Get audit log entries."""
        with self._lock:
            entries = self._audit_log.copy()

        if principal_id:
            entries = [e for e in entries if e.principal_id == principal_id]
        if action:
            entries = [e for e in entries if action in e.action]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    # =========================================================================
    # Utilities
    # =========================================================================

    def require_permission(
        self,
        resource: str,
        action: PermissionAction,
    ) -> Callable:
        """Decorator to require permission for a function."""
        def decorator(func: Callable) -> Callable:
            async def wrapper(principal: Principal, *args: Any, **kwargs: Any) -> Any:
                decision = self.authorize(principal, resource, action)
                if not decision.allowed:
                    raise PermissionError(f"Access denied: {decision.reason}")
                return await func(principal, *args, **kwargs)
            return wrapper
        return decorator

    def stats(self) -> Dict[str, Any]:
        """Get security module statistics."""
        return {
            "principals": len(self._principals),
            "roles": len(self._roles),
            "tokens": len(self._tokens),
            "sessions": len(self._sessions),
            "api_keys": len(self._api_keys),
            "audit_entries": len(self._audit_log),
            "config": self.config.to_dict(),
        }


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """CLI entry point for Security Module."""
    import argparse

    parser = argparse.ArgumentParser(description="Security Module (Step 91)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-principal command
    cp_parser = subparsers.add_parser("create-principal", help="Create a principal")
    cp_parser.add_argument("name", help="Principal name")
    cp_parser.add_argument("--roles", "-r", nargs="+", default=["viewer"])

    # create-token command
    ct_parser = subparsers.add_parser("create-token", help="Create a token")
    ct_parser.add_argument("principal_id", help="Principal ID")
    ct_parser.add_argument("--expiry", "-e", type=int, default=3600)

    # check-access command
    ca_parser = subparsers.add_parser("check-access", help="Check access")
    ca_parser.add_argument("principal_id", help="Principal ID")
    ca_parser.add_argument("resource", help="Resource path")
    ca_parser.add_argument("action", help="Action (read/write/execute/delete/admin)")

    # list-principals command
    subparsers.add_parser("list-principals", help="List principals")

    # list-roles command
    subparsers.add_parser("list-roles", help="List roles")

    # audit command
    audit_parser = subparsers.add_parser("audit", help="Show audit log")
    audit_parser.add_argument("--limit", "-n", type=int, default=20)
    audit_parser.add_argument("--json", action="store_true")

    # stats command
    subparsers.add_parser("stats", help="Show statistics")

    args = parser.parse_args()
    security = SecurityModule()

    if args.command == "create-principal":
        principal = security.create_principal(args.name, args.roles)
        print(f"Created principal: {principal.id}")
        print(f"  Name: {principal.name}")
        print(f"  Roles: {', '.join(principal.roles)}")
        return 0

    elif args.command == "create-token":
        principal = security.get_principal(args.principal_id)
        if not principal:
            print(f"Principal not found: {args.principal_id}")
            return 1

        token = security.create_token(args.principal_id, expiry_s=args.expiry)
        if token:
            token_string = security.get_token_string(token)
            print(f"Token: {token_string}")
            print(f"  Expires: {datetime.fromtimestamp(token.expires_at, tz=timezone.utc).isoformat()}")
        return 0

    elif args.command == "check-access":
        principal = security.get_principal(args.principal_id)
        if not principal:
            print(f"Principal not found: {args.principal_id}")
            return 1

        try:
            action = PermissionAction(args.action)
        except ValueError:
            print(f"Invalid action: {args.action}")
            return 1

        decision = security.authorize(principal, args.resource, action)
        print(f"Access: {'ALLOWED' if decision.allowed else 'DENIED'}")
        print(f"  Reason: {decision.reason}")
        return 0 if decision.allowed else 1

    elif args.command == "list-principals":
        for pid, principal in security._principals.items():
            print(f"{pid}: {principal.name} [{', '.join(principal.roles)}]")
        return 0

    elif args.command == "list-roles":
        for name, role in security._roles.items():
            print(f"{name}: {role.description or '(no description)'}")
            for perm in role.permissions:
                print(f"  - {perm.resource}: {perm.action.value}")
        return 0

    elif args.command == "audit":
        entries = security.get_audit_log(limit=args.limit)
        if args.json:
            print(json.dumps([e.to_dict() for e in entries], indent=2))
        else:
            for entry in entries:
                ts = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{ts}] {entry.action} {entry.resource}: {entry.result}")
        return 0

    elif args.command == "stats":
        stats = security.stats()
        print(json.dumps(stats, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
