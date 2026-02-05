#!/usr/bin/env python3
"""
Step 141: Test Security Module

Authentication, authorization, and security controls for the Test Agent.

PBTSO Phase: VERIFY, SEQUESTER
Bus Topics:
- test.security.authenticate (emits)
- test.security.authorize (emits)
- test.security.audit (emits)

Dependencies: Steps 101-140 (Test Components)
"""
from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import hmac
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ============================================================================
# Constants
# ============================================================================

class Permission(Enum):
    """Available permissions."""
    # Test operations
    TEST_READ = "test:read"
    TEST_WRITE = "test:write"
    TEST_RUN = "test:run"
    TEST_DELETE = "test:delete"

    # Coverage operations
    COVERAGE_READ = "coverage:read"
    COVERAGE_WRITE = "coverage:write"

    # Report operations
    REPORT_READ = "report:read"
    REPORT_WRITE = "report:write"
    REPORT_DELETE = "report:delete"

    # Config operations
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"

    # Admin operations
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_SYSTEM = "admin:system"

    # Agent operations
    AGENT_CONTROL = "agent:control"
    AGENT_MONITOR = "agent:monitor"


class Role(Enum):
    """Predefined roles."""
    VIEWER = "viewer"
    DEVELOPER = "developer"
    TESTER = "tester"
    ADMIN = "admin"
    SYSTEM = "system"


class AuthMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    HMAC = "hmac"
    INTERNAL = "internal"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.TEST_READ,
        Permission.COVERAGE_READ,
        Permission.REPORT_READ,
        Permission.CONFIG_READ,
    },
    Role.DEVELOPER: {
        Permission.TEST_READ,
        Permission.TEST_WRITE,
        Permission.TEST_RUN,
        Permission.COVERAGE_READ,
        Permission.REPORT_READ,
        Permission.REPORT_WRITE,
        Permission.CONFIG_READ,
    },
    Role.TESTER: {
        Permission.TEST_READ,
        Permission.TEST_WRITE,
        Permission.TEST_RUN,
        Permission.TEST_DELETE,
        Permission.COVERAGE_READ,
        Permission.COVERAGE_WRITE,
        Permission.REPORT_READ,
        Permission.REPORT_WRITE,
        Permission.CONFIG_READ,
    },
    Role.ADMIN: {p for p in Permission},  # All permissions
    Role.SYSTEM: {p for p in Permission},  # All permissions
}


# ============================================================================
# Data Types
# ============================================================================

@dataclass
class Token:
    """
    Authentication token.

    Attributes:
        token_id: Unique token ID
        subject: Token subject (user/agent ID)
        roles: Assigned roles
        permissions: Explicit permissions
        expires_at: Expiration timestamp
        issued_at: Issuance timestamp
        metadata: Additional metadata
    """
    token_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject: str = ""
    roles: List[Role] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    expires_at: Optional[float] = None
    issued_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    @property
    def all_permissions(self) -> Set[Permission]:
        """Get all permissions (role + explicit)."""
        perms = set(self.permissions)
        for role in self.roles:
            perms.update(ROLE_PERMISSIONS.get(role, set()))
        return perms

    def has_permission(self, permission: Permission) -> bool:
        """Check if token has a permission."""
        return permission in self.all_permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_id": self.token_id,
            "subject": self.subject,
            "roles": [r.value for r in self.roles],
            "permissions": [p.value for p in self.permissions],
            "expires_at": self.expires_at,
            "issued_at": self.issued_at,
            "is_expired": self.is_expired,
        }


@dataclass
class AuthResult:
    """
    Authentication result.

    Attributes:
        authenticated: Whether authentication succeeded
        token: Authentication token if successful
        error: Error message if failed
        method: Authentication method used
        subject: Authenticated subject
    """
    authenticated: bool = False
    token: Optional[Token] = None
    error: Optional[str] = None
    method: Optional[AuthMethod] = None
    subject: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "authenticated": self.authenticated,
            "token": self.token.to_dict() if self.token else None,
            "error": self.error,
            "method": self.method.value if self.method else None,
            "subject": self.subject,
        }


@dataclass
class AuthorizationResult:
    """
    Authorization result.

    Attributes:
        authorized: Whether authorization succeeded
        permission: Permission checked
        reason: Reason for denial
        policies_applied: Policies that were applied
    """
    authorized: bool = False
    permission: Optional[Permission] = None
    reason: Optional[str] = None
    policies_applied: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "authorized": self.authorized,
            "permission": self.permission.value if self.permission else None,
            "reason": self.reason,
            "policies_applied": self.policies_applied,
        }


@dataclass
class SecurityPolicy:
    """
    Security policy definition.

    Attributes:
        name: Policy name
        description: Policy description
        permissions: Required permissions
        conditions: Additional conditions
        priority: Policy priority
        enabled: Whether policy is enabled
    """
    name: str
    description: str = ""
    permissions: List[Permission] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True

    def evaluate(self, token: Token, context: Dict[str, Any]) -> tuple:
        """
        Evaluate policy against token and context.

        Returns:
            (applies, result) - Whether policy applies and if so, the result
        """
        if not self.enabled:
            return False, True

        # Check conditions
        for key, expected in self.conditions.items():
            if context.get(key) != expected:
                return False, True  # Policy doesn't apply

        # Check permissions
        for perm in self.permissions:
            if not token.has_permission(perm):
                return True, False  # Policy applies, denied

        return True, True  # Policy applies, allowed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [p.value for p in self.permissions],
            "conditions": self.conditions,
            "priority": self.priority,
            "enabled": self.enabled,
        }


@dataclass
class AuditEntry:
    """
    Security audit log entry.

    Attributes:
        entry_id: Unique entry ID
        timestamp: Entry timestamp
        event_type: Type of event
        subject: Subject (user/agent)
        action: Action performed
        resource: Resource affected
        result: Action result
        details: Additional details
        ip_address: Client IP if available
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = "security"
    subject: str = ""
    action: str = ""
    resource: str = ""
    result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
            "event_type": self.event_type,
            "subject": self.subject,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
            "ip_address": self.ip_address,
        }


@dataclass
class SecurityConfig:
    """
    Configuration for security.

    Attributes:
        output_dir: Output directory for security data
        token_expiry_s: Token expiration in seconds
        api_key_prefix: Prefix for API keys
        enable_audit: Enable audit logging
        enable_rate_limiting: Enable rate limiting
        max_failed_attempts: Max failed auth attempts
        lockout_duration_s: Lockout duration after max attempts
    """
    output_dir: str = ".pluribus/test-agent/security"
    token_expiry_s: int = 3600
    api_key_prefix: str = "ta_"
    enable_audit: bool = True
    enable_rate_limiting: bool = True
    max_failed_attempts: int = 5
    lockout_duration_s: int = 300
    secret_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "token_expiry_s": self.token_expiry_s,
            "enable_audit": self.enable_audit,
            "enable_rate_limiting": self.enable_rate_limiting,
            "max_failed_attempts": self.max_failed_attempts,
        }


# ============================================================================
# Bus Interface with File Locking
# ============================================================================

class SecurityBus:
    """Bus interface for security with file locking."""

    HEARTBEAT_INTERVAL = 300
    HEARTBEAT_TIMEOUT = 900

    def __init__(self, bus_path: Optional[Path] = None):
        self.bus_path = bus_path or self._default_bus_path()
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        self._last_heartbeat = time.time()

    def _default_bus_path(self) -> Path:
        root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        return root / ".pluribus" / "bus" / "events.ndjson"

    def emit(self, event: Dict[str, Any]) -> None:
        """Emit an event to the bus with file locking."""
        event_with_meta = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "id": str(uuid.uuid4()),
            **event,
        }

        try:
            with open(self.bus_path, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(event_with_meta) + "\n")
                    f.flush()
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def heartbeat(self, agent_id: str) -> None:
        """Send A2A heartbeat."""
        now = time.time()
        if now - self._last_heartbeat >= self.HEARTBEAT_INTERVAL:
            self.emit({
                "topic": "a2a.heartbeat",
                "kind": "heartbeat",
                "actor": agent_id,
                "data": {"status": "alive"},
            })
            self._last_heartbeat = now


# ============================================================================
# Test Security Manager
# ============================================================================

class TestSecurityManager:
    """
    Security management for the Test Agent.

    Features:
    - API key authentication
    - Token-based authorization
    - Role-based access control
    - Security policies
    - Audit logging
    - Circuit breaker for external services

    PBTSO Phase: VERIFY, SEQUESTER
    Bus Topics: test.security.authenticate, test.security.authorize, test.security.audit
    """

    BUS_TOPICS = {
        "authenticate": "test.security.authenticate",
        "authorize": "test.security.authorize",
        "audit": "test.security.audit",
    }

    def __init__(self, bus=None, config: Optional[SecurityConfig] = None):
        """
        Initialize the security manager.

        Args:
            bus: Optional bus instance
            config: Security configuration
        """
        self.bus = bus or SecurityBus()
        self.config = config or SecurityConfig()

        # Initialize secret key
        self._secret_key = (
            self.config.secret_key or
            os.environ.get("TEST_AGENT_SECRET_KEY") or
            secrets.token_hex(32)
        )

        # API keys storage
        self._api_keys: Dict[str, Dict[str, Any]] = {}

        # Active tokens
        self._tokens: Dict[str, Token] = {}

        # Security policies
        self._policies: List[SecurityPolicy] = []

        # Failed attempts tracking
        self._failed_attempts: Dict[str, List[float]] = {}

        # Audit log
        self._audit_log: List[AuditEntry] = []

        # Circuit breaker state
        self._circuit_open = False
        self._circuit_failures = 0
        self._circuit_last_failure: Optional[float] = None

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load stored keys
        self._load_api_keys()

    def generate_api_key(
        self,
        subject: str,
        roles: Optional[List[Role]] = None,
        permissions: Optional[List[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a new API key.

        Args:
            subject: Subject (user/agent ID)
            roles: Assigned roles
            permissions: Explicit permissions
            metadata: Additional metadata

        Returns:
            Generated API key
        """
        key = f"{self.config.api_key_prefix}{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(key)

        self._api_keys[key_hash] = {
            "subject": subject,
            "roles": [r.value for r in (roles or [])],
            "permissions": [p.value for p in (permissions or [])],
            "created_at": time.time(),
            "metadata": metadata or {},
        }

        self._save_api_keys()

        self._audit("api_key_created", subject, "api_key", "success", {
            "roles": [r.value for r in (roles or [])],
        })

        return key

    def revoke_api_key(self, key: str) -> bool:
        """
        Revoke an API key.

        Args:
            key: API key to revoke

        Returns:
            True if revoked
        """
        key_hash = self._hash_key(key)

        if key_hash in self._api_keys:
            subject = self._api_keys[key_hash]["subject"]
            del self._api_keys[key_hash]
            self._save_api_keys()
            self._audit("api_key_revoked", subject, "api_key", "success")
            return True

        return False

    def authenticate(
        self,
        method: AuthMethod,
        credentials: Dict[str, Any],
    ) -> AuthResult:
        """
        Authenticate a request.

        Args:
            method: Authentication method
            credentials: Authentication credentials

        Returns:
            AuthResult with authentication outcome
        """
        # Check rate limiting
        client_id = credentials.get("client_id", "unknown")
        if self._is_locked_out(client_id):
            return AuthResult(
                authenticated=False,
                error="Too many failed attempts. Please try again later.",
                method=method,
            )

        # Check circuit breaker
        if self._circuit_open and method in (AuthMethod.CERTIFICATE,):
            return AuthResult(
                authenticated=False,
                error="External authentication service unavailable",
                method=method,
            )

        result = AuthResult(method=method)

        try:
            if method == AuthMethod.API_KEY:
                result = self._authenticate_api_key(credentials)
            elif method == AuthMethod.TOKEN:
                result = self._authenticate_token(credentials)
            elif method == AuthMethod.HMAC:
                result = self._authenticate_hmac(credentials)
            elif method == AuthMethod.INTERNAL:
                result = self._authenticate_internal(credentials)
            else:
                result.error = f"Unsupported authentication method: {method}"
        except Exception as e:
            result.authenticated = False
            result.error = str(e)

        # Track failed attempts
        if not result.authenticated:
            self._record_failed_attempt(client_id)

        # Emit event and audit
        self._emit_event("authenticate", {
            "method": method.value,
            "subject": result.subject,
            "success": result.authenticated,
        })

        self._audit(
            "authentication",
            result.subject or client_id,
            "session",
            "success" if result.authenticated else "failure",
            {"method": method.value, "error": result.error},
        )

        return result

    def _authenticate_api_key(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via API key."""
        api_key = credentials.get("api_key")
        if not api_key:
            return AuthResult(
                authenticated=False,
                error="Missing API key",
                method=AuthMethod.API_KEY,
            )

        key_hash = self._hash_key(api_key)
        key_data = self._api_keys.get(key_hash)

        if not key_data:
            return AuthResult(
                authenticated=False,
                error="Invalid API key",
                method=AuthMethod.API_KEY,
            )

        # Create token
        token = Token(
            subject=key_data["subject"],
            roles=[Role(r) for r in key_data["roles"]],
            permissions={Permission(p) for p in key_data["permissions"]},
            expires_at=time.time() + self.config.token_expiry_s,
            metadata=key_data.get("metadata", {}),
        )

        self._tokens[token.token_id] = token

        return AuthResult(
            authenticated=True,
            token=token,
            method=AuthMethod.API_KEY,
            subject=key_data["subject"],
        )

    def _authenticate_token(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via token."""
        token_id = credentials.get("token_id")
        if not token_id:
            return AuthResult(
                authenticated=False,
                error="Missing token ID",
                method=AuthMethod.TOKEN,
            )

        token = self._tokens.get(token_id)

        if not token:
            return AuthResult(
                authenticated=False,
                error="Invalid token",
                method=AuthMethod.TOKEN,
            )

        if token.is_expired:
            del self._tokens[token_id]
            return AuthResult(
                authenticated=False,
                error="Token expired",
                method=AuthMethod.TOKEN,
            )

        return AuthResult(
            authenticated=True,
            token=token,
            method=AuthMethod.TOKEN,
            subject=token.subject,
        )

    def _authenticate_hmac(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via HMAC signature."""
        signature = credentials.get("signature")
        payload = credentials.get("payload", "")
        timestamp = credentials.get("timestamp", 0)
        subject = credentials.get("subject", "")

        # Check timestamp freshness (5 minute window)
        if abs(time.time() - timestamp) > 300:
            return AuthResult(
                authenticated=False,
                error="Request timestamp expired",
                method=AuthMethod.HMAC,
            )

        # Compute expected signature
        message = f"{subject}:{payload}:{timestamp}"
        expected = hmac.new(
            self._secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return AuthResult(
                authenticated=False,
                error="Invalid signature",
                method=AuthMethod.HMAC,
            )

        # Create token for HMAC-authenticated request
        token = Token(
            subject=subject,
            roles=[Role.SYSTEM],
            expires_at=time.time() + 300,  # Short-lived
        )

        self._tokens[token.token_id] = token

        return AuthResult(
            authenticated=True,
            token=token,
            method=AuthMethod.HMAC,
            subject=subject,
        )

    def _authenticate_internal(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate internal agent requests."""
        agent_id = credentials.get("agent_id")

        if not agent_id or not agent_id.startswith("oagent-"):
            return AuthResult(
                authenticated=False,
                error="Invalid agent ID",
                method=AuthMethod.INTERNAL,
            )

        # Create system token for internal requests
        token = Token(
            subject=agent_id,
            roles=[Role.SYSTEM],
            expires_at=time.time() + 3600,
        )

        self._tokens[token.token_id] = token

        return AuthResult(
            authenticated=True,
            token=token,
            method=AuthMethod.INTERNAL,
            subject=agent_id,
        )

    def authorize(
        self,
        token: Token,
        permission: Permission,
        resource: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthorizationResult:
        """
        Check authorization for an action.

        Args:
            token: Authentication token
            permission: Required permission
            resource: Resource being accessed
            context: Additional context

        Returns:
            AuthorizationResult with authorization outcome
        """
        result = AuthorizationResult(permission=permission)
        context = context or {}

        # Check token validity
        if token.is_expired:
            result.reason = "Token expired"
            self._emit_event("authorize", {
                "permission": permission.value,
                "authorized": False,
                "reason": result.reason,
            })
            return result

        # Check basic permission
        if not token.has_permission(permission):
            result.reason = f"Missing permission: {permission.value}"
            self._emit_event("authorize", {
                "permission": permission.value,
                "authorized": False,
                "reason": result.reason,
            })
            return result

        # Evaluate policies
        policies_applied = []
        for policy in sorted(self._policies, key=lambda p: -p.priority):
            applies, allowed = policy.evaluate(token, context)
            if applies:
                policies_applied.append(policy.name)
                if not allowed:
                    result.reason = f"Denied by policy: {policy.name}"
                    result.policies_applied = policies_applied
                    self._emit_event("authorize", {
                        "permission": permission.value,
                        "authorized": False,
                        "policy": policy.name,
                    })
                    return result

        result.authorized = True
        result.policies_applied = policies_applied

        self._emit_event("authorize", {
            "permission": permission.value,
            "authorized": True,
            "subject": token.subject,
        })

        self._audit(
            "authorization",
            token.subject,
            resource or permission.value,
            "allowed",
            {"permission": permission.value, "policies": policies_applied},
        )

        return result

    def add_policy(self, policy: SecurityPolicy) -> None:
        """Add a security policy."""
        self._policies.append(policy)

    def remove_policy(self, name: str) -> bool:
        """Remove a security policy by name."""
        for i, policy in enumerate(self._policies):
            if policy.name == name:
                del self._policies[i]
                return True
        return False

    def list_policies(self) -> List[SecurityPolicy]:
        """List all security policies."""
        return list(self._policies)

    def invalidate_token(self, token_id: str) -> bool:
        """Invalidate a token."""
        if token_id in self._tokens:
            del self._tokens[token_id]
            return True
        return False

    def get_audit_log(
        self,
        subject: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """
        Get audit log entries.

        Args:
            subject: Filter by subject
            event_type: Filter by event type
            since: Only entries after this timestamp
            limit: Maximum entries to return

        Returns:
            List of audit entries
        """
        entries = list(self._audit_log)

        if subject:
            entries = [e for e in entries if e.subject == subject]
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        return entries[-limit:]

    def _audit(
        self,
        action: str,
        subject: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an audit entry."""
        if not self.config.enable_audit:
            return

        entry = AuditEntry(
            event_type="security",
            subject=subject,
            action=action,
            resource=resource,
            result=result,
            details=details or {},
        )

        self._audit_log.append(entry)

        # Trim old entries
        max_entries = 10000
        if len(self._audit_log) > max_entries:
            self._audit_log = self._audit_log[-max_entries:]

        # Persist audit entry
        self._persist_audit(entry)

        self._emit_event("audit", entry.to_dict())

    def _persist_audit(self, entry: AuditEntry) -> None:
        """Persist audit entry to disk."""
        audit_file = Path(self.config.output_dir) / "audit.ndjson"

        try:
            with open(audit_file, "a") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(entry.to_dict()) + "\n")
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def _hash_key(self, key: str) -> str:
        """Hash an API key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def _is_locked_out(self, client_id: str) -> bool:
        """Check if client is locked out."""
        if not self.config.enable_rate_limiting:
            return False

        attempts = self._failed_attempts.get(client_id, [])
        recent = [a for a in attempts if time.time() - a < self.config.lockout_duration_s]

        return len(recent) >= self.config.max_failed_attempts

    def _record_failed_attempt(self, client_id: str) -> None:
        """Record a failed authentication attempt."""
        if client_id not in self._failed_attempts:
            self._failed_attempts[client_id] = []

        self._failed_attempts[client_id].append(time.time())

        # Clean old attempts
        cutoff = time.time() - self.config.lockout_duration_s
        self._failed_attempts[client_id] = [
            a for a in self._failed_attempts[client_id] if a > cutoff
        ]

    def _load_api_keys(self) -> None:
        """Load API keys from disk."""
        keys_file = Path(self.config.output_dir) / "api_keys.json"

        if keys_file.exists():
            try:
                with open(keys_file) as f:
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        self._api_keys = json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except (IOError, json.JSONDecodeError):
                pass

    def _save_api_keys(self) -> None:
        """Save API keys to disk."""
        keys_file = Path(self.config.output_dir) / "api_keys.json"

        try:
            with open(keys_file, "w") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(self._api_keys, f, indent=2)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except IOError:
            pass

    def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit a bus event."""
        topic = self.BUS_TOPICS.get(event_type, f"test.security.{event_type}")
        self.bus.emit({
            "topic": topic,
            "kind": "security",
            "actor": "test-agent",
            "data": data,
        })

    async def authenticate_async(
        self,
        method: AuthMethod,
        credentials: Dict[str, Any],
    ) -> AuthResult:
        """Async version of authenticate."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.authenticate, method, credentials)


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI entry point for Test Security Manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Security Manager")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate key command
    gen_parser = subparsers.add_parser("generate-key", help="Generate API key")
    gen_parser.add_argument("subject", help="Subject (user/agent ID)")
    gen_parser.add_argument("--roles", nargs="*", default=[], help="Roles to assign")

    # Revoke key command
    revoke_parser = subparsers.add_parser("revoke-key", help="Revoke API key")
    revoke_parser.add_argument("key", help="API key to revoke")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="View audit log")
    audit_parser.add_argument("--subject", help="Filter by subject")
    audit_parser.add_argument("--limit", type=int, default=20)

    # Policies command
    policies_parser = subparsers.add_parser("policies", help="List security policies")

    # Common arguments
    parser.add_argument("--output", "-o", default=".pluribus/test-agent/security")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    config = SecurityConfig(output_dir=args.output)
    security = TestSecurityManager(config=config)

    if args.command == "generate-key":
        roles = [Role(r) for r in args.roles if r in [role.value for role in Role]]
        key = security.generate_api_key(args.subject, roles=roles)

        if args.json:
            print(json.dumps({"api_key": key, "subject": args.subject}))
        else:
            print(f"Generated API key for {args.subject}:")
            print(f"  {key}")
            print("\nStore this key securely - it cannot be retrieved later.")

    elif args.command == "revoke-key":
        if security.revoke_api_key(args.key):
            print("API key revoked successfully")
        else:
            print("API key not found")

    elif args.command == "audit":
        entries = security.get_audit_log(subject=args.subject, limit=args.limit)

        if args.json:
            print(json.dumps([e.to_dict() for e in entries], indent=2))
        else:
            print(f"\nAudit Log ({len(entries)} entries):")
            for entry in entries:
                dt = datetime.fromtimestamp(entry.timestamp)
                print(f"  [{dt.strftime('%Y-%m-%d %H:%M:%S')}] {entry.action}")
                print(f"    Subject: {entry.subject}, Resource: {entry.resource}")
                print(f"    Result: {entry.result}")

    elif args.command == "policies":
        policies = security.list_policies()

        if args.json:
            print(json.dumps([p.to_dict() for p in policies], indent=2))
        else:
            print(f"\nSecurity Policies ({len(policies)}):")
            if not policies:
                print("  No policies configured")
            for policy in policies:
                enabled = "[ON]" if policy.enabled else "[OFF]"
                print(f"  {enabled} {policy.name} (priority: {policy.priority})")
                print(f"    {policy.description}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
