#!/usr/bin/env python3
"""
Security Module (Step 191)

Comprehensive security module providing authentication and authorization
for the Review Agent. Implements RBAC, JWT handling, and Omega veto integration.

PBTSO Phase: VERIFY, SEQUESTER
Bus Topics: review.security.auth, review.security.authz, review.security.audit

Security Features:
- JWT-based authentication
- Role-based access control (RBAC)
- Permission management
- Omega veto integration for Ring 0 operations
- Audit logging

Protocol: DKIN v30, CITIZEN v2, PAIP v16
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
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# ============================================================================
# Constants
# ============================================================================

A2A_HEARTBEAT_INTERVAL = 300
A2A_HEARTBEAT_TIMEOUT = 900

# Ring levels from DKIN protocol
RING_0_CONSTITUTIONAL = 0  # Omega authority
RING_1_INFRASTRUCTURE = 1  # Core systems
RING_2_APPLICATION = 2     # Standard agents
RING_3_USER = 3            # External interactions


# ============================================================================
# Types
# ============================================================================

class AuthMethod(Enum):
    """Authentication methods."""
    JWT = "jwt"
    API_KEY = "api_key"
    MTLS = "mtls"
    OAUTH2 = "oauth2"
    INTERNAL = "internal"


class RingLevel(Enum):
    """Security ring levels."""
    RING_0 = 0  # Constitutional - Omega authority
    RING_1 = 1  # Infrastructure - Core systems
    RING_2 = 2  # Application - Standard agents
    RING_3 = 3  # User - External interactions


class PermissionScope(Enum):
    """Permission scopes."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"
    VETO = "veto"


@dataclass
class Permission:
    """
    A permission definition.

    Attributes:
        resource: Resource being accessed
        scope: Access scope (read, write, etc.)
        constraints: Additional constraints
        ring_level: Required ring level
    """
    resource: str
    scope: PermissionScope
    constraints: Dict[str, Any] = field(default_factory=dict)
    ring_level: RingLevel = RingLevel.RING_2

    def __hash__(self):
        return hash((self.resource, self.scope))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resource": self.resource,
            "scope": self.scope.value,
            "constraints": self.constraints,
            "ring_level": self.ring_level.value,
        }

    @classmethod
    def from_string(cls, perm_str: str) -> "Permission":
        """Parse permission from string (e.g., 'review:read')."""
        parts = perm_str.split(":")
        resource = parts[0]
        scope = PermissionScope(parts[1]) if len(parts) > 1 else PermissionScope.READ
        return cls(resource=resource, scope=scope)


@dataclass
class Role:
    """
    A role with associated permissions.

    Attributes:
        name: Role name
        permissions: Set of permissions
        description: Role description
        ring_level: Role's ring level
        inherits: Roles to inherit from
    """
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    ring_level: RingLevel = RingLevel.RING_2
    inherits: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "permissions": [p.to_dict() for p in self.permissions],
            "description": self.description,
            "ring_level": self.ring_level.value,
            "inherits": self.inherits,
        }

    def has_permission(self, permission: Permission) -> bool:
        """Check if role has permission."""
        for p in self.permissions:
            if p.resource == permission.resource and p.scope == permission.scope:
                return True
            # Check wildcard
            if p.resource == "*" or permission.resource.startswith(p.resource.rstrip("*")):
                if p.scope == permission.scope or p.scope == PermissionScope.ADMIN:
                    return True
        return False


@dataclass
class Principal:
    """
    A security principal (user, agent, or service).

    Attributes:
        id: Principal identifier
        name: Display name
        roles: Assigned roles
        permissions: Direct permissions
        ring_level: Principal's ring level
        metadata: Additional metadata
        created_at: Creation timestamp
        expires_at: Expiration timestamp
    """
    id: str
    name: str
    roles: List[str] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)
    ring_level: RingLevel = RingLevel.RING_3
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    expires_at: Optional[str] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "roles": self.roles,
            "permissions": [p.to_dict() for p in self.permissions],
            "ring_level": self.ring_level.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass
class AuthResult:
    """
    Authentication/authorization result.

    Attributes:
        success: Whether auth succeeded
        principal: Authenticated principal
        method: Auth method used
        message: Status message
        token: Auth token (if applicable)
        expires_at: Token expiration
        permissions_granted: Granted permissions
    """
    success: bool
    principal: Optional[Principal] = None
    method: AuthMethod = AuthMethod.INTERNAL
    message: str = ""
    token: Optional[str] = None
    expires_at: Optional[str] = None
    permissions_granted: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "principal": self.principal.to_dict() if self.principal else None,
            "method": self.method.value,
            "message": self.message,
            "token": "***REDACTED***" if self.token else None,
            "expires_at": self.expires_at,
            "permissions_granted": self.permissions_granted,
        }


@dataclass
class AuditEntry:
    """Security audit log entry."""
    entry_id: str
    timestamp: str
    action: str
    principal_id: str
    resource: str
    result: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ============================================================================
# JWT Handling
# ============================================================================

class JWTHandler:
    """
    Minimal JWT handler for authentication.

    Note: For production, use a proper JWT library like PyJWT.
    """

    def __init__(self, secret: str, algorithm: str = "HS256"):
        """Initialize JWT handler."""
        self.secret = secret.encode() if isinstance(secret, str) else secret
        self.algorithm = algorithm

    def _base64url_encode(self, data: bytes) -> str:
        """Base64url encode."""
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    def _base64url_decode(self, data: str) -> bytes:
        """Base64url decode."""
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    def encode(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """
        Encode a JWT token.

        Args:
            payload: Token payload
            expires_in: Expiration in seconds

        Returns:
            JWT token string
        """
        header = {"alg": self.algorithm, "typ": "JWT"}

        # Add standard claims
        now = int(time.time())
        payload = {
            **payload,
            "iat": now,
            "exp": now + expires_in,
            "jti": str(uuid.uuid4())[:8],
        }

        # Encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header).encode())
        payload_b64 = self._base64url_encode(json.dumps(payload).encode())

        # Create signature
        message = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self.secret,
            message.encode(),
            hashlib.sha256,
        ).digest()
        signature_b64 = self._base64url_encode(signature)

        return f"{header_b64}.{payload_b64}.{signature_b64}"

    def decode(self, token: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Decode and verify a JWT token.

        Args:
            token: JWT token string

        Returns:
            Tuple of (valid, payload)
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return False, {"error": "Invalid token format"}

            header_b64, payload_b64, signature_b64 = parts

            # Verify signature
            message = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(
                self.secret,
                message.encode(),
                hashlib.sha256,
            ).digest()

            actual_sig = self._base64url_decode(signature_b64)

            if not hmac.compare_digest(expected_sig, actual_sig):
                return False, {"error": "Invalid signature"}

            # Decode payload
            payload = json.loads(self._base64url_decode(payload_b64))

            # Check expiration
            if "exp" in payload and payload["exp"] < time.time():
                return False, {"error": "Token expired"}

            return True, payload

        except Exception as e:
            return False, {"error": str(e)}


# ============================================================================
# Authentication Manager
# ============================================================================

class AuthenticationManager:
    """
    Manages authentication for the Review Agent.

    Example:
        auth = AuthenticationManager(jwt_secret="secret")

        # Authenticate with API key
        result = await auth.authenticate_api_key("key-123")

        # Generate JWT token
        token = auth.generate_token(principal)

        # Verify token
        result = await auth.verify_token(token)
    """

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        token_expiry: int = 3600,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize authentication manager.

        Args:
            jwt_secret: Secret for JWT signing
            token_expiry: Default token expiry in seconds
            bus_path: Path to event bus file
        """
        self.jwt_secret = jwt_secret or os.environ.get(
            "REVIEW_JWT_SECRET",
            secrets.token_hex(32),
        )
        self.token_expiry = token_expiry
        self.bus_path = bus_path or self._get_bus_path()

        self._jwt = JWTHandler(self.jwt_secret)
        self._api_keys: Dict[str, Principal] = {}
        self._sessions: Dict[str, Principal] = {}

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "security") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "security-module",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def register_api_key(self, api_key: str, principal: Principal) -> None:
        """Register an API key."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        self._api_keys[key_hash] = principal

    async def authenticate_api_key(self, api_key: str) -> AuthResult:
        """
        Authenticate using an API key.

        Args:
            api_key: API key to authenticate

        Returns:
            AuthResult
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self._api_keys:
            self._emit_event("review.security.auth", {
                "method": "api_key",
                "success": False,
                "reason": "invalid_key",
            }, kind="audit")

            return AuthResult(
                success=False,
                method=AuthMethod.API_KEY,
                message="Invalid API key",
            )

        principal = self._api_keys[key_hash]

        # Check expiration
        if principal.expires_at:
            exp_time = datetime.fromisoformat(principal.expires_at.rstrip("Z"))
            if exp_time < datetime.now(timezone.utc):
                return AuthResult(
                    success=False,
                    method=AuthMethod.API_KEY,
                    message="API key expired",
                )

        # Generate session token
        token = self.generate_token(principal)

        self._emit_event("review.security.auth", {
            "method": "api_key",
            "success": True,
            "principal_id": principal.id,
        }, kind="audit")

        return AuthResult(
            success=True,
            principal=principal,
            method=AuthMethod.API_KEY,
            message="Authentication successful",
            token=token,
            expires_at=datetime.fromtimestamp(
                time.time() + self.token_expiry,
                tz=timezone.utc,
            ).isoformat() + "Z",
        )

    def generate_token(self, principal: Principal) -> str:
        """
        Generate a JWT token for a principal.

        Args:
            principal: Authenticated principal

        Returns:
            JWT token
        """
        payload = {
            "sub": principal.id,
            "name": principal.name,
            "roles": principal.roles,
            "ring": principal.ring_level.value,
        }
        return self._jwt.encode(payload, self.token_expiry)

    async def verify_token(self, token: str) -> AuthResult:
        """
        Verify a JWT token.

        Args:
            token: JWT token to verify

        Returns:
            AuthResult
        """
        valid, payload = self._jwt.decode(token)

        if not valid:
            self._emit_event("review.security.auth", {
                "method": "jwt",
                "success": False,
                "reason": payload.get("error", "invalid"),
            }, kind="audit")

            return AuthResult(
                success=False,
                method=AuthMethod.JWT,
                message=payload.get("error", "Invalid token"),
            )

        # Reconstruct principal
        principal = Principal(
            id=payload["sub"],
            name=payload.get("name", ""),
            roles=payload.get("roles", []),
            ring_level=RingLevel(payload.get("ring", 3)),
        )

        self._emit_event("review.security.auth", {
            "method": "jwt",
            "success": True,
            "principal_id": principal.id,
        }, kind="audit")

        return AuthResult(
            success=True,
            principal=principal,
            method=AuthMethod.JWT,
            message="Token verified",
            token=token,
        )

    async def authenticate_internal(self, agent_id: str) -> AuthResult:
        """
        Authenticate an internal agent.

        Args:
            agent_id: Internal agent identifier

        Returns:
            AuthResult
        """
        # Internal agents get Ring 1 access
        principal = Principal(
            id=agent_id,
            name=f"Agent: {agent_id}",
            roles=["agent"],
            ring_level=RingLevel.RING_1,
            metadata={"type": "internal_agent"},
        )

        token = self.generate_token(principal)

        self._emit_event("review.security.auth", {
            "method": "internal",
            "success": True,
            "agent_id": agent_id,
        }, kind="audit")

        return AuthResult(
            success=True,
            principal=principal,
            method=AuthMethod.INTERNAL,
            message="Internal authentication successful",
            token=token,
        )


# ============================================================================
# Authorization Manager
# ============================================================================

class AuthorizationManager:
    """
    Manages authorization for the Review Agent.

    Example:
        authz = AuthorizationManager()

        # Define roles
        authz.define_role(Role(
            name="reviewer",
            permissions={
                Permission("review", PermissionScope.READ),
                Permission("review", PermissionScope.WRITE),
            }
        ))

        # Check authorization
        result = await authz.authorize(
            principal,
            Permission("review", PermissionScope.WRITE),
        )
    """

    def __init__(
        self,
        bus_path: Optional[Path] = None,
    ):
        """Initialize authorization manager."""
        self.bus_path = bus_path or self._get_bus_path()

        self._roles: Dict[str, Role] = {}
        self._policies: List[Callable[[Principal, Permission], bool]] = []

        # Define default roles
        self._define_default_roles()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "security") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "security-module",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    def _define_default_roles(self) -> None:
        """Define default roles."""
        # Admin role - full access
        self.define_role(Role(
            name="admin",
            permissions={
                Permission("*", PermissionScope.ADMIN),
            },
            description="Full administrative access",
            ring_level=RingLevel.RING_1,
        ))

        # Reviewer role - read/write reviews
        self.define_role(Role(
            name="reviewer",
            permissions={
                Permission("review", PermissionScope.READ),
                Permission("review", PermissionScope.WRITE),
                Permission("comment", PermissionScope.WRITE),
            },
            description="Code review access",
            ring_level=RingLevel.RING_2,
        ))

        # Read-only role
        self.define_role(Role(
            name="reader",
            permissions={
                Permission("review", PermissionScope.READ),
                Permission("comment", PermissionScope.READ),
                Permission("metrics", PermissionScope.READ),
            },
            description="Read-only access",
            ring_level=RingLevel.RING_3,
        ))

        # Agent role - internal agents
        self.define_role(Role(
            name="agent",
            permissions={
                Permission("review", PermissionScope.READ),
                Permission("review", PermissionScope.WRITE),
                Permission("review", PermissionScope.EXECUTE),
                Permission("bus", PermissionScope.WRITE),
            },
            description="Internal agent access",
            ring_level=RingLevel.RING_1,
        ))

        # Omega role - veto authority
        self.define_role(Role(
            name="omega",
            permissions={
                Permission("*", PermissionScope.ADMIN),
                Permission("veto", PermissionScope.VETO),
            },
            description="Omega constitutional authority",
            ring_level=RingLevel.RING_0,
        ))

    def define_role(self, role: Role) -> None:
        """Define a role."""
        self._roles[role.name] = role

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def add_policy(
        self,
        policy: Callable[[Principal, Permission], bool],
    ) -> None:
        """Add a custom authorization policy."""
        self._policies.append(policy)

    def _get_effective_permissions(
        self,
        principal: Principal,
    ) -> Set[Permission]:
        """Get all effective permissions for a principal."""
        permissions = set(principal.permissions)

        # Add permissions from roles
        for role_name in principal.roles:
            role = self._roles.get(role_name)
            if role:
                permissions.update(role.permissions)

                # Handle role inheritance
                for inherited_name in role.inherits:
                    inherited = self._roles.get(inherited_name)
                    if inherited:
                        permissions.update(inherited.permissions)

        return permissions

    async def authorize(
        self,
        principal: Principal,
        permission: Permission,
        resource_context: Optional[Dict[str, Any]] = None,
    ) -> AuthResult:
        """
        Check if principal is authorized for permission.

        Args:
            principal: The security principal
            permission: Required permission
            resource_context: Additional context

        Returns:
            AuthResult
        """
        # Check ring level first
        if permission.ring_level.value < principal.ring_level.value:
            self._emit_event("review.security.authz", {
                "principal_id": principal.id,
                "permission": f"{permission.resource}:{permission.scope.value}",
                "success": False,
                "reason": "insufficient_ring_level",
            }, kind="audit")

            return AuthResult(
                success=False,
                principal=principal,
                message=f"Insufficient ring level: requires {permission.ring_level.name}",
            )

        # Get effective permissions
        effective = self._get_effective_permissions(principal)

        # Check direct permission
        for p in effective:
            if p.resource == permission.resource and p.scope == permission.scope:
                self._emit_event("review.security.authz", {
                    "principal_id": principal.id,
                    "permission": f"{permission.resource}:{permission.scope.value}",
                    "success": True,
                }, kind="audit")

                return AuthResult(
                    success=True,
                    principal=principal,
                    message="Authorized",
                    permissions_granted=[f"{p.resource}:{p.scope.value}"],
                )

            # Check wildcard and admin
            if p.resource == "*" or p.scope == PermissionScope.ADMIN:
                self._emit_event("review.security.authz", {
                    "principal_id": principal.id,
                    "permission": f"{permission.resource}:{permission.scope.value}",
                    "success": True,
                    "via": "admin_or_wildcard",
                }, kind="audit")

                return AuthResult(
                    success=True,
                    principal=principal,
                    message="Authorized via admin permission",
                    permissions_granted=[f"{p.resource}:{p.scope.value}"],
                )

        # Check custom policies
        for policy in self._policies:
            if policy(principal, permission):
                self._emit_event("review.security.authz", {
                    "principal_id": principal.id,
                    "permission": f"{permission.resource}:{permission.scope.value}",
                    "success": True,
                    "via": "custom_policy",
                }, kind="audit")

                return AuthResult(
                    success=True,
                    principal=principal,
                    message="Authorized via custom policy",
                )

        # Denied
        self._emit_event("review.security.authz", {
            "principal_id": principal.id,
            "permission": f"{permission.resource}:{permission.scope.value}",
            "success": False,
            "reason": "no_permission",
        }, kind="audit")

        return AuthResult(
            success=False,
            principal=principal,
            message=f"Not authorized for {permission.resource}:{permission.scope.value}",
        )

    async def require_omega_veto(
        self,
        principal: Principal,
        action: str,
        context: Dict[str, Any],
    ) -> AuthResult:
        """
        Check if action requires Omega veto approval.

        For Ring 0 operations that need constitutional-level approval.

        Args:
            principal: The security principal
            action: Action being performed
            context: Action context

        Returns:
            AuthResult with veto status
        """
        # Check if principal has veto permission
        veto_perm = Permission("veto", PermissionScope.VETO, ring_level=RingLevel.RING_0)
        auth_result = await self.authorize(principal, veto_perm)

        if not auth_result.success:
            self._emit_event("review.security.audit", {
                "action": "veto_required",
                "principal_id": principal.id,
                "target_action": action,
                "status": "denied",
            }, kind="audit")

            return AuthResult(
                success=False,
                principal=principal,
                message="Omega veto authority required",
            )

        self._emit_event("review.security.audit", {
            "action": "veto_approved",
            "principal_id": principal.id,
            "target_action": action,
            "context": context,
        }, kind="audit")

        return AuthResult(
            success=True,
            principal=principal,
            message="Omega veto approved",
        )


# ============================================================================
# Security Module (Combined)
# ============================================================================

class SecurityModule:
    """
    Combined security module for authentication and authorization.

    Example:
        security = SecurityModule()

        # Authenticate
        auth_result = await security.authenticate(token=token)

        # Authorize
        if auth_result.success:
            authz_result = await security.authorize(
                auth_result.principal,
                "review:write",
            )
    """

    BUS_TOPICS = {
        "auth": "review.security.auth",
        "authz": "review.security.authz",
        "audit": "review.security.audit",
    }

    def __init__(
        self,
        jwt_secret: Optional[str] = None,
        bus_path: Optional[Path] = None,
    ):
        """Initialize security module."""
        self.bus_path = bus_path or self._get_bus_path()

        self.authentication = AuthenticationManager(
            jwt_secret=jwt_secret,
            bus_path=self.bus_path,
        )
        self.authorization = AuthorizationManager(
            bus_path=self.bus_path,
        )

        self._audit_log: List[AuditEntry] = []
        self._last_heartbeat = time.time()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any], kind: str = "security") -> str:
        """Emit event to bus with file locking."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "actor": "security-module",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id

    async def authenticate(
        self,
        token: Optional[str] = None,
        api_key: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> AuthResult:
        """
        Authenticate using various methods.

        Args:
            token: JWT token
            api_key: API key
            agent_id: Internal agent ID

        Returns:
            AuthResult
        """
        if token:
            return await self.authentication.verify_token(token)
        elif api_key:
            return await self.authentication.authenticate_api_key(api_key)
        elif agent_id:
            return await self.authentication.authenticate_internal(agent_id)
        else:
            return AuthResult(
                success=False,
                message="No authentication credentials provided",
            )

    async def authorize(
        self,
        principal: Principal,
        permission_str: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthResult:
        """
        Authorize a principal for a permission.

        Args:
            principal: Authenticated principal
            permission_str: Permission string (e.g., "review:write")
            context: Additional context

        Returns:
            AuthResult
        """
        permission = Permission.from_string(permission_str)
        return await self.authorization.authorize(principal, permission, context)

    def audit(
        self,
        action: str,
        principal_id: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Create an audit log entry.

        Args:
            action: Action performed
            principal_id: Principal performing action
            resource: Resource accessed
            result: Action result
            details: Additional details

        Returns:
            AuditEntry
        """
        entry = AuditEntry(
            entry_id="",
            timestamp="",
            action=action,
            principal_id=principal_id,
            resource=resource,
            result=result,
            details=details or {},
        )

        self._audit_log.append(entry)

        self._emit_event(self.BUS_TOPICS["audit"], entry.to_dict(), kind="audit")

        return entry

    def get_audit_log(
        self,
        limit: int = 100,
        principal_id: Optional[str] = None,
    ) -> List[AuditEntry]:
        """Get audit log entries."""
        entries = self._audit_log
        if principal_id:
            entries = [e for e in entries if e.principal_id == principal_id]
        return entries[-limit:]

    def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "security-module",
            "healthy": True,
            "roles_defined": len(self.authorization._roles),
            "api_keys_registered": len(self.authentication._api_keys),
            "audit_entries": len(self._audit_log),
            "last_heartbeat": self._last_heartbeat,
            "interval": A2A_HEARTBEAT_INTERVAL,
            "timeout": A2A_HEARTBEAT_TIMEOUT,
        }
        self._last_heartbeat = now

        self._emit_event("a2a.heartbeat", status, kind="heartbeat")
        return status


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Security Module."""
    import argparse

    parser = argparse.ArgumentParser(description="Security Module (Step 191)")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Auth command
    auth_parser = subparsers.add_parser("auth", help="Authenticate")
    auth_parser.add_argument("--token", help="JWT token")
    auth_parser.add_argument("--api-key", help="API key")
    auth_parser.add_argument("--agent-id", help="Internal agent ID")

    # Roles command
    subparsers.add_parser("roles", help="List roles")

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Show audit log")
    audit_parser.add_argument("--limit", type=int, default=20, help="Entries to show")

    # Generate token command
    token_parser = subparsers.add_parser("token", help="Generate token")
    token_parser.add_argument("principal_id", help="Principal ID")
    token_parser.add_argument("--name", default="User", help="Principal name")
    token_parser.add_argument("--roles", nargs="+", default=["reader"], help="Roles")

    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    security = SecurityModule()

    if args.command == "auth":
        result = asyncio.run(security.authenticate(
            token=args.token,
            api_key=args.api_key,
            agent_id=args.agent_id,
        ))
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"Authentication: {status}")
            print(f"  Message: {result.message}")
            if result.principal:
                print(f"  Principal: {result.principal.id}")
                print(f"  Ring Level: {result.principal.ring_level.name}")

    elif args.command == "roles":
        roles = security.authorization._roles
        if args.json:
            print(json.dumps({k: v.to_dict() for k, v in roles.items()}, indent=2))
        else:
            print(f"Roles: {len(roles)}")
            for name, role in roles.items():
                print(f"  {name} ({role.ring_level.name})")
                print(f"    {role.description}")
                print(f"    Permissions: {len(role.permissions)}")

    elif args.command == "audit":
        entries = security.get_audit_log(limit=args.limit)
        if args.json:
            print(json.dumps([e.to_dict() for e in entries], indent=2))
        else:
            print(f"Audit Log: {len(entries)} entries")
            for entry in entries:
                print(f"  [{entry.timestamp}] {entry.action} by {entry.principal_id}: {entry.result}")

    elif args.command == "token":
        principal = Principal(
            id=args.principal_id,
            name=args.name,
            roles=args.roles,
            ring_level=RingLevel.RING_2,
        )
        token = security.authentication.generate_token(principal)
        if args.json:
            print(json.dumps({"token": token, "principal": principal.to_dict()}, indent=2))
        else:
            print(f"Token generated for {args.principal_id}")
            print(f"  Token: {token[:50]}...")

    else:
        # Default: show status
        status = security.heartbeat()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print(f"Security Module: {status['roles_defined']} roles, {status['api_keys_registered']} API keys")

    return 0


if __name__ == "__main__":
    sys.exit(main())
