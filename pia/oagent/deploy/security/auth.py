#!/usr/bin/env python3
"""
auth.py - Security Module (Step 241)

PBTSO Phase: SEQUESTER
A2A Integration: Provides authentication/authorization via deploy.security.auth

Provides:
- AuthMethod: Authentication methods
- Permission: Permission types
- Role: Role definitions
- Principal: Security principal
- AuthToken: Authentication token
- AuthResult: Authentication result
- AccessPolicy: Access control policy
- SecurityModule: Complete authentication and authorization

Bus Topics:
- deploy.security.auth
- deploy.security.authz
- deploy.security.token
- deploy.security.audit

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
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
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


# ==============================================================================
# Bus Emission Helper with fcntl.flock()
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "security-module"
) -> str:
    """Emit an event to the Pluribus bus with file locking."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class AuthMethod(Enum):
    """Authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    OAUTH2 = "oauth2"
    MTLS = "mtls"
    BASIC = "basic"
    BEARER = "bearer"
    HMAC = "hmac"
    SAML = "saml"
    OIDC = "oidc"


class Permission(Enum):
    """Permission types."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"
    CONFIGURE = "configure"
    AUDIT = "audit"
    EXECUTE = "execute"


class TokenStatus(Enum):
    """Token status."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    PENDING = "pending"


@dataclass
class Role:
    """
    Role definition with permissions.

    Attributes:
        role_id: Unique role identifier
        name: Human-readable role name
        permissions: Set of permissions
        description: Role description
        inherits_from: Parent roles
        created_at: Creation timestamp
    """
    role_id: str
    name: str
    permissions: Set[Permission] = field(default_factory=set)
    description: str = ""
    inherits_from: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role_id": self.role_id,
            "name": self.name,
            "permissions": [p.value for p in self.permissions],
            "description": self.description,
            "inherits_from": self.inherits_from,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Role":
        data = dict(data)
        if "permissions" in data:
            data["permissions"] = {Permission(p) for p in data["permissions"]}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Principal:
    """
    Security principal (user, service, or system).

    Attributes:
        principal_id: Unique principal identifier
        name: Principal name
        principal_type: Type (user, service, system)
        roles: Assigned roles
        attributes: Additional attributes (claims)
        enabled: Whether principal is enabled
        created_at: Creation timestamp
        last_auth_at: Last authentication timestamp
    """
    principal_id: str
    name: str
    principal_type: str = "service"
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_auth_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Principal":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AuthToken:
    """
    Authentication token.

    Attributes:
        token_id: Unique token identifier
        principal_id: Associated principal
        token_hash: Hashed token value
        method: Authentication method
        scopes: Token scopes
        status: Token status
        issued_at: Issue timestamp
        expires_at: Expiration timestamp
        metadata: Additional metadata
    """
    token_id: str
    principal_id: str
    token_hash: str
    method: AuthMethod = AuthMethod.BEARER
    scopes: List[str] = field(default_factory=list)
    status: TokenStatus = TokenStatus.ACTIVE
    issued_at: float = field(default_factory=time.time)
    expires_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "principal_id": self.principal_id,
            "token_hash": self.token_hash,
            "method": self.method.value,
            "scopes": self.scopes,
            "status": self.status.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthToken":
        data = dict(data)
        if "method" in data:
            data["method"] = AuthMethod(data["method"])
        if "status" in data:
            data["status"] = TokenStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class AuthResult:
    """
    Authentication/authorization result.

    Attributes:
        success: Whether auth succeeded
        principal: Authenticated principal
        permissions: Effective permissions
        token: Auth token if issued
        error: Error message if failed
        metadata: Additional metadata
    """
    success: bool
    principal: Optional[Principal] = None
    permissions: Set[Permission] = field(default_factory=set)
    token: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "principal": self.principal.to_dict() if self.principal else None,
            "permissions": [p.value for p in self.permissions],
            "token": self.token,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AccessPolicy:
    """
    Access control policy.

    Attributes:
        policy_id: Unique policy identifier
        name: Policy name
        resource_pattern: Resource pattern (glob)
        required_permissions: Required permissions
        allowed_principals: Allowed principal patterns
        denied_principals: Denied principal patterns
        conditions: Additional conditions
        priority: Policy priority (higher wins)
        enabled: Whether policy is enabled
    """
    policy_id: str
    name: str
    resource_pattern: str = "*"
    required_permissions: Set[Permission] = field(default_factory=set)
    allowed_principals: List[str] = field(default_factory=list)
    denied_principals: List[str] = field(default_factory=list)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "name": self.name,
            "resource_pattern": self.resource_pattern,
            "required_permissions": [p.value for p in self.required_permissions],
            "allowed_principals": self.allowed_principals,
            "denied_principals": self.denied_principals,
            "conditions": self.conditions,
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AccessPolicy":
        data = dict(data)
        if "required_permissions" in data:
            data["required_permissions"] = {Permission(p) for p in data["required_permissions"]}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Security Module (Step 241)
# ==============================================================================

class SecurityModule:
    """
    Security Module - Authentication and authorization for deployments.

    PBTSO Phase: SEQUESTER

    Responsibilities:
    - Authenticate principals (users, services)
    - Issue and validate tokens
    - Manage roles and permissions
    - Enforce access policies
    - Audit security events

    A2A Heartbeat: 300s interval, 900s timeout

    Example:
        >>> security = SecurityModule()
        >>> principal = security.create_principal(
        ...     name="api-service",
        ...     principal_type="service"
        ... )
        >>> security.assign_role(principal.principal_id, "deployer")
        >>> result = await security.authenticate(
        ...     method=AuthMethod.API_KEY,
        ...     credentials={"api_key": "pk_..."}
        ... )
        >>> if result.success:
        ...     authz = await security.authorize(
        ...         principal=result.principal,
        ...         resource="deploy/api-service",
        ...         permission=Permission.DEPLOY
        ...     )
    """

    BUS_TOPICS = {
        "auth": "deploy.security.auth",
        "authz": "deploy.security.authz",
        "token": "deploy.security.token",
        "audit": "deploy.security.audit",
        "heartbeat": "a2a.security.heartbeat",
    }

    # A2A heartbeat configuration
    HEARTBEAT_INTERVAL = 300  # 5 minutes
    HEARTBEAT_TIMEOUT = 900   # 15 minutes

    # Default roles
    DEFAULT_ROLES = {
        "admin": Role(
            role_id="role-admin",
            name="admin",
            permissions={p for p in Permission},
            description="Full administrative access",
        ),
        "deployer": Role(
            role_id="role-deployer",
            name="deployer",
            permissions={Permission.READ, Permission.DEPLOY, Permission.ROLLBACK, Permission.CONFIGURE},
            description="Can deploy and rollback services",
        ),
        "operator": Role(
            role_id="role-operator",
            name="operator",
            permissions={Permission.READ, Permission.EXECUTE, Permission.CONFIGURE},
            description="Can operate and configure services",
        ),
        "auditor": Role(
            role_id="role-auditor",
            name="auditor",
            permissions={Permission.READ, Permission.AUDIT},
            description="Can audit and read resources",
        ),
        "viewer": Role(
            role_id="role-viewer",
            name="viewer",
            permissions={Permission.READ},
            description="Read-only access",
        ),
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "security-module",
        secret_key: Optional[str] = None,
    ):
        """
        Initialize the security module.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
            secret_key: Secret key for token signing
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "security"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Initialize secret key
        self._secret_key = self._init_secret_key(secret_key)

        # Storage
        self._roles: Dict[str, Role] = {}
        self._principals: Dict[str, Principal] = {}
        self._tokens: Dict[str, AuthToken] = {}
        self._api_keys: Dict[str, str] = {}  # key_hash -> principal_id
        self._policies: Dict[str, AccessPolicy] = {}
        self._audit_log: List[Dict[str, Any]] = []

        # Initialize default roles
        self._roles.update(self.DEFAULT_ROLES)

        self._load_state()

        # Heartbeat tracking
        self._last_heartbeat = time.time()

    def _init_secret_key(self, key: Optional[str]) -> bytes:
        """Initialize or load secret key."""
        key_file = self.state_dir / ".secret_key"

        if key:
            return key.encode()

        if key_file.exists():
            return key_file.read_bytes()

        # Generate new key
        new_key = secrets.token_bytes(32)
        key_file.write_bytes(new_key)
        os.chmod(key_file, 0o600)
        return new_key

    def create_principal(
        self,
        name: str,
        principal_type: str = "service",
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Principal:
        """
        Create a new security principal.

        Args:
            name: Principal name
            principal_type: Type (user, service, system)
            roles: Initial roles
            attributes: Additional attributes

        Returns:
            Created Principal
        """
        principal_id = f"principal-{uuid.uuid4().hex[:12]}"

        principal = Principal(
            principal_id=principal_id,
            name=name,
            principal_type=principal_type,
            roles=roles or [],
            attributes=attributes or {},
        )

        self._principals[principal_id] = principal
        self._save_state()

        self._audit("principal_created", {
            "principal_id": principal_id,
            "name": name,
            "principal_type": principal_type,
        })

        return principal

    def create_api_key(
        self,
        principal_id: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: int = 365,
    ) -> tuple[str, AuthToken]:
        """
        Create an API key for a principal.

        Args:
            principal_id: Principal ID
            scopes: Token scopes
            expires_in_days: Expiration in days

        Returns:
            Tuple of (raw_api_key, AuthToken)
        """
        principal = self._principals.get(principal_id)
        if not principal:
            raise ValueError(f"Principal not found: {principal_id}")

        # Generate API key
        raw_key = f"pk_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_token(raw_key)

        token_id = f"token-{uuid.uuid4().hex[:12]}"
        expires_at = time.time() + (expires_in_days * 86400)

        token = AuthToken(
            token_id=token_id,
            principal_id=principal_id,
            token_hash=key_hash,
            method=AuthMethod.API_KEY,
            scopes=scopes or [],
            expires_at=expires_at,
        )

        self._tokens[token_id] = token
        self._api_keys[key_hash] = principal_id
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["token"],
            {
                "action": "created",
                "token_id": token_id,
                "principal_id": principal_id,
                "method": "api_key",
                "expires_at": expires_at,
            },
            actor=self.actor_id,
        )

        self._audit("api_key_created", {
            "token_id": token_id,
            "principal_id": principal_id,
        })

        return raw_key, token

    def assign_role(self, principal_id: str, role_name: str) -> bool:
        """Assign a role to a principal."""
        principal = self._principals.get(principal_id)
        if not principal:
            return False

        if role_name not in self._roles:
            return False

        if role_name not in principal.roles:
            principal.roles.append(role_name)
            self._save_state()

            self._audit("role_assigned", {
                "principal_id": principal_id,
                "role": role_name,
            })

        return True

    def revoke_role(self, principal_id: str, role_name: str) -> bool:
        """Revoke a role from a principal."""
        principal = self._principals.get(principal_id)
        if not principal:
            return False

        if role_name in principal.roles:
            principal.roles.remove(role_name)
            self._save_state()

            self._audit("role_revoked", {
                "principal_id": principal_id,
                "role": role_name,
            })
            return True

        return False

    def create_role(
        self,
        name: str,
        permissions: Set[Permission],
        description: str = "",
        inherits_from: Optional[List[str]] = None,
    ) -> Role:
        """Create a custom role."""
        role_id = f"role-{uuid.uuid4().hex[:12]}"

        role = Role(
            role_id=role_id,
            name=name,
            permissions=permissions,
            description=description,
            inherits_from=inherits_from or [],
        )

        self._roles[name] = role
        self._save_state()

        self._audit("role_created", {
            "role_id": role_id,
            "name": name,
            "permissions": [p.value for p in permissions],
        })

        return role

    async def authenticate(
        self,
        method: AuthMethod,
        credentials: Dict[str, Any],
    ) -> AuthResult:
        """
        Authenticate a principal.

        Args:
            method: Authentication method
            credentials: Authentication credentials

        Returns:
            AuthResult with success/failure
        """
        _emit_bus_event(
            self.BUS_TOPICS["auth"],
            {
                "method": method.value,
                "status": "attempting",
            },
            actor=self.actor_id,
        )

        try:
            if method == AuthMethod.API_KEY:
                result = await self._auth_api_key(credentials)
            elif method == AuthMethod.BEARER:
                result = await self._auth_bearer(credentials)
            elif method == AuthMethod.BASIC:
                result = await self._auth_basic(credentials)
            elif method == AuthMethod.HMAC:
                result = await self._auth_hmac(credentials)
            else:
                result = AuthResult(
                    success=False,
                    error=f"Unsupported authentication method: {method.value}",
                )

            # Log auth event
            self._audit(
                "authentication",
                {
                    "method": method.value,
                    "success": result.success,
                    "principal_id": result.principal.principal_id if result.principal else None,
                    "error": result.error,
                },
            )

            if result.success and result.principal:
                result.principal.last_auth_at = time.time()
                result.permissions = self.get_effective_permissions(result.principal.principal_id)

            return result

        except Exception as e:
            return AuthResult(
                success=False,
                error=f"Authentication error: {str(e)}",
            )

    async def _auth_api_key(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via API key."""
        api_key = credentials.get("api_key")
        if not api_key:
            return AuthResult(success=False, error="API key required")

        key_hash = self._hash_token(api_key)
        principal_id = self._api_keys.get(key_hash)

        if not principal_id:
            return AuthResult(success=False, error="Invalid API key")

        principal = self._principals.get(principal_id)
        if not principal:
            return AuthResult(success=False, error="Principal not found")

        if not principal.enabled:
            return AuthResult(success=False, error="Principal disabled")

        # Find token and check expiration
        for token in self._tokens.values():
            if token.token_hash == key_hash:
                if token.status != TokenStatus.ACTIVE:
                    return AuthResult(success=False, error="Token not active")
                if token.expires_at > 0 and time.time() > token.expires_at:
                    token.status = TokenStatus.EXPIRED
                    return AuthResult(success=False, error="Token expired")
                break

        return AuthResult(
            success=True,
            principal=principal,
        )

    async def _auth_bearer(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via bearer token."""
        token = credentials.get("token")
        if not token:
            return AuthResult(success=False, error="Bearer token required")

        # Verify JWT-like token
        try:
            payload = self._verify_token(token)
            principal_id = payload.get("sub")
            principal = self._principals.get(principal_id)

            if not principal:
                return AuthResult(success=False, error="Principal not found")

            return AuthResult(
                success=True,
                principal=principal,
                metadata=payload,
            )
        except Exception as e:
            return AuthResult(success=False, error=f"Token verification failed: {e}")

    async def _auth_basic(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via basic auth."""
        username = credentials.get("username")
        password = credentials.get("password")

        if not username or not password:
            return AuthResult(success=False, error="Username and password required")

        # Find principal by name
        principal = self.get_principal_by_name(username)
        if not principal:
            return AuthResult(success=False, error="Invalid credentials")

        # Verify password (stored as attribute)
        stored_hash = principal.attributes.get("password_hash")
        if not stored_hash:
            return AuthResult(success=False, error="Password not configured")

        if not self._verify_password(password, stored_hash):
            return AuthResult(success=False, error="Invalid credentials")

        return AuthResult(
            success=True,
            principal=principal,
        )

    async def _auth_hmac(self, credentials: Dict[str, Any]) -> AuthResult:
        """Authenticate via HMAC signature."""
        signature = credentials.get("signature")
        timestamp = credentials.get("timestamp")
        principal_id = credentials.get("principal_id")
        message = credentials.get("message")

        if not all([signature, timestamp, principal_id, message]):
            return AuthResult(success=False, error="Missing HMAC parameters")

        # Check timestamp freshness (within 5 minutes)
        try:
            ts = float(timestamp)
            if abs(time.time() - ts) > 300:
                return AuthResult(success=False, error="Timestamp too old")
        except ValueError:
            return AuthResult(success=False, error="Invalid timestamp")

        principal = self._principals.get(principal_id)
        if not principal:
            return AuthResult(success=False, error="Principal not found")

        # Get principal's secret
        secret = principal.attributes.get("hmac_secret")
        if not secret:
            return AuthResult(success=False, error="HMAC not configured")

        # Verify signature
        expected = hmac.new(
            secret.encode(),
            f"{timestamp}:{message}".encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return AuthResult(success=False, error="Invalid signature")

        return AuthResult(
            success=True,
            principal=principal,
        )

    async def authorize(
        self,
        principal: Principal,
        resource: str,
        permission: Permission,
        context: Optional[Dict[str, Any]] = None,
    ) -> AuthResult:
        """
        Authorize access to a resource.

        Args:
            principal: Authenticated principal
            resource: Resource identifier
            permission: Required permission
            context: Additional context

        Returns:
            AuthResult with authorization decision
        """
        _emit_bus_event(
            self.BUS_TOPICS["authz"],
            {
                "principal_id": principal.principal_id,
                "resource": resource,
                "permission": permission.value,
            },
            actor=self.actor_id,
        )

        # Check if principal is enabled
        if not principal.enabled:
            return AuthResult(
                success=False,
                principal=principal,
                error="Principal disabled",
            )

        # Get effective permissions
        effective_perms = self.get_effective_permissions(principal.principal_id)

        # Check direct permission
        if Permission.ADMIN in effective_perms:
            # Admin has all permissions
            return AuthResult(
                success=True,
                principal=principal,
                permissions=effective_perms,
            )

        # Check specific permission
        if permission not in effective_perms:
            self._audit("authorization_denied", {
                "principal_id": principal.principal_id,
                "resource": resource,
                "permission": permission.value,
                "reason": "permission_not_granted",
            })
            return AuthResult(
                success=False,
                principal=principal,
                permissions=effective_perms,
                error=f"Permission denied: {permission.value}",
            )

        # Check policies
        policy_result = self._evaluate_policies(principal, resource, permission, context)
        if not policy_result:
            self._audit("authorization_denied", {
                "principal_id": principal.principal_id,
                "resource": resource,
                "permission": permission.value,
                "reason": "policy_denied",
            })
            return AuthResult(
                success=False,
                principal=principal,
                permissions=effective_perms,
                error="Access denied by policy",
            )

        self._audit("authorization_granted", {
            "principal_id": principal.principal_id,
            "resource": resource,
            "permission": permission.value,
        })

        return AuthResult(
            success=True,
            principal=principal,
            permissions=effective_perms,
        )

    def _evaluate_policies(
        self,
        principal: Principal,
        resource: str,
        permission: Permission,
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """Evaluate access policies."""
        # Get applicable policies
        applicable = []
        for policy in self._policies.values():
            if not policy.enabled:
                continue
            if self._matches_pattern(resource, policy.resource_pattern):
                applicable.append(policy)

        if not applicable:
            # No policies = allow (permissions already checked)
            return True

        # Sort by priority (higher wins)
        applicable.sort(key=lambda p: p.priority, reverse=True)

        for policy in applicable:
            # Check deny list first
            for pattern in policy.denied_principals:
                if self._matches_pattern(principal.name, pattern):
                    return False

            # Check allow list
            if policy.allowed_principals:
                allowed = False
                for pattern in policy.allowed_principals:
                    if self._matches_pattern(principal.name, pattern):
                        allowed = True
                        break
                if not allowed:
                    continue

            # Check required permissions
            if policy.required_permissions:
                effective = self.get_effective_permissions(principal.principal_id)
                if not policy.required_permissions.issubset(effective):
                    continue

            # Policy allows access
            return True

        return True

    def _matches_pattern(self, value: str, pattern: str) -> bool:
        """Match a value against a glob pattern."""
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            return value.startswith(pattern[:-1])
        if pattern.startswith("*"):
            return value.endswith(pattern[1:])
        return value == pattern

    def get_effective_permissions(self, principal_id: str) -> Set[Permission]:
        """Get all effective permissions for a principal."""
        principal = self._principals.get(principal_id)
        if not principal:
            return set()

        permissions = set()
        visited_roles = set()

        def collect_permissions(role_name: str):
            if role_name in visited_roles:
                return
            visited_roles.add(role_name)

            role = self._roles.get(role_name)
            if not role:
                return

            permissions.update(role.permissions)
            for parent in role.inherits_from:
                collect_permissions(parent)

        for role_name in principal.roles:
            collect_permissions(role_name)

        return permissions

    def create_policy(
        self,
        name: str,
        resource_pattern: str = "*",
        required_permissions: Optional[Set[Permission]] = None,
        allowed_principals: Optional[List[str]] = None,
        denied_principals: Optional[List[str]] = None,
        priority: int = 0,
    ) -> AccessPolicy:
        """Create an access policy."""
        policy_id = f"policy-{uuid.uuid4().hex[:12]}"

        policy = AccessPolicy(
            policy_id=policy_id,
            name=name,
            resource_pattern=resource_pattern,
            required_permissions=required_permissions or set(),
            allowed_principals=allowed_principals or [],
            denied_principals=denied_principals or [],
            priority=priority,
        )

        self._policies[policy_id] = policy
        self._save_state()

        self._audit("policy_created", {
            "policy_id": policy_id,
            "name": name,
            "resource_pattern": resource_pattern,
        })

        return policy

    def revoke_token(self, token_id: str, reason: str = "manual") -> bool:
        """Revoke a token."""
        token = self._tokens.get(token_id)
        if not token:
            return False

        token.status = TokenStatus.REVOKED

        # Remove from API keys if applicable
        if token.token_hash in self._api_keys:
            del self._api_keys[token.token_hash]

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["token"],
            {
                "action": "revoked",
                "token_id": token_id,
                "principal_id": token.principal_id,
                "reason": reason,
            },
            level="warn",
            actor=self.actor_id,
        )

        self._audit("token_revoked", {
            "token_id": token_id,
            "principal_id": token.principal_id,
            "reason": reason,
        })

        return True

    def issue_bearer_token(
        self,
        principal_id: str,
        scopes: Optional[List[str]] = None,
        expires_in_seconds: int = 3600,
    ) -> tuple[str, AuthToken]:
        """Issue a bearer token."""
        principal = self._principals.get(principal_id)
        if not principal:
            raise ValueError(f"Principal not found: {principal_id}")

        token_id = f"token-{uuid.uuid4().hex[:12]}"
        expires_at = time.time() + expires_in_seconds

        # Create JWT-like token
        payload = {
            "sub": principal_id,
            "name": principal.name,
            "scopes": scopes or [],
            "iat": int(time.time()),
            "exp": int(expires_at),
            "jti": token_id,
        }

        raw_token = self._sign_token(payload)
        token_hash = self._hash_token(raw_token)

        token = AuthToken(
            token_id=token_id,
            principal_id=principal_id,
            token_hash=token_hash,
            method=AuthMethod.BEARER,
            scopes=scopes or [],
            expires_at=expires_at,
        )

        self._tokens[token_id] = token
        self._save_state()

        return raw_token, token

    def _sign_token(self, payload: Dict[str, Any]) -> str:
        """Sign a token payload."""
        # Simple JWT-like structure
        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode()
        body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
        signature = hmac.new(
            self._secret_key,
            f"{header}.{body}".encode(),
            hashlib.sha256
        ).hexdigest()
        return f"{header}.{body}.{signature}"

    def _verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode a token."""
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token format")

        header, body, signature = parts

        # Verify signature
        expected = hmac.new(
            self._secret_key,
            f"{header}.{body}".encode(),
            hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected):
            raise ValueError("Invalid signature")

        # Decode payload
        payload = json.loads(base64.urlsafe_b64decode(body + "=="))

        # Check expiration
        if payload.get("exp", 0) < time.time():
            raise ValueError("Token expired")

        return payload

    def _hash_token(self, token: str) -> str:
        """Hash a token for storage."""
        return hashlib.sha256(token.encode()).hexdigest()

    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify a password against stored hash."""
        # Simple hash comparison (use bcrypt/argon2 in production)
        check_hash = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(check_hash, stored_hash)

    def get_principal(self, principal_id: str) -> Optional[Principal]:
        """Get a principal by ID."""
        return self._principals.get(principal_id)

    def get_principal_by_name(self, name: str) -> Optional[Principal]:
        """Get a principal by name."""
        for p in self._principals.values():
            if p.name == name:
                return p
        return None

    def list_principals(
        self,
        principal_type: Optional[str] = None,
    ) -> List[Principal]:
        """List principals."""
        principals = list(self._principals.values())
        if principal_type:
            principals = [p for p in principals if p.principal_type == principal_type]
        return principals

    def get_role(self, name: str) -> Optional[Role]:
        """Get a role by name."""
        return self._roles.get(name)

    def list_roles(self) -> List[Role]:
        """List all roles."""
        return list(self._roles.values())

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get audit log entries."""
        return self._audit_log[-limit:]

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        """Log an audit event."""
        entry = {
            "action": action,
            "details": details,
            "timestamp": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
        }
        self._audit_log.append(entry)
        # Keep last 10000 entries
        self._audit_log = self._audit_log[-10000:]

        _emit_bus_event(
            self.BUS_TOPICS["audit"],
            entry,
            kind="audit",
            actor=self.actor_id,
        )

    async def heartbeat(self) -> Dict[str, Any]:
        """Send A2A heartbeat."""
        now = time.time()
        status = {
            "agent": "security-module",
            "status": "healthy",
            "last_heartbeat": self._last_heartbeat,
            "principals_count": len(self._principals),
            "tokens_count": len(self._tokens),
            "policies_count": len(self._policies),
            "uptime_s": now - self._principals.get("__init__", Principal(
                principal_id="__init__",
                name="init",
                created_at=now
            )).created_at,
        }

        _emit_bus_event(
            self.BUS_TOPICS["heartbeat"],
            status,
            kind="heartbeat",
            actor=self.actor_id,
        )

        self._last_heartbeat = now
        return status

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "roles": {k: v.to_dict() for k, v in self._roles.items()
                     if k not in self.DEFAULT_ROLES},
            "principals": {k: v.to_dict() for k, v in self._principals.items()},
            "tokens": {k: v.to_dict() for k, v in self._tokens.items()},
            "api_keys": self._api_keys,
            "policies": {k: v.to_dict() for k, v in self._policies.items()},
        }

        state_file = self.state_dir / "security_state.json"
        with open(state_file, "w") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(state, f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "security_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for k, v in state.get("roles", {}).items():
                self._roles[k] = Role.from_dict(v)

            for k, v in state.get("principals", {}).items():
                self._principals[k] = Principal.from_dict(v)

            for k, v in state.get("tokens", {}).items():
                self._tokens[k] = AuthToken.from_dict(v)

            self._api_keys = state.get("api_keys", {})

            for k, v in state.get("policies", {}).items():
                self._policies[k] = AccessPolicy.from_dict(v)

        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for security module."""
    import argparse

    parser = argparse.ArgumentParser(description="Security Module (Step 241)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create-principal command
    create_parser = subparsers.add_parser("create-principal", help="Create a principal")
    create_parser.add_argument("name", help="Principal name")
    create_parser.add_argument("--type", "-t", default="service", help="Principal type")
    create_parser.add_argument("--roles", "-r", help="Comma-separated roles")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # create-api-key command
    key_parser = subparsers.add_parser("create-api-key", help="Create an API key")
    key_parser.add_argument("principal_id", help="Principal ID")
    key_parser.add_argument("--scopes", "-s", help="Comma-separated scopes")
    key_parser.add_argument("--expires", "-e", type=int, default=365, help="Expires in days")
    key_parser.add_argument("--json", action="store_true", help="JSON output")

    # authenticate command
    auth_parser = subparsers.add_parser("authenticate", help="Authenticate")
    auth_parser.add_argument("--method", "-m", default="api_key",
                            choices=["api_key", "bearer", "basic"])
    auth_parser.add_argument("--api-key", "-k", help="API key")
    auth_parser.add_argument("--token", "-t", help="Bearer token")
    auth_parser.add_argument("--user", "-u", help="Username for basic auth")
    auth_parser.add_argument("--password", "-p", help="Password for basic auth")
    auth_parser.add_argument("--json", action="store_true", help="JSON output")

    # assign-role command
    role_parser = subparsers.add_parser("assign-role", help="Assign role to principal")
    role_parser.add_argument("principal_id", help="Principal ID")
    role_parser.add_argument("role", help="Role name")

    # list-principals command
    list_parser = subparsers.add_parser("list-principals", help="List principals")
    list_parser.add_argument("--type", "-t", help="Filter by type")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # list-roles command
    roles_parser = subparsers.add_parser("list-roles", help="List roles")
    roles_parser.add_argument("--json", action="store_true", help="JSON output")

    # audit command
    audit_parser = subparsers.add_parser("audit", help="View audit log")
    audit_parser.add_argument("--limit", "-l", type=int, default=50, help="Limit entries")
    audit_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    security = SecurityModule()

    if args.command == "create-principal":
        roles = args.roles.split(",") if args.roles else []
        principal = security.create_principal(
            name=args.name,
            principal_type=args.type,
            roles=roles,
        )

        if args.json:
            print(json.dumps(principal.to_dict(), indent=2))
        else:
            print(f"Created principal: {principal.principal_id}")
            print(f"  Name: {principal.name}")
            print(f"  Type: {principal.principal_type}")
            print(f"  Roles: {', '.join(principal.roles) or 'none'}")

        return 0

    elif args.command == "create-api-key":
        scopes = args.scopes.split(",") if args.scopes else []
        try:
            raw_key, token = security.create_api_key(
                principal_id=args.principal_id,
                scopes=scopes,
                expires_in_days=args.expires,
            )

            if args.json:
                output = token.to_dict()
                output["raw_key"] = raw_key
                print(json.dumps(output, indent=2))
            else:
                print(f"Created API key: {token.token_id}")
                print(f"  Raw key: {raw_key}")
                print(f"  Expires: {datetime.fromtimestamp(token.expires_at).isoformat()}")
                print("  (Save this key - it won't be shown again)")

            return 0
        except ValueError as e:
            print(f"Error: {e}")
            return 1

    elif args.command == "authenticate":
        method = AuthMethod(args.method)
        credentials = {}

        if method == AuthMethod.API_KEY and args.api_key:
            credentials["api_key"] = args.api_key
        elif method == AuthMethod.BEARER and args.token:
            credentials["token"] = args.token
        elif method == AuthMethod.BASIC:
            credentials["username"] = args.user
            credentials["password"] = args.password

        result = asyncio.get_event_loop().run_until_complete(
            security.authenticate(method, credentials)
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Authentication successful")
                print(f"  Principal: {result.principal.name}")
                print(f"  Permissions: {', '.join(p.value for p in result.permissions)}")
            else:
                print(f"Authentication failed: {result.error}")

        return 0 if result.success else 1

    elif args.command == "assign-role":
        success = security.assign_role(args.principal_id, args.role)
        if success:
            print(f"Assigned role '{args.role}' to {args.principal_id}")
        else:
            print("Failed to assign role")
        return 0 if success else 1

    elif args.command == "list-principals":
        principals = security.list_principals(principal_type=args.type)

        if args.json:
            print(json.dumps([p.to_dict() for p in principals], indent=2))
        else:
            if not principals:
                print("No principals found")
            else:
                for p in principals:
                    print(f"{p.principal_id} ({p.name}) - {p.principal_type}")
                    print(f"  Roles: {', '.join(p.roles) or 'none'}")

        return 0

    elif args.command == "list-roles":
        roles = security.list_roles()

        if args.json:
            print(json.dumps([r.to_dict() for r in roles], indent=2))
        else:
            for r in roles:
                perms = ", ".join(p.value for p in r.permissions)
                print(f"{r.name}: {perms}")

        return 0

    elif args.command == "audit":
        entries = security.get_audit_log(limit=args.limit)

        if args.json:
            print(json.dumps(entries, indent=2))
        else:
            for entry in entries:
                ts = entry.get("iso", "")[:19]
                action = entry.get("action", "unknown")
                print(f"[{ts}] {action}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
