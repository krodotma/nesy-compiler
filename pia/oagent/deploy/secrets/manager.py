#!/usr/bin/env python3
"""
manager.py - Secret Manager (Step 211)

PBTSO Phase: SEQUESTER
A2A Integration: Manages secrets via deploy.secrets.inject

Provides:
- SecretType: Types of secrets
- SecretEntry: Secret entry definition
- SecretStore: Backend-agnostic secret storage
- VaultBackend: Vault backend type
- SecretManager: Secure secrets handling

Bus Topics:
- deploy.secrets.inject
- deploy.secrets.rotate
- deploy.secrets.revoke
- deploy.secrets.audit

Protocol: DKIN v30, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
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
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# ==============================================================================
# Bus Emission Helper
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
    actor: str = "secret-manager"
) -> str:
    """Emit an event to the Pluribus bus."""
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
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class SecretType(Enum):
    """Types of secrets."""
    API_KEY = "api_key"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    SSH_KEY = "ssh_key"
    DATABASE_URL = "database_url"
    ENCRYPTION_KEY = "encryption_key"
    OAUTH_SECRET = "oauth_secret"
    CUSTOM = "custom"


class VaultBackend(Enum):
    """Supported vault backends."""
    LOCAL = "local"  # Local encrypted file storage
    HASHICORP = "hashicorp"  # HashiCorp Vault
    AWS_SECRETS = "aws_secrets"  # AWS Secrets Manager
    GCP_SECRET = "gcp_secret"  # GCP Secret Manager
    AZURE_KEYVAULT = "azure_keyvault"  # Azure Key Vault
    KUBERNETES = "kubernetes"  # Kubernetes Secrets


@dataclass
class SecretEntry:
    """
    Secret entry definition.

    Attributes:
        secret_id: Unique secret identifier
        name: Human-readable secret name
        secret_type: Type of secret
        path: Secret path in the vault
        version: Current version
        environments: Environments where secret is available
        services: Services that can access this secret
        metadata: Additional metadata
        created_at: Timestamp when created
        updated_at: Timestamp when last updated
        expires_at: Optional expiration timestamp
        rotation_interval_days: Days between rotations
        last_rotated_at: Timestamp when last rotated
    """
    secret_id: str
    name: str
    secret_type: SecretType = SecretType.CUSTOM
    path: str = ""
    version: int = 1
    environments: List[str] = field(default_factory=lambda: ["dev", "staging", "prod"])
    services: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    rotation_interval_days: int = 90
    last_rotated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "secret_id": self.secret_id,
            "name": self.name,
            "secret_type": self.secret_type.value,
            "path": self.path,
            "version": self.version,
            "environments": self.environments,
            "services": self.services,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "rotation_interval_days": self.rotation_interval_days,
            "last_rotated_at": self.last_rotated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecretEntry":
        data = dict(data)
        if "secret_type" in data:
            data["secret_type"] = SecretType(data["secret_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class SecretStore:
    """
    Backend-agnostic secret storage configuration.

    Attributes:
        store_id: Store identifier
        backend: Vault backend type
        endpoint: Backend endpoint URL
        namespace: Namespace/path prefix
        auth_method: Authentication method
        options: Backend-specific options
    """
    store_id: str
    backend: VaultBackend = VaultBackend.LOCAL
    endpoint: str = ""
    namespace: str = "pluribus"
    auth_method: str = "token"
    options: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_id": self.store_id,
            "backend": self.backend.value,
            "endpoint": self.endpoint,
            "namespace": self.namespace,
            "auth_method": self.auth_method,
            "options": self.options,
        }


@dataclass
class InjectionResult:
    """Result of a secret injection operation."""
    success: bool
    secret_id: str
    target: str
    injected_as: str
    masked_value: str
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Secret Manager (Step 211)
# ==============================================================================

class SecretManager:
    """
    Secret Manager - secure secrets handling for deployments.

    PBTSO Phase: SEQUESTER

    Responsibilities:
    - Store and retrieve secrets securely
    - Inject secrets into deployment configurations
    - Rotate secrets on schedule
    - Audit secret access
    - Support multiple vault backends

    Example:
        >>> manager = SecretManager()
        >>> entry = manager.create_secret(
        ...     name="database_password",
        ...     value="super_secret_pass",
        ...     secret_type=SecretType.PASSWORD,
        ...     services=["api-service"]
        ... )
        >>> # Inject into environment
        >>> result = await manager.inject(
        ...     secret_id=entry.secret_id,
        ...     target="deployment",
        ...     inject_as="DB_PASSWORD"
        ... )
    """

    BUS_TOPICS = {
        "inject": "deploy.secrets.inject",
        "rotate": "deploy.secrets.rotate",
        "revoke": "deploy.secrets.revoke",
        "audit": "deploy.secrets.audit",
        "created": "deploy.secrets.created",
        "accessed": "deploy.secrets.accessed",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        encryption_key: Optional[str] = None,
        actor_id: str = "secret-manager",
    ):
        """
        Initialize the secret manager.

        Args:
            state_dir: Directory for state persistence
            encryption_key: Master encryption key (or generate new one)
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "secrets"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Initialize encryption
        self._fernet = self._init_encryption(encryption_key)

        # Storage
        self._secrets: Dict[str, SecretEntry] = {}
        self._values: Dict[str, bytes] = {}  # Encrypted values
        self._stores: Dict[str, SecretStore] = {}
        self._access_log: List[Dict[str, Any]] = []

        self._load_secrets()

    def _init_encryption(self, key: Optional[str]) -> Fernet:
        """Initialize encryption with a key."""
        key_file = self.state_dir / ".master_key"

        if key:
            # Derive key from provided string
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=b"pluribus_secret_salt",
                iterations=100000,
            )
            derived_key = base64.urlsafe_b64encode(kdf.derive(key.encode()))
            return Fernet(derived_key)

        if key_file.exists():
            # Load existing key
            return Fernet(key_file.read_bytes())

        # Generate new key
        new_key = Fernet.generate_key()
        key_file.write_bytes(new_key)
        os.chmod(key_file, 0o600)
        return Fernet(new_key)

    def create_secret(
        self,
        name: str,
        value: str,
        secret_type: SecretType = SecretType.CUSTOM,
        environments: Optional[List[str]] = None,
        services: Optional[List[str]] = None,
        rotation_interval_days: int = 90,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SecretEntry:
        """
        Create a new secret.

        Args:
            name: Human-readable secret name
            value: Secret value (will be encrypted)
            secret_type: Type of secret
            environments: Active environments
            services: Services that can access
            rotation_interval_days: Days between rotations
            metadata: Additional metadata

        Returns:
            Created SecretEntry
        """
        secret_id = f"secret-{uuid.uuid4().hex[:12]}"
        path = f"{secret_id}/{name}"

        entry = SecretEntry(
            secret_id=secret_id,
            name=name,
            secret_type=secret_type,
            path=path,
            environments=environments or ["dev", "staging", "prod"],
            services=services or [],
            rotation_interval_days=rotation_interval_days,
            metadata=metadata or {},
            last_rotated_at=time.time(),
        )

        # Encrypt and store value
        encrypted_value = self._fernet.encrypt(value.encode())
        self._values[secret_id] = encrypted_value

        self._secrets[secret_id] = entry
        self._save_secret(entry, encrypted_value)

        _emit_bus_event(
            self.BUS_TOPICS["created"],
            {
                "secret_id": secret_id,
                "name": name,
                "secret_type": secret_type.value,
                "environments": entry.environments,
                "services": entry.services,
            },
            actor=self.actor_id,
        )

        return entry

    def get_secret_value(
        self,
        secret_id: str,
        environment: str = "prod",
        service: Optional[str] = None,
        accessor: str = "system",
    ) -> Optional[str]:
        """
        Retrieve a secret value.

        Args:
            secret_id: Secret ID
            environment: Environment context
            service: Requesting service
            accessor: Who is accessing the secret

        Returns:
            Decrypted secret value or None if not found/authorized
        """
        entry = self._secrets.get(secret_id)
        if not entry:
            return None

        # Check environment authorization
        if environment not in entry.environments:
            self._log_access(secret_id, accessor, "denied", "environment_not_allowed")
            return None

        # Check service authorization
        if entry.services and service and service not in entry.services:
            self._log_access(secret_id, accessor, "denied", "service_not_allowed")
            return None

        # Check expiration
        if entry.expires_at and time.time() > entry.expires_at:
            self._log_access(secret_id, accessor, "denied", "expired")
            return None

        # Decrypt value
        encrypted = self._values.get(secret_id)
        if not encrypted:
            return None

        try:
            decrypted = self._fernet.decrypt(encrypted).decode()
            self._log_access(secret_id, accessor, "success", "retrieved")
            return decrypted
        except Exception:
            self._log_access(secret_id, accessor, "failed", "decryption_error")
            return None

    async def inject(
        self,
        secret_id: str,
        target: str,
        inject_as: str,
        environment: str = "prod",
        service: Optional[str] = None,
    ) -> InjectionResult:
        """
        Inject a secret into a deployment target.

        Args:
            secret_id: Secret ID to inject
            target: Target (e.g., "env", "file", "k8s-secret")
            inject_as: Variable/key name to inject as
            environment: Environment context
            service: Service context

        Returns:
            InjectionResult with status
        """
        value = self.get_secret_value(secret_id, environment, service, f"inject:{target}")

        if not value:
            return InjectionResult(
                success=False,
                secret_id=secret_id,
                target=target,
                injected_as=inject_as,
                masked_value="",
                error="Secret not found or access denied",
            )

        # Mask value for logging
        masked = self._mask_value(value)

        _emit_bus_event(
            self.BUS_TOPICS["inject"],
            {
                "secret_id": secret_id,
                "target": target,
                "injected_as": inject_as,
                "environment": environment,
                "service": service,
                "masked_value": masked,
            },
            actor=self.actor_id,
        )

        # Simulate injection (actual injection depends on target type)
        await asyncio.sleep(0.01)

        return InjectionResult(
            success=True,
            secret_id=secret_id,
            target=target,
            injected_as=inject_as,
            masked_value=masked,
        )

    async def rotate_secret(
        self,
        secret_id: str,
        new_value: str,
        reason: str = "scheduled",
    ) -> bool:
        """
        Rotate a secret to a new value.

        Args:
            secret_id: Secret ID to rotate
            new_value: New secret value
            reason: Rotation reason

        Returns:
            True if rotation successful
        """
        entry = self._secrets.get(secret_id)
        if not entry:
            return False

        old_version = entry.version

        # Encrypt new value
        encrypted_value = self._fernet.encrypt(new_value.encode())
        self._values[secret_id] = encrypted_value

        # Update entry
        entry.version += 1
        entry.updated_at = time.time()
        entry.last_rotated_at = time.time()

        self._save_secret(entry, encrypted_value)

        _emit_bus_event(
            self.BUS_TOPICS["rotate"],
            {
                "secret_id": secret_id,
                "name": entry.name,
                "old_version": old_version,
                "new_version": entry.version,
                "reason": reason,
            },
            actor=self.actor_id,
        )

        return True

    async def revoke_secret(self, secret_id: str, reason: str = "manual") -> bool:
        """
        Revoke (delete) a secret.

        Args:
            secret_id: Secret ID to revoke
            reason: Revocation reason

        Returns:
            True if revocation successful
        """
        entry = self._secrets.get(secret_id)
        if not entry:
            return False

        # Remove from memory
        del self._secrets[secret_id]
        if secret_id in self._values:
            del self._values[secret_id]

        # Remove from disk
        secret_file = self.state_dir / f"{secret_id}.json"
        value_file = self.state_dir / f"{secret_id}.enc"
        if secret_file.exists():
            secret_file.unlink()
        if value_file.exists():
            value_file.unlink()

        _emit_bus_event(
            self.BUS_TOPICS["revoke"],
            {
                "secret_id": secret_id,
                "name": entry.name,
                "reason": reason,
            },
            level="warn",
            actor=self.actor_id,
        )

        return True

    def check_rotation_needed(self) -> List[SecretEntry]:
        """Check which secrets need rotation."""
        needs_rotation = []
        now = time.time()

        for entry in self._secrets.values():
            days_since_rotation = (now - entry.last_rotated_at) / 86400
            if days_since_rotation >= entry.rotation_interval_days:
                needs_rotation.append(entry)

        return needs_rotation

    def generate_secret_value(
        self,
        secret_type: SecretType,
        length: int = 32,
    ) -> str:
        """
        Generate a secure secret value.

        Args:
            secret_type: Type of secret to generate
            length: Length of generated value

        Returns:
            Generated secret value
        """
        if secret_type == SecretType.API_KEY:
            return f"pk_{secrets.token_urlsafe(length)}"
        elif secret_type == SecretType.PASSWORD:
            # Strong password with mixed characters
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
            return "".join(secrets.choice(alphabet) for _ in range(length))
        elif secret_type == SecretType.TOKEN:
            return secrets.token_hex(length // 2)
        elif secret_type == SecretType.ENCRYPTION_KEY:
            return base64.b64encode(secrets.token_bytes(length)).decode()
        else:
            return secrets.token_urlsafe(length)

    def get_secret(self, secret_id: str) -> Optional[SecretEntry]:
        """Get secret metadata by ID."""
        return self._secrets.get(secret_id)

    def get_secret_by_name(self, name: str) -> Optional[SecretEntry]:
        """Get secret metadata by name."""
        for entry in self._secrets.values():
            if entry.name == name:
                return entry
        return None

    def list_secrets(
        self,
        environment: Optional[str] = None,
        service: Optional[str] = None,
        secret_type: Optional[SecretType] = None,
    ) -> List[SecretEntry]:
        """List secrets with optional filters."""
        secrets_list = list(self._secrets.values())

        if environment:
            secrets_list = [s for s in secrets_list if environment in s.environments]

        if service:
            secrets_list = [s for s in secrets_list if not s.services or service in s.services]

        if secret_type:
            secrets_list = [s for s in secrets_list if s.secret_type == secret_type]

        return secrets_list

    def get_access_log(self, secret_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get access log entries."""
        logs = self._access_log
        if secret_id:
            logs = [l for l in logs if l.get("secret_id") == secret_id]
        return logs[-limit:]

    def _mask_value(self, value: str) -> str:
        """Mask a secret value for logging."""
        if len(value) <= 8:
            return "***"
        return f"{value[:4]}...{value[-4:]}"

    def _log_access(
        self,
        secret_id: str,
        accessor: str,
        status: str,
        action: str,
    ) -> None:
        """Log secret access."""
        log_entry = {
            "secret_id": secret_id,
            "accessor": accessor,
            "status": status,
            "action": action,
            "timestamp": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
        }
        self._access_log.append(log_entry)

        _emit_bus_event(
            self.BUS_TOPICS["audit"],
            log_entry,
            kind="audit",
            level="info" if status == "success" else "warn",
            actor=self.actor_id,
        )

    def _save_secret(self, entry: SecretEntry, encrypted_value: bytes) -> None:
        """Save secret metadata and encrypted value."""
        # Save metadata
        meta_file = self.state_dir / f"{entry.secret_id}.json"
        with open(meta_file, "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

        # Save encrypted value
        value_file = self.state_dir / f"{entry.secret_id}.enc"
        value_file.write_bytes(encrypted_value)

    def _load_secrets(self) -> None:
        """Load secrets from disk."""
        for meta_file in self.state_dir.glob("*.json"):
            if meta_file.name.startswith("."):
                continue

            try:
                with open(meta_file, "r") as f:
                    data = json.load(f)

                entry = SecretEntry.from_dict(data)
                self._secrets[entry.secret_id] = entry

                # Load encrypted value
                value_file = self.state_dir / f"{entry.secret_id}.enc"
                if value_file.exists():
                    self._values[entry.secret_id] = value_file.read_bytes()

            except (json.JSONDecodeError, KeyError, IOError):
                continue


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for secret manager."""
    import argparse

    parser = argparse.ArgumentParser(description="Secret Manager (Step 211)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a secret")
    create_parser.add_argument("name", help="Secret name")
    create_parser.add_argument("--value", "-v", help="Secret value (or generate if not provided)")
    create_parser.add_argument("--type", "-t", default="custom",
                              choices=["api_key", "password", "token", "certificate", "ssh_key",
                                      "database_url", "encryption_key", "oauth_secret", "custom"])
    create_parser.add_argument("--services", "-s", help="Comma-separated services")
    create_parser.add_argument("--environments", "-e", default="dev,staging,prod", help="Comma-separated environments")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # get command
    get_parser = subparsers.add_parser("get", help="Get secret value")
    get_parser.add_argument("secret_id", help="Secret ID")
    get_parser.add_argument("--env", "-e", default="prod", help="Environment")
    get_parser.add_argument("--service", "-s", help="Service context")

    # inject command
    inject_parser = subparsers.add_parser("inject", help="Inject secret")
    inject_parser.add_argument("secret_id", help="Secret ID")
    inject_parser.add_argument("--target", "-t", default="env", help="Injection target")
    inject_parser.add_argument("--as", dest="inject_as", required=True, help="Variable name")
    inject_parser.add_argument("--env", "-e", default="prod", help="Environment")
    inject_parser.add_argument("--json", action="store_true", help="JSON output")

    # rotate command
    rotate_parser = subparsers.add_parser("rotate", help="Rotate a secret")
    rotate_parser.add_argument("secret_id", help="Secret ID")
    rotate_parser.add_argument("--value", "-v", help="New value (or generate)")
    rotate_parser.add_argument("--reason", "-r", default="manual", help="Rotation reason")

    # list command
    list_parser = subparsers.add_parser("list", help="List secrets")
    list_parser.add_argument("--env", "-e", help="Filter by environment")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--type", "-t", help="Filter by type")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # check-rotation command
    check_parser = subparsers.add_parser("check-rotation", help="Check secrets needing rotation")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    manager = SecretManager()

    if args.command == "create":
        secret_type = SecretType(args.type)
        value = args.value or manager.generate_secret_value(secret_type)
        services = args.services.split(",") if args.services else []
        environments = args.environments.split(",")

        entry = manager.create_secret(
            name=args.name,
            value=value,
            secret_type=secret_type,
            services=services,
            environments=environments,
        )

        if args.json:
            output = entry.to_dict()
            output["generated_value"] = value if not args.value else None
            print(json.dumps(output, indent=2))
        else:
            print(f"Created secret: {entry.secret_id}")
            print(f"  Name: {entry.name}")
            print(f"  Type: {entry.secret_type.value}")
            if not args.value:
                print(f"  Generated value: {value}")
            print(f"  Environments: {', '.join(entry.environments)}")

        return 0

    elif args.command == "get":
        value = manager.get_secret_value(
            args.secret_id,
            environment=args.env,
            service=args.service,
            accessor="cli",
        )

        if value:
            print(value)
            return 0
        else:
            print("Secret not found or access denied")
            return 1

    elif args.command == "inject":
        result = asyncio.get_event_loop().run_until_complete(
            manager.inject(
                secret_id=args.secret_id,
                target=args.target,
                inject_as=args.inject_as,
                environment=args.env,
            )
        )

        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            if result.success:
                print(f"Injected {args.secret_id} as {args.inject_as} to {args.target}")
                print(f"  Masked value: {result.masked_value}")
            else:
                print(f"Failed to inject: {result.error}")

        return 0 if result.success else 1

    elif args.command == "rotate":
        entry = manager.get_secret(args.secret_id)
        if not entry:
            print(f"Secret not found: {args.secret_id}")
            return 1

        new_value = args.value or manager.generate_secret_value(entry.secret_type)

        success = asyncio.get_event_loop().run_until_complete(
            manager.rotate_secret(args.secret_id, new_value, args.reason)
        )

        if success:
            print(f"Rotated secret: {args.secret_id}")
            print(f"  New version: {entry.version}")
            if not args.value:
                print(f"  New value: {new_value}")
        else:
            print("Rotation failed")

        return 0 if success else 1

    elif args.command == "list":
        secret_type = SecretType(args.type) if args.type else None
        secrets_list = manager.list_secrets(
            environment=args.env,
            service=args.service,
            secret_type=secret_type,
        )

        if args.json:
            print(json.dumps([s.to_dict() for s in secrets_list], indent=2))
        else:
            if not secrets_list:
                print("No secrets found")
            else:
                for s in secrets_list:
                    print(f"{s.secret_id} ({s.name}) - {s.secret_type.value} v{s.version}")

        return 0

    elif args.command == "check-rotation":
        needs_rotation = manager.check_rotation_needed()

        if args.json:
            print(json.dumps([s.to_dict() for s in needs_rotation], indent=2))
        else:
            if not needs_rotation:
                print("No secrets need rotation")
            else:
                print(f"Secrets needing rotation ({len(needs_rotation)}):")
                for s in needs_rotation:
                    days = (time.time() - s.last_rotated_at) / 86400
                    print(f"  {s.secret_id} ({s.name}) - {days:.0f} days old")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
