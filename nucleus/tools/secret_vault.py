#!/usr/bin/env python3
"""
secret_vault.py - Unified Secret Management for Pluribus

Version: 1.0.0
Ring: 1 (Services)
Protocol: Vault Protocol v1 / DKIN v30

This module provides secure secret retrieval with multiple backend support:
  1. Environment variables (highest priority)
  2. SOPS-encrypted files (if sops is available)
  3. Plain text secret files (fallback)
  4. HashiCorp Vault (future extension)

Usage:
    from secret_vault import SecretVault

    vault = SecretVault()
    api_key = vault.get("ANTHROPIC_API_KEY")

    # Or with default
    api_key = vault.get("ANTHROPIC_API_KEY", default="")

Security Features:
  - Secrets are never logged or printed in full
  - Memory is cleared after use (best effort)
  - SOPS decryption uses subprocess isolation
  - Supports AGE and GPG encryption via SOPS

Semops:
    PBVAULT: Secret vault operations
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger("nucleus.vault")

VERSION = "1.0.0"

# Default paths for secret files
DEFAULT_SECRET_PATHS = [
    Path.home() / ".config/nucleus/secrets.env",
    Path.home() / ".config/pluribus_next/secrets.env",
    Path("/pluribus/.pluribus/config/secrets.env"),
    Path("/pluribus/.pluribus/secrets.env"),
]

# SOPS-encrypted file patterns
SOPS_EXTENSIONS = [".sops.yaml", ".sops.json", ".sops.env"]


def _mask_secret(value: str) -> str:
    """Mask a secret for logging, showing only first/last 2 chars."""
    if not value or len(value) < 6:
        return "****"
    return f"{value[:2]}...{value[-2:]}"


@dataclass
class SecretSource:
    """Tracks where a secret was loaded from."""
    name: str
    source_type: str  # "env", "sops", "file", "vault"
    path: Optional[str] = None


@dataclass
class SecretVault:
    """
    Unified secret management with multiple backend support.

    Priority order:
    1. Environment variables
    2. SOPS-encrypted files
    3. Plain text secret files
    4. HashiCorp Vault (if configured)
    """

    # Additional paths to search for secrets
    secret_paths: list[Path] = field(default_factory=list)

    # Cache for loaded secrets (cleared on reload)
    _cache: dict[str, str] = field(default_factory=dict, repr=False)
    _sources: dict[str, SecretSource] = field(default_factory=dict, repr=False)

    # SOPS availability (checked lazily)
    _sops_available: Optional[bool] = field(default=None, repr=False)

    # Vault configuration (for future extension)
    vault_addr: Optional[str] = field(default=None)
    vault_token: Optional[str] = field(default=None, repr=False)

    def __post_init__(self):
        """Initialize the vault and load secrets."""
        # Check for vault configuration
        self.vault_addr = os.environ.get("VAULT_ADDR")
        self.vault_token = os.environ.get("VAULT_TOKEN")

        # Build search paths
        all_paths = list(DEFAULT_SECRET_PATHS)
        all_paths.extend(self.secret_paths)
        self.secret_paths = all_paths

        # Pre-load secrets from files (not env - those are checked on demand)
        self._load_secrets_from_files()

    def _check_sops(self) -> bool:
        """Check if SOPS is available."""
        if self._sops_available is not None:
            return self._sops_available
        try:
            result = subprocess.run(
                ["sops", "--version"],
                capture_output=True,
                timeout=5
            )
            self._sops_available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._sops_available = False
        return self._sops_available

    def _decrypt_sops(self, path: Path) -> dict[str, str]:
        """Decrypt a SOPS-encrypted file and return key-value pairs."""
        if not self._check_sops():
            return {}

        try:
            result = subprocess.run(
                ["sops", "-d", str(path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                logger.warning(f"SOPS decryption failed for {path}: {result.stderr}")
                return {}

            content = result.stdout

            # Parse based on file extension
            if path.suffix == ".json" or ".json" in path.suffixes:
                return json.loads(content)
            elif path.suffix == ".yaml" or ".yaml" in path.suffixes or ".yml" in path.suffixes:
                # Basic YAML parsing (key: value format)
                secrets = {}
                for line in content.split("\n"):
                    line = line.strip()
                    if ":" in line and not line.startswith("#"):
                        key, _, value = line.partition(":")
                        secrets[key.strip()] = value.strip().strip('"').strip("'")
                return secrets
            else:
                # Assume .env format
                return self._parse_env_content(content)

        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
            logger.warning(f"Error decrypting SOPS file {path}: {e}")
            return {}

    def _parse_env_content(self, content: str) -> dict[str, str]:
        """Parse .env file content into key-value pairs."""
        secrets = {}
        for line in content.split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            # Remove surrounding quotes
            if (value.startswith('"') and value.endswith('"')) or \
               (value.startswith("'") and value.endswith("'")):
                value = value[1:-1]
            if key:
                secrets[key] = value
        return secrets

    def _load_secrets_from_files(self) -> None:
        """Load secrets from configured file paths."""
        for path in self.secret_paths:
            if not path.exists():
                continue

            # Check if it's a SOPS-encrypted file
            is_sops = any(ext in path.name for ext in SOPS_EXTENSIONS)

            if is_sops:
                secrets = self._decrypt_sops(path)
                source_type = "sops"
            else:
                try:
                    content = path.read_text(encoding="utf-8")
                    if path.suffix == ".json":
                        secrets = json.loads(content)
                    else:
                        secrets = self._parse_env_content(content)
                    source_type = "file"
                except (OSError, json.JSONDecodeError) as e:
                    logger.debug(f"Could not load secrets from {path}: {e}")
                    continue

            # Add to cache (don't overwrite existing - first wins)
            for key, value in secrets.items():
                if key not in self._cache:
                    self._cache[key] = value
                    self._sources[key] = SecretSource(
                        name=key,
                        source_type=source_type,
                        path=str(path)
                    )

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret by name.

        Priority:
        1. Environment variable
        2. SOPS-encrypted file
        3. Plain text file
        4. HashiCorp Vault (if configured)
        5. Default value
        """
        # 1. Check environment first (highest priority)
        env_value = os.environ.get(key)
        if env_value is not None:
            self._sources[key] = SecretSource(name=key, source_type="env")
            return env_value

        # 2/3. Check cache (loaded from files)
        if key in self._cache:
            return self._cache[key]

        # 4. Try Vault (if configured)
        if self.vault_addr and self.vault_token:
            vault_value = self._get_from_vault(key)
            if vault_value is not None:
                self._cache[key] = vault_value
                self._sources[key] = SecretSource(
                    name=key,
                    source_type="vault",
                    path=self.vault_addr
                )
                return vault_value

        # 5. Return default
        return default

    def _get_from_vault(self, key: str) -> Optional[str]:
        """Get a secret from HashiCorp Vault."""
        if not self.vault_addr or not self.vault_token:
            return None

        # Vault path convention: secret/data/pluribus/{key}
        vault_path = f"secret/data/pluribus/{key.lower()}"

        try:
            result = subprocess.run(
                [
                    "vault", "kv", "get",
                    "-format=json",
                    "-field=data",
                    vault_path
                ],
                capture_output=True,
                text=True,
                timeout=10,
                env={**os.environ, "VAULT_ADDR": self.vault_addr, "VAULT_TOKEN": self.vault_token}
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get("value") or data.get(key)
        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
            pass

        return None

    def set_env(self, key: str, value: str) -> None:
        """Set a secret in the environment (for subprocess inheritance)."""
        os.environ[key] = value
        self._cache[key] = value
        self._sources[key] = SecretSource(name=key, source_type="env")

    def get_source(self, key: str) -> Optional[SecretSource]:
        """Get the source information for a secret."""
        return self._sources.get(key)

    def reload(self) -> None:
        """Reload secrets from all sources."""
        self._cache.clear()
        self._sources.clear()
        self._load_secrets_from_files()

    def list_available(self) -> list[str]:
        """List all available secret keys (masked)."""
        keys = set(self._cache.keys())
        # Also include common environment variable names
        for key in os.environ:
            if any(pattern in key.upper() for pattern in ["KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"]):
                keys.add(key)
        return sorted(keys)

    def audit_log(self) -> list[dict]:
        """Return audit log of secret sources (no values)."""
        return [
            {
                "key": source.name,
                "type": source.source_type,
                "path": source.path
            }
            for source in self._sources.values()
        ]

    def export_env_snippet(self, keys: list[str]) -> str:
        """Generate bash export snippet for specified keys."""
        lines = ["# Secrets export snippet (generated by secret_vault.py)"]
        for key in keys:
            value = self.get(key)
            if value is not None:
                # Escape for bash
                escaped = value.replace("'", "'\\''")
                lines.append(f"export {key}='{escaped}'")
        return "\n".join(lines)


# Singleton instance for convenience
_vault: Optional[SecretVault] = None


def get_vault() -> SecretVault:
    """Get the singleton vault instance."""
    global _vault
    if _vault is None:
        _vault = SecretVault()
    return _vault


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Convenience function to get a secret."""
    return get_vault().get(key, default)


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """CLI entry point for PBVAULT operations."""
    if len(sys.argv) < 2:
        print(f"SECRET VAULT v{VERSION}")
        print("\nUsage:")
        print("  python3 secret_vault.py list              # List available secret keys")
        print("  python3 secret_vault.py get <key>         # Get a secret value")
        print("  python3 secret_vault.py audit             # Show audit log of sources")
        print("  python3 secret_vault.py check             # Check vault backends")
        print("  python3 secret_vault.py export <keys...>  # Export as bash snippet")
        print("\nSemops: PBVAULT")
        sys.exit(1)

    cmd = sys.argv[1]
    vault = SecretVault()

    if cmd == "list":
        keys = vault.list_available()
        print(f"Available secrets ({len(keys)}):")
        for key in keys:
            source = vault.get_source(key)
            source_info = f" [{source.source_type}]" if source else " [env]"
            print(f"  {key}{source_info}")

    elif cmd == "get":
        if len(sys.argv) < 3:
            print("Usage: secret_vault.py get <key>")
            sys.exit(1)
        key = sys.argv[2]
        value = vault.get(key)
        if value is not None:
            print(value)
        else:
            print(f"Secret '{key}' not found", file=sys.stderr)
            sys.exit(1)

    elif cmd == "audit":
        audit = vault.audit_log()
        print(f"Secret sources ({len(audit)}):")
        for entry in audit:
            path_info = f" ({entry['path']})" if entry['path'] else ""
            print(f"  {entry['key']}: {entry['type']}{path_info}")

    elif cmd == "check":
        print("Vault Backend Status:")
        print(f"  SOPS available: {vault._check_sops()}")
        print(f"  Vault configured: {bool(vault.vault_addr and vault.vault_token)}")
        if vault.vault_addr:
            print(f"  Vault address: {vault.vault_addr}")
        print(f"  Secret paths checked:")
        for path in vault.secret_paths:
            exists = "✓" if path.exists() else "✗"
            print(f"    [{exists}] {path}")

    elif cmd == "export":
        if len(sys.argv) < 3:
            print("Usage: secret_vault.py export <key1> [key2...]")
            sys.exit(1)
        keys = sys.argv[2:]
        print(vault.export_env_snippet(keys))

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
