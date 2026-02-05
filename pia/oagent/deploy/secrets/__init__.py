"""
Secrets management module for Deploy Agent.

Provides:
- SecretManager: Secure secrets handling (Step 211)
"""
from .manager import (
    SecretManager,
    SecretType,
    SecretEntry,
    SecretStore,
    VaultBackend,
)

__all__ = [
    "SecretManager",
    "SecretType",
    "SecretEntry",
    "SecretStore",
    "VaultBackend",
]
