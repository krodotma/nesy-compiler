"""
DNS management module for Deploy Agent.

Provides:
- DNSManager: DNS configuration management (Step 215)
"""
from .manager import (
    DNSManager,
    DNSRecordType,
    DNSRecord,
    DNSZone,
    DNSProvider,
)

__all__ = [
    "DNSManager",
    "DNSRecordType",
    "DNSRecord",
    "DNSZone",
    "DNSProvider",
]
