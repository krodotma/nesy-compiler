#!/usr/bin/env python3
"""
manager.py - SSL Manager (Step 216)

PBTSO Phase: SEQUESTER
A2A Integration: Manages SSL certificates via deploy.ssl.renew

Provides:
- CertificateType: Types of SSL certificates
- CertificateStatus: Certificate status enum
- Certificate: SSL certificate definition
- CertificateRequest: Certificate request definition
- SSLManager: SSL certificate handling

Bus Topics:
- deploy.ssl.renew
- deploy.ssl.issued
- deploy.ssl.expired
- deploy.ssl.revoked

Protocol: DKIN v30, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    actor: str = "ssl-manager"
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

class CertificateType(Enum):
    """Types of SSL certificates."""
    DV = "domain_validation"  # Domain Validation
    OV = "organization_validation"  # Organization Validation
    EV = "extended_validation"  # Extended Validation
    WILDCARD = "wildcard"
    SELF_SIGNED = "self_signed"
    LETS_ENCRYPT = "lets_encrypt"


class CertificateStatus(Enum):
    """Certificate status enum."""
    PENDING = "pending"
    ISSUED = "issued"
    ACTIVE = "active"
    EXPIRING = "expiring"  # Within 30 days
    EXPIRED = "expired"
    REVOKED = "revoked"
    FAILED = "failed"


class ChallengeType(Enum):
    """ACME challenge types."""
    HTTP_01 = "http-01"
    DNS_01 = "dns-01"
    TLS_ALPN_01 = "tls-alpn-01"


@dataclass
class CertificateRequest:
    """
    Certificate request definition.

    Attributes:
        request_id: Unique request identifier
        domains: List of domains for certificate
        cert_type: Certificate type
        challenge_type: ACME challenge type
        organization: Organization name (OV/EV)
        contact_email: Contact email
        key_size: RSA key size
        created_at: Request creation timestamp
    """
    request_id: str
    domains: List[str]
    cert_type: CertificateType = CertificateType.DV
    challenge_type: ChallengeType = ChallengeType.HTTP_01
    organization: str = ""
    contact_email: str = ""
    key_size: int = 2048
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "domains": self.domains,
            "cert_type": self.cert_type.value,
            "challenge_type": self.challenge_type.value,
            "organization": self.organization,
            "contact_email": self.contact_email,
            "key_size": self.key_size,
            "created_at": self.created_at,
        }


@dataclass
class Certificate:
    """
    SSL certificate definition.

    Attributes:
        cert_id: Unique certificate identifier
        domains: List of domains covered
        cert_type: Certificate type
        status: Certificate status
        issued_at: Issue timestamp
        expires_at: Expiration timestamp
        serial_number: Certificate serial number
        fingerprint: Certificate fingerprint (SHA-256)
        issuer: Certificate issuer
        subject: Certificate subject
        san: Subject Alternative Names
        key_id: Associated private key ID
        chain_certs: Intermediate certificate IDs
        auto_renew: Whether to auto-renew
        renewal_days: Days before expiry to renew
        metadata: Additional metadata
    """
    cert_id: str
    domains: List[str]
    cert_type: CertificateType = CertificateType.DV
    status: CertificateStatus = CertificateStatus.PENDING
    issued_at: float = 0.0
    expires_at: float = 0.0
    serial_number: str = ""
    fingerprint: str = ""
    issuer: str = ""
    subject: str = ""
    san: List[str] = field(default_factory=list)
    key_id: str = ""
    chain_certs: List[str] = field(default_factory=list)
    auto_renew: bool = True
    renewal_days: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cert_id": self.cert_id,
            "domains": self.domains,
            "cert_type": self.cert_type.value,
            "status": self.status.value,
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "serial_number": self.serial_number,
            "fingerprint": self.fingerprint,
            "issuer": self.issuer,
            "subject": self.subject,
            "san": self.san,
            "key_id": self.key_id,
            "chain_certs": self.chain_certs,
            "auto_renew": self.auto_renew,
            "renewal_days": self.renewal_days,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Certificate":
        data = dict(data)
        if "cert_type" in data:
            data["cert_type"] = CertificateType(data["cert_type"])
        if "status" in data:
            data["status"] = CertificateStatus(data["status"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @property
    def days_until_expiry(self) -> int:
        """Get days until certificate expires."""
        if self.expires_at == 0:
            return 0
        return int((self.expires_at - time.time()) / 86400)

    @property
    def is_expiring_soon(self) -> bool:
        """Check if certificate is expiring within renewal window."""
        return 0 < self.days_until_expiry <= self.renewal_days

    @property
    def is_expired(self) -> bool:
        """Check if certificate has expired."""
        return self.expires_at > 0 and time.time() > self.expires_at


# ==============================================================================
# SSL Manager (Step 216)
# ==============================================================================

class SSLManager:
    """
    SSL Manager - manages SSL certificates for deployments.

    PBTSO Phase: SEQUESTER

    Responsibilities:
    - Request and issue SSL certificates
    - Track certificate expiration
    - Auto-renew certificates before expiry
    - Manage private keys securely
    - Support multiple certificate authorities

    Example:
        >>> manager = SSLManager()
        >>> cert = await manager.request_certificate(
        ...     domains=["api.example.com", "www.example.com"],
        ...     cert_type=CertificateType.LETS_ENCRYPT,
        ... )
        >>> # Later, check and renew expiring certs
        >>> renewed = await manager.renew_expiring()
    """

    BUS_TOPICS = {
        "renew": "deploy.ssl.renew",
        "issued": "deploy.ssl.issued",
        "expired": "deploy.ssl.expired",
        "revoked": "deploy.ssl.revoked",
        "expiring_soon": "deploy.ssl.expiring_soon",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "ssl-manager",
    ):
        """
        Initialize the SSL manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "ssl"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / "certs").mkdir(exist_ok=True)
        (self.state_dir / "keys").mkdir(exist_ok=True)

        self.actor_id = actor_id

        self._certificates: Dict[str, Certificate] = {}
        self._requests: Dict[str, CertificateRequest] = {}
        self._keys: Dict[str, bytes] = {}  # Encrypted private keys

        self._load_state()

    async def request_certificate(
        self,
        domains: List[str],
        cert_type: CertificateType = CertificateType.LETS_ENCRYPT,
        challenge_type: ChallengeType = ChallengeType.HTTP_01,
        organization: str = "",
        contact_email: str = "",
        auto_renew: bool = True,
    ) -> Certificate:
        """
        Request a new SSL certificate.

        Args:
            domains: List of domains
            cert_type: Certificate type
            challenge_type: ACME challenge type
            organization: Organization name
            contact_email: Contact email
            auto_renew: Whether to auto-renew

        Returns:
            Created Certificate
        """
        cert_id = f"cert-{uuid.uuid4().hex[:12]}"
        request_id = f"req-{uuid.uuid4().hex[:12]}"

        request = CertificateRequest(
            request_id=request_id,
            domains=domains,
            cert_type=cert_type,
            challenge_type=challenge_type,
            organization=organization,
            contact_email=contact_email,
        )
        self._requests[request_id] = request

        cert = Certificate(
            cert_id=cert_id,
            domains=domains,
            cert_type=cert_type,
            status=CertificateStatus.PENDING,
            auto_renew=auto_renew,
            san=domains,
        )

        # Generate key
        key_id = await self._generate_key(cert_id)
        cert.key_id = key_id

        self._certificates[cert_id] = cert
        self._save_state()

        # Simulate certificate issuance
        if cert_type == CertificateType.SELF_SIGNED:
            await self._issue_self_signed(cert)
        else:
            await self._issue_certificate(cert, request)

        return cert

    async def _generate_key(self, cert_id: str) -> str:
        """Generate a private key for a certificate."""
        key_id = f"key-{uuid.uuid4().hex[:12]}"

        # Simulate key generation
        # In production, use cryptography library properly
        fake_key = base64.b64encode(os.urandom(256)).decode()
        self._keys[key_id] = fake_key.encode()

        # Save key (encrypted in production)
        key_file = self.state_dir / "keys" / f"{key_id}.key"
        key_file.write_text(fake_key)
        os.chmod(key_file, 0o600)

        return key_id

    async def _issue_self_signed(self, cert: Certificate) -> None:
        """Issue a self-signed certificate."""
        await asyncio.sleep(0.1)  # Simulate generation time

        now = time.time()
        cert.issued_at = now
        cert.expires_at = now + (365 * 86400)  # 1 year
        cert.status = CertificateStatus.ACTIVE
        cert.issuer = "Self-Signed CA"
        cert.subject = f"CN={cert.domains[0]}"
        cert.serial_number = uuid.uuid4().hex[:16].upper()
        cert.fingerprint = f"SHA256:{uuid.uuid4().hex[:64].upper()}"

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["issued"],
            {
                "cert_id": cert.cert_id,
                "domains": cert.domains,
                "cert_type": cert.cert_type.value,
                "expires_at": cert.expires_at,
            },
            actor=self.actor_id,
        )

    async def _issue_certificate(
        self,
        cert: Certificate,
        request: CertificateRequest,
    ) -> None:
        """Issue a certificate via CA (simulated)."""
        # Simulate ACME challenge
        await asyncio.sleep(0.2)

        now = time.time()
        validity_days = 90 if cert.cert_type == CertificateType.LETS_ENCRYPT else 365

        cert.issued_at = now
        cert.expires_at = now + (validity_days * 86400)
        cert.status = CertificateStatus.ACTIVE
        cert.issuer = self._get_issuer_name(cert.cert_type)
        cert.subject = f"CN={cert.domains[0]}"
        cert.serial_number = uuid.uuid4().hex[:16].upper()
        cert.fingerprint = f"SHA256:{uuid.uuid4().hex[:64].upper()}"

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["issued"],
            {
                "cert_id": cert.cert_id,
                "domains": cert.domains,
                "cert_type": cert.cert_type.value,
                "expires_at": cert.expires_at,
                "days_valid": validity_days,
            },
            actor=self.actor_id,
        )

    def _get_issuer_name(self, cert_type: CertificateType) -> str:
        """Get issuer name based on certificate type."""
        issuers = {
            CertificateType.LETS_ENCRYPT: "Let's Encrypt Authority X3",
            CertificateType.DV: "DigiCert DV TLS CA",
            CertificateType.OV: "DigiCert OV TLS CA",
            CertificateType.EV: "DigiCert EV TLS CA",
            CertificateType.WILDCARD: "DigiCert Wildcard CA",
            CertificateType.SELF_SIGNED: "Self-Signed CA",
        }
        return issuers.get(cert_type, "Unknown CA")

    async def renew_certificate(self, cert_id: str) -> Optional[Certificate]:
        """
        Renew a certificate.

        Args:
            cert_id: Certificate ID to renew

        Returns:
            Renewed Certificate or None
        """
        cert = self._certificates.get(cert_id)
        if not cert:
            return None

        if cert.status == CertificateStatus.REVOKED:
            return None

        _emit_bus_event(
            self.BUS_TOPICS["renew"],
            {
                "cert_id": cert_id,
                "domains": cert.domains,
                "old_expires_at": cert.expires_at,
            },
            actor=self.actor_id,
        )

        # Re-issue certificate
        request = CertificateRequest(
            request_id=f"req-{uuid.uuid4().hex[:12]}",
            domains=cert.domains,
            cert_type=cert.cert_type,
        )

        if cert.cert_type == CertificateType.SELF_SIGNED:
            await self._issue_self_signed(cert)
        else:
            await self._issue_certificate(cert, request)

        return cert

    async def renew_expiring(self) -> List[Certificate]:
        """
        Renew all certificates expiring soon.

        Returns:
            List of renewed certificates
        """
        renewed = []

        for cert in self._certificates.values():
            if cert.auto_renew and cert.is_expiring_soon:
                _emit_bus_event(
                    self.BUS_TOPICS["expiring_soon"],
                    {
                        "cert_id": cert.cert_id,
                        "domains": cert.domains,
                        "days_until_expiry": cert.days_until_expiry,
                    },
                    level="warn",
                    actor=self.actor_id,
                )

                result = await self.renew_certificate(cert.cert_id)
                if result:
                    renewed.append(result)

        return renewed

    async def revoke_certificate(self, cert_id: str, reason: str = "") -> bool:
        """
        Revoke a certificate.

        Args:
            cert_id: Certificate ID
            reason: Revocation reason

        Returns:
            True if revoked
        """
        cert = self._certificates.get(cert_id)
        if not cert:
            return False

        cert.status = CertificateStatus.REVOKED
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["revoked"],
            {
                "cert_id": cert_id,
                "domains": cert.domains,
                "reason": reason,
            },
            level="warn",
            actor=self.actor_id,
        )

        return True

    def check_expiration(self) -> Dict[str, List[Certificate]]:
        """
        Check certificate expiration status.

        Returns:
            Dict with 'expired', 'expiring', 'valid' lists
        """
        result = {
            "expired": [],
            "expiring": [],
            "valid": [],
        }

        for cert in self._certificates.values():
            if cert.status == CertificateStatus.REVOKED:
                continue

            if cert.is_expired:
                cert.status = CertificateStatus.EXPIRED
                result["expired"].append(cert)

                _emit_bus_event(
                    self.BUS_TOPICS["expired"],
                    {
                        "cert_id": cert.cert_id,
                        "domains": cert.domains,
                        "expired_at": cert.expires_at,
                    },
                    level="error",
                    actor=self.actor_id,
                )

            elif cert.is_expiring_soon:
                cert.status = CertificateStatus.EXPIRING
                result["expiring"].append(cert)

            else:
                result["valid"].append(cert)

        self._save_state()
        return result

    def get_certificate_for_domain(self, domain: str) -> Optional[Certificate]:
        """Get an active certificate covering a domain."""
        for cert in self._certificates.values():
            if cert.status == CertificateStatus.ACTIVE:
                if domain in cert.domains:
                    return cert
                # Check wildcard
                for cert_domain in cert.domains:
                    if cert_domain.startswith("*."):
                        base = cert_domain[2:]
                        if domain.endswith(base):
                            return cert
        return None

    def get_certificate(self, cert_id: str) -> Optional[Certificate]:
        """Get a certificate by ID."""
        return self._certificates.get(cert_id)

    def list_certificates(
        self,
        status: Optional[CertificateStatus] = None,
        domain: Optional[str] = None,
    ) -> List[Certificate]:
        """List certificates with optional filters."""
        certs = list(self._certificates.values())

        if status:
            certs = [c for c in certs if c.status == status]

        if domain:
            certs = [c for c in certs if domain in c.domains or
                    any(d.startswith("*.") and domain.endswith(d[2:]) for d in c.domains)]

        return certs

    def delete_certificate(self, cert_id: str) -> bool:
        """Delete a certificate."""
        cert = self._certificates.get(cert_id)
        if not cert:
            return False

        # Delete key
        if cert.key_id:
            key_file = self.state_dir / "keys" / f"{cert.key_id}.key"
            if key_file.exists():
                key_file.unlink()
            if cert.key_id in self._keys:
                del self._keys[cert.key_id]

        del self._certificates[cert_id]
        self._save_state()
        return True

    def get_certificate_chain(self, cert_id: str) -> List[str]:
        """Get the certificate chain for a certificate."""
        cert = self._certificates.get(cert_id)
        if not cert:
            return []

        # Return simulated chain
        return [
            f"-----BEGIN CERTIFICATE-----\n{cert.fingerprint}\n-----END CERTIFICATE-----",
            "-----BEGIN CERTIFICATE-----\nINTERMEDIATE_CERT\n-----END CERTIFICATE-----",
            "-----BEGIN CERTIFICATE-----\nROOT_CERT\n-----END CERTIFICATE-----",
        ]

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "certificates": {cid: c.to_dict() for cid, c in self._certificates.items()},
        }
        state_file = self.state_dir / "ssl_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "ssl_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for cid, data in state.get("certificates", {}).items():
                self._certificates[cid] = Certificate.from_dict(data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for SSL manager."""
    import argparse

    parser = argparse.ArgumentParser(description="SSL Manager (Step 216)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # request command
    request_parser = subparsers.add_parser("request", help="Request a certificate")
    request_parser.add_argument("domains", nargs="+", help="Domain names")
    request_parser.add_argument("--type", "-t", default="lets_encrypt",
                               choices=["lets_encrypt", "dv", "ov", "ev", "wildcard", "self_signed"])
    request_parser.add_argument("--no-auto-renew", action="store_true", help="Disable auto-renew")
    request_parser.add_argument("--json", action="store_true", help="JSON output")

    # renew command
    renew_parser = subparsers.add_parser("renew", help="Renew a certificate")
    renew_parser.add_argument("cert_id", nargs="?", help="Certificate ID (or renew all expiring)")
    renew_parser.add_argument("--json", action="store_true", help="JSON output")

    # revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke a certificate")
    revoke_parser.add_argument("cert_id", help="Certificate ID")
    revoke_parser.add_argument("--reason", "-r", default="", help="Revocation reason")

    # list command
    list_parser = subparsers.add_parser("list", help="List certificates")
    list_parser.add_argument("--status", "-s", choices=["active", "expiring", "expired", "revoked"])
    list_parser.add_argument("--domain", "-d", help="Filter by domain")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # check command
    check_parser = subparsers.add_parser("check", help="Check certificate expiration")
    check_parser.add_argument("--json", action="store_true", help="JSON output")

    # info command
    info_parser = subparsers.add_parser("info", help="Get certificate info")
    info_parser.add_argument("cert_id", help="Certificate ID")
    info_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    manager = SSLManager()

    if args.command == "request":
        type_map = {
            "lets_encrypt": CertificateType.LETS_ENCRYPT,
            "dv": CertificateType.DV,
            "ov": CertificateType.OV,
            "ev": CertificateType.EV,
            "wildcard": CertificateType.WILDCARD,
            "self_signed": CertificateType.SELF_SIGNED,
        }

        cert = asyncio.get_event_loop().run_until_complete(
            manager.request_certificate(
                domains=args.domains,
                cert_type=type_map[args.type],
                auto_renew=not args.no_auto_renew,
            )
        )

        if args.json:
            print(json.dumps(cert.to_dict(), indent=2))
        else:
            print(f"Certificate issued: {cert.cert_id}")
            print(f"  Domains: {', '.join(cert.domains)}")
            print(f"  Type: {cert.cert_type.value}")
            print(f"  Expires: {datetime.fromtimestamp(cert.expires_at)}")
            print(f"  Days valid: {cert.days_until_expiry}")

        return 0

    elif args.command == "renew":
        if args.cert_id:
            cert = asyncio.get_event_loop().run_until_complete(
                manager.renew_certificate(args.cert_id)
            )
            if cert:
                print(f"Renewed: {cert.cert_id}")
            else:
                print(f"Failed to renew: {args.cert_id}")
                return 1
        else:
            renewed = asyncio.get_event_loop().run_until_complete(
                manager.renew_expiring()
            )
            if args.json:
                print(json.dumps([c.to_dict() for c in renewed], indent=2))
            else:
                print(f"Renewed {len(renewed)} certificates")
                for c in renewed:
                    print(f"  {c.cert_id}: {', '.join(c.domains)}")

        return 0

    elif args.command == "revoke":
        success = asyncio.get_event_loop().run_until_complete(
            manager.revoke_certificate(args.cert_id, args.reason)
        )
        if success:
            print(f"Revoked: {args.cert_id}")
        else:
            print(f"Failed to revoke: {args.cert_id}")
            return 1

        return 0

    elif args.command == "list":
        status_map = {
            "active": CertificateStatus.ACTIVE,
            "expiring": CertificateStatus.EXPIRING,
            "expired": CertificateStatus.EXPIRED,
            "revoked": CertificateStatus.REVOKED,
        }
        status = status_map.get(args.status) if args.status else None

        certs = manager.list_certificates(status=status, domain=args.domain)

        if args.json:
            print(json.dumps([c.to_dict() for c in certs], indent=2))
        else:
            if not certs:
                print("No certificates found")
            else:
                for c in certs:
                    days = c.days_until_expiry
                    print(f"{c.cert_id}: {', '.join(c.domains)} [{c.status.value}] ({days} days)")

        return 0

    elif args.command == "check":
        result = manager.check_expiration()

        if args.json:
            print(json.dumps({
                "expired": [c.to_dict() for c in result["expired"]],
                "expiring": [c.to_dict() for c in result["expiring"]],
                "valid": [c.to_dict() for c in result["valid"]],
            }, indent=2))
        else:
            print(f"Expired: {len(result['expired'])}")
            for c in result["expired"]:
                print(f"  {c.cert_id}: {', '.join(c.domains)}")
            print(f"Expiring soon: {len(result['expiring'])}")
            for c in result["expiring"]:
                print(f"  {c.cert_id}: {', '.join(c.domains)} ({c.days_until_expiry} days)")
            print(f"Valid: {len(result['valid'])}")

        return 0 if not result["expired"] else 1

    elif args.command == "info":
        cert = manager.get_certificate(args.cert_id)
        if not cert:
            print(f"Certificate not found: {args.cert_id}")
            return 1

        if args.json:
            print(json.dumps(cert.to_dict(), indent=2))
        else:
            print(f"Certificate: {cert.cert_id}")
            print(f"  Domains: {', '.join(cert.domains)}")
            print(f"  Type: {cert.cert_type.value}")
            print(f"  Status: {cert.status.value}")
            print(f"  Issuer: {cert.issuer}")
            print(f"  Subject: {cert.subject}")
            print(f"  Serial: {cert.serial_number}")
            print(f"  Fingerprint: {cert.fingerprint}")
            print(f"  Issued: {datetime.fromtimestamp(cert.issued_at) if cert.issued_at else 'N/A'}")
            print(f"  Expires: {datetime.fromtimestamp(cert.expires_at) if cert.expires_at else 'N/A'}")
            print(f"  Days until expiry: {cert.days_until_expiry}")
            print(f"  Auto-renew: {cert.auto_renew}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
