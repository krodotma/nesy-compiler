#!/usr/bin/env python3
"""
manager.py - DNS Manager (Step 215)

PBTSO Phase: ITERATE
A2A Integration: Manages DNS via deploy.dns.update

Provides:
- DNSRecordType: DNS record types
- DNSRecord: DNS record definition
- DNSZone: DNS zone definition
- DNSProvider: DNS provider enum
- DNSManager: DNS configuration management

Bus Topics:
- deploy.dns.update
- deploy.dns.create
- deploy.dns.delete
- deploy.dns.propagated

Protocol: DKIN v30, CITIZEN v2
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
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
    actor: str = "dns-manager"
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

class DNSRecordType(Enum):
    """DNS record types."""
    A = "A"
    AAAA = "AAAA"
    CNAME = "CNAME"
    MX = "MX"
    TXT = "TXT"
    NS = "NS"
    SRV = "SRV"
    CAA = "CAA"
    ALIAS = "ALIAS"


class DNSProvider(Enum):
    """Supported DNS providers."""
    LOCAL = "local"  # Local simulation
    ROUTE53 = "route53"
    CLOUDFLARE = "cloudflare"
    GOOGLE_DNS = "google_dns"
    AZURE_DNS = "azure_dns"
    DIGITALOCEAN = "digitalocean"


@dataclass
class DNSRecord:
    """
    DNS record definition.

    Attributes:
        record_id: Unique record identifier
        name: Record name (e.g., "api" for api.example.com)
        record_type: DNS record type
        value: Record value(s)
        ttl: Time-to-live in seconds
        priority: Priority for MX/SRV records
        weight: Weight for SRV records
        port: Port for SRV records
        zone_id: Parent zone ID
        proxied: Whether traffic is proxied (Cloudflare)
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    record_id: str
    name: str
    record_type: DNSRecordType = DNSRecordType.A
    value: List[str] = field(default_factory=list)
    ttl: int = 300
    priority: int = 0
    weight: int = 0
    port: int = 0
    zone_id: str = ""
    proxied: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "name": self.name,
            "record_type": self.record_type.value,
            "value": self.value,
            "ttl": self.ttl,
            "priority": self.priority,
            "weight": self.weight,
            "port": self.port,
            "zone_id": self.zone_id,
            "proxied": self.proxied,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNSRecord":
        data = dict(data)
        if "record_type" in data:
            data["record_type"] = DNSRecordType(data["record_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DNSZone:
    """
    DNS zone definition.

    Attributes:
        zone_id: Unique zone identifier
        domain: Zone domain (e.g., "example.com")
        provider: DNS provider
        provider_zone_id: Provider's zone ID
        nameservers: Zone nameservers
        records: Zone records
        metadata: Additional metadata
        created_at: Creation timestamp
    """
    zone_id: str
    domain: str
    provider: DNSProvider = DNSProvider.LOCAL
    provider_zone_id: str = ""
    nameservers: List[str] = field(default_factory=list)
    records: List[DNSRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "zone_id": self.zone_id,
            "domain": self.domain,
            "provider": self.provider.value,
            "provider_zone_id": self.provider_zone_id,
            "nameservers": self.nameservers,
            "records": [r.to_dict() for r in self.records],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNSZone":
        data = dict(data)
        if "provider" in data:
            data["provider"] = DNSProvider(data["provider"])
        if "records" in data:
            data["records"] = [DNSRecord.from_dict(r) for r in data["records"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# DNS Manager (Step 215)
# ==============================================================================

class DNSManager:
    """
    DNS Manager - manages DNS configuration for deployments.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Create and manage DNS zones
    - Create, update, and delete DNS records
    - Support multiple DNS providers
    - Track DNS propagation
    - Implement DNS-based routing

    Example:
        >>> manager = DNSManager()
        >>> zone = manager.create_zone("example.com")
        >>> record = await manager.create_record(
        ...     zone_id=zone.zone_id,
        ...     name="api",
        ...     record_type=DNSRecordType.A,
        ...     value=["192.168.1.100"]
        ... )
    """

    BUS_TOPICS = {
        "update": "deploy.dns.update",
        "create": "deploy.dns.create",
        "delete": "deploy.dns.delete",
        "propagated": "deploy.dns.propagated",
        "zone_created": "deploy.dns.zone.created",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "dns-manager",
    ):
        """
        Initialize the DNS manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "dns"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._zones: Dict[str, DNSZone] = {}
        self._records: Dict[str, DNSRecord] = {}
        self._propagation_status: Dict[str, Dict[str, Any]] = {}

        self._load_state()

    def create_zone(
        self,
        domain: str,
        provider: DNSProvider = DNSProvider.LOCAL,
        nameservers: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DNSZone:
        """
        Create a new DNS zone.

        Args:
            domain: Zone domain
            provider: DNS provider
            nameservers: Zone nameservers
            metadata: Additional metadata

        Returns:
            Created DNSZone
        """
        zone_id = f"zone-{uuid.uuid4().hex[:12]}"

        zone = DNSZone(
            zone_id=zone_id,
            domain=domain,
            provider=provider,
            nameservers=nameservers or [f"ns1.{domain}", f"ns2.{domain}"],
            metadata=metadata or {},
        )

        self._zones[zone_id] = zone
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["zone_created"],
            {
                "zone_id": zone_id,
                "domain": domain,
                "provider": provider.value,
            },
            actor=self.actor_id,
        )

        return zone

    async def create_record(
        self,
        zone_id: str,
        name: str,
        record_type: DNSRecordType,
        value: List[str],
        ttl: int = 300,
        priority: int = 0,
        weight: int = 0,
        port: int = 0,
        proxied: bool = False,
    ) -> Optional[DNSRecord]:
        """
        Create a DNS record.

        Args:
            zone_id: Zone ID
            name: Record name
            record_type: DNS record type
            value: Record value(s)
            ttl: Time-to-live
            priority: Priority (MX/SRV)
            weight: Weight (SRV)
            port: Port (SRV)
            proxied: Whether to proxy traffic

        Returns:
            Created DNSRecord or None
        """
        zone = self._zones.get(zone_id)
        if not zone:
            return None

        record_id = f"record-{uuid.uuid4().hex[:12]}"

        record = DNSRecord(
            record_id=record_id,
            name=name,
            record_type=record_type,
            value=value,
            ttl=ttl,
            priority=priority,
            weight=weight,
            port=port,
            zone_id=zone_id,
            proxied=proxied,
        )

        self._records[record_id] = record
        zone.records.append(record)
        self._save_state()

        # Simulate DNS API call
        await asyncio.sleep(0.1)

        _emit_bus_event(
            self.BUS_TOPICS["create"],
            {
                "record_id": record_id,
                "zone_id": zone_id,
                "name": name,
                "record_type": record_type.value,
                "value": value,
                "ttl": ttl,
            },
            actor=self.actor_id,
        )

        # Start propagation tracking
        self._track_propagation(record_id)

        return record

    async def update_record(
        self,
        record_id: str,
        value: Optional[List[str]] = None,
        ttl: Optional[int] = None,
        priority: Optional[int] = None,
    ) -> Optional[DNSRecord]:
        """
        Update a DNS record.

        Args:
            record_id: Record ID
            value: New value(s)
            ttl: New TTL
            priority: New priority

        Returns:
            Updated DNSRecord or None
        """
        record = self._records.get(record_id)
        if not record:
            return None

        old_value = record.value.copy()

        if value is not None:
            record.value = value
        if ttl is not None:
            record.ttl = ttl
        if priority is not None:
            record.priority = priority

        record.updated_at = time.time()
        self._save_state()

        # Simulate DNS API call
        await asyncio.sleep(0.1)

        _emit_bus_event(
            self.BUS_TOPICS["update"],
            {
                "record_id": record_id,
                "name": record.name,
                "old_value": old_value,
                "new_value": record.value,
            },
            actor=self.actor_id,
        )

        # Start propagation tracking
        self._track_propagation(record_id)

        return record

    async def delete_record(self, record_id: str) -> bool:
        """
        Delete a DNS record.

        Args:
            record_id: Record ID

        Returns:
            True if deleted
        """
        record = self._records.get(record_id)
        if not record:
            return False

        # Remove from zone
        zone = self._zones.get(record.zone_id)
        if zone:
            zone.records = [r for r in zone.records if r.record_id != record_id]

        del self._records[record_id]
        self._save_state()

        # Simulate DNS API call
        await asyncio.sleep(0.1)

        _emit_bus_event(
            self.BUS_TOPICS["delete"],
            {
                "record_id": record_id,
                "name": record.name,
                "zone_id": record.zone_id,
            },
            actor=self.actor_id,
        )

        return True

    async def set_a_record(
        self,
        zone_id: str,
        name: str,
        ip_addresses: List[str],
        ttl: int = 300,
    ) -> Optional[DNSRecord]:
        """Convenience method to set an A record."""
        return await self.create_record(
            zone_id=zone_id,
            name=name,
            record_type=DNSRecordType.A,
            value=ip_addresses,
            ttl=ttl,
        )

    async def set_cname_record(
        self,
        zone_id: str,
        name: str,
        target: str,
        ttl: int = 300,
    ) -> Optional[DNSRecord]:
        """Convenience method to set a CNAME record."""
        return await self.create_record(
            zone_id=zone_id,
            name=name,
            record_type=DNSRecordType.CNAME,
            value=[target],
            ttl=ttl,
        )

    async def set_txt_record(
        self,
        zone_id: str,
        name: str,
        value: str,
        ttl: int = 300,
    ) -> Optional[DNSRecord]:
        """Convenience method to set a TXT record."""
        return await self.create_record(
            zone_id=zone_id,
            name=name,
            record_type=DNSRecordType.TXT,
            value=[value],
            ttl=ttl,
        )

    async def switch_deployment(
        self,
        zone_id: str,
        name: str,
        new_target: str,
        record_type: DNSRecordType = DNSRecordType.A,
    ) -> Optional[DNSRecord]:
        """
        Switch a deployment by updating DNS.

        Args:
            zone_id: Zone ID
            name: Record name
            new_target: New target IP/CNAME
            record_type: Record type

        Returns:
            Updated or created record
        """
        # Find existing record
        zone = self._zones.get(zone_id)
        if not zone:
            return None

        existing = None
        for record in zone.records:
            if record.name == name and record.record_type == record_type:
                existing = record
                break

        if existing:
            return await self.update_record(existing.record_id, value=[new_target])
        else:
            return await self.create_record(
                zone_id=zone_id,
                name=name,
                record_type=record_type,
                value=[new_target],
            )

    def _track_propagation(self, record_id: str) -> None:
        """Start tracking DNS propagation."""
        self._propagation_status[record_id] = {
            "started_at": time.time(),
            "status": "propagating",
            "checks": [],
        }

    async def check_propagation(self, record_id: str) -> Dict[str, Any]:
        """
        Check DNS propagation status.

        Args:
            record_id: Record ID

        Returns:
            Propagation status
        """
        record = self._records.get(record_id)
        if not record:
            return {"status": "unknown", "record_id": record_id}

        zone = self._zones.get(record.zone_id)
        if not zone:
            return {"status": "unknown", "record_id": record_id}

        fqdn = f"{record.name}.{zone.domain}" if record.name != "@" else zone.domain

        # Simulate propagation check
        await asyncio.sleep(0.05)

        # Simple simulation: after TTL seconds, consider propagated
        status_info = self._propagation_status.get(record_id, {})
        started_at = status_info.get("started_at", time.time())

        if time.time() - started_at > record.ttl:
            status = "propagated"
        else:
            status = "propagating"

        result = {
            "record_id": record_id,
            "fqdn": fqdn,
            "status": status,
            "elapsed_s": time.time() - started_at,
            "ttl": record.ttl,
        }

        if status == "propagated":
            _emit_bus_event(
                self.BUS_TOPICS["propagated"],
                result,
                actor=self.actor_id,
            )

        return result

    def resolve(self, fqdn: str) -> Optional[List[str]]:
        """
        Resolve a FQDN (local simulation).

        Args:
            fqdn: Fully qualified domain name

        Returns:
            List of resolved values or None
        """
        # Parse domain and name
        parts = fqdn.split(".")
        if len(parts) < 2:
            return None

        # Try to find matching zone
        for zone in self._zones.values():
            if fqdn.endswith(zone.domain):
                name = fqdn[:-len(zone.domain) - 1] if fqdn != zone.domain else "@"

                for record in zone.records:
                    if record.name == name:
                        return record.value

        return None

    def get_zone(self, zone_id: str) -> Optional[DNSZone]:
        """Get a zone by ID."""
        return self._zones.get(zone_id)

    def get_zone_by_domain(self, domain: str) -> Optional[DNSZone]:
        """Get a zone by domain."""
        for zone in self._zones.values():
            if zone.domain == domain:
                return zone
        return None

    def get_record(self, record_id: str) -> Optional[DNSRecord]:
        """Get a record by ID."""
        return self._records.get(record_id)

    def list_zones(self) -> List[DNSZone]:
        """List all zones."""
        return list(self._zones.values())

    def list_records(self, zone_id: Optional[str] = None) -> List[DNSRecord]:
        """List records."""
        if zone_id:
            zone = self._zones.get(zone_id)
            return zone.records if zone else []
        return list(self._records.values())

    def delete_zone(self, zone_id: str) -> bool:
        """Delete a zone and all its records."""
        zone = self._zones.get(zone_id)
        if not zone:
            return False

        # Delete all records
        for record in zone.records:
            if record.record_id in self._records:
                del self._records[record.record_id]

        del self._zones[zone_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "zones": {zid: z.to_dict() for zid, z in self._zones.items()},
            "records": {rid: r.to_dict() for rid, r in self._records.items()},
        }
        state_file = self.state_dir / "dns_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "dns_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for zid, data in state.get("zones", {}).items():
                self._zones[zid] = DNSZone.from_dict(data)

            for rid, data in state.get("records", {}).items():
                self._records[rid] = DNSRecord.from_dict(data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for DNS manager."""
    import argparse

    parser = argparse.ArgumentParser(description="DNS Manager (Step 215)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # zone command
    zone_parser = subparsers.add_parser("zone", help="Create a DNS zone")
    zone_parser.add_argument("domain", help="Zone domain")
    zone_parser.add_argument("--provider", "-p", default="local",
                            choices=["local", "route53", "cloudflare"])
    zone_parser.add_argument("--json", action="store_true", help="JSON output")

    # record command
    record_parser = subparsers.add_parser("record", help="Create a DNS record")
    record_parser.add_argument("zone_id", help="Zone ID")
    record_parser.add_argument("--name", "-n", required=True, help="Record name")
    record_parser.add_argument("--type", "-t", default="A",
                              choices=["A", "AAAA", "CNAME", "TXT", "MX", "SRV"])
    record_parser.add_argument("--value", "-v", required=True, help="Record value")
    record_parser.add_argument("--ttl", type=int, default=300, help="TTL")
    record_parser.add_argument("--json", action="store_true", help="JSON output")

    # update command
    update_parser = subparsers.add_parser("update", help="Update a DNS record")
    update_parser.add_argument("record_id", help="Record ID")
    update_parser.add_argument("--value", "-v", help="New value")
    update_parser.add_argument("--ttl", type=int, help="New TTL")

    # delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a DNS record")
    delete_parser.add_argument("record_id", help="Record ID")

    # list command
    list_parser = subparsers.add_parser("list", help="List zones or records")
    list_parser.add_argument("--zone", "-z", help="Zone ID for records")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    # resolve command
    resolve_parser = subparsers.add_parser("resolve", help="Resolve a FQDN")
    resolve_parser.add_argument("fqdn", help="FQDN to resolve")

    args = parser.parse_args()
    manager = DNSManager()

    if args.command == "zone":
        zone = manager.create_zone(
            domain=args.domain,
            provider=DNSProvider(args.provider),
        )

        if args.json:
            print(json.dumps(zone.to_dict(), indent=2))
        else:
            print(f"Created zone: {zone.zone_id}")
            print(f"  Domain: {zone.domain}")
            print(f"  Provider: {zone.provider.value}")
            print(f"  Nameservers: {', '.join(zone.nameservers)}")

        return 0

    elif args.command == "record":
        record = asyncio.get_event_loop().run_until_complete(
            manager.create_record(
                zone_id=args.zone_id,
                name=args.name,
                record_type=DNSRecordType(args.type),
                value=[args.value],
                ttl=args.ttl,
            )
        )

        if not record:
            print(f"Zone not found: {args.zone_id}")
            return 1

        if args.json:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            print(f"Created record: {record.record_id}")
            print(f"  Name: {record.name}")
            print(f"  Type: {record.record_type.value}")
            print(f"  Value: {record.value}")
            print(f"  TTL: {record.ttl}")

        return 0

    elif args.command == "update":
        value = [args.value] if args.value else None
        record = asyncio.get_event_loop().run_until_complete(
            manager.update_record(
                record_id=args.record_id,
                value=value,
                ttl=args.ttl,
            )
        )

        if record:
            print(f"Updated record: {record.record_id}")
        else:
            print(f"Record not found: {args.record_id}")
            return 1

        return 0

    elif args.command == "delete":
        success = asyncio.get_event_loop().run_until_complete(
            manager.delete_record(args.record_id)
        )

        if success:
            print(f"Deleted record: {args.record_id}")
        else:
            print(f"Record not found: {args.record_id}")
            return 1

        return 0

    elif args.command == "list":
        if args.zone:
            items = manager.list_records(args.zone)
            if args.json:
                print(json.dumps([r.to_dict() for r in items], indent=2))
            else:
                for r in items:
                    print(f"{r.record_id}: {r.name} {r.record_type.value} -> {r.value}")
        else:
            items = manager.list_zones()
            if args.json:
                print(json.dumps([z.to_dict() for z in items], indent=2))
            else:
                for z in items:
                    print(f"{z.zone_id}: {z.domain} ({z.provider.value})")

        return 0

    elif args.command == "resolve":
        result = manager.resolve(args.fqdn)
        if result:
            print(f"{args.fqdn} -> {', '.join(result)}")
        else:
            print(f"Could not resolve: {args.fqdn}")
            return 1

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
