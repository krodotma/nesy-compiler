#!/usr/bin/env python3
"""
manager.py - CDN Manager (Step 219)

PBTSO Phase: ITERATE
A2A Integration: Manages CDN via deploy.cdn.configure

Provides:
- CDNProvider: CDN provider types
- CDNOrigin: CDN origin configuration
- CacheRule: Cache rule definition
- CDNDistribution: CDN distribution configuration
- CDNConfig: CDN configuration
- CDNManager: CDN configuration management

Bus Topics:
- deploy.cdn.configure
- deploy.cdn.invalidate
- deploy.cdn.distribution.create
- deploy.cdn.distribution.update

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
    actor: str = "cdn-manager"
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

class CDNProvider(Enum):
    """CDN provider types."""
    CLOUDFLARE = "cloudflare"
    CLOUDFRONT = "cloudfront"
    FASTLY = "fastly"
    AKAMAI = "akamai"
    AZURE_CDN = "azure_cdn"
    GCP_CDN = "gcp_cdn"
    LOCAL = "local"  # Simulation


class CacheStrategy(Enum):
    """Cache strategy types."""
    CACHE_ALL = "cache_all"
    CACHE_STATIC = "cache_static"
    BYPASS_CACHE = "bypass_cache"
    CACHE_BY_HEADER = "cache_by_header"
    CACHE_BY_COOKIE = "cache_by_cookie"


class OriginProtocol(Enum):
    """Origin protocol types."""
    HTTP_ONLY = "http_only"
    HTTPS_ONLY = "https_only"
    MATCH_VIEWER = "match_viewer"


@dataclass
class CDNOrigin:
    """
    CDN origin configuration.

    Attributes:
        origin_id: Unique origin identifier
        name: Origin name
        domain: Origin domain
        port: Origin port
        protocol: Origin protocol
        path: Origin path prefix
        timeout_s: Connection timeout
        headers: Custom origin headers
        ssl_protocols: Allowed SSL protocols
    """
    origin_id: str
    name: str
    domain: str
    port: int = 443
    protocol: OriginProtocol = OriginProtocol.HTTPS_ONLY
    path: str = ""
    timeout_s: int = 30
    headers: Dict[str, str] = field(default_factory=dict)
    ssl_protocols: List[str] = field(default_factory=lambda: ["TLSv1.2", "TLSv1.3"])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_id": self.origin_id,
            "name": self.name,
            "domain": self.domain,
            "port": self.port,
            "protocol": self.protocol.value,
            "path": self.path,
            "timeout_s": self.timeout_s,
            "headers": self.headers,
            "ssl_protocols": self.ssl_protocols,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CDNOrigin":
        data = dict(data)
        if "protocol" in data:
            data["protocol"] = OriginProtocol(data["protocol"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CacheRule:
    """
    Cache rule definition.

    Attributes:
        rule_id: Unique rule identifier
        name: Rule name
        path_pattern: Path pattern to match
        strategy: Cache strategy
        ttl_s: Cache TTL in seconds
        browser_ttl_s: Browser cache TTL
        edge_ttl_s: Edge cache TTL
        query_string_cache: Whether to include query strings in cache key
        compress: Whether to compress responses
        headers_to_cache: Headers to include in cache key
    """
    rule_id: str
    name: str
    path_pattern: str = "/*"
    strategy: CacheStrategy = CacheStrategy.CACHE_STATIC
    ttl_s: int = 86400  # 1 day
    browser_ttl_s: int = 3600  # 1 hour
    edge_ttl_s: int = 86400
    query_string_cache: bool = True
    compress: bool = True
    headers_to_cache: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "path_pattern": self.path_pattern,
            "strategy": self.strategy.value,
            "ttl_s": self.ttl_s,
            "browser_ttl_s": self.browser_ttl_s,
            "edge_ttl_s": self.edge_ttl_s,
            "query_string_cache": self.query_string_cache,
            "compress": self.compress,
            "headers_to_cache": self.headers_to_cache,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheRule":
        data = dict(data)
        if "strategy" in data:
            data["strategy"] = CacheStrategy(data["strategy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CDNDistribution:
    """
    CDN distribution configuration.

    Attributes:
        dist_id: Unique distribution identifier
        name: Distribution name
        domains: Custom domains
        origins: Origin configurations
        cache_rules: Cache rules
        default_origin_id: Default origin
        ssl_cert_id: SSL certificate ID
        waf_enabled: Whether WAF is enabled
        http2_enabled: Whether HTTP/2 is enabled
        ipv6_enabled: Whether IPv6 is enabled
        status: Distribution status
        cdn_domain: CDN assigned domain
        created_at: Creation timestamp
    """
    dist_id: str
    name: str
    domains: List[str] = field(default_factory=list)
    origins: List[CDNOrigin] = field(default_factory=list)
    cache_rules: List[CacheRule] = field(default_factory=list)
    default_origin_id: str = ""
    ssl_cert_id: str = ""
    waf_enabled: bool = False
    http2_enabled: bool = True
    ipv6_enabled: bool = True
    status: str = "deployed"
    cdn_domain: str = ""
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dist_id": self.dist_id,
            "name": self.name,
            "domains": self.domains,
            "origins": [o.to_dict() for o in self.origins],
            "cache_rules": [r.to_dict() for r in self.cache_rules],
            "default_origin_id": self.default_origin_id,
            "ssl_cert_id": self.ssl_cert_id,
            "waf_enabled": self.waf_enabled,
            "http2_enabled": self.http2_enabled,
            "ipv6_enabled": self.ipv6_enabled,
            "status": self.status,
            "cdn_domain": self.cdn_domain,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CDNDistribution":
        data = dict(data)
        if "origins" in data:
            data["origins"] = [CDNOrigin.from_dict(o) for o in data["origins"]]
        if "cache_rules" in data:
            data["cache_rules"] = [CacheRule.from_dict(r) for r in data["cache_rules"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class CDNConfig:
    """
    Global CDN configuration.

    Attributes:
        config_id: Configuration identifier
        provider: CDN provider
        default_ttl_s: Default cache TTL
        error_ttl_s: Error response TTL
        price_class: CDN price class
        logging_enabled: Whether logging is enabled
        log_bucket: S3 bucket for logs
        security_headers: Default security headers
    """
    config_id: str
    provider: CDNProvider = CDNProvider.LOCAL
    default_ttl_s: int = 86400
    error_ttl_s: int = 300
    price_class: str = "PriceClass_All"
    logging_enabled: bool = True
    log_bucket: str = ""
    security_headers: Dict[str, str] = field(default_factory=lambda: {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "SAMEORIGIN",
        "X-XSS-Protection": "1; mode=block",
    })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "provider": self.provider.value,
            "default_ttl_s": self.default_ttl_s,
            "error_ttl_s": self.error_ttl_s,
            "price_class": self.price_class,
            "logging_enabled": self.logging_enabled,
            "log_bucket": self.log_bucket,
            "security_headers": self.security_headers,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CDNConfig":
        data = dict(data)
        if "provider" in data:
            data["provider"] = CDNProvider(data["provider"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class InvalidationRequest:
    """CDN cache invalidation request."""
    invalidation_id: str
    dist_id: str
    paths: List[str]
    status: str = "pending"  # pending, in_progress, completed
    created_at: float = field(default_factory=time.time)
    completed_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# CDN Manager (Step 219)
# ==============================================================================

class CDNManager:
    """
    CDN Manager - manages CDN configuration.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Create and manage CDN distributions
    - Configure origin servers
    - Set up cache rules
    - Handle cache invalidation
    - Support multiple CDN providers

    Example:
        >>> cdn = CDNManager()
        >>> config = cdn.configure(provider=CDNProvider.CLOUDFLARE)
        >>> dist = cdn.create_distribution(
        ...     name="myapp-cdn",
        ...     domains=["cdn.example.com"],
        ...     origin_domain="origin.example.com"
        ... )
        >>> await cdn.invalidate(dist.dist_id, ["/assets/*"])
    """

    BUS_TOPICS = {
        "configure": "deploy.cdn.configure",
        "invalidate": "deploy.cdn.invalidate",
        "dist_create": "deploy.cdn.distribution.create",
        "dist_update": "deploy.cdn.distribution.update",
        "cache_purge": "deploy.cdn.cache.purge",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "cdn-manager",
    ):
        """
        Initialize the CDN manager.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "cdn"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._config: Optional[CDNConfig] = None
        self._distributions: Dict[str, CDNDistribution] = {}
        self._invalidations: Dict[str, InvalidationRequest] = {}

        self._load_state()

    def configure(
        self,
        provider: CDNProvider = CDNProvider.LOCAL,
        default_ttl_s: int = 86400,
        error_ttl_s: int = 300,
        logging_enabled: bool = True,
        security_headers: Optional[Dict[str, str]] = None,
    ) -> CDNConfig:
        """
        Configure CDN settings.

        Args:
            provider: CDN provider
            default_ttl_s: Default cache TTL
            error_ttl_s: Error response TTL
            logging_enabled: Enable logging
            security_headers: Security headers

        Returns:
            CDNConfig
        """
        config_id = f"cdn-{uuid.uuid4().hex[:12]}"

        config = CDNConfig(
            config_id=config_id,
            provider=provider,
            default_ttl_s=default_ttl_s,
            error_ttl_s=error_ttl_s,
            logging_enabled=logging_enabled,
        )

        if security_headers:
            config.security_headers = security_headers

        self._config = config
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["configure"],
            {
                "config_id": config_id,
                "provider": provider.value,
                "default_ttl_s": default_ttl_s,
            },
            actor=self.actor_id,
        )

        return config

    def create_distribution(
        self,
        name: str,
        domains: List[str],
        origin_domain: str,
        origin_port: int = 443,
        origin_protocol: OriginProtocol = OriginProtocol.HTTPS_ONLY,
        ssl_cert_id: str = "",
        waf_enabled: bool = False,
    ) -> CDNDistribution:
        """
        Create a CDN distribution.

        Args:
            name: Distribution name
            domains: Custom domains
            origin_domain: Origin server domain
            origin_port: Origin server port
            origin_protocol: Origin protocol
            ssl_cert_id: SSL certificate ID
            waf_enabled: Enable WAF

        Returns:
            Created CDNDistribution
        """
        dist_id = f"dist-{uuid.uuid4().hex[:12]}"
        origin_id = f"origin-{uuid.uuid4().hex[:8]}"

        # Create default origin
        origin = CDNOrigin(
            origin_id=origin_id,
            name=f"{name}-origin",
            domain=origin_domain,
            port=origin_port,
            protocol=origin_protocol,
        )

        # Create default cache rule
        static_rule = CacheRule(
            rule_id=f"rule-{uuid.uuid4().hex[:8]}",
            name="static-assets",
            path_pattern="/static/*",
            strategy=CacheStrategy.CACHE_ALL,
            ttl_s=604800,  # 7 days
        )

        dist = CDNDistribution(
            dist_id=dist_id,
            name=name,
            domains=domains,
            origins=[origin],
            cache_rules=[static_rule],
            default_origin_id=origin_id,
            ssl_cert_id=ssl_cert_id,
            waf_enabled=waf_enabled,
            cdn_domain=f"{dist_id}.cdn.local",
        )

        self._distributions[dist_id] = dist
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["dist_create"],
            {
                "dist_id": dist_id,
                "name": name,
                "domains": domains,
                "origin_domain": origin_domain,
                "cdn_domain": dist.cdn_domain,
            },
            actor=self.actor_id,
        )

        return dist

    def add_origin(
        self,
        dist_id: str,
        name: str,
        domain: str,
        port: int = 443,
        protocol: OriginProtocol = OriginProtocol.HTTPS_ONLY,
        path: str = "",
    ) -> Optional[CDNOrigin]:
        """Add an origin to a distribution."""
        dist = self._distributions.get(dist_id)
        if not dist:
            return None

        origin_id = f"origin-{uuid.uuid4().hex[:8]}"

        origin = CDNOrigin(
            origin_id=origin_id,
            name=name,
            domain=domain,
            port=port,
            protocol=protocol,
            path=path,
        )

        dist.origins.append(origin)
        self._save_state()

        return origin

    def add_cache_rule(
        self,
        dist_id: str,
        name: str,
        path_pattern: str,
        strategy: CacheStrategy = CacheStrategy.CACHE_STATIC,
        ttl_s: int = 86400,
        browser_ttl_s: int = 3600,
    ) -> Optional[CacheRule]:
        """Add a cache rule to a distribution."""
        dist = self._distributions.get(dist_id)
        if not dist:
            return None

        rule_id = f"rule-{uuid.uuid4().hex[:8]}"

        rule = CacheRule(
            rule_id=rule_id,
            name=name,
            path_pattern=path_pattern,
            strategy=strategy,
            ttl_s=ttl_s,
            browser_ttl_s=browser_ttl_s,
        )

        dist.cache_rules.append(rule)
        self._save_state()

        return rule

    async def invalidate(
        self,
        dist_id: str,
        paths: List[str],
    ) -> Optional[InvalidationRequest]:
        """
        Invalidate CDN cache for paths.

        Args:
            dist_id: Distribution ID
            paths: Paths to invalidate (supports wildcards)

        Returns:
            InvalidationRequest or None
        """
        dist = self._distributions.get(dist_id)
        if not dist:
            return None

        invalidation_id = f"inv-{uuid.uuid4().hex[:12]}"

        request = InvalidationRequest(
            invalidation_id=invalidation_id,
            dist_id=dist_id,
            paths=paths,
            status="in_progress",
        )

        self._invalidations[invalidation_id] = request

        _emit_bus_event(
            self.BUS_TOPICS["invalidate"],
            {
                "invalidation_id": invalidation_id,
                "dist_id": dist_id,
                "paths": paths,
            },
            actor=self.actor_id,
        )

        # Simulate invalidation
        await asyncio.sleep(0.2)

        request.status = "completed"
        request.completed_at = time.time()
        self._save_state()

        return request

    async def purge_all(self, dist_id: str) -> Optional[InvalidationRequest]:
        """Purge all cached content for a distribution."""
        return await self.invalidate(dist_id, ["/*"])

    def update_distribution(
        self,
        dist_id: str,
        domains: Optional[List[str]] = None,
        waf_enabled: Optional[bool] = None,
        http2_enabled: Optional[bool] = None,
        ssl_cert_id: Optional[str] = None,
    ) -> Optional[CDNDistribution]:
        """Update distribution settings."""
        dist = self._distributions.get(dist_id)
        if not dist:
            return None

        if domains is not None:
            dist.domains = domains
        if waf_enabled is not None:
            dist.waf_enabled = waf_enabled
        if http2_enabled is not None:
            dist.http2_enabled = http2_enabled
        if ssl_cert_id is not None:
            dist.ssl_cert_id = ssl_cert_id

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["dist_update"],
            {
                "dist_id": dist_id,
                "name": dist.name,
            },
            actor=self.actor_id,
        )

        return dist

    def get_cache_statistics(self, dist_id: str) -> Dict[str, Any]:
        """Get cache statistics for a distribution."""
        dist = self._distributions.get(dist_id)
        if not dist:
            return {}

        # Simulated statistics
        return {
            "dist_id": dist_id,
            "cache_hit_ratio": 0.85,
            "requests_total": 1000000,
            "bytes_served": 50 * 1024 * 1024 * 1024,  # 50 GB
            "edge_locations_active": 50,
            "bandwidth_peak_mbps": 500,
            "error_rate_pct": 0.1,
        }

    def get_config(self) -> Optional[CDNConfig]:
        """Get current CDN configuration."""
        return self._config

    def get_distribution(self, dist_id: str) -> Optional[CDNDistribution]:
        """Get a distribution by ID."""
        return self._distributions.get(dist_id)

    def list_distributions(self) -> List[CDNDistribution]:
        """List all distributions."""
        return list(self._distributions.values())

    def list_invalidations(
        self,
        dist_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[InvalidationRequest]:
        """List invalidation requests."""
        invs = list(self._invalidations.values())
        if dist_id:
            invs = [i for i in invs if i.dist_id == dist_id]
        return sorted(invs, key=lambda i: i.created_at, reverse=True)[:limit]

    def delete_distribution(self, dist_id: str) -> bool:
        """Delete a distribution."""
        if dist_id not in self._distributions:
            return False

        del self._distributions[dist_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "config": self._config.to_dict() if self._config else None,
            "distributions": {did: d.to_dict() for did, d in self._distributions.items()},
            "invalidations": {iid: i.to_dict() for iid, i in self._invalidations.items()},
        }
        state_file = self.state_dir / "cdn_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "cdn_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            if state.get("config"):
                self._config = CDNConfig.from_dict(state["config"])

            for did, data in state.get("distributions", {}).items():
                self._distributions[did] = CDNDistribution.from_dict(data)

            for iid, data in state.get("invalidations", {}).items():
                self._invalidations[iid] = InvalidationRequest(**data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for CDN manager."""
    import argparse

    parser = argparse.ArgumentParser(description="CDN Manager (Step 219)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # configure command
    config_parser = subparsers.add_parser("configure", help="Configure CDN")
    config_parser.add_argument("--provider", "-p", default="local",
                              choices=["local", "cloudflare", "cloudfront", "fastly"])
    config_parser.add_argument("--ttl", type=int, default=86400, help="Default TTL")
    config_parser.add_argument("--json", action="store_true", help="JSON output")

    # create command
    create_parser = subparsers.add_parser("create", help="Create distribution")
    create_parser.add_argument("name", help="Distribution name")
    create_parser.add_argument("--domain", "-d", required=True, help="Custom domain")
    create_parser.add_argument("--origin", "-o", required=True, help="Origin domain")
    create_parser.add_argument("--waf", action="store_true", help="Enable WAF")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # invalidate command
    invalidate_parser = subparsers.add_parser("invalidate", help="Invalidate cache")
    invalidate_parser.add_argument("dist_id", help="Distribution ID")
    invalidate_parser.add_argument("--paths", "-p", required=True, help="Comma-separated paths")
    invalidate_parser.add_argument("--json", action="store_true", help="JSON output")

    # rule command
    rule_parser = subparsers.add_parser("rule", help="Add cache rule")
    rule_parser.add_argument("dist_id", help="Distribution ID")
    rule_parser.add_argument("--name", "-n", required=True, help="Rule name")
    rule_parser.add_argument("--pattern", "-p", required=True, help="Path pattern")
    rule_parser.add_argument("--ttl", type=int, default=86400, help="Cache TTL")
    rule_parser.add_argument("--json", action="store_true", help="JSON output")

    # stats command
    stats_parser = subparsers.add_parser("stats", help="Get cache statistics")
    stats_parser.add_argument("dist_id", help="Distribution ID")
    stats_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List distributions")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    cdn = CDNManager()

    if args.command == "configure":
        config = cdn.configure(
            provider=CDNProvider(args.provider.upper() if args.provider != "local" else "LOCAL"),
            default_ttl_s=args.ttl,
        )

        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(f"Configured CDN: {config.config_id}")
            print(f"  Provider: {config.provider.value}")
            print(f"  Default TTL: {config.default_ttl_s}s")

        return 0

    elif args.command == "create":
        dist = cdn.create_distribution(
            name=args.name,
            domains=[args.domain],
            origin_domain=args.origin,
            waf_enabled=args.waf,
        )

        if args.json:
            print(json.dumps(dist.to_dict(), indent=2))
        else:
            print(f"Created distribution: {dist.dist_id}")
            print(f"  Name: {dist.name}")
            print(f"  Domains: {', '.join(dist.domains)}")
            print(f"  CDN Domain: {dist.cdn_domain}")
            print(f"  Origin: {dist.origins[0].domain}")

        return 0

    elif args.command == "invalidate":
        paths = [p.strip() for p in args.paths.split(",")]
        request = asyncio.get_event_loop().run_until_complete(
            cdn.invalidate(args.dist_id, paths)
        )

        if not request:
            print(f"Distribution not found: {args.dist_id}")
            return 1

        if args.json:
            print(json.dumps(request.to_dict(), indent=2))
        else:
            print(f"Invalidation: {request.invalidation_id}")
            print(f"  Status: {request.status}")
            print(f"  Paths: {', '.join(request.paths)}")

        return 0

    elif args.command == "rule":
        rule = cdn.add_cache_rule(
            dist_id=args.dist_id,
            name=args.name,
            path_pattern=args.pattern,
            ttl_s=args.ttl,
        )

        if not rule:
            print(f"Distribution not found: {args.dist_id}")
            return 1

        if args.json:
            print(json.dumps(rule.to_dict(), indent=2))
        else:
            print(f"Added cache rule: {rule.rule_id}")
            print(f"  Name: {rule.name}")
            print(f"  Pattern: {rule.path_pattern}")
            print(f"  TTL: {rule.ttl_s}s")

        return 0

    elif args.command == "stats":
        stats = cdn.get_cache_statistics(args.dist_id)

        if not stats:
            print(f"Distribution not found: {args.dist_id}")
            return 1

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print(f"Cache Statistics: {args.dist_id}")
            print(f"  Hit Ratio: {stats['cache_hit_ratio'] * 100:.1f}%")
            print(f"  Total Requests: {stats['requests_total']:,}")
            print(f"  Bytes Served: {stats['bytes_served'] / (1024**3):.1f} GB")
            print(f"  Error Rate: {stats['error_rate_pct']:.2f}%")

        return 0

    elif args.command == "list":
        dists = cdn.list_distributions()

        if args.json:
            print(json.dumps([d.to_dict() for d in dists], indent=2))
        else:
            if not dists:
                print("No distributions found")
            else:
                for d in dists:
                    print(f"{d.dist_id}: {d.name} [{d.status}]")
                    print(f"  CDN: {d.cdn_domain}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
