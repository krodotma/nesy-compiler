#!/usr/bin/env python3
"""
service_mesh.py - Service Mesh (Step 218)

PBTSO Phase: ITERATE
A2A Integration: Manages service mesh via deploy.mesh.configure

Provides:
- MeshProvider: Service mesh provider types
- ServiceEntry: Service mesh service entry
- VirtualService: Virtual service definition
- DestinationRule: Destination rule definition
- MeshConfig: Mesh configuration
- ServiceMesh: Service mesh integration

Bus Topics:
- deploy.mesh.configure
- deploy.mesh.service.register
- deploy.mesh.traffic.route
- deploy.mesh.policy.apply

Protocol: DKIN v30, CITIZEN v2, HOLON v2
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
    actor: str = "service-mesh"
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

class MeshProvider(Enum):
    """Service mesh provider types."""
    ISTIO = "istio"
    LINKERD = "linkerd"
    CONSUL = "consul"
    AWS_APP_MESH = "aws_app_mesh"
    KUMA = "kuma"
    LOCAL = "local"  # Simulation


class TrafficPolicy(Enum):
    """Traffic policy types."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONN = "least_conn"
    RANDOM = "random"
    PASSTHROUGH = "passthrough"


class TLSMode(Enum):
    """TLS mode for mesh traffic."""
    DISABLE = "DISABLE"
    SIMPLE = "SIMPLE"
    MUTUAL = "MUTUAL"
    ISTIO_MUTUAL = "ISTIO_MUTUAL"


@dataclass
class ServiceEntry:
    """
    Service mesh service entry.

    Attributes:
        entry_id: Unique entry identifier
        name: Service name
        hosts: Service hostnames
        ports: Service ports
        location: MESH_EXTERNAL or MESH_INTERNAL
        resolution: DNS, STATIC, or NONE
        endpoints: Service endpoints
        namespace: Kubernetes namespace
        labels: Service labels
        created_at: Creation timestamp
    """
    entry_id: str
    name: str
    hosts: List[str] = field(default_factory=list)
    ports: List[Dict[str, Any]] = field(default_factory=list)
    location: str = "MESH_INTERNAL"
    resolution: str = "DNS"
    endpoints: List[Dict[str, Any]] = field(default_factory=list)
    namespace: str = "default"
    labels: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceEntry":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class HTTPRoute:
    """HTTP route definition."""
    match: List[Dict[str, Any]] = field(default_factory=list)  # Match conditions
    route: List[Dict[str, Any]] = field(default_factory=list)  # Destination routes
    timeout: str = "30s"
    retries: Dict[str, Any] = field(default_factory=dict)
    fault: Optional[Dict[str, Any]] = None  # Fault injection
    mirror: Optional[Dict[str, Any]] = None  # Traffic mirroring

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VirtualService:
    """
    Virtual service definition.

    Attributes:
        vs_id: Unique virtual service identifier
        name: Virtual service name
        hosts: Target hosts
        gateways: Associated gateways
        http_routes: HTTP routes
        tcp_routes: TCP routes
        namespace: Kubernetes namespace
        created_at: Creation timestamp
    """
    vs_id: str
    name: str
    hosts: List[str] = field(default_factory=list)
    gateways: List[str] = field(default_factory=list)
    http_routes: List[HTTPRoute] = field(default_factory=list)
    tcp_routes: List[Dict[str, Any]] = field(default_factory=list)
    namespace: str = "default"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vs_id": self.vs_id,
            "name": self.name,
            "hosts": self.hosts,
            "gateways": self.gateways,
            "http_routes": [r.to_dict() for r in self.http_routes],
            "tcp_routes": self.tcp_routes,
            "namespace": self.namespace,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VirtualService":
        data = dict(data)
        if "http_routes" in data:
            data["http_routes"] = [HTTPRoute(**r) for r in data["http_routes"]]
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class DestinationRule:
    """
    Destination rule definition.

    Attributes:
        dr_id: Unique destination rule identifier
        name: Destination rule name
        host: Target host
        traffic_policy: Traffic policy configuration
        subsets: Service subsets (versions)
        namespace: Kubernetes namespace
        created_at: Creation timestamp
    """
    dr_id: str
    name: str
    host: str = ""
    traffic_policy: Dict[str, Any] = field(default_factory=dict)
    subsets: List[Dict[str, Any]] = field(default_factory=list)
    namespace: str = "default"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DestinationRule":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MeshConfig:
    """
    Service mesh configuration.

    Attributes:
        config_id: Unique configuration identifier
        provider: Mesh provider
        namespace: Default namespace
        mtls_enabled: Whether mTLS is enabled
        tracing_enabled: Whether tracing is enabled
        metrics_enabled: Whether metrics are enabled
        access_log_enabled: Whether access logging is enabled
        default_timeout: Default request timeout
        retry_policy: Default retry policy
        circuit_breaker: Default circuit breaker config
        rate_limit: Default rate limit config
    """
    config_id: str
    provider: MeshProvider = MeshProvider.LOCAL
    namespace: str = "default"
    mtls_enabled: bool = True
    tracing_enabled: bool = True
    metrics_enabled: bool = True
    access_log_enabled: bool = True
    default_timeout: str = "30s"
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "attempts": 3,
        "per_try_timeout": "10s",
        "retry_on": "5xx,reset,connect-failure",
    })
    circuit_breaker: Dict[str, Any] = field(default_factory=lambda: {
        "consecutive_errors": 5,
        "interval": "30s",
        "base_ejection_time": "30s",
        "max_ejection_percent": 50,
    })
    rate_limit: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config_id": self.config_id,
            "provider": self.provider.value,
            "namespace": self.namespace,
            "mtls_enabled": self.mtls_enabled,
            "tracing_enabled": self.tracing_enabled,
            "metrics_enabled": self.metrics_enabled,
            "access_log_enabled": self.access_log_enabled,
            "default_timeout": self.default_timeout,
            "retry_policy": self.retry_policy,
            "circuit_breaker": self.circuit_breaker,
            "rate_limit": self.rate_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeshConfig":
        data = dict(data)
        if "provider" in data:
            data["provider"] = MeshProvider(data["provider"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ==============================================================================
# Service Mesh (Step 218)
# ==============================================================================

class ServiceMesh:
    """
    Service Mesh - manages service mesh integration.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Configure service mesh settings
    - Register services with the mesh
    - Create virtual services for traffic routing
    - Define destination rules for load balancing
    - Manage traffic policies

    Example:
        >>> mesh = ServiceMesh()
        >>> config = mesh.configure(
        ...     provider=MeshProvider.ISTIO,
        ...     mtls_enabled=True,
        ... )
        >>> entry = mesh.register_service(
        ...     name="api-service",
        ...     hosts=["api.default.svc.cluster.local"],
        ...     ports=[{"number": 8080, "protocol": "HTTP"}]
        ... )
        >>> vs = mesh.create_virtual_service(
        ...     name="api-routing",
        ...     hosts=["api.default.svc.cluster.local"],
        ... )
    """

    BUS_TOPICS = {
        "configure": "deploy.mesh.configure",
        "register": "deploy.mesh.service.register",
        "route": "deploy.mesh.traffic.route",
        "policy": "deploy.mesh.policy.apply",
        "vs_created": "deploy.mesh.virtualservice.created",
        "dr_created": "deploy.mesh.destinationrule.created",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "service-mesh",
    ):
        """
        Initialize the service mesh.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "mesh"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._config: Optional[MeshConfig] = None
        self._services: Dict[str, ServiceEntry] = {}
        self._virtual_services: Dict[str, VirtualService] = {}
        self._destination_rules: Dict[str, DestinationRule] = {}

        self._load_state()

    def configure(
        self,
        provider: MeshProvider = MeshProvider.LOCAL,
        mtls_enabled: bool = True,
        tracing_enabled: bool = True,
        metrics_enabled: bool = True,
        default_timeout: str = "30s",
        retry_policy: Optional[Dict[str, Any]] = None,
        circuit_breaker: Optional[Dict[str, Any]] = None,
    ) -> MeshConfig:
        """
        Configure the service mesh.

        Args:
            provider: Mesh provider
            mtls_enabled: Enable mutual TLS
            tracing_enabled: Enable distributed tracing
            metrics_enabled: Enable metrics collection
            default_timeout: Default request timeout
            retry_policy: Default retry policy
            circuit_breaker: Default circuit breaker config

        Returns:
            MeshConfig
        """
        config_id = f"mesh-{uuid.uuid4().hex[:12]}"

        config = MeshConfig(
            config_id=config_id,
            provider=provider,
            mtls_enabled=mtls_enabled,
            tracing_enabled=tracing_enabled,
            metrics_enabled=metrics_enabled,
            default_timeout=default_timeout,
        )

        if retry_policy:
            config.retry_policy = retry_policy
        if circuit_breaker:
            config.circuit_breaker = circuit_breaker

        self._config = config
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["configure"],
            {
                "config_id": config_id,
                "provider": provider.value,
                "mtls_enabled": mtls_enabled,
                "tracing_enabled": tracing_enabled,
            },
            actor=self.actor_id,
        )

        return config

    def register_service(
        self,
        name: str,
        hosts: List[str],
        ports: List[Dict[str, Any]],
        location: str = "MESH_INTERNAL",
        resolution: str = "DNS",
        endpoints: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
        labels: Optional[Dict[str, str]] = None,
    ) -> ServiceEntry:
        """
        Register a service with the mesh.

        Args:
            name: Service name
            hosts: Service hostnames
            ports: Service ports
            location: MESH_INTERNAL or MESH_EXTERNAL
            resolution: DNS, STATIC, or NONE
            endpoints: Service endpoints
            namespace: Kubernetes namespace
            labels: Service labels

        Returns:
            Created ServiceEntry
        """
        entry_id = f"svc-{uuid.uuid4().hex[:12]}"

        entry = ServiceEntry(
            entry_id=entry_id,
            name=name,
            hosts=hosts,
            ports=ports,
            location=location,
            resolution=resolution,
            endpoints=endpoints or [],
            namespace=namespace,
            labels=labels or {},
        )

        self._services[entry_id] = entry
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["register"],
            {
                "entry_id": entry_id,
                "name": name,
                "hosts": hosts,
                "namespace": namespace,
            },
            actor=self.actor_id,
        )

        return entry

    def create_virtual_service(
        self,
        name: str,
        hosts: List[str],
        gateways: Optional[List[str]] = None,
        namespace: str = "default",
    ) -> VirtualService:
        """
        Create a virtual service.

        Args:
            name: Virtual service name
            hosts: Target hosts
            gateways: Associated gateways
            namespace: Kubernetes namespace

        Returns:
            Created VirtualService
        """
        vs_id = f"vs-{uuid.uuid4().hex[:12]}"

        vs = VirtualService(
            vs_id=vs_id,
            name=name,
            hosts=hosts,
            gateways=gateways or ["mesh"],
            namespace=namespace,
        )

        self._virtual_services[vs_id] = vs
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["vs_created"],
            {
                "vs_id": vs_id,
                "name": name,
                "hosts": hosts,
            },
            actor=self.actor_id,
        )

        return vs

    def add_route(
        self,
        vs_id: str,
        destination_host: str,
        weight: int = 100,
        subset: str = "",
        match_conditions: Optional[List[Dict[str, Any]]] = None,
        timeout: str = "30s",
        retries: Optional[Dict[str, Any]] = None,
    ) -> Optional[VirtualService]:
        """
        Add a route to a virtual service.

        Args:
            vs_id: Virtual service ID
            destination_host: Destination host
            weight: Traffic weight
            subset: Destination subset
            match_conditions: Match conditions
            timeout: Request timeout
            retries: Retry configuration

        Returns:
            Updated VirtualService or None
        """
        vs = self._virtual_services.get(vs_id)
        if not vs:
            return None

        destination = {"host": destination_host, "weight": weight}
        if subset:
            destination["subset"] = subset

        route = HTTPRoute(
            match=match_conditions or [],
            route=[{"destination": destination}],
            timeout=timeout,
            retries=retries or self._config.retry_policy if self._config else {},
        )

        vs.http_routes.append(route)
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["route"],
            {
                "vs_id": vs_id,
                "destination_host": destination_host,
                "weight": weight,
            },
            actor=self.actor_id,
        )

        return vs

    def create_destination_rule(
        self,
        name: str,
        host: str,
        traffic_policy: Optional[Dict[str, Any]] = None,
        subsets: Optional[List[Dict[str, Any]]] = None,
        namespace: str = "default",
    ) -> DestinationRule:
        """
        Create a destination rule.

        Args:
            name: Destination rule name
            host: Target host
            traffic_policy: Traffic policy configuration
            subsets: Service subsets
            namespace: Kubernetes namespace

        Returns:
            Created DestinationRule
        """
        dr_id = f"dr-{uuid.uuid4().hex[:12]}"

        # Default traffic policy
        default_policy = {
            "connectionPool": {
                "tcp": {"maxConnections": 100},
                "http": {"h2UpgradePolicy": "UPGRADE"},
            },
            "loadBalancer": {"simple": "ROUND_ROBIN"},
        }

        if self._config and self._config.mtls_enabled:
            default_policy["tls"] = {"mode": "ISTIO_MUTUAL"}

        dr = DestinationRule(
            dr_id=dr_id,
            name=name,
            host=host,
            traffic_policy=traffic_policy or default_policy,
            subsets=subsets or [],
            namespace=namespace,
        )

        self._destination_rules[dr_id] = dr
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["dr_created"],
            {
                "dr_id": dr_id,
                "name": name,
                "host": host,
            },
            actor=self.actor_id,
        )

        return dr

    def add_subset(
        self,
        dr_id: str,
        name: str,
        labels: Dict[str, str],
        traffic_policy: Optional[Dict[str, Any]] = None,
    ) -> Optional[DestinationRule]:
        """
        Add a subset to a destination rule.

        Args:
            dr_id: Destination rule ID
            name: Subset name (e.g., "v1", "v2")
            labels: Selector labels
            traffic_policy: Subset-specific traffic policy

        Returns:
            Updated DestinationRule or None
        """
        dr = self._destination_rules.get(dr_id)
        if not dr:
            return None

        subset = {"name": name, "labels": labels}
        if traffic_policy:
            subset["trafficPolicy"] = traffic_policy

        dr.subsets.append(subset)
        self._save_state()

        return dr

    async def apply_canary_routing(
        self,
        service_name: str,
        stable_version: str,
        canary_version: str,
        canary_weight: int = 10,
    ) -> VirtualService:
        """
        Apply canary routing configuration.

        Args:
            service_name: Service name
            stable_version: Stable version label
            canary_version: Canary version label
            canary_weight: Percentage of traffic to canary

        Returns:
            Created VirtualService
        """
        host = f"{service_name}.default.svc.cluster.local"

        # Create destination rule with subsets
        dr = self.create_destination_rule(
            name=f"{service_name}-dr",
            host=host,
        )

        self.add_subset(dr.dr_id, "stable", {"version": stable_version})
        self.add_subset(dr.dr_id, "canary", {"version": canary_version})

        # Create virtual service with weighted routing
        vs = self.create_virtual_service(
            name=f"{service_name}-vs",
            hosts=[host],
        )

        stable_weight = 100 - canary_weight

        route = HTTPRoute(
            route=[
                {"destination": {"host": host, "subset": "stable"}, "weight": stable_weight},
                {"destination": {"host": host, "subset": "canary"}, "weight": canary_weight},
            ],
        )
        vs.http_routes.append(route)

        self._save_state()

        return vs

    async def apply_circuit_breaker(
        self,
        dr_id: str,
        consecutive_errors: int = 5,
        interval: str = "30s",
        base_ejection_time: str = "30s",
        max_ejection_percent: int = 50,
    ) -> Optional[DestinationRule]:
        """
        Apply circuit breaker to a destination rule.

        Args:
            dr_id: Destination rule ID
            consecutive_errors: Errors before ejection
            interval: Analysis interval
            base_ejection_time: Minimum ejection time
            max_ejection_percent: Max percentage to eject

        Returns:
            Updated DestinationRule or None
        """
        dr = self._destination_rules.get(dr_id)
        if not dr:
            return None

        dr.traffic_policy["outlierDetection"] = {
            "consecutiveErrors": consecutive_errors,
            "interval": interval,
            "baseEjectionTime": base_ejection_time,
            "maxEjectionPercent": max_ejection_percent,
        }

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["policy"],
            {
                "dr_id": dr_id,
                "policy": "circuit_breaker",
                "consecutive_errors": consecutive_errors,
            },
            actor=self.actor_id,
        )

        return dr

    def get_config(self) -> Optional[MeshConfig]:
        """Get current mesh configuration."""
        return self._config

    def get_service(self, entry_id: str) -> Optional[ServiceEntry]:
        """Get a service entry by ID."""
        return self._services.get(entry_id)

    def get_virtual_service(self, vs_id: str) -> Optional[VirtualService]:
        """Get a virtual service by ID."""
        return self._virtual_services.get(vs_id)

    def get_destination_rule(self, dr_id: str) -> Optional[DestinationRule]:
        """Get a destination rule by ID."""
        return self._destination_rules.get(dr_id)

    def list_services(self) -> List[ServiceEntry]:
        """List all registered services."""
        return list(self._services.values())

    def list_virtual_services(self) -> List[VirtualService]:
        """List all virtual services."""
        return list(self._virtual_services.values())

    def list_destination_rules(self) -> List[DestinationRule]:
        """List all destination rules."""
        return list(self._destination_rules.values())

    def delete_service(self, entry_id: str) -> bool:
        """Delete a service entry."""
        if entry_id not in self._services:
            return False
        del self._services[entry_id]
        self._save_state()
        return True

    def delete_virtual_service(self, vs_id: str) -> bool:
        """Delete a virtual service."""
        if vs_id not in self._virtual_services:
            return False
        del self._virtual_services[vs_id]
        self._save_state()
        return True

    def delete_destination_rule(self, dr_id: str) -> bool:
        """Delete a destination rule."""
        if dr_id not in self._destination_rules:
            return False
        del self._destination_rules[dr_id]
        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "config": self._config.to_dict() if self._config else None,
            "services": {sid: s.to_dict() for sid, s in self._services.items()},
            "virtual_services": {vid: v.to_dict() for vid, v in self._virtual_services.items()},
            "destination_rules": {did: d.to_dict() for did, d in self._destination_rules.items()},
        }
        state_file = self.state_dir / "mesh_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "mesh_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            if state.get("config"):
                self._config = MeshConfig.from_dict(state["config"])

            for sid, data in state.get("services", {}).items():
                self._services[sid] = ServiceEntry.from_dict(data)

            for vid, data in state.get("virtual_services", {}).items():
                self._virtual_services[vid] = VirtualService.from_dict(data)

            for did, data in state.get("destination_rules", {}).items():
                self._destination_rules[did] = DestinationRule.from_dict(data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for service mesh."""
    import argparse

    parser = argparse.ArgumentParser(description="Service Mesh (Step 218)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # configure command
    config_parser = subparsers.add_parser("configure", help="Configure mesh")
    config_parser.add_argument("--provider", "-p", default="local",
                              choices=["local", "istio", "linkerd", "consul"])
    config_parser.add_argument("--mtls", action="store_true", help="Enable mTLS")
    config_parser.add_argument("--tracing", action="store_true", help="Enable tracing")
    config_parser.add_argument("--json", action="store_true", help="JSON output")

    # register command
    register_parser = subparsers.add_parser("register", help="Register a service")
    register_parser.add_argument("name", help="Service name")
    register_parser.add_argument("--host", "-h", required=True, help="Service host")
    register_parser.add_argument("--port", "-p", type=int, default=80, help="Service port")
    register_parser.add_argument("--namespace", "-n", default="default", help="Namespace")
    register_parser.add_argument("--json", action="store_true", help="JSON output")

    # vs command (virtual service)
    vs_parser = subparsers.add_parser("vs", help="Create virtual service")
    vs_parser.add_argument("name", help="Virtual service name")
    vs_parser.add_argument("--host", "-h", required=True, help="Target host")
    vs_parser.add_argument("--json", action="store_true", help="JSON output")

    # dr command (destination rule)
    dr_parser = subparsers.add_parser("dr", help="Create destination rule")
    dr_parser.add_argument("name", help="Destination rule name")
    dr_parser.add_argument("--host", "-h", required=True, help="Target host")
    dr_parser.add_argument("--json", action="store_true", help="JSON output")

    # canary command
    canary_parser = subparsers.add_parser("canary", help="Set up canary routing")
    canary_parser.add_argument("service", help="Service name")
    canary_parser.add_argument("--stable", "-s", default="v1", help="Stable version")
    canary_parser.add_argument("--canary", "-c", default="v2", help="Canary version")
    canary_parser.add_argument("--weight", "-w", type=int, default=10, help="Canary weight %")
    canary_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List resources")
    list_parser.add_argument("type", choices=["services", "vs", "dr"])
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    mesh = ServiceMesh()

    if args.command == "configure":
        config = mesh.configure(
            provider=MeshProvider(args.provider.upper() if args.provider != "local" else "LOCAL"),
            mtls_enabled=args.mtls,
            tracing_enabled=args.tracing,
        )

        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(f"Configured mesh: {config.config_id}")
            print(f"  Provider: {config.provider.value}")
            print(f"  mTLS: {config.mtls_enabled}")
            print(f"  Tracing: {config.tracing_enabled}")

        return 0

    elif args.command == "register":
        entry = mesh.register_service(
            name=args.name,
            hosts=[args.host],
            ports=[{"number": args.port, "protocol": "HTTP"}],
            namespace=args.namespace,
        )

        if args.json:
            print(json.dumps(entry.to_dict(), indent=2))
        else:
            print(f"Registered service: {entry.entry_id}")
            print(f"  Name: {entry.name}")
            print(f"  Hosts: {', '.join(entry.hosts)}")

        return 0

    elif args.command == "vs":
        vs = mesh.create_virtual_service(
            name=args.name,
            hosts=[args.host],
        )

        if args.json:
            print(json.dumps(vs.to_dict(), indent=2))
        else:
            print(f"Created virtual service: {vs.vs_id}")
            print(f"  Name: {vs.name}")
            print(f"  Hosts: {', '.join(vs.hosts)}")

        return 0

    elif args.command == "dr":
        dr = mesh.create_destination_rule(
            name=args.name,
            host=args.host,
        )

        if args.json:
            print(json.dumps(dr.to_dict(), indent=2))
        else:
            print(f"Created destination rule: {dr.dr_id}")
            print(f"  Name: {dr.name}")
            print(f"  Host: {dr.host}")

        return 0

    elif args.command == "canary":
        vs = asyncio.get_event_loop().run_until_complete(
            mesh.apply_canary_routing(
                service_name=args.service,
                stable_version=args.stable,
                canary_version=args.canary,
                canary_weight=args.weight,
            )
        )

        if args.json:
            print(json.dumps(vs.to_dict(), indent=2))
        else:
            print(f"Applied canary routing for {args.service}")
            print(f"  Stable ({args.stable}): {100 - args.weight}%")
            print(f"  Canary ({args.canary}): {args.weight}%")

        return 0

    elif args.command == "list":
        if args.type == "services":
            items = mesh.list_services()
            if args.json:
                print(json.dumps([s.to_dict() for s in items], indent=2))
            else:
                for s in items:
                    print(f"{s.entry_id}: {s.name} ({', '.join(s.hosts)})")

        elif args.type == "vs":
            items = mesh.list_virtual_services()
            if args.json:
                print(json.dumps([v.to_dict() for v in items], indent=2))
            else:
                for v in items:
                    print(f"{v.vs_id}: {v.name} ({', '.join(v.hosts)})")

        elif args.type == "dr":
            items = mesh.list_destination_rules()
            if args.json:
                print(json.dumps([d.to_dict() for d in items], indent=2))
            else:
                for d in items:
                    print(f"{d.dr_id}: {d.name} ({d.host})")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
