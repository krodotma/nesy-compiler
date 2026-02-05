#!/usr/bin/env python3
"""
balancer.py - Load Balancer (Step 217)

PBTSO Phase: ITERATE
A2A Integration: Configures load balancing via deploy.lb.configure

Provides:
- LoadBalancerType: Types of load balancers
- Backend: Load balancer backend definition
- HealthCheck: Health check configuration
- LoadBalancerConfig: Load balancer configuration
- LoadBalancerState: Load balancer state
- LoadBalancer: Load balancing configuration

Bus Topics:
- deploy.lb.configure
- deploy.lb.backend.add
- deploy.lb.backend.remove
- deploy.lb.health

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
    actor: str = "load-balancer"
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

class LoadBalancerType(Enum):
    """Types of load balancers."""
    APPLICATION = "application"  # Layer 7
    NETWORK = "network"  # Layer 4
    CLASSIC = "classic"
    INTERNAL = "internal"
    GLOBAL = "global"


class Algorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    IP_HASH = "ip_hash"
    WEIGHTED = "weighted"
    RANDOM = "random"
    LEAST_TIME = "least_time"


class Protocol(Enum):
    """Supported protocols."""
    HTTP = "HTTP"
    HTTPS = "HTTPS"
    TCP = "TCP"
    UDP = "UDP"
    GRPC = "gRPC"


@dataclass
class HealthCheck:
    """
    Health check configuration for backends.

    Attributes:
        check_id: Unique check identifier
        protocol: Health check protocol
        path: HTTP path for health check
        port: Health check port
        interval_s: Check interval in seconds
        timeout_s: Check timeout
        healthy_threshold: Consecutive successes to be healthy
        unhealthy_threshold: Consecutive failures to be unhealthy
        matcher: Expected response (status codes, body)
    """
    check_id: str = ""
    protocol: Protocol = Protocol.HTTP
    path: str = "/health"
    port: int = 80
    interval_s: int = 30
    timeout_s: int = 5
    healthy_threshold: int = 2
    unhealthy_threshold: int = 3
    matcher: Dict[str, Any] = field(default_factory=lambda: {"status_codes": [200]})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id,
            "protocol": self.protocol.value,
            "path": self.path,
            "port": self.port,
            "interval_s": self.interval_s,
            "timeout_s": self.timeout_s,
            "healthy_threshold": self.healthy_threshold,
            "unhealthy_threshold": self.unhealthy_threshold,
            "matcher": self.matcher,
        }


@dataclass
class Backend:
    """
    Load balancer backend definition.

    Attributes:
        backend_id: Unique backend identifier
        name: Backend name
        address: Backend address (IP or hostname)
        port: Backend port
        weight: Traffic weight
        max_connections: Maximum connections
        health_status: Current health status
        active_connections: Current active connections
        metadata: Additional metadata
    """
    backend_id: str
    name: str
    address: str
    port: int = 80
    weight: int = 100
    max_connections: int = 1000
    health_status: str = "unknown"
    active_connections: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Backend":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Listener:
    """
    Load balancer listener configuration.

    Attributes:
        listener_id: Unique listener identifier
        protocol: Listener protocol
        port: Listener port
        ssl_cert_id: SSL certificate ID (for HTTPS)
        default_action: Default action for requests
    """
    listener_id: str
    protocol: Protocol = Protocol.HTTP
    port: int = 80
    ssl_cert_id: str = ""
    default_action: str = "forward"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "listener_id": self.listener_id,
            "protocol": self.protocol.value,
            "port": self.port,
            "ssl_cert_id": self.ssl_cert_id,
            "default_action": self.default_action,
        }


@dataclass
class LoadBalancerConfig:
    """
    Load balancer configuration.

    Attributes:
        lb_id: Unique load balancer identifier
        name: Load balancer name
        lb_type: Load balancer type
        algorithm: Load balancing algorithm
        listeners: Configured listeners
        backends: Configured backends
        health_check: Health check configuration
        sticky_sessions: Whether to enable sticky sessions
        sticky_cookie: Cookie name for sticky sessions
        connection_draining: Whether to drain connections on deregister
        idle_timeout_s: Idle connection timeout
        cross_zone: Whether to balance across zones
        created_at: Creation timestamp
    """
    lb_id: str
    name: str
    lb_type: LoadBalancerType = LoadBalancerType.APPLICATION
    algorithm: Algorithm = Algorithm.ROUND_ROBIN
    listeners: List[Listener] = field(default_factory=list)
    backends: List[Backend] = field(default_factory=list)
    health_check: Optional[HealthCheck] = None
    sticky_sessions: bool = False
    sticky_cookie: str = "LBSESSION"
    connection_draining: bool = True
    idle_timeout_s: int = 60
    cross_zone: bool = True
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lb_id": self.lb_id,
            "name": self.name,
            "lb_type": self.lb_type.value,
            "algorithm": self.algorithm.value,
            "listeners": [l.to_dict() for l in self.listeners],
            "backends": [b.to_dict() for b in self.backends],
            "health_check": self.health_check.to_dict() if self.health_check else None,
            "sticky_sessions": self.sticky_sessions,
            "sticky_cookie": self.sticky_cookie,
            "connection_draining": self.connection_draining,
            "idle_timeout_s": self.idle_timeout_s,
            "cross_zone": self.cross_zone,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LoadBalancerConfig":
        data = dict(data)
        if "lb_type" in data:
            data["lb_type"] = LoadBalancerType(data["lb_type"])
        if "algorithm" in data:
            data["algorithm"] = Algorithm(data["algorithm"])
        if "listeners" in data:
            data["listeners"] = [Listener(**{**l, "protocol": Protocol(l["protocol"])})
                                for l in data["listeners"]]
        if "backends" in data:
            data["backends"] = [Backend.from_dict(b) for b in data["backends"]]
        if "health_check" in data and data["health_check"]:
            hc = data["health_check"]
            hc["protocol"] = Protocol(hc["protocol"])
            data["health_check"] = HealthCheck(**hc)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class LoadBalancerState:
    """
    Current load balancer state.

    Attributes:
        lb_id: Load balancer ID
        status: Current status
        dns_name: DNS name of the load balancer
        ip_addresses: IP addresses
        healthy_backends: Number of healthy backends
        total_backends: Total number of backends
        requests_per_second: Current RPS
        active_connections: Current active connections
        metrics: Additional metrics
    """
    lb_id: str
    status: str = "active"
    dns_name: str = ""
    ip_addresses: List[str] = field(default_factory=list)
    healthy_backends: int = 0
    total_backends: int = 0
    requests_per_second: float = 0.0
    active_connections: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# Load Balancer (Step 217)
# ==============================================================================

class LoadBalancer:
    """
    Load Balancer - manages load balancing configuration.

    PBTSO Phase: ITERATE

    Responsibilities:
    - Create and configure load balancers
    - Manage backend servers
    - Configure health checks
    - Support multiple algorithms
    - Track load balancer state

    Example:
        >>> lb = LoadBalancer()
        >>> config = lb.create(
        ...     name="api-lb",
        ...     lb_type=LoadBalancerType.APPLICATION,
        ...     algorithm=Algorithm.ROUND_ROBIN,
        ... )
        >>> lb.add_backend(config.lb_id, "api-1", "10.0.0.1", 8080)
        >>> lb.add_backend(config.lb_id, "api-2", "10.0.0.2", 8080)
    """

    BUS_TOPICS = {
        "configure": "deploy.lb.configure",
        "backend_add": "deploy.lb.backend.add",
        "backend_remove": "deploy.lb.backend.remove",
        "health": "deploy.lb.health",
        "created": "deploy.lb.created",
        "updated": "deploy.lb.updated",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "load-balancer",
    ):
        """
        Initialize the load balancer.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "lb"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        self._load_balancers: Dict[str, LoadBalancerConfig] = {}
        self._states: Dict[str, LoadBalancerState] = {}
        self._round_robin_index: Dict[str, int] = {}

        self._load_state()

    def create(
        self,
        name: str,
        lb_type: LoadBalancerType = LoadBalancerType.APPLICATION,
        algorithm: Algorithm = Algorithm.ROUND_ROBIN,
        sticky_sessions: bool = False,
        health_check: Optional[HealthCheck] = None,
    ) -> LoadBalancerConfig:
        """
        Create a new load balancer.

        Args:
            name: Load balancer name
            lb_type: Load balancer type
            algorithm: Load balancing algorithm
            sticky_sessions: Enable sticky sessions
            health_check: Health check configuration

        Returns:
            Created LoadBalancerConfig
        """
        lb_id = f"lb-{uuid.uuid4().hex[:12]}"

        if health_check and not health_check.check_id:
            health_check.check_id = f"hc-{uuid.uuid4().hex[:8]}"

        config = LoadBalancerConfig(
            lb_id=lb_id,
            name=name,
            lb_type=lb_type,
            algorithm=algorithm,
            sticky_sessions=sticky_sessions,
            health_check=health_check or HealthCheck(check_id=f"hc-{uuid.uuid4().hex[:8]}"),
        )

        # Create default listener
        listener = Listener(
            listener_id=f"listener-{uuid.uuid4().hex[:8]}",
            protocol=Protocol.HTTP,
            port=80,
        )
        config.listeners.append(listener)

        # Initialize state
        state = LoadBalancerState(
            lb_id=lb_id,
            dns_name=f"{name}.lb.local",
            ip_addresses=["10.0.0.100"],
        )

        self._load_balancers[lb_id] = config
        self._states[lb_id] = state
        self._round_robin_index[lb_id] = 0
        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["created"],
            {
                "lb_id": lb_id,
                "name": name,
                "lb_type": lb_type.value,
                "algorithm": algorithm.value,
            },
            actor=self.actor_id,
        )

        return config

    def add_backend(
        self,
        lb_id: str,
        name: str,
        address: str,
        port: int = 80,
        weight: int = 100,
        max_connections: int = 1000,
    ) -> Optional[Backend]:
        """
        Add a backend to a load balancer.

        Args:
            lb_id: Load balancer ID
            name: Backend name
            address: Backend address
            port: Backend port
            weight: Traffic weight
            max_connections: Maximum connections

        Returns:
            Added Backend or None
        """
        config = self._load_balancers.get(lb_id)
        if not config:
            return None

        backend_id = f"backend-{uuid.uuid4().hex[:8]}"

        backend = Backend(
            backend_id=backend_id,
            name=name,
            address=address,
            port=port,
            weight=weight,
            max_connections=max_connections,
            health_status="healthy",
        )

        config.backends.append(backend)

        # Update state
        state = self._states.get(lb_id)
        if state:
            state.total_backends = len(config.backends)
            state.healthy_backends = sum(1 for b in config.backends if b.health_status == "healthy")

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["backend_add"],
            {
                "lb_id": lb_id,
                "backend_id": backend_id,
                "name": name,
                "address": address,
                "port": port,
            },
            actor=self.actor_id,
        )

        return backend

    def remove_backend(self, lb_id: str, backend_id: str) -> bool:
        """
        Remove a backend from a load balancer.

        Args:
            lb_id: Load balancer ID
            backend_id: Backend ID

        Returns:
            True if removed
        """
        config = self._load_balancers.get(lb_id)
        if not config:
            return False

        backend = next((b for b in config.backends if b.backend_id == backend_id), None)
        if not backend:
            return False

        config.backends = [b for b in config.backends if b.backend_id != backend_id]

        # Update state
        state = self._states.get(lb_id)
        if state:
            state.total_backends = len(config.backends)
            state.healthy_backends = sum(1 for b in config.backends if b.health_status == "healthy")

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["backend_remove"],
            {
                "lb_id": lb_id,
                "backend_id": backend_id,
                "name": backend.name,
            },
            actor=self.actor_id,
        )

        return True

    def update_backend_health(
        self,
        lb_id: str,
        backend_id: str,
        healthy: bool,
    ) -> None:
        """Update backend health status."""
        config = self._load_balancers.get(lb_id)
        if not config:
            return

        for backend in config.backends:
            if backend.backend_id == backend_id:
                backend.health_status = "healthy" if healthy else "unhealthy"
                break

        # Update state
        state = self._states.get(lb_id)
        if state:
            state.healthy_backends = sum(1 for b in config.backends if b.health_status == "healthy")

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["health"],
            {
                "lb_id": lb_id,
                "backend_id": backend_id,
                "healthy": healthy,
            },
            kind="metric",
            actor=self.actor_id,
        )

    def get_next_backend(self, lb_id: str) -> Optional[Backend]:
        """
        Get the next backend for a request.

        Args:
            lb_id: Load balancer ID

        Returns:
            Selected Backend or None
        """
        config = self._load_balancers.get(lb_id)
        if not config:
            return None

        healthy_backends = [b for b in config.backends if b.health_status == "healthy"]
        if not healthy_backends:
            return None

        if config.algorithm == Algorithm.ROUND_ROBIN:
            idx = self._round_robin_index.get(lb_id, 0)
            backend = healthy_backends[idx % len(healthy_backends)]
            self._round_robin_index[lb_id] = idx + 1
            return backend

        elif config.algorithm == Algorithm.LEAST_CONNECTIONS:
            return min(healthy_backends, key=lambda b: b.active_connections)

        elif config.algorithm == Algorithm.WEIGHTED:
            import random
            total_weight = sum(b.weight for b in healthy_backends)
            r = random.randint(1, total_weight)
            cumulative = 0
            for backend in healthy_backends:
                cumulative += backend.weight
                if r <= cumulative:
                    return backend
            return healthy_backends[-1]

        elif config.algorithm == Algorithm.RANDOM:
            import random
            return random.choice(healthy_backends)

        return healthy_backends[0]

    def add_listener(
        self,
        lb_id: str,
        protocol: Protocol = Protocol.HTTP,
        port: int = 80,
        ssl_cert_id: str = "",
    ) -> Optional[Listener]:
        """Add a listener to a load balancer."""
        config = self._load_balancers.get(lb_id)
        if not config:
            return None

        listener = Listener(
            listener_id=f"listener-{uuid.uuid4().hex[:8]}",
            protocol=protocol,
            port=port,
            ssl_cert_id=ssl_cert_id,
        )

        config.listeners.append(listener)
        self._save_state()

        return listener

    def configure(
        self,
        lb_id: str,
        algorithm: Optional[Algorithm] = None,
        sticky_sessions: Optional[bool] = None,
        idle_timeout_s: Optional[int] = None,
        cross_zone: Optional[bool] = None,
    ) -> Optional[LoadBalancerConfig]:
        """
        Update load balancer configuration.

        Args:
            lb_id: Load balancer ID
            algorithm: New algorithm
            sticky_sessions: Enable/disable sticky sessions
            idle_timeout_s: New idle timeout
            cross_zone: Enable/disable cross-zone balancing

        Returns:
            Updated config or None
        """
        config = self._load_balancers.get(lb_id)
        if not config:
            return None

        if algorithm is not None:
            config.algorithm = algorithm
        if sticky_sessions is not None:
            config.sticky_sessions = sticky_sessions
        if idle_timeout_s is not None:
            config.idle_timeout_s = idle_timeout_s
        if cross_zone is not None:
            config.cross_zone = cross_zone

        self._save_state()

        _emit_bus_event(
            self.BUS_TOPICS["configure"],
            {
                "lb_id": lb_id,
                "name": config.name,
                "algorithm": config.algorithm.value,
                "sticky_sessions": config.sticky_sessions,
            },
            actor=self.actor_id,
        )

        return config

    def get_config(self, lb_id: str) -> Optional[LoadBalancerConfig]:
        """Get load balancer configuration."""
        return self._load_balancers.get(lb_id)

    def get_state(self, lb_id: str) -> Optional[LoadBalancerState]:
        """Get load balancer state."""
        return self._states.get(lb_id)

    def list_load_balancers(self) -> List[LoadBalancerConfig]:
        """List all load balancers."""
        return list(self._load_balancers.values())

    def delete(self, lb_id: str) -> bool:
        """Delete a load balancer."""
        if lb_id not in self._load_balancers:
            return False

        del self._load_balancers[lb_id]
        if lb_id in self._states:
            del self._states[lb_id]
        if lb_id in self._round_robin_index:
            del self._round_robin_index[lb_id]

        self._save_state()
        return True

    def _save_state(self) -> None:
        """Save state to disk."""
        state = {
            "load_balancers": {lid: lb.to_dict() for lid, lb in self._load_balancers.items()},
            "states": {lid: s.to_dict() for lid, s in self._states.items()},
        }
        state_file = self.state_dir / "lb_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        state_file = self.state_dir / "lb_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, "r") as f:
                state = json.load(f)

            for lid, data in state.get("load_balancers", {}).items():
                self._load_balancers[lid] = LoadBalancerConfig.from_dict(data)
                self._round_robin_index[lid] = 0

            for lid, data in state.get("states", {}).items():
                self._states[lid] = LoadBalancerState(**data)
        except (json.JSONDecodeError, IOError):
            pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for load balancer."""
    import argparse

    parser = argparse.ArgumentParser(description="Load Balancer (Step 217)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # create command
    create_parser = subparsers.add_parser("create", help="Create a load balancer")
    create_parser.add_argument("name", help="Load balancer name")
    create_parser.add_argument("--type", "-t", default="application",
                              choices=["application", "network", "classic"])
    create_parser.add_argument("--algorithm", "-a", default="round_robin",
                              choices=["round_robin", "least_connections", "weighted", "random"])
    create_parser.add_argument("--sticky", action="store_true", help="Enable sticky sessions")
    create_parser.add_argument("--json", action="store_true", help="JSON output")

    # backend command
    backend_parser = subparsers.add_parser("backend", help="Add a backend")
    backend_parser.add_argument("lb_id", help="Load balancer ID")
    backend_parser.add_argument("--name", "-n", required=True, help="Backend name")
    backend_parser.add_argument("--address", "-a", required=True, help="Backend address")
    backend_parser.add_argument("--port", "-p", type=int, default=80, help="Backend port")
    backend_parser.add_argument("--weight", "-w", type=int, default=100, help="Traffic weight")
    backend_parser.add_argument("--json", action="store_true", help="JSON output")

    # remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a backend")
    remove_parser.add_argument("lb_id", help="Load balancer ID")
    remove_parser.add_argument("backend_id", help="Backend ID")

    # status command
    status_parser = subparsers.add_parser("status", help="Get load balancer status")
    status_parser.add_argument("lb_id", help="Load balancer ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List load balancers")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    lb = LoadBalancer()

    if args.command == "create":
        config = lb.create(
            name=args.name,
            lb_type=LoadBalancerType(args.type.upper() if args.type != "classic" else "CLASSIC"),
            algorithm=Algorithm(args.algorithm.upper()),
            sticky_sessions=args.sticky,
        )

        if args.json:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            state = lb.get_state(config.lb_id)
            print(f"Created load balancer: {config.lb_id}")
            print(f"  Name: {config.name}")
            print(f"  Type: {config.lb_type.value}")
            print(f"  Algorithm: {config.algorithm.value}")
            if state:
                print(f"  DNS: {state.dns_name}")

        return 0

    elif args.command == "backend":
        backend = lb.add_backend(
            lb_id=args.lb_id,
            name=args.name,
            address=args.address,
            port=args.port,
            weight=args.weight,
        )

        if not backend:
            print(f"Load balancer not found: {args.lb_id}")
            return 1

        if args.json:
            print(json.dumps(backend.to_dict(), indent=2))
        else:
            print(f"Added backend: {backend.backend_id}")
            print(f"  Name: {backend.name}")
            print(f"  Address: {backend.address}:{backend.port}")
            print(f"  Weight: {backend.weight}")

        return 0

    elif args.command == "remove":
        success = lb.remove_backend(args.lb_id, args.backend_id)
        if success:
            print(f"Removed backend: {args.backend_id}")
        else:
            print(f"Backend not found: {args.backend_id}")
            return 1

        return 0

    elif args.command == "status":
        config = lb.get_config(args.lb_id)
        state = lb.get_state(args.lb_id)

        if not config or not state:
            print(f"Load balancer not found: {args.lb_id}")
            return 1

        if args.json:
            print(json.dumps({
                "config": config.to_dict(),
                "state": state.to_dict(),
            }, indent=2))
        else:
            print(f"Load Balancer: {config.lb_id}")
            print(f"  Name: {config.name}")
            print(f"  Type: {config.lb_type.value}")
            print(f"  Algorithm: {config.algorithm.value}")
            print(f"  DNS: {state.dns_name}")
            print(f"  Status: {state.status}")
            print(f"  Backends: {state.healthy_backends}/{state.total_backends} healthy")
            print(f"  Backends:")
            for b in config.backends:
                print(f"    {b.backend_id}: {b.name} ({b.address}:{b.port}) [{b.health_status}]")

        return 0

    elif args.command == "list":
        lbs = lb.list_load_balancers()

        if args.json:
            print(json.dumps([l.to_dict() for l in lbs], indent=2))
        else:
            if not lbs:
                print("No load balancers found")
            else:
                for l in lbs:
                    state = lb.get_state(l.lb_id)
                    backends = f"{state.healthy_backends}/{state.total_backends}" if state else "?"
                    print(f"{l.lb_id}: {l.name} [{l.algorithm.value}] ({backends} backends)")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
