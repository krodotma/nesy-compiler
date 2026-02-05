#!/usr/bin/env python3
"""
Network Monitor - Step 262

Tracks network health, latency, and bandwidth metrics.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.network.health (emitted)
- monitor.network.latency (emitted)
- monitor.network.alert (emitted)

Protocol: DKIN v30, CITIZEN v2, HOLON v2
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor


class ConnectionState(Enum):
    """Network connection states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNREACHABLE = "unreachable"
    UNKNOWN = "unknown"


class NetworkAlertType(Enum):
    """Types of network alerts."""
    HIGH_LATENCY = "high_latency"
    PACKET_LOSS = "packet_loss"
    CONNECTION_FAILURE = "connection_failure"
    BANDWIDTH_SATURATION = "bandwidth_saturation"
    DNS_FAILURE = "dns_failure"


@dataclass
class LatencyMetrics:
    """Network latency metrics.

    Attributes:
        target: Target host or endpoint
        min_ms: Minimum latency in ms
        max_ms: Maximum latency in ms
        avg_ms: Average latency in ms
        jitter_ms: Latency jitter (std dev)
        packet_loss_percent: Packet loss percentage
        samples: Number of samples
        timestamp: Collection timestamp
    """
    target: str
    min_ms: float
    max_ms: float
    avg_ms: float
    jitter_ms: float = 0.0
    packet_loss_percent: float = 0.0
    samples: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def is_high(self) -> bool:
        """Check if latency is high (>100ms)."""
        return self.avg_ms > 100.0

    @property
    def has_packet_loss(self) -> bool:
        """Check if there is packet loss."""
        return self.packet_loss_percent > 0.0


@dataclass
class BandwidthMetrics:
    """Network bandwidth metrics.

    Attributes:
        interface: Network interface name
        rx_bytes_per_sec: Receive throughput (bytes/sec)
        tx_bytes_per_sec: Transmit throughput (bytes/sec)
        rx_packets_per_sec: Receive packets/sec
        tx_packets_per_sec: Transmit packets/sec
        rx_errors: Receive errors
        tx_errors: Transmit errors
        rx_dropped: Receive dropped packets
        tx_dropped: Transmit dropped packets
        timestamp: Collection timestamp
    """
    interface: str
    rx_bytes_per_sec: float
    tx_bytes_per_sec: float
    rx_packets_per_sec: float = 0.0
    tx_packets_per_sec: float = 0.0
    rx_errors: int = 0
    tx_errors: int = 0
    rx_dropped: int = 0
    tx_dropped: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def rx_mbps(self) -> float:
        """Receive throughput in Mbps."""
        return self.rx_bytes_per_sec * 8 / 1_000_000

    @property
    def tx_mbps(self) -> float:
        """Transmit throughput in Mbps."""
        return self.tx_bytes_per_sec * 8 / 1_000_000

    @property
    def total_mbps(self) -> float:
        """Total throughput in Mbps."""
        return self.rx_mbps + self.tx_mbps


@dataclass
class ConnectionHealth:
    """Network connection health check result.

    Attributes:
        target: Target host/endpoint
        port: Target port (if applicable)
        state: Connection state
        latency_ms: Connection latency
        dns_resolution_ms: DNS resolution time
        ssl_valid: SSL certificate valid
        ssl_expiry_days: Days until SSL expiry
        error_message: Error message if failed
        timestamp: Check timestamp
    """
    target: str
    port: Optional[int]
    state: ConnectionState
    latency_ms: float = 0.0
    dns_resolution_ms: float = 0.0
    ssl_valid: bool = True
    ssl_expiry_days: Optional[int] = None
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "target": self.target,
            "port": self.port,
            "state": self.state.value,
            "latency_ms": self.latency_ms,
            "dns_resolution_ms": self.dns_resolution_ms,
            "ssl_valid": self.ssl_valid,
            "ssl_expiry_days": self.ssl_expiry_days,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
        }


@dataclass
class NetworkMetrics:
    """Combined network metrics.

    Attributes:
        interfaces: Bandwidth metrics per interface
        latencies: Latency metrics per target
        connections: Connection health per target
        hostname: Local hostname
        timestamp: Collection timestamp
    """
    interfaces: List[BandwidthMetrics] = field(default_factory=list)
    latencies: List[LatencyMetrics] = field(default_factory=list)
    connections: List[ConnectionHealth] = field(default_factory=list)
    hostname: str = field(default_factory=socket.gethostname)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "interfaces": [i.to_dict() for i in self.interfaces],
            "latencies": [l.to_dict() for l in self.latencies],
            "connections": [c.to_dict() for c in self.connections],
            "hostname": self.hostname,
            "timestamp": self.timestamp,
        }

    @property
    def all_healthy(self) -> bool:
        """Check if all connections are healthy."""
        return all(c.state == ConnectionState.HEALTHY for c in self.connections)

    @property
    def unhealthy_targets(self) -> List[str]:
        """Get list of unhealthy targets."""
        return [c.target for c in self.connections if c.state != ConnectionState.HEALTHY]


@dataclass
class NetworkAlert:
    """Network alert.

    Attributes:
        alert_type: Type of alert
        target: Affected target
        message: Alert message
        current_value: Current metric value
        threshold: Threshold that was exceeded
        timestamp: Alert timestamp
        alert_id: Unique alert ID
    """
    alert_type: NetworkAlertType
    target: str
    message: str
    current_value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "target": self.target,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
        }


@dataclass
class NetworkThresholds:
    """Configurable network thresholds.

    Attributes:
        latency_warning_ms: Latency warning threshold
        latency_critical_ms: Latency critical threshold
        packet_loss_warning: Packet loss warning threshold (%)
        connection_timeout_s: Connection timeout in seconds
        dns_timeout_s: DNS resolution timeout
    """
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0
    packet_loss_warning: float = 1.0
    connection_timeout_s: float = 5.0
    dns_timeout_s: float = 2.0


class NetworkMonitor:
    """
    Monitor network health and performance.

    The monitor:
    - Tracks network interface bandwidth
    - Measures latency to configured targets
    - Monitors connection health
    - Generates alerts on issues

    Example:
        monitor = NetworkMonitor(targets=["8.8.8.8", "github.com"])
        await monitor.start()

        metrics = await monitor.collect_metrics()
        for latency in metrics.latencies:
            print(f"{latency.target}: {latency.avg_ms}ms")
    """

    BUS_TOPICS = {
        "health": "monitor.network.health",
        "latency": "monitor.network.latency",
        "alert": "monitor.network.alert",
    }

    def __init__(
        self,
        targets: Optional[List[str]] = None,
        thresholds: Optional[NetworkThresholds] = None,
        collection_interval_s: int = 60,
        history_size: int = 1000,
        bus_dir: Optional[str] = None,
    ):
        """Initialize network monitor.

        Args:
            targets: List of targets to monitor (hosts/IPs)
            thresholds: Alert thresholds
            collection_interval_s: Collection interval
            history_size: History size
            bus_dir: Bus directory
        """
        self.targets = targets or ["8.8.8.8", "1.1.1.1"]
        self.thresholds = thresholds or NetworkThresholds()
        self.collection_interval_s = collection_interval_s
        self.history_size = history_size

        # History
        self._bandwidth_history: Dict[str, List[BandwidthMetrics]] = {}
        self._latency_history: Dict[str, List[LatencyMetrics]] = {}
        self._connection_history: Dict[str, List[ConnectionHealth]] = {}

        # Previous values for rate calculation
        self._prev_interface_stats: Dict[str, Dict[str, int]] = {}
        self._prev_interface_time: Optional[float] = None

        # State
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._alert_callbacks: List[Callable[[NetworkAlert], None]] = []

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    async def start(self) -> bool:
        """Start the network monitor.

        Returns:
            True if started successfully
        """
        if self._running:
            return False

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        self._emit_bus_event(
            "monitor.network.started",
            {
                "targets": self.targets,
                "collection_interval_s": self.collection_interval_s,
            }
        )

        return True

    async def stop(self) -> bool:
        """Stop the network monitor.

        Returns:
            True if stopped successfully
        """
        if not self._running:
            return False

        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        self._executor.shutdown(wait=False)
        return True

    async def collect_metrics(self) -> NetworkMetrics:
        """Collect network metrics.

        Returns:
            Network metrics
        """
        loop = asyncio.get_event_loop()

        # Collect bandwidth metrics
        interfaces = self._collect_bandwidth()

        # Collect latency metrics in parallel
        latency_tasks = [
            loop.run_in_executor(self._executor, self._ping_target, target)
            for target in self.targets
        ]
        latencies = await asyncio.gather(*latency_tasks, return_exceptions=True)
        latencies = [l for l in latencies if isinstance(l, LatencyMetrics)]

        # Check connections
        connection_tasks = [
            loop.run_in_executor(self._executor, self._check_connection, target)
            for target in self.targets
        ]
        connections = await asyncio.gather(*connection_tasks, return_exceptions=True)
        connections = [c for c in connections if isinstance(c, ConnectionHealth)]

        metrics = NetworkMetrics(
            interfaces=interfaces,
            latencies=latencies,
            connections=connections,
        )

        # Store in history
        for interface in interfaces:
            if interface.interface not in self._bandwidth_history:
                self._bandwidth_history[interface.interface] = []
            self._bandwidth_history[interface.interface].append(interface)
            if len(self._bandwidth_history[interface.interface]) > self.history_size:
                self._bandwidth_history[interface.interface] = (
                    self._bandwidth_history[interface.interface][-self.history_size:]
                )

        for latency in latencies:
            if latency.target not in self._latency_history:
                self._latency_history[latency.target] = []
            self._latency_history[latency.target].append(latency)
            if len(self._latency_history[latency.target]) > self.history_size:
                self._latency_history[latency.target] = (
                    self._latency_history[latency.target][-self.history_size:]
                )

        # Emit events
        self._emit_bus_event(self.BUS_TOPICS["health"], metrics.to_dict())

        # Check for alerts
        alerts = self._check_alerts(metrics)
        for alert in alerts:
            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert.to_dict(),
                level="warning"
            )
            for callback in self._alert_callbacks:
                callback(alert)

        return metrics

    def add_target(self, target: str) -> None:
        """Add a monitoring target.

        Args:
            target: Host/IP to monitor
        """
        if target not in self.targets:
            self.targets.append(target)

    def remove_target(self, target: str) -> bool:
        """Remove a monitoring target.

        Args:
            target: Target to remove

        Returns:
            True if removed
        """
        if target in self.targets:
            self.targets.remove(target)
            return True
        return False

    def get_latency_history(
        self,
        target: str,
        window_s: int = 3600
    ) -> List[LatencyMetrics]:
        """Get latency history for a target.

        Args:
            target: Target host
            window_s: Time window

        Returns:
            Latency history
        """
        if target not in self._latency_history:
            return []
        cutoff = time.time() - window_s
        return [l for l in self._latency_history[target] if l.timestamp >= cutoff]

    def get_average_latency(self, target: str, window_s: int = 300) -> float:
        """Get average latency for a target.

        Args:
            target: Target host
            window_s: Time window

        Returns:
            Average latency in ms
        """
        history = self.get_latency_history(target, window_s)
        if not history:
            return 0.0
        return sum(l.avg_ms for l in history) / len(history)

    def get_bandwidth_history(
        self,
        interface: str,
        window_s: int = 3600
    ) -> List[BandwidthMetrics]:
        """Get bandwidth history for an interface.

        Args:
            interface: Interface name
            window_s: Time window

        Returns:
            Bandwidth history
        """
        if interface not in self._bandwidth_history:
            return []
        cutoff = time.time() - window_s
        return [b for b in self._bandwidth_history[interface] if b.timestamp >= cutoff]

    def register_alert_callback(
        self,
        callback: Callable[[NetworkAlert], None]
    ) -> None:
        """Register alert callback.

        Args:
            callback: Callback function
        """
        self._alert_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "targets": self.targets,
            "collection_interval_s": self.collection_interval_s,
            "history_sizes": {
                "bandwidth": {k: len(v) for k, v in self._bandwidth_history.items()},
                "latency": {k: len(v) for k, v in self._latency_history.items()},
            },
        }

    def _collect_bandwidth(self) -> List[BandwidthMetrics]:
        """Collect bandwidth metrics from /proc/net/dev."""
        interfaces = []
        now = time.time()

        try:
            with open("/proc/net/dev", "r") as f:
                lines = f.readlines()[2:]  # Skip header

            current_stats = {}
            for line in lines:
                parts = line.split()
                iface = parts[0].rstrip(":")

                # Skip loopback and virtual interfaces
                if iface in ("lo", "docker0") or iface.startswith("veth"):
                    continue

                rx_bytes = int(parts[1])
                rx_packets = int(parts[2])
                rx_errors = int(parts[3])
                rx_dropped = int(parts[4])
                tx_bytes = int(parts[9])
                tx_packets = int(parts[10])
                tx_errors = int(parts[11])
                tx_dropped = int(parts[12])

                current_stats[iface] = {
                    "rx_bytes": rx_bytes,
                    "rx_packets": rx_packets,
                    "rx_errors": rx_errors,
                    "rx_dropped": rx_dropped,
                    "tx_bytes": tx_bytes,
                    "tx_packets": tx_packets,
                    "tx_errors": tx_errors,
                    "tx_dropped": tx_dropped,
                }

                # Calculate rates if we have previous values
                if (self._prev_interface_time is not None and
                    iface in self._prev_interface_stats):
                    dt = now - self._prev_interface_time
                    if dt > 0:
                        prev = self._prev_interface_stats[iface]
                        rx_bytes_sec = (rx_bytes - prev["rx_bytes"]) / dt
                        tx_bytes_sec = (tx_bytes - prev["tx_bytes"]) / dt
                        rx_packets_sec = (rx_packets - prev["rx_packets"]) / dt
                        tx_packets_sec = (tx_packets - prev["tx_packets"]) / dt

                        interfaces.append(BandwidthMetrics(
                            interface=iface,
                            rx_bytes_per_sec=max(0, rx_bytes_sec),
                            tx_bytes_per_sec=max(0, tx_bytes_sec),
                            rx_packets_per_sec=max(0, rx_packets_sec),
                            tx_packets_per_sec=max(0, tx_packets_sec),
                            rx_errors=rx_errors,
                            tx_errors=tx_errors,
                            rx_dropped=rx_dropped,
                            tx_dropped=tx_dropped,
                        ))

            self._prev_interface_stats = current_stats
            self._prev_interface_time = now

        except Exception:
            pass

        return interfaces

    def _ping_target(self, target: str, count: int = 3) -> LatencyMetrics:
        """Ping a target and collect latency metrics.

        Args:
            target: Target host
            count: Number of pings

        Returns:
            Latency metrics
        """
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), "-W", "2", target],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse ping output
                output = result.stdout

                # Extract statistics
                latencies = []
                for line in output.split("\n"):
                    if "time=" in line:
                        time_str = line.split("time=")[1].split()[0]
                        latencies.append(float(time_str.rstrip("ms")))

                if latencies:
                    avg_latency = sum(latencies) / len(latencies)
                    min_latency = min(latencies)
                    max_latency = max(latencies)

                    # Calculate jitter (standard deviation)
                    if len(latencies) > 1:
                        variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
                        jitter = variance ** 0.5
                    else:
                        jitter = 0.0

                    # Calculate packet loss
                    sent = count
                    received = len(latencies)
                    packet_loss = 100.0 * (sent - received) / sent if sent > 0 else 0.0

                    return LatencyMetrics(
                        target=target,
                        min_ms=min_latency,
                        max_ms=max_latency,
                        avg_ms=avg_latency,
                        jitter_ms=jitter,
                        packet_loss_percent=packet_loss,
                        samples=len(latencies),
                    )

            # Ping failed - 100% packet loss
            return LatencyMetrics(
                target=target,
                min_ms=0.0,
                max_ms=0.0,
                avg_ms=0.0,
                packet_loss_percent=100.0,
                samples=0,
            )

        except subprocess.TimeoutExpired:
            return LatencyMetrics(
                target=target,
                min_ms=0.0,
                max_ms=0.0,
                avg_ms=0.0,
                packet_loss_percent=100.0,
                samples=0,
            )
        except Exception:
            return LatencyMetrics(
                target=target,
                min_ms=0.0,
                max_ms=0.0,
                avg_ms=0.0,
                packet_loss_percent=100.0,
                samples=0,
            )

    def _check_connection(self, target: str) -> ConnectionHealth:
        """Check connection health to a target.

        Args:
            target: Target host

        Returns:
            Connection health
        """
        port = None
        state = ConnectionState.UNKNOWN
        latency_ms = 0.0
        dns_resolution_ms = 0.0
        error_message = None

        try:
            # DNS resolution timing
            start_dns = time.time()
            try:
                socket.gethostbyname(target)
                dns_resolution_ms = (time.time() - start_dns) * 1000
            except socket.gaierror as e:
                return ConnectionHealth(
                    target=target,
                    port=port,
                    state=ConnectionState.UNREACHABLE,
                    error_message=f"DNS resolution failed: {e}",
                )

            # Try TCP connection to common ports
            for test_port in [443, 80, 22]:
                start_conn = time.time()
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.thresholds.connection_timeout_s)
                try:
                    result = sock.connect_ex((target, test_port))
                    if result == 0:
                        latency_ms = (time.time() - start_conn) * 1000
                        port = test_port
                        state = ConnectionState.HEALTHY
                        break
                except socket.error:
                    continue
                finally:
                    sock.close()

            if state == ConnectionState.UNKNOWN:
                # Try ICMP (ping) as fallback
                result = subprocess.run(
                    ["ping", "-c", "1", "-W", "2", target],
                    capture_output=True,
                    timeout=5
                )
                if result.returncode == 0:
                    state = ConnectionState.HEALTHY
                else:
                    state = ConnectionState.UNREACHABLE
                    error_message = "No TCP ports or ICMP reachable"

            # Check if latency indicates degradation
            if state == ConnectionState.HEALTHY and latency_ms > self.thresholds.latency_critical_ms:
                state = ConnectionState.DEGRADED

        except Exception as e:
            state = ConnectionState.UNREACHABLE
            error_message = str(e)

        return ConnectionHealth(
            target=target,
            port=port,
            state=state,
            latency_ms=latency_ms,
            dns_resolution_ms=dns_resolution_ms,
            error_message=error_message,
        )

    def _check_alerts(self, metrics: NetworkMetrics) -> List[NetworkAlert]:
        """Check metrics for alerts.

        Args:
            metrics: Current metrics

        Returns:
            List of alerts
        """
        alerts = []

        # Latency alerts
        for latency in metrics.latencies:
            if latency.avg_ms >= self.thresholds.latency_critical_ms:
                alerts.append(NetworkAlert(
                    alert_type=NetworkAlertType.HIGH_LATENCY,
                    target=latency.target,
                    message=f"Critical latency to {latency.target}: {latency.avg_ms:.1f}ms",
                    current_value=latency.avg_ms,
                    threshold=self.thresholds.latency_critical_ms,
                ))
            elif latency.avg_ms >= self.thresholds.latency_warning_ms:
                alerts.append(NetworkAlert(
                    alert_type=NetworkAlertType.HIGH_LATENCY,
                    target=latency.target,
                    message=f"High latency to {latency.target}: {latency.avg_ms:.1f}ms",
                    current_value=latency.avg_ms,
                    threshold=self.thresholds.latency_warning_ms,
                ))

            # Packet loss alerts
            if latency.packet_loss_percent >= self.thresholds.packet_loss_warning:
                alerts.append(NetworkAlert(
                    alert_type=NetworkAlertType.PACKET_LOSS,
                    target=latency.target,
                    message=f"Packet loss to {latency.target}: {latency.packet_loss_percent:.1f}%",
                    current_value=latency.packet_loss_percent,
                    threshold=self.thresholds.packet_loss_warning,
                ))

        # Connection failure alerts
        for conn in metrics.connections:
            if conn.state == ConnectionState.UNREACHABLE:
                alerts.append(NetworkAlert(
                    alert_type=NetworkAlertType.CONNECTION_FAILURE,
                    target=conn.target,
                    message=f"Connection failed to {conn.target}: {conn.error_message or 'unreachable'}",
                    current_value=0.0,
                    threshold=0.0,
                ))

        return alerts

    async def _collection_loop(self) -> None:
        """Background collection loop."""
        while self._running:
            try:
                await self.collect_metrics()
            except Exception as e:
                self._emit_bus_event(
                    "monitor.network.error",
                    {"error": str(e)},
                    level="error"
                )

            await asyncio.sleep(self.collection_interval_s)

    def _emit_bus_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
        kind: str = "event"
    ) -> str:
        """Emit event to bus."""
        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": kind,
            "level": level,
            "actor": "monitor-agent",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        try:
            with open(self._bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

        return event_id


# Singleton
_monitor: Optional[NetworkMonitor] = None


def get_network_monitor() -> NetworkMonitor:
    """Get or create the network monitor singleton."""
    global _monitor
    if _monitor is None:
        _monitor = NetworkMonitor()
    return _monitor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Network Monitor (Step 262)")
    parser.add_argument("--collect", action="store_true", help="Collect metrics once")
    parser.add_argument("--target", action="append", help="Target to monitor")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    monitor = get_network_monitor()
    if args.target:
        for target in args.target:
            monitor.add_target(target)

    if args.collect:
        async def main():
            metrics = await monitor.collect_metrics()
            if args.json:
                print(json.dumps(metrics.to_dict(), indent=2))
            else:
                print("Network Metrics:")
                print("\nLatencies:")
                for lat in metrics.latencies:
                    print(f"  {lat.target}: {lat.avg_ms:.1f}ms "
                          f"(min: {lat.min_ms:.1f}, max: {lat.max_ms:.1f}, "
                          f"loss: {lat.packet_loss_percent:.1f}%)")
                print("\nConnections:")
                for conn in metrics.connections:
                    print(f"  {conn.target}: {conn.state.value} "
                          f"(latency: {conn.latency_ms:.1f}ms)")
                print("\nBandwidth:")
                for iface in metrics.interfaces:
                    print(f"  {iface.interface}: "
                          f"RX: {iface.rx_mbps:.2f} Mbps, "
                          f"TX: {iface.tx_mbps:.2f} Mbps")

        asyncio.run(main())
