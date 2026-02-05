#!/usr/bin/env python3
"""
Resource Monitor - Step 261

Tracks CPU, memory, and disk usage across the system.

PBTSO Phase: ITERATE

Bus Topics:
- monitor.resources.track (emitted)
- monitor.resources.alert (emitted)
- monitor.resources.collected (emitted)

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
from typing import Any, Callable, Dict, List, Optional, Tuple


class ResourceType(Enum):
    """Types of system resources."""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    SWAP = "swap"
    NETWORK_IO = "network_io"
    DISK_IO = "disk_io"


class AlertSeverity(Enum):
    """Resource alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class CPUMetrics:
    """CPU usage metrics.

    Attributes:
        usage_percent: Overall CPU usage percentage (0-100)
        user_percent: User-space CPU percentage
        system_percent: Kernel CPU percentage
        idle_percent: Idle CPU percentage
        iowait_percent: IO wait percentage
        load_avg_1m: 1-minute load average
        load_avg_5m: 5-minute load average
        load_avg_15m: 15-minute load average
        core_count: Number of CPU cores
        per_core_usage: Usage per core
        timestamp: Collection timestamp
    """
    usage_percent: float
    user_percent: float = 0.0
    system_percent: float = 0.0
    idle_percent: float = 0.0
    iowait_percent: float = 0.0
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0
    core_count: int = 1
    per_core_usage: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def is_high(self) -> bool:
        """Check if CPU usage is high (>80%)."""
        return self.usage_percent > 80.0

    @property
    def is_critical(self) -> bool:
        """Check if CPU usage is critical (>95%)."""
        return self.usage_percent > 95.0


@dataclass
class MemoryMetrics:
    """Memory usage metrics.

    Attributes:
        total_bytes: Total physical memory
        available_bytes: Available memory
        used_bytes: Used memory
        usage_percent: Usage percentage (0-100)
        cached_bytes: Cached memory
        buffers_bytes: Buffer memory
        swap_total_bytes: Total swap
        swap_used_bytes: Used swap
        swap_percent: Swap usage percentage
        timestamp: Collection timestamp
    """
    total_bytes: int
    available_bytes: int
    used_bytes: int
    usage_percent: float
    cached_bytes: int = 0
    buffers_bytes: int = 0
    swap_total_bytes: int = 0
    swap_used_bytes: int = 0
    swap_percent: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def total_gb(self) -> float:
        """Total memory in GB."""
        return self.total_bytes / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        """Available memory in GB."""
        return self.available_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        """Used memory in GB."""
        return self.used_bytes / (1024 ** 3)

    @property
    def is_high(self) -> bool:
        """Check if memory usage is high (>80%)."""
        return self.usage_percent > 80.0

    @property
    def is_critical(self) -> bool:
        """Check if memory usage is critical (>95%)."""
        return self.usage_percent > 95.0


@dataclass
class DiskMetrics:
    """Disk usage metrics.

    Attributes:
        mount_point: Mount point path
        device: Device name
        total_bytes: Total disk space
        used_bytes: Used space
        free_bytes: Free space
        usage_percent: Usage percentage
        inodes_total: Total inodes
        inodes_used: Used inodes
        inodes_percent: Inode usage percentage
        read_bytes_per_sec: Read throughput
        write_bytes_per_sec: Write throughput
        read_ops_per_sec: Read IOPS
        write_ops_per_sec: Write IOPS
        timestamp: Collection timestamp
    """
    mount_point: str
    device: str
    total_bytes: int
    used_bytes: int
    free_bytes: int
    usage_percent: float
    inodes_total: int = 0
    inodes_used: int = 0
    inodes_percent: float = 0.0
    read_bytes_per_sec: float = 0.0
    write_bytes_per_sec: float = 0.0
    read_ops_per_sec: float = 0.0
    write_ops_per_sec: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @property
    def total_gb(self) -> float:
        """Total space in GB."""
        return self.total_bytes / (1024 ** 3)

    @property
    def free_gb(self) -> float:
        """Free space in GB."""
        return self.free_bytes / (1024 ** 3)

    @property
    def is_high(self) -> bool:
        """Check if disk usage is high (>80%)."""
        return self.usage_percent > 80.0

    @property
    def is_critical(self) -> bool:
        """Check if disk usage is critical (>95%)."""
        return self.usage_percent > 95.0


@dataclass
class ResourceMetrics:
    """Combined resource metrics.

    Attributes:
        cpu: CPU metrics
        memory: Memory metrics
        disks: Disk metrics per mount point
        hostname: Host name
        timestamp: Collection timestamp
    """
    cpu: CPUMetrics
    memory: MemoryMetrics
    disks: List[DiskMetrics] = field(default_factory=list)
    hostname: str = field(default_factory=socket.gethostname)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "disks": [d.to_dict() for d in self.disks],
            "hostname": self.hostname,
            "timestamp": self.timestamp,
        }

    def get_alerts(self) -> List["ResourceAlert"]:
        """Get any active resource alerts."""
        alerts = []

        if self.cpu.is_critical:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.CPU,
                severity=AlertSeverity.CRITICAL,
                message=f"CPU usage critical: {self.cpu.usage_percent:.1f}%",
                current_value=self.cpu.usage_percent,
                threshold=95.0,
            ))
        elif self.cpu.is_high:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.CPU,
                severity=AlertSeverity.WARNING,
                message=f"CPU usage high: {self.cpu.usage_percent:.1f}%",
                current_value=self.cpu.usage_percent,
                threshold=80.0,
            ))

        if self.memory.is_critical:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.MEMORY,
                severity=AlertSeverity.CRITICAL,
                message=f"Memory usage critical: {self.memory.usage_percent:.1f}%",
                current_value=self.memory.usage_percent,
                threshold=95.0,
            ))
        elif self.memory.is_high:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.MEMORY,
                severity=AlertSeverity.WARNING,
                message=f"Memory usage high: {self.memory.usage_percent:.1f}%",
                current_value=self.memory.usage_percent,
                threshold=80.0,
            ))

        for disk in self.disks:
            if disk.is_critical:
                alerts.append(ResourceAlert(
                    resource_type=ResourceType.DISK,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Disk {disk.mount_point} critical: {disk.usage_percent:.1f}%",
                    current_value=disk.usage_percent,
                    threshold=95.0,
                    resource_id=disk.mount_point,
                ))
            elif disk.is_high:
                alerts.append(ResourceAlert(
                    resource_type=ResourceType.DISK,
                    severity=AlertSeverity.WARNING,
                    message=f"Disk {disk.mount_point} high: {disk.usage_percent:.1f}%",
                    current_value=disk.usage_percent,
                    threshold=80.0,
                    resource_id=disk.mount_point,
                ))

        return alerts


@dataclass
class ResourceAlert:
    """Resource usage alert.

    Attributes:
        resource_type: Type of resource
        severity: Alert severity
        message: Alert message
        current_value: Current value
        threshold: Threshold that was exceeded
        resource_id: Optional resource identifier (e.g., mount point)
        timestamp: Alert timestamp
        alert_id: Unique alert ID
    """
    resource_type: ResourceType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold: float
    resource_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "resource_type": self.resource_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "resource_id": self.resource_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ResourceThresholds:
    """Configurable resource thresholds.

    Attributes:
        cpu_warning: CPU warning threshold (%)
        cpu_critical: CPU critical threshold (%)
        memory_warning: Memory warning threshold (%)
        memory_critical: Memory critical threshold (%)
        disk_warning: Disk warning threshold (%)
        disk_critical: Disk critical threshold (%)
        swap_warning: Swap warning threshold (%)
        load_warning_factor: Load average warning (factor of cores)
    """
    cpu_warning: float = 80.0
    cpu_critical: float = 95.0
    memory_warning: float = 80.0
    memory_critical: float = 95.0
    disk_warning: float = 80.0
    disk_critical: float = 95.0
    swap_warning: float = 50.0
    load_warning_factor: float = 2.0


class ResourceMonitor:
    """
    Monitor system resources: CPU, memory, disk.

    The monitor:
    - Collects CPU, memory, and disk metrics
    - Tracks resource usage over time
    - Generates alerts on threshold breaches
    - Emits metrics to the A2A bus

    Example:
        monitor = ResourceMonitor()
        await monitor.start()

        metrics = monitor.collect_metrics()
        print(f"CPU: {metrics.cpu.usage_percent}%")

        alerts = metrics.get_alerts()
        for alert in alerts:
            print(f"Alert: {alert.message}")
    """

    BUS_TOPICS = {
        "track": "monitor.resources.track",
        "alert": "monitor.resources.alert",
        "collected": "monitor.resources.collected",
    }

    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        collection_interval_s: int = 30,
        history_size: int = 1000,
        bus_dir: Optional[str] = None,
    ):
        """Initialize resource monitor.

        Args:
            thresholds: Resource thresholds for alerts
            collection_interval_s: Collection interval in seconds
            history_size: Number of samples to retain
            bus_dir: Directory for bus events
        """
        self.thresholds = thresholds or ResourceThresholds()
        self.collection_interval_s = collection_interval_s
        self.history_size = history_size

        # History storage
        self._cpu_history: List[CPUMetrics] = []
        self._memory_history: List[MemoryMetrics] = []
        self._disk_history: Dict[str, List[DiskMetrics]] = {}

        # State
        self._running = False
        self._collection_task: Optional[asyncio.Task] = None
        self._last_collection: Optional[float] = None
        self._alert_callbacks: List[Callable[[ResourceAlert], None]] = []

        # Previous values for rate calculations
        self._prev_cpu_times: Optional[Dict[str, float]] = None
        self._prev_disk_io: Dict[str, Dict[str, float]] = {}

        # Bus path
        pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        self._bus_dir = bus_dir or os.path.join(pluribus_root, ".pluribus", "bus")
        self._bus_path = Path(self._bus_dir) / "events.ndjson"
        self._bus_path.parent.mkdir(parents=True, exist_ok=True)

    async def start(self) -> bool:
        """Start the resource monitor.

        Returns:
            True if started successfully
        """
        if self._running:
            return False

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())

        self._emit_bus_event(
            "monitor.resources.started",
            {
                "collection_interval_s": self.collection_interval_s,
                "thresholds": asdict(self.thresholds),
            }
        )

        return True

    async def stop(self) -> bool:
        """Stop the resource monitor.

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

        return True

    def collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics.

        Returns:
            Current resource metrics
        """
        cpu = self._collect_cpu()
        memory = self._collect_memory()
        disks = self._collect_disks()

        metrics = ResourceMetrics(
            cpu=cpu,
            memory=memory,
            disks=disks,
        )

        # Store in history
        self._cpu_history.append(cpu)
        if len(self._cpu_history) > self.history_size:
            self._cpu_history = self._cpu_history[-self.history_size:]

        self._memory_history.append(memory)
        if len(self._memory_history) > self.history_size:
            self._memory_history = self._memory_history[-self.history_size:]

        for disk in disks:
            if disk.mount_point not in self._disk_history:
                self._disk_history[disk.mount_point] = []
            self._disk_history[disk.mount_point].append(disk)
            if len(self._disk_history[disk.mount_point]) > self.history_size:
                self._disk_history[disk.mount_point] = self._disk_history[disk.mount_point][-self.history_size:]

        self._last_collection = time.time()

        # Emit collected event
        self._emit_bus_event(
            self.BUS_TOPICS["collected"],
            metrics.to_dict()
        )

        # Check for alerts
        alerts = self._check_alerts(metrics)
        for alert in alerts:
            self._emit_bus_event(
                self.BUS_TOPICS["alert"],
                alert.to_dict(),
                level=alert.severity.value
            )
            for callback in self._alert_callbacks:
                callback(alert)

        return metrics

    def get_cpu_history(self, window_s: int = 3600) -> List[CPUMetrics]:
        """Get CPU history within time window.

        Args:
            window_s: Time window in seconds

        Returns:
            List of CPU metrics
        """
        cutoff = time.time() - window_s
        return [m for m in self._cpu_history if m.timestamp >= cutoff]

    def get_memory_history(self, window_s: int = 3600) -> List[MemoryMetrics]:
        """Get memory history within time window.

        Args:
            window_s: Time window in seconds

        Returns:
            List of memory metrics
        """
        cutoff = time.time() - window_s
        return [m for m in self._memory_history if m.timestamp >= cutoff]

    def get_disk_history(
        self,
        mount_point: str,
        window_s: int = 3600
    ) -> List[DiskMetrics]:
        """Get disk history for mount point.

        Args:
            mount_point: Mount point to get history for
            window_s: Time window in seconds

        Returns:
            List of disk metrics
        """
        if mount_point not in self._disk_history:
            return []
        cutoff = time.time() - window_s
        return [m for m in self._disk_history[mount_point] if m.timestamp >= cutoff]

    def get_average_cpu(self, window_s: int = 300) -> float:
        """Get average CPU usage over window.

        Args:
            window_s: Time window in seconds

        Returns:
            Average CPU usage percentage
        """
        history = self.get_cpu_history(window_s)
        if not history:
            return 0.0
        return sum(m.usage_percent for m in history) / len(history)

    def get_average_memory(self, window_s: int = 300) -> float:
        """Get average memory usage over window.

        Args:
            window_s: Time window in seconds

        Returns:
            Average memory usage percentage
        """
        history = self.get_memory_history(window_s)
        if not history:
            return 0.0
        return sum(m.usage_percent for m in history) / len(history)

    def register_alert_callback(
        self,
        callback: Callable[[ResourceAlert], None]
    ) -> None:
        """Register a callback for resource alerts.

        Args:
            callback: Function to call on alert
        """
        self._alert_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get monitor status.

        Returns:
            Status dictionary
        """
        return {
            "running": self._running,
            "last_collection": self._last_collection,
            "collection_interval_s": self.collection_interval_s,
            "history_sizes": {
                "cpu": len(self._cpu_history),
                "memory": len(self._memory_history),
                "disks": {k: len(v) for k, v in self._disk_history.items()},
            },
            "thresholds": asdict(self.thresholds),
        }

    def _collect_cpu(self) -> CPUMetrics:
        """Collect CPU metrics."""
        try:
            # Read /proc/stat for CPU times
            with open("/proc/stat", "r") as f:
                line = f.readline()

            parts = line.split()
            # cpu user nice system idle iowait irq softirq steal
            user = int(parts[1])
            nice = int(parts[2])
            system = int(parts[3])
            idle = int(parts[4])
            iowait = int(parts[5]) if len(parts) > 5 else 0

            current_times = {
                "user": user + nice,
                "system": system,
                "idle": idle,
                "iowait": iowait,
            }

            total = sum(current_times.values())

            # Calculate percentages if we have previous values
            if self._prev_cpu_times:
                prev_total = sum(self._prev_cpu_times.values())
                delta_total = total - prev_total

                if delta_total > 0:
                    user_percent = 100.0 * (current_times["user"] - self._prev_cpu_times["user"]) / delta_total
                    system_percent = 100.0 * (current_times["system"] - self._prev_cpu_times["system"]) / delta_total
                    idle_percent = 100.0 * (current_times["idle"] - self._prev_cpu_times["idle"]) / delta_total
                    iowait_percent = 100.0 * (current_times["iowait"] - self._prev_cpu_times["iowait"]) / delta_total
                    usage_percent = 100.0 - idle_percent
                else:
                    user_percent = system_percent = idle_percent = iowait_percent = 0.0
                    usage_percent = 0.0
            else:
                # First reading - estimate from current totals
                if total > 0:
                    user_percent = 100.0 * current_times["user"] / total
                    system_percent = 100.0 * current_times["system"] / total
                    idle_percent = 100.0 * current_times["idle"] / total
                    iowait_percent = 100.0 * current_times["iowait"] / total
                    usage_percent = 100.0 - idle_percent
                else:
                    user_percent = system_percent = idle_percent = iowait_percent = 0.0
                    usage_percent = 0.0

            self._prev_cpu_times = current_times

            # Read load averages
            with open("/proc/loadavg", "r") as f:
                load_parts = f.read().split()
            load_1 = float(load_parts[0])
            load_5 = float(load_parts[1])
            load_15 = float(load_parts[2])

            # Get core count
            core_count = os.cpu_count() or 1

            return CPUMetrics(
                usage_percent=max(0.0, min(100.0, usage_percent)),
                user_percent=max(0.0, min(100.0, user_percent)),
                system_percent=max(0.0, min(100.0, system_percent)),
                idle_percent=max(0.0, min(100.0, idle_percent)),
                iowait_percent=max(0.0, min(100.0, iowait_percent)),
                load_avg_1m=load_1,
                load_avg_5m=load_5,
                load_avg_15m=load_15,
                core_count=core_count,
            )
        except Exception:
            # Fallback to basic metrics
            return CPUMetrics(
                usage_percent=0.0,
                core_count=os.cpu_count() or 1,
            )

    def _collect_memory(self) -> MemoryMetrics:
        """Collect memory metrics."""
        try:
            meminfo = {}
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(":")
                    value = int(parts[1]) * 1024  # Convert KB to bytes
                    meminfo[key] = value

            total = meminfo.get("MemTotal", 0)
            available = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
            cached = meminfo.get("Cached", 0)
            buffers = meminfo.get("Buffers", 0)
            used = total - available

            swap_total = meminfo.get("SwapTotal", 0)
            swap_free = meminfo.get("SwapFree", 0)
            swap_used = swap_total - swap_free
            swap_percent = 100.0 * swap_used / swap_total if swap_total > 0 else 0.0

            usage_percent = 100.0 * used / total if total > 0 else 0.0

            return MemoryMetrics(
                total_bytes=total,
                available_bytes=available,
                used_bytes=used,
                usage_percent=usage_percent,
                cached_bytes=cached,
                buffers_bytes=buffers,
                swap_total_bytes=swap_total,
                swap_used_bytes=swap_used,
                swap_percent=swap_percent,
            )
        except Exception:
            return MemoryMetrics(
                total_bytes=0,
                available_bytes=0,
                used_bytes=0,
                usage_percent=0.0,
            )

    def _collect_disks(self) -> List[DiskMetrics]:
        """Collect disk metrics."""
        disks = []

        try:
            # Read mount points
            mounts = []
            with open("/proc/mounts", "r") as f:
                for line in f:
                    parts = line.split()
                    device = parts[0]
                    mount_point = parts[1]
                    fs_type = parts[2]

                    # Filter to real filesystems
                    if fs_type in ("ext4", "xfs", "btrfs", "zfs", "ext3", "overlay"):
                        mounts.append((device, mount_point))

            for device, mount_point in mounts:
                try:
                    stat = os.statvfs(mount_point)
                    total = stat.f_blocks * stat.f_frsize
                    free = stat.f_bavail * stat.f_frsize
                    used = total - free
                    usage_percent = 100.0 * used / total if total > 0 else 0.0

                    inodes_total = stat.f_files
                    inodes_free = stat.f_ffree
                    inodes_used = inodes_total - inodes_free
                    inodes_percent = 100.0 * inodes_used / inodes_total if inodes_total > 0 else 0.0

                    disks.append(DiskMetrics(
                        mount_point=mount_point,
                        device=device,
                        total_bytes=total,
                        used_bytes=used,
                        free_bytes=free,
                        usage_percent=usage_percent,
                        inodes_total=inodes_total,
                        inodes_used=inodes_used,
                        inodes_percent=inodes_percent,
                    ))
                except (OSError, PermissionError):
                    continue
        except Exception:
            pass

        return disks

    def _check_alerts(self, metrics: ResourceMetrics) -> List[ResourceAlert]:
        """Check metrics against thresholds.

        Args:
            metrics: Current metrics

        Returns:
            List of alerts
        """
        alerts = []

        # CPU alerts
        if metrics.cpu.usage_percent >= self.thresholds.cpu_critical:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.CPU,
                severity=AlertSeverity.CRITICAL,
                message=f"CPU usage critical: {metrics.cpu.usage_percent:.1f}%",
                current_value=metrics.cpu.usage_percent,
                threshold=self.thresholds.cpu_critical,
            ))
        elif metrics.cpu.usage_percent >= self.thresholds.cpu_warning:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.CPU,
                severity=AlertSeverity.WARNING,
                message=f"CPU usage high: {metrics.cpu.usage_percent:.1f}%",
                current_value=metrics.cpu.usage_percent,
                threshold=self.thresholds.cpu_warning,
            ))

        # Load average alert
        load_threshold = metrics.cpu.core_count * self.thresholds.load_warning_factor
        if metrics.cpu.load_avg_1m >= load_threshold:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.CPU,
                severity=AlertSeverity.WARNING,
                message=f"Load average high: {metrics.cpu.load_avg_1m:.2f} (threshold: {load_threshold:.2f})",
                current_value=metrics.cpu.load_avg_1m,
                threshold=load_threshold,
                resource_id="load_avg",
            ))

        # Memory alerts
        if metrics.memory.usage_percent >= self.thresholds.memory_critical:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.MEMORY,
                severity=AlertSeverity.CRITICAL,
                message=f"Memory usage critical: {metrics.memory.usage_percent:.1f}%",
                current_value=metrics.memory.usage_percent,
                threshold=self.thresholds.memory_critical,
            ))
        elif metrics.memory.usage_percent >= self.thresholds.memory_warning:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.MEMORY,
                severity=AlertSeverity.WARNING,
                message=f"Memory usage high: {metrics.memory.usage_percent:.1f}%",
                current_value=metrics.memory.usage_percent,
                threshold=self.thresholds.memory_warning,
            ))

        # Swap alert
        if metrics.memory.swap_percent >= self.thresholds.swap_warning:
            alerts.append(ResourceAlert(
                resource_type=ResourceType.SWAP,
                severity=AlertSeverity.WARNING,
                message=f"Swap usage high: {metrics.memory.swap_percent:.1f}%",
                current_value=metrics.memory.swap_percent,
                threshold=self.thresholds.swap_warning,
            ))

        # Disk alerts
        for disk in metrics.disks:
            if disk.usage_percent >= self.thresholds.disk_critical:
                alerts.append(ResourceAlert(
                    resource_type=ResourceType.DISK,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Disk {disk.mount_point} critical: {disk.usage_percent:.1f}%",
                    current_value=disk.usage_percent,
                    threshold=self.thresholds.disk_critical,
                    resource_id=disk.mount_point,
                ))
            elif disk.usage_percent >= self.thresholds.disk_warning:
                alerts.append(ResourceAlert(
                    resource_type=ResourceType.DISK,
                    severity=AlertSeverity.WARNING,
                    message=f"Disk {disk.mount_point} high: {disk.usage_percent:.1f}%",
                    current_value=disk.usage_percent,
                    threshold=self.thresholds.disk_warning,
                    resource_id=disk.mount_point,
                ))

        return alerts

    async def _collection_loop(self) -> None:
        """Background collection loop."""
        while self._running:
            try:
                self.collect_metrics()
            except Exception as e:
                self._emit_bus_event(
                    "monitor.resources.error",
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
        """Emit event to the Pluribus bus.

        Args:
            topic: Event topic
            data: Event data
            level: Log level
            kind: Event kind

        Returns:
            Event ID
        """
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


# Singleton instance
_monitor: Optional[ResourceMonitor] = None


def get_resource_monitor() -> ResourceMonitor:
    """Get or create the resource monitor singleton.

    Returns:
        ResourceMonitor instance
    """
    global _monitor
    if _monitor is None:
        _monitor = ResourceMonitor()
    return _monitor


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resource Monitor (Step 261)")
    parser.add_argument("--collect", action="store_true", help="Collect metrics once")
    parser.add_argument("--watch", action="store_true", help="Watch metrics continuously")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval in seconds")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    monitor = get_resource_monitor()

    if args.collect:
        metrics = monitor.collect_metrics()
        if args.json:
            print(json.dumps(metrics.to_dict(), indent=2))
        else:
            print("Resource Metrics:")
            print(f"  CPU: {metrics.cpu.usage_percent:.1f}% (load: {metrics.cpu.load_avg_1m:.2f})")
            print(f"  Memory: {metrics.memory.usage_percent:.1f}% ({metrics.memory.used_gb:.1f}/{metrics.memory.total_gb:.1f} GB)")
            for disk in metrics.disks:
                print(f"  Disk {disk.mount_point}: {disk.usage_percent:.1f}% ({disk.free_gb:.1f} GB free)")

            alerts = metrics.get_alerts()
            if alerts:
                print("\nAlerts:")
                for alert in alerts:
                    print(f"  [{alert.severity.value.upper()}] {alert.message}")

    if args.watch:
        import signal
        import sys

        def signal_handler(sig, frame):
            print("\nStopping...")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        print(f"Watching resources every {args.interval}s (Ctrl+C to stop)")
        while True:
            metrics = monitor.collect_metrics()
            if args.json:
                print(json.dumps(metrics.to_dict()))
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"CPU: {metrics.cpu.usage_percent:5.1f}% | "
                      f"MEM: {metrics.memory.usage_percent:5.1f}% | "
                      f"LOAD: {metrics.cpu.load_avg_1m:5.2f}")
            time.sleep(args.interval)
