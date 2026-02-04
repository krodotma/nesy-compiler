#!/usr/bin/env python3
"""Pluribus Membrane Health - Check health of all membrane adapters.

DKIN v28 Remediation: Step 52

This tool checks the health of all membrane integrations:
- CLI_WRAPPER adapters (subprocess-based)
- MCP_BOUNDARY adapters (MCP servers)
- External tool availability

Usage:
    python3 nucleus/tools/membrane_health.py           # Check all
    python3 nucleus/tools/membrane_health.py --json    # JSON output
    python3 nucleus/tools/membrane_health.py --emit    # Emit bus events
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


class IntegrationType(str, Enum):
    CLI_WRAPPER = "CLI_WRAPPER"
    MCP_BOUNDARY = "MCP_BOUNDARY"
    DEEP_INTEGRATION = "DEEP_INTEGRATION"


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class AdapterHealth:
    """Health status of a membrane adapter."""
    name: str
    type: IntegrationType
    status: HealthStatus
    healthy: bool
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "status": self.status.value,
            "healthy": self.healthy,
            "details": self.details,
            "latency_ms": self.latency_ms,
            "error": self.error,
        }


class MembraneHealthChecker:
    """Checks health of all membrane adapters."""

    # Adapter definitions
    ADAPTERS = [
        {
            "name": "maestro",
            "type": IntegrationType.CLI_WRAPPER,
            "binary": "maestro",
            "adapter_file": "maestro_adapter.py",
            "membrane_dir": "maestro",
        },
        {
            "name": "agent-s",
            "type": IntegrationType.CLI_WRAPPER,
            "binary": None,  # Python-based
            "adapter_file": "agent_s_adapter.py",
            "membrane_dir": "agent-s",
        },
        {
            "name": "agent0",
            "type": IntegrationType.CLI_WRAPPER,
            "binary": None,  # Python-based
            "adapter_file": "agent0_adapter.py",
            "membrane_dir": "agent0",
        },
        {
            "name": "graphiti",
            "type": IntegrationType.MCP_BOUNDARY,
            "binary": None,
            "adapter_file": None,
            "mcp_server": "graphiti_server.py",
            "membrane_dir": "graphiti",
        },
        {
            "name": "mem0",
            "type": IntegrationType.MCP_BOUNDARY,
            "binary": None,
            "adapter_file": "mem0_adapter.py",
            "mcp_server": "mem0_server.py",
            "membrane_dir": "mem0-fork",
        },
        {
            "name": "codex",
            "type": IntegrationType.CLI_WRAPPER,
            "binary": "codex",
            "adapter_file": "codex_adapter.py",
            "membrane_dir": None,
        },
        {
            "name": "crush",
            "type": IntegrationType.CLI_WRAPPER,
            "binary": "crush",
            "adapter_file": "crush_adapter.py",
            "membrane_dir": None,
        },
    ]

    def __init__(
        self,
        tools_dir: Path | None = None,
        membrane_dir: Path | None = None,
        emit_bus_events: bool = False,
    ):
        """Initialize the health checker.

        Args:
            tools_dir: Path to nucleus/tools
            membrane_dir: Path to membrane directory
            emit_bus_events: Whether to emit bus events
        """
        self.tools_dir = tools_dir or Path("/pluribus/nucleus/tools")
        self.membrane_dir = membrane_dir or Path("/pluribus/membrane")
        self.emit_bus_events = emit_bus_events
        self._bus_emit = None

        if emit_bus_events:
            self._init_bus_emitter()

    def _init_bus_emitter(self) -> None:
        """Initialize bus event emitter."""
        try:
            if str(self.tools_dir) not in sys.path:
                sys.path.insert(0, str(self.tools_dir))

            from agent_bus import emit_bus_event as emit_event, resolve_bus_paths
            self._bus_paths = resolve_bus_paths(None)
            self._bus_emit = emit_event
        except (ImportError, Exception):
            self._bus_emit = None

    def _emit_event(self, health: AdapterHealth) -> None:
        """Emit bus event for adapter health."""
        if not self._bus_emit:
            return

        try:
            self._bus_emit(
                self._bus_paths,
                topic=f"membrane.health.{health.name}",
                kind="log",
                level="info" if health.healthy else "warn",
                actor="membrane-health",
                data=health.to_dict(),
                durable=True,
            )
        except Exception:
            pass

    def check_binary(self, name: str) -> tuple[bool, str | None]:
        """Check if a binary is available in PATH."""
        if not name:
            return True, None

        path = shutil.which(name)
        if path:
            return True, path
        return False, None

    def check_adapter_file(self, filename: str) -> tuple[bool, Path | None]:
        """Check if adapter file exists."""
        if not filename:
            return True, None

        path = self.tools_dir / filename
        if path.exists():
            return True, path
        return False, None

    def check_membrane_dir(self, dirname: str) -> tuple[bool, Path | None]:
        """Check if membrane directory exists."""
        if not dirname:
            return True, None

        path = self.membrane_dir / dirname
        if path.exists() and path.is_dir():
            return True, path
        return False, None

    def check_mcp_server(self, filename: str) -> tuple[bool, Path | None]:
        """Check if MCP server file exists."""
        if not filename:
            return True, None

        mcp_dir = self.tools_dir.parent / "mcp"
        path = mcp_dir / filename
        if path.exists():
            return True, path
        return False, None

    def check_adapter(self, adapter: dict) -> AdapterHealth:
        """Check health of a single adapter."""
        start = time.time()
        details = {}
        errors = []

        # Check binary
        if adapter.get("binary"):
            found, path = self.check_binary(adapter["binary"])
            details["binary_found"] = found
            if path:
                details["binary_path"] = str(path)
            if not found:
                errors.append(f"Binary '{adapter['binary']}' not found in PATH")

        # Check adapter file
        if adapter.get("adapter_file"):
            found, path = self.check_adapter_file(adapter["adapter_file"])
            details["adapter_file_found"] = found
            if path:
                details["adapter_file_path"] = str(path)
            if not found:
                errors.append(f"Adapter file '{adapter['adapter_file']}' not found")

        # Check membrane directory
        if adapter.get("membrane_dir"):
            found, path = self.check_membrane_dir(adapter["membrane_dir"])
            details["membrane_dir_found"] = found
            if path:
                details["membrane_dir_path"] = str(path)
            if not found:
                errors.append(f"Membrane dir '{adapter['membrane_dir']}' not found")

        # Check MCP server
        if adapter.get("mcp_server"):
            found, path = self.check_mcp_server(adapter["mcp_server"])
            details["mcp_server_found"] = found
            if path:
                details["mcp_server_path"] = str(path)
            if not found:
                errors.append(f"MCP server '{adapter['mcp_server']}' not found")

        latency_ms = (time.time() - start) * 1000

        # Determine status
        if not errors:
            status = HealthStatus.HEALTHY
            healthy = True
        elif len(errors) == 1:
            status = HealthStatus.DEGRADED
            healthy = False
        else:
            status = HealthStatus.UNHEALTHY
            healthy = False

        health = AdapterHealth(
            name=adapter["name"],
            type=adapter["type"],
            status=status,
            healthy=healthy,
            details=details,
            latency_ms=round(latency_ms, 2),
            error="; ".join(errors) if errors else None,
        )

        if self.emit_bus_events:
            self._emit_event(health)

        return health

    def check_all(self) -> list[AdapterHealth]:
        """Check health of all adapters."""
        results = []
        for adapter in self.ADAPTERS:
            try:
                health = self.check_adapter(adapter)
                results.append(health)
            except Exception as e:
                results.append(AdapterHealth(
                    name=adapter["name"],
                    type=adapter["type"],
                    status=HealthStatus.UNKNOWN,
                    healthy=False,
                    error=str(e),
                ))
        return results

    def get_summary(self, results: list[AdapterHealth]) -> dict[str, Any]:
        """Generate summary from health check results."""
        healthy_count = sum(1 for r in results if r.healthy)
        total = len(results)

        return {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_adapters": total,
            "healthy": healthy_count,
            "unhealthy": total - healthy_count,
            "health_percentage": round(healthy_count / total * 100, 1) if total > 0 else 0,
            "by_status": {
                status.value: sum(1 for r in results if r.status == status)
                for status in HealthStatus
            },
            "adapters": [r.to_dict() for r in results],
        }


def format_table(results: list[AdapterHealth]) -> str:
    """Format results as ASCII table."""
    lines = [
        "MEMBRANE HEALTH CHECK",
        "=" * 70,
        f"{'Adapter':<15} {'Type':<15} {'Status':<12} {'Details'}",
        "-" * 70,
    ]

    for r in results:
        symbol = "\u2713" if r.healthy else "\u2717"
        type_short = r.type.value.split("_")[0]
        detail = r.error or "OK"
        if len(detail) > 30:
            detail = detail[:27] + "..."
        lines.append(f"{symbol} {r.name:<13} {type_short:<15} {r.status.value:<12} {detail}")

    lines.append("-" * 70)
    healthy = sum(1 for r in results if r.healthy)
    lines.append(f"Total: {len(results)} | Healthy: {healthy} | Unhealthy: {len(results) - healthy}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Membrane Health Checker")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--emit", action="store_true", help="Emit bus events")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    checker = MembraneHealthChecker(emit_bus_events=args.emit)
    results = checker.check_all()

    if args.json:
        summary = checker.get_summary(results)
        print(json.dumps(summary, indent=2))
    elif args.summary:
        summary = checker.get_summary(results)
        print(f"Health: {summary['healthy']}/{summary['total_adapters']} ({summary['health_percentage']}%)")
    else:
        print(format_table(results))


if __name__ == "__main__":
    main()
