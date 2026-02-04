#!/usr/bin/env python3
"""Pluribus Architecture Counter - Automated codebase metrics.

DKIN v28 Remediation: Step 73

This tool generates accurate counts for ARCH-TOP-TO-SUBTREES.md:
- Python tools count
- Specs file count
- MCP servers count
- Systemd services count
- Semops operators count
- Axioms count
- Motifs count
- Provider integrations count

Usage:
    python3 nucleus/tools/arch_counter.py           # Show all counts
    python3 nucleus/tools/arch_counter.py --json    # JSON output
    python3 nucleus/tools/arch_counter.py --emit    # Emit bus event
    python3 nucleus/tools/arch_counter.py --update  # Update ARCH doc
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


@dataclass
class ArchCounts:
    """Architecture metrics."""
    # Tools
    python_tools: int = 0
    tools_by_category: dict[str, int] = field(default_factory=dict)

    # Specs
    spec_files_total: int = 0
    spec_files_md: int = 0
    spec_files_json: int = 0

    # MCP
    mcp_servers: int = 0
    mcp_python_files: int = 0

    # Deploy
    systemd_services: int = 0
    systemd_timers: int = 0

    # Semops
    semops_operators: int = 0

    # Axioms
    axiom_declarations: int = 0

    # Motifs
    omega_motifs: int = 0

    # Providers
    provider_integrations: int = 0

    # Membrane
    membrane_adapters: int = 0

    # Dashboard
    dashboard_components: int = 0

    # Timestamp
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "tools": {
                "python_total": self.python_tools,
                "by_category": self.tools_by_category,
            },
            "specs": {
                "total": self.spec_files_total,
                "markdown": self.spec_files_md,
                "json": self.spec_files_json,
            },
            "mcp": {
                "servers": self.mcp_servers,
                "python_files": self.mcp_python_files,
            },
            "deploy": {
                "systemd_services": self.systemd_services,
                "systemd_timers": self.systemd_timers,
                "total": self.systemd_services + self.systemd_timers,
            },
            "semops_operators": self.semops_operators,
            "axiom_declarations": self.axiom_declarations,
            "omega_motifs": self.omega_motifs,
            "provider_integrations": self.provider_integrations,
            "membrane_adapters": self.membrane_adapters,
            "dashboard_components": self.dashboard_components,
        }


class ArchCounter:
    """Counts architecture components for ARCH-TOP-TO-SUBTREES.md."""

    def __init__(self, root: Path | None = None):
        self.root = root or Path("/pluribus")
        self.nucleus = self.root / "nucleus"

    def count_python_tools(self) -> tuple[int, dict[str, int]]:
        """Count Python files in nucleus/tools/."""
        tools_dir = self.nucleus / "tools"
        if not tools_dir.exists():
            return 0, {}

        total = 0
        by_category = {
            "operators": 0,
            "daemons": 0,
            "tests": 0,
            "hooks": 0,
            "providers": 0,
            "other": 0,
        }

        for py_file in tools_dir.rglob("*.py"):
            total += 1
            name = py_file.name

            if "operator" in name:
                by_category["operators"] += 1
            elif "daemon" in name or name.startswith("oiterate"):
                by_category["daemons"] += 1
            elif "test_" in name or "_test" in name:
                by_category["tests"] += 1
            elif py_file.parent.name == "hooks":
                by_category["hooks"] += 1
            elif py_file.parent.name == "providers":
                by_category["providers"] += 1
            else:
                by_category["other"] += 1

        return total, by_category

    def count_specs(self) -> tuple[int, int, int]:
        """Count spec files in nucleus/specs/."""
        specs_dir = self.nucleus / "specs"
        if not specs_dir.exists():
            return 0, 0, 0

        md_count = len(list(specs_dir.rglob("*.md")))
        json_count = len(list(specs_dir.rglob("*.json")))
        total = md_count + json_count

        return total, md_count, json_count

    def count_mcp_servers(self) -> tuple[int, int]:
        """Count MCP servers in nucleus/mcp/."""
        mcp_dir = self.nucleus / "mcp"
        if not mcp_dir.exists():
            return 0, 0

        py_files = list(mcp_dir.glob("*.py"))
        servers = sum(1 for f in py_files if "server" in f.name.lower())

        return servers, len(py_files)

    def count_systemd(self) -> tuple[int, int]:
        """Count systemd services and timers."""
        systemd_dir = self.nucleus / "deploy" / "systemd"
        if not systemd_dir.exists():
            return 0, 0

        services = len(list(systemd_dir.glob("*.service")))
        timers = len(list(systemd_dir.glob("*.timer")))

        return services, timers

    def count_semops_operators(self) -> int:
        """Count operators in semops.json."""
        semops_path = self.nucleus / "specs" / "semops.json"
        if not semops_path.exists():
            return 0

        try:
            with open(semops_path, "r") as f:
                data = json.load(f)
            return len(data.get("operators", {}))
        except (json.JSONDecodeError, IOError):
            return 0

    def count_axioms(self) -> int:
        """Count axiom declarations."""
        axioms_dir = self.nucleus / "specs" / "axioms"
        if not axioms_dir.exists():
            return 0

        count = 0
        for axiom_file in axioms_dir.glob("*.axiom"):
            try:
                content = axiom_file.read_text()
                # Count AXIOM declarations
                count += len(re.findall(r"^AXIOM\s+", content, re.MULTILINE))
            except IOError:
                continue

        return count

    def count_motifs(self) -> int:
        """Count omega motifs."""
        motifs_path = self.nucleus / "specs" / "omega_motifs.json"
        if not motifs_path.exists():
            return 0

        try:
            with open(motifs_path, "r") as f:
                data = json.load(f)
            return len(data.get("motifs", []))
        except (json.JSONDecodeError, IOError):
            return 0

    def count_providers(self) -> int:
        """Count provider integrations."""
        providers_dir = self.nucleus / "tools" / "providers"
        if not providers_dir.exists():
            return 0

        # Count unique provider smoke tests
        return len(list(providers_dir.glob("*_smoke.py")))

    def count_membrane_adapters(self) -> int:
        """Count membrane adapters."""
        tools_dir = self.nucleus / "tools"
        if not tools_dir.exists():
            return 0

        return len(list(tools_dir.glob("*_adapter.py")))

    def count_dashboard_components(self) -> int:
        """Count dashboard React/TSX components."""
        dashboard_dir = self.nucleus / "dashboard" / "src" / "components"
        if not dashboard_dir.exists():
            return 0

        return len(list(dashboard_dir.rglob("*.tsx")))

    def count_all(self) -> ArchCounts:
        """Generate all architecture counts."""
        counts = ArchCounts()
        counts.generated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Tools
        counts.python_tools, counts.tools_by_category = self.count_python_tools()

        # Specs
        counts.spec_files_total, counts.spec_files_md, counts.spec_files_json = self.count_specs()

        # MCP
        counts.mcp_servers, counts.mcp_python_files = self.count_mcp_servers()

        # Systemd
        counts.systemd_services, counts.systemd_timers = self.count_systemd()

        # Semops
        counts.semops_operators = self.count_semops_operators()

        # Axioms
        counts.axiom_declarations = self.count_axioms()

        # Motifs
        counts.omega_motifs = self.count_motifs()

        # Providers
        counts.provider_integrations = self.count_providers()

        # Membrane
        counts.membrane_adapters = self.count_membrane_adapters()

        # Dashboard
        counts.dashboard_components = self.count_dashboard_components()

        return counts


def format_report(counts: ArchCounts) -> str:
    """Format counts as readable report."""
    lines = [
        "PLURIBUS ARCHITECTURE COUNTS",
        f"Generated: {counts.generated_at}",
        "=" * 50,
        "",
        "NUCLEUS/TOOLS:",
        f"  Python files: {counts.python_tools}",
    ]

    for cat, count in counts.tools_by_category.items():
        lines.append(f"    - {cat}: {count}")

    lines.extend([
        "",
        "NUCLEUS/SPECS:",
        f"  Total: {counts.spec_files_total}",
        f"    - Markdown: {counts.spec_files_md}",
        f"    - JSON: {counts.spec_files_json}",
        "",
        "NUCLEUS/MCP:",
        f"  Servers: {counts.mcp_servers}",
        f"  Python files: {counts.mcp_python_files}",
        "",
        "NUCLEUS/DEPLOY/SYSTEMD:",
        f"  Services: {counts.systemd_services}",
        f"  Timers: {counts.systemd_timers}",
        f"  Total: {counts.systemd_services + counts.systemd_timers}",
        "",
        "REGISTRIES:",
        f"  Semops operators: {counts.semops_operators}",
        f"  Axiom declarations: {counts.axiom_declarations}",
        f"  Omega motifs: {counts.omega_motifs}",
        f"  Provider integrations: {counts.provider_integrations}",
        "",
        "MEMBRANE:",
        f"  Adapters: {counts.membrane_adapters}",
        "",
        "DASHBOARD:",
        f"  TSX components: {counts.dashboard_components}",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Architecture Counter")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--emit", action="store_true", help="Emit bus event")
    parser.add_argument("--update", action="store_true", help="Update ARCH doc")

    args = parser.parse_args()

    counter = ArchCounter()
    counts = counter.count_all()

    if args.json:
        print(json.dumps(counts.to_dict(), indent=2))
    elif args.emit:
        # Emit bus event
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from agent_bus import emit_bus_event as emit_event, resolve_bus_paths
            bus_paths = resolve_bus_paths(None)
            emit_event(
                bus_paths,
                topic="arch.counts.updated",
                kind="metric",
                level="info",
                actor="arch-counter",
                data=counts.to_dict(),
                durable=True,
            )
            print("Emitted arch.counts.updated event")
        except Exception as e:
            print(f"Failed to emit: {e}")
    else:
        print(format_report(counts))


if __name__ == "__main__":
    main()
