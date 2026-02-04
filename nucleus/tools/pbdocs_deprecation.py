#!/usr/bin/env python3
"""PBDOCS Deprecation - Component Lifecycle Tracking.

DKIN v28 Compliant

Tracks component lifecycle states:
- stable: Production-ready, actively maintained
- experimental: In development, API may change
- deprecated: Scheduled for removal, use replacement
- archived: Removed from active use, preserved

Integrates with:
- mkdocstrings (status badges)
- Bus events (lifecycle.deprecation.*)
- PBDOCS operator (audit integration)

Usage:
    python3 nucleus/tools/pbdocs_deprecation.py scan         # Scan for markers
    python3 nucleus/tools/pbdocs_deprecation.py report       # Generate report
    python3 nucleus/tools/pbdocs_deprecation.py tag PATH STATUS  # Tag component
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

sys.dont_write_bytecode = True


LifecycleStatus = Literal["stable", "experimental", "deprecated", "archived"]


@dataclass
class LifecycleTag:
    """Component lifecycle metadata."""
    path: str
    status: LifecycleStatus
    since: str
    reason: str
    replacement: str | None = None
    removal_date: str | None = None
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "since": self.since,
            "reason": self.reason,
            "replacement": self.replacement,
            "removal_date": self.removal_date,
            "detected_at": self.detected_at,
        }

    def to_badge(self) -> str:
        """Generate MkDocs badge markup."""
        colors = {
            "stable": "green",
            "experimental": "yellow",
            "deprecated": "red",
            "archived": "gray",
        }
        return f'<span class="status-{self.status}">{self.status.upper()}</span>'


@dataclass
class LifecycleRegistry:
    """Registry of all component lifecycle states."""
    components: dict[str, LifecycleTag] = field(default_factory=dict)
    version: str = "1.0.0"
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add(self, tag: LifecycleTag) -> None:
        self.components[tag.path] = tag
        self.updated_at = datetime.utcnow().isoformat()

    def get(self, path: str) -> LifecycleTag | None:
        return self.components.get(path)

    def filter_by_status(self, status: LifecycleStatus) -> list[LifecycleTag]:
        return [t for t in self.components.values() if t.status == status]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "component_count": len(self.components),
            "by_status": {
                "stable": len(self.filter_by_status("stable")),
                "experimental": len(self.filter_by_status("experimental")),
                "deprecated": len(self.filter_by_status("deprecated")),
                "archived": len(self.filter_by_status("archived")),
            },
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "LifecycleRegistry":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        registry = cls(version=data.get("version", "1.0.0"))
        for comp_path, comp_data in data.get("components", {}).items():
            registry.add(LifecycleTag(**comp_data))
        return registry


class DeprecationScanner:
    """Scans codebase for lifecycle markers."""

    # Detection patterns
    PATTERNS = {
        "deprecated": [
            r"@deprecated",
            r"# DEPRECATED",
            r"# deprecated:",
            r"DEPRECATION_WARNING",
            r"DeprecationWarning",
            r"\[DEPRECATED\]",
            r"\.deprecated\s*=\s*True",
        ],
        "experimental": [
            r"@experimental",
            r"# EXPERIMENTAL",
            r"# experimental:",
            r"# WIP",
            r"# TODO:",
            r"\[EXPERIMENTAL\]",
            r"NotImplementedError",
        ],
        "archived": [
            r"# ARCHIVED",
            r"# archived:",
            r"\[ARCHIVED\]",
            r"\.archived\s*=\s*True",
        ],
    }

    # Files to skip
    SKIP_PATTERNS = [
        r"__pycache__",
        r"\.git",
        r"node_modules",
        r"\.venv",
        r"\.pyc$",
    ]

    def __init__(self, root: Path | None = None):
        self.root = root or Path("/pluribus")
        self.nucleus = self.root / "nucleus"

    def should_skip(self, path: Path) -> bool:
        path_str = str(path)
        return any(re.search(p, path_str) for p in self.SKIP_PATTERNS)

    def detect_status(self, content: str) -> LifecycleStatus | None:
        """Detect lifecycle status from file content."""
        for status, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return status
        return None

    def extract_metadata(self, content: str, status: LifecycleStatus) -> dict[str, str]:
        """Extract metadata from markers."""
        metadata = {
            "reason": f"Detected {status} marker in code",
            "since": "Unknown",
            "replacement": None,
        }

        # Try to extract reason
        reason_patterns = [
            rf"#\s*{status}:\s*(.+)",
            rf"@{status}\s*\((.+)\)",
            rf"{status}:\s*(.+)",
        ]
        for pattern in reason_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["reason"] = match.group(1).strip()
                break

        # Try to extract replacement
        replace_patterns = [
            r"use\s+(\S+)\s+instead",
            r"replaced\s+by\s+(\S+)",
            r"replacement:\s*(\S+)",
        ]
        for pattern in replace_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata["replacement"] = match.group(1).strip()
                break

        return metadata

    def scan_file(self, path: Path) -> LifecycleTag | None:
        """Scan single file for lifecycle markers."""
        if self.should_skip(path):
            return None

        try:
            content = path.read_text(errors="ignore")
        except Exception:
            return None

        status = self.detect_status(content)
        if not status:
            return None

        metadata = self.extract_metadata(content, status)

        return LifecycleTag(
            path=str(path.relative_to(self.root)),
            status=status,
            since=metadata["since"],
            reason=metadata["reason"],
            replacement=metadata["replacement"],
        )

    def scan_directory(self, directory: Path, extensions: list[str] = None) -> list[LifecycleTag]:
        """Scan directory for lifecycle markers."""
        extensions = extensions or [".py", ".ts", ".tsx", ".md"]
        tags = []

        for ext in extensions:
            for path in directory.rglob(f"*{ext}"):
                if self.should_skip(path):
                    continue
                tag = self.scan_file(path)
                if tag:
                    tags.append(tag)

        return tags

    def scan_tools(self) -> list[LifecycleTag]:
        """Scan nucleus/tools for lifecycle markers."""
        return self.scan_directory(self.nucleus / "tools", [".py"])

    def scan_specs(self) -> list[LifecycleTag]:
        """Scan nucleus/specs for lifecycle markers."""
        return self.scan_directory(self.nucleus / "specs", [".md", ".json"])

    def scan_dashboard(self) -> list[LifecycleTag]:
        """Scan nucleus/dashboard for lifecycle markers."""
        return self.scan_directory(self.nucleus / "dashboard", [".ts", ".tsx"])

    def scan_all(self) -> LifecycleRegistry:
        """Scan entire codebase."""
        registry = LifecycleRegistry()

        for tag in self.scan_tools():
            registry.add(tag)

        for tag in self.scan_specs():
            registry.add(tag)

        for tag in self.scan_dashboard():
            registry.add(tag)

        return registry


class DeprecationReporter:
    """Generates deprecation reports."""

    def __init__(self, registry: LifecycleRegistry):
        self.registry = registry

    def to_markdown(self) -> str:
        lines = [
            "# Component Lifecycle Report",
            "",
            f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Total Components:** {len(self.registry.components)}",
            "",
            "## Summary",
            "",
            "| Status | Count |",
            "|--------|-------|",
        ]

        for status in ["stable", "experimental", "deprecated", "archived"]:
            count = len(self.registry.filter_by_status(status))
            lines.append(f"| {status.capitalize()} | {count} |")

        # Deprecated section
        deprecated = self.registry.filter_by_status("deprecated")
        if deprecated:
            lines.extend([
                "",
                "## Deprecated Components",
                "",
                "| Component | Reason | Replacement |",
                "|-----------|--------|-------------|",
            ])
            for tag in deprecated:
                replacement = tag.replacement or "-"
                lines.append(f"| `{tag.path}` | {tag.reason[:50]} | {replacement} |")

        # Experimental section
        experimental = self.registry.filter_by_status("experimental")
        if experimental:
            lines.extend([
                "",
                "## Experimental Components",
                "",
                "| Component | Reason |",
                "|-----------|--------|",
            ])
            for tag in experimental[:20]:
                lines.append(f"| `{tag.path}` | {tag.reason[:50]} |")
            if len(experimental) > 20:
                lines.append(f"| ... | +{len(experimental) - 20} more |")

        lines.extend([
            "",
            "---",
            "*Generated by PBDOCS Deprecation Scanner - DKIN v28*",
        ])

        return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PBDOCS Deprecation Scanner")
    parser.add_argument("command", choices=["scan", "report", "tag"], help="Command to run")
    parser.add_argument("--path", help="Path to scan or tag")
    parser.add_argument("--status", choices=["stable", "experimental", "deprecated", "archived"])
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()

    root = Path("/pluribus")
    registry_path = root / ".pluribus" / "state" / "lifecycle_registry.json"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "scan":
        scanner = DeprecationScanner(root)
        registry = scanner.scan_all()
        registry.save(registry_path)

        if args.json:
            print(json.dumps(registry.to_dict(), indent=2))
        else:
            print(f"Scanned {len(registry.components)} components with lifecycle markers")
            for status in ["deprecated", "experimental", "archived"]:
                count = len(registry.filter_by_status(status))
                if count:
                    print(f"  - {status}: {count}")

    elif args.command == "report":
        registry = LifecycleRegistry.load(registry_path)
        reporter = DeprecationReporter(registry)

        if args.json:
            print(json.dumps(registry.to_dict(), indent=2))
        else:
            print(reporter.to_markdown())

    elif args.command == "tag":
        if not args.path or not args.status:
            print("Error: --path and --status required for tag command")
            sys.exit(1)

        registry = LifecycleRegistry.load(registry_path)
        tag = LifecycleTag(
            path=args.path,
            status=args.status,
            since=datetime.utcnow().strftime("%Y-%m-%d"),
            reason="Manually tagged",
        )
        registry.add(tag)
        registry.save(registry_path)
        print(f"Tagged {args.path} as {args.status}")


if __name__ == "__main__":
    main()
