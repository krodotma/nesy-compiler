#!/usr/bin/env python3
"""PBDOCS Installation Audit - Dependency and Integration Tracer.

DKIN v28 Compliant

Traces and audits all installation requirements:
- Python dependencies across all packages
- Node.js dependencies in package.json files
- System packages referenced in scripts
- Systemd service requirements
- Database and external service dependencies
- Agent-created integrations

Usage:
    python3 nucleus/tools/pbdocs_install_audit.py scan       # Full scan
    python3 nucleus/tools/pbdocs_install_audit.py python     # Python deps
    python3 nucleus/tools/pbdocs_install_audit.py node       # Node.js deps
    python3 nucleus/tools/pbdocs_install_audit.py services   # Systemd services
    python3 nucleus/tools/pbdocs_install_audit.py report     # Generate report
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


@dataclass
class PythonDependency:
    """Python package dependency."""
    name: str
    version: str | None
    source: str  # File that requires it
    optional: bool = False
    extras: list[str] = field(default_factory=list)


@dataclass
class NodeDependency:
    """Node.js package dependency."""
    name: str
    version: str
    source: str
    dev: bool = False
    peer: bool = False


@dataclass
class SystemDependency:
    """System-level dependency."""
    name: str
    source: str
    install_cmd: str | None = None
    detected_in: str = ""


@dataclass
class ServiceDependency:
    """Systemd service dependency."""
    name: str
    exec_start: str
    working_dir: str | None
    dependencies: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    requires_python: list[str] = field(default_factory=list)


@dataclass
class InstallationAudit:
    """Full installation audit report."""
    audit_id: str
    timestamp: str
    python_deps: list[PythonDependency] = field(default_factory=list)
    node_deps: list[NodeDependency] = field(default_factory=list)
    system_deps: list[SystemDependency] = field(default_factory=list)
    services: list[ServiceDependency] = field(default_factory=list)
    missing_docs: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "python_deps_count": len(self.python_deps),
            "node_deps_count": len(self.node_deps),
            "system_deps_count": len(self.system_deps),
            "services_count": len(self.services),
            "missing_docs_count": len(self.missing_docs),
            "python_deps": [
                {"name": d.name, "version": d.version, "source": d.source, "optional": d.optional}
                for d in self.python_deps
            ],
            "node_deps": [
                {"name": d.name, "version": d.version, "source": d.source, "dev": d.dev}
                for d in self.node_deps
            ],
            "system_deps": [
                {"name": d.name, "source": d.source, "install_cmd": d.install_cmd}
                for d in self.system_deps
            ],
            "services": [
                {"name": s.name, "exec_start": s.exec_start, "dependencies": s.dependencies}
                for s in self.services
            ],
            "missing_docs": self.missing_docs,
            "recommendations": self.recommendations,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Installation Requirements Audit",
            "",
            f"**Audit ID:** `{self.audit_id[:8]}`",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Summary",
            "",
            "| Category | Count |",
            "|----------|-------|",
            f"| Python Dependencies | {len(self.python_deps)} |",
            f"| Node.js Dependencies | {len(self.node_deps)} |",
            f"| System Dependencies | {len(self.system_deps)} |",
            f"| Systemd Services | {len(self.services)} |",
            f"| Missing Documentation | {len(self.missing_docs)} |",
            "",
        ]

        # Python section
        if self.python_deps:
            lines.extend([
                "## Python Dependencies",
                "",
                "| Package | Version | Source |",
                "|---------|---------|--------|",
            ])
            seen = set()
            for dep in sorted(self.python_deps, key=lambda d: d.name.lower()):
                if dep.name not in seen:
                    seen.add(dep.name)
                    ver = dep.version or "any"
                    lines.append(f"| {dep.name} | {ver} | `{dep.source}` |")
            lines.append("")

        # Node.js section
        if self.node_deps:
            lines.extend([
                "## Node.js Dependencies",
                "",
                "| Package | Version | Type | Source |",
                "|---------|---------|------|--------|",
            ])
            for dep in sorted(self.node_deps, key=lambda d: d.name.lower()):
                dep_type = "dev" if dep.dev else "prod"
                lines.append(f"| {dep.name} | {dep.version} | {dep_type} | `{dep.source}` |")
            lines.append("")

        # System deps section
        if self.system_deps:
            lines.extend([
                "## System Dependencies",
                "",
                "| Package | Install Command | Source |",
                "|---------|-----------------|--------|",
            ])
            for dep in self.system_deps:
                cmd = dep.install_cmd or "N/A"
                lines.append(f"| {dep.name} | `{cmd}` | `{dep.source}` |")
            lines.append("")

        # Services section
        if self.services:
            lines.extend([
                "## Systemd Services",
                "",
                "| Service | ExecStart | Dependencies |",
                "|---------|-----------|--------------|",
            ])
            for svc in self.services:
                deps = ", ".join(svc.dependencies[:3]) if svc.dependencies else "-"
                lines.append(f"| {svc.name} | `{svc.exec_start[:40]}...` | {deps} |")
            lines.append("")

        # Recommendations
        if self.recommendations:
            lines.extend([
                "## Recommendations",
                "",
            ])
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Missing docs
        if self.missing_docs:
            lines.extend([
                "## Missing Documentation",
                "",
            ])
            for doc in self.missing_docs[:10]:
                lines.append(f"- {doc}")
            if len(self.missing_docs) > 10:
                lines.append(f"- ... and {len(self.missing_docs) - 10} more")

        lines.extend([
            "",
            "---",
            "*Generated by PBDOCS Installation Audit - DKIN v28*",
        ])

        return "\n".join(lines)


class InstallationAuditor:
    """Audits installation requirements across the codebase."""

    # Common system packages
    SYSTEM_PACKAGES = {
        "git": "apt install git",
        "curl": "apt install curl",
        "jq": "apt install jq",
        "ripgrep": "apt install ripgrep",
        "fzf": "apt install fzf",
        "gh": "apt install gh",
        "node": "apt install nodejs",
        "npm": "apt install npm",
        "python3": "apt install python3",
        "pip": "apt install python3-pip",
        "ollama": "curl -fsSL https://ollama.com/install.sh | sh",
        "postgresql": "apt install postgresql",
        "neo4j": "apt install neo4j",
        "xvfb": "apt install xvfb",
        "chromium": "apt install chromium-browser",
        "tigervnc": "apt install tigervnc-standalone-server",
        "caddy": "apt install caddy",
    }

    def __init__(self, root: Path | None = None):
        self.root = root or Path("/pluribus")
        self.nucleus = self.root / "nucleus"

    def scan_python_requirements(self) -> list[PythonDependency]:
        """Scan all requirements.txt and pyproject.toml files."""
        deps = []

        # Find all requirements files
        for req_file in self.root.rglob("requirements*.txt"):
            deps.extend(self._parse_requirements_txt(req_file))

        # Find all pyproject.toml files
        for pyproject in self.root.rglob("pyproject.toml"):
            deps.extend(self._parse_pyproject_toml(pyproject))

        return deps

    def _parse_requirements_txt(self, path: Path) -> list[PythonDependency]:
        """Parse requirements.txt file."""
        deps = []
        try:
            content = path.read_text()
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("-"):
                    continue

                # Parse package spec
                match = re.match(r"([a-zA-Z0-9_-]+)([<>=!]+)?(.+)?", line)
                if match:
                    name = match.group(1)
                    version = f"{match.group(2) or ''}{match.group(3) or ''}".strip() or None
                    deps.append(PythonDependency(
                        name=name,
                        version=version,
                        source=str(path.relative_to(self.root)),
                    ))
        except Exception:
            pass
        return deps

    def _parse_pyproject_toml(self, path: Path) -> list[PythonDependency]:
        """Parse pyproject.toml for dependencies."""
        deps = []
        try:
            content = path.read_text()

            # Look for dependencies section
            in_deps = False
            for line in content.split("\n"):
                if "[project.dependencies]" in line or "[tool.poetry.dependencies]" in line:
                    in_deps = True
                    continue
                if in_deps:
                    if line.startswith("["):
                        break
                    match = re.match(r'"?([a-zA-Z0-9_-]+)"?\s*[=<>]', line)
                    if match:
                        deps.append(PythonDependency(
                            name=match.group(1),
                            version=None,
                            source=str(path.relative_to(self.root)),
                        ))
        except Exception:
            pass
        return deps

    def scan_node_dependencies(self) -> list[NodeDependency]:
        """Scan all package.json files."""
        deps = []

        for pkg_json in self.root.rglob("package.json"):
            # Skip node_modules
            if "node_modules" in str(pkg_json):
                continue

            try:
                data = json.loads(pkg_json.read_text())
                source = str(pkg_json.relative_to(self.root))

                for name, version in data.get("dependencies", {}).items():
                    deps.append(NodeDependency(name=name, version=version, source=source, dev=False))

                for name, version in data.get("devDependencies", {}).items():
                    deps.append(NodeDependency(name=name, version=version, source=source, dev=True))

                for name, version in data.get("peerDependencies", {}).items():
                    deps.append(NodeDependency(name=name, version=version, source=source, peer=True))

            except Exception:
                pass

        return deps

    def scan_system_dependencies(self) -> list[SystemDependency]:
        """Scan scripts for system package references."""
        deps = []
        seen = set()

        # Scan shell scripts
        for script in self.root.rglob("*.sh"):
            try:
                content = script.read_text()

                for pkg, install_cmd in self.SYSTEM_PACKAGES.items():
                    # Check if package is referenced
                    if re.search(rf"\b{pkg}\b", content):
                        if pkg not in seen:
                            seen.add(pkg)
                            deps.append(SystemDependency(
                                name=pkg,
                                source=str(script.relative_to(self.root)),
                                install_cmd=install_cmd,
                            ))

                # Check for apt install commands
                for match in re.finditer(r"apt(?:-get)?\s+install\s+([\w\s-]+)", content):
                    for pkg in match.group(1).split():
                        pkg = pkg.strip()
                        if pkg and pkg not in seen:
                            seen.add(pkg)
                            deps.append(SystemDependency(
                                name=pkg,
                                source=str(script.relative_to(self.root)),
                                install_cmd=f"apt install {pkg}",
                            ))

            except Exception:
                pass

        return deps

    def scan_services(self) -> list[ServiceDependency]:
        """Scan systemd service files."""
        services = []
        services_dir = self.nucleus / "deploy" / "systemd"

        for service_file in services_dir.glob("*.service"):
            try:
                content = service_file.read_text()

                # Parse service file
                exec_start = ""
                working_dir = None
                deps = []
                env = {}

                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("ExecStart="):
                        exec_start = line.split("=", 1)[1]
                    elif line.startswith("WorkingDirectory="):
                        working_dir = line.split("=", 1)[1]
                    elif line.startswith("After="):
                        deps.extend(line.split("=", 1)[1].split())
                    elif line.startswith("Requires="):
                        deps.extend(line.split("=", 1)[1].split())
                    elif line.startswith("Environment="):
                        env_line = line.split("=", 1)[1]
                        if "=" in env_line:
                            key, val = env_line.split("=", 1)
                            env[key.strip('"')] = val.strip('"')

                services.append(ServiceDependency(
                    name=service_file.stem,
                    exec_start=exec_start,
                    working_dir=working_dir,
                    dependencies=deps,
                    environment=env,
                ))

            except Exception:
                pass

        return services

    def check_missing_docs(self) -> list[str]:
        """Check for missing installation documentation."""
        missing = []

        required_docs = [
            "INSTALLATION.md",
            "DEPENDENCIES.md",
            "SYSTEM_REQUIREMENTS.md",
        ]

        for doc in required_docs:
            if not (self.root / doc).exists():
                missing.append(f"Missing: {doc}")

        # Check for undocumented services
        services_dir = self.nucleus / "deploy" / "systemd"
        readme = services_dir / "README.md"

        if readme.exists():
            readme_content = readme.read_text()
            for service in services_dir.glob("*.service"):
                if service.stem not in readme_content:
                    missing.append(f"Service not in README: {service.stem}")

        return missing

    def generate_recommendations(self, audit: InstallationAudit) -> list[str]:
        """Generate installation recommendations."""
        recs = []

        # Python version
        recs.append("Ensure Python 3.10+ is installed")

        # Node version
        if audit.node_deps:
            recs.append("Ensure Node.js 18+ is installed")

        # Database
        if any("postgres" in d.name.lower() or "neo4j" in d.name.lower() for d in audit.system_deps):
            recs.append("Configure database connections in .env.security")

        # Missing docs
        if audit.missing_docs:
            recs.append(f"Create {len(audit.missing_docs)} missing documentation files")

        return recs

    def run_full_audit(self) -> InstallationAudit:
        """Run full installation audit."""
        import uuid

        audit = InstallationAudit(
            audit_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        audit.python_deps = self.scan_python_requirements()
        audit.node_deps = self.scan_node_dependencies()
        audit.system_deps = self.scan_system_dependencies()
        audit.services = self.scan_services()
        audit.missing_docs = self.check_missing_docs()
        audit.recommendations = self.generate_recommendations(audit)

        return audit


def main():
    import argparse

    parser = argparse.ArgumentParser(description="PBDOCS Installation Audit")
    parser.add_argument("command", nargs="?", default="scan",
                       choices=["scan", "python", "node", "services", "report"],
                       help="Command to run")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--output", help="Output file path")

    args = parser.parse_args()

    auditor = InstallationAuditor()

    if args.command == "python":
        deps = auditor.scan_python_requirements()
        for dep in sorted(set(d.name for d in deps)):
            print(dep)

    elif args.command == "node":
        deps = auditor.scan_node_dependencies()
        for dep in sorted(set(d.name for d in deps)):
            print(dep)

    elif args.command == "services":
        services = auditor.scan_services()
        for svc in services:
            print(f"{svc.name}: {svc.exec_start[:60]}...")

    elif args.command in ["scan", "report"]:
        audit = auditor.run_full_audit()

        if args.json:
            output = json.dumps(audit.to_dict(), indent=2)
        else:
            output = audit.to_markdown()

        if args.output:
            Path(args.output).write_text(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)


if __name__ == "__main__":
    main()
