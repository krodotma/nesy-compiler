#!/usr/bin/env python3
"""
affected_projects.py - Nx-Equivalent Affected Project Detection

First-principles Git-native implementation using iso_git.mjs (isomorphic-git).
Provides affected detection without GitHub dependency.

DKIN v29 | PAIP v15 | CITIZEN v1

Features:
- Changed file detection via iso_git.mjs tree comparison
- Project mapping with dependency graph traversal
- Transitive affect propagation
- Bus event emission for observability
- Cache-friendly output (stable ordering)

Usage:
    python3 affected_projects.py                          # Compare HEAD to origin/main
    python3 affected_projects.py --base main --head HEAD  # Custom refs
    python3 affected_projects.py --target build           # Filter by target
    python3 affected_projects.py --json                   # JSON output
    python3 affected_projects.py --emit-bus               # Emit to Pluribus bus

Reference: nucleus/specs/projects.json
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

# =============================================================================
# Constants
# =============================================================================

PROTOCOL_VERSION = "29"
OPERATOR_NAME = "PBAFFECTED"

# Paths
TOOLS_DIR = Path(__file__).parent.resolve()
NUCLEUS_DIR = TOOLS_DIR.parent
REPO_ROOT = NUCLEUS_DIR.parent
PROJECTS_JSON = NUCLEUS_DIR / "specs" / "projects.json"
ISO_GIT_PATH = TOOLS_DIR / "iso_git.mjs"


# =============================================================================
# Helpers
# =============================================================================

def now_ts() -> float:
    return time.time()


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_bus_dir() -> Path:
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if bus_dir:
        return Path(bus_dir).expanduser().resolve()
    root = os.environ.get("PLURIBUS_ROOT") or str(REPO_ROOT)
    return Path(root) / ".pluribus" / "bus"


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or "pbaffected"


# =============================================================================
# File I/O
# =============================================================================

try:
    import fcntl
    def lock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    def unlock_file(handle) -> None:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
except ImportError:
    def lock_file(handle) -> None: pass
    def unlock_file(handle) -> None: pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        lock_file(f)
        try:
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
        finally:
            unlock_file(f)


def emit_bus_event(topic: str, kind: str, level: str, actor: str, data: dict) -> str:
    evt_id = str(uuid.uuid4())
    evt = {
        "id": evt_id,
        "ts": now_ts(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    bus_path = resolve_bus_dir() / "events.ndjson"
    append_ndjson(bus_path, evt)
    return evt_id


# =============================================================================
# Project Registry
# =============================================================================

@dataclass
class Project:
    """Represents a project in the workspace."""
    id: str
    root: str
    type: str
    language: str
    targets: dict[str, dict]
    dependencies: list[str]
    inputs: list[str]
    tags: list[str]

    def matches_file(self, filepath: str) -> bool:
        """Check if a file belongs to this project."""
        # Direct path match
        if filepath.startswith(self.root + "/") or filepath == self.root:
            return True
        # Input glob patterns
        for pattern in self.inputs:
            if fnmatch.fnmatch(filepath, pattern):
                return True
        return False


@dataclass
class ProjectGraph:
    """Project dependency graph."""
    projects: dict[str, Project]
    implicit_deps: dict[str, list[str]]
    workspace_config: dict

    def get_dependents(self, project_id: str) -> list[str]:
        """Get all projects that depend on the given project (reverse deps)."""
        dependents = []
        for pid, proj in self.projects.items():
            if project_id in proj.dependencies:
                dependents.append(pid)
        return dependents

    def get_transitive_dependents(self, project_ids: set[str]) -> set[str]:
        """Get all transitively affected projects (BFS)."""
        affected = set(project_ids)
        queue = deque(project_ids)

        while queue:
            current = queue.popleft()
            for dependent in self.get_dependents(current):
                if dependent not in affected:
                    affected.add(dependent)
                    queue.append(dependent)

        return affected

    def find_project_for_file(self, filepath: str) -> Optional[str]:
        """Find which project owns a file."""
        # Check direct path containment (most specific first)
        candidates = []
        for pid, proj in self.projects.items():
            if filepath.startswith(proj.root + "/"):
                candidates.append((len(proj.root), pid))

        if candidates:
            # Return most specific (longest path match)
            candidates.sort(reverse=True)
            return candidates[0][1]

        # Check input patterns
        for pid, proj in self.projects.items():
            if proj.matches_file(filepath):
                return pid

        return None

    def check_implicit_deps(self, filepath: str) -> list[str]:
        """Check if a file triggers implicit dependencies."""
        affected = []
        for pattern, targets in self.implicit_deps.items():
            if fnmatch.fnmatch(filepath, pattern) or filepath == pattern:
                if "*" in targets:
                    # Affects all projects
                    affected.extend(self.projects.keys())
                else:
                    for target in targets:
                        if target.endswith("/*"):
                            # Pattern match
                            prefix = target[:-2]
                            affected.extend(
                                pid for pid in self.projects.keys()
                                if pid.startswith(prefix)
                            )
                        else:
                            affected.append(target)
        return affected


def load_project_graph() -> ProjectGraph:
    """Load project graph from projects.json."""
    if not PROJECTS_JSON.exists():
        raise FileNotFoundError(f"Project registry not found: {PROJECTS_JSON}")

    data = json.loads(PROJECTS_JSON.read_text(encoding="utf-8"))

    projects = {}
    for key, proj_data in data.get("projects", {}).items():
        projects[key] = Project(
            id=proj_data.get("id", key),
            root=proj_data.get("root", key),
            type=proj_data.get("type", "library"),
            language=proj_data.get("language", "unknown"),
            targets=proj_data.get("targets", {}),
            dependencies=proj_data.get("dependencies", []),
            inputs=proj_data.get("inputs", []),
            tags=proj_data.get("tags", []),
        )

    return ProjectGraph(
        projects=projects,
        implicit_deps=data.get("implicit_dependencies", {}),
        workspace_config=data.get("workspace", {}),
    )


# =============================================================================
# Changed File Detection
# =============================================================================

def get_changed_files_git(base: str, head: str, repo_dir: Path) -> list[str]:
    """Get changed files using native git diff (fallback)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", base, head],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        pass
    return []


def get_changed_files_iso_git(base: str, head: str, repo_dir: Path) -> list[str]:
    """Get changed files using iso_git.mjs (first-principles, no GitHub)."""
    if not ISO_GIT_PATH.exists():
        # Fallback to native git
        return get_changed_files_git(base, head, repo_dir)

    try:
        # Use iso_git.mjs diff command
        result = subprocess.run(
            ["node", str(ISO_GIT_PATH), "diff", "--base", base, "--head", head, "--json"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PLURIBUS_ROOT": str(repo_dir)},
        )

        if result.returncode == 0 and result.stdout.strip():
            try:
                data = json.loads(result.stdout)
                return data.get("changed_files", [])
            except json.JSONDecodeError:
                # Parse line-by-line output
                return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]

    except subprocess.TimeoutExpired:
        pass
    except Exception:
        pass

    # Fallback to native git
    return get_changed_files_git(base, head, repo_dir)


def get_changed_files(base: str, head: str, repo_dir: Path) -> list[str]:
    """Get changed files between refs, preferring iso_git.mjs."""
    # Try iso_git first (first-principles), then fallback to native git
    files = get_changed_files_iso_git(base, head, repo_dir)
    if not files:
        files = get_changed_files_git(base, head, repo_dir)

    # Filter out skippable paths
    skip_dirs = {".git", ".pluribus", ".pluribus_local", "node_modules", ".venv", "__pycache__"}
    filtered = []
    for f in files:
        parts = f.split("/")
        if not any(p in skip_dirs for p in parts):
            filtered.append(f)

    return sorted(set(filtered))


# =============================================================================
# Affected Detection
# =============================================================================

@dataclass
class AffectedProject:
    """Represents an affected project with reason."""
    id: str
    root: str
    reason: str
    direct: bool
    changed_files: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "root": self.root,
            "reason": self.reason,
            "direct": self.direct,
            "changed_files": self.changed_files,
            "targets": self.targets,
        }


@dataclass
class AffectedReport:
    """Complete affected analysis report."""
    req_id: str
    base: str
    head: str
    timestamp: str
    changed_files: list[str]
    affected: list[AffectedProject]
    all_targets: list[str]

    @property
    def counts(self) -> dict:
        return {
            "changed_files": len(self.changed_files),
            "affected_projects": len(self.affected),
            "direct": sum(1 for a in self.affected if a.direct),
            "transitive": sum(1 for a in self.affected if not a.direct),
        }

    def to_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "base": self.base,
            "head": self.head,
            "timestamp": self.timestamp,
            "counts": self.counts,
            "changed_files": self.changed_files,
            "affected": [a.to_dict() for a in self.affected],
            "all_targets": self.all_targets,
            "protocol_version": PROTOCOL_VERSION,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Affected Projects Report",
            "",
            f"**Base:** `{self.base}`",
            f"**Head:** `{self.head}`",
            f"**Timestamp:** {self.timestamp}",
            "",
            "## Summary",
            "",
            f"| Metric | Count |",
            f"|--------|-------|",
            f"| Changed Files | {self.counts['changed_files']} |",
            f"| Affected Projects | {self.counts['affected_projects']} |",
            f"| Direct | {self.counts['direct']} |",
            f"| Transitive | {self.counts['transitive']} |",
            "",
        ]

        if self.affected:
            lines.extend([
                "## Affected Projects",
                "",
                "| Project | Type | Targets | Reason |",
                "|---------|------|---------|--------|",
            ])
            for a in self.affected:
                type_icon = "D" if a.direct else "T"
                targets = ", ".join(a.targets[:3]) or "-"
                lines.append(f"| `{a.id}` | {type_icon} | {targets} | {a.reason} |")
            lines.append("")

        if self.changed_files:
            lines.extend([
                "## Changed Files",
                "",
            ])
            for f in self.changed_files[:20]:
                lines.append(f"- `{f}`")
            if len(self.changed_files) > 20:
                lines.append(f"- ... and {len(self.changed_files) - 20} more")
            lines.append("")

        lines.extend([
            "---",
            f"*Generated by PBAFFECTED - DKIN v{PROTOCOL_VERSION}*",
        ])

        return "\n".join(lines)


def detect_affected(
    base: str,
    head: str,
    repo_dir: Path,
    target_filter: Optional[str] = None,
) -> AffectedReport:
    """Detect affected projects between two refs."""
    req_id = f"affected-{uuid.uuid4().hex[:12]}"

    # Load project graph
    graph = load_project_graph()

    # Get changed files
    changed_files = get_changed_files(base, head, repo_dir)

    # Map files to projects
    directly_affected: dict[str, list[str]] = {}  # project_id -> [files]
    implicit_affected: set[str] = set()

    for filepath in changed_files:
        # Check implicit dependencies first
        implicit_projects = graph.check_implicit_deps(filepath)
        implicit_affected.update(implicit_projects)

        # Find owning project
        project_id = graph.find_project_for_file(filepath)
        if project_id:
            if project_id not in directly_affected:
                directly_affected[project_id] = []
            directly_affected[project_id].append(filepath)

    # Combine direct and implicit
    all_direct = set(directly_affected.keys()) | implicit_affected

    # Get transitive dependents
    all_affected = graph.get_transitive_dependents(all_direct)

    # Build affected report
    affected_list: list[AffectedProject] = []
    all_targets: set[str] = set()

    for project_id in sorted(all_affected):
        if project_id not in graph.projects:
            continue

        proj = graph.projects[project_id]
        is_direct = project_id in all_direct
        files = directly_affected.get(project_id, [])

        # Determine reason
        if project_id in directly_affected:
            reason = f"Direct: {len(files)} file(s) changed"
        elif project_id in implicit_affected:
            reason = "Implicit dependency triggered"
        else:
            # Find which direct affect caused this
            deps_affected = [d for d in proj.dependencies if d in all_direct]
            if deps_affected:
                reason = f"Transitive: depends on {', '.join(deps_affected[:2])}"
            else:
                reason = "Transitive dependency"

        # Get targets
        targets = list(proj.targets.keys())

        # Filter by target if specified
        if target_filter and target_filter not in targets:
            continue

        all_targets.update(targets)

        affected_list.append(AffectedProject(
            id=project_id,
            root=proj.root,
            reason=reason,
            direct=is_direct,
            changed_files=files,
            targets=targets,
        ))

    return AffectedReport(
        req_id=req_id,
        base=base,
        head=head,
        timestamp=now_iso_utc(),
        changed_files=changed_files,
        affected=affected_list,
        all_targets=sorted(all_targets),
    )


# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="affected_projects.py",
        description="PBAFFECTED - Nx-equivalent affected project detection (iso_git native)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect affected projects (HEAD vs origin/main)
  python3 affected_projects.py

  # Custom base and head refs
  python3 affected_projects.py --base main --head feature-branch

  # Filter by target
  python3 affected_projects.py --target test

  # JSON output for scripting
  python3 affected_projects.py --json

  # Emit to Pluribus bus
  python3 affected_projects.py --emit-bus

  # List only project names (for nx run-many equivalent)
  python3 affected_projects.py --names-only
""",
    )
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Base ref for comparison (default: origin/main)",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="Head ref for comparison (default: HEAD)",
    )
    parser.add_argument(
        "--target",
        help="Filter to projects with specific target (build, test, lint)",
    )
    parser.add_argument(
        "--format",
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Shorthand for --format json",
    )
    parser.add_argument(
        "--names-only",
        action="store_true",
        help="Output only project names (one per line)",
    )
    parser.add_argument(
        "--emit-bus",
        action="store_true",
        help="Emit report to Pluribus bus",
    )
    parser.add_argument(
        "--repo",
        default=str(REPO_ROOT),
        help=f"Repository root (default: {REPO_ROOT})",
    )
    return parser


def format_text(report: AffectedReport) -> str:
    lines = [
        "=" * 60,
        f"PBAFFECTED - Affected Project Detection",
        "=" * 60,
        "",
        f"Base: {report.base}",
        f"Head: {report.head}",
        f"Changed Files: {report.counts['changed_files']}",
        f"Affected Projects: {report.counts['affected_projects']} ({report.counts['direct']} direct, {report.counts['transitive']} transitive)",
        "",
    ]

    if report.affected:
        lines.append("AFFECTED PROJECTS:")
        for a in report.affected:
            icon = "D" if a.direct else "T"
            targets_str = ", ".join(a.targets[:3]) if a.targets else "-"
            lines.append(f"  [{icon}] {a.id:30} targets: {targets_str}")
            if a.direct and a.changed_files:
                for f in a.changed_files[:3]:
                    lines.append(f"        > {f}")
                if len(a.changed_files) > 3:
                    lines.append(f"        > ... +{len(a.changed_files) - 3} more")
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    repo_dir = Path(args.repo).resolve()

    # Detect affected projects
    report = detect_affected(
        base=args.base,
        head=args.head,
        repo_dir=repo_dir,
        target_filter=args.target,
    )

    # Emit to bus if requested
    if args.emit_bus:
        emit_bus_event(
            topic="operator.pbaffected.report",
            kind="metric",
            level="info",
            actor=default_actor(),
            data=report.to_dict(),
        )

    # Output
    if args.names_only:
        for a in report.affected:
            print(a.id)
    elif args.json or args.format == "json":
        print(json.dumps(report.to_dict(), indent=2))
    elif args.format == "markdown":
        print(report.to_markdown())
    else:
        print(format_text(report))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
