#!/usr/bin/env python3
"""
Membrane Manager
================
Manages external SOTA tool integrations via git submodules and subtrees.
Part of the Clade Manager Protocol (CMP) - handles the "Membrane" layer.

Membrane entries are tracked in `.clade-manifest.json` under the "membrane" key.

Commands:
  add-submodule <name> <remote> [--version <tag>] [--adapter <path>]
  add-subtree <name> <remote> [--version <tag>] [--prefix <dir>]
  update <name> [--version <tag>]
  sync-all
  check-updates
  remove <name>
  list
  verify-adapters
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Literal

sys.dont_write_bytecode = True

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("membrane_manager")


# =============================================================================
# Constants
# =============================================================================

MANIFEST_FILENAME = ".clade-manifest.json"
DEFAULT_MEMBRANE_DIR = "membrane"
DEFAULT_ADAPTERS_DIR = "tools"

# Golden ratio thresholds (from CMP spec)
PHI = 1.618033988749895


# =============================================================================
# Data Structures
# =============================================================================

class MembraneType(str, Enum):
    SUBMODULE = "submodule"
    SUBTREE = "subtree"


class EntryStatus(str, Enum):
    OK = "ok"
    MISSING = "missing"
    DIRTY = "dirty"
    OUTDATED = "outdated"
    ERROR = "error"


@dataclass
class MembraneEntry:
    """A membrane entry representing an external SOTA tool integration."""
    name: str
    type: Literal["submodule", "subtree"]
    remote: str
    pinned: str  # Version tag or commit SHA
    adapter: str | None = None  # Path to adapter file relative to repo root
    prefix: str | None = None  # For subtrees: directory prefix
    last_updated: str | None = None
    upstream_latest: str | None = None  # Latest available version

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = {
            "type": self.type,
            "remote": self.remote,
            "pinned": self.pinned,
        }
        if self.adapter:
            d["adapter"] = self.adapter
        if self.prefix:
            d["prefix"] = self.prefix
        if self.last_updated:
            d["last_updated"] = self.last_updated
        if self.upstream_latest:
            d["upstream_latest"] = self.upstream_latest
        return d

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "MembraneEntry":
        """Create from dictionary."""
        # Normalize type: planned_submodule -> submodule, planned_subtree -> subtree
        raw_type = data.get("type", "submodule")
        if raw_type.startswith("planned_"):
            normalized_type = raw_type.replace("planned_", "")
        else:
            normalized_type = raw_type

        return cls(
            name=name,
            type=normalized_type if normalized_type in ("submodule", "subtree") else "submodule",
            remote=data.get("remote", ""),
            pinned=data.get("pinned") or data.get("recommended_version"),
            adapter=data.get("adapter"),
            prefix=data.get("prefix"),
            last_updated=data.get("last_updated"),
            upstream_latest=data.get("upstream_latest"),
        )


@dataclass
class MembraneStatus:
    """Status of a membrane entry."""
    entry: MembraneEntry
    status: EntryStatus
    message: str = ""
    current_commit: str | None = None
    has_local_changes: bool = False


# =============================================================================
# Git Operations
# =============================================================================

class GitError(Exception):
    """Git command execution error."""
    pass


class GitOps:
    """Git operations wrapper using subprocess."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self._git = shutil.which("git")
        if not self._git:
            raise RuntimeError("git executable not found in PATH")

    def _run(
        self,
        args: list[str],
        cwd: Path | None = None,
        check: bool = True,
        capture: bool = True,
    ) -> subprocess.CompletedProcess:
        """Execute a git command."""
        cmd = [self._git] + args
        work_dir = cwd or self.repo_root
        logger.debug(f"Running: {' '.join(cmd)} in {work_dir}")

        try:
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=capture,
                text=True,
                timeout=120,  # 2 minute timeout
            )
            if check and result.returncode != 0:
                raise GitError(
                    f"git {args[0]} failed (rc={result.returncode}): {result.stderr}"
                )
            return result
        except subprocess.TimeoutExpired as e:
            raise GitError(f"git {args[0]} timed out after 120s") from e

    def is_worktree_clean(self) -> bool:
        """Check if the worktree has uncommitted changes."""
        result = self._run(["status", "--porcelain"], check=False)
        return len(result.stdout.strip()) == 0

    def get_current_branch(self) -> str:
        """Get the current branch name."""
        result = self._run(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def resolve_ref(self, ref: str, remote: str | None = None) -> str | None:
        """Resolve a ref (tag, branch, commit) to a commit SHA."""
        if remote:
            # For remote refs, use ls-remote
            result = self._run(
                ["ls-remote", remote, ref],
                check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Format: <sha>\t<ref>
                return result.stdout.split()[0]
            # Try as exact SHA
            if len(ref) == 40 and all(c in "0123456789abcdef" for c in ref.lower()):
                return ref
            return None
        else:
            result = self._run(["rev-parse", ref], check=False)
            if result.returncode == 0:
                return result.stdout.strip()
            return None

    def get_latest_tag(self, remote: str) -> str | None:
        """Get the latest version tag from a remote."""
        result = self._run(["ls-remote", "--tags", "--sort=-v:refname", remote], check=False)
        if result.returncode != 0:
            return None

        # Parse tags and find the latest semver-like tag
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                ref = parts[1]
                # Extract tag name, skip ^{} dereferenced refs
                if "^{}" in ref:
                    continue
                tag = ref.replace("refs/tags/", "")
                # Prefer semver tags (v0.1.0, 0.1.0, etc.)
                if re.match(r"^v?\d+\.\d+", tag):
                    return tag
        return None

    # =========================================================================
    # Submodule Operations
    # =========================================================================

    def submodule_add(
        self,
        name: str,
        remote: str,
        path: str | None = None,
        branch: str | None = None,
    ) -> None:
        """Add a git submodule."""
        submodule_path = path or f"{DEFAULT_MEMBRANE_DIR}/{name}"
        args = ["submodule", "add"]
        if branch:
            args.extend(["-b", branch])
        args.extend(["--name", name, remote, submodule_path])
        self._run(args)

    def submodule_update(
        self,
        name: str | None = None,
        init: bool = True,
        recursive: bool = True,
    ) -> None:
        """Update submodules."""
        args = ["submodule", "update"]
        if init:
            args.append("--init")
        if recursive:
            args.append("--recursive")
        if name:
            args.append(name)
        self._run(args)

    def submodule_checkout(self, path: str, ref: str) -> None:
        """Checkout a specific ref in a submodule."""
        submodule_dir = self.repo_root / path
        if not submodule_dir.exists():
            raise GitError(f"Submodule path does not exist: {path}")

        self._run(["fetch", "--all", "--tags"], cwd=submodule_dir)
        self._run(["checkout", ref], cwd=submodule_dir)

    def submodule_remove(self, name: str, path: str) -> None:
        """Remove a git submodule."""
        # Deinit the submodule
        self._run(["submodule", "deinit", "-f", path], check=False)
        # Remove from .git/modules
        modules_path = self.repo_root / ".git" / "modules" / name
        if modules_path.exists():
            shutil.rmtree(modules_path)
        # Remove the working tree
        self._run(["rm", "-f", path], check=False)
        # Clean up .gitmodules
        self._run(["config", "-f", ".gitmodules", "--remove-section", f"submodule.{name}"], check=False)

    def get_submodule_commit(self, path: str) -> str | None:
        """Get the current commit of a submodule."""
        submodule_dir = self.repo_root / path
        if not submodule_dir.exists():
            return None
        result = self._run(["rev-parse", "HEAD"], cwd=submodule_dir, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
        return None

    # =========================================================================
    # Subtree Operations
    # =========================================================================

    def subtree_add(
        self,
        prefix: str,
        remote: str,
        ref: str = "main",
        squash: bool = True,
    ) -> None:
        """Add a git subtree."""
        args = ["subtree", "add", "--prefix", prefix]
        if squash:
            args.append("--squash")
        args.extend([remote, ref])
        self._run(args)

    def subtree_pull(
        self,
        prefix: str,
        remote: str,
        ref: str = "main",
        squash: bool = True,
    ) -> None:
        """Pull updates for a git subtree."""
        args = ["subtree", "pull", "--prefix", prefix]
        if squash:
            args.append("--squash")
        args.extend([remote, ref])
        self._run(args)

    def subtree_has_changes(self, prefix: str) -> bool:
        """Check if a subtree has uncommitted changes."""
        result = self._run(["status", "--porcelain", prefix], check=False)
        return len(result.stdout.strip()) > 0


# =============================================================================
# Event Bus Integration
# =============================================================================

def emit_membrane_event(
    topic: str,
    data: dict,
    level: str = "info",
) -> None:
    """Emit an event to the Pluribus bus for membrane changes."""
    try:
        # Try importing from the same package
        try:
            from agent_bus import emit_event, resolve_bus_paths
        except ImportError:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from agent_bus import emit_event, resolve_bus_paths

        paths = resolve_bus_paths(None)
        emit_event(
            paths,
            topic=f"cmp.membrane.{topic}",
            kind="lifecycle",
            level=level,
            actor=os.environ.get("PLURIBUS_ACTOR", "membrane-manager"),
            data=data,
            trace_id=os.environ.get("PLURIBUS_TRACE_ID"),
            run_id=os.environ.get("PLURIBUS_RUN_ID"),
            durable=True,
        )
        logger.debug(f"Emitted event: cmp.membrane.{topic}")
    except Exception as e:
        logger.warning(f"Failed to emit bus event: {e}")


# =============================================================================
# Manifest Management
# =============================================================================

class ManifestManager:
    """Manages the .clade-manifest.json file."""

    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        self.manifest_path = repo_root / MANIFEST_FILENAME

    def _default_manifest(self) -> dict:
        """Create a default manifest structure."""
        return {
            "schema_version": 1,
            "phi": PHI,
            "trunk": "main",
            "fitness_thresholds": {
                "excellent": 1.0,
                "good": 0.618,
                "fair": 0.382,
                "poor": 0.236,
            },
            "clades": {},
            "membrane": {},
            "extinct_archive": [],
        }

    def load(self) -> dict:
        """Load the manifest, creating default if it doesn't exist."""
        if not self.manifest_path.exists():
            logger.info(f"Creating new manifest at {self.manifest_path}")
            return self._default_manifest()

        try:
            with open(self.manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in manifest: {e}")

    def save(self, manifest: dict) -> None:
        """Save the manifest."""
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
            f.write("\n")
        logger.debug(f"Saved manifest to {self.manifest_path}")

    def get_membrane_entries(self) -> dict[str, MembraneEntry]:
        """Get all membrane entries."""
        manifest = self.load()
        membrane = manifest.get("membrane", {})
        return {
            name: MembraneEntry.from_dict(name, data)
            for name, data in membrane.items()
        }

    def set_membrane_entry(self, entry: MembraneEntry) -> None:
        """Add or update a membrane entry."""
        manifest = self.load()
        if "membrane" not in manifest:
            manifest["membrane"] = {}
        manifest["membrane"][entry.name] = entry.to_dict()
        self.save(manifest)

    def remove_membrane_entry(self, name: str) -> bool:
        """Remove a membrane entry. Returns True if entry existed."""
        manifest = self.load()
        if name in manifest.get("membrane", {}):
            del manifest["membrane"][name]
            self.save(manifest)
            return True
        return False


# =============================================================================
# Membrane Manager
# =============================================================================

class MembraneManager:
    """
    Manages external SOTA tool integrations via git submodules and subtrees.

    The membrane layer sits between the nucleus (core code) and the cytoplasm
    (evolutionary clades), providing a controlled interface to external tools.
    """

    def __init__(self, repo_root: Path | str | None = None):
        """
        Initialize the Membrane Manager.

        Args:
            repo_root: Repository root path. If None, will search upward for .git
        """
        self.repo_root = self._find_repo_root(repo_root)
        self.git = GitOps(self.repo_root)
        self.manifest = ManifestManager(self.repo_root)
        logger.info(f"MembraneManager initialized at {self.repo_root}")

    def _find_repo_root(self, provided_root: Path | str | None) -> Path:
        """Find the repository root."""
        if provided_root:
            root = Path(provided_root).resolve()
            if (root / ".git").exists():
                return root
            raise ValueError(f"Not a git repository: {root}")

        # Search upward for .git
        cwd = Path.cwd().resolve()
        for p in [cwd, *cwd.parents]:
            if (p / ".git").exists():
                return p
        raise ValueError("Not inside a git repository")

    def _get_submodule_path(self, name: str) -> str:
        """Get the default submodule path for a name."""
        return f"{DEFAULT_MEMBRANE_DIR}/{name}"

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # =========================================================================
    # Core Operations
    # =========================================================================

    def add_submodule(
        self,
        name: str,
        remote: str,
        version: str | None = None,
        adapter_path: str | None = None,
    ) -> MembraneEntry:
        """
        Add a git submodule for an external tool.

        Args:
            name: Unique name for this membrane entry
            remote: Git remote URL
            version: Tag or commit to pin to (defaults to latest tag)
            adapter_path: Path to adapter file relative to repo root

        Returns:
            The created MembraneEntry

        Raises:
            GitError: If git operations fail
            ValueError: If entry already exists or worktree is dirty
        """
        # Check for existing entry
        entries = self.manifest.get_membrane_entries()
        if name in entries:
            raise ValueError(f"Membrane entry '{name}' already exists")

        # Check for dirty worktree
        if not self.git.is_worktree_clean():
            raise ValueError("Worktree has uncommitted changes. Commit or stash first.")

        # Resolve version
        if not version:
            version = self.git.get_latest_tag(remote)
            if not version:
                version = "main"
            logger.info(f"No version specified, using: {version}")

        # Validate version exists
        resolved = self.git.resolve_ref(version, remote=remote)
        if not resolved:
            raise ValueError(f"Cannot resolve version '{version}' in remote")

        # Add the submodule
        submodule_path = self._get_submodule_path(name)
        logger.info(f"Adding submodule '{name}' from {remote} at {version}")

        self.git.submodule_add(name, remote, submodule_path)
        self.git.submodule_update(init=True)

        # Checkout specific version if it's not a branch
        if version != "main" and version != "master":
            self.git.submodule_checkout(submodule_path, version)

        # Create and save entry
        entry = MembraneEntry(
            name=name,
            type="submodule",
            remote=remote,
            pinned=version,
            adapter=adapter_path,
            last_updated=self._now_iso(),
        )
        self.manifest.set_membrane_entry(entry)

        # Emit event
        emit_membrane_event("added", {
            "name": name,
            "type": "submodule",
            "remote": remote,
            "version": version,
            "adapter": adapter_path,
        })

        logger.info(f"Successfully added submodule '{name}'")
        return entry

    def add_subtree(
        self,
        name: str,
        remote: str,
        version: str | None = None,
        prefix: str | None = None,
        adapter_path: str | None = None,
    ) -> MembraneEntry:
        """
        Add a git subtree for a forked external tool.

        Subtrees are used when we need to maintain our own modifications
        to an external tool while still being able to pull upstream updates.

        Args:
            name: Unique name for this membrane entry
            remote: Git remote URL
            version: Tag or branch to pull from (defaults to main)
            prefix: Directory prefix for the subtree
            adapter_path: Path to adapter file relative to repo root

        Returns:
            The created MembraneEntry
        """
        # Check for existing entry
        entries = self.manifest.get_membrane_entries()
        if name in entries:
            raise ValueError(f"Membrane entry '{name}' already exists")

        # Check for dirty worktree
        if not self.git.is_worktree_clean():
            raise ValueError("Worktree has uncommitted changes. Commit or stash first.")

        # Determine prefix
        subtree_prefix = prefix or f"{DEFAULT_MEMBRANE_DIR}/{name}-fork"

        # Resolve version
        if not version:
            version = self.git.get_latest_tag(remote) or "main"
            logger.info(f"No version specified, using: {version}")

        # Add the subtree
        logger.info(f"Adding subtree '{name}' from {remote} at {version}")
        self.git.subtree_add(subtree_prefix, remote, version, squash=True)

        # Create and save entry
        entry = MembraneEntry(
            name=name,
            type="subtree",
            remote=remote,
            pinned=version,
            prefix=subtree_prefix,
            adapter=adapter_path,
            last_updated=self._now_iso(),
        )
        self.manifest.set_membrane_entry(entry)

        # Emit event
        emit_membrane_event("added", {
            "name": name,
            "type": "subtree",
            "remote": remote,
            "version": version,
            "prefix": subtree_prefix,
            "adapter": adapter_path,
        })

        logger.info(f"Successfully added subtree '{name}'")
        return entry

    def update(
        self,
        name: str,
        version: str | None = None,
    ) -> MembraneEntry:
        """
        Update a membrane entry to a specific version.

        Args:
            name: Name of the membrane entry
            version: Target version (tag/commit). If None, updates to latest tag.

        Returns:
            The updated MembraneEntry
        """
        entries = self.manifest.get_membrane_entries()
        if name not in entries:
            raise ValueError(f"Membrane entry '{name}' not found")

        entry = entries[name]
        old_version = entry.pinned

        # Resolve target version
        if not version:
            version = self.git.get_latest_tag(entry.remote)
            if not version:
                raise ValueError(f"Cannot determine latest version for '{name}'")
            logger.info(f"Updating to latest version: {version}")

        if version == old_version:
            logger.info(f"'{name}' is already at version {version}")
            return entry

        # Validate version exists
        resolved = self.git.resolve_ref(version, remote=entry.remote)
        if not resolved:
            raise ValueError(f"Cannot resolve version '{version}' in remote")

        logger.info(f"Updating '{name}' from {old_version} to {version}")

        if entry.type == "submodule":
            # Update submodule
            submodule_path = self._get_submodule_path(name)
            self.git.submodule_update(init=True)
            self.git.submodule_checkout(submodule_path, version)
        else:
            # Pull subtree
            if not entry.prefix:
                raise ValueError(f"Subtree entry '{name}' missing prefix")
            self.git.subtree_pull(entry.prefix, entry.remote, version, squash=True)

        # Update entry
        entry.pinned = version
        entry.last_updated = self._now_iso()
        self.manifest.set_membrane_entry(entry)

        # Emit event
        emit_membrane_event("updated", {
            "name": name,
            "type": entry.type,
            "old_version": old_version,
            "new_version": version,
        })

        logger.info(f"Successfully updated '{name}' to {version}")
        return entry

    def sync_all(self) -> list[tuple[str, bool, str]]:
        """
        Sync all membrane entries to their pinned versions.

        Returns:
            List of (name, success, message) tuples
        """
        entries = self.manifest.get_membrane_entries()
        results: list[tuple[str, bool, str]] = []

        if not entries:
            logger.info("No membrane entries to sync")
            return results

        logger.info(f"Syncing {len(entries)} membrane entries")

        for name, entry in entries.items():
            try:
                if entry.type == "submodule":
                    submodule_path = self._get_submodule_path(name)
                    self.git.submodule_update(name, init=True)
                    self.git.submodule_checkout(submodule_path, entry.pinned)
                else:
                    if not entry.prefix:
                        raise ValueError(f"Subtree '{name}' missing prefix")
                    # For subtrees, we can only pull if there are no local changes
                    if self.git.subtree_has_changes(entry.prefix):
                        results.append((name, False, "Has uncommitted changes"))
                        continue
                    self.git.subtree_pull(entry.prefix, entry.remote, entry.pinned, squash=True)

                entry.last_updated = self._now_iso()
                self.manifest.set_membrane_entry(entry)
                results.append((name, True, f"Synced to {entry.pinned}"))
                logger.info(f"Synced '{name}' to {entry.pinned}")

            except Exception as e:
                results.append((name, False, str(e)))
                logger.error(f"Failed to sync '{name}': {e}")

        # Emit summary event
        emit_membrane_event("synced", {
            "total": len(entries),
            "successful": sum(1 for _, success, _ in results if success),
            "failed": sum(1 for _, success, _ in results if not success),
        })

        return results

    def check_updates(self) -> list[tuple[str, str, str | None]]:
        """
        Check for available upstream updates.

        Returns:
            List of (name, current_version, latest_version) tuples.
            latest_version is None if check failed.
        """
        entries = self.manifest.get_membrane_entries()
        results: list[tuple[str, str, str | None]] = []

        logger.info(f"Checking updates for {len(entries)} membrane entries")

        for name, entry in entries.items():
            try:
                latest = self.git.get_latest_tag(entry.remote)
                if latest and latest != entry.pinned:
                    logger.info(f"'{name}': update available {entry.pinned} -> {latest}")
                    entry.upstream_latest = latest
                    self.manifest.set_membrane_entry(entry)
                results.append((name, entry.pinned, latest))
            except Exception as e:
                logger.error(f"Failed to check updates for '{name}': {e}")
                results.append((name, entry.pinned, None))

        return results

    def remove(self, name: str, force: bool = False) -> bool:
        """
        Remove a membrane entry.

        Args:
            name: Name of the entry to remove
            force: If True, remove even if worktree is dirty

        Returns:
            True if removed successfully
        """
        entries = self.manifest.get_membrane_entries()
        if name not in entries:
            raise ValueError(f"Membrane entry '{name}' not found")

        if not force and not self.git.is_worktree_clean():
            raise ValueError("Worktree has uncommitted changes. Use --force to override.")

        entry = entries[name]
        logger.info(f"Removing membrane entry '{name}'")

        if entry.type == "submodule":
            submodule_path = self._get_submodule_path(name)
            self.git.submodule_remove(name, submodule_path)
        else:
            # For subtrees, we just remove the directory
            if entry.prefix:
                prefix_path = self.repo_root / entry.prefix
                if prefix_path.exists():
                    shutil.rmtree(prefix_path)
                    self.git._run(["add", "-A", entry.prefix])

        # Remove from manifest
        self.manifest.remove_membrane_entry(name)

        # Emit event
        emit_membrane_event("removed", {
            "name": name,
            "type": entry.type,
        })

        logger.info(f"Successfully removed '{name}'")
        return True

    def list_entries(self) -> list[MembraneStatus]:
        """
        List all membrane entries with their status.

        Returns:
            List of MembraneStatus objects
        """
        entries = self.manifest.get_membrane_entries()
        statuses: list[MembraneStatus] = []

        for name, entry in entries.items():
            status = EntryStatus.OK
            message = ""
            current_commit = None
            has_changes = False

            try:
                if entry.type == "submodule":
                    submodule_path = self._get_submodule_path(name)
                    path = self.repo_root / submodule_path

                    if not path.exists():
                        status = EntryStatus.MISSING
                        message = "Submodule directory missing"
                    else:
                        current_commit = self.git.get_submodule_commit(submodule_path)
                        # Check for local changes in submodule
                        result = self.git._run(
                            ["status", "--porcelain"],
                            cwd=path,
                            check=False,
                        )
                        if result.stdout.strip():
                            has_changes = True
                            status = EntryStatus.DIRTY
                            message = "Has uncommitted changes"
                else:
                    if entry.prefix:
                        path = self.repo_root / entry.prefix
                        if not path.exists():
                            status = EntryStatus.MISSING
                            message = "Subtree directory missing"
                        elif self.git.subtree_has_changes(entry.prefix):
                            has_changes = True
                            status = EntryStatus.DIRTY
                            message = "Has uncommitted changes"
                    else:
                        status = EntryStatus.ERROR
                        message = "Missing prefix configuration"

                # Check if outdated
                if status == EntryStatus.OK and entry.upstream_latest:
                    if entry.upstream_latest != entry.pinned:
                        status = EntryStatus.OUTDATED
                        message = f"Update available: {entry.upstream_latest}"

            except Exception as e:
                status = EntryStatus.ERROR
                message = str(e)

            statuses.append(MembraneStatus(
                entry=entry,
                status=status,
                message=message,
                current_commit=current_commit,
                has_local_changes=has_changes,
            ))

        return statuses

    def verify_adapters(self) -> list[tuple[str, str | None, bool]]:
        """
        Verify that adapter files exist for each membrane entry.

        Returns:
            List of (name, adapter_path, exists) tuples
        """
        entries = self.manifest.get_membrane_entries()
        results: list[tuple[str, str | None, bool]] = []

        for name, entry in entries.items():
            if entry.adapter:
                adapter_path = self.repo_root / entry.adapter
                exists = adapter_path.exists()
                results.append((name, entry.adapter, exists))
                if not exists:
                    logger.warning(f"Adapter missing for '{name}': {entry.adapter}")
            else:
                results.append((name, None, True))  # No adapter required

        return results


# =============================================================================
# CLI Interface
# =============================================================================

def format_status_table(statuses: list[MembraneStatus]) -> str:
    """Format status list as a table."""
    if not statuses:
        return "No membrane entries found."

    lines = [
        f"{'NAME':<20} {'TYPE':<15} {'VERSION':<15} {'STATUS':<10} {'MESSAGE'}",
        "-" * 85,
    ]

    status_icons = {
        EntryStatus.OK: "[OK]",
        EntryStatus.MISSING: "[MISSING]",
        EntryStatus.DIRTY: "[DIRTY]",
        EntryStatus.OUTDATED: "[UPDATE]",
        EntryStatus.ERROR: "[ERROR]",
    }

    for s in statuses:
        icon = status_icons.get(s.status, "[?]")
        version = s.entry.pinned or "(not pinned)"
        entry_type = s.entry.type or "unknown"
        lines.append(
            f"{s.entry.name:<20} {entry_type:<15} {version:<15} {icon:<10} {s.message}"
        )

    return "\n".join(lines)


def cmd_add_submodule(args: argparse.Namespace) -> int:
    """Handle add-submodule command."""
    try:
        manager = MembraneManager(args.repo)
        entry = manager.add_submodule(
            name=args.name,
            remote=args.remote,
            version=args.version,
            adapter_path=args.adapter,
        )
        print(f"Added submodule '{entry.name}' at version {entry.pinned}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_add_subtree(args: argparse.Namespace) -> int:
    """Handle add-subtree command."""
    try:
        manager = MembraneManager(args.repo)
        entry = manager.add_subtree(
            name=args.name,
            remote=args.remote,
            version=args.version,
            prefix=args.prefix,
            adapter_path=args.adapter,
        )
        print(f"Added subtree '{entry.name}' at {entry.prefix}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_update(args: argparse.Namespace) -> int:
    """Handle update command."""
    try:
        manager = MembraneManager(args.repo)
        entry = manager.update(args.name, args.version)
        print(f"Updated '{entry.name}' to version {entry.pinned}")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_sync_all(args: argparse.Namespace) -> int:
    """Handle sync-all command."""
    try:
        manager = MembraneManager(args.repo)
        results = manager.sync_all()

        if not results:
            print("No membrane entries to sync.")
            return 0

        print(f"{'NAME':<20} {'STATUS':<10} {'MESSAGE'}")
        print("-" * 60)

        failures = 0
        for name, success, message in results:
            status = "OK" if success else "FAILED"
            print(f"{name:<20} {status:<10} {message}")
            if not success:
                failures += 1

        print(f"\nSynced {len(results) - failures}/{len(results)} entries.")
        return 1 if failures > 0 else 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_check_updates(args: argparse.Namespace) -> int:
    """Handle check-updates command."""
    try:
        manager = MembraneManager(args.repo)
        results = manager.check_updates()

        if not results:
            print("No membrane entries found.")
            return 0

        print(f"{'NAME':<20} {'CURRENT':<15} {'LATEST':<15} {'STATUS'}")
        print("-" * 70)

        updates_available = 0
        for name, current, latest in results:
            current_str = current or "(not pinned)"
            latest_str = latest or "N/A"
            if latest is None:
                status = "CHECK FAILED"
            elif latest == current:
                status = "UP TO DATE"
            else:
                status = "UPDATE AVAILABLE"
                updates_available += 1
            print(f"{name:<20} {current_str:<15} {latest_str:<15} {status}")

        if updates_available > 0:
            print(f"\n{updates_available} update(s) available.")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_remove(args: argparse.Namespace) -> int:
    """Handle remove command."""
    try:
        manager = MembraneManager(args.repo)
        manager.remove(args.name, force=args.force)
        print(f"Removed membrane entry '{args.name}'")
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """Handle list command."""
    try:
        manager = MembraneManager(args.repo)
        statuses = manager.list_entries()
        print(format_status_table(statuses))
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def cmd_verify_adapters(args: argparse.Namespace) -> int:
    """Handle verify-adapters command."""
    try:
        manager = MembraneManager(args.repo)
        results = manager.verify_adapters()

        if not results:
            print("No membrane entries found.")
            return 0

        print(f"{'NAME':<20} {'ADAPTER':<40} {'STATUS'}")
        print("-" * 70)

        missing = 0
        for name, adapter, exists in results:
            if adapter:
                status = "OK" if exists else "MISSING"
                if not exists:
                    missing += 1
            else:
                adapter = "(none configured)"
                status = "-"
            print(f"{name:<20} {adapter:<40} {status}")

        if missing > 0:
            print(f"\n{missing} adapter(s) missing!")
            return 1
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="membrane_manager.py",
        description="Manage external SOTA tool integrations via git submodules/subtrees.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=None,
        help="Repository root path (default: auto-detect)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # add-submodule
    p_add_sub = subparsers.add_parser(
        "add-submodule",
        help="Add a git submodule for an external tool",
    )
    p_add_sub.add_argument("name", help="Unique name for this membrane entry")
    p_add_sub.add_argument("remote", help="Git remote URL")
    p_add_sub.add_argument("--version", help="Tag or commit to pin to")
    p_add_sub.add_argument("--adapter", help="Path to adapter file")
    p_add_sub.set_defaults(func=cmd_add_submodule)

    # add-subtree
    p_add_tree = subparsers.add_parser(
        "add-subtree",
        help="Add a git subtree for a forked tool",
    )
    p_add_tree.add_argument("name", help="Unique name for this membrane entry")
    p_add_tree.add_argument("remote", help="Git remote URL")
    p_add_tree.add_argument("--version", help="Tag or branch to pull")
    p_add_tree.add_argument("--prefix", help="Directory prefix for subtree")
    p_add_tree.add_argument("--adapter", help="Path to adapter file")
    p_add_tree.set_defaults(func=cmd_add_subtree)

    # update
    p_update = subparsers.add_parser(
        "update",
        help="Update a membrane entry to a specific version",
    )
    p_update.add_argument("name", help="Name of the membrane entry")
    p_update.add_argument("--version", help="Target version (default: latest tag)")
    p_update.set_defaults(func=cmd_update)

    # sync-all
    p_sync = subparsers.add_parser(
        "sync-all",
        help="Sync all membrane entries to pinned versions",
    )
    p_sync.set_defaults(func=cmd_sync_all)

    # check-updates
    p_check = subparsers.add_parser(
        "check-updates",
        help="Check for available upstream updates",
    )
    p_check.set_defaults(func=cmd_check_updates)

    # remove
    p_remove = subparsers.add_parser(
        "remove",
        help="Remove a membrane entry",
    )
    p_remove.add_argument("name", help="Name of the entry to remove")
    p_remove.add_argument("--force", "-f", action="store_true", help="Force removal")
    p_remove.set_defaults(func=cmd_remove)

    # list
    p_list = subparsers.add_parser(
        "list",
        help="List all membrane entries with status",
    )
    p_list.set_defaults(func=cmd_list)

    # verify-adapters
    p_verify = subparsers.add_parser(
        "verify-adapters",
        help="Verify adapter files exist for each entry",
    )
    p_verify.set_defaults(func=cmd_verify_adapters)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
