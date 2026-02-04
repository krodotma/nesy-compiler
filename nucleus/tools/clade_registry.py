#!/usr/bin/env python3
"""
Clade Registry - Core CMP (Clade Manager Protocol) Implementation
==================================================================

Provides phylogenetic management of development clades with golden-ratio
weighted fitness metrics, lifecycle state management, and bus event emission.

Key Concepts:
- **Clade**: An evolutionary branch (git branch) with tracked metadata
- **Speciation**: Creating a new clade from a parent
- **Fitness**: Golden-ratio weighted health score (0.0 - 1.0+)
- **Membrane**: External SOTA tool integrations (submodules/subtrees)

Fitness Thresholds (Golden Ratio Powers):
- EXCELLENT: >= 1.0    - Ready for immediate merge
- GOOD:      >= 0.618  - Converging, schedule merge (1/phi)
- FAIR:      >= 0.382  - Active development, healthy (1/phi^2)
- POOR:      >= 0.236  - Needs attention (1/phi^3)
- CRITICAL:  < 0.236   - Consider extinction

Usage:
    from clade_registry import CladeRegistry

    registry = CladeRegistry("/path/to/repo")
    registry.load()

    # Create a new clade
    registry.speciate("agent-b", parent="main", pressure="vision-api", mutation_rate=0.5)

    # Evaluate fitness
    fitness = registry.evaluate_fitness("agent-b")

    # Get merge recommendations
    ready = registry.recommend_merges()

    registry.save()
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from functools import reduce
from pathlib import Path
from typing import Any, Literal

# Golden ratio constants
PHI = 1.618033988749895
PHI_INV = 1.0 / PHI          # 0.618... (1/phi)
PHI_INV_2 = 1.0 / (PHI ** 2)  # 0.382... (1/phi^2)
PHI_INV_3 = 1.0 / (PHI ** 3)  # 0.236... (1/phi^3)

# Fitness thresholds
THRESHOLD_EXCELLENT = 1.0
THRESHOLD_GOOD = PHI_INV       # 0.618
THRESHOLD_FAIR = PHI_INV_2     # 0.382
THRESHOLD_POOR = PHI_INV_3     # 0.236

# Default manifest filename
MANIFEST_FILENAME = ".clade-manifest.json"
MANIFEST_SCHEMA_VERSION = 1


class CladeStatus(str, Enum):
    """Lifecycle states for a clade."""
    ACTIVE = "active"
    DORMANT = "dormant"
    CONVERGING = "converging"
    EXTINCT = "extinct"
    MERGED = "merged"


class FitnessLevel(str, Enum):
    """Fitness classification based on golden ratio thresholds."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


@dataclass
class FitnessMetrics:
    """
    Fitness metrics for a clade, used to calculate overall health.

    All values are 0.0-1.0 normalized (except score which can exceed 1.0).

    Attributes:
        score: Computed fitness score (geometric mean of weighted factors)
        last_evaluated: ISO timestamp of last evaluation
        task_completion: Ratio of completed tasks (0.0-1.0)
        test_coverage: Test coverage percentage (0.0-1.0)
        bug_rate: Bug rate (lower is better, 0.0-1.0)
        review_velocity: Speed of code review completion (0.0-1.0)
        divergence_ratio: How far the clade has diverged from parent (0.0-1.0)
    """
    score: float = 0.0
    last_evaluated: str = ""
    task_completion: float = 0.0
    test_coverage: float = 0.0
    bug_rate: float = 0.0
    review_velocity: float = 0.5
    divergence_ratio: float = 0.0

    def classify(self) -> FitnessLevel:
        """Classify fitness score into a level."""
        if self.score >= THRESHOLD_EXCELLENT:
            return FitnessLevel.EXCELLENT
        elif self.score >= THRESHOLD_GOOD:
            return FitnessLevel.GOOD
        elif self.score >= THRESHOLD_FAIR:
            return FitnessLevel.FAIR
        elif self.score >= THRESHOLD_POOR:
            return FitnessLevel.POOR
        else:
            return FitnessLevel.CRITICAL

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "score": round(self.score, 4),
            "last_evaluated": self.last_evaluated,
            "task_completion": round(self.task_completion, 4),
            "test_coverage": round(self.test_coverage, 4),
            "bug_rate": round(self.bug_rate, 4),
            "review_velocity": round(self.review_velocity, 4),
            "divergence_ratio": round(self.divergence_ratio, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FitnessMetrics:
        """Create from dictionary."""
        return cls(
            score=float(data.get("score", 0.0)),
            last_evaluated=str(data.get("last_evaluated", "")),
            task_completion=float(data.get("task_completion", 0.0)),
            test_coverage=float(data.get("test_coverage", 0.0)),
            bug_rate=float(data.get("bug_rate", 0.0)),
            review_velocity=float(data.get("review_velocity", 0.5)),
            divergence_ratio=float(data.get("divergence_ratio", 0.0)),
        )


@dataclass
class CladeMetadata:
    """
    Metadata for a single clade (evolutionary branch).

    Attributes:
        clade_id: Unique identifier (branch name without prefix)
        parent: Parent clade or trunk name
        speciation_commit: Git commit SHA where clade was created
        speciation_date: ISO timestamp of clade creation
        selection_pressure: Description of evolutionary pressure/goal
        mutation_rate: Experimentation level (0.0=conservative, 1.0=experimental)
        generation: Number of generations from trunk
        status: Current lifecycle state
        fitness: Current fitness metrics (optional)
        extra: Additional fields preserved from manifest (e.g., branch, description, commits)
    """
    clade_id: str
    parent: str
    speciation_commit: str
    speciation_date: str
    selection_pressure: str
    mutation_rate: float
    generation: int
    status: Literal["active", "dormant", "converging", "extinct", "merged"] = "active"
    fitness: FitnessMetrics | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "parent": self.parent,
            "speciation_commit": self.speciation_commit,
            "speciation_date": self.speciation_date,
            "selection_pressure": self.selection_pressure,
            "mutation_rate": round(self.mutation_rate, 4),
            "generation": self.generation,
            "status": self.status,
        }
        if self.fitness:
            result["fitness"] = self.fitness.to_dict()
        # Preserve extra fields (branch, description, commits, etc.)
        for key, value in self.extra.items():
            if key not in result:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, clade_id: str, data: dict[str, Any]) -> CladeMetadata:
        """Create from dictionary."""
        fitness_data = data.get("fitness")
        fitness = FitnessMetrics.from_dict(fitness_data) if fitness_data else None

        # Collect known fields
        known_fields = {
            "parent", "speciation_commit", "speciation_date", "selection_pressure",
            "mutation_rate", "generation", "status", "fitness"
        }
        # Preserve unknown fields in extra
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            clade_id=clade_id,
            parent=str(data.get("parent", "main")),
            speciation_commit=str(data.get("speciation_commit", "")),
            speciation_date=str(data.get("speciation_date", "")),
            selection_pressure=str(data.get("selection_pressure", "")),
            mutation_rate=float(data.get("mutation_rate", 0.5)),
            generation=int(data.get("generation", 1)),
            status=data.get("status", "active"),
            fitness=fitness,
            extra=extra,
        )


@dataclass
class MembraneEntry:
    """
    External SOTA tool integration entry.

    Attributes:
        name: Integration name (e.g., "graphiti", "mem0")
        type: Integration type ("submodule", "subtree", "planned_submodule", "planned_subtree")
        remote: Remote repository URL
        pinned: Pinned version/tag/commit (can be None for planned)
        adapter: Path to adapter bridge module (optional)
        prefix: Subtree prefix path (for subtree type only)
        extra: Additional fields preserved from manifest
    """
    name: str
    type: str  # "submodule", "subtree", "planned_submodule", "planned_subtree"
    remote: str
    pinned: str | None
    adapter: str | None = None
    prefix: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result: dict[str, Any] = {
            "type": self.type,
            "remote": self.remote,
            "pinned": self.pinned,
        }
        if self.adapter:
            result["adapter"] = self.adapter
        if self.prefix:
            result["prefix"] = self.prefix
        # Preserve extra fields (upstream_latest, description, integration_status, etc.)
        for key, value in self.extra.items():
            if key not in result:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> MembraneEntry:
        """Create from dictionary."""
        # Collect known fields
        known_fields = {"type", "remote", "pinned", "adapter", "prefix"}
        # Preserve unknown fields in extra
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            name=name,
            type=data.get("type", "submodule"),
            remote=str(data.get("remote", "")),
            pinned=data.get("pinned"),  # Can be None
            adapter=data.get("adapter"),
            prefix=data.get("prefix"),
            extra=extra,
        )


@dataclass
class ExtinctEntry:
    """
    Archive entry for an extinct clade with preserved learnings.

    Attributes:
        clade_id: Original clade identifier
        extinction_date: ISO timestamp of extinction
        learnings: Documented learnings from the clade
        commits_preserved: Number of commits in the extinct branch
        final_fitness: Final fitness score before extinction
    """
    clade_id: str
    extinction_date: str
    learnings: str
    commits_preserved: int = 0
    final_fitness: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clade_id": self.clade_id,
            "extinction_date": self.extinction_date,
            "learnings": self.learnings,
            "commits_preserved": self.commits_preserved,
            "final_fitness": round(self.final_fitness, 4),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExtinctEntry:
        """Create from dictionary."""
        return cls(
            clade_id=str(data.get("clade_id", "")),
            extinction_date=str(data.get("extinction_date", "")),
            learnings=str(data.get("learnings", "")),
            commits_preserved=int(data.get("commits_preserved", 0)),
            final_fitness=float(data.get("final_fitness", 0.0)),
        )


@dataclass
class CladeManifest:
    """
    Root manifest structure containing all CMP state.

    Attributes:
        schema_version: Manifest schema version
        phi: Golden ratio constant (for documentation)
        trunk: Main/trunk branch name
        fitness_thresholds: Dictionary of threshold values
        clades: Dictionary of clade_id -> CladeMetadata
        membrane: Dictionary of name -> MembraneEntry
        extinct_archive: List of ExtinctEntry
        extra: Additional fields preserved from manifest (e.g., meta, phi_inverse)
    """
    schema_version: int = MANIFEST_SCHEMA_VERSION
    phi: float = PHI
    trunk: str = "main"
    fitness_thresholds: dict[str, float] = field(default_factory=lambda: {
        "excellent": THRESHOLD_EXCELLENT,
        "good": THRESHOLD_GOOD,
        "fair": THRESHOLD_FAIR,
        "poor": THRESHOLD_POOR,
    })
    clades: dict[str, CladeMetadata] = field(default_factory=dict)
    membrane: dict[str, MembraneEntry] = field(default_factory=dict)
    extinct_archive: list[ExtinctEntry] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "schema_version": self.schema_version,
            "phi": self.phi,
            "trunk": self.trunk,
            "fitness_thresholds": self.fitness_thresholds,
            "clades": {k: v.to_dict() for k, v in self.clades.items()},
            "membrane": {k: v.to_dict() for k, v in self.membrane.items()},
            "extinct_archive": [e.to_dict() for e in self.extinct_archive],
        }
        # Preserve extra fields (meta, phi_inverse, etc.)
        for key, value in self.extra.items():
            if key not in result:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CladeManifest:
        """Create from dictionary."""
        clades = {}
        for clade_id, clade_data in data.get("clades", {}).items():
            clades[clade_id] = CladeMetadata.from_dict(clade_id, clade_data)

        membrane = {}
        for name, entry_data in data.get("membrane", {}).items():
            membrane[name] = MembraneEntry.from_dict(name, entry_data)

        extinct = [ExtinctEntry.from_dict(e) for e in data.get("extinct_archive", [])]

        # Collect known fields
        known_fields = {
            "schema_version", "phi", "trunk", "fitness_thresholds",
            "clades", "membrane", "extinct_archive"
        }
        # Preserve unknown fields in extra
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            schema_version=int(data.get("schema_version", MANIFEST_SCHEMA_VERSION)),
            phi=float(data.get("phi", PHI)),
            trunk=str(data.get("trunk", "main")),
            fitness_thresholds=data.get("fitness_thresholds", {
                "excellent": THRESHOLD_EXCELLENT,
                "good": THRESHOLD_GOOD,
                "fair": THRESHOLD_FAIR,
                "poor": THRESHOLD_POOR,
            }),
            clades=clades,
            membrane=membrane,
            extra=extra,
            extinct_archive=extinct,
        )


def _now_iso() -> str:
    """Return current UTC time as ISO string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _emit_bus_event(
    topic: str,
    data: dict[str, Any],
    *,
    kind: str = "lifecycle",
    level: str = "info",
) -> None:
    """
    Emit an event to the Pluribus agent bus.

    Best-effort: failures are logged but do not raise exceptions.
    """
    try:
        bus_script = Path(__file__).parent / "agent_bus.py"
        if not bus_script.exists():
            return

        actor = os.environ.get("PLURIBUS_ACTOR", "clade-registry")
        payload = json.dumps(data)

        subprocess.run(
            [
                sys.executable,
                str(bus_script),
                "pub",
                "--topic", topic,
                "--kind", kind,
                "--level", level,
                "--actor", actor,
                "--data", payload,
            ],
            capture_output=True,
            timeout=5,
        )
    except Exception:
        # Best-effort: bus emission should not break core functionality
        pass


def calculate_fitness(metrics: FitnessMetrics) -> float:
    """
    Calculate fitness score using golden-ratio weighted factors.

    Formula uses geometric mean of phi-weighted factors:
    - task_completion: weight PHI (most important)
    - test_coverage: weight 1.0
    - bug_rate: weight 1/PHI (inverse - lower is better)
    - review_velocity: weight 1/PHI^2
    - divergence_ratio: weight 1/PHI^3

    Args:
        metrics: FitnessMetrics with raw values

    Returns:
        Fitness score (0.0-1.0+, can exceed 1.0 for exceptional clades)
    """
    weights = {
        "task_completion": PHI,
        "test_coverage": 1.0,
        "bug_rate": PHI_INV,
        "review_velocity": PHI_INV_2,
        "divergence_ratio": PHI_INV_3,
    }

    # Clamp inputs to valid ranges
    task_completion = max(0.001, min(1.0, metrics.task_completion))
    test_coverage = max(0.001, min(1.0, metrics.test_coverage))
    bug_rate = max(0.0, min(0.999, metrics.bug_rate))
    review_velocity = max(0.001, min(1.0, metrics.review_velocity))
    divergence_ratio = max(0.0, min(0.999, metrics.divergence_ratio))

    factors = [
        pow(task_completion, weights["task_completion"]),
        pow(test_coverage, weights["test_coverage"]),
        pow(1.0 - bug_rate, weights["bug_rate"]),
        pow(review_velocity, weights["review_velocity"]),
        pow(1.0 - divergence_ratio, weights["divergence_ratio"]),
    ]

    # Geometric mean
    product = reduce(lambda x, y: x * y, factors, 1.0)
    return pow(product, 1.0 / len(factors))


class CladeRegistry:
    """
    Core registry for managing evolutionary clades.

    Provides CRUD operations for clades, fitness evaluation,
    merge recommendations, and lifecycle state management.

    All state changes emit bus events to topic `cmp.clade.*`.

    Example:
        registry = CladeRegistry("/path/to/repo")
        registry.load()

        # Create clade
        registry.speciate("feature-x", "main", "performance", 0.3)

        # Evaluate and check
        fitness = registry.evaluate_fitness("feature-x")
        if fitness.classify() == FitnessLevel.GOOD:
            print("Ready for merge!")

        registry.save()
    """

    def __init__(self, repo_path: str | Path):
        """
        Initialize registry for a repository.

        Args:
            repo_path: Path to git repository root
        """
        self.repo_path = Path(repo_path).resolve()
        self.manifest_path = self.repo_path / MANIFEST_FILENAME
        self.manifest = CladeManifest()
        self._loaded = False

    def load(self) -> CladeManifest:
        """
        Load manifest from disk.

        Creates default manifest if file doesn't exist.

        Returns:
            Loaded CladeManifest
        """
        if self.manifest_path.exists():
            try:
                data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                self.manifest = CladeManifest.from_dict(data)
            except (json.JSONDecodeError, KeyError) as e:
                # Corrupted manifest - create fresh one
                self.manifest = CladeManifest()
                _emit_bus_event(
                    "cmp.clade.manifest_corrupt",
                    {"path": str(self.manifest_path), "error": str(e)},
                    kind="log",
                    level="warn",
                )
        else:
            self.manifest = CladeManifest()

        self._loaded = True
        return self.manifest

    def save(self) -> None:
        """
        Save manifest to disk.

        Ensures parent directories exist and writes atomically.
        """
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)

        # Write atomically via temp file
        temp_path = self.manifest_path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(self.manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            temp_path.replace(self.manifest_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _ensure_loaded(self) -> None:
        """Ensure manifest is loaded before operations."""
        if not self._loaded:
            self.load()

    def _get_current_commit(self) -> str:
        """Get current HEAD commit SHA using isomorphic git or fallback."""
        iso_git = Path(__file__).parent / "iso_git.mjs"

        # Try isomorphic git first
        if iso_git.exists():
            try:
                result = subprocess.run(
                    ["node", str(iso_git), "log", str(self.repo_path)],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    commits = data.get("commits", [])
                    if commits:
                        return commits[0].get("sha", "")[:7]
            except Exception:
                pass

        # Fallback to native git
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass

        return ""

    def _get_parent_generation(self, parent: str) -> int:
        """Get generation number of parent clade."""
        if parent == self.manifest.trunk:
            return 0
        if parent in self.manifest.clades:
            return self.manifest.clades[parent].generation
        return 0

    def speciate(
        self,
        clade_id: str,
        parent: str,
        pressure: str,
        mutation_rate: float = 0.5,
    ) -> CladeMetadata:
        """
        Create a new clade (speciation event).

        Args:
            clade_id: Unique identifier for the new clade
            parent: Parent clade or trunk name
            pressure: Selection pressure / goal description
            mutation_rate: Experimentation level (0.0-1.0)

        Returns:
            Created CladeMetadata

        Raises:
            ValueError: If clade_id already exists
        """
        self._ensure_loaded()

        if clade_id in self.manifest.clades:
            raise ValueError(f"Clade '{clade_id}' already exists")

        commit = self._get_current_commit()
        now = _now_iso()
        generation = self._get_parent_generation(parent) + 1

        clade = CladeMetadata(
            clade_id=clade_id,
            parent=parent,
            speciation_commit=commit,
            speciation_date=now,
            selection_pressure=pressure,
            mutation_rate=max(0.0, min(1.0, mutation_rate)),
            generation=generation,
            status="active",
            fitness=None,
        )

        self.manifest.clades[clade_id] = clade

        _emit_bus_event(
            "cmp.clade.speciated",
            {
                "clade_id": clade_id,
                "parent": parent,
                "pressure": pressure,
                "commit": commit,
                "generation": generation,
                "mutation_rate": mutation_rate,
            },
        )

        return clade

    def evaluate_fitness(self, clade_id: str) -> FitnessMetrics:
        """
        Calculate fitness metrics for a clade from git statistics.

        Gathers metrics from:
        - Git commit history (divergence)
        - Task completion (from commit messages)
        - Test coverage (if available)
        - Bug rate (from commit patterns)

        Args:
            clade_id: Clade to evaluate

        Returns:
            Calculated FitnessMetrics

        Raises:
            KeyError: If clade_id not found
        """
        self._ensure_loaded()

        if clade_id not in self.manifest.clades:
            raise KeyError(f"Clade '{clade_id}' not found")

        clade = self.manifest.clades[clade_id]

        # Gather raw metrics from git
        metrics = self._gather_git_metrics(clade_id, clade.parent)

        # Calculate weighted fitness score
        metrics.score = calculate_fitness(metrics)
        metrics.last_evaluated = _now_iso()

        # Update clade
        clade.fitness = metrics

        # Auto-update status based on fitness
        if metrics.score >= THRESHOLD_GOOD and clade.status == "active":
            clade.status = "converging"
        elif metrics.score < THRESHOLD_POOR and clade.status in ("active", "converging"):
            # Don't auto-extinct, just note it needs attention
            pass

        _emit_bus_event(
            "cmp.clade.fitness_evaluated",
            {
                "clade_id": clade_id,
                "fitness": metrics.score,
                "level": metrics.classify().value,
                "recommendation": self._get_recommendation(metrics),
                "metrics": metrics.to_dict(),
            },
            kind="metric",
        )

        return metrics

    def _gather_git_metrics(self, clade_id: str, parent: str) -> FitnessMetrics:
        """
        Gather fitness metrics from git statistics.

        Uses heuristics based on commit history:
        - task_completion: Ratio of "fix"/"feat"/"complete" commits
        - test_coverage: Presence of test files in commits
        - bug_rate: Ratio of "fix"/"bug" commits
        - review_velocity: Commits per day (normalized)
        - divergence_ratio: Commits ahead of parent / total
        """
        metrics = FitnessMetrics()

        try:
            # Get commit log for the clade branch
            result = subprocess.run(
                ["git", "log", "--oneline", f"clade/{clade_id}", "--not", parent, "--", "."],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Try without clade/ prefix
                result = subprocess.run(
                    ["git", "log", "--oneline", clade_id, "--not", parent, "--", "."],
                    cwd=str(self.repo_path),
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

            commits = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
            total_commits = len(commits)

            if total_commits == 0:
                # No commits yet - return defaults with slight activity
                metrics.task_completion = 0.1
                metrics.test_coverage = 0.1
                metrics.bug_rate = 0.0
                metrics.review_velocity = 0.1
                metrics.divergence_ratio = 0.0
                return metrics

            # Analyze commit messages
            feat_commits = sum(1 for c in commits if any(kw in c.lower() for kw in ["feat", "add", "implement", "complete"]))
            fix_commits = sum(1 for c in commits if any(kw in c.lower() for kw in ["fix", "bug", "patch", "hotfix"]))
            test_commits = sum(1 for c in commits if any(kw in c.lower() for kw in ["test", "spec", "coverage"]))

            # Calculate metrics
            metrics.task_completion = min(1.0, (feat_commits + fix_commits) / max(total_commits, 1))
            metrics.test_coverage = min(1.0, (test_commits / max(total_commits, 1)) * 3)  # Boost test ratio
            metrics.bug_rate = min(0.5, fix_commits / max(total_commits, 1))  # Cap at 0.5

            # Review velocity: commits / expected (heuristic: 5 commits is good velocity)
            metrics.review_velocity = min(1.0, total_commits / 5)

            # Divergence: get total parent commits for ratio
            parent_result = subprocess.run(
                ["git", "rev-list", "--count", parent],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            parent_commits = int(parent_result.stdout.strip()) if parent_result.returncode == 0 else 100
            metrics.divergence_ratio = min(0.9, total_commits / max(parent_commits, 1))

        except Exception:
            # Return safe defaults on any error
            metrics.task_completion = 0.5
            metrics.test_coverage = 0.3
            metrics.bug_rate = 0.1
            metrics.review_velocity = 0.5
            metrics.divergence_ratio = 0.1

        return metrics

    def _get_recommendation(self, metrics: FitnessMetrics) -> str:
        """Get action recommendation based on fitness."""
        level = metrics.classify()
        if level == FitnessLevel.EXCELLENT:
            return "immediate_merge"
        elif level == FitnessLevel.GOOD:
            return "converging"
        elif level == FitnessLevel.FAIR:
            return "active_development"
        elif level == FitnessLevel.POOR:
            return "needs_attention"
        else:
            return "consider_extinction"

    def update_status(
        self,
        clade_id: str,
        status: Literal["active", "dormant", "converging", "extinct", "merged"],
    ) -> CladeMetadata:
        """
        Update clade lifecycle status.

        Args:
            clade_id: Clade to update
            status: New status

        Returns:
            Updated CladeMetadata

        Raises:
            KeyError: If clade_id not found
        """
        self._ensure_loaded()

        if clade_id not in self.manifest.clades:
            raise KeyError(f"Clade '{clade_id}' not found")

        clade = self.manifest.clades[clade_id]
        old_status = clade.status
        clade.status = status

        _emit_bus_event(
            "cmp.clade.status_changed",
            {
                "clade_id": clade_id,
                "old_status": old_status,
                "new_status": status,
            },
        )

        return clade

    def recommend_merges(self) -> list[CladeMetadata]:
        """
        Get list of clades ready for merge (fitness >= 0.618).

        Returns:
            List of CladeMetadata with fitness >= THRESHOLD_GOOD,
            sorted by fitness score descending
        """
        self._ensure_loaded()

        ready = []
        for clade in self.manifest.clades.values():
            if clade.status in ("extinct", "merged"):
                continue
            if clade.fitness and clade.fitness.score >= THRESHOLD_GOOD:
                ready.append(clade)

        # Sort by fitness descending
        ready.sort(key=lambda c: c.fitness.score if c.fitness else 0, reverse=True)

        _emit_bus_event(
            "cmp.clade.merge_recommendations",
            {
                "count": len(ready),
                "clades": [
                    {
                        "clade_id": c.clade_id,
                        "fitness": c.fitness.score if c.fitness else 0,
                        "parent": c.parent,
                    }
                    for c in ready
                ],
            },
            kind="metric",
        )

        return ready

    def mark_extinct(
        self,
        clade_id: str,
        learnings: str,
    ) -> ExtinctEntry:
        """
        Mark a clade as extinct and archive learnings.

        Args:
            clade_id: Clade to mark extinct
            learnings: Documented learnings from the clade

        Returns:
            Created ExtinctEntry

        Raises:
            KeyError: If clade_id not found
        """
        self._ensure_loaded()

        if clade_id not in self.manifest.clades:
            raise KeyError(f"Clade '{clade_id}' not found")

        clade = self.manifest.clades[clade_id]

        # Count commits (best effort)
        commits_count = 0
        try:
            result = subprocess.run(
                ["git", "rev-list", "--count", f"clade/{clade_id}", "--not", clade.parent],
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                commits_count = int(result.stdout.strip())
        except Exception:
            pass

        # Create archive entry
        entry = ExtinctEntry(
            clade_id=clade_id,
            extinction_date=_now_iso(),
            learnings=learnings,
            commits_preserved=commits_count,
            final_fitness=clade.fitness.score if clade.fitness else 0.0,
        )

        # Update status and archive
        clade.status = "extinct"
        self.manifest.extinct_archive.append(entry)

        _emit_bus_event(
            "cmp.clade.extinct",
            {
                "clade_id": clade_id,
                "learnings": learnings,
                "commits_preserved": commits_count,
                "final_fitness": entry.final_fitness,
            },
        )

        return entry

    def get_lineage(self, clade_id: str) -> list[str]:
        """
        Get ancestry chain for a clade.

        Args:
            clade_id: Clade to trace

        Returns:
            List of clade IDs from trunk to given clade

        Raises:
            KeyError: If clade_id not found
        """
        self._ensure_loaded()

        if clade_id not in self.manifest.clades:
            raise KeyError(f"Clade '{clade_id}' not found")

        lineage = [clade_id]
        current = self.manifest.clades[clade_id]

        # Walk up the tree
        max_depth = 100  # Prevent infinite loops
        while current.parent != self.manifest.trunk and max_depth > 0:
            if current.parent not in self.manifest.clades:
                break
            lineage.append(current.parent)
            current = self.manifest.clades[current.parent]
            max_depth -= 1

        # Add trunk
        lineage.append(self.manifest.trunk)

        # Reverse to get trunk -> clade order
        lineage.reverse()

        return lineage

    def get_clade(self, clade_id: str) -> CladeMetadata | None:
        """
        Get a clade by ID.

        Args:
            clade_id: Clade identifier

        Returns:
            CladeMetadata or None if not found
        """
        self._ensure_loaded()
        return self.manifest.clades.get(clade_id)

    def list_clades(
        self,
        status: str | None = None,
    ) -> list[CladeMetadata]:
        """
        List all clades, optionally filtered by status.

        Args:
            status: Optional status filter

        Returns:
            List of CladeMetadata
        """
        self._ensure_loaded()

        clades = list(self.manifest.clades.values())
        if status:
            clades = [c for c in clades if c.status == status]

        return clades

    def add_membrane_entry(
        self,
        name: str,
        entry_type: Literal["submodule", "subtree"],
        remote: str,
        pinned: str,
        adapter: str | None = None,
        prefix: str | None = None,
    ) -> MembraneEntry:
        """
        Add or update a membrane (external integration) entry.

        Args:
            name: Integration name
            entry_type: "submodule" or "subtree"
            remote: Remote repository URL
            pinned: Pinned version/tag
            adapter: Optional adapter module path
            prefix: Subtree prefix (for subtree type)

        Returns:
            Created/updated MembraneEntry
        """
        self._ensure_loaded()

        entry = MembraneEntry(
            name=name,
            type=entry_type,
            remote=remote,
            pinned=pinned,
            adapter=adapter,
            prefix=prefix,
        )

        self.manifest.membrane[name] = entry

        _emit_bus_event(
            "cmp.membrane.updated",
            {
                "name": name,
                "type": entry_type,
                "remote": remote,
                "pinned": pinned,
            },
        )

        return entry


# CLI entry point
def main() -> int:
    """CLI entry point for clade_registry."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clade Manager Protocol Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s speciate agent-b --parent main --pressure "vision-api"
  %(prog)s evaluate agent-a
  %(prog)s recommend
  %(prog)s extinct failed-exp --learnings "Approach X doesn't scale"
  %(prog)s lineage agent-a
        """,
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository path (default: current directory)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # speciate
    sp = subparsers.add_parser("speciate", help="Create a new clade")
    sp.add_argument("clade_id", help="New clade identifier")
    sp.add_argument("--parent", default="main", help="Parent clade/trunk")
    sp.add_argument("--pressure", required=True, help="Selection pressure/goal")
    sp.add_argument("--mutation-rate", type=float, default=0.5, help="0.0-1.0")

    # evaluate
    ev = subparsers.add_parser("evaluate", help="Evaluate clade fitness")
    ev.add_argument("clade_id", help="Clade to evaluate")

    # status
    st = subparsers.add_parser("status", help="Update clade status")
    st.add_argument("clade_id", help="Clade to update")
    st.add_argument("new_status", choices=["active", "dormant", "converging", "merged"])

    # recommend
    subparsers.add_parser("recommend", help="List merge-ready clades")

    # extinct
    ex = subparsers.add_parser("extinct", help="Mark clade as extinct")
    ex.add_argument("clade_id", help="Clade to archive")
    ex.add_argument("--learnings", required=True, help="Documented learnings")

    # lineage
    ln = subparsers.add_parser("lineage", help="Show clade ancestry")
    ln.add_argument("clade_id", help="Clade to trace")

    # list
    ls = subparsers.add_parser("list", help="List all clades")
    ls.add_argument("--status", help="Filter by status")

    # show
    sh = subparsers.add_parser("show", help="Show clade details")
    sh.add_argument("clade_id", help="Clade to show")

    args = parser.parse_args()

    registry = CladeRegistry(args.repo)
    registry.load()

    try:
        if args.command == "speciate":
            clade = registry.speciate(
                args.clade_id,
                args.parent,
                args.pressure,
                args.mutation_rate,
            )
            registry.save()
            print(f"Created clade: {clade.clade_id}")
            print(f"  Parent: {clade.parent}")
            print(f"  Generation: {clade.generation}")
            print(f"  Pressure: {clade.selection_pressure}")

        elif args.command == "evaluate":
            metrics = registry.evaluate_fitness(args.clade_id)
            registry.save()
            print(f"Fitness for {args.clade_id}:")
            print(f"  Score: {metrics.score:.4f} ({metrics.classify().value})")
            print(f"  Task Completion: {metrics.task_completion:.2%}")
            print(f"  Test Coverage: {metrics.test_coverage:.2%}")
            print(f"  Bug Rate: {metrics.bug_rate:.2%}")
            print(f"  Review Velocity: {metrics.review_velocity:.2%}")
            print(f"  Divergence: {metrics.divergence_ratio:.2%}")

        elif args.command == "status":
            clade = registry.update_status(args.clade_id, args.new_status)
            registry.save()
            print(f"Updated {args.clade_id} status to: {clade.status}")

        elif args.command == "recommend":
            ready = registry.recommend_merges()
            if not ready:
                print("No clades ready for merge (fitness < 0.618)")
            else:
                print("Merge-ready clades:")
                for c in ready:
                    score = c.fitness.score if c.fitness else 0
                    print(f"  {c.clade_id}: {score:.3f} -> {c.parent}")

        elif args.command == "extinct":
            entry = registry.mark_extinct(args.clade_id, args.learnings)
            registry.save()
            print(f"Archived extinct clade: {args.clade_id}")
            print(f"  Commits preserved: {entry.commits_preserved}")
            print(f"  Final fitness: {entry.final_fitness:.3f}")
            print(f"  Learnings: {entry.learnings}")

        elif args.command == "lineage":
            chain = registry.get_lineage(args.clade_id)
            print(f"Lineage for {args.clade_id}:")
            print("  " + " -> ".join(chain))

        elif args.command == "list":
            clades = registry.list_clades(args.status)
            if not clades:
                print("No clades found")
            else:
                print("Clades:")
                for c in clades:
                    score = f"{c.fitness.score:.3f}" if c.fitness else "n/a"
                    print(f"  {c.clade_id}: {c.status} (fitness: {score})")

        elif args.command == "show":
            clade = registry.get_clade(args.clade_id)
            if not clade:
                print(f"Clade '{args.clade_id}' not found")
                return 1
            print(json.dumps(clade.to_dict(), indent=2))

    except (KeyError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
