#!/usr/bin/env python3
"""
Central CMP Registry for Pluribus RGMA
=======================================

Implements the Clade Meta-Productivity (CMP) fitness tracking system with:
- CladeMetric dataclass for lineage fitness over time
- CMPLarge class with CRUD operations and Thompson sampling
- LineageBandit for compute allocation via Thompson sampling
- Bus event emission for cmp.score, cmp.decay_warning, rgma.lineage.extinct
- Persistence to .pluribus/cmp_registry.ndjson

Golden ratio thresholds:
- EXCELLENT: >= 1.0
- GOOD: >= 0.618 (1/PHI)
- FAIR: >= 0.382 (1/PHI^2)
- CRITICAL: < 0.236 (1/PHI^3) -> extinction

Reference: /pluribus/nucleus/specs/rhizome_godel_alpha.md

Usage:
    # Get a CMP score
    python3 cmp_large.py get <lineage_id>

    # Set/update a CMP score
    python3 cmp_large.py set <lineage_id> --score 0.75 --components '{"task_completion": 0.9}'

    # Sample next lineage for compute allocation
    python3 cmp_large.py sample

    # Decay all lineages (run periodically)
    python3 cmp_large.py decay --rate 0.01

    # List all lineages with status
    python3 cmp_large.py list --status active

    # Show lineage history
    python3 cmp_large.py history <lineage_id>
"""
from __future__ import annotations

import argparse
import getpass
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Literal

sys.dont_write_bytecode = True


# =============================================================================
# Constants: Golden Ratio Thresholds
# =============================================================================

PHI: float = 1.618033988749895
PHI_INV: float = 1.0 / PHI      # ~0.618 (GOOD threshold)
PHI_INV2: float = 1.0 / PHI**2  # ~0.382 (FAIR threshold)
PHI_INV3: float = 1.0 / PHI**3  # ~0.236 (CRITICAL / extinction threshold)

# Fitness thresholds
THRESHOLD_EXCELLENT: float = 1.0
THRESHOLD_GOOD: float = PHI_INV      # ~0.618
THRESHOLD_FAIR: float = PHI_INV2     # ~0.382
THRESHOLD_CRITICAL: float = PHI_INV3  # ~0.236

# Component weights (from RGMA spec)
COMPONENT_WEIGHTS: dict[str, float] = {
    "task_completion": PHI,         # ~1.618 - Primary
    "test_coverage": 1.0,
    "guard_pass_rate": 1.0,
    "motif_recurrence": PHI_INV,    # ~0.618
    "mdl_complexity": PHI_INV2,     # ~0.382 (lower is better)
    "descendant_cmp": PHI_INV3,     # ~0.236 - Recursive lineage health
}

# Default Beta prior parameters for Thompson sampling
DEFAULT_ALPHA: float = 1.0
DEFAULT_BETA: float = 1.0


# =============================================================================
# Utility Functions
# =============================================================================

def now_iso_utc() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    """Return the current actor name from environment or system user."""
    return os.environ.get("PLURIBUS_ACTOR") or os.environ.get("USER") or getpass.getuser()


def find_rhizome_root(start: Path) -> Path | None:
    """Find the rhizome root by searching for .pluribus/rhizome.json."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def ensure_dir(p: Path) -> None:
    """Ensure a directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    """Append a JSON object as a line to an NDJSON file."""
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    """Iterate over lines in an NDJSON file, yielding parsed JSON objects."""
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def emit_bus(
    bus_dir: str | None,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
    trace_id: str | None = None,
) -> None:
    """Emit an event to the Pluribus bus."""
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    cmd = [
        sys.executable,
        str(tool),
        "--bus-dir",
        bus_dir,
        "pub",
        "--topic",
        topic,
        "--kind",
        kind,
        "--level",
        level,
        "--actor",
        actor,
        "--data",
        json.dumps(data, ensure_ascii=False),
    ]
    if trace_id:
        cmd.extend(["--trace-id", trace_id])
    subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def resolve_root(raw_root: str | None) -> Path:
    """Resolve the rhizome root directory."""
    if raw_root:
        return Path(raw_root).expanduser().resolve()
    return find_rhizome_root(Path.cwd()) or Path.cwd().resolve()


def base_dir(root: Path) -> Path:
    """Get the CMP-LARGE index directory."""
    return root / ".pluribus" / "index" / "cmp_large"


def geometric_mean(values: list[float]) -> float:
    """Compute the geometric mean of a list of values."""
    if not values:
        return 0.0
    # Filter out non-positive values
    positive = [v for v in values if v > 0]
    if not positive:
        return 0.0
    product = 1.0
    for v in positive:
        product *= v
    return product ** (1.0 / len(positive))


def classify_fitness(score: float) -> Literal["excellent", "good", "fair", "poor", "critical"]:
    """Classify a fitness score into a category using golden ratio thresholds."""
    if score >= THRESHOLD_EXCELLENT:
        return "excellent"
    elif score >= THRESHOLD_GOOD:
        return "good"
    elif score >= THRESHOLD_FAIR:
        return "fair"
    elif score >= THRESHOLD_CRITICAL:
        return "poor"
    else:
        return "critical"


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FitnessComponents:
    """Individual fitness components for CMP calculation."""
    task_completion: float = 0.0
    test_coverage: float = 0.0
    guard_pass_rate: float = 0.0
    motif_recurrence: float = 0.0
    mdl_complexity: float = 1.0  # Lower is better, default to 1.0
    descendant_cmp: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FitnessComponents":
        """Create from dictionary."""
        return cls(
            task_completion=float(d.get("task_completion", 0.0)),
            test_coverage=float(d.get("test_coverage", 0.0)),
            guard_pass_rate=float(d.get("guard_pass_rate", 0.0)),
            motif_recurrence=float(d.get("motif_recurrence", 0.0)),
            mdl_complexity=float(d.get("mdl_complexity", 1.0)),
            descendant_cmp=float(d.get("descendant_cmp", 0.0)),
        )


@dataclass
class BetaPrior:
    """
    Beta distribution prior for Thompson sampling.

    Alpha represents successes, Beta represents failures.
    The posterior mean is alpha / (alpha + beta).
    """
    alpha: float = DEFAULT_ALPHA
    beta: float = DEFAULT_BETA

    def sample(self) -> float:
        """Sample from the Beta distribution."""
        # Use random.betavariate (standard library)
        return random.betavariate(max(0.01, self.alpha), max(0.01, self.beta))

    def update(self, success: bool, magnitude: float = 1.0) -> None:
        """
        Update the prior with an observation.

        Args:
            success: Whether the outcome was a success
            magnitude: Weight of the update (default 1.0)
        """
        if success:
            self.alpha += magnitude
        else:
            self.beta += magnitude

    def mean(self) -> float:
        """Return the posterior mean."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Return the posterior variance."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BetaPrior":
        """Create from dictionary."""
        return cls(
            alpha=float(d.get("alpha", DEFAULT_ALPHA)),
            beta=float(d.get("beta", DEFAULT_BETA)),
        )


@dataclass
class CladeMetric:
    """
    Tracks lineage fitness over time with CMP scoring.

    Each lineage has:
    - A unique ID
    - Current CMP score
    - Component breakdown
    - Thompson sampling prior
    - History of score updates
    - Status (active, dormant, converging, extinct, merged)
    """
    lineage_id: str
    score: float = 0.0
    classification: str = "poor"
    status: Literal["active", "dormant", "converging", "extinct", "merged"] = "active"
    components: FitnessComponents = field(default_factory=FitnessComponents)
    prior: BetaPrior = field(default_factory=BetaPrior)
    parent_id: str | None = None
    generation: int = 0
    created_iso: str = field(default_factory=now_iso_utc)
    updated_iso: str = field(default_factory=now_iso_utc)
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def compute_score(self) -> float:
        """
        Compute CMP score from components using golden ratio weights.

        The score is a weighted geometric mean of component factors.
        """
        factors = []
        weights_sum = 0.0

        # Task completion (most important)
        if self.components.task_completion > 0:
            weight = COMPONENT_WEIGHTS["task_completion"]
            factors.append((self.components.task_completion, weight))
            weights_sum += weight

        # Test coverage
        if self.components.test_coverage > 0:
            weight = COMPONENT_WEIGHTS["test_coverage"]
            factors.append((self.components.test_coverage, weight))
            weights_sum += weight

        # Guard pass rate
        if self.components.guard_pass_rate > 0:
            weight = COMPONENT_WEIGHTS["guard_pass_rate"]
            factors.append((self.components.guard_pass_rate, weight))
            weights_sum += weight

        # Motif recurrence
        if self.components.motif_recurrence > 0:
            weight = COMPONENT_WEIGHTS["motif_recurrence"]
            factors.append((self.components.motif_recurrence, weight))
            weights_sum += weight

        # MDL complexity (lower is better, so use 1 - normalized)
        if 0 < self.components.mdl_complexity <= 1:
            weight = COMPONENT_WEIGHTS["mdl_complexity"]
            factors.append((1.0 - self.components.mdl_complexity + 0.01, weight))
            weights_sum += weight

        # Descendant CMP (recursive lineage health)
        if self.components.descendant_cmp > 0:
            weight = COMPONENT_WEIGHTS["descendant_cmp"]
            factors.append((self.components.descendant_cmp, weight))
            weights_sum += weight

        if not factors or weights_sum == 0:
            return 0.0

        # Compute weighted geometric mean
        log_sum = 0.0
        for value, weight in factors:
            if value > 0:
                log_sum += weight * math.log(value)

        return math.exp(log_sum / weights_sum)

    def update_score(self) -> None:
        """Update the CMP score from current components."""
        old_score = self.score
        self.score = self.compute_score()
        self.classification = classify_fitness(self.score)
        self.updated_iso = now_iso_utc()

        # Record history entry
        self.history.append({
            "ts": time.time(),
            "iso": self.updated_iso,
            "score": self.score,
            "old_score": old_score,
            "classification": self.classification,
            "components": self.components.to_dict(),
        })

        # Update prior based on score change
        if self.score > old_score:
            self.prior.update(True, min(1.0, self.score - old_score))
        elif self.score < old_score:
            self.prior.update(False, min(1.0, old_score - self.score))

        # Check for status transitions based on thresholds
        if self.status != "extinct" and self.status != "merged":
            if self.score < THRESHOLD_CRITICAL:
                self.status = "extinct"
            elif self.score >= THRESHOLD_GOOD:
                self.status = "converging"
            elif self.status == "converging" and self.score < THRESHOLD_GOOD:
                self.status = "active"

    def apply_decay(self, rate: float = 0.01) -> None:
        """
        Apply temporal decay to the score.

        Args:
            rate: Decay rate per period (default 0.01 = 1%)
        """
        old_score = self.score
        self.score = max(0.0, self.score * (1.0 - rate))
        self.classification = classify_fitness(self.score)
        self.updated_iso = now_iso_utc()

        if old_score != self.score:
            self.history.append({
                "ts": time.time(),
                "iso": self.updated_iso,
                "score": self.score,
                "old_score": old_score,
                "classification": self.classification,
                "event": "decay",
                "rate": rate,
            })

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "lineage_id": self.lineage_id,
            "score": self.score,
            "classification": self.classification,
            "status": self.status,
            "components": self.components.to_dict(),
            "prior": self.prior.to_dict(),
            "parent_id": self.parent_id,
            "generation": self.generation,
            "created_iso": self.created_iso,
            "updated_iso": self.updated_iso,
            "history": self.history[-50:],  # Keep last 50 entries
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CladeMetric":
        """Deserialize from dictionary."""
        return cls(
            lineage_id=d["lineage_id"],
            score=float(d.get("score", 0.0)),
            classification=d.get("classification", "poor"),
            status=d.get("status", "active"),
            components=FitnessComponents.from_dict(d.get("components", {})),
            prior=BetaPrior.from_dict(d.get("prior", {})),
            parent_id=d.get("parent_id"),
            generation=int(d.get("generation", 0)),
            created_iso=d.get("created_iso", now_iso_utc()),
            updated_iso=d.get("updated_iso", now_iso_utc()),
            history=d.get("history", []),
            metadata=d.get("metadata", {}),
        )


# =============================================================================
# LineageBandit: Thompson Sampling for Compute Allocation
# =============================================================================

class LineageBandit:
    """
    Multi-armed bandit over lineages using Thompson sampling.

    Each lineage arm has a Beta prior updated by CMP outcomes.
    This enables principled exploration-exploitation for compute allocation.

    Reference: RGMA spec Section 3.3
    """

    def __init__(self) -> None:
        self.arms: dict[str, BetaPrior] = {}

    def add_arm(self, lineage_id: str, prior: BetaPrior | None = None) -> None:
        """Add a new lineage arm to the bandit."""
        self.arms[lineage_id] = prior or BetaPrior()

    def remove_arm(self, lineage_id: str) -> None:
        """Remove a lineage arm (e.g., on extinction)."""
        self.arms.pop(lineage_id, None)

    def sample_lineage(self) -> str | None:
        """
        Sample from posteriors to select lineage for compute allocation.

        Returns the lineage ID with the highest sampled value.
        """
        if not self.arms:
            return None

        samples = {
            lineage_id: prior.sample()
            for lineage_id, prior in self.arms.items()
        }
        return max(samples, key=lambda k: samples[k])

    def update(self, lineage_id: str, cmp_delta: float) -> None:
        """
        Update posterior based on observed CMP improvement.

        Args:
            lineage_id: The lineage that received compute
            cmp_delta: Change in CMP score (positive = improvement)
        """
        if lineage_id not in self.arms:
            return

        prior = self.arms[lineage_id]
        if cmp_delta > 0:
            prior.update(True, min(1.0, cmp_delta))
        else:
            prior.update(False, min(1.0, abs(cmp_delta)))

    def get_rankings(self) -> list[tuple[str, float, float]]:
        """
        Get lineages ranked by posterior mean.

        Returns list of (lineage_id, mean, variance) tuples.
        """
        rankings = [
            (lid, prior.mean(), prior.variance())
            for lid, prior in self.arms.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Serialize arms to dictionary."""
        return {lid: prior.to_dict() for lid, prior in self.arms.items()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LineageBandit":
        """Deserialize from dictionary."""
        bandit = cls()
        for lid, prior_data in d.items():
            bandit.arms[lid] = BetaPrior.from_dict(prior_data)
        return bandit


# =============================================================================
# CMPLarge: Central CMP Registry
# =============================================================================

class CMPLarge:
    """
    Central CMP Registry for Pluribus RGMA.

    Provides:
    - CRUD operations for CMP scores (get_cmp, set_cmp, update_cmp)
    - Thompson sampler for lineage selection
    - Lineage history tracking
    - Golden ratio threshold classification
    - Persistence to NDJSON
    - Bus event emission

    Thresholds (golden ratio powers):
    - EXCELLENT: >= 1.0
    - GOOD: >= 0.618 (1/PHI) - Ready for merge
    - FAIR: >= 0.382 (1/PHI^2) - Active development
    - POOR: >= 0.236 (1/PHI^3) - Needs attention
    - CRITICAL: < 0.236 - Consider extinction
    """

    def __init__(
        self,
        root: Path | None = None,
        bus_dir: str | None = None,
        actor: str | None = None,
    ) -> None:
        """
        Initialize the CMP Registry.

        Args:
            root: Rhizome root directory (defaults to auto-detect)
            bus_dir: Bus directory for event emission
            actor: Actor name for provenance
        """
        self.root = root or find_rhizome_root(Path.cwd()) or Path("/pluribus")
        self.bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
        self.actor = actor or default_actor()

        # Storage paths
        self.storage_dir = self.root / ".pluribus"
        self.registry_path = self.storage_dir / "cmp_registry.ndjson"
        self.bandit_path = self.storage_dir / "cmp_bandit.json"

        # In-memory state
        self.lineages: dict[str, CladeMetric] = {}
        self.bandit = LineageBandit()

        # Load existing state
        self._load_state()

    def _load_state(self) -> None:
        """Load persisted state from disk."""
        ensure_dir(self.storage_dir)

        # Load lineages from registry
        for record in iter_ndjson(self.registry_path):
            if record.get("type") == "lineage":
                try:
                    metric = CladeMetric.from_dict(record["data"])
                    self.lineages[metric.lineage_id] = metric
                    # Also add to bandit if active
                    if metric.status in ("active", "dormant", "converging"):
                        self.bandit.add_arm(metric.lineage_id, metric.prior)
                except (KeyError, TypeError):
                    continue

        # Load bandit state
        if self.bandit_path.exists():
            try:
                data = json.loads(self.bandit_path.read_text(encoding="utf-8"))
                # Merge with existing arms (prefer registry data)
                for lid, prior_data in data.get("arms", {}).items():
                    if lid not in self.bandit.arms:
                        self.bandit.add_arm(lid, BetaPrior.from_dict(prior_data))
            except (json.JSONDecodeError, KeyError):
                pass

    def _persist_lineage(self, metric: CladeMetric) -> None:
        """Persist a lineage update to the registry."""
        record = {
            "type": "lineage",
            "ts": time.time(),
            "iso": now_iso_utc(),
            "actor": self.actor,
            "data": metric.to_dict(),
        }
        append_ndjson(self.registry_path, record)

    def _persist_bandit(self) -> None:
        """Persist bandit state."""
        data = {
            "ts": time.time(),
            "iso": now_iso_utc(),
            "arms": self.bandit.to_dict(),
        }
        self.bandit_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _emit(self, topic: str, kind: str, level: str, data: dict, trace_id: str | None = None) -> None:
        """Emit a bus event."""
        emit_bus(
            self.bus_dir,
            topic=topic,
            kind=kind,
            level=level,
            actor=self.actor,
            data=data,
            trace_id=trace_id,
        )

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    def get_cmp(self, lineage_id: str) -> CladeMetric | None:
        """
        Get CMP data for a lineage.

        Args:
            lineage_id: The lineage identifier

        Returns:
            CladeMetric if found, None otherwise
        """
        return self.lineages.get(lineage_id)

    def set_cmp(
        self,
        lineage_id: str,
        *,
        score: float | None = None,
        components: FitnessComponents | dict | None = None,
        parent_id: str | None = None,
        generation: int | None = None,
        status: str | None = None,
        metadata: dict | None = None,
    ) -> CladeMetric:
        """
        Create or replace CMP data for a lineage.

        Args:
            lineage_id: The lineage identifier
            score: Optional explicit score (otherwise computed from components)
            components: Fitness component breakdown
            parent_id: Parent lineage ID
            generation: Lineage generation number
            status: Lineage status
            metadata: Additional metadata

        Returns:
            The created/updated CladeMetric
        """
        is_new = lineage_id not in self.lineages

        if isinstance(components, dict):
            components = FitnessComponents.from_dict(components)

        metric = CladeMetric(
            lineage_id=lineage_id,
            components=components or FitnessComponents(),
            parent_id=parent_id,
            generation=generation or 0,
            status=status or "active",
            metadata=metadata or {},
        )

        # Compute score from components, or use explicit score
        if score is not None:
            metric.score = score
            metric.classification = classify_fitness(score)
        else:
            metric.update_score()

        self.lineages[lineage_id] = metric
        self._persist_lineage(metric)

        # Add to bandit if active
        if metric.status in ("active", "dormant", "converging"):
            self.bandit.add_arm(lineage_id, metric.prior)
            self._persist_bandit()

        # Emit events
        self._emit(
            "cmp.score" if not is_new else "rgma.lineage.created",
            kind="metric",
            level="info",
            data={
                "lineage_id": lineage_id,
                "score": metric.score,
                "classification": metric.classification,
                "status": metric.status,
                "is_new": is_new,
            },
        )

        return metric

    def update_cmp(
        self,
        lineage_id: str,
        *,
        components: FitnessComponents | dict | None = None,
        component_deltas: dict[str, float] | None = None,
        status: str | None = None,
        metadata_update: dict | None = None,
    ) -> CladeMetric | None:
        """
        Update CMP data for an existing lineage.

        Args:
            lineage_id: The lineage identifier
            components: New component values (replaces existing)
            component_deltas: Incremental changes to components
            status: New status
            metadata_update: Metadata to merge

        Returns:
            Updated CladeMetric, or None if lineage not found
        """
        metric = self.lineages.get(lineage_id)
        if not metric:
            return None

        old_score = metric.score
        old_status = metric.status

        # Update components
        if components is not None:
            if isinstance(components, dict):
                components = FitnessComponents.from_dict(components)
            metric.components = components
        elif component_deltas:
            for key, delta in component_deltas.items():
                if hasattr(metric.components, key):
                    old_val = getattr(metric.components, key)
                    setattr(metric.components, key, max(0.0, old_val + delta))

        # Update metadata
        if metadata_update:
            metric.metadata.update(metadata_update)

        # Recompute score
        metric.update_score()

        # Check for status override
        if status:
            metric.status = status
        elif old_status != "extinct" and old_status != "merged":
            # Auto-transition based on thresholds
            if metric.score < THRESHOLD_CRITICAL:
                metric.status = "extinct"

        self.lineages[lineage_id] = metric
        self._persist_lineage(metric)

        # Update bandit
        cmp_delta = metric.score - old_score
        self.bandit.update(lineage_id, cmp_delta)

        # Handle extinction
        if metric.status == "extinct" and old_status != "extinct":
            self.bandit.remove_arm(lineage_id)
            self._emit(
                "rgma.lineage.extinct",
                kind="metric",
                level="warn",
                data={
                    "lineage_id": lineage_id,
                    "final_score": metric.score,
                    "reason": "cmp_below_critical",
                    "threshold": THRESHOLD_CRITICAL,
                },
            )

        self._persist_bandit()

        # Emit score event
        self._emit(
            "cmp.score",
            kind="metric",
            level="info" if metric.score >= THRESHOLD_FAIR else "warn",
            data={
                "lineage_id": lineage_id,
                "score": metric.score,
                "old_score": old_score,
                "delta": cmp_delta,
                "classification": metric.classification,
                "status": metric.status,
            },
        )

        # Emit decay warning if approaching critical
        if THRESHOLD_CRITICAL <= metric.score < THRESHOLD_FAIR:
            self._emit(
                "cmp.decay_warning",
                kind="metric",
                level="warn",
                data={
                    "lineage_id": lineage_id,
                    "score": metric.score,
                    "classification": metric.classification,
                    "distance_to_extinction": metric.score - THRESHOLD_CRITICAL,
                },
            )

        return metric

    def delete_cmp(self, lineage_id: str) -> bool:
        """
        Mark a lineage as extinct (soft delete).

        Args:
            lineage_id: The lineage identifier

        Returns:
            True if lineage existed and was marked extinct
        """
        metric = self.lineages.get(lineage_id)
        if not metric:
            return False

        metric.status = "extinct"
        metric.updated_iso = now_iso_utc()
        self._persist_lineage(metric)
        self.bandit.remove_arm(lineage_id)
        self._persist_bandit()

        self._emit(
            "rgma.lineage.extinct",
            kind="metric",
            level="warn",
            data={
                "lineage_id": lineage_id,
                "final_score": metric.score,
                "reason": "manual_deletion",
            },
        )

        return True

    # =========================================================================
    # Thompson Sampling
    # =========================================================================

    def sample_lineage(self) -> str | None:
        """
        Sample a lineage for compute allocation using Thompson sampling.

        Returns:
            The selected lineage ID, or None if no active lineages
        """
        # Filter to active lineages only
        active_arms = {
            lid: prior
            for lid, prior in self.bandit.arms.items()
            if lid in self.lineages and self.lineages[lid].status in ("active", "converging")
        }

        if not active_arms:
            return None

        samples = {
            lid: prior.sample()
            for lid, prior in active_arms.items()
        }

        selected = max(samples, key=lambda k: samples[k])

        self._emit(
            "cmp.lineage_sampled",
            kind="metric",
            level="debug",
            data={
                "lineage_id": selected,
                "sample_value": samples[selected],
                "active_arms": len(active_arms),
            },
        )

        return selected

    def get_rankings(self) -> list[dict[str, Any]]:
        """
        Get lineages ranked by Thompson sampling posterior mean.

        Returns:
            List of dicts with lineage_id, mean, variance, score, classification
        """
        rankings = []
        for lid, mean, variance in self.bandit.get_rankings():
            metric = self.lineages.get(lid)
            if metric:
                rankings.append({
                    "lineage_id": lid,
                    "posterior_mean": mean,
                    "posterior_variance": variance,
                    "cmp_score": metric.score,
                    "classification": metric.classification,
                    "status": metric.status,
                })
        return rankings

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def decay_all(self, rate: float = 0.01) -> list[str]:
        """
        Apply temporal decay to all active lineages.

        Args:
            rate: Decay rate (default 0.01 = 1%)

        Returns:
            List of lineage IDs that became extinct
        """
        extinct = []

        for lineage_id, metric in list(self.lineages.items()):
            if metric.status in ("extinct", "merged"):
                continue

            old_score = metric.score
            old_status = metric.status
            metric.apply_decay(rate)

            # Check for extinction
            if metric.score < THRESHOLD_CRITICAL and old_status != "extinct":
                metric.status = "extinct"
                extinct.append(lineage_id)
                self.bandit.remove_arm(lineage_id)

                self._emit(
                    "rgma.lineage.extinct",
                    kind="metric",
                    level="warn",
                    data={
                        "lineage_id": lineage_id,
                        "final_score": metric.score,
                        "reason": "decay_below_critical",
                        "threshold": THRESHOLD_CRITICAL,
                    },
                )
            elif THRESHOLD_CRITICAL <= metric.score < THRESHOLD_FAIR:
                self._emit(
                    "cmp.decay_warning",
                    kind="metric",
                    level="warn",
                    data={
                        "lineage_id": lineage_id,
                        "score": metric.score,
                        "old_score": old_score,
                        "classification": metric.classification,
                        "decay_rate": rate,
                    },
                )

            self._persist_lineage(metric)

        if extinct:
            self._persist_bandit()

        return extinct

    def list_lineages(
        self,
        status_filter: str | None = None,
        min_score: float | None = None,
        max_score: float | None = None,
    ) -> list[CladeMetric]:
        """
        List lineages with optional filtering.

        Args:
            status_filter: Filter by status
            min_score: Minimum score
            max_score: Maximum score

        Returns:
            List of matching CladeMetric objects
        """
        results = []
        for metric in self.lineages.values():
            if status_filter and metric.status != status_filter:
                continue
            if min_score is not None and metric.score < min_score:
                continue
            if max_score is not None and metric.score > max_score:
                continue
            results.append(metric)

        return sorted(results, key=lambda m: m.score, reverse=True)

    def get_history(self, lineage_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """
        Get score history for a lineage.

        Args:
            lineage_id: The lineage identifier
            limit: Maximum history entries to return

        Returns:
            List of history entries, newest first
        """
        metric = self.lineages.get(lineage_id)
        if not metric:
            return []
        return list(reversed(metric.history[-limit:]))

    def get_statistics(self) -> dict[str, Any]:
        """
        Get overall registry statistics.

        Returns:
            Dict with counts, averages, and distribution info
        """
        total = len(self.lineages)
        active = sum(1 for m in self.lineages.values() if m.status == "active")
        converging = sum(1 for m in self.lineages.values() if m.status == "converging")
        dormant = sum(1 for m in self.lineages.values() if m.status == "dormant")
        extinct = sum(1 for m in self.lineages.values() if m.status == "extinct")
        merged = sum(1 for m in self.lineages.values() if m.status == "merged")

        active_scores = [m.score for m in self.lineages.values() if m.status in ("active", "converging")]
        avg_score = sum(active_scores) / len(active_scores) if active_scores else 0.0

        classifications = {"excellent": 0, "good": 0, "fair": 0, "poor": 0, "critical": 0}
        for m in self.lineages.values():
            if m.status in ("active", "converging", "dormant"):
                classifications[m.classification] += 1

        return {
            "total_lineages": total,
            "by_status": {
                "active": active,
                "converging": converging,
                "dormant": dormant,
                "extinct": extinct,
                "merged": merged,
            },
            "by_classification": classifications,
            "average_active_score": avg_score,
            "thresholds": {
                "excellent": THRESHOLD_EXCELLENT,
                "good": THRESHOLD_GOOD,
                "fair": THRESHOLD_FAIR,
                "critical": THRESHOLD_CRITICAL,
            },
            "phi": PHI,
        }


# =============================================================================
# Legacy API (backward compatibility)
# =============================================================================

def record_episode(
    episode_id: str,
    clade_id: str,
    generation: int,
    cmp_score: float,
    cmp_delta: float,
    fitness_components: dict,
    *,
    root: str | Path | None = None,
    bus_dir: str | None = None,
    actor: str | None = None,
    metadata: dict | None = None,
) -> str:
    """
    Record a training episode to the CMP-LARGE lineage ledger.

    This function is called by AgentLightningTrainer after episode completion
    to persist CMP lineage data for RGMA evolution tracking.

    Args:
        episode_id: Unique identifier for the episode
        clade_id: Lineage/clade identifier
        generation: Generation number within lineage
        cmp_score: Computed CMP score after episode
        cmp_delta: Change in CMP from parent
        fitness_components: Dict of individual fitness component scores
        root: Rhizome root directory
        bus_dir: Bus directory for event emission
        actor: Actor identifier
        metadata: Additional metadata to record

    Returns:
        Record ID
    """
    resolved_root = resolve_root(root)
    d = base_dir(resolved_root)
    ensure_dir(d)

    actor = actor or default_actor()
    bus_dir = bus_dir or os.environ.get("PLURIBUS_BUS_DIR")

    record_id = str(uuid.uuid4())
    record = {
        "id": record_id,
        "episode_id": episode_id,
        "clade_id": clade_id,
        "generation": generation,
        "cmp_score": cmp_score,
        "cmp_delta": cmp_delta,
        "fitness_components": fitness_components,
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "episode",
        "actor": actor,
        "metadata": metadata or {},
    }

    # Append to lineage ledger
    lineage_path = d / "lineage.ndjson"
    append_ndjson(lineage_path, record)

    # Also append to runs for backward compatibility
    runs_path = d / "runs.ndjson"
    append_ndjson(runs_path, {
        "id": record_id,
        "ts": record["ts"],
        "iso": record["iso"],
        "kind": "run",
        "type": "episode",
        "episode_id": episode_id,
        "clade_id": clade_id,
        "cmp_score": cmp_score,
    })

    # Emit bus event for RGMA integration
    emit_bus(
        bus_dir,
        topic="rgma.lineage.cmp_updated",
        kind="metric",
        level="info",
        actor=actor,
        data={
            "record_id": record_id,
            "episode_id": episode_id,
            "clade_id": clade_id,
            "generation": generation,
            "cmp_score": cmp_score,
            "cmp_delta": cmp_delta,
            "fitness_components": fitness_components,
        },
    )

    return record_id


def get_lineage_history(
    clade_id: str,
    *,
    root: str | Path | None = None,
    limit: int = 100,
) -> list[dict]:
    """
    Retrieve CMP history for a lineage.

    Args:
        clade_id: Lineage identifier
        root: Rhizome root directory
        limit: Maximum records to return

    Returns:
        List of episode records for the lineage, newest first
    """
    resolved_root = resolve_root(root)
    d = base_dir(resolved_root)
    lineage_path = d / "lineage.ndjson"

    if not lineage_path.exists():
        return []

    records = []
    for record in iter_ndjson(lineage_path):
        if record.get("clade_id") == clade_id:
            records.append(record)

    # Sort by timestamp descending, limit
    records.sort(key=lambda r: r.get("ts", 0), reverse=True)
    return records[:limit]


def compute_lineage_cmp(
    clade_id: str,
    *,
    root: str | Path | None = None,
    horizon: int = 10,
) -> float:
    """
    Compute aggregate CMP for a lineage using PHI-discounted descendant scores.

    Per RGMA spec section 3.1, CMP aggregates descendant returns with
    temporal discount.

    Args:
        clade_id: Lineage identifier
        root: Rhizome root directory
        horizon: Number of generations to consider

    Returns:
        Aggregate CMP score (0.0-1.0+)
    """
    discount = 1 / PHI  # ~0.618 per generation

    history = get_lineage_history(clade_id, root=root, limit=horizon * 10)
    if not history:
        return 0.5  # neutral

    # Group by generation
    by_gen: dict[int, list[float]] = {}
    for record in history:
        gen = record.get("generation", 0)
        cmp = record.get("cmp_score", 0.5)
        if gen not in by_gen:
            by_gen[gen] = []
        by_gen[gen].append(cmp)

    if not by_gen:
        return 0.5

    # Compute weighted average across generations
    weighted_sum = 0.0
    weight_total = 0.0

    generations = sorted(by_gen.keys())
    base_gen = generations[0] if generations else 0

    for gen in generations[:horizon]:
        gen_offset = gen - base_gen
        weight = discount ** gen_offset
        gen_scores = by_gen[gen]

        # Geometric mean of generation scores
        if gen_scores:
            log_sum = sum(math.log(max(0.001, s)) for s in gen_scores)
            geo_mean = math.exp(log_sum / len(gen_scores))
            weighted_sum += weight * geo_mean
            weight_total += weight

    return weighted_sum / weight_total if weight_total > 0 else 0.5


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_init(args: argparse.Namespace) -> int:
    """Initialize CMP registry."""
    root = resolve_root(args.root)

    # Legacy directory
    d = base_dir(root)
    ensure_dir(d)
    (d / "runs.ndjson").touch(exist_ok=True)
    (d / "datasets.ndjson").touch(exist_ok=True)

    # New registry
    registry = CMPLarge(root=root, bus_dir=args.bus_dir)
    ensure_dir(registry.storage_dir)
    registry.registry_path.touch(exist_ok=True)

    if args.json:
        sys.stdout.write(json.dumps({
            "legacy_dir": str(d),
            "registry_path": str(registry.registry_path),
            "bandit_path": str(registry.bandit_path),
        }, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Initialized CMP registry at {registry.storage_dir}\n")
        sys.stdout.write(f"Registry file: {registry.registry_path}\n")
        sys.stdout.write(f"Legacy dir: {d}\n")

    return 0


def cmd_ingest_priors(args: argparse.Namespace) -> int:
    """Register world priors as a CMP-LARGE dataset."""
    root = resolve_root(args.root)
    actor = default_actor()
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    d = base_dir(root)
    ensure_dir(d)

    priors = root / ".pluribus" / "index" / "world_priors.ndjson"
    if not priors.exists():
        sys.stderr.write("missing world priors; run: world_priors.py build\n")
        return 2

    ds = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "kind": "dataset",
        "type": "world_priors",
        "path": str(priors),
        "notes": "Equilibrium snapshots intended as selector inputs for CMP/CMP-LARGE.",
        "provenance": {"added_by": actor},
    }
    append_ndjson(d / "datasets.ndjson", ds)
    emit_bus(bus_dir, topic="cmp_large.dataset.added", kind="artifact", level="info", actor=actor, data=ds)
    sys.stdout.write(ds["id"] + "\n")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    """Get CMP data for a lineage."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    metric = registry.get_cmp(args.lineage_id)
    if not metric:
        sys.stderr.write(f"Lineage not found: {args.lineage_id}\n")
        return 1

    if args.json:
        sys.stdout.write(json.dumps(metric.to_dict(), ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Lineage: {metric.lineage_id}\n")
        sys.stdout.write(f"Score: {metric.score:.4f} ({metric.classification})\n")
        sys.stdout.write(f"Status: {metric.status}\n")
        sys.stdout.write(f"Generation: {metric.generation}\n")
        if metric.parent_id:
            sys.stdout.write(f"Parent: {metric.parent_id}\n")
        sys.stdout.write(f"Updated: {metric.updated_iso}\n")
        sys.stdout.write(f"Components: {json.dumps(metric.components.to_dict())}\n")

    return 0


def cmd_set(args: argparse.Namespace) -> int:
    """Set/create CMP data for a lineage."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    components = None
    if args.components:
        try:
            components = json.loads(args.components)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Invalid components JSON: {e}\n")
            return 1

    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Invalid metadata JSON: {e}\n")
            return 1

    metric = registry.set_cmp(
        args.lineage_id,
        score=args.score,
        components=components,
        parent_id=args.parent,
        generation=args.generation,
        status=args.status,
        metadata=metadata,
    )

    if args.json:
        sys.stdout.write(json.dumps(metric.to_dict(), ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Set {metric.lineage_id}: score={metric.score:.4f} ({metric.classification})\n")

    return 0


def cmd_update(args: argparse.Namespace) -> int:
    """Update CMP data for a lineage."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    components = None
    if args.components:
        try:
            components = json.loads(args.components)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Invalid components JSON: {e}\n")
            return 1

    deltas = None
    if args.deltas:
        try:
            deltas = json.loads(args.deltas)
        except json.JSONDecodeError as e:
            sys.stderr.write(f"Invalid deltas JSON: {e}\n")
            return 1

    metric = registry.update_cmp(
        args.lineage_id,
        components=components,
        component_deltas=deltas,
        status=args.status,
    )

    if not metric:
        sys.stderr.write(f"Lineage not found: {args.lineage_id}\n")
        return 1

    if args.json:
        sys.stdout.write(json.dumps(metric.to_dict(), ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Updated {metric.lineage_id}: score={metric.score:.4f} ({metric.classification})\n")

    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Sample a lineage for compute allocation."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    lineage_id = registry.sample_lineage()
    if not lineage_id:
        sys.stderr.write("No active lineages available\n")
        return 1

    metric = registry.get_cmp(lineage_id)

    if args.json:
        sys.stdout.write(json.dumps({
            "lineage_id": lineage_id,
            "score": metric.score if metric else None,
            "classification": metric.classification if metric else None,
        }, ensure_ascii=False) + "\n")
    else:
        sys.stdout.write(f"{lineage_id}\n")

    return 0


def cmd_decay(args: argparse.Namespace) -> int:
    """Apply decay to all lineages."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    extinct = registry.decay_all(rate=args.rate)

    if args.json:
        sys.stdout.write(json.dumps({
            "decay_rate": args.rate,
            "extinct_lineages": extinct,
            "extinct_count": len(extinct),
        }, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(f"Applied {args.rate * 100:.1f}% decay\n")
        if extinct:
            sys.stdout.write(f"Extinct lineages: {', '.join(extinct)}\n")

    return 0


def cmd_list(args: argparse.Namespace) -> int:
    """List lineages with optional filtering."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    lineages = registry.list_lineages(
        status_filter=args.status,
        min_score=args.min_score,
        max_score=args.max_score,
    )

    if args.json:
        sys.stdout.write(json.dumps([m.to_dict() for m in lineages], ensure_ascii=False, indent=2) + "\n")
    else:
        if not lineages:
            sys.stdout.write("No lineages found\n")
            return 0

        sys.stdout.write(f"{'ID':<36} {'Score':>8} {'Class':>10} {'Status':>12}\n")
        sys.stdout.write("-" * 70 + "\n")
        for m in lineages:
            sys.stdout.write(f"{m.lineage_id:<36} {m.score:>8.4f} {m.classification:>10} {m.status:>12}\n")

    return 0


def cmd_history(args: argparse.Namespace) -> int:
    """Show score history for a lineage."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    history = registry.get_history(args.lineage_id, limit=args.limit)
    if not history:
        sys.stderr.write(f"No history for lineage: {args.lineage_id}\n")
        return 1

    if args.json:
        sys.stdout.write(json.dumps(history, ensure_ascii=False, indent=2) + "\n")
    else:
        for entry in history:
            ts = entry.get("iso", "unknown")
            score = entry.get("score", 0.0)
            classification = entry.get("classification", "unknown")
            event = entry.get("event", "update")
            sys.stdout.write(f"{ts}: {score:.4f} ({classification}) [{event}]\n")

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Show registry statistics."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    stats = registry.get_statistics()

    if args.json:
        sys.stdout.write(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write("CMP Registry Statistics\n")
        sys.stdout.write("=" * 40 + "\n")
        sys.stdout.write(f"Total Lineages: {stats['total_lineages']}\n")
        sys.stdout.write(f"Average Active Score: {stats['average_active_score']:.4f}\n")
        sys.stdout.write("\nBy Status:\n")
        for status, count in stats["by_status"].items():
            sys.stdout.write(f"  {status}: {count}\n")
        sys.stdout.write("\nBy Classification:\n")
        for cls, count in stats["by_classification"].items():
            sys.stdout.write(f"  {cls}: {count}\n")
        sys.stdout.write("\nThresholds (Golden Ratio):\n")
        for name, value in stats["thresholds"].items():
            sys.stdout.write(f"  {name}: {value:.4f}\n")

    return 0


def cmd_rankings(args: argparse.Namespace) -> int:
    """Show Thompson sampling rankings."""
    registry = CMPLarge(
        root=resolve_root(args.root),
        bus_dir=args.bus_dir,
    )

    rankings = registry.get_rankings()

    if args.json:
        sys.stdout.write(json.dumps(rankings, ensure_ascii=False, indent=2) + "\n")
    else:
        if not rankings:
            sys.stdout.write("No active lineages\n")
            return 0

        sys.stdout.write(f"{'Rank':>4} {'ID':<36} {'Mean':>8} {'CMP':>8} {'Status':>12}\n")
        sys.stdout.write("-" * 72 + "\n")
        for i, r in enumerate(rankings[:args.limit], 1):
            sys.stdout.write(
                f"{i:>4} {r['lineage_id']:<36} "
                f"{r['posterior_mean']:>8.4f} {r['cmp_score']:>8.4f} {r['status']:>12}\n"
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="cmp_large.py",
        description="Central CMP Registry for Pluribus RGMA",
    )
    p.add_argument("--root", default=None, help="Rhizome root directory")
    p.add_argument("--bus-dir", default=None, help="Bus directory (or PLURIBUS_BUS_DIR)")
    p.add_argument("--json", action="store_true", help="Output as JSON")

    sub = p.add_subparsers(dest="cmd", required=True)

    # init
    init_p = sub.add_parser("init", help="Initialize CMP registry")
    init_p.set_defaults(func=cmd_init)

    # ingest-priors (legacy)
    ip = sub.add_parser("ingest-priors", help="Register world priors as a dataset")
    ip.set_defaults(func=cmd_ingest_priors)

    # get
    get_p = sub.add_parser("get", help="Get CMP data for a lineage")
    get_p.add_argument("lineage_id", help="Lineage identifier")
    get_p.set_defaults(func=cmd_get)

    # set
    set_p = sub.add_parser("set", help="Set/create CMP data for a lineage")
    set_p.add_argument("lineage_id", help="Lineage identifier")
    set_p.add_argument("--score", type=float, help="Explicit score (otherwise computed)")
    set_p.add_argument("--components", help="Components JSON")
    set_p.add_argument("--parent", help="Parent lineage ID")
    set_p.add_argument("--generation", type=int, help="Generation number")
    set_p.add_argument("--status", choices=["active", "dormant", "converging", "extinct", "merged"])
    set_p.add_argument("--metadata", help="Metadata JSON")
    set_p.set_defaults(func=cmd_set)

    # update
    update_p = sub.add_parser("update", help="Update CMP data for a lineage")
    update_p.add_argument("lineage_id", help="Lineage identifier")
    update_p.add_argument("--components", help="New components JSON")
    update_p.add_argument("--deltas", help="Component deltas JSON")
    update_p.add_argument("--status", choices=["active", "dormant", "converging", "extinct", "merged"])
    update_p.set_defaults(func=cmd_update)

    # sample
    sample_p = sub.add_parser("sample", help="Sample lineage via Thompson sampling")
    sample_p.set_defaults(func=cmd_sample)

    # decay
    decay_p = sub.add_parser("decay", help="Apply temporal decay to all lineages")
    decay_p.add_argument("--rate", type=float, default=0.01, help="Decay rate (default 0.01 = 1%%)")
    decay_p.set_defaults(func=cmd_decay)

    # list
    list_p = sub.add_parser("list", help="List lineages")
    list_p.add_argument("--status", choices=["active", "dormant", "converging", "extinct", "merged"])
    list_p.add_argument("--min-score", type=float)
    list_p.add_argument("--max-score", type=float)
    list_p.set_defaults(func=cmd_list)

    # history
    hist_p = sub.add_parser("history", help="Show score history for a lineage")
    hist_p.add_argument("lineage_id", help="Lineage identifier")
    hist_p.add_argument("--limit", type=int, default=50, help="Max entries to show")
    hist_p.set_defaults(func=cmd_history)

    # stats
    stats_p = sub.add_parser("stats", help="Show registry statistics")
    stats_p.set_defaults(func=cmd_stats)

    # rankings
    rank_p = sub.add_parser("rankings", help="Show Thompson sampling rankings")
    rank_p.add_argument("--limit", type=int, default=20, help="Max lineages to show")
    rank_p.set_defaults(func=cmd_rankings)

    return p


def main(argv: list[str] | None = None) -> int:
    """Main entry point."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
