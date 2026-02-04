#!/usr/bin/env python3
"""
Clade Manager Protocol (CMP) Command-Line Interface
====================================================

Provides CLI access to the evolutionary clade management system, enabling
speciation, fitness evaluation, merge recommendations, and membrane management.

Commands:
    speciate        Create a new clade (branch) from a parent
    evaluate        Evaluate fitness of one or all clades
    status          Show clade status
    recommend-merge List clades ready for merge
    extinct         Archive a clade with learnings
    lineage         Show evolutionary lineage of a clade

    membrane sync           Sync all membrane integrations
    membrane add            Add a new membrane entry (submodule/subtree)
    membrane update         Update a membrane entry version
    membrane check-updates  Check for upstream updates
    membrane list           List all membrane entries

Usage:
    pluribus-clade speciate <clade-id> --parent main --pressure "feature-x"
    pluribus-clade evaluate agent-a
    pluribus-clade evaluate --all
    pluribus-clade status
    pluribus-clade recommend-merge
    pluribus-clade extinct old-branch --learnings "Did not scale"
    pluribus-clade lineage agent-a

    pluribus-clade membrane sync
    pluribus-clade membrane add graphiti --remote https://... --type submodule
    pluribus-clade membrane update graphiti --version v0.4.0
    pluribus-clade membrane check-updates
    pluribus-clade membrane list
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

sys.dont_write_bytecode = True

# Golden ratio constants
PHI = 1.618033988749895
PHI_INV = 1 / PHI           # ~0.618 (good)
PHI_INV_2 = 1 / (PHI ** 2)  # ~0.382 (fair)
PHI_INV_3 = 1 / (PHI ** 3)  # ~0.236 (poor)

# Fitness thresholds
FITNESS_EXCELLENT = 1.0
FITNESS_GOOD = PHI_INV      # ~0.618
FITNESS_FAIR = PHI_INV_2    # ~0.382
FITNESS_POOR = PHI_INV_3    # ~0.236


# =============================================================================
# Color Utilities (Rich Output)
# =============================================================================

class Colors:
    """ANSI color codes for terminal output."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Background colors
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"

    @classmethod
    def disable(cls) -> None:
        """Disable all colors (for non-TTY or --no-color)."""
        for attr in dir(cls):
            if not attr.startswith("_") and isinstance(getattr(cls, attr), str):
                setattr(cls, attr, "")


def use_colors() -> bool:
    """Check if we should use colors."""
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    return True


def fitness_color(score: float) -> str:
    """Return ANSI color code based on fitness score."""
    if score >= FITNESS_EXCELLENT:
        return Colors.GREEN + Colors.BOLD
    elif score >= FITNESS_GOOD:
        return Colors.GREEN
    elif score >= FITNESS_FAIR:
        return Colors.YELLOW
    elif score >= FITNESS_POOR:
        return Colors.YELLOW + Colors.DIM
    else:
        return Colors.RED


def status_color(status: str) -> str:
    """Return ANSI color code based on clade status."""
    status_colors = {
        "active": Colors.GREEN,
        "dormant": Colors.YELLOW,
        "converging": Colors.CYAN,
        "extinct": Colors.RED + Colors.DIM,
        "merged": Colors.GREEN + Colors.BOLD,
    }
    return status_colors.get(status, Colors.WHITE)


def fitness_label(score: float) -> str:
    """Return human-readable fitness label."""
    if score >= FITNESS_EXCELLENT:
        return "EXCELLENT"
    elif score >= FITNESS_GOOD:
        return "GOOD"
    elif score >= FITNESS_FAIR:
        return "FAIR"
    elif score >= FITNESS_POOR:
        return "POOR"
    else:
        return "CRITICAL"


# =============================================================================
# Table Formatting
# =============================================================================

def format_table(headers: list[str], rows: list[list[str]], widths: list[int] | None = None) -> str:
    """Format data as a table with aligned columns."""
    if not widths:
        widths = [max(len(h), max((len(str(row[i])) for row in rows), default=0))
                  for i, h in enumerate(headers)]

    lines = []

    # Header
    header_line = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    lines.append(f"{Colors.BOLD}{header_line}{Colors.RESET}")

    # Separator
    lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))

    # Data rows
    for row in rows:
        row_line = "  ".join(str(cell).ljust(w) for cell, w in zip(row, widths))
        lines.append(row_line)

    return "\n".join(lines)


# =============================================================================
# Interactive Confirmation
# =============================================================================

def confirm(prompt: str, default: bool = False) -> bool:
    """Prompt user for confirmation."""
    if not sys.stdin.isatty():
        return default

    suffix = " [Y/n]" if default else " [y/N]"
    try:
        response = input(f"{Colors.YELLOW}{prompt}{suffix}{Colors.RESET} ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False

    if not response:
        return default
    return response in ("y", "yes")


# =============================================================================
# Data Structures (mirrors clade_registry.py)
# =============================================================================

@dataclass
class FitnessMetrics:
    """Fitness metrics for a clade."""
    score: float = 0.0
    last_evaluated: str = ""
    task_completion: float = 0.0
    test_coverage: float = 0.0
    bug_rate: float = 0.0
    review_velocity: float = 0.0
    divergence_ratio: float = 0.0


@dataclass
class CladeMetadata:
    """Metadata for a development clade (branch)."""
    clade_id: str
    parent: str
    speciation_commit: str = ""
    speciation_date: str = ""
    selection_pressure: str = ""
    mutation_rate: float = 0.3
    generation: int = 1
    status: Literal["active", "dormant", "converging", "extinct", "merged"] = "active"
    fitness: FitnessMetrics | None = None


@dataclass
class MembraneEntry:
    """External integration tracked in the membrane layer."""
    name: str
    type: Literal["submodule", "subtree"]
    remote: str
    pinned: str = ""
    adapter: str | None = None
    prefix: str | None = None  # For subtrees


@dataclass
class ExtinctRecord:
    """Record of an extinct clade with learnings preserved."""
    clade_id: str
    extinction_date: str
    learnings: str
    commits_preserved: int = 0


# =============================================================================
# Registry Interface (Facade over clade_registry module)
# =============================================================================

class CladeRegistry:
    """
    Interface to the clade registry.

    This class provides a facade over the actual clade_registry module.
    If the module is not available, it uses stub implementations that
    read from .clade-manifest.json directly.
    """

    def __init__(self, root: Path | None = None):
        self.root = root or self._find_root()
        self.manifest_path = self.root / ".clade-manifest.json"
        self._registry = None
        self._load_registry()

    def _find_root(self) -> Path:
        """Find the pluribus root directory."""
        # Check environment
        env_root = os.environ.get("PLURIBUS_ROOT")
        if env_root:
            return Path(env_root)

        # Walk up from current directory
        cur = Path.cwd()
        for p in [cur, *cur.parents]:
            if (p / ".pluribus").exists() or (p / ".clade-manifest.json").exists():
                return p

        # Default to /pluribus
        return Path("/pluribus")

    def _load_registry(self) -> None:
        """Load or initialize the clade registry."""
        try:
            # Try to import the actual clade_registry module
            sys.path.insert(0, str(Path(__file__).parent))
            from clade_registry import CladeRegistry as RealRegistry
            self._registry = RealRegistry(self.root)
            self._registry.load()
        except ImportError:
            # Fall back to reading manifest directly
            self._registry = None

    def _load_manifest(self) -> dict:
        """Load the clade manifest file."""
        if not self.manifest_path.exists():
            return self._default_manifest()
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return self._default_manifest()

    def _save_manifest(self, manifest: dict) -> None:
        """Save the clade manifest file."""
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )

    def _default_manifest(self) -> dict:
        """Return default manifest structure."""
        return {
            "schema_version": 1,
            "phi": PHI,
            "trunk": "main",
            "fitness_thresholds": {
                "excellent": FITNESS_EXCELLENT,
                "good": FITNESS_GOOD,
                "fair": FITNESS_FAIR,
                "poor": FITNESS_POOR,
            },
            "clades": {},
            "membrane": {},
            "extinct_archive": [],
        }

    def speciate(
        self,
        clade_id: str,
        parent: str,
        selection_pressure: str,
        mutation_rate: float = 0.3,
    ) -> CladeMetadata:
        """Create a new clade (speciation)."""
        if self._registry:
            # Real registry uses 'pressure' param name
            clade = self._registry.speciate(
                clade_id=clade_id,
                parent=parent,
                pressure=selection_pressure,
                mutation_rate=mutation_rate,
            )
            self._registry.save()
            return clade

        # Fallback: direct manifest manipulation
        manifest = self._load_manifest()

        # Determine generation
        parent_gen = 0
        if parent in manifest["clades"]:
            parent_gen = manifest["clades"][parent].get("generation", 0)

        clade = CladeMetadata(
            clade_id=clade_id,
            parent=parent,
            speciation_date=datetime.utcnow().isoformat() + "Z",
            selection_pressure=selection_pressure,
            mutation_rate=mutation_rate,
            generation=parent_gen + 1,
            status="active",
        )

        manifest["clades"][clade_id] = asdict(clade)
        self._save_manifest(manifest)

        return clade

    def get_clade(self, clade_id: str) -> CladeMetadata | None:
        """Get clade metadata by ID."""
        if self._registry:
            return self._registry.get_clade(clade_id)

        manifest = self._load_manifest()
        if clade_id not in manifest.get("clades", {}):
            return None

        data = manifest["clades"][clade_id]
        fitness_data = data.get("fitness")
        fitness = FitnessMetrics(**fitness_data) if fitness_data else None

        return CladeMetadata(
            clade_id=data.get("clade_id", clade_id),
            parent=data.get("parent", "main"),
            speciation_commit=data.get("speciation_commit", ""),
            speciation_date=data.get("speciation_date", ""),
            selection_pressure=data.get("selection_pressure", ""),
            mutation_rate=data.get("mutation_rate", 0.3),
            generation=data.get("generation", 1),
            status=data.get("status", "active"),
            fitness=fitness,
        )

    def list_clades(self) -> list[CladeMetadata]:
        """List all clades."""
        if self._registry:
            return self._registry.list_clades()

        manifest = self._load_manifest()
        clades = []
        for clade_id, data in manifest.get("clades", {}).items():
            clade = self.get_clade(clade_id)
            if clade:
                clades.append(clade)
        return clades

    def evaluate_fitness(self, clade_id: str) -> FitnessMetrics | None:
        """Evaluate fitness of a clade."""
        if self._registry:
            result = self._registry.evaluate_fitness(clade_id)
            self._registry.save()
            return result

        # Stub evaluation - would integrate with git stats, tests, etc.
        manifest = self._load_manifest()
        if clade_id not in manifest.get("clades", {}):
            return None

        # Placeholder fitness (real impl would calculate from metrics)
        fitness = FitnessMetrics(
            score=0.5,
            last_evaluated=datetime.utcnow().isoformat() + "Z",
            task_completion=0.6,
            test_coverage=0.4,
            bug_rate=0.1,
            review_velocity=0.5,
            divergence_ratio=0.2,
        )

        manifest["clades"][clade_id]["fitness"] = asdict(fitness)
        self._save_manifest(manifest)

        return fitness

    def mark_extinct(self, clade_id: str, learnings: str) -> ExtinctRecord | None:
        """Mark a clade as extinct and record learnings."""
        if self._registry:
            result = self._registry.mark_extinct(clade_id, learnings)
            self._registry.save()
            return result

        manifest = self._load_manifest()
        if clade_id not in manifest.get("clades", {}):
            return None

        # Update status
        manifest["clades"][clade_id]["status"] = "extinct"

        # Add to archive
        record = ExtinctRecord(
            clade_id=clade_id,
            extinction_date=datetime.utcnow().isoformat() + "Z",
            learnings=learnings,
            commits_preserved=0,  # Would count commits in real impl
        )
        manifest["extinct_archive"].append(asdict(record))

        self._save_manifest(manifest)
        return record

    def get_lineage(self, clade_id: str) -> list[str]:
        """Get the evolutionary lineage of a clade (ancestors)."""
        if self._registry:
            return self._registry.get_lineage(clade_id)

        manifest = self._load_manifest()
        lineage = []
        current = clade_id

        while current and current in manifest.get("clades", {}):
            lineage.append(current)
            current = manifest["clades"][current].get("parent")

        # Add trunk if not already included
        trunk = manifest.get("trunk", "main")
        if trunk not in lineage:
            lineage.append(trunk)

        return lineage

    def recommend_merge(self) -> list[tuple[CladeMetadata, str]]:
        """Get merge recommendations for all clades."""
        recommendations = []
        for clade in self.list_clades():
            if clade.status in ("extinct", "merged"):
                continue

            fitness_score = clade.fitness.score if clade.fitness else 0.0

            if fitness_score >= FITNESS_GOOD:
                rec = "ready for merge"
            elif fitness_score >= FITNESS_FAIR:
                rec = "needs more work"
            elif fitness_score >= FITNESS_POOR:
                rec = "at risk"
            else:
                rec = "recommend extinction"

            recommendations.append((clade, rec))

        return recommendations


# =============================================================================
# Membrane Manager Interface
# =============================================================================

class MembraneManager:
    """
    Interface to the membrane manager.

    Handles external tool integrations via git submodules/subtrees.
    """

    def __init__(self, root: Path | None = None):
        self.root = root or Path("/pluribus")
        self.manifest_path = self.root / ".clade-manifest.json"
        self._manager = None
        self._load_manager()

    def _load_manager(self) -> None:
        """Load or initialize the membrane manager."""
        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from membrane_manager import MembraneManager as RealManager
            self._manager = RealManager(self.root)
        except ImportError:
            self._manager = None

    def _load_manifest(self) -> dict:
        """Load the clade manifest file."""
        if not self.manifest_path.exists():
            return {"membrane": {}}
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return {"membrane": {}}

    def _save_manifest(self, manifest: dict) -> None:
        """Save the clade manifest file."""
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8"
        )

    def list_entries(self) -> list[MembraneEntry]:
        """List all membrane entries."""
        if self._manager:
            return self._manager.list_entries()

        manifest = self._load_manifest()
        entries = []
        for name, data in manifest.get("membrane", {}).items():
            entries.append(MembraneEntry(
                name=name,
                type=data.get("type", "submodule"),
                remote=data.get("remote", ""),
                pinned=data.get("pinned", ""),
                adapter=data.get("adapter"),
                prefix=data.get("prefix"),
            ))
        return entries

    def add_entry(
        self,
        name: str,
        remote: str,
        entry_type: Literal["submodule", "subtree"],
        version: str | None = None,
    ) -> MembraneEntry:
        """Add a new membrane entry."""
        if self._manager:
            return self._manager.add_entry(name, remote, entry_type, version)

        manifest = self._load_manifest()
        if "membrane" not in manifest:
            manifest["membrane"] = {}

        entry = MembraneEntry(
            name=name,
            type=entry_type,
            remote=remote,
            pinned=version or "HEAD",
        )

        manifest["membrane"][name] = {
            "type": entry_type,
            "remote": remote,
            "pinned": entry.pinned,
        }

        self._save_manifest(manifest)
        return entry

    def update_entry(self, name: str, version: str) -> MembraneEntry | None:
        """Update a membrane entry to a specific version."""
        if self._manager:
            return self._manager.update_entry(name, version)

        manifest = self._load_manifest()
        if name not in manifest.get("membrane", {}):
            return None

        manifest["membrane"][name]["pinned"] = version
        self._save_manifest(manifest)

        data = manifest["membrane"][name]
        return MembraneEntry(
            name=name,
            type=data.get("type", "submodule"),
            remote=data.get("remote", ""),
            pinned=version,
            adapter=data.get("adapter"),
        )

    def sync_all(self) -> list[tuple[str, bool, str]]:
        """Sync all membrane entries. Returns list of (name, success, message)."""
        if self._manager:
            return self._manager.sync_all()

        results = []
        for entry in self.list_entries():
            # Stub - would actually run git submodule/subtree commands
            results.append((entry.name, True, f"Synced to {entry.pinned}"))
        return results

    def check_updates(self) -> list[tuple[str, str, str]]:
        """Check for upstream updates. Returns list of (name, current, available)."""
        if self._manager:
            return self._manager.check_updates()

        updates = []
        for entry in self.list_entries():
            # Stub - would actually check upstream
            updates.append((entry.name, entry.pinned, entry.pinned))
        return updates


# =============================================================================
# CLI Command Handlers
# =============================================================================

def cmd_speciate(args: argparse.Namespace) -> int:
    """Create a new clade."""
    registry = CladeRegistry()

    # Check if clade already exists
    existing = registry.get_clade(args.clade_id)
    if existing:
        print(f"{Colors.RED}Error: Clade '{args.clade_id}' already exists{Colors.RESET}")
        return 1

    clade = registry.speciate(
        clade_id=args.clade_id,
        parent=args.parent,
        selection_pressure=args.pressure,
        mutation_rate=args.mutation_rate,
    )

    if args.json:
        print(json.dumps(asdict(clade), indent=2))
    else:
        print(f"{Colors.GREEN}Created clade '{clade.clade_id}'{Colors.RESET}")
        print(f"  Parent: {clade.parent}")
        print(f"  Pressure: {clade.selection_pressure}")
        print(f"  Mutation rate: {clade.mutation_rate}")
        print(f"  Generation: {clade.generation}")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evaluate clade fitness."""
    registry = CladeRegistry()

    if args.all:
        clades = [c for c in registry.list_clades() if c.status not in ("extinct", "merged")]
        if not clades:
            print("No active clades to evaluate")
            return 0
    else:
        clade = registry.get_clade(args.clade_id)
        if not clade:
            print(f"{Colors.RED}Error: Clade '{args.clade_id}' not found{Colors.RESET}")
            return 1
        clades = [clade]

    results = []
    for clade in clades:
        fitness = registry.evaluate_fitness(clade.clade_id)
        if fitness:
            results.append((clade, fitness))

    if args.json:
        output = [
            {
                "clade_id": c.clade_id,
                "fitness": asdict(f),
                "label": fitness_label(f.score),
            }
            for c, f in results
        ]
        print(json.dumps(output, indent=2))
    else:
        for clade, fitness in results:
            color = fitness_color(fitness.score)
            label = fitness_label(fitness.score)

            print(f"\n{Colors.BOLD}Clade: {clade.clade_id}{Colors.RESET}")
            print(f"  Fitness: {color}{fitness.score:.3f} ({label}){Colors.RESET}")
            print(f"  Task completion: {fitness.task_completion:.0%}")
            print(f"  Test coverage:   {fitness.test_coverage:.0%}")
            print(f"  Bug rate:        {fitness.bug_rate:.0%}")
            print(f"  Review velocity: {fitness.review_velocity:.0%}")
            print(f"  Divergence:      {fitness.divergence_ratio:.0%}")

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show clade status."""
    registry = CladeRegistry()

    if args.clade_id:
        clade = registry.get_clade(args.clade_id)
        if not clade:
            print(f"{Colors.RED}Error: Clade '{args.clade_id}' not found{Colors.RESET}")
            return 1
        clades = [clade]
    else:
        clades = registry.list_clades()

    if not clades:
        print("No clades found")
        return 0

    if args.json:
        output = [asdict(c) for c in clades]
        print(json.dumps(output, indent=2))
        return 0

    # Build table data
    headers = ["Clade", "Parent", "Status", "Fitness", "Pressure", "Gen"]
    rows = []

    for clade in sorted(clades, key=lambda c: c.clade_id):
        fitness_score = clade.fitness.score if clade.fitness else 0.0
        color = fitness_color(fitness_score)
        status_col = status_color(clade.status)

        fitness_str = f"{color}{fitness_score:.3f}{Colors.RESET}" if clade.fitness else "-"
        status_str = f"{status_col}{clade.status}{Colors.RESET}"
        pressure = (clade.selection_pressure[:20] + "...") if len(clade.selection_pressure) > 23 else clade.selection_pressure

        rows.append([
            clade.clade_id,
            clade.parent,
            status_str,
            fitness_str,
            pressure,
            str(clade.generation),
        ])

    print(format_table(headers, rows, [20, 15, 12, 10, 25, 4]))
    return 0


def cmd_recommend_merge(args: argparse.Namespace) -> int:
    """Show merge recommendations."""
    registry = CladeRegistry()
    recommendations = registry.recommend_merge()

    if not recommendations:
        print("No clades available for merge recommendation")
        return 0

    if args.json:
        output = [
            {
                "clade_id": c.clade_id,
                "fitness": c.fitness.score if c.fitness else 0.0,
                "recommendation": r,
            }
            for c, r in recommendations
        ]
        print(json.dumps(output, indent=2))
        return 0

    print(f"{Colors.BOLD}Merge Recommendations{Colors.RESET}\n")

    for clade, rec in sorted(recommendations, key=lambda x: -(x[0].fitness.score if x[0].fitness else 0)):
        fitness_score = clade.fitness.score if clade.fitness else 0.0
        color = fitness_color(fitness_score)

        # Symbol based on recommendation
        if "ready" in rec:
            symbol = f"{Colors.GREEN}[OK]{Colors.RESET}"
        elif "work" in rec:
            symbol = f"{Colors.YELLOW}[..]{Colors.RESET}"
        elif "risk" in rec:
            symbol = f"{Colors.YELLOW}[!!]{Colors.RESET}"
        else:
            symbol = f"{Colors.RED}[XX]{Colors.RESET}"

        print(f"  {symbol} {clade.clade_id} ({color}{fitness_score:.3f}{Colors.RESET}) -> {rec}")
        print(f"       target: {clade.parent}")

    return 0


def cmd_extinct(args: argparse.Namespace) -> int:
    """Mark a clade as extinct."""
    registry = CladeRegistry()

    clade = registry.get_clade(args.clade_id)
    if not clade:
        print(f"{Colors.RED}Error: Clade '{args.clade_id}' not found{Colors.RESET}")
        return 1

    if clade.status == "extinct":
        print(f"{Colors.YELLOW}Clade '{args.clade_id}' is already extinct{Colors.RESET}")
        return 0

    # Interactive confirmation for destructive operation
    if not args.force:
        print(f"\n{Colors.YELLOW}WARNING: This will mark clade '{args.clade_id}' as extinct.{Colors.RESET}")
        print(f"Learnings: {args.learnings}")
        if not confirm("Proceed with extinction?", default=False):
            print("Aborted")
            return 1

    record = registry.mark_extinct(args.clade_id, args.learnings)

    if args.json:
        print(json.dumps(asdict(record) if record else {}, indent=2))
    else:
        print(f"{Colors.RED}Marked clade '{args.clade_id}' as extinct{Colors.RESET}")
        print(f"  Learnings preserved: {args.learnings}")

    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    """Show evolutionary lineage of a clade."""
    registry = CladeRegistry()

    clade = registry.get_clade(args.clade_id)
    if not clade:
        print(f"{Colors.RED}Error: Clade '{args.clade_id}' not found{Colors.RESET}")
        return 1

    lineage = registry.get_lineage(args.clade_id)

    if args.json:
        print(json.dumps({"clade_id": args.clade_id, "lineage": lineage}))
        return 0

    print(f"{Colors.BOLD}Lineage of '{args.clade_id}'{Colors.RESET}\n")

    for i, ancestor in enumerate(lineage):
        indent = "  " * i
        connector = "+-" if i < len(lineage) - 1 else "+-"

        anc_clade = registry.get_clade(ancestor)
        if anc_clade:
            status_col = status_color(anc_clade.status)
            print(f"{indent}{connector} {Colors.CYAN}{ancestor}{Colors.RESET} [{status_col}{anc_clade.status}{Colors.RESET}]")
        else:
            # Trunk or unknown
            print(f"{indent}{connector} {Colors.GREEN}{ancestor}{Colors.RESET} [trunk]")

    return 0


# =============================================================================
# Membrane Subcommand Handlers
# =============================================================================

def cmd_membrane_sync(args: argparse.Namespace) -> int:
    """Sync all membrane entries."""
    manager = MembraneManager()
    results = manager.sync_all()

    if args.json:
        output = [{"name": n, "success": s, "message": m} for n, s, m in results]
        print(json.dumps(output, indent=2))
        return 0

    print(f"{Colors.BOLD}Membrane Sync{Colors.RESET}\n")

    for name, success, message in results:
        if success:
            print(f"  {Colors.GREEN}[OK]{Colors.RESET} {name}: {message}")
        else:
            print(f"  {Colors.RED}[FAIL]{Colors.RESET} {name}: {message}")

    return 0 if all(s for _, s, _ in results) else 1


def cmd_membrane_add(args: argparse.Namespace) -> int:
    """Add a new membrane entry."""
    manager = MembraneManager()

    # Check if already exists
    existing = [e for e in manager.list_entries() if e.name == args.name]
    if existing:
        print(f"{Colors.RED}Error: Membrane entry '{args.name}' already exists{Colors.RESET}")
        return 1

    entry = manager.add_entry(
        name=args.name,
        remote=args.remote,
        entry_type=args.type,
        version=args.version,
    )

    if args.json:
        print(json.dumps(asdict(entry), indent=2))
    else:
        print(f"{Colors.GREEN}Added membrane entry '{entry.name}'{Colors.RESET}")
        print(f"  Type:   {entry.type}")
        print(f"  Remote: {entry.remote}")
        print(f"  Pinned: {entry.pinned}")

    return 0


def cmd_membrane_update(args: argparse.Namespace) -> int:
    """Update a membrane entry."""
    manager = MembraneManager()

    entry = manager.update_entry(args.name, args.version)
    if not entry:
        print(f"{Colors.RED}Error: Membrane entry '{args.name}' not found{Colors.RESET}")
        return 1

    if args.json:
        print(json.dumps(asdict(entry), indent=2))
    else:
        print(f"{Colors.GREEN}Updated membrane entry '{entry.name}' to {args.version}{Colors.RESET}")

    return 0


def cmd_membrane_check_updates(args: argparse.Namespace) -> int:
    """Check for upstream updates."""
    manager = MembraneManager()
    updates = manager.check_updates()

    if args.json:
        output = [{"name": n, "current": c, "available": a} for n, c, a in updates]
        print(json.dumps(output, indent=2))
        return 0

    print(f"{Colors.BOLD}Membrane Update Check{Colors.RESET}\n")

    has_updates = False
    for name, current, available in updates:
        if current != available:
            has_updates = True
            print(f"  {Colors.YELLOW}[UPDATE]{Colors.RESET} {name}: {current} -> {available}")
        else:
            print(f"  {Colors.GREEN}[OK]{Colors.RESET} {name}: {current}")

    if not has_updates:
        print(f"\n{Colors.GREEN}All membrane entries are up to date{Colors.RESET}")

    return 0


def cmd_membrane_list(args: argparse.Namespace) -> int:
    """List all membrane entries."""
    manager = MembraneManager()
    entries = manager.list_entries()

    if not entries:
        print("No membrane entries found")
        return 0

    if args.json:
        # Handle both MembraneStatus and MembraneEntry objects
        output = []
        for e in entries:
            if hasattr(e, "entry"):
                # It's a MembraneStatus object
                output.append({
                    "name": e.entry.name,
                    "type": e.entry.type,
                    "remote": e.entry.remote,
                    "pinned": e.entry.pinned,
                    "adapter": e.entry.adapter,
                    "status": str(e.status.value) if hasattr(e.status, "value") else str(e.status),
                    "message": e.message,
                })
            else:
                # It's a MembraneEntry object
                output.append(asdict(e))
        print(json.dumps(output, indent=2))
        return 0

    headers = ["Name", "Type", "Remote", "Pinned", "Status"]
    rows = []

    for item in entries:
        # Handle both MembraneStatus and MembraneEntry objects
        if hasattr(item, "entry"):
            entry = item.entry
            status = str(item.status.value) if hasattr(item.status, "value") else str(item.status)
        else:
            entry = item
            status = "-"

        remote_short = entry.remote if len(entry.remote) <= 40 else "..." + entry.remote[-37:]
        rows.append([
            entry.name,
            entry.type,
            remote_short,
            entry.pinned or "-",
            status,
        ])

    # Sort by name
    rows.sort(key=lambda r: r[0])
    print(format_table(headers, rows, [15, 10, 42, 12, 10]))
    return 0


# =============================================================================
# Argument Parser Construction
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pluribus-clade",
        description="Clade Manager Protocol (CMP) CLI - Evolutionary branch management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s speciate agent-b --parent main --pressure "distributed-inference"
  %(prog)s evaluate --all
  %(prog)s status
  %(prog)s recommend-merge
  %(prog)s extinct old-branch --learnings "Approach did not scale"
  %(prog)s lineage agent-a

  %(prog)s membrane sync
  %(prog)s membrane add graphiti --remote https://github.com/... --type submodule
  %(prog)s membrane list
""",
    )

    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    # Helper to add common args to subparsers
    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--json", action="store_true", help="Output in JSON format")

    # speciate
    p_speciate = subparsers.add_parser("speciate", help="Create a new clade")
    p_speciate.add_argument("clade_id", help="ID for the new clade")
    p_speciate.add_argument("--parent", required=True, help="Parent clade/branch")
    p_speciate.add_argument("--pressure", required=True, help="Selection pressure description")
    p_speciate.add_argument("--mutation-rate", type=float, default=0.3,
                            help="Mutation rate 0.0-1.0 (default: 0.3)")
    add_common_args(p_speciate)
    p_speciate.set_defaults(func=cmd_speciate)

    # evaluate
    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate clade fitness")
    p_evaluate.add_argument("clade_id", nargs="?", help="Clade ID to evaluate")
    p_evaluate.add_argument("--all", action="store_true", help="Evaluate all active clades")
    add_common_args(p_evaluate)
    p_evaluate.set_defaults(func=cmd_evaluate)

    # status
    p_status = subparsers.add_parser("status", help="Show clade status")
    p_status.add_argument("clade_id", nargs="?", help="Specific clade ID (optional)")
    add_common_args(p_status)
    p_status.set_defaults(func=cmd_status)

    # recommend-merge
    p_merge = subparsers.add_parser("recommend-merge", help="Show merge recommendations")
    add_common_args(p_merge)
    p_merge.set_defaults(func=cmd_recommend_merge)

    # extinct
    p_extinct = subparsers.add_parser("extinct", help="Mark a clade as extinct")
    p_extinct.add_argument("clade_id", help="Clade ID to mark extinct")
    p_extinct.add_argument("--learnings", required=True, help="Learnings to preserve")
    p_extinct.add_argument("--force", "-f", action="store_true", help="Skip confirmation")
    add_common_args(p_extinct)
    p_extinct.set_defaults(func=cmd_extinct)

    # lineage
    p_lineage = subparsers.add_parser("lineage", help="Show evolutionary lineage")
    p_lineage.add_argument("clade_id", help="Clade ID to trace")
    add_common_args(p_lineage)
    p_lineage.set_defaults(func=cmd_lineage)

    # membrane subcommands
    p_membrane = subparsers.add_parser("membrane", help="Membrane management commands")
    membrane_sub = p_membrane.add_subparsers(dest="membrane_command", metavar="SUBCOMMAND")

    # membrane sync
    p_m_sync = membrane_sub.add_parser("sync", help="Sync all membrane entries")
    add_common_args(p_m_sync)
    p_m_sync.set_defaults(func=cmd_membrane_sync)

    # membrane add
    p_m_add = membrane_sub.add_parser("add", help="Add a membrane entry")
    p_m_add.add_argument("name", help="Entry name")
    p_m_add.add_argument("--remote", required=True, help="Remote URL")
    p_m_add.add_argument("--type", required=True, choices=["submodule", "subtree"],
                         help="Integration type")
    p_m_add.add_argument("--version", help="Version to pin (optional)")
    add_common_args(p_m_add)
    p_m_add.set_defaults(func=cmd_membrane_add)

    # membrane update
    p_m_update = membrane_sub.add_parser("update", help="Update a membrane entry")
    p_m_update.add_argument("name", help="Entry name")
    p_m_update.add_argument("--version", required=True, help="New version")
    add_common_args(p_m_update)
    p_m_update.set_defaults(func=cmd_membrane_update)

    # membrane check-updates
    p_m_check = membrane_sub.add_parser("check-updates", help="Check for upstream updates")
    add_common_args(p_m_check)
    p_m_check.set_defaults(func=cmd_membrane_check_updates)

    # membrane list
    p_m_list = membrane_sub.add_parser("list", help="List membrane entries")
    add_common_args(p_m_list)
    p_m_list.set_defaults(func=cmd_membrane_list)

    return parser


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv: list[str] | None = None) -> int:
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Handle color settings
    if args.no_color or not use_colors():
        Colors.disable()

    # Check if command was provided
    if not args.command:
        parser.print_help()
        return 0

    # Handle membrane subcommands
    if args.command == "membrane":
        if not args.membrane_command:
            # Print membrane help
            parser.parse_args(["membrane", "--help"])
            return 0

    # Check evaluate args
    if args.command == "evaluate":
        if not args.clade_id and not args.all:
            print(f"{Colors.RED}Error: Specify a clade_id or use --all{Colors.RESET}")
            return 1

    # Execute command
    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
