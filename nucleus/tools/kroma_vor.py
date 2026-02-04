#!/usr/bin/env python3
"""
kroma_vor.py — VOR/MCTS Self-Verification and Optimization Routine

Implements the VOR (VHF Omnidirectional Range) navigation metaphor for
autonomous Pluribus self-checking and self-improvement.

VOR Phase               MCTS Phase              Implementation
─────────────────────────────────────────────────────────────────────
Tune & identify beacon  Selection (UCT)         select_check_domain()
Dial radial             Expansion               expand_checks()
Read CDI needle         Simulation              run_checks()
Wind correction         Backpropagation         apply_fixes()
Cross-check & correct   Next iteration          loop()

References:
- kroma.md: VOR/MCTS Metaphor (lines 67-85)
- entelexis.md: CMP-LARGE / Huxley-Gödel Machine
- Gauss-Schläfli-Hertz lineage for geometric interpretation

Axiom: Each VOR cycle refines structural compliance with formal Pluribus spec.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

sys.dont_write_bytecode = True


# --- Formal Pluribus Specification ---

PLURIBUS_SPEC = {
    "version": "1.0",
    "domains": {
        "structural": {
            "description": "Required files and directories",
            "checks": [
                {"id": "rhizome_json", "path": ".pluribus/rhizome.json", "type": "file", "required": True},
                {"id": "bus_dir", "path": ".pluribus/bus", "type": "dir", "required": True},
                {"id": "bus_events", "path": ".pluribus/bus/events.ndjson", "type": "file", "required": True},
                {"id": "index_dir", "path": ".pluribus/index", "type": "dir", "required": True},
                {"id": "objects_dir", "path": ".pluribus/objects", "type": "dir", "required": True},
                {"id": "kroma_dir", "path": ".pluribus/kroma", "type": "dir", "required": False},
                {"id": "mcp_dir", "path": ".pluribus/mcp", "type": "dir", "required": False},
            ]
        },
        "indices": {
            "description": "Required index files",
            "checks": [
                {"id": "sota_index", "path": ".pluribus/index/sota.ndjson", "type": "file", "required": True},
                {"id": "curation_index", "path": ".pluribus/index/curation.ndjson", "type": "file", "required": True},
                {"id": "kg_nodes", "path": ".pluribus/index/kg_nodes.ndjson", "type": "file", "required": True},
                {"id": "kg_edges", "path": ".pluribus/index/kg_edges.ndjson", "type": "file", "required": True},
            ]
        },
        "protocol": {
            "description": "Protocol compliance (nexus bridge)",
            "checks": [
                {"id": "nexus_readme", "path": "nexus_bridge/README.md", "type": "file", "required": True},
                {"id": "sextext", "path": "sextext.md", "type": "file", "required": True},
                {"id": "auom", "path": "auom.md", "type": "file_or_alt", "alt": "AuOM.md", "required": True},
            ]
        },
        "agents": {
            "description": "Agent home directories",
            "checks": [
                {"id": "agent_homes_dir", "path": ".pluribus/agent_homes", "type": "dir", "required": False},
                {"id": "claude_home", "path": ".pluribus/agent_homes/claude", "type": "dir", "required": False},
                {"id": "codex_home", "path": ".pluribus/agent_homes/codex", "type": "dir", "required": False},
            ]
        },
        "kroma": {
            "description": "Kroma infrastructure",
            "checks": [
                {"id": "kroma_md", "path": "kroma.md", "type": "file", "required": False},
                {"id": "entelexis_md", "path": "entelexis.md", "type": "file", "required": False},
                {"id": "kroma_exact", "path": "nucleus/tools/kroma_exact.py", "type": "file", "required": False},
                {"id": "kroma_inertia", "path": "nucleus/tools/kroma_inertia.py", "type": "file", "required": False},
                {"id": "omega_automata", "path": "nucleus/tools/omega_automata.py", "type": "file", "required": True},
            ]
        },
        "liveness": {
            "description": "ω-automata liveness specs",
            "checks": [
                {"id": "omega_specs_dir", "path": ".pluribus/kroma/omega_specs", "type": "dir", "required": False},
                {"id": "inertia_tensor", "path": ".pluribus/kroma/inertia_tensor.json", "type": "file", "required": False},
            ]
        }
    }
}


# --- Data Structures ---

@dataclass
class CheckResult:
    """Result of a single check."""
    check_id: str
    domain: str
    passed: bool
    message: str
    path: str
    fixable: bool = False
    fix_action: Optional[str] = None


@dataclass
class VORCycle:
    """One complete VOR cycle result."""
    cycle_id: str
    timestamp: float
    iso: str
    total_checks: int
    passed: int
    failed: int
    fixed: int
    cdi: float  # Course Deviation Indicator: 0.0 = on course, 1.0 = fully off
    results: List[CheckResult] = field(default_factory=list)
    fixes_applied: List[dict] = field(default_factory=list)


@dataclass
class VORState:
    """Persistent VOR state across cycles."""
    cycles_run: int = 0
    last_cdi: float = 1.0
    cdi_history: List[float] = field(default_factory=list)
    cumulative_fixes: int = 0
    domains_healthy: Dict[str, bool] = field(default_factory=dict)


# --- Check Functions ---

def check_file_exists(root: Path, check: dict) -> CheckResult:
    """Check if a file exists."""
    path = root / check["path"]
    exists = path.is_file()
    return CheckResult(
        check_id=check["id"],
        domain=check.get("domain", "structural"),
        passed=exists,
        message=f"File exists: {check['path']}" if exists else f"Missing file: {check['path']}",
        path=check["path"],
        fixable=not check.get("required", True),  # Non-required can be created
        fix_action="create_file" if not exists else None,
    )


def check_dir_exists(root: Path, check: dict) -> CheckResult:
    """Check if a directory exists."""
    path = root / check["path"]
    exists = path.is_dir()
    return CheckResult(
        check_id=check["id"],
        domain=check.get("domain", "structural"),
        passed=exists,
        message=f"Directory exists: {check['path']}" if exists else f"Missing directory: {check['path']}",
        path=check["path"],
        fixable=True,
        fix_action="create_dir" if not exists else None,
    )


def check_file_or_alt(root: Path, check: dict) -> CheckResult:
    """Check if file or its alternate exists."""
    path = root / check["path"]
    alt_path = root / check.get("alt", check["path"])
    exists = path.is_file() or alt_path.is_file()
    which = check["path"] if path.is_file() else (check.get("alt") if alt_path.is_file() else None)
    return CheckResult(
        check_id=check["id"],
        domain=check.get("domain", "structural"),
        passed=exists,
        message=f"File exists: {which}" if exists else f"Missing file: {check['path']} (or {check.get('alt')})",
        path=check["path"],
        fixable=False,
    )


# --- VOR Engine ---

class VOREngine:
    """
    VOR/MCTS Self-Verification and Optimization Engine.

    Implements Gauss-Schläfli-Hertz lineage:
    - Gauss: Manifold priors for interpreting signals in curved space
    - Schläfli: Polytope geometry for phase-encoded representations
    - Hertz: Physical validation of signal propagation
    """

    def __init__(self, root: Path, spec: dict = PLURIBUS_SPEC):
        self.root = root
        self.spec = spec
        self.state = self._load_state()

    def _load_state(self) -> VORState:
        """Load persistent VOR state."""
        state_path = self.root / ".pluribus" / "kroma" / "vor_state.json"
        if state_path.exists():
            data = json.loads(state_path.read_text(encoding="utf-8"))
            state = VORState(
                cycles_run=data.get("cycles_run", 0),
                last_cdi=data.get("last_cdi", 1.0),
                cdi_history=data.get("cdi_history", []),
                cumulative_fixes=data.get("cumulative_fixes", 0),
                domains_healthy=data.get("domains_healthy", {}),
            )
            return state
        return VORState()

    def _save_state(self) -> None:
        """Save persistent VOR state."""
        kroma_dir = self.root / ".pluribus" / "kroma"
        kroma_dir.mkdir(parents=True, exist_ok=True)
        state_path = kroma_dir / "vor_state.json"
        data = {
            "cycles_run": self.state.cycles_run,
            "last_cdi": self.state.last_cdi,
            "cdi_history": self.state.cdi_history[-100:],  # Keep last 100
            "cumulative_fixes": self.state.cumulative_fixes,
            "domains_healthy": self.state.domains_healthy,
        }
        state_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Phase 1: Selection (Tune & identify beacon)
    def select_domains(self, focus: Optional[List[str]] = None) -> List[str]:
        """Select which domains to check (UCT-style selection)."""
        if focus:
            return [d for d in focus if d in self.spec["domains"]]

        # Prioritize unhealthy domains
        domains = list(self.spec["domains"].keys())
        unhealthy = [d for d in domains if not self.state.domains_healthy.get(d, False)]
        healthy = [d for d in domains if self.state.domains_healthy.get(d, False)]

        # UCT-style: explore unhealthy first, but occasionally check healthy
        return unhealthy + healthy

    # Phase 2: Expansion (Dial radial)
    def expand_checks(self, domains: List[str]) -> List[dict]:
        """Expand domains into individual checks."""
        checks = []
        for domain in domains:
            domain_spec = self.spec["domains"].get(domain, {})
            for check in domain_spec.get("checks", []):
                check_with_domain = {**check, "domain": domain}
                checks.append(check_with_domain)
        return checks

    # Phase 3: Simulation (Read CDI needle)
    def run_checks(self, checks: List[dict]) -> List[CheckResult]:
        """Run all checks and return results."""
        results = []
        for check in checks:
            check_type = check.get("type", "file")

            if check_type == "file":
                result = check_file_exists(self.root, check)
            elif check_type == "dir":
                result = check_dir_exists(self.root, check)
            elif check_type == "file_or_alt":
                result = check_file_or_alt(self.root, check)
            else:
                result = CheckResult(
                    check_id=check["id"],
                    domain=check.get("domain", "unknown"),
                    passed=False,
                    message=f"Unknown check type: {check_type}",
                    path=check.get("path", ""),
                )

            results.append(result)

        return results

    # Phase 4: Backpropagation (Wind correction)
    def apply_fixes(self, results: List[CheckResult], auto_fix: bool = True) -> List[dict]:
        """Apply fixes for failed checks where possible."""
        fixes = []

        if not auto_fix:
            return fixes

        for result in results:
            if result.passed or not result.fixable:
                continue

            fix = {"check_id": result.check_id, "action": result.fix_action, "path": result.path}

            if result.fix_action == "create_dir":
                path = self.root / result.path
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    fix["success"] = True
                    fix["message"] = f"Created directory: {result.path}"
                except Exception as e:
                    fix["success"] = False
                    fix["message"] = f"Failed to create directory: {e}"

            elif result.fix_action == "create_file":
                path = self.root / result.path
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        # Create empty file or with minimal content
                        if result.path.endswith(".ndjson"):
                            path.write_text("", encoding="utf-8")
                        elif result.path.endswith(".json"):
                            path.write_text("{}", encoding="utf-8")
                        else:
                            path.write_text("", encoding="utf-8")
                        fix["success"] = True
                        fix["message"] = f"Created file: {result.path}"
                except Exception as e:
                    fix["success"] = False
                    fix["message"] = f"Failed to create file: {e}"
            else:
                fix["success"] = False
                fix["message"] = f"Unknown fix action: {result.fix_action}"

            fixes.append(fix)

        return fixes

    # Phase 5: Iteration (Cross-check & correct)
    def compute_cdi(self, results: List[CheckResult]) -> float:
        """
        Compute Course Deviation Indicator.

        CDI = 0.0 means perfectly on course (all checks pass)
        CDI = 1.0 means fully off course (all required checks fail)
        """
        if not results:
            return 1.0

        # Weight required checks more heavily
        total_weight = 0.0
        deviation = 0.0

        for result in results:
            weight = 2.0 if not result.fixable else 1.0  # Required checks weighted 2x
            total_weight += weight
            if not result.passed:
                deviation += weight

        return deviation / total_weight if total_weight > 0 else 1.0

    def run_cycle(self, focus: Optional[List[str]] = None, auto_fix: bool = True) -> VORCycle:
        """Run one complete VOR cycle."""
        cycle_id = str(uuid.uuid4())[:8]
        ts = time.time()
        iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))

        # Phase 1: Selection
        domains = self.select_domains(focus)

        # Phase 2: Expansion
        checks = self.expand_checks(domains)

        # Phase 3: Simulation
        results = self.run_checks(checks)

        # Phase 4: Backpropagation
        fixes = self.apply_fixes(results, auto_fix)

        # Re-run checks after fixes
        if fixes:
            results = self.run_checks(checks)

        # Phase 5: Compute CDI
        cdi = self.compute_cdi(results)

        # Update state
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed)
        fixed = sum(1 for f in fixes if f.get("success", False))

        self.state.cycles_run += 1
        self.state.last_cdi = cdi
        self.state.cdi_history.append(cdi)
        self.state.cumulative_fixes += fixed

        # Update domain health
        domain_results: Dict[str, List[bool]] = {}
        for r in results:
            domain_results.setdefault(r.domain, []).append(r.passed)
        for domain, outcomes in domain_results.items():
            self.state.domains_healthy[domain] = all(outcomes)

        self._save_state()

        return VORCycle(
            cycle_id=cycle_id,
            timestamp=ts,
            iso=iso,
            total_checks=len(results),
            passed=passed,
            failed=failed,
            fixed=fixed,
            cdi=cdi,
            results=results,
            fixes_applied=fixes,
        )


# --- Bus Integration ---

def emit_bus(bus_dir: str | None, *, topic: str, kind: str, level: str, actor: str, data) -> None:
    """Emit event to agent bus."""
    if not bus_dir:
        return
    tool = Path(__file__).with_name("agent_bus.py")
    if not tool.exists():
        return
    subprocess.run(
        [
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
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# --- CLI Commands ---

def find_pluribus_root(start: Path) -> Path | None:
    """Find Pluribus root directory."""
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def cmd_check(args: argparse.Namespace) -> int:
    """Run VOR check cycle."""
    root = Path(args.root).expanduser().resolve() if args.root else find_pluribus_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    engine = VOREngine(root)
    focus = args.domain.split(",") if args.domain else None
    cycle = engine.run_cycle(focus=focus, auto_fix=args.fix)

    # Output
    output = {
        "cycle_id": cycle.cycle_id,
        "iso": cycle.iso,
        "cdi": round(cycle.cdi, 4),
        "total": cycle.total_checks,
        "passed": cycle.passed,
        "failed": cycle.failed,
        "fixed": cycle.fixed,
        "on_course": cycle.cdi < 0.1,
    }

    if args.verbose:
        output["results"] = [asdict(r) for r in cycle.results]
        output["fixes"] = cycle.fixes_applied

    print(json.dumps(output, indent=2))

    # Emit to bus
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if bus_dir:
        emit_bus(
            bus_dir,
            topic="kroma.vor.cycle",
            kind="metric",
            level="info" if cycle.cdi < 0.3 else "warn",
            actor=os.environ.get("PLURIBUS_ACTOR", "kroma"),
            data=output,
        )

    return 0 if cycle.cdi < 0.3 else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show VOR status."""
    root = Path(args.root).expanduser().resolve() if args.root else find_pluribus_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    engine = VOREngine(root)

    # Compute CDI trend
    history = engine.state.cdi_history
    trend = "stable"
    if len(history) >= 3:
        recent = history[-3:]
        if recent[-1] < recent[0]:
            trend = "improving"
        elif recent[-1] > recent[0]:
            trend = "degrading"

    output = {
        "cycles_run": engine.state.cycles_run,
        "last_cdi": round(engine.state.last_cdi, 4),
        "trend": trend,
        "cumulative_fixes": engine.state.cumulative_fixes,
        "domains_healthy": engine.state.domains_healthy,
        "on_course": engine.state.last_cdi < 0.1,
    }

    print(json.dumps(output, indent=2))
    return 0


def cmd_spec(args: argparse.Namespace) -> int:
    """Show the formal Pluribus specification."""
    print(json.dumps(PLURIBUS_SPEC, indent=2))
    return 0


def cmd_integrate_pluribuscheck(args: argparse.Namespace) -> int:
    """
    Run VOR as part of PLURIBUSCHECK response.

    This is the Gödel/CMP integration point: each PLURIBUSCHECK
    triggers a VOR cycle that self-verifies and self-improves.
    """
    root = Path(args.root).expanduser().resolve() if args.root else find_pluribus_root(Path.cwd())
    if not root:
        sys.stderr.write("Error: No Pluribus root found.\n")
        return 1

    engine = VOREngine(root)
    cycle = engine.run_cycle(auto_fix=True)

    # Generate PLURIBUSCHECK-compatible report
    report = {
        "topic": "pluribus.check.report",
        "kind": "metric",
        "actor": os.environ.get("PLURIBUS_ACTOR", "kroma_vor"),
        "data": {
            "status": "working" if cycle.cdi > 0.3 else "idle",
            "queue_depth": cycle.failed,
            "current_task": {
                "goal": "VOR self-verification cycle",
                "step": f"CDI={cycle.cdi:.2f}",
                "started_iso": cycle.iso,
            },
            "blockers": [r.message for r in cycle.results if not r.passed and not r.fixable],
            "context": {
                "cwd": str(root),
                "last_tool_used": "kroma_vor",
                "vor_cycle_id": cycle.cycle_id,
            },
            "health": "nominal" if cycle.cdi < 0.1 else ("degraded" if cycle.cdi < 0.5 else "critical"),
            "vor_metrics": {
                "cdi": round(cycle.cdi, 4),
                "passed": cycle.passed,
                "failed": cycle.failed,
                "fixed": cycle.fixed,
                "cycles_total": engine.state.cycles_run,
            }
        }
    }

    print(json.dumps(report, indent=2))

    # Emit to bus
    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    if bus_dir:
        emit_bus(
            bus_dir,
            topic="pluribus.check.report",
            kind="metric",
            level="info",
            actor=os.environ.get("PLURIBUS_ACTOR", "kroma_vor"),
            data=report["data"],
        )

    return 0


# --- Main ---

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kroma_vor.py",
        description="VOR/MCTS Self-Verification and Optimization Routine for Pluribus",
    )
    p.add_argument("--root", default=None, help="Pluribus root directory")
    p.add_argument("--bus-dir", default=None, help="Agent bus directory")

    sub = p.add_subparsers(dest="cmd", required=True)

    # check
    check_p = sub.add_parser("check", help="Run VOR check cycle")
    check_p.add_argument("--domain", default=None, help="Comma-separated domains to check")
    check_p.add_argument("--fix", action="store_true", help="Auto-fix fixable issues")
    check_p.add_argument("--verbose", "-v", action="store_true", help="Show detailed results")
    check_p.set_defaults(func=cmd_check)

    # status
    status_p = sub.add_parser("status", help="Show VOR status")
    status_p.set_defaults(func=cmd_status)

    # spec
    spec_p = sub.add_parser("spec", help="Show formal Pluribus specification")
    spec_p.set_defaults(func=cmd_spec)

    # pluribuscheck (integration point)
    pc_p = sub.add_parser("pluribuscheck", help="Run VOR as PLURIBUSCHECK response")
    pc_p.set_defaults(func=cmd_integrate_pluribuscheck)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
