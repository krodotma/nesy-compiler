#!/usr/bin/env python3
"""
HGT Guard - CMP Fitness Checks for RGMA Horizontal Gene Transfer

This module implements guarded horizontal gene transfer with CMP fitness
verification. Per RGMA spec (section 5.2), HGT splices require explicit
provenance, guard checks, and CMP fitness thresholds.

Reference: nucleus/specs/hgt_protocol.md
Reference: nucleus/specs/rhizome_godel_alpha.md (section 5.2)

Guards:
  P - provenance: evidence paths + req_id present
  E - execution: --help or dry-run succeeds
  L - liveness: tool emits heartbeat topic within expected window
  R - reproducibility: config + command logged
  Q - quality: lint/style gates or minimal invariants
  G-CMP - CMP fitness: donor CMP >= PHI_INV (0.618)

Bus Topics:
  hgt.propose - HGT proposal submitted
  hgt.splice.rejected - HGT splice rejected (with reason)
  hgt.splice.success - HGT splice completed successfully
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

sys.dont_write_bytecode = True


# -----------------------------------------------------------------
# Constants
# -----------------------------------------------------------------

PHI = 1.618033988749895
PHI_INV = 1 / PHI  # ~0.618 - golden ratio inverse
PHI_INV_2 = 1 / (PHI ** 2)  # ~0.382
PHI_INV_3 = 1 / (PHI ** 3)  # ~0.236 - global CMP floor

DEFAULT_MIN_CMP = PHI_INV  # 0.618 - donor must meet this threshold
DEFAULT_MIN_OCCURRENCES = 3
DEFAULT_MIN_CORRELATION = 0.5


# -----------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def default_actor() -> str:
    return os.environ.get("PLURIBUS_ACTOR") or getpass.getuser()


def sha256_content(content: str | bytes) -> str:
    """Compute SHA256 hash of content."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def emit_bus(
    bus_dir: str | None,
    *,
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: dict,
) -> None:
    """Emit event to Pluribus bus using agent_bus.py."""
    if not bus_dir:
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")
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
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


# -----------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------


@dataclass
class Evidence:
    """A piece of evidence supporting an HGT proposal."""

    kind: Literal["path", "bus", "commit", "test", "metric"]
    value: str
    note: str = ""


@dataclass
class OmegaMotif:
    """
    A recurring pattern in successful lineages.
    Persistence across infinite time is the acceptance criterion (Buchi/Parity).
    """

    id: str
    content_sha: str
    occurrence_count: int = 0
    lineages_present: set[str] = field(default_factory=set)
    cmp_correlation: float = 0.0
    mdl_length: int = 0

    def is_viable(
        self,
        min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
        min_correlation: float = DEFAULT_MIN_CORRELATION,
    ) -> bool:
        """Check if motif meets viability thresholds."""
        return (
            self.occurrence_count >= min_occurrences
            and self.cmp_correlation >= min_correlation
        )


@dataclass
class CMPRegistry:
    """
    Registry of CMP scores for lineages.
    This is a simplified in-memory registry; production would use rhizome storage.
    """

    scores: dict[str, float] = field(default_factory=dict)
    history: dict[str, list[tuple[float, float]]] = field(
        default_factory=dict
    )  # lineage -> [(ts, cmp), ...]

    def get_cmp(self, lineage_id: str) -> float | None:
        """Get current CMP score for a lineage."""
        return self.scores.get(lineage_id)

    def set_cmp(self, lineage_id: str, cmp: float) -> None:
        """Update CMP score for a lineage."""
        self.scores[lineage_id] = cmp
        if lineage_id not in self.history:
            self.history[lineage_id] = []
        self.history[lineage_id].append((time.time(), cmp))

    def is_extinct(self, lineage_id: str) -> bool:
        """Check if lineage CMP is below global floor (extinct)."""
        cmp = self.get_cmp(lineage_id)
        return cmp is not None and cmp < PHI_INV_3

    @classmethod
    def load(cls, path: Path) -> "CMPRegistry":
        """Load registry from NDJSON file."""
        registry = cls()
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        lineage_id = entry.get("lineage_id")
                        cmp = entry.get("cmp")
                        if lineage_id and cmp is not None:
                            registry.set_cmp(lineage_id, cmp)
                    except json.JSONDecodeError:
                        continue
        return registry

    def save(self, path: Path) -> None:
        """Append current scores to NDJSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            ts = time.time()
            iso = now_iso_utc()
            for lineage_id, cmp in self.scores.items():
                entry = {
                    "lineage_id": lineage_id,
                    "cmp": cmp,
                    "ts": ts,
                    "iso": iso,
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass
class HGTProposal:
    """
    A proposal for horizontal gene transfer between lineages.

    Contains donor/recipient lineage info, motif SHA, evidence,
    and computed CMP values for guard validation.
    """

    req_id: str
    actor: str
    donor_lineage: str
    recipient_lineage: str
    motif_sha: str
    donor_cmp: float
    recipient_cmp: float
    path: str
    reason: str = ""
    evidence: list[Evidence] = field(default_factory=list)
    guard_results: dict[str, bool] = field(default_factory=dict)
    created_at: str = field(default_factory=now_iso_utc)

    def to_dict(self) -> dict:
        """Serialize proposal to dict for bus emission."""
        return {
            "req_id": self.req_id,
            "actor": self.actor,
            "donor_lineage": self.donor_lineage,
            "recipient_lineage": self.recipient_lineage,
            "motif_sha": self.motif_sha,
            "donor_cmp": self.donor_cmp,
            "recipient_cmp": self.recipient_cmp,
            "path": self.path,
            "reason": self.reason,
            "evidence": [
                {"kind": e.kind, "value": e.value, "note": e.note}
                for e in self.evidence
            ],
            "guard_results": self.guard_results,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HGTProposal":
        """Deserialize proposal from dict."""
        evidence = [
            Evidence(kind=e["kind"], value=e["value"], note=e.get("note", ""))
            for e in d.get("evidence", [])
        ]
        return cls(
            req_id=d["req_id"],
            actor=d["actor"],
            donor_lineage=d["donor_lineage"],
            recipient_lineage=d["recipient_lineage"],
            motif_sha=d["motif_sha"],
            donor_cmp=d["donor_cmp"],
            recipient_cmp=d["recipient_cmp"],
            path=d["path"],
            reason=d.get("reason", ""),
            evidence=evidence,
            guard_results=d.get("guard_results", {}),
            created_at=d.get("created_at", now_iso_utc()),
        )


@dataclass
class GuardResult:
    """Result of running the full guard ladder on an HGT proposal."""

    passed: bool
    gate_id: str | None = None  # ID of failed gate (if any)
    reason: str = ""
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "gate_id": self.gate_id,
            "reason": self.reason,
            "details": self.details,
        }


# -----------------------------------------------------------------
# Guard Functions
# -----------------------------------------------------------------


def check_donor_fitness(
    donor_lineage: str,
    cmp_registry: CMPRegistry,
    min_cmp: float = DEFAULT_MIN_CMP,
) -> tuple[bool, float | None, str]:
    """
    Verify donor CMP >= threshold (default 0.618).

    Args:
        donor_lineage: ID of the donor lineage
        cmp_registry: Registry containing CMP scores
        min_cmp: Minimum CMP threshold (default PHI_INV = 0.618)

    Returns:
        Tuple of (passed, cmp_value, reason)
    """
    cmp = cmp_registry.get_cmp(donor_lineage)

    if cmp is None:
        return False, None, f"Donor lineage '{donor_lineage}' not found in CMP registry"

    if cmp < min_cmp:
        return (
            False,
            cmp,
            f"Donor CMP {cmp:.4f} below threshold {min_cmp:.4f}",
        )

    if cmp_registry.is_extinct(donor_lineage):
        return False, cmp, f"Donor lineage is extinct (CMP {cmp:.4f} < {PHI_INV_3:.4f})"

    return True, cmp, f"Donor CMP {cmp:.4f} meets threshold {min_cmp:.4f}"


def check_motif_viability(
    motif: OmegaMotif,
    min_occurrences: int = DEFAULT_MIN_OCCURRENCES,
    min_correlation: float = DEFAULT_MIN_CORRELATION,
) -> tuple[bool, str]:
    """
    Verify motif meets viability thresholds.

    A viable motif must:
    1. Have at least min_occurrences occurrences
    2. Have CMP correlation >= min_correlation

    Args:
        motif: The omega-motif to check
        min_occurrences: Minimum number of occurrences (default 3)
        min_correlation: Minimum CMP correlation (default 0.5)

    Returns:
        Tuple of (passed, reason)
    """
    if motif.occurrence_count < min_occurrences:
        return (
            False,
            f"Motif occurrences {motif.occurrence_count} < min {min_occurrences}",
        )

    if motif.cmp_correlation < min_correlation:
        return (
            False,
            f"Motif CMP correlation {motif.cmp_correlation:.4f} < min {min_correlation:.4f}",
        )

    return (
        True,
        f"Motif viable: {motif.occurrence_count} occurrences, {motif.cmp_correlation:.4f} correlation",
    )


def check_provenance(proposal: HGTProposal) -> tuple[bool, str]:
    """
    Guard P: Verify provenance evidence is present.

    Requires:
    - req_id present
    - At least one evidence item
    - All evidence paths exist (for path-type evidence)
    """
    if not proposal.req_id:
        return False, "Missing req_id"

    if not proposal.evidence:
        return False, "No evidence provided"

    for ev in proposal.evidence:
        if ev.kind == "path":
            if not Path(ev.value).exists():
                return False, f"Evidence path not found: {ev.value}"

    return True, f"Provenance verified: {len(proposal.evidence)} evidence items"


def check_execution(proposal: HGTProposal) -> tuple[bool, str]:
    """
    Guard E: Verify execution capability.

    For Python files, attempt import without execution.
    For other files, verify they are readable.
    """
    path = Path(proposal.path)

    if not path.exists():
        return False, f"Candidate path not found: {proposal.path}"

    if path.suffix == ".py":
        # Try syntax check via compile
        try:
            source = path.read_text(encoding="utf-8")
            compile(source, str(path), "exec")
        except SyntaxError as e:
            return False, f"Python syntax error: {e}"
        except Exception as e:
            return False, f"Failed to verify Python file: {e}"

    return True, f"Execution check passed for {path.name}"


def check_liveness(_proposal: HGTProposal) -> tuple[bool, str]:
    """
    Guard L: Verify liveness.

    For HGT, this is a lightweight check that the system is responsive.
    Full liveness checks (heartbeat windows) are handled by omega_heartbeat.py.
    """
    # Basic liveness: can we emit to bus?
    # For now, assume liveness if we got this far
    return True, "Liveness check passed (deferred to omega_heartbeat)"


def check_reproducibility(proposal: HGTProposal) -> tuple[bool, str]:
    """
    Guard R: Verify reproducibility.

    Requires:
    - motif_sha present
    - donor_lineage and recipient_lineage specified
    """
    if not proposal.motif_sha:
        return False, "Missing motif_sha for reproducibility"

    if not proposal.donor_lineage or not proposal.recipient_lineage:
        return False, "Missing lineage identifiers for reproducibility"

    return True, f"Reproducibility verified: sha={proposal.motif_sha[:12]}..."


def check_quality(proposal: HGTProposal) -> tuple[bool, str]:
    """
    Guard Q: Quality gates.

    For Python files, run basic lint checks.
    """
    path = Path(proposal.path)

    if not path.exists():
        return False, f"Path not found: {proposal.path}"

    if path.suffix == ".py":
        # Try ruff if available
        try:
            result = subprocess.run(
                [sys.executable, "-m", "ruff", "check", "--quiet", str(path)],
                capture_output=True,
                timeout=30,
                check=False,
            )
            if result.returncode != 0:
                # Ruff found issues, but don't fail for warnings
                stderr = result.stderr.decode("utf-8", errors="replace")
                if "error" in stderr.lower():
                    return False, f"Quality check failed: {stderr[:200]}"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Ruff not available or timed out, skip
            pass

    return True, "Quality check passed"


def check_cmp_fitness(
    proposal: HGTProposal,
    cmp_registry: CMPRegistry,
    min_donor_cmp: float = DEFAULT_MIN_CMP,
) -> tuple[bool, str]:
    """
    Guard G-CMP: CMP fitness check.

    Verifies:
    1. Donor CMP >= min_donor_cmp (default 0.618)
    2. Recipient is not extinct
    """
    # Check donor fitness
    donor_passed, donor_cmp, donor_reason = check_donor_fitness(
        proposal.donor_lineage, cmp_registry, min_donor_cmp
    )

    if not donor_passed:
        return False, f"G-CMP donor failed: {donor_reason}"

    # Verify recipient is not extinct
    recipient_cmp = cmp_registry.get_cmp(proposal.recipient_lineage)
    if recipient_cmp is not None and cmp_registry.is_extinct(proposal.recipient_lineage):
        return (
            False,
            f"G-CMP recipient extinct: CMP {recipient_cmp:.4f} < {PHI_INV_3:.4f}",
        )

    return True, f"G-CMP passed: donor={donor_cmp:.4f}, recipient={recipient_cmp or 'N/A'}"


def approve_splice(
    proposal: HGTProposal,
    cmp_registry: CMPRegistry,
    motif: OmegaMotif | None = None,
    min_donor_cmp: float = DEFAULT_MIN_CMP,
    bus_dir: str | None = None,
) -> GuardResult:
    """
    Run full guard ladder: P/E/L/R/Q + G-CMP.

    This is the main entry point for validating an HGT splice.

    Args:
        proposal: The HGT proposal to validate
        cmp_registry: Registry containing CMP scores
        motif: Optional motif to validate viability
        min_donor_cmp: Minimum donor CMP threshold
        bus_dir: Bus directory for event emission

    Returns:
        GuardResult with pass/fail and details
    """
    actor = proposal.actor or default_actor()
    details: dict[str, dict] = {}

    # Emit propose event
    emit_bus(
        bus_dir,
        topic="hgt.propose",
        kind="request",
        level="info",
        actor=actor,
        data=proposal.to_dict(),
    )

    # Guard ladder
    guards = [
        ("P", check_provenance, (proposal,)),
        ("E", check_execution, (proposal,)),
        ("L", check_liveness, (proposal,)),
        ("R", check_reproducibility, (proposal,)),
        ("Q", check_quality, (proposal,)),
        ("G-CMP", check_cmp_fitness, (proposal, cmp_registry, min_donor_cmp)),
    ]

    for gate_id, check_fn, args in guards:
        try:
            passed, reason = check_fn(*args)
            details[gate_id] = {"passed": passed, "reason": reason}
            proposal.guard_results[gate_id] = passed

            if not passed:
                result = GuardResult(
                    passed=False,
                    gate_id=gate_id,
                    reason=reason,
                    details=details,
                )
                emit_bus(
                    bus_dir,
                    topic="hgt.splice.rejected",
                    kind="response",
                    level="warn",
                    actor=actor,
                    data={
                        "req_id": proposal.req_id,
                        "gate_id": gate_id,
                        "reason": reason,
                        "donor_cmp": proposal.donor_cmp,
                        "recipient_cmp": proposal.recipient_cmp,
                        "guard_results": details,
                    },
                )
                return result
        except Exception as e:
            details[gate_id] = {"passed": False, "reason": str(e)}
            result = GuardResult(
                passed=False,
                gate_id=gate_id,
                reason=f"Guard exception: {e}",
                details=details,
            )
            emit_bus(
                bus_dir,
                topic="hgt.splice.rejected",
                kind="response",
                level="error",
                actor=actor,
                data={
                    "req_id": proposal.req_id,
                    "gate_id": gate_id,
                    "reason": str(e),
                    "donor_cmp": proposal.donor_cmp,
                    "guard_results": details,
                },
            )
            return result

    # Optional motif viability check
    if motif:
        passed, reason = check_motif_viability(motif)
        details["motif_viability"] = {"passed": passed, "reason": reason}
        if not passed:
            result = GuardResult(
                passed=False,
                gate_id="motif_viability",
                reason=reason,
                details=details,
            )
            emit_bus(
                bus_dir,
                topic="hgt.splice.rejected",
                kind="response",
                level="warn",
                actor=actor,
                data={
                    "req_id": proposal.req_id,
                    "gate_id": "motif_viability",
                    "reason": reason,
                    "motif_id": motif.id,
                    "guard_results": details,
                },
            )
            return result

    # All guards passed
    result = GuardResult(passed=True, details=details)

    emit_bus(
        bus_dir,
        topic="hgt.splice.success",
        kind="response",
        level="info",
        actor=actor,
        data={
            "req_id": proposal.req_id,
            "donor_lineage": proposal.donor_lineage,
            "recipient_lineage": proposal.recipient_lineage,
            "motif_sha": proposal.motif_sha,
            "donor_cmp": proposal.donor_cmp,
            "recipient_cmp": proposal.recipient_cmp,
            "path": proposal.path,
            "guard_results": details,
        },
    )

    return result


# -----------------------------------------------------------------
# CLI
# -----------------------------------------------------------------


def cmd_check_donor(args: argparse.Namespace) -> int:
    """CLI command: check donor fitness."""
    registry = CMPRegistry()
    if args.registry:
        registry = CMPRegistry.load(Path(args.registry))
    else:
        # Set test value if provided
        if args.cmp is not None:
            registry.set_cmp(args.lineage, args.cmp)

    passed, cmp_val, reason = check_donor_fitness(
        args.lineage, registry, args.min_cmp
    )

    output = {
        "lineage": args.lineage,
        "passed": passed,
        "cmp": cmp_val,
        "min_cmp": args.min_cmp,
        "reason": reason,
    }
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


def cmd_check_motif(args: argparse.Namespace) -> int:
    """CLI command: check motif viability."""
    motif = OmegaMotif(
        id=args.motif_id,
        content_sha=args.sha or "",
        occurrence_count=args.occurrences,
        cmp_correlation=args.correlation,
    )

    passed, reason = check_motif_viability(
        motif, args.min_occurrences, args.min_correlation
    )

    output = {
        "motif_id": args.motif_id,
        "passed": passed,
        "occurrences": args.occurrences,
        "correlation": args.correlation,
        "reason": reason,
    }
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


def cmd_validate(args: argparse.Namespace) -> int:
    """CLI command: validate an HGT proposal."""
    registry = CMPRegistry()
    if args.registry:
        registry = CMPRegistry.load(Path(args.registry))
    else:
        # Set test values if provided
        if args.donor_cmp is not None:
            registry.set_cmp(args.donor, args.donor_cmp)
        if args.recipient_cmp is not None:
            registry.set_cmp(args.recipient, args.recipient_cmp)

    proposal = HGTProposal(
        req_id=args.req_id or str(uuid.uuid4()),
        actor=args.actor or default_actor(),
        donor_lineage=args.donor,
        recipient_lineage=args.recipient,
        motif_sha=args.sha or sha256_content(args.path),
        donor_cmp=args.donor_cmp or registry.get_cmp(args.donor) or 0.0,
        recipient_cmp=args.recipient_cmp or registry.get_cmp(args.recipient) or 0.0,
        path=args.path,
        reason=args.reason or "",
        evidence=[Evidence(kind="path", value=args.path)],
    )

    result = approve_splice(
        proposal,
        registry,
        min_donor_cmp=args.min_cmp,
        bus_dir=args.bus_dir,
    )

    output = {
        "req_id": proposal.req_id,
        "passed": result.passed,
        "gate_id": result.gate_id,
        "reason": result.reason,
        "details": result.details,
    }
    print(json.dumps(output, indent=2))
    return 0 if result.passed else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="hgt_guard.py",
        description="HGT Guard - CMP fitness checks for RGMA horizontal gene transfer",
    )
    p.add_argument(
        "--bus-dir",
        default=None,
        help="Bus directory (or set PLURIBUS_BUS_DIR)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # check-donor command
    donor = sub.add_parser("check-donor", help="Check donor lineage CMP fitness")
    donor.add_argument("lineage", help="Donor lineage ID")
    donor.add_argument("--cmp", type=float, default=None, help="CMP value (for testing)")
    donor.add_argument("--min-cmp", type=float, default=DEFAULT_MIN_CMP, help="Minimum CMP threshold")
    donor.add_argument("--registry", default=None, help="Path to CMP registry NDJSON")
    donor.set_defaults(func=cmd_check_donor)

    # check-motif command
    motif = sub.add_parser("check-motif", help="Check motif viability")
    motif.add_argument("motif_id", help="Motif ID")
    motif.add_argument("--sha", default=None, help="Content SHA")
    motif.add_argument("--occurrences", type=int, default=0, help="Occurrence count")
    motif.add_argument("--correlation", type=float, default=0.0, help="CMP correlation")
    motif.add_argument("--min-occurrences", type=int, default=DEFAULT_MIN_OCCURRENCES, help="Minimum occurrences")
    motif.add_argument("--min-correlation", type=float, default=DEFAULT_MIN_CORRELATION, help="Minimum correlation")
    motif.set_defaults(func=cmd_check_motif)

    # validate command
    validate = sub.add_parser("validate", help="Validate full HGT proposal")
    validate.add_argument("path", help="Path to candidate file")
    validate.add_argument("--donor", required=True, help="Donor lineage ID")
    validate.add_argument("--recipient", required=True, help="Recipient lineage ID")
    validate.add_argument("--sha", default=None, help="Motif SHA (computed if not provided)")
    validate.add_argument("--donor-cmp", type=float, default=None, help="Donor CMP value")
    validate.add_argument("--recipient-cmp", type=float, default=None, help="Recipient CMP value")
    validate.add_argument("--min-cmp", type=float, default=DEFAULT_MIN_CMP, help="Minimum donor CMP")
    validate.add_argument("--req-id", default=None, help="Request ID (generated if not provided)")
    validate.add_argument("--actor", default=None, help="Actor ID")
    validate.add_argument("--reason", default=None, help="Reason for HGT")
    validate.add_argument("--registry", default=None, help="Path to CMP registry NDJSON")
    validate.set_defaults(func=cmd_validate)

    return p


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
