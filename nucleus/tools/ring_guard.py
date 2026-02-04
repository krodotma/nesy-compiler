#!/usr/bin/env python3
"""
Ring Guard - SCI/SAP-inspired access control enforcement.

Historical Context:
-------------------
This implementation draws from verified government security practices:

1. MANHATTAN PROJECT (1942-1946):
   - General Leslie Groves implemented "compartmentalization" as "the very heart of security"
   - Workers at Oak Ridge operated centrifuges but didn't know they were enriching uranium
   - Color-coded badges (red/blue for low clearance, white for high) controlled access
   - Knowledge was fragmented so capture of any individual minimized damage

2. SCI/SAP SYSTEM (1946-present):
   - Sensitive Compartmented Information (SCI) emerged from atomic secrets
   - Special Access Programs (SAPs) add layers beyond Top Secret
   - "Need-to-know" is enforced separately from clearance level
   - Read-in ceremonies formally grant access; read-out revokes it

3. INTELLIGENCE CELL STRUCTURES:
   - Cut-outs prevent direct contact between network members
   - Each cell member knows only their handler and one contact
   - Soviet operations used "block" (aware of network) and "chain" (minimal awareness) cut-outs
   - The Prosper network collapse (1943) showed consequences of poor compartmentalization

Technical Implementation:
------------------------
- Ring 0: Kernel/DNA (like Q clearance + SCI for nuclear weapons design)
- Ring 1: Operators (like Top Secret/SCI for intelligence operations)
- Ring 2: Application (like Secret clearance for general classified work)
- Ring 3: Ephemeral (like Confidential or uncleared contractors in SCIFs with escorts)

Every access check follows the principle: clearance alone is insufficient.
The agent must have both the ring level AND the compartment read-in.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus
except ImportError:
    agent_bus = None


# =============================================================================
# RING TAXONOMY
# =============================================================================

class Ring(IntEnum):
    """
    Security rings inspired by x86 privilege levels and SCI/SAP hierarchy.

    Historical parallel:
    - Ring 0 = Q Clearance + SCI (nuclear weapons, most sensitive)
    - Ring 1 = Top Secret/SCI (intelligence operations)
    - Ring 2 = Secret (routine classified work)
    - Ring 3 = Confidential/Uncleared (escorted access only)
    """
    KERNEL = 0      # DNA, protocols, PQC infrastructure
    OPERATOR = 1    # Semantic operators, bus infrastructure, omega
    APPLICATION = 2 # Dashboard, agent homes, reports
    EPHEMERAL = 3   # PAIP clones, sandboxed execution


# =============================================================================
# COMPARTMENTS (SCI-style)
# =============================================================================

# These mirror real SCI compartments like TK (TALENT KEYHOLE), SI (SPECIAL INTELLIGENCE)
COMPARTMENTS = {
    "PQC": {
        "description": "Post-quantum cryptography infrastructure",
        "min_ring": Ring.KERNEL,
        "gate": "crypto_audit",
        "paths": ["nucleus/specs/pqc_*", ".pluribus/secrets/**"],
    },
    "EVOLUTION": {
        "description": "HGT/VGT evolutionary systems",
        "min_ring": Ring.OPERATOR,
        "gate": "cmp_compliance",
        "paths": ["nucleus/tools/hgt_*", "nucleus/tools/vgt_*"],
    },
    "OMEGA": {
        "description": "Liveness verification automata",
        "min_ring": Ring.OPERATOR,
        "gate": "automata_training",
        "paths": ["nucleus/tools/omega_*", "nucleus/specs/omega_*"],
    },
    "GENESIS": {
        "description": "LUCA modification (most restricted)",
        "min_ring": Ring.KERNEL,
        "gate": "multi_sig",
        "paths": ["GENOME.json", "pluribus-dna/**"],
    },
    "METATOOL": {
        "description": "Core infrastructure tooling",
        "min_ring": Ring.KERNEL,
        "gate": "ring0_approval",
        "paths": ["nucleus/tools/agent_bus.py", "nucleus/tools/lens_*"],
    },
}


# =============================================================================
# CLEARANCE CREDENTIAL (Digital Badge)
# =============================================================================

@dataclass
class AgentClearance:
    """
    Digital equivalent of a Manhattan Project security badge.

    Historical parallel:
    - agent_id = Employee name/number
    - ring = Badge color (red/blue=low, white=high)
    - compartments = SCI program access (TK, SI, etc.)
    - issuer_lineage = Security officer who granted access
    - nda_signature = Signed nondisclosure agreement
    - read_in_timestamp = Date of security briefing
    """
    agent_id: str
    ring: Ring
    compartments: set[str] = field(default_factory=set)
    issuer_lineage: str = ""
    nda_signature: str = ""
    read_in_timestamp: float = 0.0
    last_heartbeat: float = 0.0
    violation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "ring": self.ring.value,
            "compartments": sorted(self.compartments),
            "issuer_lineage": self.issuer_lineage,
            "nda_signature": self.nda_signature,
            "read_in_timestamp": self.read_in_timestamp,
            "last_heartbeat": self.last_heartbeat,
            "violation_count": self.violation_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentClearance":
        return cls(
            agent_id=data["agent_id"],
            ring=Ring(data["ring"]),
            compartments=set(data.get("compartments", [])),
            issuer_lineage=data.get("issuer_lineage", ""),
            nda_signature=data.get("nda_signature", ""),
            read_in_timestamp=data.get("read_in_timestamp", 0.0),
            last_heartbeat=data.get("last_heartbeat", 0.0),
            violation_count=data.get("violation_count", 0),
        )

    def badge_color(self) -> str:
        """Return Manhattan Project-style badge color."""
        if self.ring == Ring.KERNEL:
            return "WHITE"  # Full access (senior scientists)
        elif self.ring == Ring.OPERATOR:
            return "GOLD"   # Operator access
        elif self.ring == Ring.APPLICATION:
            return "BLUE"   # Application worker
        else:
            return "RED"    # Ephemeral/escorted


# =============================================================================
# CLEARANCE LEDGER (Persistent Credential Store)
# =============================================================================

class ClearanceLedger:
    """
    Persistent store for agent clearances.

    Historical parallel:
    - Central personnel security files maintained by security officers
    - Records of all read-ins, read-outs, and access grants
    - Audit trail for counterintelligence review
    """

    def __init__(self, ledger_path: Path):
        self.ledger_path = ledger_path
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, AgentClearance] = {}
        self._load()

    def _load(self):
        if self.ledger_path.exists():
            try:
                with self.ledger_path.open("r") as f:
                    data = json.load(f)
                for agent_id, clearance_data in data.get("clearances", {}).items():
                    self._cache[agent_id] = AgentClearance.from_dict(clearance_data)
            except Exception:
                pass

    def _save(self):
        data = {
            "version": 1,
            "updated_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "clearances": {
                agent_id: clearance.to_dict()
                for agent_id, clearance in self._cache.items()
            },
        }
        with self.ledger_path.open("w") as f:
            json.dump(data, f, indent=2)

    def get(self, agent_id: str) -> Optional[AgentClearance]:
        return self._cache.get(agent_id)

    def set(self, clearance: AgentClearance):
        self._cache[clearance.agent_id] = clearance
        self._save()

    def revoke(self, agent_id: str) -> bool:
        if agent_id in self._cache:
            del self._cache[agent_id]
            self._save()
            return True
        return False

    def list_by_ring(self, ring: Ring) -> list[AgentClearance]:
        return [c for c in self._cache.values() if c.ring == ring]

    def list_by_compartment(self, compartment: str) -> list[AgentClearance]:
        return [c for c in self._cache.values() if compartment in c.compartments]


# =============================================================================
# ACCESS CONTROL POLICY
# =============================================================================

@dataclass
class AccessPolicy:
    """
    Path-based access control rule.

    Historical parallel:
    - SCIF access lists defining who can enter which facility
    - Document classification markings (TS/SCI//TK//SI)
    - "SPECIAL ACCESS REQUIRED" banners on SAP documents
    """
    path_pattern: str
    min_ring: Ring
    compartments_required: list[str] = field(default_factory=list)
    compartments_any: list[str] = field(default_factory=list)  # OR logic
    description: str = ""


class PolicyEngine:
    """
    Evaluates access requests against policies.

    Implements the dual-key principle:
    1. Ring level check (like clearance level)
    2. Compartment check (like SCI program access)

    Both must pass - having Top Secret doesn't mean you can read TK material.
    """

    def __init__(self, policies_path: Optional[Path] = None):
        self.policies: list[AccessPolicy] = []
        if policies_path and policies_path.exists():
            self._load_policies(policies_path)
        else:
            self._load_default_policies()

    def _load_default_policies(self):
        """Default policies matching COMPARTMENTALIZATION_SPEC.md"""
        defaults = [
            AccessPolicy("nucleus/specs/pqc_*", Ring.KERNEL, ["PQC"], [], "PQC infrastructure"),
            AccessPolicy("nucleus/specs/dkin_*", Ring.KERNEL, [], [], "DKIN protocols"),
            AccessPolicy("nucleus/tools/omega_*", Ring.OPERATOR, [], ["OMEGA"], "Omega verification"),
            AccessPolicy("nucleus/dashboard/**", Ring.APPLICATION, [], [], "Dashboard code"),
            AccessPolicy("/tmp/pluribus_*/**", Ring.EPHEMERAL, [], [], "PAIP clones"),
            AccessPolicy(".pluribus/secrets/**", Ring.KERNEL, ["PQC"], [], "Secret storage"),
            AccessPolicy("GENOME.json", Ring.KERNEL, ["GENESIS"], [], "DNA manifest"),
            AccessPolicy("nucleus/tools/agent_bus.py", Ring.OPERATOR, [], ["METATOOL"], "Bus infrastructure"),
            AccessPolicy("agent_reports/**", Ring.APPLICATION, [], [], "Agent reports"),
            AccessPolicy(".pluribus/agent_homes/**", Ring.APPLICATION, [], [], "Agent homes"),
        ]
        self.policies = defaults

    def _load_policies(self, path: Path):
        try:
            with path.open("r") as f:
                data = json.load(f)
            for p in data.get("policies", []):
                self.policies.append(AccessPolicy(
                    path_pattern=p["path"],
                    min_ring=Ring(p["min_ring"]),
                    compartments_required=p.get("compartments_required", []),
                    compartments_any=p.get("compartments_any", []),
                    description=p.get("description", ""),
                ))
        except Exception:
            self._load_default_policies()

    def _match_path(self, pattern: str, path: str) -> bool:
        """Simple glob-style matching."""
        import fnmatch
        return fnmatch.fnmatch(path, pattern)

    def find_policy(self, path: str) -> Optional[AccessPolicy]:
        """Find the most specific policy matching a path."""
        matches = [p for p in self.policies if self._match_path(p.path_pattern, path)]
        if not matches:
            return None
        # Return most specific (longest pattern)
        return max(matches, key=lambda p: len(p.path_pattern))


# =============================================================================
# RING GUARD (Main Enforcement)
# =============================================================================

@dataclass
class AccessDecision:
    """Result of an access check."""
    granted: bool
    reason: str
    policy: Optional[AccessPolicy] = None
    clearance: Optional[AgentClearance] = None


class RingGuard:
    """
    Central access control enforcer.

    Historical parallel:
    - Security officer at SCIF entrance checking badges
    - Guard verifying both clearance level AND program access
    - "Two-person integrity" for nuclear weapons access

    Implementation notes:
    - Every access attempt is logged (continuous audit like SCIF monitoring)
    - Violations trigger demotion consideration
    - Heartbeats maintain active status (liveness proof)
    """

    def __init__(
        self,
        *,
        ledger_path: Optional[Path] = None,
        policies_path: Optional[Path] = None,
        bus_dir: Optional[str] = None,
    ):
        self.ledger = ClearanceLedger(
            ledger_path or Path(REPO_ROOT / ".pluribus/state/clearance_ledger.json")
        )
        self.policy_engine = PolicyEngine(policies_path)
        self.bus_dir = bus_dir or str(REPO_ROOT / ".pluribus/bus")

        # Heartbeat timeout (like SCIF access card timeout)
        self.heartbeat_timeout_s = 300  # 5 minutes

        # Violation threshold before demotion
        self.max_violations = 3

    def _emit_bus(self, topic: str, level: str, data: dict):
        """Emit access event to bus for audit trail."""
        if agent_bus is None:
            return
        try:
            paths = agent_bus.resolve_bus_paths(self.bus_dir)
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind="log",
                level=level,
                actor="ring_guard",
                data=data,
                trace_id=None,
                run_id=None,
                durable=True,
            )
        except Exception:
            pass

    def check_access(
        self,
        agent_id: str,
        path: str,
        operation: str = "read",
    ) -> AccessDecision:
        """
        Check if an agent can access a resource.

        This implements the SCI/SAP dual-key principle:
        1. Does the agent have sufficient ring level?
        2. Does the agent have the required compartment access?

        Historical parallel:
        - Guard checks badge color (ring level)
        - Guard checks SCI access list (compartment membership)
        - Both must pass for entry
        """
        # Get agent clearance
        clearance = self.ledger.get(agent_id)
        if clearance is None:
            decision = AccessDecision(
                granted=False,
                reason=f"Agent {agent_id} has no clearance on file",
            )
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
            })
            return decision

        # Check heartbeat liveness (like access card expiry)
        now = time.time()
        if now - clearance.last_heartbeat > self.heartbeat_timeout_s:
            decision = AccessDecision(
                granted=False,
                reason=f"Agent {agent_id} heartbeat expired ({self.heartbeat_timeout_s}s timeout)",
                clearance=clearance,
            )
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
                "last_heartbeat": clearance.last_heartbeat,
            })
            return decision

        # Find applicable policy
        policy = self.policy_engine.find_policy(path)
        if policy is None:
            # No policy = default allow for Ring 2+
            if clearance.ring <= Ring.APPLICATION:
                decision = AccessDecision(
                    granted=True,
                    reason="No policy defined, default allow for Ring 0-2",
                    clearance=clearance,
                )
                self._emit_bus("ring.access.granted", "info", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "ring": clearance.ring.value,
                    "badge": clearance.badge_color(),
                })
                return decision
            else:
                decision = AccessDecision(
                    granted=False,
                    reason="Ring 3 agents require explicit policy",
                    clearance=clearance,
                )
                self._emit_bus("ring.access.denied", "warn", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "reason": decision.reason,
                })
                return decision

        # Check ring level (like clearance level)
        if clearance.ring > policy.min_ring:
            decision = AccessDecision(
                granted=False,
                reason=f"Insufficient ring: has {clearance.ring.name} ({clearance.ring}), needs {policy.min_ring.name} ({policy.min_ring})",
                policy=policy,
                clearance=clearance,
            )
            self._record_violation(clearance, path, decision.reason)
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
                "policy": policy.path_pattern,
            })
            return decision

        # Check required compartments (AND logic - must have ALL)
        missing_required = set(policy.compartments_required) - clearance.compartments
        if missing_required:
            decision = AccessDecision(
                granted=False,
                reason=f"Missing required compartments: {sorted(missing_required)}",
                policy=policy,
                clearance=clearance,
            )
            self._record_violation(clearance, path, decision.reason)
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
                "missing": list(missing_required),
            })
            return decision

        # Check any compartments (OR logic - must have at least one)
        if policy.compartments_any:
            has_any = bool(set(policy.compartments_any) & clearance.compartments)
            if not has_any:
                decision = AccessDecision(
                    granted=False,
                    reason=f"Need one of compartments: {policy.compartments_any}",
                    policy=policy,
                    clearance=clearance,
                )
                self._record_violation(clearance, path, decision.reason)
                self._emit_bus("ring.access.denied", "warn", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "reason": decision.reason,
                    "needs_one_of": policy.compartments_any,
                })
                return decision

        # Access granted
        decision = AccessDecision(
            granted=True,
            reason="Access granted",
            policy=policy,
            clearance=clearance,
        )
        self._emit_bus("ring.access.granted", "info", {
            "agent_id": agent_id,
            "path": path,
            "operation": operation,
            "ring": clearance.ring.value,
            "badge": clearance.badge_color(),
            "compartments": list(clearance.compartments),
        })
        return decision

    def _record_violation(self, clearance: AgentClearance, path: str, reason: str):
        """Record access violation - may trigger demotion."""
        clearance.violation_count += 1
        self.ledger.set(clearance)

        if clearance.violation_count >= self.max_violations:
            self._emit_bus("ring.violation.threshold", "error", {
                "agent_id": clearance.agent_id,
                "violations": clearance.violation_count,
                "action": "demotion_recommended",
            })

    def record_heartbeat(self, agent_id: str) -> bool:
        """
        Record agent heartbeat to maintain active status.

        Historical parallel:
        - Regular check-ins with security officer
        - Access card swipes at SCIF entrance
        - Continuous monitoring of cleared personnel
        """
        clearance = self.ledger.get(agent_id)
        if clearance is None:
            return False

        clearance.last_heartbeat = time.time()
        self.ledger.set(clearance)

        self._emit_bus(f"ring.heartbeat.ring{clearance.ring}", "info", {
            "agent_id": agent_id,
            "ring": clearance.ring.value,
            "compartments": list(clearance.compartments),
            "badge": clearance.badge_color(),
        })
        return True

    def grant_clearance(
        self,
        agent_id: str,
        ring: Ring,
        compartments: set[str],
        sponsor_id: str,
    ) -> AgentClearance:
        """
        Grant clearance to an agent (read-in ceremony).

        Historical parallel:
        - Security briefing where agent signs NDA
        - Formal "read-in" to SCI program
        - Sponsor must have higher clearance
        """
        # Verify sponsor has authority
        sponsor = self.ledger.get(sponsor_id)
        if sponsor is None or sponsor.ring > ring:
            raise PermissionError(f"Sponsor {sponsor_id} cannot grant Ring {ring.name}")

        # Create NDA signature (hash of agent + sponsor + time)
        nda_content = f"{agent_id}:{sponsor_id}:{time.time()}"
        nda_signature = hashlib.sha256(nda_content.encode()).hexdigest()[:16]

        clearance = AgentClearance(
            agent_id=agent_id,
            ring=ring,
            compartments=compartments,
            issuer_lineage=sponsor_id,
            nda_signature=nda_signature,
            read_in_timestamp=time.time(),
            last_heartbeat=time.time(),
        )

        self.ledger.set(clearance)

        self._emit_bus("ring.compartment.read_in", "info", {
            "agent_id": agent_id,
            "ring": ring.value,
            "compartments": list(compartments),
            "sponsor": sponsor_id,
            "nda_signature": nda_signature,
        })

        return clearance

    def revoke_clearance(self, agent_id: str, reason: str = "") -> bool:
        """
        Revoke agent clearance (read-out).

        Historical parallel:
        - Debriefing when leaving classified program
        - Reminder of ongoing NDA obligations
        - Removal from access lists
        """
        clearance = self.ledger.get(agent_id)
        if clearance is None:
            return False

        self._emit_bus("ring.compartment.read_out", "info", {
            "agent_id": agent_id,
            "former_ring": clearance.ring.value,
            "former_compartments": list(clearance.compartments),
            "reason": reason,
        })

        return self.ledger.revoke(agent_id)

    def demote(self, agent_id: str, new_ring: Ring, reason: str = "") -> bool:
        """
        Demote agent to lower ring (like clearance downgrade).

        Historical parallel:
        - Security incidents leading to clearance review
        - Temporary suspension pending investigation
        - Administrative downgrade
        """
        clearance = self.ledger.get(agent_id)
        if clearance is None:
            return False

        old_ring = clearance.ring
        clearance.ring = new_ring

        # Remove compartments that require higher ring
        for compartment, info in COMPARTMENTS.items():
            if info["min_ring"] < new_ring and compartment in clearance.compartments:
                clearance.compartments.discard(compartment)

        self.ledger.set(clearance)

        self._emit_bus("ring.revocation.complete", "warn", {
            "agent_id": agent_id,
            "old_ring": old_ring.value,
            "new_ring": new_ring.value,
            "reason": reason,
        })

        return True


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Ring Guard - Access Control Enforcement")
    subparsers = parser.add_subparsers(dest="command")

    # Check access
    check_parser = subparsers.add_parser("check", help="Check access to a path")
    check_parser.add_argument("agent_id")
    check_parser.add_argument("path")
    check_parser.add_argument("--operation", default="read")

    # Grant clearance
    grant_parser = subparsers.add_parser("grant", help="Grant clearance to an agent")
    grant_parser.add_argument("agent_id")
    grant_parser.add_argument("--ring", type=int, required=True)
    grant_parser.add_argument("--compartments", nargs="*", default=[])
    grant_parser.add_argument("--sponsor", required=True)

    # Revoke clearance
    revoke_parser = subparsers.add_parser("revoke", help="Revoke clearance")
    revoke_parser.add_argument("agent_id")
    revoke_parser.add_argument("--reason", default="")

    # Heartbeat
    hb_parser = subparsers.add_parser("heartbeat", help="Record heartbeat")
    hb_parser.add_argument("agent_id")

    # List clearances
    list_parser = subparsers.add_parser("list", help="List clearances")
    list_parser.add_argument("--ring", type=int)
    list_parser.add_argument("--compartment")

    # Status
    status_parser = subparsers.add_parser("status", help="Show agent clearance status")
    status_parser.add_argument("agent_id")

    args = parser.parse_args()
    guard = RingGuard()

    if args.command == "check":
        decision = guard.check_access(args.agent_id, args.path, args.operation)
        print(json.dumps({
            "granted": decision.granted,
            "reason": decision.reason,
            "badge": decision.clearance.badge_color() if decision.clearance else None,
        }, indent=2))
        sys.exit(0 if decision.granted else 1)

    elif args.command == "grant":
        try:
            clearance = guard.grant_clearance(
                args.agent_id,
                Ring(args.ring),
                set(args.compartments),
                args.sponsor,
            )
            print(json.dumps(clearance.to_dict(), indent=2))
        except PermissionError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "revoke":
        success = guard.revoke_clearance(args.agent_id, args.reason)
        print("Revoked" if success else "Not found")
        sys.exit(0 if success else 1)

    elif args.command == "heartbeat":
        success = guard.record_heartbeat(args.agent_id)
        print("Recorded" if success else "Agent not found")
        sys.exit(0 if success else 1)

    elif args.command == "list":
        if args.ring is not None:
            clearances = guard.ledger.list_by_ring(Ring(args.ring))
        elif args.compartment:
            clearances = guard.ledger.list_by_compartment(args.compartment)
        else:
            clearances = list(guard.ledger._cache.values())

        for c in clearances:
            print(f"[{c.badge_color()}] {c.agent_id}: Ring {c.ring.name}, Compartments: {sorted(c.compartments)}")

    elif args.command == "status":
        clearance = guard.ledger.get(args.agent_id)
        if clearance:
            print(json.dumps(clearance.to_dict(), indent=2))
        else:
            print("No clearance on file")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
