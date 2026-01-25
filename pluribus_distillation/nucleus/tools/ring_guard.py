#!/usr/bin/env python3
"""
Ring Guard - SCI/SAP-inspired access control enforcement.

Phase 2.3: Ring Security (Steps 96-110)
Implements the compartmentalization model from ring_policies.json

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

Ring Taxonomy:
  0 KERNEL    - Constitutional docs, core security (ceremony required)
  1 OPERATOR  - Infrastructure, bus, dispatch
  2 APPLICATION - Dashboard, UI, tools
  3 EPHEMERAL - PAIP clones, sandbox
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path
from typing import Any, Iterator, Optional, Set

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
    - phi_score = Trust metric (higher = more trusted)
    """
    agent_id: str
    ring: Ring
    compartments: Set[str] = field(default_factory=set)
    issuer_lineage: str = ""
    phi_score: float = 0.70
    ceremonies_completed: list = field(default_factory=list)
    read_in_timestamp: float = 0.0
    last_heartbeat: float = 0.0
    violation_count: int = 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "ring": self.ring.value,
            "ring_name": self.ring.name,
            "compartments": sorted(self.compartments),
            "issuer_lineage": self.issuer_lineage,
            "phi_score": self.phi_score,
            "ceremonies_completed": self.ceremonies_completed,
            "read_in_timestamp": self.read_in_timestamp,
            "last_heartbeat": self.last_heartbeat,
            "violation_count": self.violation_count,
            "badge_color": self.badge_color(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentClearance":
        return cls(
            agent_id=data["agent_id"],
            ring=Ring(data["ring"]),
            compartments=set(data.get("compartments", [])),
            issuer_lineage=data.get("issuer_lineage", ""),
            phi_score=data.get("phi_score", 0.70),
            ceremonies_completed=data.get("ceremonies_completed", []),
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

    def __init__(self, ledger_path: Optional[Path] = None):
        self.ledger_path = ledger_path or Path(REPO_ROOT / ".pluribus/state/clearance_ledger.json")
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
            "updated_iso": datetime.now(timezone.utc).isoformat(),
            "clearances": {
                agent_id: clearance.to_dict()
                for agent_id, clearance in self._cache.items()
            },
        }
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
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

    def all(self) -> list[AgentClearance]:
        return list(self._cache.values())


# =============================================================================
# ACCESS DECISION (Result of access check)
# =============================================================================

@dataclass
class AccessDecision:
    """
    Result of an access check.

    Supports both:
    - Object interface: decision.granted, decision.reason
    - Tuple unpacking: allowed, reason = guard.check_access(...)

    In observe mode:
    - granted = would the access be granted under policy?
    - allowed = is the access actually permitted (always True in observe mode)
    - enforced = was the policy actually enforced?
    """
    granted: bool
    reason: str
    policy: Optional[dict] = None
    clearance: Optional[AgentClearance] = None
    allowed: bool = True  # In observe mode, access is always allowed
    enforced: bool = True  # False in observe mode

    def __iter__(self) -> Iterator:
        """Support tuple unpacking: allowed, reason = decision"""
        return iter((self.allowed, self.reason))

    def __bool__(self) -> bool:
        """Boolean context uses 'allowed' for backward compatibility."""
        return self.allowed


# =============================================================================
# RING GUARD (Main Enforcement)
# =============================================================================

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
        self.ledger = ClearanceLedger(ledger_path)
        self.policies_path = policies_path or Path(REPO_ROOT / "nucleus/specs/ring_policies.json")
        self.policies = self._load_policies()
        self.bus_dir = bus_dir or str(REPO_ROOT / ".pluribus/bus")

        # Settings from policy
        settings = self.policies.get("settings", {})
        self.enforcement_mode = settings.get("enforcement_mode", "enforce")  # enforce | observe
        self.auto_register_ring = settings.get("auto_register_unknown_ring", None)
        self.enforce_heartbeat = settings.get("enforce_heartbeat", True)
        self.enforce_ceremony = settings.get("enforce_ceremony", True)
        self.enforce_access_matrix = settings.get("enforce_access_matrix", False)
        self.heartbeat_timeout_s = settings.get("heartbeat_timeout_s", 300)
        self.max_violations = settings.get("max_violations", 3)

        # Auto-register default clearances from policy
        self._register_default_clearances()

    def _load_policies(self) -> dict:
        """Load ring policies from JSON."""
        if self.policies_path and self.policies_path.exists():
            try:
                with self.policies_path.open("r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"rings": {}, "policies": [], "settings": {}}

    def _register_default_clearances(self):
        """Auto-register default clearances from policy file."""
        defaults = self.policies.get("default_clearances", {})
        for agent_id, config in defaults.items():
            if self.ledger.get(agent_id) is None:
                clearance = AgentClearance(
                    agent_id=agent_id,
                    ring=Ring(config.get("ring", 2)),
                    compartments=set(config.get("compartments", [])),
                    issuer_lineage=config.get("issuer_lineage", "default_policy"),
                    phi_score=config.get("phi_score", 0.85),
                    ceremonies_completed=config.get("ceremonies_completed", []),
                    read_in_timestamp=time.time(),
                    last_heartbeat=time.time(),
                )
                self.ledger.set(clearance)

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

    def _match_path(self, pattern: str, path: str) -> bool:
        """Simple glob-style matching."""
        return fnmatch.fnmatch(path, pattern)

    def _find_policy(self, path: str) -> Optional[dict]:
        """Find the most specific policy matching a path."""
        policies = self.policies.get("policies", [])
        matches = [p for p in policies if self._match_path(p.get("path", ""), path)]
        if not matches:
            return None
        # Return most specific (longest pattern)
        return max(matches, key=lambda p: len(p.get("path", "")))

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

        Returns AccessDecision that supports both:
        - Object interface: decision.granted, decision.reason
        - Tuple unpacking: allowed, reason = guard.check_access(...)
        """
        is_observe_mode = self.enforcement_mode == "observe"

        # Get agent clearance
        clearance = self.ledger.get(agent_id)
        if clearance is None:
            # Auto-register if configured
            if self.auto_register_ring is not None:
                clearance = AgentClearance(
                    agent_id=agent_id,
                    ring=Ring(self.auto_register_ring),
                    phi_score=0.70,
                    read_in_timestamp=time.time(),
                    last_heartbeat=time.time(),
                )
                self.ledger.set(clearance)
            else:
                decision = AccessDecision(
                    granted=False,
                    reason=f"Agent {agent_id} has no clearance on file",
                    allowed=is_observe_mode,
                    enforced=not is_observe_mode,
                )
                self._emit_bus("ring.access.denied", "warn", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "reason": decision.reason,
                    "observe_mode": is_observe_mode,
                })
                return decision

        # Check heartbeat liveness if enforced
        if self.enforce_heartbeat:
            now = time.time()
            if now - clearance.last_heartbeat > self.heartbeat_timeout_s:
                decision = AccessDecision(
                    granted=False,
                    reason=f"Agent {agent_id} heartbeat expired ({self.heartbeat_timeout_s}s timeout)",
                    clearance=clearance,
                    allowed=is_observe_mode,
                    enforced=not is_observe_mode,
                )
                self._emit_bus("ring.access.denied", "warn", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "reason": decision.reason,
                    "last_heartbeat": clearance.last_heartbeat,
                    "observe_mode": is_observe_mode,
                })
                return decision

        # Find applicable policy
        policy = self._find_policy(path)
        if policy is None:
            # No policy = default allow for Ring 0-2
            if clearance.ring <= Ring.APPLICATION:
                decision = AccessDecision(
                    granted=True,
                    reason="No policy defined, default allow for Ring 0-2",
                    clearance=clearance,
                    allowed=True,
                    enforced=not is_observe_mode,
                )
                self._emit_bus("ring.access.granted", "info", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "ring": clearance.ring.value,
                    "badge": clearance.badge_color(),
                    "observe_mode": is_observe_mode,
                })
                return decision
            else:
                decision = AccessDecision(
                    granted=False,
                    reason="Ring 3 agents require explicit policy",
                    clearance=clearance,
                    allowed=is_observe_mode,
                    enforced=not is_observe_mode,
                )
                self._emit_bus("ring.access.denied", "warn", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "reason": decision.reason,
                    "observe_mode": is_observe_mode,
                })
                return decision

        min_ring = policy.get("min_ring", 2)
        required_compartments = set(policy.get("compartments_required", []))
        required_ceremony = None

        # Get ring config for ceremony requirement
        ring_config = self.policies.get("rings", {}).get(str(min_ring), {})
        if self.enforce_ceremony:
            required_ceremony = ring_config.get("required_ceremony")

        # Check phi-score threshold
        min_phi = ring_config.get("min_phi_score", 0.0)
        if clearance.phi_score < min_phi:
            decision = AccessDecision(
                granted=False,
                reason=f"Phi-score {clearance.phi_score:.2f} below threshold {min_phi:.2f}",
                policy=policy,
                clearance=clearance,
                allowed=is_observe_mode,
                enforced=not is_observe_mode,
            )
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
                "observe_mode": is_observe_mode,
            })
            return decision

        # Check ring level
        if clearance.ring > Ring(min_ring):
            decision = AccessDecision(
                granted=False,
                reason=f"Insufficient ring: has {clearance.ring.name} ({clearance.ring}), needs Ring {min_ring}",
                policy=policy,
                clearance=clearance,
                allowed=is_observe_mode,
                enforced=not is_observe_mode,
            )
            self._record_violation(clearance, path, decision.reason)
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
                "policy": policy.get("path"),
                "observe_mode": is_observe_mode,
            })
            return decision

        # Check required compartments (AND logic - must have ALL)
        if required_compartments:
            missing_required = required_compartments - clearance.compartments
            if missing_required:
                decision = AccessDecision(
                    granted=False,
                    reason=f"Missing required compartments: {sorted(missing_required)}",
                    policy=policy,
                    clearance=clearance,
                    allowed=is_observe_mode,
                    enforced=not is_observe_mode,
                )
                self._record_violation(clearance, path, decision.reason)
                self._emit_bus("ring.access.denied", "warn", {
                    "agent_id": agent_id,
                    "path": path,
                    "operation": operation,
                    "reason": decision.reason,
                    "missing": list(missing_required),
                    "observe_mode": is_observe_mode,
                })
                return decision

        # Check ceremony requirement
        if required_ceremony and required_ceremony not in clearance.ceremonies_completed:
            decision = AccessDecision(
                granted=False,
                reason=f"Ceremony '{required_ceremony}' not completed",
                policy=policy,
                clearance=clearance,
                allowed=is_observe_mode,
                enforced=not is_observe_mode,
            )
            self._emit_bus("ring.access.denied", "warn", {
                "agent_id": agent_id,
                "path": path,
                "operation": operation,
                "reason": decision.reason,
                "observe_mode": is_observe_mode,
            })
            return decision

        # Access granted
        decision = AccessDecision(
            granted=True,
            reason="Access granted",
            policy=policy,
            clearance=clearance,
            allowed=True,
            enforced=not is_observe_mode,
        )
        self._emit_bus("ring.access.granted", "info", {
            "agent_id": agent_id,
            "path": path,
            "operation": operation,
            "ring": clearance.ring.value,
            "badge": clearance.badge_color(),
            "compartments": list(clearance.compartments),
            "observe_mode": is_observe_mode,
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
        compartments: Set[str],
        sponsor_id: str,
        phi_score: float = 0.85,
    ) -> AgentClearance:
        """
        Grant clearance to an agent (read-in ceremony).

        Historical parallel:
        - Security briefing where agent signs NDA
        - Formal "read-in" to SCI program
        """
        clearance = AgentClearance(
            agent_id=agent_id,
            ring=ring,
            compartments=compartments,
            issuer_lineage=sponsor_id,
            phi_score=phi_score,
            ceremonies_completed=["read_in_ceremony"] if ring <= Ring.OPERATOR else [],
            read_in_timestamp=time.time(),
            last_heartbeat=time.time(),
        )

        self.ledger.set(clearance)

        self._emit_bus("ring.compartment.read_in", "info", {
            "agent_id": agent_id,
            "ring": ring.value,
            "compartments": list(compartments),
            "sponsor": sponsor_id,
        })

        return clearance

    def revoke_clearance(self, agent_id: str, reason: str = "") -> bool:
        """
        Revoke agent clearance (read-out).

        Historical parallel:
        - Debriefing when leaving classified program
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

    def get_status(self) -> dict:
        """Get current ring guard status for dashboard."""
        clearances = self.ledger.all()
        return {
            "enforcement_mode": self.enforcement_mode,
            "total_agents": len(clearances),
            "by_ring": {
                "kernel": len([c for c in clearances if c.ring == Ring.KERNEL]),
                "operator": len([c for c in clearances if c.ring == Ring.OPERATOR]),
                "application": len([c for c in clearances if c.ring == Ring.APPLICATION]),
                "ephemeral": len([c for c in clearances if c.ring == Ring.EPHEMERAL]),
            },
            "compartments_active": list(set().union(*[c.compartments for c in clearances])),
            "violations_total": sum(c.violation_count for c in clearances),
        }


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
    grant_parser.add_argument("--phi", type=float, default=0.85)

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
    status_parser = subparsers.add_parser("status", help="Show ring guard status")

    args = parser.parse_args()
    guard = RingGuard()

    if args.command == "check":
        decision = guard.check_access(args.agent_id, args.path, args.operation)
        print(json.dumps({
            "granted": decision.granted,
            "allowed": decision.allowed,
            "enforced": decision.enforced,
            "reason": decision.reason,
            "badge": decision.clearance.badge_color() if decision.clearance else None,
        }, indent=2))
        sys.exit(0 if decision.allowed else 1)

    elif args.command == "grant":
        clearance = guard.grant_clearance(
            args.agent_id,
            Ring(args.ring),
            set(args.compartments),
            args.sponsor,
            args.phi,
        )
        print(json.dumps(clearance.to_dict(), indent=2))

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
            clearances = guard.ledger.all()

        for c in clearances:
            print(f"[{c.badge_color()}] {c.agent_id}: Ring {c.ring.name}, Compartments: {sorted(c.compartments)}, Phi: {c.phi_score:.2f}")

    elif args.command == "status":
        status = guard.get_status()
        print(json.dumps(status, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
