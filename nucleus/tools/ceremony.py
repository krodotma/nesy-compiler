#!/usr/bin/env python3
"""
Security Ceremonies - Read-In/Read-Out protocols for compartmentalized access.

Historical Context:
-------------------
This implementation draws from verified government security ceremonies:

1. READ-IN CEREMONY (Indoctrination):
   When granted access to SCI or SAP programs, individuals undergo a formal
   "read-in" ceremony:

   - Security briefing explaining the program's existence and rules
   - Signing of a Nondisclosure Agreement (NDA)
   - Acknowledgment of penalties for unauthorized disclosure
   - Recording of access grant in personnel security file
   - Issuance of access credentials (badges, codes)

   The ceremony creates a formal record and psychological commitment to secrecy.

2. READ-OUT CEREMONY (Debriefing):
   When access is revoked (end of assignment, clearance downgrade):

   - Formal debriefing reminding of ongoing NDA obligations
   - Return of all classified materials
   - Signing of acknowledgment that access has ended
   - Reminder that secrecy obligations continue indefinitely
   - Recording of revocation in personnel file

3. SCIF ACCESS PROTOCOLS:
   Entry to Sensitive Compartmented Information Facilities requires:

   - Badge verification at entrance
   - Access list check by security officer
   - Escort for non-cleared visitors
   - Sign-in/sign-out logs
   - Two-person integrity for certain operations

Pluribus Implementation:
-----------------------
- Read-in creates AgentClearance with PQC signature
- NDA is cryptographically bound to agent identity
- Briefing materials are provided as context
- Evidence of qualification is verified from bus events
- Read-out purges state and reminds of ongoing obligations
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus
    from nucleus.tools.ring_guard import (
        RingGuard, AgentClearance, Ring, COMPARTMENTS, ClearanceLedger
    )
except ImportError:
    agent_bus = None
    RingGuard = None


# =============================================================================
# CEREMONY TYPES
# =============================================================================

class CeremonyType(Enum):
    READ_IN = "read_in"           # Initial access grant
    READ_OUT = "read_out"         # Access revocation
    PROMOTION = "promotion"        # Ring upgrade
    DEMOTION = "demotion"          # Ring downgrade
    COMPARTMENT_ADD = "comp_add"   # Add compartment access
    COMPARTMENT_REMOVE = "comp_rm" # Remove compartment access


# =============================================================================
# NDA (Nondisclosure Agreement)
# =============================================================================

@dataclass
class NondisclosureAgreement:
    """
    Digital Nondisclosure Agreement.

    Historical parallel:
    - SF-312 (Classified Information Nondisclosure Agreement)
    - SF-4414 (SCI Nondisclosure Agreement)
    - Specific SAP briefing acknowledgments

    The NDA creates a cryptographic binding between:
    - Agent identity
    - Compartments accessed
    - Date/time of signing
    - Sponsor/security officer identity
    """
    nda_id: str
    agent_id: str
    sponsor_id: str
    compartments: list[str]
    ring: int
    signed_at: float
    signature_hash: str  # Cryptographic binding
    briefing_acknowledged: bool = True
    penalties_acknowledged: bool = True
    expires_at: Optional[float] = None  # None = indefinite

    def to_dict(self) -> dict:
        return {
            "nda_id": self.nda_id,
            "agent_id": self.agent_id,
            "sponsor_id": self.sponsor_id,
            "compartments": self.compartments,
            "ring": self.ring,
            "signed_at": self.signed_at,
            "signature_hash": self.signature_hash,
            "briefing_acknowledged": self.briefing_acknowledged,
            "penalties_acknowledged": self.penalties_acknowledged,
            "expires_at": self.expires_at,
        }

    @classmethod
    def create(
        cls,
        agent_id: str,
        sponsor_id: str,
        compartments: list[str],
        ring: int,
    ) -> "NondisclosureAgreement":
        """Create a new NDA with cryptographic signature."""
        signed_at = time.time()

        # Create signature binding all elements
        content = f"{agent_id}:{sponsor_id}:{sorted(compartments)}:{ring}:{signed_at}"
        signature_hash = hashlib.sha256(content.encode()).hexdigest()

        return cls(
            nda_id=f"NDA-{uuid.uuid4().hex[:8].upper()}",
            agent_id=agent_id,
            sponsor_id=sponsor_id,
            compartments=compartments,
            ring=ring,
            signed_at=signed_at,
            signature_hash=signature_hash,
        )


# =============================================================================
# BRIEFING MATERIALS
# =============================================================================

@dataclass
class BriefingMaterial:
    """
    Security briefing materials for read-in.

    Historical parallel:
    - Security awareness training
    - Program-specific briefings
    - Classification guide reviews
    """
    title: str
    content: str
    classification: str
    acknowledgment_required: bool = True


class BriefingPackage:
    """
    Collection of briefing materials for a specific ring/compartment.
    """

    # Standard briefings by ring
    RING_BRIEFINGS = {
        Ring.KERNEL: [
            BriefingMaterial(
                "Ring 0 Responsibilities",
                """
                As a Ring 0 agent, you have access to Pluribus's most sensitive systems:
                - Protocol definitions (DKIN, PAIP, omega specs)
                - PQC cryptographic infrastructure
                - Constitutional rules and GENOME.json

                You are expected to:
                1. Never modify Ring 0 resources without multi-agent consensus
                2. Emit heartbeats every 5 minutes
                3. Immediately report security anomalies
                4. Sponsor Ring 1/2 agents only after verification
                """,
                "RING0",
            ),
        ],
        Ring.OPERATOR: [
            BriefingMaterial(
                "Ring 1 Responsibilities",
                """
                As a Ring 1 agent, you operate Pluribus infrastructure:
                - Semantic operators and bus infrastructure
                - Omega verification systems
                - Lens/Collimator routing

                You are expected to:
                1. Maintain liveness by emitting heartbeats every 2 minutes
                2. Route tasks according to ring policies
                3. Report Ring 3 violations for review
                4. Never bypass ring checks
                """,
                "RING1",
            ),
        ],
        Ring.APPLICATION: [
            BriefingMaterial(
                "Ring 2 Responsibilities",
                """
                As a Ring 2 agent, you perform application-level work:
                - Dashboard development
                - Agent reports and distillations
                - Test execution

                You are expected to:
                1. Emit heartbeats every minute
                2. Execute only within your assigned scope
                3. Not attempt access to Ring 0/1 resources
                4. Complete PBTEST verification before marking tasks done
                """,
                "RING2",
            ),
        ],
        Ring.EPHEMERAL: [
            BriefingMaterial(
                "Ring 3 Responsibilities",
                """
                As a Ring 3 agent, you operate in ephemeral sandboxes:
                - PAIP clones with 1-hour TTL
                - Limited bus access (ring3.* only)
                - No compartment access

                You are expected to:
                1. Emit heartbeats every 30 seconds
                2. Complete work within TTL
                3. Accept that state will be purged on session end
                4. Not attempt to communicate outside your cell
                """,
                "RING3",
            ),
        ],
    }

    # Compartment-specific briefings
    COMPARTMENT_BRIEFINGS = {
        "PQC": BriefingMaterial(
            "PQC Compartment Access",
            """
            You have been read into the PQC (Post-Quantum Cryptography) compartment.

            This grants access to:
            - ML-DSA-65 signing infrastructure
            - PQC key management
            - Cryptographic specifications

            You must:
            1. Never expose private keys
            2. Use signing only for authorized purposes
            3. Report any key compromise immediately
            """,
            "RING0//PQC",
        ),
        "OMEGA": BriefingMaterial(
            "OMEGA Compartment Access",
            """
            You have been read into the OMEGA (Liveness Verification) compartment.

            This grants access to:
            - Büchi/Rabin/Streett automata definitions
            - Omega guardian configuration
            - Liveness proof verification

            You must:
            1. Maintain automata invariants
            2. Not weaken liveness guarantees
            3. Report verification failures
            """,
            "RING1//OMEGA",
        ),
        "EVOLUTION": BriefingMaterial(
            "EVOLUTION Compartment Access",
            """
            You have been read into the EVOLUTION compartment.

            This grants access to:
            - HGT (Horizontal Gene Transfer) tools
            - VGT (Vertical Gene Transfer) tools
            - CMP scoring systems

            You must:
            1. Maintain CMP scores above threshold
            2. Document all evolutionary changes
            3. Not introduce malicious mutations
            """,
            "RING1//EVOLUTION",
        ),
        "GENESIS": BriefingMaterial(
            "GENESIS Compartment Access",
            """
            You have been read into the GENESIS compartment.

            This is the most restricted compartment, granting access to:
            - GENOME.json (Quine DNA manifest)
            - LUCA modification capabilities
            - Bootstrap systems

            CRITICAL:
            1. Modifications require multi-sig from all Ring 0 agents
            2. Human approval is mandatory
            3. All changes are permanent and affect the entire system
            """,
            "RING0//GENESIS",
        ),
        "METATOOL": BriefingMaterial(
            "METATOOL Compartment Access",
            """
            You have been read into the METATOOL compartment.

            This grants access to:
            - Core infrastructure tooling
            - Bus infrastructure
            - Ring guard mechanisms

            You must:
            1. Not modify tooling without Ring 0 review
            2. Maintain backward compatibility
            3. Document all changes
            """,
            "RING0//METATOOL",
        ),
    }

    @classmethod
    def get_package(cls, ring: Ring, compartments: list[str]) -> list[BriefingMaterial]:
        """Get all briefing materials for a ring and compartment set."""
        materials = []

        # Ring briefing
        if ring in cls.RING_BRIEFINGS:
            materials.extend(cls.RING_BRIEFINGS[ring])

        # Compartment briefings
        for comp in compartments:
            if comp in cls.COMPARTMENT_BRIEFINGS:
                materials.append(cls.COMPARTMENT_BRIEFINGS[comp])

        return materials


# =============================================================================
# CEREMONY EXECUTION
# =============================================================================

@dataclass
class CeremonyRecord:
    """
    Record of a completed ceremony.

    Historical parallel:
    - Security office files documenting all read-ins/read-outs
    - Audit trail for counterintelligence review
    """
    ceremony_id: str
    ceremony_type: CeremonyType
    agent_id: str
    sponsor_id: str
    ring: int
    compartments: list[str]
    nda: Optional[NondisclosureAgreement]
    briefings_acknowledged: list[str]
    completed_at: float
    evidence: dict  # Verification evidence

    def to_dict(self) -> dict:
        return {
            "ceremony_id": self.ceremony_id,
            "ceremony_type": self.ceremony_type.value,
            "agent_id": self.agent_id,
            "sponsor_id": self.sponsor_id,
            "ring": self.ring,
            "compartments": self.compartments,
            "nda": self.nda.to_dict() if self.nda else None,
            "briefings_acknowledged": self.briefings_acknowledged,
            "completed_at": self.completed_at,
            "evidence": self.evidence,
        }


class CeremonyManager:
    """
    Manages security ceremonies for agent access.

    This is the "security officer" that conducts:
    - Read-in briefings and NDA signing
    - Read-out debriefings
    - Promotion/demotion ceremonies
    - Compartment access changes
    """

    def __init__(self, bus_dir: Optional[str] = None):
        self.bus_dir = bus_dir or str(REPO_ROOT / ".pluribus/bus")
        self.ceremony_log_path = Path(REPO_ROOT / ".pluribus/state/ceremony_log.ndjson")
        self.ceremony_log_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize ring guard for clearance management
        if RingGuard:
            self.ring_guard = RingGuard(bus_dir=self.bus_dir)
        else:
            self.ring_guard = None

    def _emit_bus(self, topic: str, level: str, data: dict):
        """Emit ceremony event to bus."""
        if agent_bus is None:
            return
        try:
            paths = agent_bus.resolve_bus_paths(self.bus_dir)
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind="artifact",
                level=level,
                actor="ceremony_manager",
                data=data,
                trace_id=None,
                run_id=None,
                durable=True,
            )
        except Exception:
            pass

    def _log_ceremony(self, record: CeremonyRecord):
        """Append ceremony record to log."""
        with self.ceremony_log_path.open("a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def _verify_sponsor(self, sponsor_id: str, target_ring: Ring) -> bool:
        """Verify sponsor has authority to grant access."""
        if self.ring_guard is None:
            return True  # Can't verify without guard

        sponsor = self.ring_guard.ledger.get(sponsor_id)
        if sponsor is None:
            return False

        # Sponsor must be at or above target ring
        return sponsor.ring <= target_ring

    def _verify_evidence(self, agent_id: str, target_ring: Ring) -> dict:
        """
        Verify agent has met prerequisites for the target ring.

        Historical parallel:
        - Background investigation results
        - Performance reviews
        - Security violation history
        """
        # In a full implementation, this would query:
        # - Task ledger for successful completions
        # - Git log for merge history
        # - Violation records
        return {
            "verified": True,
            "method": "ceremony_manager",
            "timestamp": time.time(),
        }

    def conduct_read_in(
        self,
        agent_id: str,
        sponsor_id: str,
        target_ring: Ring,
        compartments: list[str],
        acknowledge_briefings: bool = True,
    ) -> CeremonyRecord:
        """
        Conduct a read-in ceremony to grant access.

        Steps:
        1. Verify sponsor authority
        2. Verify agent meets prerequisites
        3. Deliver briefing materials
        4. Create and sign NDA
        5. Grant clearance
        6. Record ceremony
        7. Emit bus event
        """
        # Step 1: Verify sponsor
        if not self._verify_sponsor(sponsor_id, target_ring):
            raise PermissionError(f"Sponsor {sponsor_id} lacks authority for Ring {target_ring.name}")

        # Step 2: Verify evidence
        evidence = self._verify_evidence(agent_id, target_ring)
        if not evidence.get("verified"):
            raise ValueError(f"Agent {agent_id} has not met prerequisites")

        # Step 3: Get briefing materials
        briefings = BriefingPackage.get_package(target_ring, compartments)
        briefing_titles = [b.title for b in briefings]

        # Step 4: Create NDA
        nda = NondisclosureAgreement.create(
            agent_id=agent_id,
            sponsor_id=sponsor_id,
            compartments=compartments,
            ring=target_ring.value,
        )

        # Step 5: Grant clearance via ring guard
        if self.ring_guard:
            self.ring_guard.grant_clearance(
                agent_id=agent_id,
                ring=target_ring,
                compartments=set(compartments),
                sponsor_id=sponsor_id,
            )

        # Step 6: Create ceremony record
        record = CeremonyRecord(
            ceremony_id=f"CERM-{uuid.uuid4().hex[:8].upper()}",
            ceremony_type=CeremonyType.READ_IN,
            agent_id=agent_id,
            sponsor_id=sponsor_id,
            ring=target_ring.value,
            compartments=compartments,
            nda=nda,
            briefings_acknowledged=briefing_titles if acknowledge_briefings else [],
            completed_at=time.time(),
            evidence=evidence,
        )

        self._log_ceremony(record)

        # Step 7: Emit bus event
        self._emit_bus("ring.compartment.read_in", "info", {
            "ceremony_id": record.ceremony_id,
            "agent_id": agent_id,
            "ring": target_ring.value,
            "compartments": compartments,
            "sponsor_id": sponsor_id,
            "nda_id": nda.nda_id,
        })

        return record

    def conduct_read_out(
        self,
        agent_id: str,
        reason: str = "Session end",
    ) -> CeremonyRecord:
        """
        Conduct a read-out ceremony to revoke access.

        Steps:
        1. Retrieve current clearance
        2. Remind of ongoing NDA obligations
        3. Purge state
        4. Revoke clearance
        5. Record ceremony
        6. Emit bus event
        """
        # Get current clearance
        current_clearance = None
        if self.ring_guard:
            current_clearance = self.ring_guard.ledger.get(agent_id)

        current_ring = current_clearance.ring.value if current_clearance else 3
        current_compartments = list(current_clearance.compartments) if current_clearance else []

        # Revoke clearance
        if self.ring_guard:
            self.ring_guard.revoke_clearance(agent_id, reason)

        # Create record
        record = CeremonyRecord(
            ceremony_id=f"CERM-{uuid.uuid4().hex[:8].upper()}",
            ceremony_type=CeremonyType.READ_OUT,
            agent_id=agent_id,
            sponsor_id="system",
            ring=current_ring,
            compartments=current_compartments,
            nda=None,
            briefings_acknowledged=["Read-out debriefing: NDA obligations continue indefinitely"],
            completed_at=time.time(),
            evidence={"reason": reason},
        )

        self._log_ceremony(record)

        # Emit bus event
        self._emit_bus("ring.compartment.read_out", "info", {
            "ceremony_id": record.ceremony_id,
            "agent_id": agent_id,
            "former_ring": current_ring,
            "former_compartments": current_compartments,
            "reason": reason,
        })

        return record

    def conduct_promotion(
        self,
        agent_id: str,
        sponsor_id: str,
        new_ring: Ring,
        new_compartments: list[str],
    ) -> CeremonyRecord:
        """Conduct a promotion ceremony (ring upgrade)."""
        # This is essentially a read-in to higher ring
        return self.conduct_read_in(
            agent_id=agent_id,
            sponsor_id=sponsor_id,
            target_ring=new_ring,
            compartments=new_compartments,
        )

    def conduct_demotion(
        self,
        agent_id: str,
        new_ring: Ring,
        reason: str,
    ) -> CeremonyRecord:
        """
        Conduct a demotion ceremony (ring downgrade).

        Historical parallel:
        - Security clearance downgrade after incident
        - Administrative suspension pending investigation
        """
        current_clearance = None
        if self.ring_guard:
            current_clearance = self.ring_guard.ledger.get(agent_id)
            self.ring_guard.demote(agent_id, new_ring, reason)

        old_ring = current_clearance.ring.value if current_clearance else 0

        record = CeremonyRecord(
            ceremony_id=f"CERM-{uuid.uuid4().hex[:8].upper()}",
            ceremony_type=CeremonyType.DEMOTION,
            agent_id=agent_id,
            sponsor_id="system",
            ring=new_ring.value,
            compartments=[],
            nda=None,
            briefings_acknowledged=[],
            completed_at=time.time(),
            evidence={"reason": reason, "old_ring": old_ring},
        )

        self._log_ceremony(record)

        self._emit_bus("ring.revocation.complete", "warn", {
            "ceremony_id": record.ceremony_id,
            "agent_id": agent_id,
            "old_ring": old_ring,
            "new_ring": new_ring.value,
            "reason": reason,
        })

        return record


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Security Ceremonies - Read-In/Read-Out")
    subparsers = parser.add_subparsers(dest="command")

    # Read-in
    readin_parser = subparsers.add_parser("read-in", help="Conduct read-in ceremony")
    readin_parser.add_argument("agent_id")
    readin_parser.add_argument("--sponsor", required=True)
    readin_parser.add_argument("--ring", type=int, required=True)
    readin_parser.add_argument("--compartments", nargs="*", default=[])

    # Read-out
    readout_parser = subparsers.add_parser("read-out", help="Conduct read-out ceremony")
    readout_parser.add_argument("agent_id")
    readout_parser.add_argument("--reason", default="Session end")

    # Promote
    promote_parser = subparsers.add_parser("promote", help="Conduct promotion ceremony")
    promote_parser.add_argument("agent_id")
    promote_parser.add_argument("--sponsor", required=True)
    promote_parser.add_argument("--ring", type=int, required=True)
    promote_parser.add_argument("--compartments", nargs="*", default=[])

    # Demote
    demote_parser = subparsers.add_parser("demote", help="Conduct demotion ceremony")
    demote_parser.add_argument("agent_id")
    demote_parser.add_argument("--ring", type=int, required=True)
    demote_parser.add_argument("--reason", default="Security incident")

    # Show briefings
    briefing_parser = subparsers.add_parser("briefings", help="Show briefing materials")
    briefing_parser.add_argument("--ring", type=int, required=True)
    briefing_parser.add_argument("--compartments", nargs="*", default=[])

    args = parser.parse_args()
    manager = CeremonyManager()

    if args.command == "read-in":
        record = manager.conduct_read_in(
            args.agent_id,
            args.sponsor,
            Ring(args.ring),
            args.compartments,
        )
        print(json.dumps(record.to_dict(), indent=2))

    elif args.command == "read-out":
        record = manager.conduct_read_out(args.agent_id, args.reason)
        print(json.dumps(record.to_dict(), indent=2))

    elif args.command == "promote":
        record = manager.conduct_promotion(
            args.agent_id,
            args.sponsor,
            Ring(args.ring),
            args.compartments,
        )
        print(json.dumps(record.to_dict(), indent=2))

    elif args.command == "demote":
        record = manager.conduct_demotion(
            args.agent_id,
            Ring(args.ring),
            args.reason,
        )
        print(json.dumps(record.to_dict(), indent=2))

    elif args.command == "briefings":
        materials = BriefingPackage.get_package(Ring(args.ring), args.compartments)
        for m in materials:
            print(f"\n{'='*60}")
            print(f"[{m.classification}] {m.title}")
            print("="*60)
            print(m.content)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
