#!/usr/bin/env python3
"""
witness.py - Witness Pattern Implementation (VOR Replacement)

Every mutation must have a Witness.
Witnesses produce Attestations.
Attestations are evidence of Entelecheia.

Ring: 1 (Operator)
Protocol: DKIN v28 | PAIP v16 | Citizen v1
"""

import json
import time
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class WitnessType(Enum):
    """Types of witnesses in the Pluribus DNA."""
    VERIFICATION = "verification"   # Saw the action succeed/fail
    OBSERVATION = "observation"     # Can report what happened
    REPRODUCTION = "reproduction"   # Can repeat the action


@dataclass
class Attestation:
    """
    Evidence produced by a Witness.
    
    An attestation is the cryptographic/semantic record that
    a witness observed an action and can testify to its outcome.
    """
    attestation_id: str
    witness_id: str
    witness_type: str
    subject: str  # What was witnessed (commit, episode, handoff, etc.)
    subject_hash: str  # SHA256 of the subject content
    verdict: str  # "confirmed" | "denied" | "uncertain"
    evidence: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    entelecheia_delta: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attestation_id": self.attestation_id,
            "witness_id": self.witness_id,
            "witness_type": self.witness_type,
            "subject": self.subject,
            "subject_hash": self.subject_hash,
            "verdict": self.verdict,
            "evidence": self.evidence,
            "timestamp": self.timestamp,
            "entelecheia_delta": self.entelecheia_delta
        }
    
    def signature(self) -> str:
        """Generate a verification signature for this attestation."""
        content = json.dumps({
            "witness_id": self.witness_id,
            "subject": self.subject,
            "subject_hash": self.subject_hash,
            "verdict": self.verdict,
            "timestamp": self.timestamp
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class Witness:
    """
    A Witness observes mutations and produces Attestations.
    
    Replaces the VOR (Verification, Observability, Reproducibility) pattern
    with a more meaningful metaphor: Witnesses testify to entelecheia.
    """
    
    def __init__(self, 
                 witness_id: str,
                 witness_type: WitnessType = WitnessType.OBSERVATION,
                 bus_path: Optional[Path] = None,
                 ledger_path: Optional[Path] = None):
        self.witness_id = witness_id
        self.witness_type = witness_type
        self.bus_path = bus_path or Path(".pluribus/bus/events.ndjson")
        self.ledger_path = ledger_path or Path(".pluribus/dkin/attestation_ledger.ndjson")
        self.attestations: List[Attestation] = []
    
    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info"):
        """Emit event to bus."""
        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "attestation",
            "level": level,
            "actor": self.witness_id,
            "data": data
        }
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.bus_path, 'a') as f:
            f.write(json.dumps(event) + '\n')
    
    def _append_ledger(self, attestation: Attestation):
        """Append attestation to persistent ledger."""
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, 'a') as f:
            f.write(json.dumps(attestation.to_dict()) + '\n')
        self.attestations.append(attestation)
    
    def attest(self,
               subject: str,
               subject_content: Any,
               verdict: str,
               evidence: Dict[str, Any] = None,
               entelecheia_delta: Dict[str, float] = None) -> Attestation:
        """
        Produce an attestation for a witnessed subject.
        
        Args:
            subject: Identifier of what was witnessed
            subject_content: The actual content (for hashing)
            verdict: "confirmed" | "denied" | "uncertain"
            evidence: Supporting evidence dict
            entelecheia_delta: Optional telos/coherence/resonance changes
        
        Returns:
            The produced Attestation
        """
        subject_str = json.dumps(subject_content, sort_keys=True, default=str)
        subject_hash = hashlib.sha256(subject_str.encode()).hexdigest()
        
        attestation = Attestation(
            attestation_id=f"ATT-{uuid.uuid4().hex[:8].upper()}",
            witness_id=self.witness_id,
            witness_type=self.witness_type.value,
            subject=subject,
            subject_hash=subject_hash,
            verdict=verdict,
            evidence=evidence or {},
            entelecheia_delta=entelecheia_delta
        )
        
        self._append_ledger(attestation)
        self._emit_bus_event("witness.attestation.produced", {
            "attestation_id": attestation.attestation_id,
            "subject": subject,
            "verdict": verdict,
            "signature": attestation.signature()
        })
        
        return attestation
    
    def verify(self, subject: str, subject_content: Any, 
               check_fn: callable) -> Attestation:
        """
        Witness a verification action.
        
        Args:
            subject: What is being verified
            subject_content: Content to verify
            check_fn: Function that returns (success: bool, evidence: dict)
        
        Returns:
            Attestation with verdict based on check_fn result
        """
        try:
            success, evidence = check_fn(subject_content)
            verdict = "confirmed" if success else "denied"
        except Exception as e:
            verdict = "uncertain"
            evidence = {"error": str(e)}
        
        return self.attest(
            subject=subject,
            subject_content=subject_content,
            verdict=verdict,
            evidence=evidence
        )
    
    def observe(self, subject: str, observation: Dict[str, Any]) -> Attestation:
        """
        Witness an observation (no verification, just recording).
        """
        return self.attest(
            subject=subject,
            subject_content=observation,
            verdict="confirmed",
            evidence={"observation": observation}
        )


def main():
    """Demo the Witness pattern."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Witness - Attestation Pattern")
    parser.add_argument("command", choices=["attest", "verify", "observe", "test"])
    parser.add_argument("--subject", help="Subject to witness")
    parser.add_argument("--verdict", default="confirmed", help="Verdict")
    
    args = parser.parse_args()
    witness = Witness("witness_cli", WitnessType.OBSERVATION)
    
    if args.command == "test":
        att = witness.attest(
            subject="test-episode-001",
            subject_content={"action": "test", "result": "success"},
            verdict="confirmed",
            evidence={"test": True}
        )
        print(f"✅ Attestation produced: {att.attestation_id}")
        print(f"   Signature: {att.signature()}")
    
    return 0


if __name__ == "__main__":
    exit(main())
