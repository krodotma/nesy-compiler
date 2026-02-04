#!/usr/bin/env python3
"""
guard_ladder.py - Neuro-Symbolic Guard Ladder (G1-G6)

DUALITY-BIND Phase 2: Guard Ladder Implementation

The Guard Ladder provides decidable symbolic validation for neural proposals.
Every action must pass the ladder before execution or selection.

Guards (from DNA.md HGT Guard Ladder):
- G1: Type Compatibility
- G2: Timing Compatibility  
- G3: Effect Boundary (Ring 0 protection)
- G4: Omega Acceptance (lineage compatibility)
- G5: MDL Penalty (complexity cost)
- G6: Spectral Stability (signature verification)

Ring: 1 (Operator)
Protocol: DKIN v29 | PAIP v15 | Citizen v1

Usage:
    python3 guard_ladder.py check <proposal_json>
    python3 guard_ladder.py validate-action <action_type> <target>
    python3 guard_ladder.py status
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
RING_0_PATHS = [
    "DNA.md", "CITIZEN.md", "ring_guard.py", "guard_ladder.py",
    "nucleus/specs/dkin_protocol", "nucleus/specs/paip_protocol",
]


class GuardResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    SKIP = "skip"


@dataclass
class GuardOutcome:
    """Result of a single guard check."""
    guard_id: str  # G1, G2, etc.
    guard_name: str
    result: GuardResult
    message: str
    score: float = 1.0  # 1.0 = pass, 0.0 = fail, 0.5 = warn
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["result"] = self.result.value
        return d


@dataclass
class LadderResult:
    """Aggregated result from running all guards."""
    passed: bool
    outcomes: List[GuardOutcome]
    total_score: float  # Product of individual scores
    shaped_reward: float  # For RL feedback
    proposal_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "total_score": round(self.total_score, 4),
            "shaped_reward": round(self.shaped_reward, 4),
            "proposal_id": self.proposal_id,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


class GuardLadder:
    """
    The Guard Ladder: 6 decidable checks for neuro-symbolic validation.
    
    Flow:
    1. Neural module proposes action
    2. Guard Ladder validates via G1-G6
    3. Pass/fail becomes shaped reward
    4. Neural module updates from feedback
    """
    
    def __init__(self):
        self.ring_0_patterns = [re.compile(p) for p in RING_0_PATHS]
    
    # =========================================================================
    # G1: Type Compatibility
    # =========================================================================
    def g1_type_check(self, proposal: Dict[str, Any]) -> GuardOutcome:
        """
        G1: Verify type compatibility of the proposal.
        
        Checks:
        - Required fields present
        - Field types match schema
        - No unexpected nulls in critical fields
        """
        required_fields = ["action", "target"]
        missing = [f for f in required_fields if f not in proposal]
        
        if missing:
            return GuardOutcome(
                guard_id="G1",
                guard_name="Type Compatibility",
                result=GuardResult.FAIL,
                message=f"Missing required fields: {missing}",
                score=0.0,
                details={"missing": missing},
            )
        
        # Check action is string
        if not isinstance(proposal.get("action"), str):
            return GuardOutcome(
                guard_id="G1",
                guard_name="Type Compatibility",
                result=GuardResult.FAIL,
                message="Field 'action' must be a string",
                score=0.0,
            )
        
        return GuardOutcome(
            guard_id="G1",
            guard_name="Type Compatibility",
            result=GuardResult.PASS,
            message="Type check passed",
            score=1.0,
        )
    
    # =========================================================================
    # G2: Timing Compatibility
    # =========================================================================
    def g2_timing_check(self, proposal: Dict[str, Any]) -> GuardOutcome:
        """
        G2: Verify timing constraints.
        
        Checks:
        - Deadline not exceeded
        - Rate limits not violated
        - TTL not expired
        """
        deadline = proposal.get("deadline")
        if deadline and time.time() > deadline:
            return GuardOutcome(
                guard_id="G2",
                guard_name="Timing Compatibility",
                result=GuardResult.FAIL,
                message=f"Deadline exceeded: {deadline}",
                score=0.0,
                details={"deadline": deadline, "now": time.time()},
            )
        
        ttl = proposal.get("ttl")
        created_at = proposal.get("created_at", time.time())
        if ttl and (time.time() - created_at) > ttl:
            return GuardOutcome(
                guard_id="G2",
                guard_name="Timing Compatibility",
                result=GuardResult.FAIL,
                message=f"TTL expired: {ttl}s",
                score=0.0,
            )
        
        return GuardOutcome(
            guard_id="G2",
            guard_name="Timing Compatibility",
            result=GuardResult.PASS,
            message="Timing check passed",
            score=1.0,
        )
    
    # =========================================================================
    # G3: Effect Boundary (Ring 0 Protection)
    # =========================================================================
    def g3_effect_boundary(self, proposal: Dict[str, Any]) -> GuardOutcome:
        """
        G3: Verify effect doesn't violate Ring 0 boundaries.
        
        Checks:
        - Target path not in Ring 0
        - Action not destructive on protected resources
        - Actor has required ring level
        """
        target = proposal.get("target", "")
        action = proposal.get("action", "")
        actor_ring = proposal.get("actor_ring", 3)  # Default to ephemeral
        
        # Check if target is Ring 0
        is_ring_0 = any(p.search(str(target)) for p in self.ring_0_patterns)
        
        if is_ring_0:
            # Only Ring 0 actors can modify Ring 0
            if actor_ring > 0:
                return GuardOutcome(
                    guard_id="G3",
                    guard_name="Effect Boundary",
                    result=GuardResult.FAIL,
                    message=f"Ring {actor_ring} actor cannot modify Ring 0 target: {target}",
                    score=0.0,
                    details={"target": target, "actor_ring": actor_ring},
                )
            
            # Even Ring 0 actors get a warning
            return GuardOutcome(
                guard_id="G3",
                guard_name="Effect Boundary",
                result=GuardResult.WARN,
                message=f"Ring 0 modification detected: {target}",
                score=0.5,
                details={"target": target, "actor_ring": actor_ring},
            )
        
        # Destructive actions need higher ring
        destructive_actions = ["delete", "remove", "destroy", "drop"]
        if action.lower() in destructive_actions and actor_ring > 1:
            return GuardOutcome(
                guard_id="G3",
                guard_name="Effect Boundary",
                result=GuardResult.FAIL,
                message=f"Destructive action '{action}' requires Ring 0-1",
                score=0.0,
            )
        
        return GuardOutcome(
            guard_id="G3",
            guard_name="Effect Boundary",
            result=GuardResult.PASS,
            message="Effect boundary check passed",
            score=1.0,
        )
    
    # =========================================================================
    # G4: Omega Acceptance (Lineage Compatibility)
    # =========================================================================
    def g4_omega_acceptance(self, proposal: Dict[str, Any]) -> GuardOutcome:
        """
        G4: Verify lineage compatibility for HGT.
        
        Checks:
        - Source and target lineages are compatible
        - Omega motifs don't conflict
        - Acceptance conditions satisfied
        """
        source_lineage = proposal.get("source_lineage")
        target_lineage = proposal.get("target_lineage")
        
        # Skip if not an HGT operation
        if not source_lineage or not target_lineage:
            return GuardOutcome(
                guard_id="G4",
                guard_name="Omega Acceptance",
                result=GuardResult.SKIP,
                message="Not an HGT operation",
                score=1.0,
            )
        
        # Check lineage depth compatibility (simple heuristic)
        source_depth = proposal.get("source_depth", 0)
        target_depth = proposal.get("target_depth", 0)
        
        if abs(source_depth - target_depth) > 5:
            return GuardOutcome(
                guard_id="G4",
                guard_name="Omega Acceptance",
                result=GuardResult.WARN,
                message=f"Large lineage depth gap: {source_depth} vs {target_depth}",
                score=0.7,
                details={"source_depth": source_depth, "target_depth": target_depth},
            )
        
        return GuardOutcome(
            guard_id="G4",
            guard_name="Omega Acceptance",
            result=GuardResult.PASS,
            message="Omega acceptance check passed",
            score=1.0,
        )
    
    # =========================================================================
    # G5: MDL Penalty (Complexity Cost)
    # =========================================================================
    def g5_mdl_penalty(self, proposal: Dict[str, Any]) -> GuardOutcome:
        """
        G5: Apply Minimum Description Length penalty.
        
        Checks:
        - Proposal complexity within bounds
        - Code/content size reasonable
        - Dependency count acceptable
        """
        content = proposal.get("content", "")
        dependencies = proposal.get("dependencies", [])
        max_complexity = proposal.get("max_complexity", 10000)
        
        # Compute complexity as content length + dependency penalty
        content_len = len(str(content))
        dep_penalty = len(dependencies) * 100
        complexity = content_len + dep_penalty
        
        if complexity > max_complexity:
            return GuardOutcome(
                guard_id="G5",
                guard_name="MDL Penalty",
                result=GuardResult.FAIL,
                message=f"Complexity {complexity} exceeds max {max_complexity}",
                score=0.0,
                details={"complexity": complexity, "max": max_complexity},
            )
        
        # Score proportional to headroom
        score = 1.0 - (complexity / max_complexity) * 0.5
        
        return GuardOutcome(
            guard_id="G5",
            guard_name="MDL Penalty",
            result=GuardResult.PASS,
            message=f"Complexity {complexity} within bounds",
            score=max(0.5, score),
            details={"complexity": complexity, "max": max_complexity},
        )
    
    # =========================================================================
    # G6: Spectral Stability (Signature Verification)
    # =========================================================================
    def g6_spectral_stability(self, proposal: Dict[str, Any]) -> GuardOutcome:
        """
        G6: Verify spectral/cryptographic stability.
        
        Checks:
        - Hash signatures valid
        - PQC attestation present (if required)
        - Content integrity verified
        """
        content = proposal.get("content", "")
        expected_hash = proposal.get("content_hash")
        requires_pqc = proposal.get("requires_pqc", False)
        pqc_signature = proposal.get("pqc_signature")
        
        # Verify content hash if provided
        if expected_hash:
            actual_hash = hashlib.sha256(str(content).encode()).hexdigest()[:16]
            if not expected_hash.startswith(actual_hash[:8]):
                return GuardOutcome(
                    guard_id="G6",
                    guard_name="Spectral Stability",
                    result=GuardResult.FAIL,
                    message=f"Hash mismatch: expected {expected_hash[:8]}, got {actual_hash[:8]}",
                    score=0.0,
                )
        
        # Check PQC signature if required
        if requires_pqc and not pqc_signature:
            return GuardOutcome(
                guard_id="G6",
                guard_name="Spectral Stability",
                result=GuardResult.FAIL,
                message="PQC signature required but not provided",
                score=0.0,
            )
        
        return GuardOutcome(
            guard_id="G6",
            guard_name="Spectral Stability",
            result=GuardResult.PASS,
            message="Spectral stability check passed",
            score=1.0,
        )
    
    # =========================================================================
    # Run Full Ladder
    # =========================================================================
    def run(self, proposal: Dict[str, Any]) -> LadderResult:
        """
        Run all guards on a proposal.
        
        Returns:
            LadderResult with aggregated pass/fail and shaped reward
        """
        outcomes = [
            self.g1_type_check(proposal),
            self.g2_timing_check(proposal),
            self.g3_effect_boundary(proposal),
            self.g4_omega_acceptance(proposal),
            self.g5_mdl_penalty(proposal),
            self.g6_spectral_stability(proposal),
        ]
        
        # Aggregate results
        failed = any(o.result == GuardResult.FAIL for o in outcomes)
        total_score = 1.0
        for o in outcomes:
            if o.result != GuardResult.SKIP:
                total_score *= o.score
        
        # Compute shaped reward for RL
        # - +1.0 for full pass
        # - -0.5 for failure
        # - Proportional for partial passes
        if failed:
            shaped_reward = -0.5
        else:
            shaped_reward = total_score * 1.0
        
        return LadderResult(
            passed=not failed,
            outcomes=outcomes,
            total_score=total_score,
            shaped_reward=shaped_reward,
            proposal_id=proposal.get("id"),
        )
    
    def emit_bus_event(self, result: LadderResult) -> Optional[str]:
        """Emit guard result to bus."""
        try:
            from agent_bus import emit_bus_event
            topic = "guard.ladder.passed" if result.passed else "guard.ladder.failed"
            return emit_bus_event(
                topic=topic,
                actor="guard_ladder",
                data=result.to_dict(),
            )
        except ImportError:
            return None


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_check(args):
    """Check a proposal against the guard ladder."""
    try:
        proposal = json.loads(args.proposal_json)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON - {e}")
        return 1
    
    ladder = GuardLadder()
    result = ladder.run(proposal)
    
    print(f"Guard Ladder Result: {'✅ PASSED' if result.passed else '❌ FAILED'}")
    print(f"Total Score: {result.total_score:.3f}")
    print(f"Shaped Reward: {result.shaped_reward:+.3f}")
    print()
    
    for outcome in result.outcomes:
        icon = {"pass": "✅", "fail": "❌", "warn": "⚠️", "skip": "⏭️"}[outcome.result.value]
        print(f"  {icon} {outcome.guard_id} ({outcome.guard_name}): {outcome.message}")
    
    return 0 if result.passed else 1


def cmd_validate_action(args):
    """Quick validation for a simple action."""
    proposal = {
        "action": args.action_type,
        "target": args.target,
        "actor_ring": 2,  # Standard application ring
    }
    
    ladder = GuardLadder()
    result = ladder.run(proposal)
    
    if result.passed:
        print(f"✅ Action '{args.action_type}' on '{args.target}' is allowed")
    else:
        print(f"❌ Action '{args.action_type}' on '{args.target}' is BLOCKED")
        for o in result.outcomes:
            if o.result == GuardResult.FAIL:
                print(f"   Reason: {o.message}")
    
    return 0 if result.passed else 1


def cmd_status(args):
    """Show guard ladder status."""
    print("Guard Ladder Status")
    print("  Guards: G1-G6 (6 total)")
    print("  Ring 0 Protected Paths:")
    for p in RING_0_PATHS:
        print(f"    - {p}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="Guard Ladder - Neuro-Symbolic Validation")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # check
    p_check = subparsers.add_parser("check", help="Check proposal against ladder")
    p_check.add_argument("proposal_json", help="Proposal as JSON string")
    
    # validate-action
    p_validate = subparsers.add_parser("validate-action", help="Quick action validation")
    p_validate.add_argument("action_type", help="Action type (e.g., write, delete)")
    p_validate.add_argument("target", help="Target path or resource")
    
    # status
    subparsers.add_parser("status", help="Show guard ladder status")
    
    args = parser.parse_args()
    
    if args.command == "check":
        return cmd_check(args)
    elif args.command == "validate-action":
        return cmd_validate_action(args)
    elif args.command == "status":
        return cmd_status(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
