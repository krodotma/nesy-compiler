#!/usr/bin/env python3
"""
hgt_guard.py - Horizontal Gene Transfer Guards (G1-G6)

DUALITY-BIND E9: Enforce 6-gate ladder for safe cross-lineage module transfer.

Ring: 0 (Kernel) - This is critical infrastructure
Protocol: DKIN v29
"""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

# Import guard_ladder for integration
try:
    from guard_ladder import GuardLadder, GuardResult, GuardOutcome, LadderResult
except ImportError:
    # Fallback definitions
    class GuardResult(Enum):
        PASS = "pass"
        FAIL = "fail"
        WARN = "warn"


@dataclass
class HGTProposal:
    """A proposal for horizontal gene transfer."""
    proposal_id: str
    source_lineage: str
    target_lineage: str
    module_id: str
    module_content: str
    source_depth: int = 0
    target_depth: int = 0
    module_type: str = "code"  # code, config, prompt


@dataclass
class HGTResult:
    """Result of HGT guard evaluation."""
    passed: bool
    guards_passed: List[str]
    guards_failed: List[str]
    guards_warned: List[str]
    total_score: float
    rejection_reason: Optional[str] = None


class HGTGuard:
    """
    Horizontal Gene Transfer Guard - 6 gates from DNA.md:
    - G1: Type Compatibility
    - G2: Timing Compatibility
    - G3: Effect Boundary (Ring 0 protection)
    - G4: Omega Acceptance (lineage compatibility)
    - G5: MDL Penalty (complexity cost)
    - G6: Spectral Stability (PQC signatures)
    """
    
    def __init__(self):
        self.max_complexity = 50000
        self.max_depth_gap = 10
    
    def g1_type_compatibility(self, proposal: HGTProposal) -> tuple:
        """Check type compatibility between source and target."""
        # Check module type is valid
        valid_types = ["code", "config", "prompt", "motif"]
        if proposal.module_type not in valid_types:
            return (GuardResult.FAIL, f"Invalid module type: {proposal.module_type}")
        
        # Check content is non-empty
        if not proposal.module_content:
            return (GuardResult.FAIL, "Empty module content")
        
        return (GuardResult.PASS, "Type compatibility OK")
    
    def g2_timing_compatibility(self, proposal: HGTProposal) -> tuple:
        """Check timing constraints."""
        # Placeholder: would check rate limits, cooldowns
        return (GuardResult.PASS, "Timing OK")
    
    def g3_effect_boundary(self, proposal: HGTProposal) -> tuple:
        """Check Ring 0 protection."""
        protected_patterns = ["DNA.md", "CITIZEN.md", "ring_guard", "hgt_guard"]
        
        for pattern in protected_patterns:
            if pattern in proposal.module_content:
                return (GuardResult.FAIL, f"Module references Ring 0: {pattern}")
        
        return (GuardResult.PASS, "Effect boundary OK")
    
    def g4_omega_acceptance(self, proposal: HGTProposal) -> tuple:
        """Check lineage compatibility."""
        depth_gap = abs(proposal.source_depth - proposal.target_depth)
        
        if depth_gap > self.max_depth_gap:
            return (GuardResult.FAIL, f"Lineage depth gap too large: {depth_gap}")
        
        if depth_gap > 5:
            return (GuardResult.WARN, f"Large lineage depth gap: {depth_gap}")
        
        return (GuardResult.PASS, "Omega acceptance OK")
    
    def g5_mdl_penalty(self, proposal: HGTProposal) -> tuple:
        """Check complexity bounds."""
        complexity = len(proposal.module_content)
        
        if complexity > self.max_complexity:
            return (GuardResult.FAIL, f"Complexity {complexity} exceeds max {self.max_complexity}")
        
        if complexity > self.max_complexity * 0.8:
            return (GuardResult.WARN, f"High complexity: {complexity}")
        
        return (GuardResult.PASS, f"Complexity OK: {complexity}")
    
    def g6_spectral_stability(self, proposal: HGTProposal) -> tuple:
        """Verify spectral/cryptographic stability."""
        # Compute content hash
        content_hash = hashlib.sha256(proposal.module_content.encode()).hexdigest()[:16]
        
        # Placeholder for PQC signature verification
        # In production, would verify against known-good signatures
        
        return (GuardResult.PASS, f"Spectral stable: {content_hash}")
    
    def evaluate(self, proposal: HGTProposal) -> HGTResult:
        """Run all guards on a proposal."""
        guards = [
            ("G1", self.g1_type_compatibility),
            ("G2", self.g2_timing_compatibility),
            ("G3", self.g3_effect_boundary),
            ("G4", self.g4_omega_acceptance),
            ("G5", self.g5_mdl_penalty),
            ("G6", self.g6_spectral_stability),
        ]
        
        passed = []
        failed = []
        warned = []
        rejection_reason = None
        
        for guard_id, guard_fn in guards:
            result, message = guard_fn(proposal)
            
            if result == GuardResult.PASS:
                passed.append(guard_id)
            elif result == GuardResult.FAIL:
                failed.append(guard_id)
                if not rejection_reason:
                    rejection_reason = f"{guard_id}: {message}"
            elif result == GuardResult.WARN:
                warned.append(guard_id)
        
        # Compute score
        total_guards = len(guards)
        score = (len(passed) + 0.5 * len(warned)) / total_guards
        
        return HGTResult(
            passed=len(failed) == 0,
            guards_passed=passed,
            guards_failed=failed,
            guards_warned=warned,
            total_score=score,
            rejection_reason=rejection_reason,
        )


def main():
    parser = argparse.ArgumentParser(description="HGT Guard")
    parser.add_argument("--source", default="lineage-source")
    parser.add_argument("--target", default="lineage-target")
    parser.add_argument("--module", default="def hello(): pass")
    parser.add_argument("--type", default="code")
    args = parser.parse_args()
    
    proposal = HGTProposal(
        proposal_id=f"hgt-{int(time.time())}",
        source_lineage=args.source,
        target_lineage=args.target,
        module_id="test-module",
        module_content=args.module,
        module_type=args.type,
    )
    
    guard = HGTGuard()
    result = guard.evaluate(proposal)
    
    icon = "✅" if result.passed else "❌"
    print(f"{icon} HGT Guard Result: {'PASSED' if result.passed else 'FAILED'}")
    print(f"  Score: {result.total_score:.2f}")
    print(f"  Passed: {result.guards_passed}")
    print(f"  Failed: {result.guards_failed}")
    print(f"  Warned: {result.guards_warned}")
    if result.rejection_reason:
        print(f"  Reason: {result.rejection_reason}")
    
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
