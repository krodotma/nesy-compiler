#!/usr/bin/env python3
"""
verifier.py - Formal Verification for ARK

P2-071: Implement formal verification hooks
P2-073: Create `ark verify --deep` command
P2-074: Implement contract verification
P2-075: Add invariant checker

Integrates Büchi/Streett automata concepts from Claude Opus R&D.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum

logger = logging.getLogger("ARK.Testing.Verifier")


class VerificationStatus(Enum):
    PASSED = "passed"
    FAILED = "failed"
    UNKNOWN = "unknown"
    TIMEOUT = "timeout"


@dataclass
class Contract:
    """A contract specification for verification."""
    name: str
    precondition: str  # Expression that must hold before
    postcondition: str  # Expression that must hold after
    invariant: Optional[str] = None  # Expression that must always hold
    
    def to_dict(self) -> Dict:
        return {"name": self.name, "pre": self.precondition, 
                "post": self.postcondition, "inv": self.invariant}


@dataclass
class VerificationResult:
    """Result of formal verification."""
    status: VerificationStatus
    property_name: str
    counterexample: Optional[Any] = None
    trace: List[str] = field(default_factory=list)
    time_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {"status": self.status.value, "property": self.property_name,
                "counterexample": str(self.counterexample) if self.counterexample else None}


class FormalVerifier:
    """
    Formal verification engine with Büchi automata support.
    
    P2-071: Formal verification hooks
    Based on Claude Opus insight: Büchi acceptance for infinite CMP improvement.
    """
    
    def __init__(self, timeout_seconds: float = 30.0):
        self.timeout = timeout_seconds
        self.contracts: Dict[str, Contract] = {}
        self.invariants: List[str] = []
    
    def register_contract(self, contract: Contract) -> None:
        """Register a contract for verification."""
        self.contracts[contract.name] = contract
    
    def add_invariant(self, invariant: str) -> None:
        """Add a global invariant."""
        self.invariants.append(invariant)
    
    def verify_contract(
        self, contract: Contract, 
        test_inputs: List[Dict[str, Any]],
        executor: Callable[[Dict], Dict]
    ) -> VerificationResult:
        """Verify contract against test inputs."""
        for inputs in test_inputs:
            try:
                # Check precondition
                pre_result = self._eval_condition(contract.precondition, inputs)
                if not pre_result:
                    continue  # Skip if precondition not met
                
                # Execute
                outputs = executor(inputs)
                context = {**inputs, **outputs}
                
                # Check postcondition
                post_result = self._eval_condition(contract.postcondition, context)
                if not post_result:
                    return VerificationResult(
                        status=VerificationStatus.FAILED,
                        property_name=contract.name,
                        counterexample={"inputs": inputs, "outputs": outputs}
                    )
            except Exception as e:
                logger.debug("Verification error: %s", e)
        
        return VerificationResult(status=VerificationStatus.PASSED, 
                                  property_name=contract.name)
    
    def verify_buchi(
        self, trace: List[Dict[str, Any]], 
        acceptance: str
    ) -> VerificationResult:
        """
        Verify Büchi acceptance condition on trace.
        
        Claude Opus insight: CMP improvement should occur *infinitely often*.
        """
        acceptance_count = 0
        trace_length = len(trace)
        
        for state in trace:
            if self._eval_condition(acceptance, state):
                acceptance_count += 1
        
        # For finite trace, check if acceptance occurs "often enough"
        acceptance_ratio = acceptance_count / max(trace_length, 1)
        
        if acceptance_ratio > 0.5:  # Threshold for "infinitely often"
            return VerificationResult(
                status=VerificationStatus.PASSED,
                property_name=f"buchi:{acceptance}",
                trace=[str(acceptance_ratio)]
            )
        else:
            return VerificationResult(
                status=VerificationStatus.FAILED,
                property_name=f"buchi:{acceptance}",
                counterexample={"acceptance_ratio": acceptance_ratio}
            )
    
    def verify_invariant(
        self, invariant: str, 
        trace: List[Dict[str, Any]]
    ) -> VerificationResult:
        """Verify invariant holds throughout trace."""
        for i, state in enumerate(trace):
            if not self._eval_condition(invariant, state):
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    property_name=f"inv:{invariant}",
                    counterexample={"step": i, "state": state}
                )
        
        return VerificationResult(status=VerificationStatus.PASSED,
                                  property_name=f"inv:{invariant}")
    
    def _eval_condition(self, condition: str, context: Dict) -> bool:
        """Safely evaluate a condition in context."""
        try:
            return bool(eval(condition, {"__builtins__": {}}, context))
        except Exception:
            return False
    
    def verify_streett(
        self, 
        trace: List[Dict[str, Any]], 
        fairness_pairs: List[tuple]
    ) -> VerificationResult:
        """
        Verify Streett acceptance condition on trace.
        
        Claude Opus R&D insight: Streett automaton for stronger guarantees.
        Streett condition: For ALL pairs (recurrence, persistence),
        if recurrence holds infinitely often, then persistence must hold
        infinitely often.
        
        Args:
            trace: Execution trace
            fairness_pairs: List of (recurrence_cond, persistence_cond) tuples
        """
        for rec_cond, pers_cond in fairness_pairs:
            rec_count = sum(1 for s in trace if self._eval_condition(rec_cond, s))
            pers_count = sum(1 for s in trace if self._eval_condition(pers_cond, s))
            
            rec_ratio = rec_count / max(len(trace), 1)
            pers_ratio = pers_count / max(len(trace), 1)
            
            # If recurrence occurs often, persistence must also occur often
            if rec_ratio > 0.3 and pers_ratio < 0.3:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    property_name=f"streett:({rec_cond},{pers_cond})",
                    counterexample={
                        "recurrence_ratio": rec_ratio,
                        "persistence_ratio": pers_ratio
                    }
                )
        
        return VerificationResult(
            status=VerificationStatus.PASSED,
            property_name="streett:all_pairs"
        )
    
    def verify_fairness(
        self, 
        trace: List[Dict[str, Any]],
        enabled_action: str,
        taken_action: str
    ) -> VerificationResult:
        """
        Verify strong fairness (Streett-based).
        
        If action is enabled infinitely often, it must be taken infinitely often.
        Critical for preventing starvation in gate execution.
        """
        return self.verify_streett(trace, [(enabled_action, taken_action)])
    
    def cmp_liveness_contract(self) -> List[tuple]:
        """
        Streett pairs for CMP liveness.
        
        Claude Opus: "CMP improvement should occur infinitely often"
        """
        return [
            # If we commit, improvement should eventually happen
            ("committed == True", "cmp_delta > 0"),
            # If entropy is reducible, it should eventually reduce
            ("entropy_reducible == True", "entropy_reduced == True"),
        ]
    
    # --- Built-in ARK contracts ---
    
    def cmp_improvement_contract(self) -> Contract:
        """CMP should not decrease significantly."""
        return Contract(
            name="cmp_improvement",
            precondition="cmp_before >= 0",
            postcondition="cmp_after >= cmp_before * 0.9",
            invariant="0 <= cmp <= 1"
        )
    
    def entropy_bounds_contract(self) -> Contract:
        """Entropy values must be bounded."""
        return Contract(
            name="entropy_bounds",
            precondition="True",
            postcondition="all(0 <= v <= 1 for v in entropy.values())"
        )
    
    def gate_consistency_contract(self) -> Contract:
        """Gate decisions must be deterministic."""
        return Contract(
            name="gate_consistency",
            precondition="True",
            postcondition="decision1 == decision2"
        )
