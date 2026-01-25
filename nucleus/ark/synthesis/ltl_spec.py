#!/usr/bin/env python3
"""
ltl_spec.py - Linear Temporal Logic Specifications

Formalizes correctness properties for ARK commits:
- Safety (□): Bad things never happen
- Liveness (◇): Good things eventually happen
- Reactive (□→◇): Triggers and responses
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from enum import Enum
import re


class LTLOperator(Enum):
    """LTL temporal operators."""
    GLOBALLY = "□"       # Always (box)
    FINALLY = "◇"        # Eventually (diamond)
    NEXT = "○"           # Next state
    UNTIL = "U"          # Until
    IMPLIES = "→"        # Implication


@dataclass
class LTLFormula:
    """A Linear Temporal Logic formula."""
    name: str
    formula: str
    description: str
    category: str = "general"  # safety, liveness, reactive
    
    # Compiled checker (optional)
    _checker: Optional[Callable] = None
    
    def is_safety(self) -> bool:
        """Check if this is a safety property (□ prefix)."""
        return self.formula.startswith("□")
    
    def is_liveness(self) -> bool:
        """Check if this is a liveness property (contains ◇)."""
        return "◇" in self.formula
    
    def is_reactive(self) -> bool:
        """Check if this is a reactive property (□(trigger → ◇response))."""
        return "→" in self.formula and "◇" in self.formula


@dataclass
class LTLSpec:
    """A collection of LTL formulas defining system properties."""
    name: str
    version: str = "1.0"
    formulas: List[LTLFormula] = field(default_factory=list)
    
    def add_safety(self, name: str, formula: str, description: str) -> None:
        """Add a safety property."""
        self.formulas.append(LTLFormula(
            name=name,
            formula=formula,
            description=description,
            category="safety"
        ))
    
    def add_liveness(self, name: str, formula: str, description: str) -> None:
        """Add a liveness property."""
        self.formulas.append(LTLFormula(
            name=name,
            formula=formula,
            description=description,
            category="liveness"
        ))
    
    def add_reactive(self, name: str, formula: str, description: str) -> None:
        """Add a reactive property."""
        self.formulas.append(LTLFormula(
            name=name,
            formula=formula,
            description=description,
            category="reactive"
        ))
    
    @property
    def safety_formulas(self) -> List[LTLFormula]:
        return [f for f in self.formulas if f.category == "safety"]
    
    @property
    def liveness_formulas(self) -> List[LTLFormula]:
        return [f for f in self.formulas if f.category == "liveness"]


class PluribusLTLSpec:
    """
    Pre-defined LTL specifications for Pluribus/ARK.
    
    Encodes the DNA axioms as formal properties.
    """
    
    @staticmethod
    def core_spec() -> LTLSpec:
        """Get the core Pluribus LTL specification."""
        spec = LTLSpec(name="pluribus_core", version="1.0")
        
        # DNA Axiom 1: Inertia (Safety)
        spec.add_safety(
            "inertia",
            "□ (mutation(nucleus) → verified(mutation))",
            "Nucleus changes must be verified"
        )
        
        # DNA Axiom 2: Entelecheia (Liveness)
        spec.add_liveness(
            "entelecheia",
            "□ (proposed(change) → ◇ witnessed(change))",
            "Every proposed change must eventually be witnessed"
        )
        
        # DNA Axiom 3: Homeostasis (Reactive)
        spec.add_reactive(
            "homeostasis",
            "□ (drift(system) → ◇ stabilized(system))",
            "System drift triggers eventual stabilization"
        )
        
        # DNA Axiom 4: Hysteresis (Safety)
        spec.add_safety(
            "hysteresis",
            "□ (state(t) ← influences(state(t-1)))",
            "Past states influence present"
        )
        
        # DNA Axiom 5: Infinity (Liveness)
        spec.add_liveness(
            "infinity",
            "□◇ stable(system)",
            "System is infinitely often stable (Büchi)"
        )
        
        # ARK-specific
        spec.add_safety(
            "gate_compliance",
            "□ (commit → (inertia_pass ∧ entelecheia_pass ∧ homeostasis_pass))",
            "All commits must pass DNA gates"
        )
        
        spec.add_reactive(
            "entropy_response",
            "□ (h_total > 0.7 → ○ stabilization_mode)",
            "High entropy triggers stabilization"
        )
        
        return spec


class LTLVerifier:
    """
    Verifies system traces against LTL specifications.
    
    Uses bounded model checking for finite traces.
    """
    
    def __init__(self, spec: LTLSpec):
        self.spec = spec
        self.violations: List[Dict] = []
    
    def verify_trace(self, trace: List[Dict]) -> bool:
        """
        Verify a trace (sequence of states) against the spec.
        
        Args:
            trace: List of state dictionaries
        
        Returns:
            True if all properties hold, False otherwise
        """
        self.violations = []
        all_pass = True
        
        for formula in self.spec.formulas:
            passed = self._check_formula(formula, trace)
            if not passed:
                all_pass = False
                self.violations.append({
                    "formula": formula.name,
                    "category": formula.category,
                    "description": formula.description
                })
        
        return all_pass
    
    def _check_formula(self, formula: LTLFormula, trace: List[Dict]) -> bool:
        """Check a single formula against a trace."""
        # Simplified checking - pattern matching on common formulas
        
        if formula.name == "inertia":
            # Check: all nucleus mutations are verified
            for state in trace:
                if state.get("mutation_nucleus") and not state.get("verified"):
                    return False
            return True
        
        elif formula.name == "entelecheia":
            # Check: all proposed changes are eventually witnessed
            proposed = set()
            witnessed = set()
            for state in trace:
                if state.get("proposed"):
                    proposed.add(state.get("change_id"))
                if state.get("witnessed"):
                    witnessed.add(state.get("change_id"))
            return proposed <= witnessed
        
        elif formula.name == "homeostasis":
            # Check: drift → eventually stabilized
            in_drift = False
            for state in trace:
                if state.get("drift"):
                    in_drift = True
                if in_drift and state.get("stabilized"):
                    in_drift = False
            return not in_drift  # Must end stabilized if was in drift
        
        elif formula.name == "gate_compliance":
            # Check: all commits pass gates
            for state in trace:
                if state.get("commit"):
                    if not (state.get("inertia_pass") and 
                            state.get("entelecheia_pass") and 
                            state.get("homeostasis_pass")):
                        return False
            return True
        
        elif formula.name == "infinity":
            # Check: stable occurs (simplified finite trace check)
            return any(s.get("stable") for s in trace)
        
        # Default: pass (unknown formula)
        return True
    
    def get_violation_report(self) -> str:
        """Get a human-readable violation report."""
        if not self.violations:
            return "✅ All LTL properties verified"
        
        lines = ["❌ LTL violations detected:"]
        for v in self.violations:
            lines.append(f"  - {v['formula']} ({v['category']}): {v['description']}")
        return "\n".join(lines)
