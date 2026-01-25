"""
Theia Reflexive Omega Domain — Ω ≅ [Ω → Ω].

The reflexive domain Omega is the fixed point of the continuous
function space constructor: Ω = [Ω → Ω].

This enables:
    - Self-reference: The system can model itself
    - Metacognition: Reasoning about reasoning
    - Gödel-style self-improvement: Modify own computation

Mathematical Foundation:
    - Scott domains (D∞ construction)
    - Corecursive types
    - Fixed-point combinators

Integration with Theia:
    - L2 mHC: Omega can modify attractor landscapes
    - L3 Birkhoff: Omega can adjust crystallization parameters
    - L4 DNA: Omega is the domain of self-teaching states
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List, Tuple
from abc import ABC, abstractmethod
import hashlib
import time


# =============================================================================
# REFLEXIVE DOMAIN CORE
# =============================================================================

@dataclass
class OmegaState:
    """
    State in the reflexive domain Ω.
    
    An OmegaState represents a point in the self-referential space
    where the system can model and modify itself.
    
    Attributes:
        id: Unique state identifier
        level: Nesting level (0 = base, 1 = meta, 2 = meta-meta, ...)
        content: The actual content (may be another OmegaState)
        transform: How this state transforms other states
        metadata: Additional state information
    """
    id: str
    level: int = 0
    content: Any = None
    transform: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def apply(self, other: "OmegaState") -> "OmegaState":
        """
        Apply this state as a function to another state.
        
        This is the Ω → Ω structure: states are also transformations.
        """
        if self.transform is None:
            # Identity if no transform
            return other
        
        # Apply transform and wrap result
        result = self.transform(other)
        
        if isinstance(result, OmegaState):
            return result
        else:
            return OmegaState(
                id=f"{self.id}({other.id})",
                level=max(self.level, other.level),
                content=result,
            )
    
    def compose(self, other: "OmegaState") -> "OmegaState":
        """
        Compose two omega states: (f ∘ g)(x) = f(g(x)).
        """
        def composed_transform(x: OmegaState) -> OmegaState:
            intermediate = other.apply(x)
            return self.apply(intermediate)
        
        return OmegaState(
            id=f"({self.id}∘{other.id})",
            level=max(self.level, other.level) + 1,
            transform=composed_transform,
        )
    
    def fix(self, iterations: int = 10) -> "OmegaState":
        """
        Compute fixed point: x such that f(x) = x.
        
        Uses iterative application until convergence or max iterations.
        """
        current = self
        for i in range(iterations):
            next_state = self.apply(current)
            
            # Check for convergence (content equality)
            if (next_state.content is not None and 
                current.content is not None and
                np.array_equal(next_state.content, current.content)):
                next_state.metadata["fixed_point_iterations"] = i + 1
                return next_state
            
            current = next_state
        
        current.metadata["fixed_point_iterations"] = iterations
        current.metadata["converged"] = False
        return current


class OmegaDomain:
    """
    The reflexive domain Ω ≅ [Ω → Ω].
    
    Provides the infrastructure for self-referential computation:
        - State management
        - Fixed-point computation
        - Meta-level access
    """
    
    def __init__(self):
        self._states: Dict[str, OmegaState] = {}
        self._history: List[Tuple[str, OmegaState, float]] = []
        self._meta_level = 0
    
    def register(self, state: OmegaState) -> str:
        """Register a state in the domain."""
        self._states[state.id] = state
        self._history.append((state.id, state, time.time()))
        return state.id
    
    def get(self, state_id: str) -> Optional[OmegaState]:
        """Get state by ID."""
        return self._states.get(state_id)
    
    def apply(self, f_id: str, x_id: str) -> OmegaState:
        """
        Apply function state to argument state.
        
        This is the core operation: Ω × Ω → Ω
        """
        f = self._states.get(f_id)
        x = self._states.get(x_id)
        
        if f is None or x is None:
            raise ValueError(f"State not found: {f_id if f is None else x_id}")
        
        result = f.apply(x)
        self.register(result)
        return result
    
    def lift(self, value: Any, transform: Optional[Callable] = None) -> OmegaState:
        """
        Lift a value into the omega domain.
        
        Creates an OmegaState from any value.
        """
        state = OmegaState(
            id=f"lift-{hashlib.md5(str(value).encode()).hexdigest()[:8]}",
            level=0,
            content=value,
            transform=transform,
        )
        self.register(state)
        return state
    
    def meta(self) -> "OmegaDomain":
        """
        Access meta-level of the domain.
        
        Returns a view of the domain at the next meta-level,
        where states become transformations on transformations.
        """
        meta_domain = OmegaDomain()
        meta_domain._meta_level = self._meta_level + 1
        
        # Lift all states to meta-level
        for state_id, state in self._states.items():
            meta_state = OmegaState(
                id=f"meta({state_id})",
                level=state.level + 1,
                content=state,
                transform=lambda x, s=state: s.apply(x),
            )
            meta_domain.register(meta_state)
        
        return meta_domain
    
    def fixed_point(self, f: OmegaState, initial: OmegaState) -> OmegaState:
        """
        Compute fixed point of f starting from initial.
        
        Finds x such that f(x) ≈ x.
        """
        return f.fix(iterations=20)


# =============================================================================
# METACOGNITION INTERFACE
# =============================================================================

@dataclass
class MetacognitiveState:
    """
    State for metacognitive processing.
    
    Tracks the system's understanding of its own state.
    """
    confidence: float = 0.0
    uncertainty: float = 1.0
    reasoning_trace: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)


class Metacognition:
    """
    Metacognitive capabilities for self-monitoring.
    
    Provides:
        - Confidence estimation
        - Uncertainty tracking
        - Self-reflection
        - Intervention planning
    """
    
    def __init__(self, omega: OmegaDomain):
        self.omega = omega
        self._state = MetacognitiveState()
    
    def reflect(self, observation: Any) -> MetacognitiveState:
        """
        Reflect on an observation.
        
        Updates metacognitive state with new information.
        """
        # Lift observation to omega domain
        obs_state = self.omega.lift(observation)
        
        # Check if we've seen similar states
        similar_count = sum(
            1 for s in self.omega._states.values()
            if s.level == obs_state.level
        )
        
        # Update confidence based on familiarity
        self._state.confidence = min(1.0, similar_count / 10.0)
        self._state.uncertainty = 1.0 - self._state.confidence
        
        self._state.reasoning_trace.append(
            f"Observed: {obs_state.id}, similar states: {similar_count}"
        )
        
        return self._state
    
    def should_intervene(self) -> bool:
        """
        Decide if metacognitive intervention is needed.
        
        High uncertainty triggers intervention.
        """
        return self._state.uncertainty > 0.7
    
    def plan_intervention(self) -> List[str]:
        """
        Plan interventions based on current state.
        """
        interventions = []
        
        if self._state.uncertainty > 0.9:
            interventions.append("request_human_feedback")
        elif self._state.uncertainty > 0.7:
            interventions.append("explore_alternatives")
        elif self._state.confidence < 0.5:
            interventions.append("gather_more_data")
        
        self._state.interventions = interventions
        return interventions
    
    def get_state(self) -> Dict[str, Any]:
        """Get current metacognitive state as dict."""
        return {
            "confidence": self._state.confidence,
            "uncertainty": self._state.uncertainty,
            "traces": len(self._state.reasoning_trace),
            "interventions": self._state.interventions,
        }


# =============================================================================
# INTEGRATION WITH THEIA LAYERS
# =============================================================================

class OmegaModifier:
    """
    Uses Omega domain to modify lower layers.
    
    This is the reentry mechanism: higher-level states
    modify the dynamics of lower-level computation.
    """
    
    def __init__(self, omega: OmegaDomain):
        self.omega = omega
        self._modifications: List[Dict[str, Any]] = []
    
    def modify_mhc_beta(self, delta: float) -> OmegaState:
        """
        Create omega state that modifies mHC temperature.
        """
        def modifier(state: OmegaState) -> OmegaState:
            # Modify beta parameter in state metadata
            new_meta = state.metadata.copy()
            new_meta["mhc_beta_delta"] = delta
            return OmegaState(
                id=f"{state.id}_beta{delta:+.2f}",
                level=state.level,
                content=state.content,
                metadata=new_meta,
            )
        
        mod_state = OmegaState(
            id=f"mod_beta_{delta:+.2f}",
            level=1,
            transform=modifier,
        )
        self.omega.register(mod_state)
        
        self._modifications.append({
            "type": "mhc_beta",
            "delta": delta,
            "state_id": mod_state.id,
            "timestamp": time.time(),
        })
        
        return mod_state
    
    def modify_crystallization(self, pressure_factor: float) -> OmegaState:
        """
        Create omega state that modifies crystallization pressure.
        """
        def modifier(state: OmegaState) -> OmegaState:
            new_meta = state.metadata.copy()
            new_meta["crystallization_pressure_factor"] = pressure_factor
            return OmegaState(
                id=f"{state.id}_cryst{pressure_factor:.2f}",
                level=state.level,
                content=state.content,
                metadata=new_meta,
            )
        
        mod_state = OmegaState(
            id=f"mod_cryst_{pressure_factor:.2f}",
            level=1,
            transform=modifier,
        )
        self.omega.register(mod_state)
        
        self._modifications.append({
            "type": "crystallization",
            "pressure_factor": pressure_factor,
            "state_id": mod_state.id,
            "timestamp": time.time(),
        })
        
        return mod_state
    
    def get_pending_modifications(self) -> List[Dict[str, Any]]:
        """Get list of pending modifications."""
        return self._modifications.copy()


__all__ = [
    "OmegaState",
    "OmegaDomain",
    "MetacognitiveState",
    "Metacognition",
    "OmegaModifier",
]
