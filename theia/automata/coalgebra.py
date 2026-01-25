"""
Theia DNA Coalgebra — Dual Neurosymbolic Automata state machine.

Coalgebraic structure: Q → F(Q) × Mod(E)
    - Q: State space
    - F(Q): Next state functor (observation + transition)
    - Mod(E): Energy landscape modification (reentrant self-improvement)

The DUAL structure enables:
    - Forward execution: coalgebra Q → F(Q)
    - Backward constraint propagation: algebra F*(E) → E

This is the core of self-teaching: higher-level states can modify
the energy landscapes of lower levels.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any, Generic, TypeVar
from enum import Enum, auto
from abc import ABC, abstractmethod


# Type variables for generic state machine
S = TypeVar('S')  # State type
O = TypeVar('O')  # Observation type
A = TypeVar('A')  # Action type


class StateType(Enum):
    """Types of automaton states."""
    JUDGMENT = auto()   # Branching decision
    PROCESSING = auto() # Execute action
    TERMINAL = auto()   # End state
    REENTRY = auto()    # Self-modification state


@dataclass
class State:
    """
    Automaton state with typed transitions.
    
    Attributes:
        id: Unique state identifier
        type: State type (judgment, processing, etc.)
        action: Action to execute (for processing states)
        transitions: Map from observation → next state
        metadata: Additional state information
    """
    id: str
    type: StateType
    action: Optional[str] = None
    transitions: Dict[Any, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_next(self, observation: Any) -> Optional[str]:
        """Get next state given observation."""
        return self.transitions.get(observation)


@dataclass
class Modification:
    """
    Energy landscape modification descriptor.
    
    Used by reentry states to modify lower-level dynamics.
    """
    target_layer: str  # Which layer to modify
    operation: str     # "scale", "shift", "replace", "blend"
    parameters: Dict[str, Any] = field(default_factory=dict)


class Coalgebra(ABC):
    """
    Abstract coalgebra interface: Q → F(Q).
    
    Coalgebras are the categorical dual of algebras.
    They encode observation and state transition.
    """
    
    @abstractmethod
    def observe(self, state: S) -> O:
        """Extract observation from state."""
        pass
    
    @abstractmethod
    def transition(self, state: S, observation: O) -> S:
        """Transition to next state given observation."""
        pass
    
    def unfold(self, initial: S, steps: int) -> List[Tuple[S, O]]:
        """Unfold coalgebra from initial state."""
        trace = []
        state = initial
        
        for _ in range(steps):
            obs = self.observe(state)
            trace.append((state, obs))
            state = self.transition(state, obs)
        
        return trace


class DNAAutomaton:
    """
    Dual Neurosymbolic Automaton.
    
    Implements coalgebraic state machine with reentrant modification:
        Q → F(Q) × Mod(E)
    
    Attributes:
        states: Map of state ID → State
        initial: Initial state ID
        current: Current state ID
        energy_modifiers: Map of layer → modification history
    """
    
    def __init__(self, initial_state: str = "start"):
        self.states: Dict[str, State] = {}
        self.initial = initial_state
        self.current = initial_state
        self.energy_modifiers: Dict[str, List[Modification]] = {}
        self._execution_trace: List[str] = []
    
    def add_state(self, state: State) -> None:
        """Add state to automaton."""
        self.states[state.id] = state
    
    def add_transition(self, from_id: str, observation: Any, to_id: str) -> None:
        """Add transition between states."""
        if from_id in self.states:
            self.states[from_id].transitions[observation] = to_id
    
    def observe(self, environment: Any) -> Any:
        """
        Coalgebraic observation: extract observable from current state + environment.
        """
        state = self.states.get(self.current)
        if not state:
            return None
        
        if state.type == StateType.JUDGMENT:
            # Judgment states observe environment to decide
            return self._evaluate_condition(state, environment)
        elif state.type == StateType.PROCESSING:
            # Processing states observe action result
            return self._execute_action(state, environment)
        elif state.type == StateType.REENTRY:
            # Reentry states observe modification result
            return self._apply_modification(state, environment)
        else:
            return None
    
    def _evaluate_condition(self, state: State, env: Any) -> Any:
        """Evaluate judgment condition."""
        condition = state.metadata.get("condition")
        if callable(condition):
            return condition(env)
        return env  # Default: environment is the observation
    
    def _execute_action(self, state: State, env: Any) -> Any:
        """Execute processing action."""
        # In real implementation, this would dispatch to action handlers
        return f"executed:{state.action}"
    
    def _apply_modification(self, state: State, env: Any) -> Modification:
        """Apply reentry modification to energy landscape."""
        mod = Modification(
            target_layer=state.metadata.get("target", "mHC"),
            operation=state.metadata.get("operation", "scale"),
            parameters=state.metadata.get("params", {}),
        )
        
        # Record modification
        layer = mod.target_layer
        if layer not in self.energy_modifiers:
            self.energy_modifiers[layer] = []
        self.energy_modifiers[layer].append(mod)
        
        return mod
    
    def step(self, environment: Any) -> Tuple[str, Any]:
        """
        Execute one step of automaton.
        
        Returns:
            (next_state_id, observation)
        """
        obs = self.observe(environment)
        
        state = self.states.get(self.current)
        if not state:
            return self.current, obs
        
        # Get next state from transitions
        next_id = state.get_next(obs)
        if next_id is None:
            # Default transition
            next_id = state.transitions.get("default", self.current)
        
        self._execution_trace.append(self.current)
        self.current = next_id
        
        return next_id, obs
    
    def run(self, environment: Any, max_steps: int = 100) -> List[Tuple[str, Any]]:
        """
        Run automaton until terminal or max steps.
        
        Returns:
            Trace of (state_id, observation) pairs
        """
        trace = []
        
        for _ in range(max_steps):
            state = self.states.get(self.current)
            if not state or state.type == StateType.TERMINAL:
                break
            
            next_id, obs = self.step(environment)
            trace.append((next_id, obs))
        
        return trace
    
    def reset(self) -> None:
        """Reset automaton to initial state."""
        self.current = self.initial
        self._execution_trace = []
    
    def get_modifications(self, layer: str) -> List[Modification]:
        """Get all modifications applied to a layer."""
        return self.energy_modifiers.get(layer, [])


def create_simple_dna(
    states: List[Tuple[str, StateType, Optional[str]]],
    transitions: List[Tuple[str, Any, str]],
    initial: str = None,
) -> DNAAutomaton:
    """
    Factory for simple DNA automaton.
    
    Args:
        states: List of (id, type, action) tuples
        transitions: List of (from, observation, to) tuples
        initial: Initial state ID (default: first state)
        
    Returns:
        Configured DNAAutomaton
    """
    if not states:
        raise ValueError("Need at least one state")
    
    initial = initial or states[0][0]
    dna = DNAAutomaton(initial_state=initial)
    
    for state_id, state_type, action in states:
        dna.add_state(State(
            id=state_id,
            type=state_type,
            action=action,
        ))
    
    for from_id, obs, to_id in transitions:
        dna.add_transition(from_id, obs, to_id)
    
    return dna


# Example: Browser automation DNA
def create_browser_dna() -> DNAAutomaton:
    """
    Create DNA for browser automation task.
    
    States: navigate → wait → interact → verify → done
    """
    dna = DNAAutomaton(initial_state="navigate")
    
    dna.add_state(State("navigate", StateType.PROCESSING, action="open_url"))
    dna.add_state(State("wait", StateType.PROCESSING, action="wait_load"))
    dna.add_state(State("interact", StateType.JUDGMENT))
    dna.add_state(State("fill", StateType.PROCESSING, action="fill_input"))
    dna.add_state(State("click", StateType.PROCESSING, action="click_button"))
    dna.add_state(State("verify", StateType.JUDGMENT))
    dna.add_state(State("success", StateType.TERMINAL))
    dna.add_state(State("retry", StateType.REENTRY, metadata={
        "target": "mHC",
        "operation": "boost",
        "params": {"beta_factor": 1.5},
    }))
    
    # Transitions
    dna.add_transition("navigate", "default", "wait")
    dna.add_transition("wait", "default", "interact")
    dna.add_transition("interact", "input", "fill")
    dna.add_transition("interact", "button", "click")
    dna.add_transition("fill", "default", "interact")
    dna.add_transition("click", "default", "verify")
    dna.add_transition("verify", True, "success")
    dna.add_transition("verify", False, "retry")
    dna.add_transition("retry", "default", "navigate")
    
    return dna


__all__ = [
    "StateType",
    "State", 
    "Modification",
    "Coalgebra",
    "DNAAutomaton",
    "create_simple_dna",
    "create_browser_dna",
]
