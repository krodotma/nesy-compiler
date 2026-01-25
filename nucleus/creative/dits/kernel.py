"""
DiTS Kernel - Diegetic Transition System Core
==============================================

Implements the core DiTS kernel with mu/nu (mu/nu) calculus for
modeling modal fixpoint logic and transition systems.

The mu/nu calculus provides:
- mu (least fixpoint): Reachability, eventuality - "something will happen"
- nu (greatest fixpoint): Safety, invariance - "something always holds"

This forms the basis for narrative transition modeling where:
- States represent narrative moments/positions
- Transitions represent possible narrative moves
- Fixpoints define narrative constraints and goals
"""

from __future__ import annotations

import time
import uuid
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Set,
    Callable,
    TypeVar,
    Generic,
    Iterator,
    Tuple,
    Union,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import hashlib
import json


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

StateId = str
TransitionLabel = str
T = TypeVar("T")


class FixpointType(str, Enum):
    """Type of fixpoint operator."""
    MU = "mu"      # Least fixpoint (reachability)
    NU = "nu"      # Greatest fixpoint (safety)


class TransitionType(str, Enum):
    """Type of narrative transition."""
    SEQUENTIAL = "sequential"        # Linear progression
    BRANCHING = "branching"          # Choice point
    MERGING = "merging"              # Convergence
    LOOPING = "looping"              # Recursive return
    DIEGETIC = "diegetic"            # Story-world internal
    EXTRADIEGETIC = "extradiegetic"  # Meta-narrative


class EvaluationMode(Enum):
    """Mode for fixpoint evaluation."""
    EAGER = auto()      # Evaluate immediately
    LAZY = auto()       # Evaluate on demand
    MEMOIZED = auto()   # Cache results


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class DiTSSpec:
    """
    Specification for a Diegetic Transition System.

    Defines the structure and constraints of a narrative state machine.

    Attributes:
        name: Human-readable name for the system
        version: Specification version
        states: Set of valid state identifiers
        initial_states: Set of starting states
        final_states: Set of accepting/terminal states
        alphabet: Set of valid transition labels
        transitions: Mapping from (state, label) to target states
        mu_formulas: Least fixpoint formulas (reachability goals)
        nu_formulas: Greatest fixpoint formulas (safety invariants)
        metadata: Additional specification data
    """
    name: str
    version: str = "1.0.0"
    states: Set[StateId] = field(default_factory=set)
    initial_states: Set[StateId] = field(default_factory=set)
    final_states: Set[StateId] = field(default_factory=set)
    alphabet: Set[TransitionLabel] = field(default_factory=set)
    transitions: Dict[Tuple[StateId, TransitionLabel], Set[StateId]] = field(
        default_factory=dict
    )
    mu_formulas: Dict[str, "Formula"] = field(default_factory=dict)
    nu_formulas: Dict[str, "Formula"] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate specification after initialization."""
        # Ensure initial states are valid
        if self.initial_states and self.states:
            invalid = self.initial_states - self.states
            if invalid:
                raise ValueError(f"Invalid initial states: {invalid}")

        # Ensure final states are valid
        if self.final_states and self.states:
            invalid = self.final_states - self.states
            if invalid:
                raise ValueError(f"Invalid final states: {invalid}")

    def add_state(self, state_id: StateId, initial: bool = False, final: bool = False) -> None:
        """Add a state to the specification."""
        self.states.add(state_id)
        if initial:
            self.initial_states.add(state_id)
        if final:
            self.final_states.add(state_id)

    def add_transition(
        self,
        source: StateId,
        label: TransitionLabel,
        target: StateId,
    ) -> None:
        """Add a transition to the specification."""
        self.states.add(source)
        self.states.add(target)
        self.alphabet.add(label)

        key = (source, label)
        if key not in self.transitions:
            self.transitions[key] = set()
        self.transitions[key].add(target)

    def get_successors(
        self,
        state: StateId,
        label: Optional[TransitionLabel] = None,
    ) -> Set[StateId]:
        """Get successor states from a given state."""
        if label is not None:
            return self.transitions.get((state, label), set())

        # All successors across all labels
        successors = set()
        for (s, _), targets in self.transitions.items():
            if s == state:
                successors.update(targets)
        return successors

    def get_predecessors(
        self,
        state: StateId,
        label: Optional[TransitionLabel] = None,
    ) -> Set[StateId]:
        """Get predecessor states that can reach the given state."""
        predecessors = set()
        for (source, trans_label), targets in self.transitions.items():
            if state in targets:
                if label is None or trans_label == label:
                    predecessors.add(source)
        return predecessors

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "states": list(self.states),
            "initial_states": list(self.initial_states),
            "final_states": list(self.final_states),
            "alphabet": list(self.alphabet),
            "transitions": {
                f"{s}:{l}": list(targets)
                for (s, l), targets in self.transitions.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiTSSpec":
        """Create specification from dictionary."""
        transitions = {}
        for key, targets in data.get("transitions", {}).items():
            source, label = key.split(":", 1)
            transitions[(source, label)] = set(targets)

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            states=set(data.get("states", [])),
            initial_states=set(data.get("initial_states", [])),
            final_states=set(data.get("final_states", [])),
            alphabet=set(data.get("alphabet", [])),
            transitions=transitions,
            metadata=data.get("metadata", {}),
        )

    def hash(self) -> str:
        """Compute hash of specification for caching."""
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class DiTSState:
    """
    Current state of a DiTS execution.

    Tracks the runtime state during narrative traversal.

    Attributes:
        spec: The specification being executed
        current_states: Set of currently active states (for non-determinism)
        history: Sequence of (state, label, state) transitions taken
        evaluation_cache: Cached fixpoint evaluation results
        metadata: Runtime metadata
        created_at: Timestamp of state creation
    """
    spec: DiTSSpec
    current_states: Set[StateId] = field(default_factory=set)
    history: List[Tuple[StateId, TransitionLabel, StateId]] = field(
        default_factory=list
    )
    evaluation_cache: Dict[str, Set[StateId]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize current states from spec if not provided."""
        if not self.current_states:
            self.current_states = self.spec.initial_states.copy()

    @property
    def is_terminal(self) -> bool:
        """Check if current states are all terminal."""
        if not self.current_states:
            return True
        return all(s in self.spec.final_states for s in self.current_states)

    @property
    def is_accepting(self) -> bool:
        """Check if any current state is accepting/final."""
        return bool(self.current_states & self.spec.final_states)

    @property
    def available_transitions(self) -> Set[TransitionLabel]:
        """Get all available transition labels from current states."""
        labels = set()
        for state in self.current_states:
            for (s, label), _ in self.spec.transitions.items():
                if s == state:
                    labels.add(label)
        return labels

    def step(self, label: TransitionLabel) -> "DiTSState":
        """
        Take a transition step, returning new state.

        Args:
            label: The transition label to follow

        Returns:
            New DiTSState after the transition
        """
        new_states = set()
        new_history = self.history.copy()

        for state in self.current_states:
            successors = self.spec.get_successors(state, label)
            for succ in successors:
                new_history.append((state, label, succ))
            new_states.update(successors)

        if not new_states:
            raise ValueError(f"No valid transition for label '{label}' from states {self.current_states}")

        return DiTSState(
            spec=self.spec,
            current_states=new_states,
            history=new_history,
            evaluation_cache={},  # Clear cache on transition
            metadata=self.metadata.copy(),
        )

    def reachable_states(self, max_depth: int = 100) -> Set[StateId]:
        """Compute all reachable states from current states."""
        visited = set()
        frontier = self.current_states.copy()
        depth = 0

        while frontier and depth < max_depth:
            visited.update(frontier)
            next_frontier = set()
            for state in frontier:
                next_frontier.update(self.spec.get_successors(state))
            frontier = next_frontier - visited
            depth += 1

        return visited

    def clone(self) -> "DiTSState":
        """Create a deep copy of this state."""
        return DiTSState(
            spec=self.spec,
            current_states=self.current_states.copy(),
            history=self.history.copy(),
            evaluation_cache=self.evaluation_cache.copy(),
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "spec_hash": self.spec.hash(),
            "current_states": list(self.current_states),
            "history": [
                {"from": s, "label": l, "to": t}
                for s, l, t in self.history
            ],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


# =============================================================================
# FORMULA SYSTEM (for mu/nu calculus)
# =============================================================================

class Formula(ABC):
    """
    Abstract base class for mu/nu calculus formulas.

    Formulas define properties that can be evaluated against DiTS states.
    """

    @abstractmethod
    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        """
        Evaluate the formula and return satisfying states.

        Args:
            spec: The DiTS specification
            variable_bindings: Current variable bindings

        Returns:
            Set of states satisfying the formula
        """
        pass

    @abstractmethod
    def to_string(self) -> str:
        """Return string representation of formula."""
        pass

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class AtomicFormula(Formula):
    """Atomic proposition (state predicate)."""
    predicate: Callable[[StateId], bool]
    name: str = "atom"

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        return {s for s in spec.states if self.predicate(s)}

    def to_string(self) -> str:
        return self.name


@dataclass
class StateSetFormula(Formula):
    """Formula for a specific set of states."""
    states: Set[StateId]

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        return self.states & spec.states

    def to_string(self) -> str:
        return f"{{{','.join(sorted(self.states))}}}"


@dataclass
class VariableFormula(Formula):
    """Variable reference in fixpoint formula."""
    name: str

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        return variable_bindings.get(self.name, set())

    def to_string(self) -> str:
        return self.name


@dataclass
class NotFormula(Formula):
    """Negation of a formula."""
    inner: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        inner_result = self.inner.evaluate(spec, variable_bindings)
        return spec.states - inner_result

    def to_string(self) -> str:
        return f"NOT({self.inner.to_string()})"


@dataclass
class AndFormula(Formula):
    """Conjunction of formulas."""
    left: Formula
    right: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        left_result = self.left.evaluate(spec, variable_bindings)
        right_result = self.right.evaluate(spec, variable_bindings)
        return left_result & right_result

    def to_string(self) -> str:
        return f"({self.left.to_string()} AND {self.right.to_string()})"


@dataclass
class OrFormula(Formula):
    """Disjunction of formulas."""
    left: Formula
    right: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        left_result = self.left.evaluate(spec, variable_bindings)
        right_result = self.right.evaluate(spec, variable_bindings)
        return left_result | right_result

    def to_string(self) -> str:
        return f"({self.left.to_string()} OR {self.right.to_string()})"


@dataclass
class DiamondFormula(Formula):
    """
    Diamond modality: <label>phi

    States from which there EXISTS a transition labeled 'label'
    to a state satisfying phi.
    """
    label: Optional[TransitionLabel]  # None means any label
    inner: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        inner_states = self.inner.evaluate(spec, variable_bindings)
        result = set()

        for (source, trans_label), targets in spec.transitions.items():
            if self.label is None or trans_label == self.label:
                if targets & inner_states:
                    result.add(source)

        return result

    def to_string(self) -> str:
        label_str = self.label if self.label else "-"
        return f"<{label_str}>{self.inner.to_string()}"


@dataclass
class BoxFormula(Formula):
    """
    Box modality: [label]phi

    States from which ALL transitions labeled 'label'
    lead to states satisfying phi.
    """
    label: Optional[TransitionLabel]  # None means any label
    inner: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        inner_states = self.inner.evaluate(spec, variable_bindings)
        result = set()

        for state in spec.states:
            successors = spec.get_successors(state, self.label)
            if not successors or successors <= inner_states:
                result.add(state)

        return result

    def to_string(self) -> str:
        label_str = self.label if self.label else "-"
        return f"[{label_str}]{self.inner.to_string()}"


@dataclass
class MuFormula(Formula):
    """
    Least fixpoint: mu X. phi(X)

    Computes the smallest set X such that X = phi(X).
    Used for reachability properties (eventually reaching a goal).
    """
    variable: str
    body: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        # Start with empty set (least fixpoint)
        current = set()
        bindings = variable_bindings.copy()

        # Iterate until fixpoint
        while True:
            bindings[self.variable] = current
            next_set = self.body.evaluate(spec, bindings)

            if next_set == current:
                break
            current = next_set

        return current

    def to_string(self) -> str:
        return f"mu {self.variable}. {self.body.to_string()}"


@dataclass
class NuFormula(Formula):
    """
    Greatest fixpoint: nu X. phi(X)

    Computes the largest set X such that X = phi(X).
    Used for safety properties (always maintaining an invariant).
    """
    variable: str
    body: Formula

    def evaluate(
        self,
        spec: DiTSSpec,
        variable_bindings: Dict[str, Set[StateId]],
    ) -> Set[StateId]:
        # Start with all states (greatest fixpoint)
        current = spec.states.copy()
        bindings = variable_bindings.copy()

        # Iterate until fixpoint
        while True:
            bindings[self.variable] = current
            next_set = self.body.evaluate(spec, bindings)

            if next_set == current:
                break
            current = next_set

        return current

    def to_string(self) -> str:
        return f"nu {self.variable}. {self.body.to_string()}"


# =============================================================================
# DiTS KERNEL
# =============================================================================

@dataclass
class EvaluationResult:
    """Result of a formula evaluation."""
    formula: str
    satisfying_states: Set[StateId]
    evaluation_time: float
    fixpoint_type: Optional[FixpointType] = None
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DiTSKernel:
    """
    DiTS Kernel - Core engine for Diegetic Transition System evaluation.

    Implements mu/nu calculus evaluation, state space exploration,
    and narrative transition management.

    Example usage:
        >>> spec = DiTSSpec(name="story")
        >>> spec.add_state("intro", initial=True)
        >>> spec.add_state("climax")
        >>> spec.add_state("resolution", final=True)
        >>> spec.add_transition("intro", "develop", "climax")
        >>> spec.add_transition("climax", "resolve", "resolution")
        >>>
        >>> kernel = DiTSKernel(spec)
        >>> state = kernel.create_initial_state()
        >>> state = kernel.step(state, "develop")
        >>> print(kernel.check_reachability(state, {"resolution"}))
    """

    def __init__(
        self,
        spec: DiTSSpec,
        evaluation_mode: EvaluationMode = EvaluationMode.MEMOIZED,
    ):
        """
        Initialize DiTS Kernel.

        Args:
            spec: The DiTS specification to operate on
            evaluation_mode: How to evaluate formulas
        """
        self.spec = spec
        self.evaluation_mode = evaluation_mode
        self._formula_cache: Dict[str, Set[StateId]] = {}
        self._stats = {
            "evaluations": 0,
            "cache_hits": 0,
            "transitions": 0,
        }

    def create_initial_state(self) -> DiTSState:
        """Create an initial execution state."""
        return DiTSState(spec=self.spec)

    def step(self, state: DiTSState, label: TransitionLabel) -> DiTSState:
        """
        Execute a transition step.

        Args:
            state: Current state
            label: Transition label to follow

        Returns:
            New state after transition
        """
        self._stats["transitions"] += 1
        return state.step(label)

    def evaluate_formula(
        self,
        formula: Formula,
        variable_bindings: Optional[Dict[str, Set[StateId]]] = None,
    ) -> EvaluationResult:
        """
        Evaluate a formula against the specification.

        Args:
            formula: The formula to evaluate
            variable_bindings: Initial variable bindings

        Returns:
            EvaluationResult with satisfying states
        """
        bindings = variable_bindings or {}
        formula_key = formula.to_string()

        # Check cache for memoized mode
        if self.evaluation_mode == EvaluationMode.MEMOIZED:
            if formula_key in self._formula_cache:
                self._stats["cache_hits"] += 1
                return EvaluationResult(
                    formula=formula_key,
                    satisfying_states=self._formula_cache[formula_key],
                    evaluation_time=0.0,
                )

        self._stats["evaluations"] += 1
        start_time = time.time()

        result_states = formula.evaluate(self.spec, bindings)

        evaluation_time = time.time() - start_time

        # Cache result
        if self.evaluation_mode == EvaluationMode.MEMOIZED:
            self._formula_cache[formula_key] = result_states

        # Determine fixpoint type
        fixpoint_type = None
        if isinstance(formula, MuFormula):
            fixpoint_type = FixpointType.MU
        elif isinstance(formula, NuFormula):
            fixpoint_type = FixpointType.NU

        return EvaluationResult(
            formula=formula_key,
            satisfying_states=result_states,
            evaluation_time=evaluation_time,
            fixpoint_type=fixpoint_type,
        )

    def check_reachability(
        self,
        from_state: DiTSState,
        target_states: Set[StateId],
    ) -> bool:
        """
        Check if target states are reachable from current state.

        Uses mu-calculus: mu X. target OR <->X
        (least fixpoint of: target states or can reach X)

        Args:
            from_state: Starting state
            target_states: Goal states to reach

        Returns:
            True if any target is reachable
        """
        # Build reachability formula
        target_formula = StateSetFormula(target_states)
        var_formula = VariableFormula("X")
        diamond = DiamondFormula(None, var_formula)  # Any transition
        body = OrFormula(target_formula, diamond)
        reachability = MuFormula("X", body)

        result = self.evaluate_formula(reachability)
        return bool(from_state.current_states & result.satisfying_states)

    def check_safety(
        self,
        from_state: DiTSState,
        invariant: Set[StateId],
    ) -> bool:
        """
        Check if invariant is maintained from current state.

        Uses nu-calculus: nu X. invariant AND [-]X
        (greatest fixpoint of: invariant holds and all successors satisfy X)

        Args:
            from_state: Starting state
            invariant: States that should always be maintained

        Returns:
            True if invariant always holds
        """
        # Build safety formula
        inv_formula = StateSetFormula(invariant)
        var_formula = VariableFormula("X")
        box = BoxFormula(None, var_formula)  # All transitions
        body = AndFormula(inv_formula, box)
        safety = NuFormula("X", body)

        result = self.evaluate_formula(safety)
        return from_state.current_states <= result.satisfying_states

    def compute_winning_states(
        self,
        objective: Formula,
    ) -> Set[StateId]:
        """
        Compute states from which the objective can be achieved.

        Args:
            objective: The goal formula

        Returns:
            Set of winning states
        """
        result = self.evaluate_formula(objective)
        return result.satisfying_states

    def explore_state_space(
        self,
        initial_state: Optional[DiTSState] = None,
        max_depth: int = 100,
    ) -> Dict[str, Any]:
        """
        Explore the state space from initial state.

        Args:
            initial_state: Starting state (or use spec's initial states)
            max_depth: Maximum exploration depth

        Returns:
            Exploration statistics and results
        """
        state = initial_state or self.create_initial_state()

        visited = set()
        frontier = [(s, 0) for s in state.current_states]
        paths: Dict[StateId, List[Tuple[TransitionLabel, StateId]]] = {
            s: [] for s in state.current_states
        }

        while frontier:
            current, depth = frontier.pop(0)

            if current in visited or depth >= max_depth:
                continue

            visited.add(current)

            for (s, label), targets in self.spec.transitions.items():
                if s == current:
                    for target in targets:
                        if target not in visited:
                            frontier.append((target, depth + 1))
                            if target not in paths:
                                paths[target] = paths[current] + [(label, target)]

        return {
            "visited_states": visited,
            "state_count": len(visited),
            "reachable_from_initial": len(visited),
            "final_states_reachable": visited & self.spec.final_states,
            "paths": paths,
            "max_depth_reached": max_depth if len(visited) == max_depth else len(visited),
        }

    def find_path(
        self,
        from_states: Set[StateId],
        to_states: Set[StateId],
        max_length: int = 100,
    ) -> Optional[List[Tuple[StateId, TransitionLabel, StateId]]]:
        """
        Find a path between state sets.

        Args:
            from_states: Starting states
            to_states: Goal states
            max_length: Maximum path length

        Returns:
            List of transitions forming a path, or None if no path exists
        """
        # BFS for shortest path
        queue: List[Tuple[StateId, List[Tuple[StateId, TransitionLabel, StateId]]]] = [
            (s, []) for s in from_states
        ]
        visited = set()

        while queue:
            current, path = queue.pop(0)

            if current in to_states:
                return path

            if current in visited or len(path) >= max_length:
                continue

            visited.add(current)

            for (s, label), targets in self.spec.transitions.items():
                if s == current:
                    for target in targets:
                        if target not in visited:
                            new_path = path + [(current, label, target)]
                            queue.append((target, new_path))

        return None

    def bisimulation_quotient(self) -> DiTSSpec:
        """
        Compute the bisimulation quotient of the specification.

        Minimizes the state space while preserving behavioral equivalence.

        Returns:
            A minimized DiTSSpec
        """
        # Partition refinement algorithm
        partitions: List[Set[StateId]] = [self.spec.states.copy()]

        changed = True
        while changed:
            changed = False
            new_partitions = []

            for partition in partitions:
                # Try to split partition based on transitions
                splits: Dict[Tuple, Set[StateId]] = {}

                for state in partition:
                    # Compute signature: which partitions are reachable
                    signature = []
                    for label in self.spec.alphabet:
                        targets = self.spec.get_successors(state, label)
                        target_parts = frozenset(
                            i for i, p in enumerate(partitions) if targets & p
                        )
                        signature.append((label, target_parts))

                    sig_tuple = tuple(sorted(signature))
                    if sig_tuple not in splits:
                        splits[sig_tuple] = set()
                    splits[sig_tuple].add(state)

                if len(splits) > 1:
                    changed = True

                new_partitions.extend(splits.values())

            partitions = new_partitions

        # Build quotient automaton
        quotient = DiTSSpec(name=f"{self.spec.name}_quotient")

        # Map states to partition representatives
        state_to_repr: Dict[StateId, StateId] = {}
        for partition in partitions:
            repr_state = min(partition)  # Use smallest state as representative
            for state in partition:
                state_to_repr[state] = repr_state
            quotient.states.add(repr_state)

            if partition & self.spec.initial_states:
                quotient.initial_states.add(repr_state)
            if partition & self.spec.final_states:
                quotient.final_states.add(repr_state)

        # Add transitions
        for (source, label), targets in self.spec.transitions.items():
            repr_source = state_to_repr[source]
            for target in targets:
                repr_target = state_to_repr[target]
                quotient.add_transition(repr_source, label, repr_target)

        quotient.metadata["original_states"] = len(self.spec.states)
        quotient.metadata["quotient_states"] = len(quotient.states)

        return quotient

    def clear_cache(self) -> None:
        """Clear the formula evaluation cache."""
        self._formula_cache.clear()

    @property
    def statistics(self) -> Dict[str, int]:
        """Get kernel statistics."""
        return self._stats.copy()

    def reset_statistics(self) -> None:
        """Reset kernel statistics."""
        self._stats = {
            "evaluations": 0,
            "cache_hits": 0,
            "transitions": 0,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core types
    "StateId",
    "TransitionLabel",
    "FixpointType",
    "TransitionType",
    "EvaluationMode",
    # Data structures
    "DiTSSpec",
    "DiTSState",
    "EvaluationResult",
    # Formulas
    "Formula",
    "AtomicFormula",
    "StateSetFormula",
    "VariableFormula",
    "NotFormula",
    "AndFormula",
    "OrFormula",
    "DiamondFormula",
    "BoxFormula",
    "MuFormula",
    "NuFormula",
    # Kernel
    "DiTSKernel",
]


if __name__ == "__main__":
    # Demo usage
    spec = DiTSSpec(name="simple_story")
    spec.add_state("intro", initial=True)
    spec.add_state("rising_action")
    spec.add_state("climax")
    spec.add_state("falling_action")
    spec.add_state("resolution", final=True)

    spec.add_transition("intro", "develop", "rising_action")
    spec.add_transition("rising_action", "escalate", "climax")
    spec.add_transition("climax", "turn", "falling_action")
    spec.add_transition("falling_action", "conclude", "resolution")
    spec.add_transition("climax", "twist", "rising_action")  # Recursive option

    kernel = DiTSKernel(spec)
    state = kernel.create_initial_state()

    print(f"DiTS Kernel Demo: {spec.name}")
    print(f"Initial states: {state.current_states}")
    print(f"Available transitions: {state.available_transitions}")

    # Check reachability to resolution
    can_reach = kernel.check_reachability(state, {"resolution"})
    print(f"Can reach resolution: {can_reach}")

    # Take some steps
    state = kernel.step(state, "develop")
    print(f"After 'develop': {state.current_states}")

    state = kernel.step(state, "escalate")
    print(f"After 'escalate': {state.current_states}")

    # Explore state space
    exploration = kernel.explore_state_space()
    print(f"Reachable states: {exploration['visited_states']}")

    print(f"Kernel statistics: {kernel.statistics}")
