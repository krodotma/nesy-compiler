"""
DiTS Subsystem Benchmarks
=========================

Benchmarks for Diegetic Transition System kernel evaluation and narrative generation.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Any, Optional, Dict, List, Set
from enum import Enum

from .bench_runner import BenchmarkSuite


class TransitionType(Enum):
    """Types of diegetic transitions."""
    MU = "mu"  # Least fixpoint (finite unfolding)
    NU = "nu"  # Greatest fixpoint (infinite unfolding)
    DIRECT = "direct"  # Direct state transition
    RECURSIVE = "recursive"  # Self-referential transition


@dataclass
class DiTSState:
    """State in a Diegetic Transition System."""
    id: str
    properties: Dict[str, Any]
    successors: List[str] = field(default_factory=list)
    is_terminal: bool = False
    depth: int = 0

    @classmethod
    def random(cls, id_prefix: str = "s", depth: int = 0) -> "DiTSState":
        """Create random state."""
        return cls(
            id=f"{id_prefix}_{random.randint(0, 10000)}",
            properties={
                "value": random.random(),
                "label": random.choice(["A", "B", "C", "D"]),
                "weight": random.gauss(1.0, 0.3),
            },
            successors=[],
            is_terminal=random.random() < 0.1,
            depth=depth,
        )


@dataclass
class Transition:
    """Transition between states."""
    source: str
    target: str
    label: str
    transition_type: TransitionType
    probability: float = 1.0

    @classmethod
    def random(cls, source: str, targets: List[str]) -> "Transition":
        """Create random transition."""
        return cls(
            source=source,
            target=random.choice(targets) if targets else source,
            label=random.choice(["alpha", "beta", "gamma", "delta"]),
            transition_type=random.choice(list(TransitionType)),
            probability=random.random(),
        )


@dataclass
class DiTSSpec:
    """Specification for a Diegetic Transition System."""
    states: Dict[str, DiTSState]
    transitions: List[Transition]
    initial_state: str
    accepting_states: Set[str]

    @classmethod
    def random(cls, n_states: int, n_transitions: int) -> "DiTSSpec":
        """Create random DiTS specification."""
        states = {}
        for i in range(n_states):
            state = DiTSState.random(f"s{i}", depth=i // 5)
            states[state.id] = state

        state_ids = list(states.keys())
        initial = state_ids[0] if state_ids else "s0"
        accepting = set(random.sample(state_ids, min(3, len(state_ids))))

        transitions = []
        for _ in range(n_transitions):
            source = random.choice(state_ids) if state_ids else "s0"
            trans = Transition.random(source, state_ids)
            transitions.append(trans)

            # Add successor reference
            if source in states and trans.target not in states[source].successors:
                states[source].successors.append(trans.target)

        return cls(
            states=states,
            transitions=transitions,
            initial_state=initial,
            accepting_states=accepting,
        )


class DiTSKernel:
    """Kernel for evaluating Diegetic Transition Systems."""

    def __init__(self, spec: DiTSSpec, max_depth: int = 100):
        self.spec = spec
        self.max_depth = max_depth
        self._cache: Dict[str, Any] = {}

    def evaluate_mu(self, state_id: str, formula: str, depth: int = 0) -> bool:
        """
        Evaluate mu-calculus formula (least fixpoint).

        Mu computes finite unfoldings - what must eventually happen.
        """
        cache_key = f"mu:{state_id}:{formula}:{depth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if depth >= self.max_depth:
            result = False  # Mu returns false at limit (conservative)
        elif state_id not in self.spec.states:
            result = False
        else:
            state = self.spec.states[state_id]

            # Simplified evaluation
            if formula.startswith("prop:"):
                prop = formula[5:]
                result = prop in state.properties
            elif formula.startswith("reach:"):
                target = formula[6:]
                result = self._can_reach(state_id, target, depth)
            else:
                # Default: check if accepting
                result = state_id in self.spec.accepting_states

        self._cache[cache_key] = result
        return result

    def evaluate_nu(self, state_id: str, formula: str, depth: int = 0) -> bool:
        """
        Evaluate nu-calculus formula (greatest fixpoint).

        Nu computes infinite unfoldings - what may always happen.
        """
        cache_key = f"nu:{state_id}:{formula}:{depth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if depth >= self.max_depth:
            result = True  # Nu returns true at limit (optimistic)
        elif state_id not in self.spec.states:
            result = True
        else:
            state = self.spec.states[state_id]

            # Simplified evaluation
            if formula.startswith("safe:"):
                prop = formula[5:]
                result = self._always_safe(state_id, prop, depth)
            elif formula.startswith("invariant:"):
                target = formula[10:]
                result = self._is_invariant(state_id, target, depth)
            else:
                # Default: check if not terminal
                result = not state.is_terminal

        self._cache[cache_key] = result
        return result

    def _can_reach(self, from_state: str, to_state: str, depth: int) -> bool:
        """Check if target is reachable."""
        if from_state == to_state:
            return True
        if depth >= self.max_depth:
            return False

        state = self.spec.states.get(from_state)
        if not state:
            return False

        for succ in state.successors:
            if self._can_reach(succ, to_state, depth + 1):
                return True

        return False

    def _always_safe(self, state_id: str, prop: str, depth: int) -> bool:
        """Check safety property holds."""
        if depth >= self.max_depth:
            return True

        state = self.spec.states.get(state_id)
        if not state:
            return True

        # Check property holds here
        if prop in state.properties and not state.properties[prop]:
            return False

        # Check successors
        for succ in state.successors:
            if not self._always_safe(succ, prop, depth + 1):
                return False

        return True

    def _is_invariant(self, state_id: str, target: str, depth: int) -> bool:
        """Check invariant property."""
        return self._always_safe(state_id, target, depth)

    def step(self, state_id: str) -> List[tuple[str, Transition]]:
        """Take one step from current state."""
        results = []

        for trans in self.spec.transitions:
            if trans.source == state_id:
                if random.random() < trans.probability:
                    results.append((trans.target, trans))

        return results

    def run(self, max_steps: int = 100) -> List[str]:
        """Run simulation from initial state."""
        path = [self.spec.initial_state]
        current = self.spec.initial_state

        for _ in range(max_steps):
            if current in self.spec.accepting_states:
                break

            successors = self.step(current)
            if not successors:
                break

            current, _ = random.choice(successors)
            path.append(current)

        return path


@dataclass
class Episode:
    """Narrative episode."""
    id: str
    content: str
    mood: str
    characters: List[str]
    transitions: List[str]

    @classmethod
    def random(cls, id_prefix: str = "ep") -> "Episode":
        """Create random episode."""
        moods = ["tense", "calm", "exciting", "mysterious", "dramatic"]
        characters = ["Alice", "Bob", "Charlie", "Dana", "Eve"]

        return cls(
            id=f"{id_prefix}_{random.randint(0, 1000)}",
            content=" ".join(random.choices(
                ["The", "hero", "found", "a", "secret", "path", "through", "darkness"],
                k=random.randint(10, 30),
            )),
            mood=random.choice(moods),
            characters=random.sample(characters, random.randint(1, 3)),
            transitions=[],
        )


@dataclass
class Narrative:
    """Complete narrative structure."""
    episodes: List[Episode]
    arc: str
    theme: str

    @classmethod
    def random(cls, n_episodes: int) -> "Narrative":
        """Create random narrative."""
        arcs = ["hero_journey", "tragedy", "comedy", "mystery", "romance"]
        themes = ["redemption", "discovery", "loss", "growth", "conflict"]

        return cls(
            episodes=[Episode.random(f"ep{i}") for i in range(n_episodes)],
            arc=random.choice(arcs),
            theme=random.choice(themes),
        )


class NarrativeEngine:
    """Engine for generating narratives from DiTS."""

    def __init__(self, kernel: DiTSKernel):
        self.kernel = kernel
        self._templates = [
            "In {state}, the {character} discovered {prop}.",
            "Transitioning from {state} via {label}, everything changed.",
            "The {mood} atmosphere of {state} set the scene.",
        ]

    def generate_episode(self, state: DiTSState, prev_state: Optional[DiTSState] = None) -> Episode:
        """Generate episode from state."""
        mood = state.properties.get("label", "neutral")
        content = random.choice(self._templates).format(
            state=state.id,
            character="protagonist",
            prop=list(state.properties.keys())[0] if state.properties else "nothing",
            label="transition",
            mood=mood,
        )

        return Episode(
            id=f"ep_{state.id}",
            content=content,
            mood=mood,
            characters=["protagonist"],
            transitions=state.successors[:3],
        )

    def generate_narrative(self, max_episodes: int = 10) -> Narrative:
        """Generate complete narrative from DiTS execution."""
        path = self.kernel.run(max_steps=max_episodes)

        episodes = []
        prev_state = None

        for state_id in path:
            state = self.kernel.spec.states.get(state_id)
            if state:
                episode = self.generate_episode(state, prev_state)
                episodes.append(episode)
                prev_state = state

        return Narrative(
            episodes=episodes,
            arc="generated",
            theme="emergence",
        )

    def evaluate_coherence(self, narrative: Narrative) -> float:
        """Evaluate narrative coherence score."""
        if not narrative.episodes:
            return 0.0

        # Simplified coherence: check mood transitions
        score = 0.0
        for i in range(len(narrative.episodes) - 1):
            curr = narrative.episodes[i]
            next_ep = narrative.episodes[i + 1]

            # Same mood = coherent
            if curr.mood == next_ep.mood:
                score += 1.0
            # Related moods
            elif (curr.mood, next_ep.mood) in [("tense", "exciting"), ("calm", "mysterious")]:
                score += 0.5

        return score / max(1, len(narrative.episodes) - 1)


class DiTSBenchmark(BenchmarkSuite):
    """Benchmark suite for DiTS subsystem."""

    @property
    def name(self) -> str:
        return "dits"

    @property
    def description(self) -> str:
        return "DiTS kernel evaluation and narrative generation benchmarks"

    def __init__(self):
        self._specs: Dict[str, DiTSSpec] = {}
        self._kernels: Dict[str, DiTSKernel] = {}
        self._engines: Dict[str, NarrativeEngine] = {}
        self._narratives: List[Narrative] = []

    def setup(self) -> None:
        """Setup test data."""
        # Pre-generate DiTS specifications
        self._specs = {
            "tiny": DiTSSpec.random(10, 20),
            "small": DiTSSpec.random(50, 100),
            "medium": DiTSSpec.random(200, 500),
            "large": DiTSSpec.random(1000, 2000),
        }

        # Create kernels
        self._kernels = {
            name: DiTSKernel(spec)
            for name, spec in self._specs.items()
        }

        # Create engines
        self._engines = {
            name: NarrativeEngine(kernel)
            for name, kernel in self._kernels.items()
        }

        # Pre-generate narratives
        self._narratives = [Narrative.random(10) for _ in range(5)]

    def get_benchmarks(self) -> list[tuple[str, Callable[[], Any]]]:
        """Get all DiTS benchmarks."""
        return [
            # Spec creation
            ("spec_create_tiny", self._spec_create_tiny),
            ("spec_create_small", self._spec_create_small),
            ("spec_create_medium", self._spec_create_medium),

            # State/Transition creation
            ("state_create", self._state_create),
            ("transition_create", self._transition_create),

            # Mu-calculus evaluation
            ("mu_eval_simple", self._mu_eval_simple),
            ("mu_eval_reach", self._mu_eval_reach),
            ("mu_eval_deep", self._mu_eval_deep),

            # Nu-calculus evaluation
            ("nu_eval_simple", self._nu_eval_simple),
            ("nu_eval_safe", self._nu_eval_safe),
            ("nu_eval_deep", self._nu_eval_deep),

            # Kernel operations
            ("kernel_step", self._kernel_step),
            ("kernel_run_short", self._kernel_run_short),
            ("kernel_run_long", self._kernel_run_long),

            # Narrative generation
            ("narrative_episode", self._narrative_episode),
            ("narrative_generate_short", self._narrative_generate_short),
            ("narrative_generate_long", self._narrative_generate_long),
            ("narrative_coherence", self._narrative_coherence),
        ]

    def _spec_create_tiny(self) -> DiTSSpec:
        """Create tiny DiTS spec."""
        return DiTSSpec.random(10, 20)

    def _spec_create_small(self) -> DiTSSpec:
        """Create small DiTS spec."""
        return DiTSSpec.random(50, 100)

    def _spec_create_medium(self) -> DiTSSpec:
        """Create medium DiTS spec."""
        return DiTSSpec.random(200, 500)

    def _state_create(self) -> DiTSState:
        """Create single state."""
        return DiTSState.random()

    def _transition_create(self) -> Transition:
        """Create single transition."""
        return Transition.random("s0", ["s1", "s2", "s3"])

    def _mu_eval_simple(self) -> bool:
        """Evaluate simple mu formula."""
        kernel = self._kernels["small"]
        initial = kernel.spec.initial_state
        return kernel.evaluate_mu(initial, "prop:value")

    def _mu_eval_reach(self) -> bool:
        """Evaluate mu reachability."""
        kernel = self._kernels["small"]
        initial = kernel.spec.initial_state
        accepting = list(kernel.spec.accepting_states)[0] if kernel.spec.accepting_states else initial
        return kernel.evaluate_mu(initial, f"reach:{accepting}")

    def _mu_eval_deep(self) -> bool:
        """Evaluate deep mu formula."""
        kernel = self._kernels["medium"]
        kernel.max_depth = 50
        initial = kernel.spec.initial_state
        return kernel.evaluate_mu(initial, "prop:label", depth=0)

    def _nu_eval_simple(self) -> bool:
        """Evaluate simple nu formula."""
        kernel = self._kernels["small"]
        initial = kernel.spec.initial_state
        return kernel.evaluate_nu(initial, "safe:value")

    def _nu_eval_safe(self) -> bool:
        """Evaluate nu safety property."""
        kernel = self._kernels["small"]
        initial = kernel.spec.initial_state
        return kernel.evaluate_nu(initial, "safe:label")

    def _nu_eval_deep(self) -> bool:
        """Evaluate deep nu formula."""
        kernel = self._kernels["medium"]
        kernel.max_depth = 50
        initial = kernel.spec.initial_state
        return kernel.evaluate_nu(initial, "invariant:weight", depth=0)

    def _kernel_step(self) -> List[tuple[str, Transition]]:
        """Take single kernel step."""
        kernel = self._kernels["small"]
        return kernel.step(kernel.spec.initial_state)

    def _kernel_run_short(self) -> List[str]:
        """Run kernel for 10 steps."""
        kernel = self._kernels["small"]
        return kernel.run(max_steps=10)

    def _kernel_run_long(self) -> List[str]:
        """Run kernel for 100 steps."""
        kernel = self._kernels["medium"]
        return kernel.run(max_steps=100)

    def _narrative_episode(self) -> Episode:
        """Generate single episode."""
        engine = self._engines["small"]
        state = list(engine.kernel.spec.states.values())[0]
        return engine.generate_episode(state)

    def _narrative_generate_short(self) -> Narrative:
        """Generate short narrative."""
        engine = self._engines["small"]
        return engine.generate_narrative(max_episodes=5)

    def _narrative_generate_long(self) -> Narrative:
        """Generate long narrative."""
        engine = self._engines["medium"]
        return engine.generate_narrative(max_episodes=50)

    def _narrative_coherence(self) -> List[float]:
        """Evaluate coherence of narratives."""
        engine = self._engines["small"]
        return [engine.evaluate_coherence(n) for n in self._narratives]
