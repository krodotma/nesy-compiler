"""
DiTS Rheomode Engine
====================

Implements David Bohm's Rheomode concept for language and thought:
- Language as flowing movement (rheo = flow)
- Verbs as primary, nouns as secondary
- Thought as process, not static entities

The rheomode provides a way to express narrative transitions that
emphasizes becoming over being, process over product.

Key concepts:
- Verb-first language structure
- Levate roots (le-vidate, re-levate, ir-relevate)
- Flowing state rather than fixed entities
- Reflexive self-observation

Reference: David Bohm, "Wholeness and the Implicate Order" (1980)
"""

from __future__ import annotations

import time
import uuid
import re
from typing import (
    Dict,
    Any,
    List,
    Optional,
    Set,
    Callable,
    Iterator,
    Tuple,
    Union,
)
from dataclasses import dataclass, field
from enum import Enum, auto


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class VerbMode(str, Enum):
    """Mode of verb in rheomode."""
    LEVATE = "levate"           # Base form (to lift into attention)
    RELEVATE = "relevate"       # Re-lift (bring back attention)
    IRRELEVATE = "irrelevate"   # Not-lift (allow to subside)
    INDICATE = "indicate"       # Point to relationship
    VIDATE = "vidate"           # See/perceive
    DIVIDATE = "dividate"       # See as separate


class FlowState(str, Enum):
    """State of flow in rheomode processing."""
    DORMANT = "dormant"         # Not actively flowing
    EMERGING = "emerging"       # Beginning to flow
    FLOWING = "flowing"         # Active flow
    TURBULENT = "turbulent"     # Conflicting flows
    SUBSIDING = "subsiding"     # Flow diminishing
    REFLECTING = "reflecting"   # Self-observing flow


class AttentionLevel(str, Enum):
    """Level of attention in rheomode."""
    PERIPHERAL = "peripheral"   # Background awareness
    EMERGING = "emerging"       # Coming into focus
    FOCAL = "focal"             # Central attention
    ABSORBED = "absorbed"       # Deep engagement
    TRANSCENDENT = "transcendent"  # Beyond normal attention


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class VerbInfo:
    """
    Information about a verb in rheomode.

    In rheomode, verbs are primary and carry the essential meaning.
    This structure captures the full semantic content of a verb.

    Attributes:
        root: The verb root (e.g., "levate", "vidate")
        mode: The verb mode (levate, relevate, irrelevate)
        tense: Temporal aspect (though rheomode de-emphasizes static time)
        aspect: Process aspect (beginning, continuing, completing)
        agent: Who/what performs the action
        patient: Who/what receives the action
        manner: How the action occurs
        context: Surrounding context
        intensity: Strength of the verb action (0.0 to 1.0)
        reflexive: Whether the verb is self-referential
        metadata: Additional verb data
    """
    root: str
    mode: VerbMode = VerbMode.LEVATE
    tense: str = "present"
    aspect: str = "continuous"
    agent: Optional[str] = None
    patient: Optional[str] = None
    manner: str = ""
    context: str = ""
    intensity: float = 0.5
    reflexive: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_form(self) -> str:
        """Get the full verb form."""
        prefix = ""
        if self.mode == VerbMode.RELEVATE:
            prefix = "re-"
        elif self.mode == VerbMode.IRRELEVATE:
            prefix = "ir-"

        return f"{prefix}{self.root}"

    def to_phrase(self) -> str:
        """Convert verb to natural language phrase."""
        parts = []

        if self.agent:
            parts.append(self.agent)

        # Conjugate based on tense/aspect
        verb = self.full_form
        if self.tense == "past":
            verb = f"has {verb}d"
        elif self.aspect == "beginning":
            verb = f"begins to {verb}"
        elif self.aspect == "completing":
            verb = f"completes {verb}ing"
        else:
            verb = f"{verb}s"

        parts.append(verb)

        if self.patient:
            parts.append(self.patient)

        if self.manner:
            parts.append(self.manner)

        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "root": self.root,
            "mode": self.mode.value,
            "tense": self.tense,
            "aspect": self.aspect,
            "agent": self.agent,
            "patient": self.patient,
            "manner": self.manner,
            "context": self.context,
            "intensity": self.intensity,
            "reflexive": self.reflexive,
            "full_form": self.full_form,
            "metadata": self.metadata,
        }


@dataclass
class FlowElement:
    """
    A single element in a rheomode flow.

    Represents one moment or beat in the flowing thought process.
    """
    id: str
    content: str
    verb: Optional[VerbInfo] = None
    attention_level: AttentionLevel = AttentionLevel.EMERGING
    coherence: float = 1.0  # How well this fits the flow
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RheomodeFlow:
    """
    A complete flow in rheomode.

    Represents a flowing stream of thought/language following
    Bohm's rheomode principles.

    Attributes:
        id: Unique flow identifier
        title: Optional title for the flow
        state: Current flow state
        elements: List of flow elements
        dominant_verb: The primary verb driving this flow
        coherence_score: Overall coherence (0.0 to 1.0)
        attention_focus: What the flow is attending to
        emergence_patterns: Patterns emerging from the flow
        reflexive_depth: How self-referential the flow is
        metadata: Additional flow data
        created_at: Timestamp of creation
    """
    id: str
    title: str = ""
    state: FlowState = FlowState.DORMANT
    elements: List[FlowElement] = field(default_factory=list)
    dominant_verb: Optional[VerbInfo] = None
    coherence_score: float = 1.0
    attention_focus: str = ""
    emergence_patterns: List[str] = field(default_factory=list)
    reflexive_depth: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, title: str = "") -> "RheomodeFlow":
        """Factory method to create a flow."""
        return cls(
            id=f"flow_{uuid.uuid4().hex[:12]}",
            title=title,
        )

    def add_element(
        self,
        content: str,
        verb: Optional[VerbInfo] = None,
        attention: AttentionLevel = AttentionLevel.EMERGING,
    ) -> FlowElement:
        """Add an element to the flow."""
        element = FlowElement(
            id=f"elem_{uuid.uuid4().hex[:8]}",
            content=content,
            verb=verb,
            attention_level=attention,
        )
        self.elements.append(element)
        self._update_flow_state()
        return element

    def _update_flow_state(self) -> None:
        """Update flow state based on elements."""
        if not self.elements:
            self.state = FlowState.DORMANT
            return

        # Determine state based on recent elements
        recent = self.elements[-3:] if len(self.elements) >= 3 else self.elements

        avg_coherence = sum(e.coherence for e in recent) / len(recent)

        if avg_coherence < 0.3:
            self.state = FlowState.TURBULENT
        elif avg_coherence < 0.6:
            self.state = FlowState.EMERGING
        elif len(self.elements) > 5 and avg_coherence > 0.8:
            self.state = FlowState.FLOWING
        else:
            self.state = FlowState.EMERGING

        # Check for reflexivity
        reflexive_count = sum(
            1 for e in self.elements
            if e.verb and e.verb.reflexive
        )
        self.reflexive_depth = reflexive_count

        if reflexive_count > len(self.elements) * 0.3:
            self.state = FlowState.REFLECTING

        self.coherence_score = avg_coherence

    @property
    def duration(self) -> float:
        """Get flow duration in seconds."""
        if not self.elements:
            return 0.0
        return self.elements[-1].timestamp - self.elements[0].timestamp

    @property
    def is_active(self) -> bool:
        """Check if flow is currently active."""
        return self.state in [
            FlowState.EMERGING,
            FlowState.FLOWING,
            FlowState.TURBULENT,
        ]

    def get_verb_sequence(self) -> List[VerbInfo]:
        """Get sequence of verbs in this flow."""
        return [e.verb for e in self.elements if e.verb]

    def to_text(self) -> str:
        """Convert flow to natural text."""
        return " ".join(e.content for e in self.elements)

    def to_dict(self) -> Dict[str, Any]:
        """Convert flow to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "state": self.state.value,
            "elements": [
                {
                    "id": e.id,
                    "content": e.content,
                    "verb": e.verb.to_dict() if e.verb else None,
                    "attention_level": e.attention_level.value,
                    "coherence": e.coherence,
                    "timestamp": e.timestamp,
                }
                for e in self.elements
            ],
            "dominant_verb": self.dominant_verb.to_dict() if self.dominant_verb else None,
            "coherence_score": self.coherence_score,
            "attention_focus": self.attention_focus,
            "emergence_patterns": self.emergence_patterns,
            "reflexive_depth": self.reflexive_depth,
            "duration": self.duration,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


# =============================================================================
# RHEOMODE ENGINE
# =============================================================================

@dataclass
class RheomodeConfig:
    """Configuration for rheomode processing."""
    min_coherence: float = 0.3
    max_turbulence_duration: float = 60.0
    reflexivity_threshold: float = 0.4
    attention_decay_rate: float = 0.1
    emergence_threshold: int = 3
    enable_self_observation: bool = True


class RheomodeEngine:
    """
    Engine for processing and generating rheomode language flows.

    Implements Bohm's rheomode concept with support for:
    - Verb-primary language construction
    - Flow state management
    - Coherence tracking
    - Reflexive self-observation
    - Pattern emergence detection

    Example usage:
        >>> engine = RheomodeEngine()
        >>> flow = engine.create_flow("Contemplation")
        >>> engine.add_to_flow(flow, "The thought begins to emerge")
        >>> engine.levate(flow, "understanding")
        >>> print(flow.state)
    """

    # Standard rheomode verb roots
    VERB_ROOTS = {
        "levate": "to lift into attention",
        "vidate": "to see/perceive",
        "dividate": "to see as separate",
        "indicate": "to point to relationship",
        "ordinate": "to order/arrange",
        "terminate": "to end/complete",
        "generate": "to bring forth",
        "relate": "to connect",
        "create": "to bring into being",
        "integrate": "to bring together",
    }

    # Attention level thresholds
    ATTENTION_THRESHOLDS = {
        AttentionLevel.PERIPHERAL: 0.2,
        AttentionLevel.EMERGING: 0.4,
        AttentionLevel.FOCAL: 0.6,
        AttentionLevel.ABSORBED: 0.8,
        AttentionLevel.TRANSCENDENT: 0.95,
    }

    def __init__(self, config: Optional[RheomodeConfig] = None):
        """
        Initialize RheomodeEngine.

        Args:
            config: Optional configuration
        """
        self.config = config or RheomodeConfig()
        self._flows: Dict[str, RheomodeFlow] = {}
        self._active_flow_id: Optional[str] = None
        self._observation_stack: List[str] = []
        self._stats = {
            "flows_created": 0,
            "elements_added": 0,
            "verb_operations": 0,
            "reflexive_events": 0,
        }

    def create_flow(self, title: str = "") -> RheomodeFlow:
        """
        Create a new rheomode flow.

        Args:
            title: Optional title for the flow

        Returns:
            New RheomodeFlow instance
        """
        flow = RheomodeFlow.create(title)
        self._flows[flow.id] = flow
        self._active_flow_id = flow.id
        self._stats["flows_created"] += 1
        return flow

    def get_flow(self, flow_id: str) -> Optional[RheomodeFlow]:
        """Get a flow by ID."""
        return self._flows.get(flow_id)

    @property
    def active_flow(self) -> Optional[RheomodeFlow]:
        """Get the currently active flow."""
        if self._active_flow_id:
            return self._flows.get(self._active_flow_id)
        return None

    def add_to_flow(
        self,
        flow: RheomodeFlow,
        content: str,
        verb: Optional[VerbInfo] = None,
    ) -> FlowElement:
        """
        Add content to a flow.

        Args:
            flow: The flow to add to
            content: Text content to add
            verb: Optional verb info

        Returns:
            The created FlowElement
        """
        # Detect attention level from content
        attention = self._detect_attention_level(content, flow)

        # Auto-detect verb if not provided
        if verb is None:
            verb = self._extract_verb(content)

        element = flow.add_element(content, verb, attention)
        self._stats["elements_added"] += 1

        # Update coherence
        element.coherence = self._calculate_coherence(element, flow)

        # Check for emergence patterns
        self._detect_emergence(flow)

        return element

    def levate(
        self,
        flow: RheomodeFlow,
        focus: str,
        intensity: float = 0.5,
    ) -> FlowElement:
        """
        Levate: Lift something into attention.

        The primary rheomode operation - bringing something
        into the focus of flowing thought.

        Args:
            flow: The flow to operate on
            focus: What to bring into attention
            intensity: How strongly to levate (0.0 to 1.0)

        Returns:
            The created FlowElement
        """
        verb = VerbInfo(
            root="levate",
            mode=VerbMode.LEVATE,
            patient=focus,
            intensity=intensity,
            aspect="beginning" if flow.state == FlowState.DORMANT else "continuing",
        )

        content = f"Attention lifts to {focus}."
        flow.attention_focus = focus

        self._stats["verb_operations"] += 1
        return self.add_to_flow(flow, content, verb)

    def relevate(
        self,
        flow: RheomodeFlow,
        focus: str,
        intensity: float = 0.7,
    ) -> FlowElement:
        """
        Relevate: Re-lift something back into attention.

        Returns something previously attended to back into focus.

        Args:
            flow: The flow to operate on
            focus: What to bring back into attention
            intensity: How strongly to relevate

        Returns:
            The created FlowElement
        """
        verb = VerbInfo(
            root="levate",
            mode=VerbMode.RELEVATE,
            patient=focus,
            intensity=intensity,
            aspect="returning",
        )

        content = f"Attention returns to {focus}."
        flow.attention_focus = focus

        self._stats["verb_operations"] += 1
        return self.add_to_flow(flow, content, verb)

    def irrelevate(
        self,
        flow: RheomodeFlow,
        focus: str,
    ) -> FlowElement:
        """
        Irrelevate: Allow something to subside from attention.

        Lets something fade from focal attention.

        Args:
            flow: The flow to operate on
            focus: What to allow to subside

        Returns:
            The created FlowElement
        """
        verb = VerbInfo(
            root="levate",
            mode=VerbMode.IRRELEVATE,
            patient=focus,
            intensity=0.2,
            aspect="completing",
        )

        content = f"{focus} subsides from attention."

        if flow.attention_focus == focus:
            flow.attention_focus = ""

        self._stats["verb_operations"] += 1
        return self.add_to_flow(flow, content, verb)

    def vidate(
        self,
        flow: RheomodeFlow,
        what: str,
        how: str = "",
    ) -> FlowElement:
        """
        Vidate: See/perceive something.

        The act of perception in flowing thought.

        Args:
            flow: The flow to operate on
            what: What is being perceived
            how: Optional manner of perception

        Returns:
            The created FlowElement
        """
        verb = VerbInfo(
            root="vidate",
            mode=VerbMode.VIDATE,
            patient=what,
            manner=how,
            intensity=0.6,
        )

        content = f"Perceiving {what}"
        if how:
            content += f" {how}"
        content += "."

        self._stats["verb_operations"] += 1
        return self.add_to_flow(flow, content, verb)

    def reflect(
        self,
        flow: RheomodeFlow,
        depth: int = 1,
    ) -> FlowElement:
        """
        Create a reflexive observation of the flow itself.

        The flow observes its own flowing - a key rheomode concept.

        Args:
            flow: The flow to reflect on
            depth: Depth of reflection (1 = flow observes self,
                   2 = flow observes observation, etc.)

        Returns:
            The created FlowElement
        """
        if not self.config.enable_self_observation:
            # Add neutral element instead
            return self.add_to_flow(
                flow,
                "The flow continues.",
            )

        # Build reflexive description
        observer = "The flow"
        for _ in range(depth - 1):
            observer = f"the observation of {observer}"

        observed = self._describe_flow_state(flow)

        verb = VerbInfo(
            root="vidate",
            mode=VerbMode.VIDATE,
            agent=observer,
            patient=f"its own {observed}",
            reflexive=True,
            intensity=0.8,
        )

        content = f"{observer.capitalize()} observes its own {observed}."

        # Push to observation stack
        self._observation_stack.append(flow.id)
        flow.reflexive_depth = max(flow.reflexive_depth, depth)

        self._stats["reflexive_events"] += 1
        self._stats["verb_operations"] += 1

        return self.add_to_flow(flow, content, verb)

    def _detect_attention_level(
        self,
        content: str,
        flow: RheomodeFlow,
    ) -> AttentionLevel:
        """Detect attention level from content and context."""
        # Keywords suggesting different attention levels
        focal_keywords = ["focus", "attention", "central", "important", "key"]
        absorbed_keywords = ["deeply", "fully", "completely", "absorbed", "immersed"]
        peripheral_keywords = ["background", "aside", "peripheral", "meanwhile"]

        content_lower = content.lower()

        for word in absorbed_keywords:
            if word in content_lower:
                return AttentionLevel.ABSORBED

        for word in focal_keywords:
            if word in content_lower:
                return AttentionLevel.FOCAL

        for word in peripheral_keywords:
            if word in content_lower:
                return AttentionLevel.PERIPHERAL

        # Default based on flow state
        if flow.state == FlowState.FLOWING:
            return AttentionLevel.FOCAL
        elif flow.state == FlowState.DORMANT:
            return AttentionLevel.EMERGING
        else:
            return AttentionLevel.EMERGING

    def _extract_verb(self, content: str) -> Optional[VerbInfo]:
        """Extract verb information from content."""
        content_lower = content.lower()

        # Check for rheomode verb roots
        for root, meaning in self.VERB_ROOTS.items():
            if root in content_lower:
                # Determine mode
                mode = VerbMode.LEVATE
                if f"re-{root}" in content_lower or f"re{root}" in content_lower:
                    mode = VerbMode.RELEVATE
                elif f"ir-{root}" in content_lower or f"ir{root}" in content_lower:
                    mode = VerbMode.IRRELEVATE

                return VerbInfo(root=root, mode=mode)

        # Check for common verbs that map to rheomode
        verb_mappings = {
            "begins": ("levate", VerbMode.LEVATE),
            "starts": ("levate", VerbMode.LEVATE),
            "returns": ("levate", VerbMode.RELEVATE),
            "fades": ("levate", VerbMode.IRRELEVATE),
            "sees": ("vidate", VerbMode.VIDATE),
            "perceives": ("vidate", VerbMode.VIDATE),
            "separates": ("dividate", VerbMode.DIVIDATE),
            "points": ("indicate", VerbMode.INDICATE),
        }

        for word, (root, mode) in verb_mappings.items():
            if word in content_lower:
                return VerbInfo(root=root, mode=mode)

        return None

    def _calculate_coherence(
        self,
        element: FlowElement,
        flow: RheomodeFlow,
    ) -> float:
        """Calculate coherence of element with flow."""
        if not flow.elements:
            return 1.0

        coherence = 1.0

        # Check attention continuity
        if len(flow.elements) > 1:
            prev = flow.elements[-2]
            attention_diff = abs(
                self._attention_to_value(element.attention_level) -
                self._attention_to_value(prev.attention_level)
            )
            coherence -= attention_diff * 0.3

        # Check verb coherence
        if element.verb and flow.dominant_verb:
            if element.verb.root == flow.dominant_verb.root:
                coherence += 0.1
            elif element.verb.mode != flow.dominant_verb.mode:
                coherence -= 0.1

        # Check for topic continuity
        if flow.attention_focus and element.content:
            if flow.attention_focus.lower() in element.content.lower():
                coherence += 0.1

        return max(0.0, min(1.0, coherence))

    def _attention_to_value(self, level: AttentionLevel) -> float:
        """Convert attention level to numeric value."""
        return self.ATTENTION_THRESHOLDS.get(level, 0.5)

    def _describe_flow_state(self, flow: RheomodeFlow) -> str:
        """Generate a description of the flow's current state."""
        state_descriptions = {
            FlowState.DORMANT: "stillness",
            FlowState.EMERGING: "emergence",
            FlowState.FLOWING: "flowing",
            FlowState.TURBULENT: "turbulence",
            FlowState.SUBSIDING: "subsiding",
            FlowState.REFLECTING: "self-reflection",
        }
        return state_descriptions.get(flow.state, "state")

    def _detect_emergence(self, flow: RheomodeFlow) -> None:
        """Detect emergence patterns in the flow."""
        if len(flow.elements) < self.config.emergence_threshold:
            return

        # Look for repeated verb roots
        verbs = flow.get_verb_sequence()
        if len(verbs) >= 3:
            root_counts: Dict[str, int] = {}
            for v in verbs:
                root_counts[v.root] = root_counts.get(v.root, 0) + 1

            for root, count in root_counts.items():
                if count >= 3 and f"recurring_{root}" not in flow.emergence_patterns:
                    flow.emergence_patterns.append(f"recurring_{root}")

        # Look for attention patterns
        recent = flow.elements[-5:]
        attention_levels = [e.attention_level for e in recent]

        # Check for crescendo (increasing attention)
        values = [self._attention_to_value(a) for a in attention_levels]
        if all(values[i] <= values[i + 1] for i in range(len(values) - 1)):
            if "crescendo" not in flow.emergence_patterns:
                flow.emergence_patterns.append("crescendo")

        # Check for oscillation
        if len(values) >= 4:
            diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
            sign_changes = sum(
                1 for i in range(len(diffs) - 1)
                if diffs[i] * diffs[i + 1] < 0
            )
            if sign_changes >= 2 and "oscillation" not in flow.emergence_patterns:
                flow.emergence_patterns.append("oscillation")

    def analyze_flow(self, flow: RheomodeFlow) -> Dict[str, Any]:
        """
        Analyze a rheomode flow.

        Args:
            flow: The flow to analyze

        Returns:
            Analysis results
        """
        verbs = flow.get_verb_sequence()

        # Verb analysis
        verb_roots = [v.root for v in verbs]
        verb_modes = [v.mode.value for v in verbs]

        root_distribution = {}
        for root in verb_roots:
            root_distribution[root] = root_distribution.get(root, 0) + 1

        mode_distribution = {}
        for mode in verb_modes:
            mode_distribution[mode] = mode_distribution.get(mode, 0) + 1

        # Attention analysis
        attention_values = [
            self._attention_to_value(e.attention_level)
            for e in flow.elements
        ]

        # Coherence analysis
        coherence_values = [e.coherence for e in flow.elements]

        return {
            "id": flow.id,
            "title": flow.title,
            "state": flow.state.value,
            "element_count": len(flow.elements),
            "duration": flow.duration,
            "verb_analysis": {
                "total_verbs": len(verbs),
                "root_distribution": root_distribution,
                "mode_distribution": mode_distribution,
                "dominant_root": max(root_distribution, key=root_distribution.get)
                if root_distribution else None,
            },
            "attention_analysis": {
                "mean": sum(attention_values) / len(attention_values)
                if attention_values else 0,
                "max": max(attention_values) if attention_values else 0,
                "trajectory": attention_values,
            },
            "coherence_analysis": {
                "mean": sum(coherence_values) / len(coherence_values)
                if coherence_values else 0,
                "final": coherence_values[-1] if coherence_values else 1.0,
                "flow_coherence": flow.coherence_score,
            },
            "emergence_patterns": flow.emergence_patterns,
            "reflexive_depth": flow.reflexive_depth,
            "is_active": flow.is_active,
        }

    def merge_flows(
        self,
        flows: List[RheomodeFlow],
        mode: str = "sequential",
    ) -> RheomodeFlow:
        """
        Merge multiple flows into one.

        Args:
            flows: Flows to merge
            mode: "sequential" or "interleaved"

        Returns:
            Merged flow
        """
        if not flows:
            raise ValueError("No flows to merge")

        merged = self.create_flow(
            title=f"Merged: {', '.join(f.title or f.id for f in flows)}"
        )

        if mode == "sequential":
            for flow in flows:
                for element in flow.elements:
                    merged.elements.append(element)
        else:  # interleaved
            max_len = max(len(f.elements) for f in flows)
            for i in range(max_len):
                for flow in flows:
                    if i < len(flow.elements):
                        merged.elements.append(flow.elements[i])

        # Recalculate flow state
        merged._update_flow_state()

        # Combine emergence patterns
        for flow in flows:
            for pattern in flow.emergence_patterns:
                if pattern not in merged.emergence_patterns:
                    merged.emergence_patterns.append(pattern)

        return merged

    @property
    def statistics(self) -> Dict[str, int]:
        """Get engine statistics."""
        return self._stats.copy()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "VerbMode",
    "FlowState",
    "AttentionLevel",
    # Data structures
    "VerbInfo",
    "FlowElement",
    "RheomodeFlow",
    "RheomodeConfig",
    # Engine
    "RheomodeEngine",
]


if __name__ == "__main__":
    # Demo usage
    engine = RheomodeEngine()

    # Create a flow
    flow = engine.create_flow("Contemplation of Understanding")

    print(f"Created flow: {flow.title}")
    print(f"Initial state: {flow.state.value}")
    print()

    # Perform rheomode operations
    engine.levate(flow, "the question of meaning")
    print(f"After levate: {flow.state.value}")

    engine.vidate(flow, "patterns emerging", "with growing clarity")
    print(f"After vidate: {flow.state.value}")

    engine.add_to_flow(flow, "The understanding begins to cohere.")

    engine.relevate(flow, "earlier insights")
    print(f"After relevate: {flow.state.value}")

    engine.reflect(flow)
    print(f"After reflection: {flow.state.value}")

    engine.irrelevate(flow, "distracting thoughts")

    # Analyze the flow
    print()
    print("Flow Analysis:")
    analysis = engine.analyze_flow(flow)
    print(f"  Elements: {analysis['element_count']}")
    print(f"  Duration: {analysis['duration']:.2f}s")
    print(f"  Mean coherence: {analysis['coherence_analysis']['mean']:.2f}")
    print(f"  Emergence patterns: {analysis['emergence_patterns']}")
    print(f"  Reflexive depth: {analysis['reflexive_depth']}")

    print()
    print("Flow text:")
    print(flow.to_text())

    print()
    print(f"Engine statistics: {engine.statistics}")
