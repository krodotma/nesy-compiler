"""
DiTS Omega Bridge
=================

Bridge between the DiTS subsystem and higher-level Omega (reflexive) systems.

The Omega layer represents the highest level of the Pluribus architecture:
- L5: Omega Reflexive Domain (from Theia architecture)
- Self-observation and meta-cognition
- Integration of multiple subsystems
- Emergence detection and response

This bridge allows DiTS narratives and transitions to connect to:
- Metalearning systems (omega_metalearning.py)
- Reflexive monitors
- Cross-system coherence checking
- Higher-order narrative reasoning

Key concepts:
- Omega State: Higher-level cognitive state
- Reflexive loops: Self-observation capabilities
- Emergence detection: Recognizing novel patterns
- Integration: Connecting narrative to meaning
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
    Iterator,
    Tuple,
    Union,
    Protocol,
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import json


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class OmegaLevel(str, Enum):
    """Levels of omega processing."""
    L0_GROUND = "ground"          # Base reality/data level
    L1_PATTERN = "pattern"        # Pattern recognition
    L2_MEANING = "meaning"        # Semantic interpretation
    L3_CONTEXT = "context"        # Contextual reasoning
    L4_REFLECTION = "reflection"  # Self-observation
    L5_OMEGA = "omega"            # Full reflexive integration


class IntegrationMode(str, Enum):
    """Mode of cross-system integration."""
    PASSIVE = "passive"       # Observe only
    REACTIVE = "reactive"     # Respond to events
    PROACTIVE = "proactive"   # Anticipate and prepare
    GENERATIVE = "generative" # Create new structures


class CoherenceStatus(str, Enum):
    """Status of system coherence."""
    COHERENT = "coherent"
    PARTIAL = "partial"
    FRAGMENTED = "fragmented"
    CONFLICTED = "conflicted"
    EMERGENT = "emergent"


class OmegaEventType(str, Enum):
    """Types of omega-level events."""
    STATE_CHANGE = "state_change"
    EMERGENCE_DETECTED = "emergence_detected"
    COHERENCE_SHIFT = "coherence_shift"
    REFLECTION_TRIGGERED = "reflection_triggered"
    INTEGRATION_COMPLETE = "integration_complete"
    ANOMALY_DETECTED = "anomaly_detected"


# =============================================================================
# PROTOCOL DEFINITIONS
# =============================================================================

class OmegaObserver(Protocol):
    """Protocol for observers of omega events."""

    def on_omega_event(self, event: "OmegaEvent") -> None:
        """Handle an omega event."""
        ...


class Integrable(Protocol):
    """Protocol for systems that can be integrated at the omega level."""

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of current state."""
        ...

    def get_coherence_metrics(self) -> Dict[str, float]:
        """Get coherence metrics."""
        ...


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class OmegaEvent:
    """An event at the omega level."""
    id: str
    type: OmegaEventType
    source: str
    level: OmegaLevel
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event_type: OmegaEventType,
        source: str,
        level: OmegaLevel = OmegaLevel.L4_REFLECTION,
        **data,
    ) -> "OmegaEvent":
        """Factory method to create an event."""
        return cls(
            id=f"omega_{uuid.uuid4().hex[:12]}",
            type=event_type,
            source=source,
            level=level,
            timestamp=time.time(),
            data=data,
        )


@dataclass
class ReflexiveLoop:
    """A reflexive observation loop."""
    id: str
    depth: int
    observer: str
    observed: str
    observation: str
    timestamp: float = field(default_factory=time.time)
    stable: bool = True

    @property
    def is_self_referential(self) -> bool:
        """Check if this is a true self-reference."""
        return self.observer == self.observed


@dataclass
class EmergenceSignal:
    """Signal indicating emergence of novel patterns."""
    id: str
    pattern_type: str
    description: str
    confidence: float
    source_systems: List[str]
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OmegaState:
    """
    State of the omega-level system.

    Represents the highest-level cognitive state integrating
    all subsystems and maintaining reflexive awareness.

    Attributes:
        id: Unique state identifier
        level: Current omega processing level
        coherence_status: Overall coherence status
        coherence_score: Numeric coherence (0.0 to 1.0)
        active_integrations: Currently active integration processes
        reflexive_loops: Active self-observation loops
        emergence_signals: Detected emergence patterns
        attention_focus: Current high-level attention
        integration_mode: How subsystems are being integrated
        subsystem_states: Summaries from integrated subsystems
        metadata: Additional state data
        created_at: Timestamp of creation
        updated_at: Timestamp of last update
    """
    id: str
    level: OmegaLevel = OmegaLevel.L1_PATTERN
    coherence_status: CoherenceStatus = CoherenceStatus.COHERENT
    coherence_score: float = 1.0
    active_integrations: List[str] = field(default_factory=list)
    reflexive_loops: List[ReflexiveLoop] = field(default_factory=list)
    emergence_signals: List[EmergenceSignal] = field(default_factory=list)
    attention_focus: str = ""
    integration_mode: IntegrationMode = IntegrationMode.PASSIVE
    subsystem_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @classmethod
    def create(cls, level: OmegaLevel = OmegaLevel.L1_PATTERN) -> "OmegaState":
        """Factory method to create omega state."""
        return cls(
            id=f"omega_state_{uuid.uuid4().hex[:12]}",
            level=level,
        )

    def update(self) -> None:
        """Mark state as updated."""
        self.updated_at = time.time()

    def add_reflexive_loop(
        self,
        observer: str,
        observed: str,
        observation: str,
        depth: int = 1,
    ) -> ReflexiveLoop:
        """Add a reflexive observation loop."""
        loop = ReflexiveLoop(
            id=f"loop_{uuid.uuid4().hex[:8]}",
            depth=depth,
            observer=observer,
            observed=observed,
            observation=observation,
        )
        self.reflexive_loops.append(loop)
        self.update()
        return loop

    def add_emergence_signal(
        self,
        pattern_type: str,
        description: str,
        confidence: float,
        sources: List[str],
    ) -> EmergenceSignal:
        """Add an emergence signal."""
        signal = EmergenceSignal(
            id=f"emerge_{uuid.uuid4().hex[:8]}",
            pattern_type=pattern_type,
            description=description,
            confidence=confidence,
            source_systems=sources,
        )
        self.emergence_signals.append(signal)
        self.update()
        return signal

    @property
    def reflexive_depth(self) -> int:
        """Get maximum reflexive loop depth."""
        if not self.reflexive_loops:
            return 0
        return max(loop.depth for loop in self.reflexive_loops)

    @property
    def is_stable(self) -> bool:
        """Check if omega state is stable."""
        # Stable if coherent and no unstable loops
        if self.coherence_status in [
            CoherenceStatus.FRAGMENTED,
            CoherenceStatus.CONFLICTED,
        ]:
            return False

        if any(not loop.stable for loop in self.reflexive_loops):
            return False

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary."""
        return {
            "id": self.id,
            "level": self.level.value,
            "coherence_status": self.coherence_status.value,
            "coherence_score": self.coherence_score,
            "active_integrations": self.active_integrations,
            "reflexive_loops": [
                {
                    "id": l.id,
                    "depth": l.depth,
                    "observer": l.observer,
                    "observed": l.observed,
                    "observation": l.observation,
                    "stable": l.stable,
                }
                for l in self.reflexive_loops
            ],
            "emergence_signals": [
                {
                    "id": s.id,
                    "pattern_type": s.pattern_type,
                    "description": s.description,
                    "confidence": s.confidence,
                    "sources": s.source_systems,
                }
                for s in self.emergence_signals
            ],
            "attention_focus": self.attention_focus,
            "integration_mode": self.integration_mode.value,
            "subsystem_states": self.subsystem_states,
            "is_stable": self.is_stable,
            "reflexive_depth": self.reflexive_depth,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }


# =============================================================================
# OMEGA BRIDGE
# =============================================================================

@dataclass
class BridgeConfig:
    """Configuration for the omega bridge."""
    max_reflexive_depth: int = 5
    coherence_threshold: float = 0.5
    emergence_confidence_threshold: float = 0.7
    integration_timeout: float = 30.0
    enable_auto_reflection: bool = True
    enable_emergence_detection: bool = True


class OmegaBridge:
    """
    Bridge between DiTS and the Omega reflexive layer.

    Provides:
    - State management at the omega level
    - Reflexive loop creation and monitoring
    - Cross-system integration
    - Emergence detection
    - Coherence maintenance

    Example usage:
        >>> bridge = OmegaBridge()
        >>> state = bridge.create_omega_state()
        >>> bridge.integrate_subsystem("dits", dits_state)
        >>> bridge.reflect("system", "narrative coherence")
        >>> emergence = bridge.check_emergence()
    """

    def __init__(
        self,
        config: Optional[BridgeConfig] = None,
        event_callback: Optional[Callable[[OmegaEvent], None]] = None,
    ):
        """
        Initialize OmegaBridge.

        Args:
            config: Optional configuration
            event_callback: Optional callback for omega events
        """
        self.config = config or BridgeConfig()
        self._event_callback = event_callback
        self._state: Optional[OmegaState] = None
        self._observers: List[OmegaObserver] = []
        self._subsystems: Dict[str, Any] = {}
        self._event_history: List[OmegaEvent] = []
        self._stats = {
            "events_emitted": 0,
            "reflections": 0,
            "integrations": 0,
            "emergences_detected": 0,
        }

    def _emit_event(self, event: OmegaEvent) -> None:
        """Emit an omega event."""
        self._event_history.append(event)
        self._stats["events_emitted"] += 1

        if self._event_callback:
            self._event_callback(event)

        for observer in self._observers:
            observer.on_omega_event(event)

    def add_observer(self, observer: OmegaObserver) -> None:
        """Add an event observer."""
        self._observers.append(observer)

    def remove_observer(self, observer: OmegaObserver) -> None:
        """Remove an event observer."""
        if observer in self._observers:
            self._observers.remove(observer)

    def create_omega_state(
        self,
        level: OmegaLevel = OmegaLevel.L1_PATTERN,
    ) -> OmegaState:
        """
        Create a new omega state.

        Args:
            level: Initial omega processing level

        Returns:
            New OmegaState instance
        """
        self._state = OmegaState.create(level)

        self._emit_event(OmegaEvent.create(
            OmegaEventType.STATE_CHANGE,
            source="omega_bridge",
            level=level,
            action="state_created",
            state_id=self._state.id,
        ))

        return self._state

    @property
    def state(self) -> Optional[OmegaState]:
        """Get current omega state."""
        return self._state

    def integrate_subsystem(
        self,
        name: str,
        subsystem: Any,
        mode: IntegrationMode = IntegrationMode.PASSIVE,
    ) -> None:
        """
        Integrate a subsystem into the omega layer.

        Args:
            name: Subsystem name
            subsystem: The subsystem to integrate
            mode: Integration mode
        """
        if not self._state:
            self.create_omega_state()

        self._subsystems[name] = subsystem

        # Get state summary if available
        if hasattr(subsystem, "get_state_summary"):
            self._state.subsystem_states[name] = subsystem.get_state_summary()
        elif hasattr(subsystem, "to_dict"):
            self._state.subsystem_states[name] = subsystem.to_dict()
        else:
            self._state.subsystem_states[name] = {"integrated": True}

        self._state.active_integrations.append(name)
        self._state.integration_mode = mode
        self._state.update()

        self._stats["integrations"] += 1

        self._emit_event(OmegaEvent.create(
            OmegaEventType.INTEGRATION_COMPLETE,
            source="omega_bridge",
            subsystem=name,
            mode=mode.value,
        ))

    def deintegrate_subsystem(self, name: str) -> None:
        """Remove a subsystem from omega integration."""
        if not self._state:
            return

        if name in self._subsystems:
            del self._subsystems[name]

        if name in self._state.subsystem_states:
            del self._state.subsystem_states[name]

        if name in self._state.active_integrations:
            self._state.active_integrations.remove(name)

        self._state.update()

    def reflect(
        self,
        observer: str,
        subject: str,
        depth: int = 1,
    ) -> ReflexiveLoop:
        """
        Create a reflexive observation.

        Args:
            observer: What is observing
            subject: What is being observed
            depth: Reflexive depth

        Returns:
            The created ReflexiveLoop
        """
        if not self._state:
            self.create_omega_state(OmegaLevel.L4_REFLECTION)

        if depth > self.config.max_reflexive_depth:
            depth = self.config.max_reflexive_depth

        # Generate observation
        observation = self._generate_observation(observer, subject)

        loop = self._state.add_reflexive_loop(
            observer=observer,
            observed=subject,
            observation=observation,
            depth=depth,
        )

        # Elevate to reflection level
        if self._state.level.value < OmegaLevel.L4_REFLECTION.value:
            self._state.level = OmegaLevel.L4_REFLECTION

        self._stats["reflections"] += 1

        self._emit_event(OmegaEvent.create(
            OmegaEventType.REFLECTION_TRIGGERED,
            source="omega_bridge",
            observer=observer,
            subject=subject,
            depth=depth,
            observation=observation,
        ))

        return loop

    def _generate_observation(self, observer: str, subject: str) -> str:
        """Generate an observation description."""
        if observer == subject:
            return f"{observer} observes its own process of observation."

        # Check if subject is an integrated subsystem
        if subject in self._subsystems:
            subsystem = self._subsystems[subject]
            if hasattr(subsystem, "state"):
                return f"{observer} observes {subject} in state: {getattr(subsystem, 'state', 'unknown')}"

        return f"{observer} observes the nature of {subject}."

    def check_coherence(self) -> CoherenceStatus:
        """
        Check overall system coherence.

        Returns:
            Current coherence status
        """
        if not self._state:
            return CoherenceStatus.COHERENT

        scores = []

        # Check subsystem coherence
        for name, subsystem in self._subsystems.items():
            if hasattr(subsystem, "get_coherence_metrics"):
                metrics = subsystem.get_coherence_metrics()
                scores.extend(metrics.values())
            elif hasattr(subsystem, "coherence_score"):
                scores.append(subsystem.coherence_score)

        # Check reflexive loop stability
        unstable_loops = sum(
            1 for loop in self._state.reflexive_loops
            if not loop.stable
        )

        if not scores:
            avg_score = 1.0
        else:
            avg_score = sum(scores) / len(scores)

        # Determine status
        if unstable_loops > 0:
            status = CoherenceStatus.CONFLICTED
        elif avg_score >= 0.8:
            status = CoherenceStatus.COHERENT
        elif avg_score >= 0.5:
            status = CoherenceStatus.PARTIAL
        elif len(self._state.emergence_signals) > 0:
            status = CoherenceStatus.EMERGENT
        else:
            status = CoherenceStatus.FRAGMENTED

        old_status = self._state.coherence_status
        self._state.coherence_status = status
        self._state.coherence_score = avg_score

        if status != old_status:
            self._emit_event(OmegaEvent.create(
                OmegaEventType.COHERENCE_SHIFT,
                source="omega_bridge",
                old_status=old_status.value,
                new_status=status.value,
                score=avg_score,
            ))

        return status

    def check_emergence(self) -> List[EmergenceSignal]:
        """
        Check for emergence patterns across subsystems.

        Returns:
            List of detected emergence signals
        """
        if not self.config.enable_emergence_detection:
            return []

        if not self._state:
            return []

        new_signals = []

        # Look for cross-system patterns
        subsystem_patterns: Dict[str, List[str]] = {}

        for name, subsystem in self._subsystems.items():
            patterns = []
            if hasattr(subsystem, "emergence_patterns"):
                patterns = getattr(subsystem, "emergence_patterns", [])
            elif hasattr(subsystem, "get_patterns"):
                patterns = subsystem.get_patterns()

            if patterns:
                subsystem_patterns[name] = patterns

        # Find patterns that appear in multiple subsystems
        all_patterns: Dict[str, List[str]] = {}
        for name, patterns in subsystem_patterns.items():
            for pattern in patterns:
                if pattern not in all_patterns:
                    all_patterns[pattern] = []
                all_patterns[pattern].append(name)

        # Patterns in multiple subsystems indicate emergence
        for pattern, sources in all_patterns.items():
            if len(sources) >= 2:
                confidence = len(sources) / len(self._subsystems)

                if confidence >= self.config.emergence_confidence_threshold:
                    signal = self._state.add_emergence_signal(
                        pattern_type="cross_system",
                        description=f"Pattern '{pattern}' emerging across {sources}",
                        confidence=confidence,
                        sources=sources,
                    )
                    new_signals.append(signal)
                    self._stats["emergences_detected"] += 1

        # Check for reflexive emergence
        if self._state.reflexive_depth >= 3:
            signal = self._state.add_emergence_signal(
                pattern_type="reflexive",
                description="Deep self-observation creating emergent awareness",
                confidence=0.8,
                sources=["omega_bridge"],
            )
            new_signals.append(signal)
            self._stats["emergences_detected"] += 1

        if new_signals:
            self._state.level = OmegaLevel.L5_OMEGA

            for signal in new_signals:
                self._emit_event(OmegaEvent.create(
                    OmegaEventType.EMERGENCE_DETECTED,
                    source="omega_bridge",
                    signal_id=signal.id,
                    pattern_type=signal.pattern_type,
                    confidence=signal.confidence,
                ))

        return new_signals

    def elevate_level(self, target: OmegaLevel) -> bool:
        """
        Attempt to elevate the omega processing level.

        Args:
            target: Target level

        Returns:
            True if elevation successful
        """
        if not self._state:
            self.create_omega_state(target)
            return True

        current_value = list(OmegaLevel).index(self._state.level)
        target_value = list(OmegaLevel).index(target)

        if target_value <= current_value:
            return True  # Already at or above target

        # Check requirements for elevation
        requirements_met = True

        if target == OmegaLevel.L2_MEANING:
            # Need pattern recognition first
            requirements_met = len(self._subsystems) > 0

        elif target == OmegaLevel.L3_CONTEXT:
            # Need meaning interpretation
            requirements_met = self._state.level.value >= OmegaLevel.L2_MEANING.value

        elif target == OmegaLevel.L4_REFLECTION:
            # Need reflexive capability
            requirements_met = len(self._state.reflexive_loops) > 0 or \
                              self.config.enable_auto_reflection

        elif target == OmegaLevel.L5_OMEGA:
            # Need emergence signals
            requirements_met = len(self._state.emergence_signals) > 0 or \
                              self._state.reflexive_depth >= 2

        if requirements_met:
            old_level = self._state.level
            self._state.level = target
            self._state.update()

            self._emit_event(OmegaEvent.create(
                OmegaEventType.STATE_CHANGE,
                source="omega_bridge",
                action="level_elevated",
                old_level=old_level.value,
                new_level=target.value,
            ))

            return True

        return False

    def get_state_summary(self) -> Dict[str, Any]:
        """Get a summary of the current omega state."""
        if not self._state:
            return {"status": "uninitialized"}

        return {
            "id": self._state.id,
            "level": self._state.level.value,
            "coherence": {
                "status": self._state.coherence_status.value,
                "score": self._state.coherence_score,
            },
            "integrations": len(self._state.active_integrations),
            "reflexive_depth": self._state.reflexive_depth,
            "emergence_count": len(self._state.emergence_signals),
            "is_stable": self._state.is_stable,
            "attention": self._state.attention_focus,
        }

    def focus_attention(self, focus: str) -> None:
        """Set the omega-level attention focus."""
        if not self._state:
            self.create_omega_state()

        self._state.attention_focus = focus
        self._state.update()

    def get_recent_events(self, limit: int = 10) -> List[OmegaEvent]:
        """Get recent omega events."""
        return self._event_history[-limit:]

    @property
    def statistics(self) -> Dict[str, int]:
        """Get bridge statistics."""
        return self._stats.copy()

    def reset(self) -> None:
        """Reset the omega bridge."""
        self._state = None
        self._subsystems.clear()
        self._event_history.clear()
        self._stats = {
            "events_emitted": 0,
            "reflections": 0,
            "integrations": 0,
            "emergences_detected": 0,
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "OmegaLevel",
    "IntegrationMode",
    "CoherenceStatus",
    "OmegaEventType",
    # Protocols
    "OmegaObserver",
    "Integrable",
    # Data structures
    "OmegaEvent",
    "ReflexiveLoop",
    "EmergenceSignal",
    "OmegaState",
    "BridgeConfig",
    # Bridge
    "OmegaBridge",
]


if __name__ == "__main__":
    # Demo usage
    bridge = OmegaBridge()

    # Create omega state
    state = bridge.create_omega_state(OmegaLevel.L1_PATTERN)
    print(f"Created omega state: {state.id}")
    print(f"Initial level: {state.level.value}")
    print()

    # Simulate integrating a subsystem
    class MockDiTS:
        def __init__(self):
            self.state = "flowing"
            self.coherence_score = 0.85
            self.emergence_patterns = ["crescendo", "oscillation"]

        def to_dict(self):
            return {"state": self.state, "coherence": self.coherence_score}

        def get_coherence_metrics(self):
            return {"narrative": self.coherence_score}

    mock_dits = MockDiTS()
    bridge.integrate_subsystem("dits", mock_dits, IntegrationMode.REACTIVE)
    print(f"Integrated subsystem: dits")
    print(f"Active integrations: {state.active_integrations}")
    print()

    # Create reflexive observations
    loop1 = bridge.reflect("omega_bridge", "dits")
    print(f"Reflection: {loop1.observation}")

    loop2 = bridge.reflect("omega_bridge", "omega_bridge", depth=2)
    print(f"Self-reflection: {loop2.observation}")
    print()

    # Check coherence
    coherence = bridge.check_coherence()
    print(f"Coherence status: {coherence.value}")
    print(f"Coherence score: {state.coherence_score:.2f}")
    print()

    # Check for emergence
    signals = bridge.check_emergence()
    print(f"Emergence signals detected: {len(signals)}")
    for signal in signals:
        print(f"  - {signal.pattern_type}: {signal.description}")
    print()

    # Get state summary
    summary = bridge.get_state_summary()
    print("State Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    print()
    print(f"Bridge statistics: {bridge.statistics}")
