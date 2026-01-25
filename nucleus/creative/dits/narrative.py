"""
DiTS Narrative Engine
======================

Provides narrative construction and management on top of the DiTS kernel.

Episodes are discrete narrative units that can be composed into larger
narratives. The NarrativeEngine manages episode creation, sequencing,
and narrative arc construction.

Integrates with:
- kernel.py: Uses DiTSSpec and DiTSState for narrative structure
- rheomode.py: Uses flowing language constructs for narrative expression
- omega_bridge.py: Connects to higher-level narrative reasoning
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
)
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import json


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class NarrativeMode(str, Enum):
    """Mode of narrative construction."""
    LINEAR = "linear"           # Sequential progression
    BRANCHING = "branching"     # Multiple paths
    CYCLIC = "cyclic"           # Recursive/looping
    EMERGENT = "emergent"       # Self-organizing
    DIALOGIC = "dialogic"       # Conversational


class EpisodeType(str, Enum):
    """Type of narrative episode."""
    EXPOSITION = "exposition"       # Setting/introduction
    INCITING = "inciting"           # Catalyst event
    RISING = "rising"               # Building tension
    CLIMAX = "climax"               # Peak moment
    FALLING = "falling"             # Resolution begins
    RESOLUTION = "resolution"       # Conclusion
    REFLECTION = "reflection"       # Meta-narrative
    TRANSITION = "transition"       # Bridge between arcs


class NarrativeArc(str, Enum):
    """Standard narrative arc types."""
    FREYTAG = "freytag"             # Classic 5-act structure
    HERO_JOURNEY = "hero_journey"   # Campbell's monomyth
    KISHOTEN = "kishoten"           # Japanese 4-act (ki-sho-ten-ketsu)
    RAGS_RICHES = "rags_riches"     # Rise narrative
    TRAGEDY = "tragedy"             # Fall narrative
    QUEST = "quest"                 # Goal-oriented
    VOYAGE = "voyage"               # Journey and return
    REBIRTH = "rebirth"             # Transformation


class EmotionalTone(str, Enum):
    """Emotional tone of narrative content."""
    NEUTRAL = "neutral"
    HOPEFUL = "hopeful"
    TENSE = "tense"
    JOYFUL = "joyful"
    SAD = "sad"
    FEARFUL = "fearful"
    ANGRY = "angry"
    SURPRISED = "surprised"
    CONTEMPLATIVE = "contemplative"


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Episode:
    """
    A single narrative episode.

    An episode is the atomic unit of narrative construction,
    representing a discrete moment, scene, or beat in the story.

    Attributes:
        id: Unique episode identifier
        type: Classification of episode's narrative function
        title: Human-readable title
        content: The narrative content/text
        summary: Brief summary of the episode
        characters: Characters involved in this episode
        location: Setting/location of the episode
        emotional_tone: Dominant emotional quality
        tension_level: Narrative tension (0.0 to 1.0)
        preconditions: Requirements for this episode to occur
        postconditions: State changes after this episode
        metadata: Additional episode data
        created_at: Timestamp of creation
        duration: Estimated narrative duration (seconds)
    """
    id: str
    type: EpisodeType
    title: str
    content: str
    summary: str = ""
    characters: List[str] = field(default_factory=list)
    location: str = ""
    emotional_tone: EmotionalTone = EmotionalTone.NEUTRAL
    tension_level: float = 0.5
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    duration: float = 60.0  # seconds

    def __post_init__(self):
        """Generate summary if not provided."""
        if not self.summary and self.content:
            # Auto-generate summary from first sentence
            sentences = self.content.split(".")
            if sentences:
                self.summary = sentences[0].strip() + "."

    @classmethod
    def create(
        cls,
        type: EpisodeType,
        title: str,
        content: str,
        **kwargs,
    ) -> "Episode":
        """Factory method to create an episode."""
        return cls(
            id=f"ep_{uuid.uuid4().hex[:12]}",
            type=type,
            title=title,
            content=content,
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert episode to dictionary."""
        return {
            "id": self.id,
            "type": self.type.value,
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "characters": self.characters,
            "location": self.location,
            "emotional_tone": self.emotional_tone.value,
            "tension_level": self.tension_level,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "duration": self.duration,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Episode":
        """Create episode from dictionary."""
        return cls(
            id=data["id"],
            type=EpisodeType(data["type"]),
            title=data["title"],
            content=data["content"],
            summary=data.get("summary", ""),
            characters=data.get("characters", []),
            location=data.get("location", ""),
            emotional_tone=EmotionalTone(data.get("emotional_tone", "neutral")),
            tension_level=data.get("tension_level", 0.5),
            preconditions=data.get("preconditions", {}),
            postconditions=data.get("postconditions", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            duration=data.get("duration", 60.0),
        )


@dataclass
class NarrativeEdge:
    """
    A transition between episodes in a narrative.

    Represents the connection and causality between episodes.
    """
    source_id: str
    target_id: str
    label: str = "next"
    probability: float = 1.0
    condition: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Narrative:
    """
    A complete narrative composed of episodes.

    Represents the full narrative structure including episodes,
    their connections, and the overall narrative properties.

    Attributes:
        id: Unique narrative identifier
        title: Narrative title
        arc: The narrative arc structure being followed
        mode: How the narrative progresses
        episodes: Dictionary of episodes by ID
        edges: Connections between episodes
        current_episode_id: Currently active episode
        entry_point_id: Starting episode
        exit_points: Set of ending episode IDs
        world_state: Persistent narrative world state
        metadata: Additional narrative data
        created_at: Timestamp of creation
    """
    id: str
    title: str
    arc: NarrativeArc = NarrativeArc.FREYTAG
    mode: NarrativeMode = NarrativeMode.LINEAR
    episodes: Dict[str, Episode] = field(default_factory=dict)
    edges: List[NarrativeEdge] = field(default_factory=list)
    current_episode_id: Optional[str] = None
    entry_point_id: Optional[str] = None
    exit_points: Set[str] = field(default_factory=set)
    world_state: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        """Initialize entry point if episodes exist."""
        if self.episodes and not self.entry_point_id:
            # Use first episode as entry point
            self.entry_point_id = next(iter(self.episodes.keys()))
        if self.entry_point_id and not self.current_episode_id:
            self.current_episode_id = self.entry_point_id

    @classmethod
    def create(
        cls,
        title: str,
        arc: NarrativeArc = NarrativeArc.FREYTAG,
        mode: NarrativeMode = NarrativeMode.LINEAR,
        **kwargs,
    ) -> "Narrative":
        """Factory method to create a narrative."""
        return cls(
            id=f"narr_{uuid.uuid4().hex[:12]}",
            title=title,
            arc=arc,
            mode=mode,
            **kwargs,
        )

    def add_episode(
        self,
        episode: Episode,
        as_entry: bool = False,
        as_exit: bool = False,
    ) -> None:
        """Add an episode to the narrative."""
        self.episodes[episode.id] = episode

        if as_entry or not self.entry_point_id:
            self.entry_point_id = episode.id
            if not self.current_episode_id:
                self.current_episode_id = episode.id

        if as_exit:
            self.exit_points.add(episode.id)

    def connect_episodes(
        self,
        source_id: str,
        target_id: str,
        label: str = "next",
        probability: float = 1.0,
        condition: Optional[str] = None,
    ) -> None:
        """Create a connection between episodes."""
        if source_id not in self.episodes:
            raise ValueError(f"Source episode not found: {source_id}")
        if target_id not in self.episodes:
            raise ValueError(f"Target episode not found: {target_id}")

        edge = NarrativeEdge(
            source_id=source_id,
            target_id=target_id,
            label=label,
            probability=probability,
            condition=condition,
        )
        self.edges.append(edge)

    def get_next_episodes(
        self,
        from_id: Optional[str] = None,
    ) -> List[Tuple[Episode, NarrativeEdge]]:
        """Get possible next episodes from a given episode."""
        source = from_id or self.current_episode_id
        if not source:
            return []

        results = []
        for edge in self.edges:
            if edge.source_id == source:
                if edge.target_id in self.episodes:
                    results.append((self.episodes[edge.target_id], edge))
        return results

    def advance(self, edge_label: Optional[str] = None) -> Optional[Episode]:
        """
        Advance to the next episode.

        Args:
            edge_label: Optional specific edge to follow

        Returns:
            The new current episode, or None if no valid transition
        """
        if not self.current_episode_id:
            return None

        next_options = self.get_next_episodes()
        if not next_options:
            return None

        # Find matching edge
        for episode, edge in next_options:
            if edge_label is None or edge.label == edge_label:
                self.current_episode_id = episode.id
                return episode

        return None

    @property
    def current_episode(self) -> Optional[Episode]:
        """Get the current episode."""
        if self.current_episode_id:
            return self.episodes.get(self.current_episode_id)
        return None

    @property
    def is_complete(self) -> bool:
        """Check if narrative has reached an exit point."""
        return self.current_episode_id in self.exit_points

    @property
    def tension_curve(self) -> List[Tuple[str, float]]:
        """Get the tension levels across the narrative."""
        return [
            (ep.title, ep.tension_level)
            for ep in self.ordered_episodes
        ]

    @property
    def ordered_episodes(self) -> List[Episode]:
        """Get episodes in traversal order."""
        if not self.entry_point_id:
            return list(self.episodes.values())

        ordered = []
        visited = set()
        queue = [self.entry_point_id]

        while queue:
            ep_id = queue.pop(0)
            if ep_id in visited:
                continue

            visited.add(ep_id)
            if ep_id in self.episodes:
                ordered.append(self.episodes[ep_id])

            for edge in self.edges:
                if edge.source_id == ep_id and edge.target_id not in visited:
                    queue.append(edge.target_id)

        return ordered

    def to_dict(self) -> Dict[str, Any]:
        """Convert narrative to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "arc": self.arc.value,
            "mode": self.mode.value,
            "episodes": {
                ep_id: ep.to_dict()
                for ep_id, ep in self.episodes.items()
            },
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "label": e.label,
                    "probability": e.probability,
                    "condition": e.condition,
                }
                for e in self.edges
            ],
            "current_episode_id": self.current_episode_id,
            "entry_point_id": self.entry_point_id,
            "exit_points": list(self.exit_points),
            "world_state": self.world_state,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Narrative":
        """Create narrative from dictionary."""
        episodes = {
            ep_id: Episode.from_dict(ep_data)
            for ep_id, ep_data in data.get("episodes", {}).items()
        }

        edges = [
            NarrativeEdge(
                source_id=e["source"],
                target_id=e["target"],
                label=e.get("label", "next"),
                probability=e.get("probability", 1.0),
                condition=e.get("condition"),
            )
            for e in data.get("edges", [])
        ]

        return cls(
            id=data["id"],
            title=data["title"],
            arc=NarrativeArc(data.get("arc", "freytag")),
            mode=NarrativeMode(data.get("mode", "linear")),
            episodes=episodes,
            edges=edges,
            current_episode_id=data.get("current_episode_id"),
            entry_point_id=data.get("entry_point_id"),
            exit_points=set(data.get("exit_points", [])),
            world_state=data.get("world_state", {}),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
        )


# =============================================================================
# NARRATIVE ENGINE
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for narrative generation."""
    arc: NarrativeArc = NarrativeArc.FREYTAG
    mode: NarrativeMode = NarrativeMode.LINEAR
    min_episodes: int = 3
    max_episodes: int = 10
    target_duration: float = 300.0  # seconds
    emotional_arc: List[EmotionalTone] = field(default_factory=list)
    required_characters: List[str] = field(default_factory=list)
    setting: str = ""
    theme: str = ""
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NarrativeEvent:
    """Event emitted during narrative processing."""
    type: str
    narrative_id: str
    episode_id: Optional[str]
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


class NarrativeEngine:
    """
    Engine for generating and managing narratives.

    Provides methods for:
    - Creating narratives from specifications
    - Generating episodes based on arc structures
    - Managing narrative progression
    - Analyzing narrative properties

    Example usage:
        >>> engine = NarrativeEngine()
        >>> config = GenerationConfig(arc=NarrativeArc.HERO_JOURNEY)
        >>> narrative = engine.generate_narrative("My Story", config)
        >>> engine.advance_narrative(narrative)
    """

    # Arc templates defining episode type sequences
    ARC_TEMPLATES: Dict[NarrativeArc, List[EpisodeType]] = {
        NarrativeArc.FREYTAG: [
            EpisodeType.EXPOSITION,
            EpisodeType.INCITING,
            EpisodeType.RISING,
            EpisodeType.CLIMAX,
            EpisodeType.FALLING,
            EpisodeType.RESOLUTION,
        ],
        NarrativeArc.HERO_JOURNEY: [
            EpisodeType.EXPOSITION,    # Ordinary World
            EpisodeType.INCITING,      # Call to Adventure
            EpisodeType.TRANSITION,    # Crossing Threshold
            EpisodeType.RISING,        # Tests and Allies
            EpisodeType.CLIMAX,        # Ordeal
            EpisodeType.FALLING,       # Reward
            EpisodeType.TRANSITION,    # Road Back
            EpisodeType.RESOLUTION,    # Return with Elixir
        ],
        NarrativeArc.KISHOTEN: [
            EpisodeType.EXPOSITION,    # Ki - Introduction
            EpisodeType.RISING,        # Sho - Development
            EpisodeType.CLIMAX,        # Ten - Twist
            EpisodeType.RESOLUTION,    # Ketsu - Conclusion
        ],
        NarrativeArc.QUEST: [
            EpisodeType.EXPOSITION,
            EpisodeType.INCITING,
            EpisodeType.RISING,
            EpisodeType.RISING,
            EpisodeType.CLIMAX,
            EpisodeType.RESOLUTION,
        ],
        NarrativeArc.TRAGEDY: [
            EpisodeType.EXPOSITION,
            EpisodeType.RISING,
            EpisodeType.CLIMAX,
            EpisodeType.FALLING,
            EpisodeType.FALLING,
            EpisodeType.RESOLUTION,
        ],
        NarrativeArc.REBIRTH: [
            EpisodeType.EXPOSITION,
            EpisodeType.FALLING,
            EpisodeType.FALLING,
            EpisodeType.CLIMAX,
            EpisodeType.RISING,
            EpisodeType.RESOLUTION,
        ],
    }

    # Tension curves for different arcs (normalized 0-1)
    TENSION_CURVES: Dict[NarrativeArc, List[float]] = {
        NarrativeArc.FREYTAG: [0.2, 0.4, 0.6, 1.0, 0.6, 0.2],
        NarrativeArc.HERO_JOURNEY: [0.2, 0.4, 0.5, 0.7, 1.0, 0.6, 0.4, 0.3],
        NarrativeArc.KISHOTEN: [0.3, 0.5, 1.0, 0.3],
        NarrativeArc.QUEST: [0.2, 0.4, 0.6, 0.8, 1.0, 0.3],
        NarrativeArc.TRAGEDY: [0.3, 0.6, 1.0, 0.8, 0.6, 0.9],
        NarrativeArc.REBIRTH: [0.5, 0.7, 0.9, 0.3, 0.5, 0.3],
    }

    def __init__(
        self,
        event_callback: Optional[Callable[[NarrativeEvent], None]] = None,
    ):
        """
        Initialize NarrativeEngine.

        Args:
            event_callback: Optional callback for narrative events
        """
        self._event_callback = event_callback
        self._narratives: Dict[str, Narrative] = {}
        self._generation_history: List[Dict[str, Any]] = []
        self._stats = {
            "narratives_created": 0,
            "episodes_generated": 0,
            "advances": 0,
        }

    def _emit_event(
        self,
        event_type: str,
        narrative_id: str,
        episode_id: Optional[str] = None,
        **data,
    ) -> None:
        """Emit a narrative event."""
        if self._event_callback:
            event = NarrativeEvent(
                type=event_type,
                narrative_id=narrative_id,
                episode_id=episode_id,
                timestamp=time.time(),
                data=data,
            )
            self._event_callback(event)

    def create_narrative(
        self,
        title: str,
        arc: NarrativeArc = NarrativeArc.FREYTAG,
        mode: NarrativeMode = NarrativeMode.LINEAR,
    ) -> Narrative:
        """
        Create an empty narrative structure.

        Args:
            title: Narrative title
            arc: Narrative arc to follow
            mode: Narrative progression mode

        Returns:
            New Narrative instance
        """
        narrative = Narrative.create(title=title, arc=arc, mode=mode)
        self._narratives[narrative.id] = narrative
        self._stats["narratives_created"] += 1

        self._emit_event("narrative_created", narrative.id)
        return narrative

    def generate_narrative(
        self,
        title: str,
        config: Optional[GenerationConfig] = None,
    ) -> Narrative:
        """
        Generate a complete narrative from configuration.

        Args:
            title: Narrative title
            config: Generation configuration

        Returns:
            Generated Narrative with episodes
        """
        config = config or GenerationConfig()
        narrative = self.create_narrative(title, config.arc, config.mode)

        # Get arc template
        template = self.ARC_TEMPLATES.get(
            config.arc,
            self.ARC_TEMPLATES[NarrativeArc.FREYTAG],
        )
        tension_curve = self.TENSION_CURVES.get(
            config.arc,
            self.TENSION_CURVES[NarrativeArc.FREYTAG],
        )

        # Generate episodes based on template
        prev_episode_id = None
        for i, episode_type in enumerate(template):
            # Calculate tension for this position
            tension = tension_curve[i] if i < len(tension_curve) else 0.5

            # Create episode
            episode = Episode.create(
                type=episode_type,
                title=f"{title} - {episode_type.value.title()}",
                content=self._generate_episode_content(
                    episode_type,
                    config,
                    narrative,
                    i,
                ),
                characters=config.required_characters,
                location=config.setting,
                tension_level=tension,
                metadata={
                    "arc_position": i,
                    "arc_total": len(template),
                },
            )

            # Add to narrative
            is_first = i == 0
            is_last = i == len(template) - 1
            narrative.add_episode(
                episode,
                as_entry=is_first,
                as_exit=is_last,
            )

            # Connect to previous episode
            if prev_episode_id:
                narrative.connect_episodes(prev_episode_id, episode.id)

            prev_episode_id = episode.id
            self._stats["episodes_generated"] += 1

        self._generation_history.append({
            "narrative_id": narrative.id,
            "config": {
                "arc": config.arc.value,
                "mode": config.mode.value,
                "episode_count": len(template),
            },
            "timestamp": time.time(),
        })

        self._emit_event(
            "narrative_generated",
            narrative.id,
            episode_count=len(narrative.episodes),
        )

        return narrative

    def _generate_episode_content(
        self,
        episode_type: EpisodeType,
        config: GenerationConfig,
        narrative: Narrative,
        position: int,
    ) -> str:
        """Generate content for an episode based on its type."""
        # Content templates based on episode type
        templates = {
            EpisodeType.EXPOSITION: (
                f"In {config.setting or 'a world not unlike our own'}, "
                f"we find {', '.join(config.required_characters) or 'our protagonist'}. "
                f"The theme of {config.theme or 'transformation'} begins to emerge."
            ),
            EpisodeType.INCITING: (
                "Something changes. An event occurs that sets everything in motion. "
                "The ordinary gives way to the extraordinary."
            ),
            EpisodeType.RISING: (
                "Tension builds as challenges mount. "
                "Each obstacle overcome reveals new depths of character."
            ),
            EpisodeType.CLIMAX: (
                "Everything converges in this moment of maximum tension. "
                "The central conflict reaches its peak."
            ),
            EpisodeType.FALLING: (
                "The consequences of the climax unfold. "
                "Characters begin to understand what has truly changed."
            ),
            EpisodeType.RESOLUTION: (
                "A new equilibrium emerges. "
                f"The journey through {config.theme or 'this story'} reaches its conclusion."
            ),
            EpisodeType.REFLECTION: (
                "Stepping back, we see the larger pattern. "
                "What has been learned ripples outward."
            ),
            EpisodeType.TRANSITION: (
                "A bridge between what was and what will be. "
                "The narrative shifts its focus."
            ),
        }

        return templates.get(episode_type, "The story continues...")

    def add_episode(
        self,
        narrative: Narrative,
        episode: Episode,
        connect_from: Optional[str] = None,
        as_exit: bool = False,
    ) -> None:
        """
        Add an episode to an existing narrative.

        Args:
            narrative: Target narrative
            episode: Episode to add
            connect_from: Episode ID to connect from
            as_exit: Whether this is an exit point
        """
        narrative.add_episode(episode, as_exit=as_exit)

        if connect_from:
            narrative.connect_episodes(connect_from, episode.id)

        self._stats["episodes_generated"] += 1
        self._emit_event(
            "episode_added",
            narrative.id,
            episode.id,
        )

    def advance_narrative(
        self,
        narrative: Narrative,
        choice: Optional[str] = None,
    ) -> Optional[Episode]:
        """
        Advance a narrative to the next episode.

        Args:
            narrative: Narrative to advance
            choice: Optional edge label to follow

        Returns:
            The new current episode, or None if cannot advance
        """
        episode = narrative.advance(choice)

        if episode:
            self._stats["advances"] += 1
            self._emit_event(
                "narrative_advanced",
                narrative.id,
                episode.id,
            )

        return episode

    def get_narrative(self, narrative_id: str) -> Optional[Narrative]:
        """Get a narrative by ID."""
        return self._narratives.get(narrative_id)

    def analyze_narrative(self, narrative: Narrative) -> Dict[str, Any]:
        """
        Analyze properties of a narrative.

        Args:
            narrative: Narrative to analyze

        Returns:
            Analysis results
        """
        episodes = list(narrative.episodes.values())

        # Tension analysis
        tensions = [ep.tension_level for ep in narrative.ordered_episodes]
        peak_tension = max(tensions) if tensions else 0
        avg_tension = sum(tensions) / len(tensions) if tensions else 0

        # Character analysis
        all_characters = set()
        for ep in episodes:
            all_characters.update(ep.characters)

        # Episode type distribution
        type_counts: Dict[str, int] = {}
        for ep in episodes:
            ep_type = ep.type.value
            type_counts[ep_type] = type_counts.get(ep_type, 0) + 1

        # Emotional arc
        emotional_arc = [ep.emotional_tone.value for ep in narrative.ordered_episodes]

        # Structure analysis
        entry_count = 1 if narrative.entry_point_id else 0
        exit_count = len(narrative.exit_points)
        branching_factor = len(narrative.edges) / max(len(episodes), 1)

        return {
            "id": narrative.id,
            "title": narrative.title,
            "arc": narrative.arc.value,
            "mode": narrative.mode.value,
            "episode_count": len(episodes),
            "edge_count": len(narrative.edges),
            "tension_analysis": {
                "peak": peak_tension,
                "average": avg_tension,
                "curve": tensions,
            },
            "character_count": len(all_characters),
            "characters": list(all_characters),
            "episode_type_distribution": type_counts,
            "emotional_arc": emotional_arc,
            "structure": {
                "entry_points": entry_count,
                "exit_points": exit_count,
                "branching_factor": branching_factor,
                "is_linear": branching_factor <= 1.0,
            },
            "completion_status": {
                "is_complete": narrative.is_complete,
                "current_position": narrative.current_episode_id,
            },
            "total_duration": sum(ep.duration for ep in episodes),
        }

    def merge_narratives(
        self,
        narratives: List[Narrative],
        connection_type: str = "sequential",
    ) -> Narrative:
        """
        Merge multiple narratives into one.

        Args:
            narratives: Narratives to merge
            connection_type: How to connect them ("sequential" or "parallel")

        Returns:
            Merged narrative
        """
        if not narratives:
            raise ValueError("No narratives to merge")

        merged = Narrative.create(
            title=f"Merged: {', '.join(n.title for n in narratives)}",
            arc=narratives[0].arc,
            mode=NarrativeMode.BRANCHING if connection_type == "parallel" else NarrativeMode.LINEAR,
        )

        # Add all episodes
        for narrative in narratives:
            for episode in narrative.episodes.values():
                merged.episodes[episode.id] = episode

            # Add all edges
            merged.edges.extend(narrative.edges)

        if connection_type == "sequential":
            # Connect exit points of each narrative to entry of next
            for i in range(len(narratives) - 1):
                for exit_id in narratives[i].exit_points:
                    if narratives[i + 1].entry_point_id:
                        merged.connect_episodes(
                            exit_id,
                            narratives[i + 1].entry_point_id,
                            label="continue",
                        )

            # Set entry and exit
            if narratives[0].entry_point_id:
                merged.entry_point_id = narratives[0].entry_point_id
                merged.current_episode_id = merged.entry_point_id
            merged.exit_points = narratives[-1].exit_points.copy()

        else:  # parallel
            # Create junction points
            junction_start = Episode.create(
                type=EpisodeType.TRANSITION,
                title="Junction: Paths Diverge",
                content="Multiple paths present themselves.",
            )
            junction_end = Episode.create(
                type=EpisodeType.TRANSITION,
                title="Junction: Paths Converge",
                content="The paths come together.",
            )

            merged.add_episode(junction_start, as_entry=True)
            merged.add_episode(junction_end, as_exit=True)

            # Connect junction to all narrative entries
            for narrative in narratives:
                if narrative.entry_point_id:
                    merged.connect_episodes(
                        junction_start.id,
                        narrative.entry_point_id,
                        label=narrative.title,
                    )

                # Connect exits to junction end
                for exit_id in narrative.exit_points:
                    merged.connect_episodes(exit_id, junction_end.id)

        self._narratives[merged.id] = merged
        self._stats["narratives_created"] += 1

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
    "NarrativeMode",
    "EpisodeType",
    "NarrativeArc",
    "EmotionalTone",
    # Data structures
    "Episode",
    "NarrativeEdge",
    "Narrative",
    "GenerationConfig",
    "NarrativeEvent",
    # Engine
    "NarrativeEngine",
]


if __name__ == "__main__":
    # Demo usage
    engine = NarrativeEngine()

    # Generate a narrative using Hero's Journey arc
    config = GenerationConfig(
        arc=NarrativeArc.HERO_JOURNEY,
        mode=NarrativeMode.LINEAR,
        required_characters=["Hero", "Mentor", "Shadow"],
        setting="A land between worlds",
        theme="self-discovery",
    )

    narrative = engine.generate_narrative("The Threshold", config)

    print(f"Generated Narrative: {narrative.title}")
    print(f"Arc: {narrative.arc.value}")
    print(f"Episodes: {len(narrative.episodes)}")
    print()

    # Print episode sequence
    for i, episode in enumerate(narrative.ordered_episodes):
        print(f"{i + 1}. [{episode.type.value}] {episode.title}")
        print(f"   Tension: {'*' * int(episode.tension_level * 10)}")
        print(f"   {episode.content[:60]}...")
        print()

    # Analyze narrative
    analysis = engine.analyze_narrative(narrative)
    print("Analysis:")
    print(f"  Peak tension: {analysis['tension_analysis']['peak']:.2f}")
    print(f"  Characters: {analysis['characters']}")
    print(f"  Duration: {analysis['total_duration']}s")

    print(f"\nEngine statistics: {engine.statistics}")
