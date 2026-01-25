"""
Storyboard Module
=================

Storyboard generation from parsed scripts.
Converts screenplay scenes into visual shot breakdowns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any, Iterator

from .script_parser import Script, SceneElement, ElementType


class ShotType(Enum):
    """Types of camera shots for storyboarding."""
    EXTREME_WIDE = auto()      # Establishing shot, landscapes
    WIDE = auto()              # Full scene, multiple characters
    MEDIUM_WIDE = auto()       # Knee up, cowboy shot
    MEDIUM = auto()            # Waist up
    MEDIUM_CLOSE = auto()      # Chest up
    CLOSE = auto()             # Face/head
    EXTREME_CLOSE = auto()     # Detail shot
    OVER_SHOULDER = auto()     # OTS shot
    TWO_SHOT = auto()          # Two characters
    GROUP = auto()             # Multiple characters
    POV = auto()               # Point of view
    INSERT = auto()            # Detail insert
    REACTION = auto()          # Reaction shot
    CUTAWAY = auto()           # Cutaway shot


class CameraMovement(Enum):
    """Types of camera movements."""
    STATIC = auto()
    PAN_LEFT = auto()
    PAN_RIGHT = auto()
    TILT_UP = auto()
    TILT_DOWN = auto()
    DOLLY_IN = auto()
    DOLLY_OUT = auto()
    TRUCK_LEFT = auto()
    TRUCK_RIGHT = auto()
    ZOOM_IN = auto()
    ZOOM_OUT = auto()
    CRANE_UP = auto()
    CRANE_DOWN = auto()
    STEADICAM = auto()
    HANDHELD = auto()
    RACK_FOCUS = auto()


class ShotAngle(Enum):
    """Camera angles for shots."""
    EYE_LEVEL = auto()
    LOW_ANGLE = auto()
    HIGH_ANGLE = auto()
    DUTCH_ANGLE = auto()
    BIRDS_EYE = auto()
    WORMS_EYE = auto()


@dataclass
class StoryboardPanel:
    """
    A single panel in a storyboard.

    Attributes:
        panel_id: Unique identifier for this panel
        scene_number: The scene this panel belongs to
        shot_number: Sequential shot number within scene
        shot_type: The type of camera shot
        movement: Camera movement type
        angle: Camera angle
        description: Visual description of the shot
        dialogue: Any dialogue in this shot
        action: Action occurring in this shot
        characters: Characters visible in this shot
        duration_seconds: Estimated duration of shot
        notes: Director/technical notes
        reference_image: Path to reference image if any
        metadata: Additional metadata
    """
    panel_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    scene_number: int = 1
    shot_number: int = 1
    shot_type: ShotType = ShotType.MEDIUM
    movement: CameraMovement = CameraMovement.STATIC
    angle: ShotAngle = ShotAngle.EYE_LEVEL
    description: str = ""
    dialogue: str = ""
    action: str = ""
    characters: List[str] = field(default_factory=list)
    duration_seconds: float = 3.0
    notes: str = ""
    reference_image: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert panel to dictionary representation."""
        return {
            'panel_id': self.panel_id,
            'scene_number': self.scene_number,
            'shot_number': self.shot_number,
            'shot_type': self.shot_type.name,
            'movement': self.movement.name,
            'angle': self.angle.name,
            'description': self.description,
            'dialogue': self.dialogue,
            'action': self.action,
            'characters': self.characters,
            'duration_seconds': self.duration_seconds,
            'notes': self.notes,
            'reference_image': self.reference_image,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StoryboardPanel':
        """Create panel from dictionary representation."""
        return cls(
            panel_id=data.get('panel_id', str(uuid.uuid4())[:8]),
            scene_number=data.get('scene_number', 1),
            shot_number=data.get('shot_number', 1),
            shot_type=ShotType[data.get('shot_type', 'MEDIUM')],
            movement=CameraMovement[data.get('movement', 'STATIC')],
            angle=ShotAngle[data.get('angle', 'EYE_LEVEL')],
            description=data.get('description', ''),
            dialogue=data.get('dialogue', ''),
            action=data.get('action', ''),
            characters=data.get('characters', []),
            duration_seconds=data.get('duration_seconds', 3.0),
            notes=data.get('notes', ''),
            reference_image=data.get('reference_image'),
            metadata=data.get('metadata', {}),
        )

    def __repr__(self) -> str:
        return f"StoryboardPanel(scene={self.scene_number}, shot={self.shot_number}, {self.shot_type.name})"


@dataclass
class Storyboard:
    """
    A complete storyboard for a script.

    Attributes:
        title: Title of the storyboard/project
        panels: Ordered list of storyboard panels
        metadata: Project-level metadata
    """
    title: str = ""
    panels: List[StoryboardPanel] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_panel(self, panel: StoryboardPanel) -> None:
        """Add a panel to the storyboard."""
        self.panels.append(panel)

    def get_scene_panels(self, scene_number: int) -> List[StoryboardPanel]:
        """Get all panels for a specific scene."""
        return [p for p in self.panels if p.scene_number == scene_number]

    def get_scenes(self) -> List[int]:
        """Get list of unique scene numbers."""
        return sorted(set(p.scene_number for p in self.panels))

    def total_duration(self) -> float:
        """Calculate total estimated duration in seconds."""
        return sum(p.duration_seconds for p in self.panels)

    def panel_count(self) -> int:
        """Return total number of panels."""
        return len(self.panels)

    def to_dict(self) -> Dict[str, Any]:
        """Convert storyboard to dictionary representation."""
        return {
            'title': self.title,
            'panels': [p.to_dict() for p in self.panels],
            'metadata': self.metadata,
            'total_duration': self.total_duration(),
            'panel_count': self.panel_count(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Storyboard':
        """Create storyboard from dictionary representation."""
        storyboard = cls(
            title=data.get('title', ''),
            metadata=data.get('metadata', {}),
        )
        for panel_data in data.get('panels', []):
            storyboard.add_panel(StoryboardPanel.from_dict(panel_data))
        return storyboard

    def __iter__(self) -> Iterator[StoryboardPanel]:
        return iter(self.panels)

    def __len__(self) -> int:
        return len(self.panels)


class StoryboardGenerator:
    """
    Generates storyboards from parsed scripts.

    Uses heuristics to determine shot types, camera movements,
    and panel breakdowns based on script elements.
    """

    # Heuristics for shot type selection
    DIALOGUE_SHOT_TYPES = [ShotType.MEDIUM, ShotType.MEDIUM_CLOSE, ShotType.CLOSE, ShotType.OVER_SHOULDER]
    ACTION_SHOT_TYPES = [ShotType.WIDE, ShotType.MEDIUM_WIDE, ShotType.MEDIUM]
    ESTABLISHING_SHOT_TYPES = [ShotType.EXTREME_WIDE, ShotType.WIDE]

    def __init__(self,
                 default_shot_duration: float = 3.0,
                 establishing_shot_duration: float = 4.0,
                 dialogue_word_rate: float = 2.5):
        """
        Initialize the storyboard generator.

        Args:
            default_shot_duration: Default duration for non-dialogue shots
            establishing_shot_duration: Duration for establishing shots
            dialogue_word_rate: Words per second for dialogue timing
        """
        self.default_shot_duration = default_shot_duration
        self.establishing_shot_duration = establishing_shot_duration
        self.dialogue_word_rate = dialogue_word_rate

    def generate(self, script: Script) -> Storyboard:
        """
        Generate a storyboard from a parsed script.

        Args:
            script: A parsed Script object

        Returns:
            A Storyboard with panels for each shot.
        """
        storyboard = Storyboard(title=script.title)
        scenes = script.get_scenes()

        for scene_num, scene_elements in enumerate(scenes, 1):
            scene_panels = self._generate_scene_panels(scene_num, scene_elements)
            for panel in scene_panels:
                storyboard.add_panel(panel)

        return storyboard

    def _generate_scene_panels(self, scene_num: int, elements: List[SceneElement]) -> List[StoryboardPanel]:
        """Generate panels for a single scene."""
        panels = []
        shot_num = 0
        current_characters: List[str] = []

        for i, element in enumerate(elements):
            if element.element_type == ElementType.SCENE_HEADING:
                # Establishing shot for scene heading
                shot_num += 1
                panel = self._create_establishing_panel(scene_num, shot_num, element)
                panels.append(panel)

            elif element.element_type == ElementType.ACTION:
                # Action shots
                shot_num += 1
                panel = self._create_action_panel(scene_num, shot_num, element, current_characters)
                panels.append(panel)

            elif element.element_type == ElementType.CHARACTER:
                # Track character for upcoming dialogue
                if element.character_name:
                    if element.character_name not in current_characters:
                        current_characters.append(element.character_name)

            elif element.element_type == ElementType.DIALOGUE:
                # Dialogue shot
                shot_num += 1
                panel = self._create_dialogue_panel(scene_num, shot_num, element, current_characters)
                panels.append(panel)

            elif element.element_type == ElementType.PARENTHETICAL:
                # May affect the next dialogue shot but doesn't create its own panel
                pass

            elif element.element_type == ElementType.TRANSITION:
                # Transitions may affect timing but don't create panels
                pass

        return panels

    def _create_establishing_panel(self, scene_num: int, shot_num: int, element: SceneElement) -> StoryboardPanel:
        """Create an establishing shot panel for a scene heading."""
        location = element.get_location() or element.content
        time_of_day = element.get_time_of_day() or ""

        description = f"Establishing shot: {location}"
        if time_of_day:
            description += f" - {time_of_day}"

        return StoryboardPanel(
            scene_number=scene_num,
            shot_number=shot_num,
            shot_type=ShotType.WIDE,
            movement=CameraMovement.STATIC,
            angle=ShotAngle.EYE_LEVEL,
            description=description,
            action=element.content,
            duration_seconds=self.establishing_shot_duration,
            metadata={'element_type': 'scene_heading', 'location': location, 'time': time_of_day}
        )

    def _create_action_panel(self, scene_num: int, shot_num: int, element: SceneElement, characters: List[str]) -> StoryboardPanel:
        """Create an action shot panel."""
        # Determine shot type based on action content
        content_lower = element.content.lower()

        if any(word in content_lower for word in ['enters', 'walks', 'runs', 'moves']):
            shot_type = ShotType.MEDIUM_WIDE
            movement = CameraMovement.PAN_LEFT if 'left' in content_lower else (
                CameraMovement.PAN_RIGHT if 'right' in content_lower else CameraMovement.STATIC
            )
        elif any(word in content_lower for word in ['close', 'detail', 'hand', 'face']):
            shot_type = ShotType.CLOSE
            movement = CameraMovement.STATIC
        else:
            shot_type = ShotType.MEDIUM
            movement = CameraMovement.STATIC

        return StoryboardPanel(
            scene_number=scene_num,
            shot_number=shot_num,
            shot_type=shot_type,
            movement=movement,
            angle=ShotAngle.EYE_LEVEL,
            description=element.content[:100],
            action=element.content,
            characters=list(characters),
            duration_seconds=self._estimate_action_duration(element.content),
            metadata={'element_type': 'action'}
        )

    def _create_dialogue_panel(self, scene_num: int, shot_num: int, element: SceneElement, characters: List[str]) -> StoryboardPanel:
        """Create a dialogue shot panel."""
        # Determine shot type based on dialogue context
        if len(characters) >= 2:
            # Conversation: alternate between OTS and medium close
            shot_type = ShotType.OVER_SHOULDER if shot_num % 2 == 0 else ShotType.MEDIUM_CLOSE
        else:
            shot_type = ShotType.MEDIUM_CLOSE

        speaker = element.character_name or "UNKNOWN"

        return StoryboardPanel(
            scene_number=scene_num,
            shot_number=shot_num,
            shot_type=shot_type,
            movement=CameraMovement.STATIC,
            angle=ShotAngle.EYE_LEVEL,
            description=f"{speaker} speaks",
            dialogue=element.content,
            characters=[speaker],
            duration_seconds=self._estimate_dialogue_duration(element.content),
            metadata={'element_type': 'dialogue', 'speaker': speaker}
        )

    def _estimate_dialogue_duration(self, dialogue: str) -> float:
        """Estimate duration of dialogue based on word count."""
        words = len(dialogue.split())
        duration = words / self.dialogue_word_rate
        return max(2.0, min(duration, 15.0))  # Clamp between 2-15 seconds

    def _estimate_action_duration(self, action: str) -> float:
        """Estimate duration of action based on description length."""
        words = len(action.split())
        # Rough heuristic: longer descriptions suggest longer actions
        if words < 10:
            return self.default_shot_duration
        elif words < 30:
            return self.default_shot_duration * 1.5
        else:
            return self.default_shot_duration * 2

    def regenerate_panel(self, panel: StoryboardPanel, **kwargs) -> StoryboardPanel:
        """
        Regenerate a panel with modified parameters.

        Args:
            panel: The panel to regenerate
            **kwargs: Parameters to override

        Returns:
            A new panel with updated parameters.
        """
        panel_dict = panel.to_dict()
        panel_dict.update(kwargs)

        # Handle enum conversions
        if 'shot_type' in kwargs and isinstance(kwargs['shot_type'], str):
            panel_dict['shot_type'] = kwargs['shot_type']
        elif 'shot_type' in kwargs and isinstance(kwargs['shot_type'], ShotType):
            panel_dict['shot_type'] = kwargs['shot_type'].name

        if 'movement' in kwargs and isinstance(kwargs['movement'], CameraMovement):
            panel_dict['movement'] = kwargs['movement'].name

        if 'angle' in kwargs and isinstance(kwargs['angle'], ShotAngle):
            panel_dict['angle'] = kwargs['angle'].name

        return StoryboardPanel.from_dict(panel_dict)


__all__ = [
    'ShotType',
    'CameraMovement',
    'ShotAngle',
    'StoryboardPanel',
    'Storyboard',
    'StoryboardGenerator',
]
