"""
Bus Topic Definitions
=====================

Topic constants and validation functions for Creative subsystem events.

Topics follow a hierarchical namespacing convention:
    creative.<subsystem>.<action>

Subsystems:
- grammars: CGP/EGGP synthesis events
- cinema: Video generation events
- visual: Image generation events
- auralux: Audio/voice synthesis events
- avatars: 3D avatar generation events
- dits: Diegetic Transition System events
"""

from dataclasses import dataclass
from typing import Optional


# =============================================================================
# TOPIC NAMESPACE CONSTANTS
# =============================================================================

# Root namespace for all Creative subsystem events
NAMESPACE_ROOT = "creative"

# Subsystem namespaces
NAMESPACE_GRAMMARS = f"{NAMESPACE_ROOT}.grammars"
NAMESPACE_CINEMA = f"{NAMESPACE_ROOT}.cinema"
NAMESPACE_VISUAL = f"{NAMESPACE_ROOT}.visual"
NAMESPACE_AURALUX = f"{NAMESPACE_ROOT}.auralux"
NAMESPACE_AVATARS = f"{NAMESPACE_ROOT}.avatars"
NAMESPACE_DITS = f"{NAMESPACE_ROOT}.dits"

# Common action suffixes
ACTION_STARTED = "started"
ACTION_PROGRESS = "progress"
ACTION_COMPLETED = "completed"
ACTION_FAILED = "failed"


@dataclass(frozen=True)
class TopicRegistry:
    """
    Registry of all Creative subsystem bus topics.

    This frozen dataclass provides typed access to all valid topic
    strings used by the Creative subsystem. Topics are organized
    by subsystem and follow the pattern: creative.<subsystem>.<action>

    Usage:
        >>> from nucleus.creative.bus import TOPICS
        >>> TOPICS.VISUAL_RENDER
        'creative.visual.render'
        >>> TOPICS.GENERATION_STARTED
        'creative.generation.started'
    """

    # =========================================================================
    # GENERATION LIFECYCLE EVENTS
    # =========================================================================

    #: Emitted when any generation job starts
    GENERATION_STARTED: str = f"{NAMESPACE_ROOT}.generation.started"

    #: Emitted periodically with generation progress updates
    GENERATION_PROGRESS: str = f"{NAMESPACE_ROOT}.generation.progress"

    #: Emitted when generation completes successfully
    GENERATION_COMPLETED: str = f"{NAMESPACE_ROOT}.generation.completed"

    #: Emitted when generation fails
    GENERATION_FAILED: str = f"{NAMESPACE_ROOT}.generation.failed"

    # =========================================================================
    # GRAMMARS SUBSYSTEM EVENTS
    # =========================================================================

    #: Request to synthesize a grammar
    GRAMMAR_SYNTHESIZE: str = f"{NAMESPACE_GRAMMARS}.synthesize"

    #: Grammar has been evolved/optimized
    GRAMMAR_EVOLVED: str = f"{NAMESPACE_GRAMMARS}.evolved"

    #: Grammar parsing started
    GRAMMAR_PARSE_STARTED: str = f"{NAMESPACE_GRAMMARS}.parse.started"

    #: Grammar parsing completed
    GRAMMAR_PARSE_COMPLETED: str = f"{NAMESPACE_GRAMMARS}.parse.completed"

    #: AST visualization ready
    GRAMMAR_AST_READY: str = f"{NAMESPACE_GRAMMARS}.ast.ready"

    #: Metagrammar applied
    GRAMMAR_META_APPLIED: str = f"{NAMESPACE_GRAMMARS}.meta.applied"

    # =========================================================================
    # CINEMA SUBSYSTEM EVENTS
    # =========================================================================

    #: Request to generate video
    CINEMA_GENERATE: str = f"{NAMESPACE_CINEMA}.generate"

    #: Individual frame ready
    CINEMA_FRAME_READY: str = f"{NAMESPACE_CINEMA}.frame_ready"

    #: Storyboard created
    CINEMA_STORYBOARD_READY: str = f"{NAMESPACE_CINEMA}.storyboard.ready"

    #: Scene composition started
    CINEMA_SCENE_STARTED: str = f"{NAMESPACE_CINEMA}.scene.started"

    #: Scene composition completed
    CINEMA_SCENE_COMPLETED: str = f"{NAMESPACE_CINEMA}.scene.completed"

    #: Temporal consistency check
    CINEMA_TEMPORAL_CHECK: str = f"{NAMESPACE_CINEMA}.temporal.check"

    #: Multi-shot assembly started
    CINEMA_ASSEMBLY_STARTED: str = f"{NAMESPACE_CINEMA}.assembly.started"

    #: Multi-shot assembly completed
    CINEMA_ASSEMBLY_COMPLETED: str = f"{NAMESPACE_CINEMA}.assembly.completed"

    # =========================================================================
    # VISUAL SUBSYSTEM EVENTS
    # =========================================================================

    #: Request to render an image
    VISUAL_RENDER: str = f"{NAMESPACE_VISUAL}.render"

    #: Style transfer applied
    VISUAL_STYLE_APPLIED: str = f"{NAMESPACE_VISUAL}.style_applied"

    #: Text-to-image generation started
    VISUAL_T2I_STARTED: str = f"{NAMESPACE_VISUAL}.t2i.started"

    #: Text-to-image generation completed
    VISUAL_T2I_COMPLETED: str = f"{NAMESPACE_VISUAL}.t2i.completed"

    #: Image-to-image transformation started
    VISUAL_I2I_STARTED: str = f"{NAMESPACE_VISUAL}.i2i.started"

    #: Image-to-image transformation completed
    VISUAL_I2I_COMPLETED: str = f"{NAMESPACE_VISUAL}.i2i.completed"

    #: Upscaling started
    VISUAL_UPSCALE_STARTED: str = f"{NAMESPACE_VISUAL}.upscale.started"

    #: Upscaling completed
    VISUAL_UPSCALE_COMPLETED: str = f"{NAMESPACE_VISUAL}.upscale.completed"

    #: Diffusion step progress
    VISUAL_DIFFUSION_STEP: str = f"{NAMESPACE_VISUAL}.diffusion.step"

    # =========================================================================
    # AURALUX SUBSYSTEM EVENTS
    # =========================================================================

    #: Request to synthesize audio
    AURALUX_SYNTHESIZE: str = f"{NAMESPACE_AURALUX}.synthesize"

    #: Speaker profile ready
    AURALUX_SPEAKER_READY: str = f"{NAMESPACE_AURALUX}.speaker_ready"

    #: Text-to-speech started
    AURALUX_TTS_STARTED: str = f"{NAMESPACE_AURALUX}.tts.started"

    #: Text-to-speech completed
    AURALUX_TTS_COMPLETED: str = f"{NAMESPACE_AURALUX}.tts.completed"

    #: Speech-to-text started
    AURALUX_STT_STARTED: str = f"{NAMESPACE_AURALUX}.stt.started"

    #: Speech-to-text completed
    AURALUX_STT_COMPLETED: str = f"{NAMESPACE_AURALUX}.stt.completed"

    #: Voice cloning started
    AURALUX_CLONE_STARTED: str = f"{NAMESPACE_AURALUX}.clone.started"

    #: Voice cloning completed
    AURALUX_CLONE_COMPLETED: str = f"{NAMESPACE_AURALUX}.clone.completed"

    #: Audio enhancement applied
    AURALUX_ENHANCED: str = f"{NAMESPACE_AURALUX}.enhanced"

    # =========================================================================
    # AVATARS SUBSYSTEM EVENTS
    # =========================================================================

    #: Request to generate avatar
    AVATAR_GENERATE: str = f"{NAMESPACE_AVATARS}.generate"

    #: Mesh geometry ready
    AVATAR_MESH_READY: str = f"{NAMESPACE_AVATARS}.mesh_ready"

    #: 3D Gaussian Splatting started
    AVATAR_3DGS_STARTED: str = f"{NAMESPACE_AVATARS}.3dgs.started"

    #: 3D Gaussian Splatting completed
    AVATAR_3DGS_COMPLETED: str = f"{NAMESPACE_AVATARS}.3dgs.completed"

    #: SMPL-X body model fitted
    AVATAR_SMPLX_FITTED: str = f"{NAMESPACE_AVATARS}.smplx.fitted"

    #: Neural PBR materials applied
    AVATAR_PBR_APPLIED: str = f"{NAMESPACE_AVATARS}.pbr.applied"

    #: Animation rigging completed
    AVATAR_RIGGED: str = f"{NAMESPACE_AVATARS}.rigged"

    #: Deformation applied
    AVATAR_DEFORMED: str = f"{NAMESPACE_AVATARS}.deformed"

    # =========================================================================
    # DITS (Diegetic Transition System) EVENTS
    # =========================================================================

    #: Request to evaluate narrative state
    DITS_EVALUATE: str = f"{NAMESPACE_DITS}.evaluate"

    #: Narrative transition occurred
    DITS_TRANSITION: str = f"{NAMESPACE_DITS}.transition"

    #: Mu-calculus evaluation started
    DITS_MU_STARTED: str = f"{NAMESPACE_DITS}.mu.started"

    #: Nu-calculus evaluation started
    DITS_NU_STARTED: str = f"{NAMESPACE_DITS}.nu.started"

    #: Omega loop entered
    DITS_OMEGA_ENTERED: str = f"{NAMESPACE_DITS}.omega.entered"

    #: Omega loop exited
    DITS_OMEGA_EXITED: str = f"{NAMESPACE_DITS}.omega.exited"

    #: Rheomode transformation
    DITS_RHEOMODE: str = f"{NAMESPACE_DITS}.rheomode"

    #: Narrative construction completed
    DITS_CONSTRUCTED: str = f"{NAMESPACE_DITS}.constructed"


# Singleton instance for convenient access
TOPICS = TopicRegistry()


# =============================================================================
# LEGACY COMPATIBILITY DICT
# =============================================================================

# This dict mirrors the BUS_TOPICS format from the main creative __init__.py
# for backwards compatibility
BUS_TOPICS_DICT = {
    "generation_started": TOPICS.GENERATION_STARTED,
    "generation_progress": TOPICS.GENERATION_PROGRESS,
    "generation_completed": TOPICS.GENERATION_COMPLETED,
    "generation_failed": TOPICS.GENERATION_FAILED,
    # Grammar events
    "grammar_synthesize": TOPICS.GRAMMAR_SYNTHESIZE,
    "grammar_evolved": TOPICS.GRAMMAR_EVOLVED,
    # Cinema events
    "cinema_generate": TOPICS.CINEMA_GENERATE,
    "cinema_frame_ready": TOPICS.CINEMA_FRAME_READY,
    # Visual events
    "visual_render": TOPICS.VISUAL_RENDER,
    "visual_style_applied": TOPICS.VISUAL_STYLE_APPLIED,
    # Auralux events
    "auralux_synthesize": TOPICS.AURALUX_SYNTHESIZE,
    "auralux_speaker_ready": TOPICS.AURALUX_SPEAKER_READY,
    # Avatar events
    "avatar_generate": TOPICS.AVATAR_GENERATE,
    "avatar_mesh_ready": TOPICS.AVATAR_MESH_READY,
    # DiTS events
    "dits_evaluate": TOPICS.DITS_EVALUATE,
    "dits_transition": TOPICS.DITS_TRANSITION,
}


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

# Set of all valid topics for validation
_ALL_TOPICS: frozenset[str] = frozenset(
    value for key, value in vars(TopicRegistry).items()
    if not key.startswith("_") and isinstance(value, str)
)

# Mapping of subsystems to their topics
_SUBSYSTEM_TOPICS: dict[str, list[str]] = {
    "grammars": [t for t in _ALL_TOPICS if t.startswith(NAMESPACE_GRAMMARS)],
    "cinema": [t for t in _ALL_TOPICS if t.startswith(NAMESPACE_CINEMA)],
    "visual": [t for t in _ALL_TOPICS if t.startswith(NAMESPACE_VISUAL)],
    "auralux": [t for t in _ALL_TOPICS if t.startswith(NAMESPACE_AURALUX)],
    "avatars": [t for t in _ALL_TOPICS if t.startswith(NAMESPACE_AVATARS)],
    "dits": [t for t in _ALL_TOPICS if t.startswith(NAMESPACE_DITS)],
}


def validate_topic(topic: str) -> tuple[bool, Optional[str]]:
    """
    Validate a topic string.

    Args:
        topic: Topic string to validate.

    Returns:
        Tuple of (is_valid, error_message).
        If valid, error_message is None.

    Example:
        >>> validate_topic("creative.visual.render")
        (True, None)
        >>> validate_topic("invalid.topic")
        (False, "Topic must start with 'creative.'")
    """
    if not topic:
        return False, "Topic cannot be empty"

    if not topic.startswith(NAMESPACE_ROOT + "."):
        return False, f"Topic must start with '{NAMESPACE_ROOT}.'"

    parts = topic.split(".")
    if len(parts) < 3:
        return False, "Topic must have at least 3 parts: creative.<subsystem>.<action>"

    # Check if it's a known topic (but allow unknown for extensibility)
    if topic not in _ALL_TOPICS:
        # Still valid format, just not pre-defined
        pass

    return True, None


def is_creative_topic(topic: str) -> bool:
    """
    Check if a topic belongs to the Creative subsystem.

    Args:
        topic: Topic string to check.

    Returns:
        True if topic starts with 'creative.', False otherwise.

    Example:
        >>> is_creative_topic("creative.visual.render")
        True
        >>> is_creative_topic("agent.status")
        False
    """
    return topic.startswith(NAMESPACE_ROOT + ".")


def get_subsystem_from_topic(topic: str) -> Optional[str]:
    """
    Extract the subsystem name from a topic.

    Args:
        topic: Topic string (e.g., "creative.visual.render").

    Returns:
        Subsystem name (e.g., "visual") or None if invalid.

    Example:
        >>> get_subsystem_from_topic("creative.visual.render")
        'visual'
        >>> get_subsystem_from_topic("creative.grammars.synthesize")
        'grammars'
        >>> get_subsystem_from_topic("invalid")
        None
    """
    if not is_creative_topic(topic):
        return None

    parts = topic.split(".")
    if len(parts) >= 2:
        subsystem = parts[1]
        # Handle special case for generation lifecycle events
        if subsystem == "generation":
            return None  # Cross-cutting concern, not a specific subsystem
        return subsystem

    return None


def list_topics_for_subsystem(subsystem: str) -> list[str]:
    """
    Get all defined topics for a subsystem.

    Args:
        subsystem: Subsystem name (e.g., "visual", "cinema").

    Returns:
        List of topic strings for the subsystem.

    Example:
        >>> topics = list_topics_for_subsystem("visual")
        >>> "creative.visual.render" in topics
        True
    """
    return _SUBSYSTEM_TOPICS.get(subsystem, [])


def get_all_topics() -> list[str]:
    """
    Get all defined topics.

    Returns:
        List of all topic strings.
    """
    return sorted(_ALL_TOPICS)


def get_subsystems() -> list[str]:
    """
    Get all subsystem names.

    Returns:
        List of subsystem names.
    """
    return list(_SUBSYSTEM_TOPICS.keys())
