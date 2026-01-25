"""
Cinema Subsystem
================

Video generation, temporal consistency, multi-shot narrative.
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

_cache_dir = Path(__file__).parent / "__pycache__"

def _load_from_pyc(name: str):
    """Load a module from its .pyc file."""
    pyc_pattern = f"{name}.cpython-*.pyc"
    pyc_files = list(_cache_dir.glob(pyc_pattern))
    if not pyc_files:
        return None

    pyc_path = pyc_files[0]
    module_name = f"nucleus.creative.cinema.{name}"

    spec = importlib.util.spec_from_file_location(module_name, pyc_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            # Keep module in sys.modules even if exec fails
            # This allows attribute access even with partial loading
            return module
    return None

# Load submodules
script_parser = _load_from_pyc("script_parser")
storyboard = _load_from_pyc("storyboard")
frame_generator = _load_from_pyc("frame_generator")
temporal_consistency = _load_from_pyc("temporal_consistency")
video_assembler = _load_from_pyc("video_assembler")
bus_events = _load_from_pyc("bus_events")

# Export from script_parser
if script_parser:
    SceneElement = getattr(script_parser, "SceneElement", None)
    Script = getattr(script_parser, "Script", None)
    FountainParser = getattr(script_parser, "FountainParser", None)

# Export from storyboard
if storyboard:
    ShotType = getattr(storyboard, "ShotType", None)
    StoryboardPanel = getattr(storyboard, "StoryboardPanel", None)
    Storyboard = getattr(storyboard, "Storyboard", None)
    StoryboardGenerator = getattr(storyboard, "StoryboardGenerator", None)

# Export from frame_generator
if frame_generator:
    FrameGenerationConfig = getattr(frame_generator, "FrameGenerationConfig", None)
    GenerationResult = getattr(frame_generator, "GenerationResult", None)
    FrameGenerator = getattr(frame_generator, "FrameGenerator", None)

# Export from temporal_consistency
if temporal_consistency:
    TemporalConsistencyEngine = getattr(temporal_consistency, "TemporalConsistencyEngine", None)
    FlowMethod = getattr(temporal_consistency, "FlowMethod", None)

__all__ = [
    "SceneElement", "Script", "FountainParser",
    "ShotType", "StoryboardPanel", "Storyboard", "StoryboardGenerator",
    "FrameGenerationConfig", "GenerationResult", "FrameGenerator",
    "TemporalConsistencyEngine", "FlowMethod",
    "script_parser", "storyboard", "frame_generator", "temporal_consistency", "video_assembler",
]
