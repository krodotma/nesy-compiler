"""
Visual Subsystem
================

Image generation, style transfer, diffusion models.
"""

from __future__ import annotations
import importlib
import importlib.util
import sys
from pathlib import Path

_this_dir = Path(__file__).parent
_cache_dir = _this_dir / "__pycache__"


def _load_module(name: str):
    """Load a module, preferring source files over pyc."""
    module_name = f"nucleus.creative.visual.{name}"

    # Check if already loaded
    if module_name in sys.modules:
        return sys.modules[module_name]

    # Try source file first
    source_path = _this_dir / f"{name}.py"
    if source_path.exists():
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                return module
            except Exception:
                del sys.modules[module_name]
                # Fall through to try pyc

    # Try pyc file as fallback
    pyc_pattern = f"{name}.cpython-*.pyc"
    pyc_files = list(_cache_dir.glob(pyc_pattern))
    if pyc_files:
        pyc_path = pyc_files[0]
        spec = importlib.util.spec_from_file_location(module_name, pyc_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
                return module
            except Exception:
                del sys.modules[module_name]

    return None


# Load submodules
generator = _load_module("generator")
style_transfer = _load_module("style_transfer")
upscaler = _load_module("upscaler")
bus_events = _load_module("bus_events")

# Initialize exports with None defaults
ImageGenerationConfig = None
GenerationResult = None
ImageProvider = None
ImageGenerator = None
StyleConfig = None
StyleResult = None
NeuralStyleTransfer = None
STYLE_PRESETS = {}
UpscaleResult = None
RealESRGANUpscaler = None
GFPGANFaceEnhancer = None
Upscaler = None
SimpleUpscaler = None

# Export from generator
if generator:
    ImageGenerationConfig = getattr(generator, "ImageGenerationConfig", None)
    GenerationResult = getattr(generator, "GenerationResult", None)
    ImageProvider = getattr(generator, "ImageProvider", None)
    ImageGenerator = getattr(generator, "ImageGenerator", None)

# Export from style_transfer
if style_transfer:
    StyleConfig = getattr(style_transfer, "StyleConfig", None)
    StyleResult = getattr(style_transfer, "StyleResult", None)
    NeuralStyleTransfer = getattr(style_transfer, "NeuralStyleTransfer", None)
    STYLE_PRESETS = getattr(style_transfer, "STYLE_PRESETS", {})

# Export from upscaler
if upscaler:
    UpscaleResult = getattr(upscaler, "UpscaleResult", None)
    RealESRGANUpscaler = getattr(upscaler, "RealESRGANUpscaler", None)
    GFPGANFaceEnhancer = getattr(upscaler, "GFPGANFaceEnhancer", None)
    Upscaler = getattr(upscaler, "Upscaler", None)
    SimpleUpscaler = getattr(upscaler, "SimpleUpscaler", None)

__all__ = [
    "ImageGenerationConfig", "GenerationResult", "ImageProvider", "ImageGenerator",
    "StyleConfig", "StyleResult", "NeuralStyleTransfer", "STYLE_PRESETS",
    "UpscaleResult", "RealESRGANUpscaler", "GFPGANFaceEnhancer", "Upscaler", "SimpleUpscaler",
    "generator", "style_transfer", "upscaler", "bus_events",
]
