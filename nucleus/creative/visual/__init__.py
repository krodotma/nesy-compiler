"""
Visual Subsystem
================

Image generation, style transfer, diffusion models.
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
    module_name = f"nucleus.creative.visual.{name}"

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
    return None

# Load submodules
generator = _load_from_pyc("generator")
style_transfer = _load_from_pyc("style_transfer")
upscaler = _load_from_pyc("upscaler")
bus_events = _load_from_pyc("bus_events")

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
    "generator", "style_transfer", "upscaler",
]
