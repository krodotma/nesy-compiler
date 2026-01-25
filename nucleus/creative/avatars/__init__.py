"""
Avatars Subsystem
=================

3DGS, Neural PBR, SMPL-X, procedural parameters.
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
    module_name = f"nucleus.creative.avatars.{name}"

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
smplx_extractor = _load_from_pyc("smplx_extractor")
gaussian_splatting = _load_from_pyc("gaussian_splatting")
deformable = _load_from_pyc("deformable")
neural_pbr = _load_from_pyc("neural_pbr")
procedural = _load_from_pyc("procedural")
bus_events = _load_from_pyc("bus_events")

# Export from smplx_extractor
if smplx_extractor:
    SMPLXParams = getattr(smplx_extractor, "SMPLXParams", None)
    SMPLXExtractor = getattr(smplx_extractor, "SMPLXExtractor", None)

# Export from gaussian_splatting
if gaussian_splatting:
    Gaussian3D = getattr(gaussian_splatting, "Gaussian3D", None)
    GaussianSplatCloud = getattr(gaussian_splatting, "GaussianSplatCloud", None)
    GaussianCloudOperations = getattr(gaussian_splatting, "GaussianCloudOperations", None)

# Export from deformable
if deformable:
    DeformableGaussians = getattr(deformable, "DeformableGaussians", None)
    DeformationResult = getattr(deformable, "DeformationResult", None)

__all__ = [
    "SMPLXParams", "SMPLXExtractor",
    "Gaussian3D", "GaussianSplatCloud", "GaussianCloudOperations",
    "DeformableGaussians", "DeformationResult",
    "smplx_extractor", "gaussian_splatting", "deformable", "neural_pbr", "procedural",
]
