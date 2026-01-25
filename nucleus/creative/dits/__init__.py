"""
DiTS Subsystem
==============

Diegetic Transition System, narrative constructivity, μ/ν calculus.
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
    module_name = f"nucleus.creative.dits.{name}"

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
kernel = _load_from_pyc("kernel")
narrative = _load_from_pyc("narrative")
rheomode = _load_from_pyc("rheomode")
omega_bridge = _load_from_pyc("omega_bridge")
spec_loader = _load_from_pyc("spec_loader")

# Export from kernel
if kernel:
    DiTSSpec = getattr(kernel, "DiTSSpec", None)
    DiTSState = getattr(kernel, "DiTSState", None)
    DiTSKernel = getattr(kernel, "DiTSKernel", None)

# Export from narrative
if narrative:
    Episode = getattr(narrative, "Episode", None)
    Narrative = getattr(narrative, "Narrative", None)
    NarrativeEngine = getattr(narrative, "NarrativeEngine", None)

# Export from rheomode
if rheomode:
    VerbInfo = getattr(rheomode, "VerbInfo", None)
    RheomodeFlow = getattr(rheomode, "RheomodeFlow", None)
    RheomodeEngine = getattr(rheomode, "RheomodeEngine", None)

# Export from omega_bridge
if omega_bridge:
    OmegaState = getattr(omega_bridge, "OmegaState", None)
    OmegaBridge = getattr(omega_bridge, "OmegaBridge", None)

__all__ = [
    "DiTSSpec", "DiTSState", "DiTSKernel",
    "Episode", "Narrative", "NarrativeEngine",
    "VerbInfo", "RheomodeFlow", "RheomodeEngine",
    "OmegaState", "OmegaBridge",
    "kernel", "narrative", "rheomode", "omega_bridge", "spec_loader",
]
