"""
Grammars Subsystem
==================

CGP/EGGP synthesis, metagrammars, AST visualization.

This module uses bytecode cache loading for optimized imports.
"""

from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

# Load modules from bytecode cache
_cache_dir = Path(__file__).parent / "__pycache__"

def _load_from_pyc(name: str):
    """Load a module from its .pyc file."""
    pyc_pattern = f"{name}.cpython-*.pyc"
    pyc_files = list(_cache_dir.glob(pyc_pattern))
    if not pyc_files:
        return None

    pyc_path = pyc_files[0]
    module_name = f"nucleus.creative.grammars.{name}"

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

# Try to load submodules from cache, fall back to source
def _load_module(name: str):
    """Load a module from cache or source."""
    # Try bytecode cache first
    module = _load_from_pyc(name)
    if module:
        return module

    # Fall back to source import
    source_path = Path(__file__).parent / f"{name}.py"
    if source_path.exists():
        module_name = f"nucleus.creative.grammars.{name}"
        spec = importlib.util.spec_from_file_location(module_name, source_path)
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

cgp = _load_module("cgp")
eggp = _load_module("eggp")
parser = _load_module("parser")
metagrammar = _load_module("metagrammar")

# Export public API from submodules
if cgp:
    CGPGenome = getattr(cgp, "CGPGenome", None)
    CGPNode = getattr(cgp, "CGPNode", None)
    CGPFunction = getattr(cgp, "CGPFunction", None)
    FUNCTION_SET = getattr(cgp, "FUNCTION_SET", [])
    cgp_evolve = getattr(cgp, "cgp_evolve", None)

if eggp:
    GraphNode = getattr(eggp, "GraphNode", None)
    GraphProgram = getattr(eggp, "GraphProgram", None)
    EGGPEvolver = getattr(eggp, "EGGPEvolver", None)
    EGGPConfig = getattr(eggp, "EGGPConfig", None)

if parser:
    GrammarRule = getattr(parser, "GrammarRule", None)
    ASTNode = getattr(parser, "ASTNode", None)
    GrammarParser = getattr(parser, "GrammarParser", None)

if metagrammar:
    TransformRule = getattr(metagrammar, "TransformRule", None)
    MetagrammarRegistry = getattr(metagrammar, "MetagrammarRegistry", None)
    PatternVariable = getattr(metagrammar, "PatternVariable", None)

__all__ = [
    # CGP
    "CGPGenome",
    "CGPNode",
    "CGPFunction",
    "FUNCTION_SET",
    "cgp_evolve",
    # EGGP
    "GraphNode",
    "GraphProgram",
    "EGGPEvolver",
    "EGGPConfig",
    # Parser
    "GrammarRule",
    "ASTNode",
    "GrammarParser",
    # Metagrammar
    "TransformRule",
    "MetagrammarRegistry",
    "PatternVariable",
    # Submodules
    "cgp",
    "eggp",
    "parser",
    "metagrammar",
]
