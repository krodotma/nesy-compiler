"""
pluribus_evolution.observer

Observes the primary pluribus trunk for patterns, drift, and optimization opportunities.

Components:
- code_analyzer: AST analysis, pattern detection
- vector_profiler: Embedding/manifold analysis
- drift_detector: Temporal drift monitoring (genotype/phenotype)
"""

from __future__ import annotations

__all__ = [
    "CodeAnalyzer",
    "VectorProfiler",
    "DriftDetector",
    "AnalysisResult",
    "CodePattern",
    "DriftReport",
    "DriftSignal",
    "VectorProfile",
    "ManifoldSnapshot",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "CodeAnalyzer":
        from .code_analyzer import CodeAnalyzer
        return CodeAnalyzer
    if name == "AnalysisResult":
        from .code_analyzer import AnalysisResult
        return AnalysisResult
    if name == "CodePattern":
        from .code_analyzer import CodePattern
        return CodePattern
    if name == "VectorProfiler":
        from .vector_profiler import VectorProfiler
        return VectorProfiler
    if name == "VectorProfile":
        from .vector_profiler import VectorProfile
        return VectorProfile
    if name == "ManifoldSnapshot":
        from .vector_profiler import ManifoldSnapshot
        return ManifoldSnapshot
    if name == "DriftDetector":
        from .drift_detector import DriftDetector
        return DriftDetector
    if name == "DriftReport":
        from .drift_detector import DriftReport
        return DriftReport
    if name == "DriftSignal":
        from .drift_detector import DriftSignal
        return DriftSignal
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
