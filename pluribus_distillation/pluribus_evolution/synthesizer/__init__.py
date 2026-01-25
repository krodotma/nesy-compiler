"""
pluribus_evolution.synthesizer

Generates refined code, tests, and documentation from proposals.

Components:
- patch_generator: Generates code patches from proposals
- test_generator: Generates tests for proposed changes
- doc_synthesizer: Generates documentation updates
"""

from __future__ import annotations

__all__ = [
    "PatchGenerator",
    "TestGenerator",
    "DocSynthesizer",
]

# Lazy imports
def __getattr__(name: str):
    if name == "PatchGenerator":
        from .patch_generator import PatchGenerator
        return PatchGenerator
    if name == "TestGenerator":
        from .test_generator import TestGenerator
        return TestGenerator
    if name == "DocSynthesizer":
        from .doc_synthesizer import DocSynthesizer
        return DocSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
