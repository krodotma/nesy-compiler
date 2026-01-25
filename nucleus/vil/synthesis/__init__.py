"""
VIL Synthesis Module
Program synthesis from vision using CGP and EGGP.

Version: 1.0
Date: 2026-01-25
"""

from .evolution import (
    GenotypeType,
    Genotype,
    Phenotype,
    SynthesisResult,
    CartesianGeneticProgramming,
    EGGPEvolver,
    ProgramSynthesis,
    create_program_synthesis,
)

__all__ = [
    "GenotypeType",
    "Genotype",
    "Phenotype",
    "SynthesisResult",
    "CartesianGeneticProgramming",
    "EGGPEvolver",
    "ProgramSynthesis",
    "create_program_synthesis",
]
