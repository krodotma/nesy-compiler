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
from .pipeline import (
    SynthesisMethod,
    SynthesisRequest,
    SynthesisTracking,
    SynthesisPipelineIntegrator,
    create_synthesis_pipeline_integrator,
)
from .distillation import (
    TeacherModel,
    TeacherOutput,
    DistillationBatch,
    DistillationResult,
    VLMDistiller,
    create_vlm_distiller,
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
    "SynthesisMethod",
    "SynthesisRequest",
    "SynthesisTracking",
    "SynthesisPipelineIntegrator",
    "create_synthesis_pipeline_integrator",
    "TeacherModel",
    "TeacherOutput",
    "DistillationBatch",
    "DistillationResult",
    "VLMDistiller",
    "create_vlm_distiller",
]
