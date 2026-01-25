"""
DiTS Subsystem
==============

Diegetic Transition System, narrative constructivity, mu/nu calculus.

The DiTS subsystem provides:
- kernel: Core mu/nu calculus transition system
- narrative: Episode and narrative construction
- rheomode: Bohm-inspired flowing language processing
- omega_bridge: Bridge to higher-level reflexive systems
- spec_loader: Specification loading and validation

Example usage:
    >>> from nucleus.creative.dits import DiTSSpec, DiTSKernel, NarrativeEngine
    >>> spec = DiTSSpec(name="story")
    >>> kernel = DiTSKernel(spec)
    >>> engine = NarrativeEngine()
"""

from __future__ import annotations

# Import submodules
from . import kernel
from . import narrative
from . import rheomode
from . import omega_bridge
from . import spec_loader

# Export from kernel
from .kernel import (
    DiTSSpec,
    DiTSState,
    DiTSKernel,
    # Additional kernel exports
    StateId,
    TransitionLabel,
    FixpointType,
    TransitionType,
    EvaluationMode,
    EvaluationResult,
    Formula,
    AtomicFormula,
    StateSetFormula,
    VariableFormula,
    NotFormula,
    AndFormula,
    OrFormula,
    DiamondFormula,
    BoxFormula,
    MuFormula,
    NuFormula,
)

# Export from narrative
from .narrative import (
    Episode,
    Narrative,
    NarrativeEngine,
    # Additional narrative exports
    NarrativeMode,
    EpisodeType,
    NarrativeArc,
    EmotionalTone,
    NarrativeEdge,
    GenerationConfig,
    NarrativeEvent,
)

# Export from rheomode
from .rheomode import (
    VerbInfo,
    RheomodeFlow,
    RheomodeEngine,
    # Additional rheomode exports
    VerbMode,
    FlowState,
    AttentionLevel,
    FlowElement,
    RheomodeConfig,
)

# Export from omega_bridge
from .omega_bridge import (
    OmegaState,
    OmegaBridge,
    # Additional omega_bridge exports
    OmegaLevel,
    IntegrationMode,
    CoherenceStatus,
    OmegaEventType,
    OmegaEvent,
    ReflexiveLoop,
    EmergenceSignal,
    BridgeConfig,
)

# Export from spec_loader
from .spec_loader import (
    SpecLoader,
    LoadResult,
    ValidationResult,
    # Additional spec_loader exports
    SpecFormat,
    ValidationLevel,
    ValidationError,
    LoaderConfig,
    PRESET_SPECS,
    load_from_dict,
    load_from_json,
    load_from_yaml,
    load_preset,
    list_presets,
    validate_spec,
    load_spec,
    create_spec,
)

__all__ = [
    # Core types
    "DiTSSpec", "DiTSState", "DiTSKernel",
    "Episode", "Narrative", "NarrativeEngine",
    "VerbInfo", "RheomodeFlow", "RheomodeEngine",
    "OmegaState", "OmegaBridge",
    "SpecLoader", "LoadResult", "ValidationResult",
    # Submodules
    "kernel", "narrative", "rheomode", "omega_bridge", "spec_loader",
    # Kernel types
    "StateId", "TransitionLabel", "FixpointType", "TransitionType",
    "EvaluationMode", "EvaluationResult",
    "Formula", "AtomicFormula", "StateSetFormula", "VariableFormula",
    "NotFormula", "AndFormula", "OrFormula",
    "DiamondFormula", "BoxFormula", "MuFormula", "NuFormula",
    # Narrative types
    "NarrativeMode", "EpisodeType", "NarrativeArc", "EmotionalTone",
    "NarrativeEdge", "GenerationConfig", "NarrativeEvent",
    # Rheomode types
    "VerbMode", "FlowState", "AttentionLevel",
    "FlowElement", "RheomodeConfig",
    # Omega bridge types
    "OmegaLevel", "IntegrationMode", "CoherenceStatus", "OmegaEventType",
    "OmegaEvent", "ReflexiveLoop", "EmergenceSignal", "BridgeConfig",
    # Spec loader types
    "SpecFormat", "ValidationLevel", "ValidationError", "LoaderConfig",
    "PRESET_SPECS", "load_from_dict", "load_from_json", "load_from_yaml",
    "load_preset", "list_presets", "validate_spec", "load_spec", "create_spec",
]

__version__ = "1.0.0"
