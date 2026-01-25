"""
Theia Memory Layer (L2) — Modern Hopfield Continuum.

Energy: E(x) = -Σ log(Σ exp(β⟨x,ξ⟩))
Dynamics: ẋ = -∇E(x) = softmax retrieval
Capacity: Exponential in dimension

Attractors ≅ stored patterns
Attention IS Hopfield update (Ramsauer et al.)

CTM-Derived Extensions:
- Temporal sync for ARC prescience (Darlow et al., arXiv:2505.05522)
"""

from theia.memory.hopfield import (
    HopfieldMemory,
    create_memory,
    store_patterns,
    capacity_estimate,
)

from theia.memory.attractor import (
    Attractor,
    AttractorLandscape,
    create_icl_memory,
)

from theia.memory.temporal_sync import (
    SyncPhase,
    TemporalNeuron,
    SyncGroup,
    TemporalHierarchy,
    PrescienceState,
    ARCPrescienceEngine,
    create_sync_meter_data,
    create_hierarchy_viz,
)

__all__ = [
    # Hopfield
    "HopfieldMemory",
    "create_memory",
    "store_patterns",
    "capacity_estimate",
    # Attractor
    "Attractor",
    "AttractorLandscape",
    "create_icl_memory",
    # CTM-Derived Temporal Sync
    "SyncPhase",
    "TemporalNeuron",
    "SyncGroup",
    "TemporalHierarchy",
    "PrescienceState",
    "ARCPrescienceEngine",
    "create_sync_meter_data",
    "create_hierarchy_viz",
]


