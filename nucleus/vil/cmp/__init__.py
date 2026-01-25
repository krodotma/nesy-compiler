"""
VIL CMP Module
Clade Manager Protocol integration for vision-metalearning.

Version: 1.0
Date: 2026-01-25
"""

from .manager import (
    CladeState,
    VisionCMPMetrics,
    CladeCMP,
    VILCMPManager,
    create_vil_cmp_manager,
    PHI,
    CMP_DISCOUNT,
    GLOBAL_CMP_FLOOR,
)

__all__ = [
    "CladeState",
    "VisionCMPMetrics",
    "CladeCMP",
    "VILCMPManager",
    "create_vil_cmp_manager",
    "PHI",
    "CMP_DISCOUNT",
    "GLOBAL_CMP_FLOOR",
]
