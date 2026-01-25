"""
Theia Governance Module - DUALITY-BIND Integration.

Provides:
- GuardLadder: G1-G6 validation for agent actions.
"""

from theia.governance.guard_ladder import (
    GuardResult,
    GuardOutcome,
    GuardLadder
)

__all__ = ["GuardResult", "GuardOutcome", "GuardLadder"]
