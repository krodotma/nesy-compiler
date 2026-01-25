"""
pluribus_evolution.bridge

Connects the evolution trunk to the primary pluribus trunk.

Components:
- bus_mirror: Mirrors bus events for analysis
- rhizome_sync: Synchronizes rhizome content-addressed storage
"""

from __future__ import annotations

__all__ = [
    "BusMirror",
    "RhizomeSync",
]
