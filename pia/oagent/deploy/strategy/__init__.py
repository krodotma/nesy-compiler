#!/usr/bin/env python3
"""
Strategy submodule for Deploy Agent.

Provides deployment strategy implementations (blue-green, canary).
"""
from __future__ import annotations

from .blue_green import BlueGreenDeploymentManager, BlueGreenState, SlotType
from .canary import CanaryDeploymentManager, CanaryState, CanaryConfig

__all__ = [
    "BlueGreenDeploymentManager",
    "BlueGreenState",
    "SlotType",
    "CanaryDeploymentManager",
    "CanaryState",
    "CanaryConfig",
]
