#!/usr/bin/env python3
"""
Flags submodule for Deploy Agent.

Provides feature flag management infrastructure.
"""
from __future__ import annotations

from .manager import FeatureFlagManager, FeatureFlag, FlagType, FlagState

__all__ = ["FeatureFlagManager", "FeatureFlag", "FlagType", "FlagState"]
