#!/usr/bin/env python3
"""
Feedback module for Research Agent (Steps 26-27)

Contains feedback integration and incremental update components.
"""
from __future__ import annotations

from .feedback_integrator import FeedbackIntegrator, FeedbackConfig, Feedback
from .incremental_updater import IncrementalUpdater, UpdaterConfig, UpdateEvent

__all__ = [
    "FeedbackIntegrator",
    "FeedbackConfig",
    "Feedback",
    "IncrementalUpdater",
    "UpdaterConfig",
    "UpdateEvent",
]
