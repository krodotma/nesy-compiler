#!/usr/bin/env python3
"""
API module for Research Agent (Step 29)

Contains the REST API server for research queries.
"""
from __future__ import annotations

from .research_api import ResearchAPI, APIConfig

__all__ = [
    "ResearchAPI",
    "APIConfig",
]
