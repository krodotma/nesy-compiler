#!/usr/bin/env python3
"""
Analysis module for Research Agent.

Contains reference resolution, impact analysis, and pattern detection.

Steps implemented:
- Step 13: Reference Resolver
- Step 14: Impact Analyzer
- Step 15: Pattern Detector
- Step 16: Architecture Mapper
- Step 17: Knowledge Extractor
"""
from __future__ import annotations

from .reference_resolver import ReferenceResolver, Reference, ResolvedReference
from .impact_analyzer import ImpactAnalyzer, Impact, ImpactReport
from .pattern_detector import PatternDetector, Pattern, PatternMatch
from .architecture_mapper import ArchitectureMapper, ArchitectureMap, Component
from .knowledge_extractor import KnowledgeExtractor, Knowledge, KnowledgeGraph

__all__ = [
    "ReferenceResolver",
    "Reference",
    "ResolvedReference",
    "ImpactAnalyzer",
    "Impact",
    "ImpactReport",
    "PatternDetector",
    "Pattern",
    "PatternMatch",
    "ArchitectureMapper",
    "ArchitectureMap",
    "Component",
    "KnowledgeExtractor",
    "Knowledge",
    "KnowledgeGraph",
]
