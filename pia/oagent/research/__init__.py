#!/usr/bin/env python3
"""
Research Agent - Oagent Subagent 1 (Steps 1-50)

The Research Agent handles codebase exploration, documentation retrieval,
knowledge graph population, and context optimization.

PBTSO Phases: SKILL, SEQUESTER, RESEARCH, DISTILL, PLAN

Bus Topics:
- a2a.research.bootstrap.start
- a2a.research.bootstrap.complete
- research.scan.start
- research.scan.progress
- research.scan.complete
- research.parser.register
- research.capability.register
- research.parse.python
- research.parse.typescript
- research.symbols.extracted
- research.index.symbol
- research.query.symbols
- research.docs.extracted
- research.docs.indexed
- research.readme.parsed
- research.project.structure
- research.graph.dependency
- research.imports.resolved
- research.graph.calls
- research.functions.analyzed

Protocol: DKIN v30, PAIP v16, HOLON v2
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "isomorphics"

from .bootstrap import ResearchAgentBootstrap, ResearchAgentConfig
from .scanner import CodebaseScanner

__all__ = [
    "ResearchAgentBootstrap",
    "ResearchAgentConfig",
    "CodebaseScanner",
]
