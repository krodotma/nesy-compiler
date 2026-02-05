#!/usr/bin/env python3
"""
Research Agent - Oagent Subagent 1 (Steps 1-50)

The Research Agent handles codebase exploration, documentation retrieval,
knowledge graph population, and context optimization.

PBTSO Phases: SKILL, SEQUESTER, RESEARCH, DISTILL, PLAN, ITERATE, INTERFACE

Steps 21-30 (This module adds):
- Step 21: Query Executor - Execute planned research queries
- Step 22: Result Ranker - Rank and prioritize results
- Step 23: Answer Synthesizer - Generate coherent answers
- Step 24: Citation Generator - Track and format citations
- Step 25: Confidence Scorer - Score answer confidence
- Step 26: Feedback Integrator - Learn from feedback
- Step 27: Incremental Updater - Update indexes incrementally
- Step 28: Multi-Repo Manager - Handle multiple repositories
- Step 29: Research API - REST API for research queries
- Step 30: Research CLI - Complete CLI interface

Bus Topics:
- a2a.research.bootstrap.start
- a2a.research.bootstrap.complete
- a2a.research.execute.start
- a2a.research.execute.complete
- a2a.research.rank.start
- a2a.research.rank.complete
- a2a.research.synthesize.start
- a2a.research.synthesize.complete
- a2a.research.cite.generate
- a2a.research.confidence.score
- a2a.research.feedback.receive
- a2a.research.update.start
- a2a.research.repo.register
- a2a.research.api.request
- a2a.research.cli.execute
- research.scan.start
- research.scan.progress
- research.scan.complete
- research.parser.register
- research.capability.register
- research.index.symbol
- research.query.symbols

Protocol: DKIN v30, PAIP v16, HOLON v2, CITIZEN v2
"""
from __future__ import annotations

__version__ = "0.2.0"
__author__ = "isomorphics"

from .bootstrap import ResearchAgentBootstrap, ResearchAgentConfig
from .scanner import CodebaseScanner

# Steps 21-25: Executor components
from .executor import (
    QueryExecutor,
    ExecutionConfig,
    ExecutedQuery,
    ResultRanker,
    RankerConfig,
    RankedResult,
    AnswerSynthesizer,
    SynthesizerConfig,
    SynthesizedAnswer,
    CitationGenerator,
    CitationConfig,
    Citation,
    ConfidenceScorer,
    ScorerConfig,
    ConfidenceScore,
)

# Steps 26-27: Feedback components
from .feedback import (
    FeedbackIntegrator,
    FeedbackConfig,
    Feedback,
    IncrementalUpdater,
    UpdaterConfig,
    UpdateEvent,
)

# Step 28: Multi-repo management
from .multi_repo import (
    MultiRepoManager,
    RepoConfig,
    Repository,
)

# Step 29: API
from .api import (
    ResearchAPI,
    APIConfig,
)

# Step 30: CLI
from .cli import (
    ResearchCLI,
)

__all__ = [
    # Core
    "ResearchAgentBootstrap",
    "ResearchAgentConfig",
    "CodebaseScanner",
    # Step 21: Query Executor
    "QueryExecutor",
    "ExecutionConfig",
    "ExecutedQuery",
    # Step 22: Result Ranker
    "ResultRanker",
    "RankerConfig",
    "RankedResult",
    # Step 23: Answer Synthesizer
    "AnswerSynthesizer",
    "SynthesizerConfig",
    "SynthesizedAnswer",
    # Step 24: Citation Generator
    "CitationGenerator",
    "CitationConfig",
    "Citation",
    # Step 25: Confidence Scorer
    "ConfidenceScorer",
    "ScorerConfig",
    "ConfidenceScore",
    # Step 26: Feedback Integrator
    "FeedbackIntegrator",
    "FeedbackConfig",
    "Feedback",
    # Step 27: Incremental Updater
    "IncrementalUpdater",
    "UpdaterConfig",
    "UpdateEvent",
    # Step 28: Multi-Repo Manager
    "MultiRepoManager",
    "RepoConfig",
    "Repository",
    # Step 29: Research API
    "ResearchAPI",
    "APIConfig",
    # Step 30: Research CLI
    "ResearchCLI",
]
