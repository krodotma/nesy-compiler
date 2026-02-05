#!/usr/bin/env python3
"""
Review Agent - OAGENT Subagent 4 (Steps 151-200)

Provides code review, static analysis, security scanning, and quality assessment.

Architecture:
- Bootstrap: Agent initialization and A2A registration
- Static Analysis: Code quality via linters (ruff, eslint)
- Security Scanner: OWASP-based vulnerability detection
- Code Smell Detector: Anti-pattern identification
- Architecture Checker: Consistency validation
- Documentation Checker: Completeness verification
- Comment Generator: Review comment synthesis
- Dependency Scanner: CVE vulnerability checking
- License Checker: Compliance validation
- Orchestrator: Pipeline coordination

PBTSO Phases:
- SKILL: Bootstrap initialization
- SEQUESTER: Security sandboxing
- VERIFY: All analysis and checking operations
- DISTILL: Comment generation and summary

Bus Topics:
- a2a.review.bootstrap.start
- a2a.review.bootstrap.complete
- review.static.analyze
- review.security.scan
- review.smells.detect
- review.architecture.check
- review.docs.check
- review.comments.generate
- review.deps.scan
- review.license.check
- a2a.review.orchestrate
- review.pipeline.complete

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from .bootstrap import ReviewAgentBootstrap, ReviewAgentConfig
from .orchestrator import ReviewOrchestrator, ReviewPipeline, ReviewResult

__all__ = [
    "ReviewAgentBootstrap",
    "ReviewAgentConfig",
    "ReviewOrchestrator",
    "ReviewPipeline",
    "ReviewResult",
]

__version__ = "0.1.0"
__step_range__ = "151-160"
