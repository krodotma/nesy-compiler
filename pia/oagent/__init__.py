#!/usr/bin/env python3
"""
oagent - Oagent Multi-Agent System (300-Step Plan)

The Oagent system consists of 6 specialized subagents:
1. Research Agent (Steps 1-50): Codebase exploration, documentation, knowledge graphs
2. Planning Agent (Steps 51-100): Task decomposition, strategy, resource allocation
3. Coding Agent (Steps 101-150): Code generation, refactoring, implementation
4. Testing Agent (Steps 151-200): Test generation, execution, coverage analysis
5. Review Agent (Steps 201-250): Code review, quality analysis, suggestions
6. Integration Agent (Steps 251-300): Deployment, CI/CD, documentation generation

This package provides the core infrastructure and subagent implementations
for the PBTSO 9-phase pipeline.

Protocol: DKIN v30, PAIP v16, HOLON v2
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "isomorphics"

# Subagent imports
from .research import (
    ResearchAgentBootstrap,
    ResearchAgentConfig,
    CodebaseScanner,
)

from .code import (
    CodeAgentBootstrap,
    CodeAgentConfig,
)

from .deploy import (
    DeployAgentBootstrap,
    DeployAgentConfig,
    BuildOrchestrator,
    ArtifactPackager,
    ContainerBuilder,
    EnvironmentProvisioner,
    BlueGreenDeploymentManager,
    CanaryDeploymentManager,
    RollbackAutomator,
    FeatureFlagManager,
    DeployOrchestrator,
)

from .review import (
    ReviewAgentBootstrap,
    ReviewAgentConfig,
    ReviewOrchestrator,
    ReviewPipeline,
    ReviewResult,
)

__all__ = [
    # Research Agent
    "ResearchAgentBootstrap",
    "ResearchAgentConfig",
    "CodebaseScanner",
    # Code Agent
    "CodeAgentBootstrap",
    "CodeAgentConfig",
    # Deploy Agent (Steps 201-210)
    "DeployAgentBootstrap",
    "DeployAgentConfig",
    "BuildOrchestrator",
    "ArtifactPackager",
    "ContainerBuilder",
    "EnvironmentProvisioner",
    "BlueGreenDeploymentManager",
    "CanaryDeploymentManager",
    "RollbackAutomator",
    "FeatureFlagManager",
    "DeployOrchestrator",
    # Review Agent (Steps 151-160)
    "ReviewAgentBootstrap",
    "ReviewAgentConfig",
    "ReviewOrchestrator",
    "ReviewPipeline",
    "ReviewResult",
]
