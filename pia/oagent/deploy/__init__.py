#!/usr/bin/env python3
"""
Deploy Agent - CI/CD Pipeline Orchestration

Part of the OAGENT (300-step) multi-agent architecture for the Pluribus system.
Implements Steps 201-250 of the OAGENT_300_STEP_PLAN.

The Deploy Agent handles:
- Build orchestration and artifact packaging
- Container building and registry management
- Environment provisioning (dev/staging/prod)
- Blue-green and canary deployment strategies
- Rollback automation and feature flag management
- Full deployment pipeline orchestration

PBTSO Phases:
- SKILL: Bootstrap and initialization
- SEQUESTER: Environment isolation
- ITERATE: Build/deploy cycles
- DISTILL: Artifact packaging

A2A Bus Topics:
- a2a.deploy.bootstrap.start/complete
- deploy.build.start/complete/failed
- deploy.artifact.package/ready
- deploy.container.build/pushed
- deploy.env.provision/ready
- deploy.bluegreen.switch/rollback
- deploy.canary.start/progress/complete
- deploy.rollback.trigger/complete
- deploy.flag.toggle/status
- a2a.deploy.orchestrate
- deploy.pipeline.complete

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "DeployAgentConfig",
    "DeployAgentBootstrap",
    "BuildOrchestrator",
    "BuildStatus",
    "BuildResult",
    "ArtifactPackager",
    "ContainerBuilder",
    "EnvironmentProvisioner",
    "BlueGreenDeploymentManager",
    "CanaryDeploymentManager",
    "RollbackAutomator",
    "FeatureFlagManager",
    "DeployOrchestrator",
]

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    if name == "DeployAgentConfig":
        from .bootstrap import DeployAgentConfig
        return DeployAgentConfig
    if name == "DeployAgentBootstrap":
        from .bootstrap import DeployAgentBootstrap
        return DeployAgentBootstrap
    if name == "BuildOrchestrator":
        from .build.orchestrator import BuildOrchestrator
        return BuildOrchestrator
    if name == "BuildStatus":
        from .build.orchestrator import BuildStatus
        return BuildStatus
    if name == "BuildResult":
        from .build.orchestrator import BuildResult
        return BuildResult
    if name == "ArtifactPackager":
        from .artifact.packager import ArtifactPackager
        return ArtifactPackager
    if name == "ContainerBuilder":
        from .container.builder import ContainerBuilder
        return ContainerBuilder
    if name == "EnvironmentProvisioner":
        from .env.provisioner import EnvironmentProvisioner
        return EnvironmentProvisioner
    if name == "BlueGreenDeploymentManager":
        from .strategy.blue_green import BlueGreenDeploymentManager
        return BlueGreenDeploymentManager
    if name == "CanaryDeploymentManager":
        from .strategy.canary import CanaryDeploymentManager
        return CanaryDeploymentManager
    if name == "RollbackAutomator":
        from .rollback.automator import RollbackAutomator
        return RollbackAutomator
    if name == "FeatureFlagManager":
        from .flags.manager import FeatureFlagManager
        return FeatureFlagManager
    if name == "DeployOrchestrator":
        from .orchestrator import DeployOrchestrator
        return DeployOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
