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
- Deployment metrics and analytics (Step 221)
- Deployment reports (Step 222)
- Scheduled deployments (Step 223)
- Approval workflows (Step 224)
- Deployment notifications (Step 225)
- Deployment history/audit (Step 226)
- Deployment comparison (Step 227)
- CI/CD integrations (Step 228)
- REST API (Step 229)
- CLI interface (Step 230)

PBTSO Phases:
- SKILL: Bootstrap and initialization
- SEQUESTER: Environment isolation
- ITERATE: Build/deploy cycles
- DISTILL: Artifact packaging
- PLAN: Scheduling and approvals
- VERIFY: Metrics and comparison
- DISTRIBUTE: API and integrations

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
- deploy.metrics.record/aggregate/alert
- deploy.reports.generate/complete
- deploy.scheduler.create/trigger
- deploy.approval.request/approve/reject
- deploy.notifications.send/sent
- deploy.history.record/event
- deploy.comparison.compare/diff
- deploy.integration.trigger/webhook
- deploy.api.request/response

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

__version__ = "0.2.0"
__all__ = [
    # Bootstrap (Step 201)
    "DeployAgentConfig",
    "DeployAgentBootstrap",
    # Build (Step 202)
    "BuildOrchestrator",
    "BuildStatus",
    "BuildResult",
    # Artifact (Step 203)
    "ArtifactPackager",
    # Container (Step 204)
    "ContainerBuilder",
    # Environment (Step 205)
    "EnvironmentProvisioner",
    # Strategies (Steps 206-207)
    "BlueGreenDeploymentManager",
    "CanaryDeploymentManager",
    # Rollback (Step 208)
    "RollbackAutomator",
    # Feature Flags (Step 209)
    "FeatureFlagManager",
    # Orchestrator (Step 210)
    "DeployOrchestrator",
    # Metrics (Step 221)
    "DeploymentMetricsCollector",
    # Reports (Step 222)
    "DeploymentReportGenerator",
    # Scheduler (Step 223)
    "DeploymentScheduler",
    # Approval (Step 224)
    "DeploymentApprovalGate",
    # Notifications (Step 225)
    "DeploymentNotificationSystem",
    # History (Step 226)
    "DeploymentHistoryTracker",
    # Comparison (Step 227)
    "DeploymentComparator",
    # Integration (Step 228)
    "DeploymentIntegrationHub",
    # API (Step 229)
    "DeploymentAPI",
    # CLI (Step 230)
    "DeployCLI",
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
    # Step 221: Metrics Collector
    if name == "DeploymentMetricsCollector":
        from .metrics.collector import DeploymentMetricsCollector
        return DeploymentMetricsCollector
    # Step 222: Report Generator
    if name == "DeploymentReportGenerator":
        from .reports.generator import DeploymentReportGenerator
        return DeploymentReportGenerator
    # Step 223: Scheduler
    if name == "DeploymentScheduler":
        from .scheduler.scheduler import DeploymentScheduler
        return DeploymentScheduler
    # Step 224: Approval Gate
    if name == "DeploymentApprovalGate":
        from .approval.gate import DeploymentApprovalGate
        return DeploymentApprovalGate
    # Step 225: Notification System
    if name == "DeploymentNotificationSystem":
        from .notifications.notifier import DeploymentNotificationSystem
        return DeploymentNotificationSystem
    # Step 226: History Tracker
    if name == "DeploymentHistoryTracker":
        from .history.tracker import DeploymentHistoryTracker
        return DeploymentHistoryTracker
    # Step 227: Comparator
    if name == "DeploymentComparator":
        from .comparison.comparator import DeploymentComparator
        return DeploymentComparator
    # Step 228: Integration Hub
    if name == "DeploymentIntegrationHub":
        from .integration.hub import DeploymentIntegrationHub
        return DeploymentIntegrationHub
    # Step 229: API
    if name == "DeploymentAPI":
        from .api.server import DeploymentAPI
        return DeploymentAPI
    # Step 230: CLI
    if name == "DeployCLI":
        from .cli.main import DeployCLI
        return DeployCLI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
