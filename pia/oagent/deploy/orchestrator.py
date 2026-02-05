#!/usr/bin/env python3
"""
orchestrator.py - Deploy Agent Orchestrator (Step 210)

PBTSO Phase: PLAN, DISTRIBUTE
A2A Integration: Orchestrates deployment pipeline via a2a.deploy.orchestrate

This is the main orchestrator that coordinates all deploy agent components:
- Build Orchestrator
- Artifact Packager
- Container Builder
- Environment Provisioner
- Blue-Green Deployment Manager
- Canary Deployment Manager
- Rollback Automator
- Feature Flag Manager

Bus Topics:
- a2a.deploy.orchestrate
- deploy.pipeline.start
- deploy.pipeline.complete
- deploy.pipeline.failed

Protocol: DKIN v30, CITIZEN v2, PAIP v16, HOLON v2
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Local imports
from .bootstrap import DeployAgentConfig, DeployAgentBootstrap
from .build.orchestrator import BuildOrchestrator, BuildConfig, BuildResult, BuildStatus
from .artifact.packager import ArtifactPackager, ArtifactManifest, PackageFormat
from .container.builder import ContainerBuilder, ContainerImage, ContainerBuildResult
from .env.provisioner import EnvironmentProvisioner, Environment, EnvironmentConfig, EnvironmentState
from .strategy.blue_green import BlueGreenDeploymentManager, BlueGreenState, SlotType
from .strategy.canary import CanaryDeploymentManager, CanaryState, CanaryConfig
from .rollback.automator import RollbackAutomator, RollbackConfig, RollbackRecord, RollbackTrigger
from .flags.manager import FeatureFlagManager, FeatureFlag, FlagType, FlagState


# ==============================================================================
# Bus Emission Helper
# ==============================================================================

def _get_bus_path() -> Path:
    """Get the bus event file path."""
    pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
    return Path(bus_dir) / "events.ndjson"


def _emit_bus_event(
    topic: str,
    data: Dict[str, Any],
    kind: str = "event",
    level: str = "info",
    actor: str = "deploy-orchestrator"
) -> str:
    """Emit an event to the Pluribus bus."""
    bus_path = _get_bus_path()
    bus_path.parent.mkdir(parents=True, exist_ok=True)

    event_id = str(uuid.uuid4())
    event = {
        "id": event_id,
        "ts": time.time(),
        "iso": datetime.now(timezone.utc).isoformat() + "Z",
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "data": data,
    }

    try:
        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    except IOError:
        pass

    return event_id


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class DeployStrategy(Enum):
    """Deployment strategy types."""
    ROLLING = "rolling"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    IMMEDIATE = "immediate"


class PipelinePhase(Enum):
    """Deployment pipeline phases."""
    INIT = "init"
    BUILD = "build"
    PACKAGE = "package"
    CONTAINERIZE = "containerize"
    PROVISION = "provision"
    DEPLOY = "deploy"
    VERIFY = "verify"
    PROMOTE = "promote"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class PipelineConfig:
    """
    Configuration for a deployment pipeline.

    Attributes:
        name: Pipeline name
        service_name: Service being deployed
        version: Version to deploy
        source_dir: Source directory for build
        strategy: Deployment strategy
        target_environments: Target environments
        build_config: Build configuration
        container_enabled: Whether to build container
        canary_config: Canary configuration (if strategy is canary)
        rollback_config: Rollback configuration
        feature_flags: Feature flags to toggle
        timeout_s: Pipeline timeout in seconds
    """
    name: str
    service_name: str
    version: str
    source_dir: str = "."
    strategy: DeployStrategy = DeployStrategy.BLUE_GREEN
    target_environments: List[str] = field(default_factory=lambda: ["staging"])
    build_config: Optional[BuildConfig] = None
    container_enabled: bool = True
    canary_config: Optional[CanaryConfig] = None
    rollback_config: Optional[RollbackConfig] = None
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    timeout_s: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "service_name": self.service_name,
            "version": self.version,
            "source_dir": self.source_dir,
            "strategy": self.strategy.value,
            "target_environments": self.target_environments,
            "build_config": self.build_config.to_dict() if self.build_config else None,
            "container_enabled": self.container_enabled,
            "canary_config": self.canary_config.to_dict() if self.canary_config else None,
            "rollback_config": self.rollback_config.to_dict() if self.rollback_config else None,
            "feature_flags": self.feature_flags,
            "timeout_s": self.timeout_s,
        }


@dataclass
class PipelineState:
    """
    State of a deployment pipeline.

    Attributes:
        pipeline_id: Unique pipeline identifier
        config: Pipeline configuration
        phase: Current phase
        started_at: Timestamp when started
        completed_at: Timestamp when completed
        build_result: Build result
        artifact: Artifact manifest
        container_image: Container image
        environments: Provisioned environments
        deployment_state: Blue-green or canary state
        error: Error message if failed
        metrics: Pipeline metrics
    """
    pipeline_id: str
    config: PipelineConfig
    phase: PipelinePhase = PipelinePhase.INIT
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    build_result: Optional[BuildResult] = None
    artifact: Optional[ArtifactManifest] = None
    container_image: Optional[ContainerImage] = None
    environments: List[EnvironmentState] = field(default_factory=list)
    deployment_state: Optional[Union[BlueGreenState, CanaryState]] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "config": self.config.to_dict(),
            "phase": self.phase.value,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "build_result": self.build_result.to_dict() if self.build_result else None,
            "artifact": self.artifact.to_dict() if self.artifact else None,
            "container_image": self.container_image.to_dict() if self.container_image else None,
            "environments": [e.to_dict() for e in self.environments],
            "deployment_state": self.deployment_state.to_dict() if self.deployment_state else None,
            "error": self.error,
            "metrics": self.metrics,
        }


# ==============================================================================
# Deploy Agent Orchestrator (Step 210)
# ==============================================================================

class DeployOrchestrator:
    """
    Deploy Agent Orchestrator - coordinates deployment pipeline.

    PBTSO Phase: PLAN, DISTRIBUTE

    This is the main orchestrator that ties together all deploy agent components
    to execute complete deployment pipelines.

    Pipeline Flow:
    1. INIT - Initialize pipeline and validate configuration
    2. BUILD - Build the application
    3. PACKAGE - Package artifacts
    4. CONTAINERIZE - Build container image (optional)
    5. PROVISION - Provision target environment(s)
    6. DEPLOY - Deploy using selected strategy
    7. VERIFY - Verify deployment health
    8. PROMOTE - Promote to 100% traffic (for canary)
    9. COMPLETE - Pipeline complete

    Example:
        >>> orchestrator = DeployOrchestrator()
        >>> config = PipelineConfig(
        ...     name="myapp-deploy",
        ...     service_name="myapp",
        ...     version="v2.0.0",
        ...     source_dir="/app",
        ...     strategy=DeployStrategy.BLUE_GREEN,
        ... )
        >>> state = await orchestrator.run_pipeline(config)
        >>> print(f"Pipeline {state.phase.value}")
    """

    BUS_TOPICS = {
        "orchestrate": "a2a.deploy.orchestrate",
        "pipeline_start": "deploy.pipeline.start",
        "pipeline_complete": "deploy.pipeline.complete",
        "pipeline_failed": "deploy.pipeline.failed",
        "phase_change": "deploy.pipeline.phase",
    }

    def __init__(
        self,
        agent_config: Optional[DeployAgentConfig] = None,
        state_dir: Optional[str] = None,
        actor_id: str = "deploy-orchestrator",
    ):
        """
        Initialize the deploy orchestrator.

        Args:
            agent_config: Deploy agent configuration
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        self.agent_config = agent_config or DeployAgentConfig()

        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            self.state_dir = Path(self.agent_config.state_dir) / "pipelines"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Initialize component managers
        self._bootstrap = DeployAgentBootstrap(self.agent_config)
        self._build_orchestrator = None  # Lazy init per pipeline
        self._artifact_packager = ArtifactPackager(
            str(Path(self.agent_config.state_dir) / "artifacts")
        )
        self._container_builder = ContainerBuilder(
            registry=self.agent_config.registry_url
        )
        self._env_provisioner = EnvironmentProvisioner()
        self._bluegreen_manager = BlueGreenDeploymentManager()
        self._canary_manager = CanaryDeploymentManager()
        self._rollback_automator = RollbackAutomator()
        self._flag_manager = FeatureFlagManager()

        self._pipelines: Dict[str, PipelineState] = {}
        self._load_pipelines()

    async def run_pipeline(self, config: PipelineConfig) -> PipelineState:
        """
        Run a complete deployment pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            PipelineState with final status
        """
        pipeline_id = f"pipeline-{uuid.uuid4().hex[:12]}"

        state = PipelineState(
            pipeline_id=pipeline_id,
            config=config,
        )

        self._pipelines[pipeline_id] = state

        # Emit orchestrate event
        _emit_bus_event(
            self.BUS_TOPICS["orchestrate"],
            {
                "pipeline_id": pipeline_id,
                "service_name": config.service_name,
                "version": config.version,
                "strategy": config.strategy.value,
                "environments": config.target_environments,
            },
            actor=self.actor_id,
        )

        # Emit pipeline start
        _emit_bus_event(
            self.BUS_TOPICS["pipeline_start"],
            {
                "pipeline_id": pipeline_id,
                "config": config.to_dict(),
            },
            actor=self.actor_id,
        )

        try:
            # Execute pipeline phases
            await self._phase_init(state)
            await self._phase_build(state)
            await self._phase_package(state)

            if config.container_enabled:
                await self._phase_containerize(state)

            await self._phase_provision(state)
            await self._phase_deploy(state)
            await self._phase_verify(state)

            if config.strategy == DeployStrategy.CANARY:
                await self._phase_promote(state)

            # Toggle feature flags
            for flag_name, enabled in config.feature_flags.items():
                flag = self._flag_manager.get_flag_by_name(flag_name)
                if flag:
                    self._flag_manager.toggle(flag.flag_id, enabled=enabled)

            await self._phase_complete(state)

        except Exception as e:
            state.phase = PipelinePhase.FAILED
            state.error = str(e)
            state.completed_at = time.time()

            _emit_bus_event(
                self.BUS_TOPICS["pipeline_failed"],
                {
                    "pipeline_id": pipeline_id,
                    "phase": state.phase.value,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )

            # Attempt rollback if configured
            if self.agent_config.rollback_enabled and state.deployment_state:
                await self._rollback_deployment(state)

        self._save_pipeline(state)
        return state

    async def _phase_init(self, state: PipelineState) -> None:
        """Initialize pipeline."""
        state.phase = PipelinePhase.INIT
        self._emit_phase_change(state)

        # Validate configuration
        if not state.config.service_name:
            raise ValueError("service_name is required")
        if not state.config.version:
            raise ValueError("version is required")

        # Initialize build orchestrator
        self._build_orchestrator = BuildOrchestrator(state.config.source_dir)

        await asyncio.sleep(0.01)  # Allow bus events to propagate

    async def _phase_build(self, state: PipelineState) -> None:
        """Build phase."""
        state.phase = PipelinePhase.BUILD
        self._emit_phase_change(state)

        build_config = state.config.build_config or BuildConfig()

        result = await self._build_orchestrator.build(build_config)
        state.build_result = result

        if result.status != BuildStatus.SUCCESS:
            raise RuntimeError(f"Build failed: {result.error}")

    async def _phase_package(self, state: PipelineState) -> None:
        """Package artifacts phase."""
        state.phase = PipelinePhase.PACKAGE
        self._emit_phase_change(state)

        if not state.build_result:
            raise RuntimeError("No build result available")

        build_config = state.config.build_config or BuildConfig()

        artifact = self._artifact_packager.package(
            source_dir=str(Path(state.config.source_dir) / build_config.output_dir),
            version=state.config.version,
            build_id=state.build_result.build_id,
        )
        state.artifact = artifact

    async def _phase_containerize(self, state: PipelineState) -> None:
        """Build container image phase."""
        state.phase = PipelinePhase.CONTAINERIZE
        self._emit_phase_change(state)

        result = await self._container_builder.build(
            context_dir=state.config.source_dir,
            image_name=state.config.service_name,
            tag=state.config.version,
        )

        if result.status != BuildStatus.SUCCESS:
            raise RuntimeError(f"Container build failed: {result.error}")

        state.container_image = result.image

    async def _phase_provision(self, state: PipelineState) -> None:
        """Provision environment phase."""
        state.phase = PipelinePhase.PROVISION
        self._emit_phase_change(state)

        for env_name in state.config.target_environments:
            env_type = Environment(env_name) if env_name in [e.value for e in Environment] else Environment.STAGING

            env_config = EnvironmentConfig(
                env_type=env_type,
                name=f"{state.config.service_name}-{env_name}",
            )

            env_state = await self._env_provisioner.provision(env_config)
            state.environments.append(env_state)

    async def _phase_deploy(self, state: PipelineState) -> None:
        """Deploy phase using selected strategy."""
        state.phase = PipelinePhase.DEPLOY
        self._emit_phase_change(state)

        artifact_id = state.artifact.artifact_id if state.artifact else ""

        if state.config.strategy == DeployStrategy.BLUE_GREEN:
            bg_state = self._bluegreen_manager.create_deployment(state.config.service_name)

            await self._bluegreen_manager.deploy_to_standby(
                bg_state,
                artifact_id=artifact_id,
                version=state.config.version,
            )

            state.deployment_state = bg_state

        elif state.config.strategy == DeployStrategy.CANARY:
            canary_config = state.config.canary_config or CanaryConfig()

            canary_state = await self._canary_manager.start_canary(
                service_name=state.config.service_name,
                artifact_id=artifact_id,
                version=state.config.version,
                config=canary_config,
            )

            state.deployment_state = canary_state

        elif state.config.strategy == DeployStrategy.ROLLING:
            # Simulate rolling deployment
            await asyncio.sleep(0.1)

        else:  # IMMEDIATE
            await asyncio.sleep(0.05)

    async def _phase_verify(self, state: PipelineState) -> None:
        """Verify deployment health phase."""
        state.phase = PipelinePhase.VERIFY
        self._emit_phase_change(state)

        if state.config.strategy == DeployStrategy.BLUE_GREEN:
            if isinstance(state.deployment_state, BlueGreenState):
                healthy = await self._bluegreen_manager.verify_standby(state.deployment_state)
                if not healthy:
                    raise RuntimeError("Standby verification failed")

                # Switch traffic
                await self._bluegreen_manager.switch_traffic(state.deployment_state)

        elif state.config.strategy == DeployStrategy.CANARY:
            # Canary verification happens during promote phase
            pass

        # Check environment health
        for env in state.environments:
            health = await self._env_provisioner.health_check(env.env_id)
            if health.get("status") != "healthy":
                raise RuntimeError(f"Environment {env.env_id} unhealthy")

    async def _phase_promote(self, state: PipelineState) -> None:
        """Promote canary to full traffic."""
        state.phase = PipelinePhase.PROMOTE
        self._emit_phase_change(state)

        if isinstance(state.deployment_state, CanaryState):
            canary = state.deployment_state

            # Advance through all canary steps
            while not self._canary_manager.is_complete(canary):
                success = await self._canary_manager.advance(canary)
                if not success and canary.phase.value in ("rolled_back", "failed"):
                    raise RuntimeError(f"Canary promotion failed: {canary.error}")

                # Wait between steps
                await asyncio.sleep(0.1)

    async def _phase_complete(self, state: PipelineState) -> None:
        """Complete pipeline phase."""
        state.phase = PipelinePhase.COMPLETE
        state.completed_at = time.time()

        state.metrics = {
            "total_duration_ms": (state.completed_at - state.started_at) * 1000,
            "build_duration_ms": state.build_result.duration_ms if state.build_result else 0,
            "environments_provisioned": len(state.environments),
            "strategy": state.config.strategy.value,
        }

        self._emit_phase_change(state)

        _emit_bus_event(
            self.BUS_TOPICS["pipeline_complete"],
            {
                "pipeline_id": state.pipeline_id,
                "service_name": state.config.service_name,
                "version": state.config.version,
                "duration_ms": state.metrics["total_duration_ms"],
            },
            actor=self.actor_id,
        )

    async def _rollback_deployment(self, state: PipelineState) -> None:
        """Rollback a failed deployment."""
        state.phase = PipelinePhase.ROLLED_BACK

        try:
            if isinstance(state.deployment_state, BlueGreenState):
                await self._bluegreen_manager.rollback(state.deployment_state)

            elif isinstance(state.deployment_state, CanaryState):
                await self._canary_manager.rollback(
                    state.deployment_state,
                    reason=state.error or "Pipeline failure",
                )

        except Exception as e:
            # Log rollback failure but don't raise
            state.metrics["rollback_error"] = str(e)

    def _emit_phase_change(self, state: PipelineState) -> None:
        """Emit phase change event."""
        _emit_bus_event(
            self.BUS_TOPICS["phase_change"],
            {
                "pipeline_id": state.pipeline_id,
                "phase": state.phase.value,
                "service_name": state.config.service_name,
            },
            actor=self.actor_id,
        )

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineState]:
        """Get a pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    def list_pipelines(
        self,
        service_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[PipelineState]:
        """List pipelines."""
        pipelines = list(self._pipelines.values())

        if service_name:
            pipelines = [p for p in pipelines if p.config.service_name == service_name]

        pipelines.sort(key=lambda p: p.started_at, reverse=True)
        return pipelines[:limit]

    def _save_pipeline(self, state: PipelineState) -> None:
        """Save pipeline state to disk."""
        state_file = self.state_dir / f"{state.pipeline_id}.json"
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_pipelines(self) -> None:
        """Load pipelines from disk."""
        # For simplicity, we only track in-memory in this implementation
        # A production version would persist and load state
        pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deploy orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Agent Orchestrator (Step 210)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Run deployment pipeline")
    deploy_parser.add_argument("service_name", help="Service name")
    deploy_parser.add_argument("--version", "-v", required=True, help="Version")
    deploy_parser.add_argument("--source", "-s", default=".", help="Source directory")
    deploy_parser.add_argument("--strategy", default="blue_green", choices=["rolling", "blue_green", "canary", "immediate"])
    deploy_parser.add_argument("--env", "-e", default="staging", help="Target environment(s)")
    deploy_parser.add_argument("--no-container", action="store_true", help="Skip container build")
    deploy_parser.add_argument("--json", action="store_true", help="JSON output")

    # status command
    status_parser = subparsers.add_parser("status", help="Get pipeline status")
    status_parser.add_argument("pipeline_id", help="Pipeline ID")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # list command
    list_parser = subparsers.add_parser("list", help="List pipelines")
    list_parser.add_argument("--service", "-s", help="Filter by service")
    list_parser.add_argument("--limit", "-l", type=int, default=10, help="Limit results")
    list_parser.add_argument("--json", action="store_true", help="JSON output")

    args = parser.parse_args()
    orchestrator = DeployOrchestrator()

    if args.command == "deploy":
        config = PipelineConfig(
            name=f"{args.service_name}-deploy",
            service_name=args.service_name,
            version=args.version,
            source_dir=args.source,
            strategy=DeployStrategy(args.strategy),
            target_environments=args.env.split(","),
            container_enabled=not args.no_container,
        )

        state = asyncio.get_event_loop().run_until_complete(
            orchestrator.run_pipeline(config)
        )

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            status_icon = "OK" if state.phase == PipelinePhase.COMPLETE else "FAIL"
            print(f"[{status_icon}] Pipeline {state.pipeline_id}")
            print(f"  Service: {state.config.service_name}")
            print(f"  Version: {state.config.version}")
            print(f"  Phase: {state.phase.value}")
            print(f"  Strategy: {state.config.strategy.value}")
            if state.metrics:
                print(f"  Duration: {state.metrics.get('total_duration_ms', 0):.1f}ms")
            if state.error:
                print(f"  Error: {state.error}")

        return 0 if state.phase == PipelinePhase.COMPLETE else 1

    elif args.command == "status":
        state = orchestrator.get_pipeline(args.pipeline_id)
        if not state:
            print(f"Pipeline not found: {args.pipeline_id}")
            return 1

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            print(f"Pipeline: {state.pipeline_id}")
            print(f"  Service: {state.config.service_name}")
            print(f"  Version: {state.config.version}")
            print(f"  Phase: {state.phase.value}")

        return 0

    elif args.command == "list":
        pipelines = orchestrator.list_pipelines(service_name=args.service, limit=args.limit)

        if args.json:
            print(json.dumps([p.to_dict() for p in pipelines], indent=2))
        else:
            if not pipelines:
                print("No pipelines found")
            else:
                for p in pipelines:
                    print(f"{p.pipeline_id} ({p.config.service_name}:{p.config.version}) - {p.phase.value}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
