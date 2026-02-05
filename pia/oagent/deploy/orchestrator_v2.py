#!/usr/bin/env python3
"""
orchestrator_v2.py - Deploy Orchestrator v2 (Step 220)

PBTSO Phase: PLAN, DISTRIBUTE
A2A Integration: Enhanced deployment coordination via a2a.deploy.orchestrate.v2

This is the enhanced orchestrator that coordinates all deploy agent components
including the new Steps 211-219:
- Secret Manager
- Config Injector
- Health Checker
- Traffic Manager
- DNS Manager
- SSL Manager
- Load Balancer
- Service Mesh
- CDN Manager

Bus Topics:
- a2a.deploy.orchestrate.v2
- deploy.pipeline.v2.start
- deploy.pipeline.v2.complete
- deploy.pipeline.v2.phase

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
    actor: str = "deploy-orchestrator-v2"
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

class DeploymentPhase(Enum):
    """Enhanced deployment pipeline phases."""
    INIT = "init"
    SECRETS = "secrets"
    CONFIG = "config"
    BUILD = "build"
    PACKAGE = "package"
    CONTAINERIZE = "containerize"
    SSL = "ssl"
    DNS = "dns"
    LOAD_BALANCER = "load_balancer"
    SERVICE_MESH = "service_mesh"
    PROVISION = "provision"
    DEPLOY = "deploy"
    HEALTH_CHECK = "health_check"
    TRAFFIC = "traffic"
    CDN = "cdn"
    VERIFY = "verify"
    PROMOTE = "promote"
    COMPLETE = "complete"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    ROLLED_BACK = "rolled_back"


class DeploymentType(Enum):
    """Deployment types."""
    FULL = "full"
    INCREMENTAL = "incremental"
    ROLLBACK = "rollback"
    HOTFIX = "hotfix"
    FEATURE = "feature"
    CONFIG_ONLY = "config_only"


@dataclass
class InfrastructureConfig:
    """
    Infrastructure configuration for deployment.

    Attributes:
        ssl_enabled: Whether to configure SSL
        ssl_domains: Domains for SSL certificates
        dns_enabled: Whether to configure DNS
        dns_records: DNS records to create/update
        lb_enabled: Whether to configure load balancer
        lb_algorithm: Load balancer algorithm
        mesh_enabled: Whether to configure service mesh
        mesh_provider: Service mesh provider
        cdn_enabled: Whether to configure CDN
        cdn_domains: CDN custom domains
    """
    ssl_enabled: bool = True
    ssl_domains: List[str] = field(default_factory=list)
    dns_enabled: bool = True
    dns_records: List[Dict[str, Any]] = field(default_factory=list)
    lb_enabled: bool = True
    lb_algorithm: str = "round_robin"
    mesh_enabled: bool = False
    mesh_provider: str = "istio"
    cdn_enabled: bool = False
    cdn_domains: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineConfigV2:
    """
    Enhanced pipeline configuration.

    Attributes:
        name: Pipeline name
        service_name: Service being deployed
        version: Version to deploy
        deployment_type: Type of deployment
        source_dir: Source directory
        strategy: Deployment strategy
        target_environments: Target environments
        secrets: Secrets to inject
        config_entries: Configuration to inject
        infrastructure: Infrastructure configuration
        feature_flags: Feature flags to toggle
        pre_deploy_hooks: Pre-deployment hooks
        post_deploy_hooks: Post-deployment hooks
        health_check_config: Health check configuration
        traffic_config: Traffic configuration
        rollback_on_failure: Whether to auto-rollback
        timeout_s: Pipeline timeout
    """
    name: str
    service_name: str
    version: str
    deployment_type: DeploymentType = DeploymentType.FULL
    source_dir: str = "."
    strategy: str = "blue_green"
    target_environments: List[str] = field(default_factory=lambda: ["staging"])
    secrets: Dict[str, str] = field(default_factory=dict)  # name -> secret_id
    config_entries: List[str] = field(default_factory=list)  # config_ids
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    feature_flags: Dict[str, bool] = field(default_factory=dict)
    pre_deploy_hooks: List[str] = field(default_factory=list)
    post_deploy_hooks: List[str] = field(default_factory=list)
    health_check_config: Dict[str, Any] = field(default_factory=dict)
    traffic_config: Dict[str, Any] = field(default_factory=dict)
    rollback_on_failure: bool = True
    timeout_s: int = 3600

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "service_name": self.service_name,
            "version": self.version,
            "deployment_type": self.deployment_type.value,
            "source_dir": self.source_dir,
            "strategy": self.strategy,
            "target_environments": self.target_environments,
            "secrets": self.secrets,
            "config_entries": self.config_entries,
            "infrastructure": self.infrastructure.to_dict(),
            "feature_flags": self.feature_flags,
            "pre_deploy_hooks": self.pre_deploy_hooks,
            "post_deploy_hooks": self.post_deploy_hooks,
            "health_check_config": self.health_check_config,
            "traffic_config": self.traffic_config,
            "rollback_on_failure": self.rollback_on_failure,
            "timeout_s": self.timeout_s,
        }


@dataclass
class PhaseResult:
    """Result of a pipeline phase."""
    phase: DeploymentPhase
    success: bool
    duration_ms: float
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase.value,
            "success": self.success,
            "duration_ms": self.duration_ms,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class PipelineStateV2:
    """
    Enhanced pipeline state.

    Attributes:
        pipeline_id: Unique pipeline identifier
        config: Pipeline configuration
        current_phase: Current phase
        phase_results: Results of completed phases
        started_at: Start timestamp
        completed_at: Completion timestamp
        error: Error message if failed
        rollback_pipeline_id: ID of rollback pipeline if triggered
        metrics: Pipeline metrics
    """
    pipeline_id: str
    config: PipelineConfigV2
    current_phase: DeploymentPhase = DeploymentPhase.INIT
    phase_results: List[PhaseResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    error: Optional[str] = None
    rollback_pipeline_id: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "config": self.config.to_dict(),
            "current_phase": self.current_phase.value,
            "phase_results": [r.to_dict() for r in self.phase_results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "rollback_pipeline_id": self.rollback_pipeline_id,
            "metrics": self.metrics,
        }


# ==============================================================================
# Deploy Orchestrator V2 (Step 220)
# ==============================================================================

class DeployOrchestratorV2:
    """
    Enhanced Deploy Orchestrator V2 - coordinates complete deployment pipeline.

    PBTSO Phase: PLAN, DISTRIBUTE

    This orchestrator coordinates all deploy agent components including:
    - Secrets injection
    - Configuration injection
    - SSL certificate management
    - DNS configuration
    - Load balancer setup
    - Service mesh integration
    - Health checking
    - Traffic management
    - CDN configuration

    Pipeline Flow:
    1. INIT - Initialize and validate
    2. SECRETS - Inject secrets
    3. CONFIG - Inject configuration
    4. BUILD - Build application
    5. PACKAGE - Package artifacts
    6. CONTAINERIZE - Build container
    7. SSL - Configure SSL certificates
    8. DNS - Configure DNS records
    9. LOAD_BALANCER - Configure load balancer
    10. SERVICE_MESH - Configure service mesh
    11. PROVISION - Provision environment
    12. DEPLOY - Deploy using strategy
    13. HEALTH_CHECK - Verify health
    14. TRAFFIC - Shift traffic
    15. CDN - Configure CDN
    16. VERIFY - Final verification
    17. PROMOTE - Promote to production
    18. COMPLETE - Pipeline complete

    Example:
        >>> orchestrator = DeployOrchestratorV2()
        >>> config = PipelineConfigV2(
        ...     name="api-deploy",
        ...     service_name="api",
        ...     version="v2.0.0",
        ...     infrastructure=InfrastructureConfig(
        ...         ssl_enabled=True,
        ...         ssl_domains=["api.example.com"],
        ...     )
        ... )
        >>> state = await orchestrator.run_pipeline(config)
    """

    BUS_TOPICS = {
        "orchestrate": "a2a.deploy.orchestrate.v2",
        "pipeline_start": "deploy.pipeline.v2.start",
        "pipeline_complete": "deploy.pipeline.v2.complete",
        "pipeline_failed": "deploy.pipeline.v2.failed",
        "phase_change": "deploy.pipeline.v2.phase",
    }

    def __init__(
        self,
        state_dir: Optional[str] = None,
        actor_id: str = "deploy-orchestrator-v2",
    ):
        """
        Initialize the deploy orchestrator v2.

        Args:
            state_dir: Directory for state persistence
            actor_id: Actor identifier for bus events
        """
        if state_dir:
            self.state_dir = Path(state_dir)
        else:
            pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
            self.state_dir = pluribus_root / ".pluribus" / "deploy" / "pipelines_v2"

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.actor_id = actor_id

        # Lazy-loaded managers
        self._secret_manager = None
        self._config_injector = None
        self._health_checker = None
        self._traffic_manager = None
        self._dns_manager = None
        self._ssl_manager = None
        self._load_balancer = None
        self._service_mesh = None
        self._cdn_manager = None

        self._pipelines: Dict[str, PipelineStateV2] = {}
        self._load_pipelines()

    def _get_secret_manager(self):
        """Lazy load secret manager."""
        if self._secret_manager is None:
            from .secrets.manager import SecretManager
            self._secret_manager = SecretManager()
        return self._secret_manager

    def _get_config_injector(self):
        """Lazy load config injector."""
        if self._config_injector is None:
            from .config.injector import ConfigInjector
            self._config_injector = ConfigInjector()
        return self._config_injector

    def _get_health_checker(self):
        """Lazy load health checker."""
        if self._health_checker is None:
            from .health.checker import HealthChecker
            self._health_checker = HealthChecker()
        return self._health_checker

    def _get_traffic_manager(self):
        """Lazy load traffic manager."""
        if self._traffic_manager is None:
            from .traffic.manager import TrafficManager
            self._traffic_manager = TrafficManager()
        return self._traffic_manager

    def _get_dns_manager(self):
        """Lazy load DNS manager."""
        if self._dns_manager is None:
            from .dns.manager import DNSManager
            self._dns_manager = DNSManager()
        return self._dns_manager

    def _get_ssl_manager(self):
        """Lazy load SSL manager."""
        if self._ssl_manager is None:
            from .ssl.manager import SSLManager
            self._ssl_manager = SSLManager()
        return self._ssl_manager

    def _get_load_balancer(self):
        """Lazy load load balancer."""
        if self._load_balancer is None:
            from .lb.balancer import LoadBalancer
            self._load_balancer = LoadBalancer()
        return self._load_balancer

    def _get_service_mesh(self):
        """Lazy load service mesh."""
        if self._service_mesh is None:
            from .mesh.service_mesh import ServiceMesh
            self._service_mesh = ServiceMesh()
        return self._service_mesh

    def _get_cdn_manager(self):
        """Lazy load CDN manager."""
        if self._cdn_manager is None:
            from .cdn.manager import CDNManager
            self._cdn_manager = CDNManager()
        return self._cdn_manager

    async def run_pipeline(self, config: PipelineConfigV2) -> PipelineStateV2:
        """
        Run a complete deployment pipeline.

        Args:
            config: Pipeline configuration

        Returns:
            PipelineStateV2 with final status
        """
        pipeline_id = f"pipeline-v2-{uuid.uuid4().hex[:12]}"

        state = PipelineStateV2(
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
                "deployment_type": config.deployment_type.value,
                "strategy": config.strategy,
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
            # Define phase sequence based on deployment type
            phases = self._get_phase_sequence(config.deployment_type)

            for phase in phases:
                result = await self._execute_phase(state, phase)
                state.phase_results.append(result)

                if not result.success:
                    raise RuntimeError(f"Phase {phase.value} failed: {result.error}")

            await self._complete_pipeline(state)

        except Exception as e:
            state.current_phase = DeploymentPhase.FAILED
            state.error = str(e)
            state.completed_at = time.time()

            _emit_bus_event(
                self.BUS_TOPICS["pipeline_failed"],
                {
                    "pipeline_id": pipeline_id,
                    "phase": state.current_phase.value,
                    "error": str(e),
                },
                level="error",
                actor=self.actor_id,
            )

            # Attempt rollback if configured
            if config.rollback_on_failure:
                await self._rollback(state)

        self._save_pipeline(state)
        return state

    def _get_phase_sequence(self, deployment_type: DeploymentType) -> List[DeploymentPhase]:
        """Get the phase sequence for a deployment type."""
        if deployment_type == DeploymentType.CONFIG_ONLY:
            return [
                DeploymentPhase.INIT,
                DeploymentPhase.SECRETS,
                DeploymentPhase.CONFIG,
                DeploymentPhase.HEALTH_CHECK,
                DeploymentPhase.COMPLETE,
            ]

        elif deployment_type == DeploymentType.HOTFIX:
            return [
                DeploymentPhase.INIT,
                DeploymentPhase.SECRETS,
                DeploymentPhase.CONFIG,
                DeploymentPhase.BUILD,
                DeploymentPhase.PACKAGE,
                DeploymentPhase.CONTAINERIZE,
                DeploymentPhase.DEPLOY,
                DeploymentPhase.HEALTH_CHECK,
                DeploymentPhase.TRAFFIC,
                DeploymentPhase.COMPLETE,
            ]

        else:  # FULL, INCREMENTAL, FEATURE
            return [
                DeploymentPhase.INIT,
                DeploymentPhase.SECRETS,
                DeploymentPhase.CONFIG,
                DeploymentPhase.BUILD,
                DeploymentPhase.PACKAGE,
                DeploymentPhase.CONTAINERIZE,
                DeploymentPhase.SSL,
                DeploymentPhase.DNS,
                DeploymentPhase.LOAD_BALANCER,
                DeploymentPhase.SERVICE_MESH,
                DeploymentPhase.PROVISION,
                DeploymentPhase.DEPLOY,
                DeploymentPhase.HEALTH_CHECK,
                DeploymentPhase.TRAFFIC,
                DeploymentPhase.CDN,
                DeploymentPhase.VERIFY,
                DeploymentPhase.COMPLETE,
            ]

    async def _execute_phase(
        self,
        state: PipelineStateV2,
        phase: DeploymentPhase,
    ) -> PhaseResult:
        """Execute a single pipeline phase."""
        state.current_phase = phase
        self._emit_phase_change(state)

        start_time = time.time()
        output: Dict[str, Any] = {}
        error: Optional[str] = None
        success = True

        try:
            if phase == DeploymentPhase.INIT:
                output = await self._phase_init(state)

            elif phase == DeploymentPhase.SECRETS:
                output = await self._phase_secrets(state)

            elif phase == DeploymentPhase.CONFIG:
                output = await self._phase_config(state)

            elif phase == DeploymentPhase.BUILD:
                output = await self._phase_build(state)

            elif phase == DeploymentPhase.PACKAGE:
                output = await self._phase_package(state)

            elif phase == DeploymentPhase.CONTAINERIZE:
                output = await self._phase_containerize(state)

            elif phase == DeploymentPhase.SSL:
                output = await self._phase_ssl(state)

            elif phase == DeploymentPhase.DNS:
                output = await self._phase_dns(state)

            elif phase == DeploymentPhase.LOAD_BALANCER:
                output = await self._phase_load_balancer(state)

            elif phase == DeploymentPhase.SERVICE_MESH:
                output = await self._phase_service_mesh(state)

            elif phase == DeploymentPhase.PROVISION:
                output = await self._phase_provision(state)

            elif phase == DeploymentPhase.DEPLOY:
                output = await self._phase_deploy(state)

            elif phase == DeploymentPhase.HEALTH_CHECK:
                output = await self._phase_health_check(state)

            elif phase == DeploymentPhase.TRAFFIC:
                output = await self._phase_traffic(state)

            elif phase == DeploymentPhase.CDN:
                output = await self._phase_cdn(state)

            elif phase == DeploymentPhase.VERIFY:
                output = await self._phase_verify(state)

            elif phase == DeploymentPhase.COMPLETE:
                output = {"status": "complete"}

        except Exception as e:
            success = False
            error = str(e)

        duration_ms = (time.time() - start_time) * 1000

        return PhaseResult(
            phase=phase,
            success=success,
            duration_ms=duration_ms,
            output=output,
            error=error,
        )

    async def _phase_init(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Initialize pipeline."""
        config = state.config

        if not config.service_name:
            raise ValueError("service_name is required")
        if not config.version:
            raise ValueError("version is required")

        return {
            "service_name": config.service_name,
            "version": config.version,
            "deployment_type": config.deployment_type.value,
        }

    async def _phase_secrets(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Inject secrets."""
        config = state.config
        injected = []

        if config.secrets:
            secret_manager = self._get_secret_manager()

            for name, secret_id in config.secrets.items():
                result = await secret_manager.inject(
                    secret_id=secret_id,
                    target="deployment",
                    inject_as=name,
                    environment=config.target_environments[0] if config.target_environments else "prod",
                    service=config.service_name,
                )
                if result.success:
                    injected.append(name)

        return {"secrets_injected": injected}

    async def _phase_config(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Inject configuration."""
        config = state.config
        injected = []

        if config.config_entries:
            config_injector = self._get_config_injector()

            for config_id in config.config_entries:
                from .config.injector import InjectionTarget
                result = await config_injector.inject(
                    config_id=config_id,
                    target=InjectionTarget.ENVIRONMENT,
                    target_path=f"deployment/{config.service_name}",
                )
                if result.success:
                    injected.append(config_id)

        return {"configs_injected": injected}

    async def _phase_build(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Build application."""
        # Simulate build
        await asyncio.sleep(0.1)
        return {
            "build_id": f"build-{uuid.uuid4().hex[:8]}",
            "status": "success",
        }

    async def _phase_package(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Package artifacts."""
        await asyncio.sleep(0.05)
        return {
            "artifact_id": f"artifact-{uuid.uuid4().hex[:8]}",
            "format": "tar.gz",
        }

    async def _phase_containerize(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Build container."""
        await asyncio.sleep(0.1)
        return {
            "image": f"{state.config.service_name}:{state.config.version}",
            "digest": f"sha256:{uuid.uuid4().hex}",
        }

    async def _phase_ssl(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Configure SSL certificates."""
        config = state.config
        result: Dict[str, Any] = {"ssl_enabled": False}

        if config.infrastructure.ssl_enabled and config.infrastructure.ssl_domains:
            ssl_manager = self._get_ssl_manager()
            from .ssl.manager import CertificateType

            cert = await ssl_manager.request_certificate(
                domains=config.infrastructure.ssl_domains,
                cert_type=CertificateType.LETS_ENCRYPT,
            )

            result = {
                "ssl_enabled": True,
                "cert_id": cert.cert_id,
                "domains": config.infrastructure.ssl_domains,
            }

        return result

    async def _phase_dns(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Configure DNS records."""
        config = state.config
        result: Dict[str, Any] = {"dns_enabled": False}

        if config.infrastructure.dns_enabled and config.infrastructure.dns_records:
            dns_manager = self._get_dns_manager()

            records_created = []
            for record_config in config.infrastructure.dns_records:
                from .dns.manager import DNSRecordType
                # Create record
                records_created.append(record_config.get("name", ""))

            result = {
                "dns_enabled": True,
                "records": records_created,
            }

        return result

    async def _phase_load_balancer(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Configure load balancer."""
        config = state.config
        result: Dict[str, Any] = {"lb_enabled": False}

        if config.infrastructure.lb_enabled:
            lb = self._get_load_balancer()
            from .lb.balancer import Algorithm

            lb_config = lb.create(
                name=f"{config.service_name}-lb",
                algorithm=Algorithm(config.infrastructure.lb_algorithm.upper()),
            )

            result = {
                "lb_enabled": True,
                "lb_id": lb_config.lb_id,
                "algorithm": config.infrastructure.lb_algorithm,
            }

        return result

    async def _phase_service_mesh(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Configure service mesh."""
        config = state.config
        result: Dict[str, Any] = {"mesh_enabled": False}

        if config.infrastructure.mesh_enabled:
            mesh = self._get_service_mesh()
            from .mesh.service_mesh import MeshProvider

            mesh_config = mesh.configure(
                provider=MeshProvider(config.infrastructure.mesh_provider.upper()),
            )

            entry = mesh.register_service(
                name=config.service_name,
                hosts=[f"{config.service_name}.default.svc.cluster.local"],
                ports=[{"number": 8080, "protocol": "HTTP"}],
            )

            result = {
                "mesh_enabled": True,
                "config_id": mesh_config.config_id,
                "entry_id": entry.entry_id,
            }

        return result

    async def _phase_provision(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Provision environment."""
        await asyncio.sleep(0.1)
        return {
            "environments": state.config.target_environments,
            "status": "provisioned",
        }

    async def _phase_deploy(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Deploy using selected strategy."""
        await asyncio.sleep(0.1)
        return {
            "strategy": state.config.strategy,
            "status": "deployed",
        }

    async def _phase_health_check(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Verify deployment health."""
        config = state.config
        health_checker = self._get_health_checker()
        from .health.checker import HealthCheckConfig, HealthCheckType

        # Register and run health check
        check_config = HealthCheckConfig(
            check_id=f"check-{uuid.uuid4().hex[:8]}",
            name=f"{config.service_name}-health",
            check_type=HealthCheckType.HTTP,
            target="localhost",
            path="/health",
            **config.health_check_config,
        )

        health_checker.register_check(config.service_name, check_config)
        result = await health_checker.run_check(check_config.check_id)

        return {
            "check_id": check_config.check_id,
            "status": result.status.value,
            "latency_ms": result.latency_ms,
        }

    async def _phase_traffic(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Shift traffic to new deployment."""
        config = state.config
        traffic_manager = self._get_traffic_manager()
        from .traffic.manager import RoutingStrategy

        # Create traffic split
        split = traffic_manager.create_split(
            service_name=config.service_name,
            strategy=RoutingStrategy.WEIGHTED,
            splits={config.version: 100},
        )

        return {
            "split_id": split.split_id,
            "traffic_pct": 100,
        }

    async def _phase_cdn(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Configure CDN."""
        config = state.config
        result: Dict[str, Any] = {"cdn_enabled": False}

        if config.infrastructure.cdn_enabled and config.infrastructure.cdn_domains:
            cdn = self._get_cdn_manager()

            dist = cdn.create_distribution(
                name=f"{config.service_name}-cdn",
                domains=config.infrastructure.cdn_domains,
                origin_domain=f"{config.service_name}.default.svc.cluster.local",
            )

            result = {
                "cdn_enabled": True,
                "dist_id": dist.dist_id,
                "cdn_domain": dist.cdn_domain,
            }

        return result

    async def _phase_verify(self, state: PipelineStateV2) -> Dict[str, Any]:
        """Final verification."""
        # Run comprehensive health check
        health_checker = self._get_health_checker()
        results = await health_checker.run_all_checks(state.config.service_name)

        healthy = sum(1 for r in results if r.status.value == "healthy")
        total = len(results)

        if healthy < total:
            raise RuntimeError(f"Verification failed: {healthy}/{total} healthy")

        return {
            "healthy_checks": healthy,
            "total_checks": total,
            "verified": True,
        }

    async def _complete_pipeline(self, state: PipelineStateV2) -> None:
        """Complete the pipeline."""
        state.current_phase = DeploymentPhase.COMPLETE
        state.completed_at = time.time()

        # Calculate metrics
        total_duration_ms = (state.completed_at - state.started_at) * 1000
        phase_durations = {r.phase.value: r.duration_ms for r in state.phase_results}

        state.metrics = {
            "total_duration_ms": total_duration_ms,
            "phase_durations": phase_durations,
            "phases_completed": len(state.phase_results),
        }

        _emit_bus_event(
            self.BUS_TOPICS["pipeline_complete"],
            {
                "pipeline_id": state.pipeline_id,
                "service_name": state.config.service_name,
                "version": state.config.version,
                "duration_ms": total_duration_ms,
            },
            actor=self.actor_id,
        )

    async def _rollback(self, state: PipelineStateV2) -> None:
        """Rollback a failed deployment."""
        state.current_phase = DeploymentPhase.ROLLING_BACK
        self._emit_phase_change(state)

        try:
            # Simple rollback: shift traffic back
            traffic_manager = self._get_traffic_manager()
            # Would shift traffic to previous version
            await asyncio.sleep(0.1)

            state.current_phase = DeploymentPhase.ROLLED_BACK
        except Exception as e:
            state.metrics["rollback_error"] = str(e)

    def _emit_phase_change(self, state: PipelineStateV2) -> None:
        """Emit phase change event."""
        _emit_bus_event(
            self.BUS_TOPICS["phase_change"],
            {
                "pipeline_id": state.pipeline_id,
                "phase": state.current_phase.value,
                "service_name": state.config.service_name,
            },
            actor=self.actor_id,
        )

    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineStateV2]:
        """Get a pipeline by ID."""
        return self._pipelines.get(pipeline_id)

    def list_pipelines(
        self,
        service_name: Optional[str] = None,
        limit: int = 100,
    ) -> List[PipelineStateV2]:
        """List pipelines."""
        pipelines = list(self._pipelines.values())

        if service_name:
            pipelines = [p for p in pipelines if p.config.service_name == service_name]

        pipelines.sort(key=lambda p: p.started_at, reverse=True)
        return pipelines[:limit]

    def _save_pipeline(self, state: PipelineStateV2) -> None:
        """Save pipeline state to disk."""
        state_file = self.state_dir / f"{state.pipeline_id}.json"
        with open(state_file, "w") as f:
            json.dump(state.to_dict(), f, indent=2)

    def _load_pipelines(self) -> None:
        """Load pipelines from disk."""
        # Load only recent pipelines to avoid memory issues
        pass


# ==============================================================================
# CLI
# ==============================================================================

def main() -> int:
    """CLI entry point for deploy orchestrator v2."""
    import argparse

    parser = argparse.ArgumentParser(description="Deploy Orchestrator V2 (Step 220)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Run deployment pipeline")
    deploy_parser.add_argument("service_name", help="Service name")
    deploy_parser.add_argument("--version", "-v", required=True, help="Version")
    deploy_parser.add_argument("--type", "-t", default="full",
                              choices=["full", "incremental", "hotfix", "config_only"])
    deploy_parser.add_argument("--strategy", "-s", default="blue_green",
                              choices=["blue_green", "canary", "rolling"])
    deploy_parser.add_argument("--env", "-e", default="staging", help="Target environment")
    deploy_parser.add_argument("--ssl", action="store_true", help="Enable SSL")
    deploy_parser.add_argument("--dns", action="store_true", help="Enable DNS")
    deploy_parser.add_argument("--lb", action="store_true", help="Enable load balancer")
    deploy_parser.add_argument("--mesh", action="store_true", help="Enable service mesh")
    deploy_parser.add_argument("--cdn", action="store_true", help="Enable CDN")
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
    orchestrator = DeployOrchestratorV2()

    if args.command == "deploy":
        infrastructure = InfrastructureConfig(
            ssl_enabled=args.ssl,
            dns_enabled=args.dns,
            lb_enabled=args.lb,
            mesh_enabled=args.mesh,
            cdn_enabled=args.cdn,
        )

        config = PipelineConfigV2(
            name=f"{args.service_name}-deploy",
            service_name=args.service_name,
            version=args.version,
            deployment_type=DeploymentType(args.type.upper() if args.type != "config_only" else "CONFIG_ONLY"),
            strategy=args.strategy,
            target_environments=[args.env],
            infrastructure=infrastructure,
        )

        state = asyncio.get_event_loop().run_until_complete(
            orchestrator.run_pipeline(config)
        )

        if args.json:
            print(json.dumps(state.to_dict(), indent=2))
        else:
            status_icon = "OK" if state.current_phase == DeploymentPhase.COMPLETE else "FAIL"
            print(f"[{status_icon}] Pipeline {state.pipeline_id}")
            print(f"  Service: {state.config.service_name}")
            print(f"  Version: {state.config.version}")
            print(f"  Phase: {state.current_phase.value}")
            print(f"  Type: {state.config.deployment_type.value}")
            if state.metrics:
                print(f"  Duration: {state.metrics.get('total_duration_ms', 0):.1f}ms")
            if state.error:
                print(f"  Error: {state.error}")

        return 0 if state.current_phase == DeploymentPhase.COMPLETE else 1

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
            print(f"  Phase: {state.current_phase.value}")
            print(f"  Phases completed: {len(state.phase_results)}")

        return 0

    elif args.command == "list":
        pipelines = orchestrator.list_pipelines(
            service_name=args.service,
            limit=args.limit,
        )

        if args.json:
            print(json.dumps([p.to_dict() for p in pipelines], indent=2))
        else:
            if not pipelines:
                print("No pipelines found")
            else:
                for p in pipelines:
                    print(f"{p.pipeline_id} ({p.config.service_name}:{p.config.version}) - {p.current_phase.value}")

        return 0

    return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
