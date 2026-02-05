#!/usr/bin/env python3
"""
final_orchestrator.py - Final Research Agent Orchestrator (Step 50)

Complete orchestration of all Research Agent components.
Provides unified interface for all 50 steps of the Research Agent.

PBTSO Phases: ALL (SKILL, SEQUESTER, RESEARCH, DISTILL, PLAN, ITERATE,
              INTERFACE, EXTEND, MONITOR, PROTECT, OPTIMIZE, COORDINATE)

Bus Topics:
- a2a.research.orchestrator.init
- a2a.research.orchestrator.ready
- a2a.research.orchestrator.shutdown
- a2a.research.heartbeat

Protocol: DKIN v30, PAIP v16, HOLON v2, CITIZEN v2

A2A Heartbeat: 300s interval, 900s timeout
FalkorDB: port 6380
"""
from __future__ import annotations

import atexit
import fcntl
import json
import os
import signal
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ..bootstrap import AgentBus, ResearchAgentBootstrap


# ============================================================================
# Configuration
# ============================================================================


class AgentState(Enum):
    """Research Agent lifecycle states."""
    INITIALIZING = "initializing"
    BOOTSTRAPPING = "bootstrapping"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    ERROR = "error"


class ComponentStatus(Enum):
    """Component health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# A2A Protocol constants
A2A_HEARTBEAT_INTERVAL = 300  # seconds
A2A_HEARTBEAT_TIMEOUT = 900   # seconds
FALKORDB_PORT = 6380


@dataclass
class OrchestratorConfig:
    """Configuration for the Research Agent Orchestrator."""

    agent_name: str = "research-agent"
    agent_version: str = "0.2.0"
    pluribus_root: str = ""

    # Component toggles
    enable_security: bool = True
    enable_validation: bool = True
    enable_testing: bool = True
    enable_documentation: bool = True
    enable_migration: bool = True
    enable_backup: bool = True
    enable_telemetry: bool = True
    enable_versioning: bool = True
    enable_deprecation: bool = True

    # A2A Protocol
    a2a_heartbeat_interval: int = A2A_HEARTBEAT_INTERVAL
    a2a_heartbeat_timeout: int = A2A_HEARTBEAT_TIMEOUT

    # FalkorDB
    falkordb_host: str = "localhost"
    falkordb_port: int = FALKORDB_PORT

    # Bus
    bus_path: Optional[str] = None

    def __post_init__(self):
        if not self.pluribus_root:
            self.pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
        if self.bus_path is None:
            self.bus_path = f"{self.pluribus_root}/.pluribus/bus/events.ndjson"


@dataclass
class ComponentInfo:
    """Information about a registered component."""

    name: str
    step: int
    status: ComponentStatus = ComponentStatus.UNKNOWN
    instance: Any = None
    initialized: bool = False
    last_check: float = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "step": self.step,
            "status": self.status.value,
            "initialized": self.initialized,
            "last_check": self.last_check,
            "error": self.error,
        }


# ============================================================================
# Final Orchestrator
# ============================================================================


class ResearchAgentOrchestrator:
    """
    Final orchestrator for the Research Agent.

    This is Step 50 of the OAGENT 300-step plan, completing the Research Agent
    (Steps 1-50). It provides unified orchestration of all Research Agent
    components including:

    Steps 1-10: Core scanning and parsing
    Steps 11-20: Search and analysis
    Steps 21-30: Execution and API
    Steps 31-40: Infrastructure
    Steps 41-50: Final components (security, validation, testing, etc.)

    PBTSO Phases: All phases are coordinated through this orchestrator.

    Protocol Compliance:
    - DKIN v30: Knowledge graph integration
    - PAIP v16: Agent interaction protocol
    - HOLON v2: Holonic structure
    - CITIZEN v2: Citizenship protocol

    Example:
        orchestrator = ResearchAgentOrchestrator()

        # Initialize all components
        orchestrator.initialize()

        # Start the agent
        orchestrator.start()

        # Execute a research query
        result = orchestrator.research("Find all Python classes implementing Observer pattern")

        # Graceful shutdown
        orchestrator.shutdown()
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        bus: Optional[AgentBus] = None,
    ):
        """
        Initialize the Research Agent Orchestrator.

        Args:
            config: Orchestrator configuration
            bus: AgentBus for event emission
        """
        self.config = config or OrchestratorConfig()
        self.bus = bus or AgentBus()

        # State
        self._state = AgentState.INITIALIZING
        self._start_time: Optional[float] = None
        self._last_heartbeat: float = 0

        # Components registry
        self._components: Dict[str, ComponentInfo] = {}

        # Heartbeat thread
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

        # Lock for thread safety
        self._lock = threading.RLock()

        # Statistics
        self._stats = {
            "queries_processed": 0,
            "errors_handled": 0,
            "uptime_seconds": 0,
            "heartbeats_sent": 0,
        }

        # Register signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Register shutdown handler
        atexit.register(self._atexit_handler)

    @property
    def state(self) -> AgentState:
        """Get current agent state."""
        return self._state

    @property
    def is_ready(self) -> bool:
        """Check if agent is ready to process requests."""
        return self._state in [AgentState.READY, AgentState.RUNNING]

    def initialize(self) -> bool:
        """
        Initialize all Research Agent components.

        Returns:
            True if initialization successful
        """
        self._state = AgentState.BOOTSTRAPPING

        self._emit_event("a2a.research.orchestrator.init", {
            "agent": self.config.agent_name,
            "version": self.config.agent_version,
        })

        try:
            # Initialize Steps 1-40 via bootstrap
            self._initialize_core_components()

            # Initialize Steps 41-50
            self._initialize_final_components()

            # Verify all components
            self._verify_components()

            self._state = AgentState.READY
            self._start_time = time.time()

            self._emit_event("a2a.research.orchestrator.ready", {
                "components": len(self._components),
                "state": self._state.value,
            })

            return True

        except Exception as e:
            self._state = AgentState.ERROR
            self._emit_event("a2a.research.orchestrator.init", {
                "error": str(e),
            }, level="error")
            return False

    def start(self) -> None:
        """Start the Research Agent."""
        if not self.is_ready:
            raise RuntimeError("Agent not initialized")

        self._state = AgentState.RUNNING
        self._running = True

        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop the Research Agent."""
        self._running = False
        self._state = AgentState.PAUSED

    def shutdown(self) -> None:
        """Gracefully shutdown the Research Agent."""
        self._state = AgentState.SHUTTING_DOWN
        self._running = False

        self._emit_event("a2a.research.orchestrator.shutdown", {
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "queries_processed": self._stats["queries_processed"],
        })

        # Shutdown components in reverse order
        for name in reversed(list(self._components.keys())):
            try:
                self._shutdown_component(name)
            except Exception:
                pass

        self._state = AgentState.STOPPED

    def research(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a research query.

        Args:
            query: Research query
            context: Optional query context

        Returns:
            Research results
        """
        if not self.is_ready:
            return {"error": "Agent not ready", "success": False}

        self._stats["queries_processed"] += 1

        # Validate input if enabled
        if self.config.enable_validation:
            validation_result = self._validate_query(query)
            if not validation_result["valid"]:
                return {"error": validation_result["error"], "success": False}

        # Track telemetry if enabled
        start_time = time.time()

        try:
            # Execute through research pipeline
            result = self._execute_research_pipeline(query, context)

            # Track metrics
            if self.config.enable_telemetry:
                self._track_research_telemetry(query, result, start_time)

            return {"data": result, "success": True}

        except Exception as e:
            self._stats["errors_handled"] += 1

            if self.config.enable_telemetry:
                self._track_error_telemetry(query, e)

            return {"error": str(e), "success": False}

    def get_component(self, name: str) -> Optional[Any]:
        """Get a component instance by name."""
        info = self._components.get(name)
        return info.instance if info else None

    def get_health(self) -> Dict[str, Any]:
        """Get overall agent health status."""
        component_health = {}
        healthy_count = 0
        total_count = len(self._components)

        for name, info in self._components.items():
            component_health[name] = info.to_dict()
            if info.status == ComponentStatus.HEALTHY:
                healthy_count += 1

        overall = ComponentStatus.HEALTHY
        if healthy_count < total_count:
            overall = ComponentStatus.DEGRADED
        if healthy_count == 0:
            overall = ComponentStatus.UNHEALTHY

        return {
            "status": overall.value,
            "state": self._state.value,
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "components": component_health,
            "healthy_count": healthy_count,
            "total_count": total_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        if self._start_time:
            self._stats["uptime_seconds"] = time.time() - self._start_time

        return {
            **self._stats,
            "state": self._state.value,
            "component_count": len(self._components),
            "config": {
                "agent_name": self.config.agent_name,
                "agent_version": self.config.agent_version,
                "falkordb_port": self.config.falkordb_port,
                "a2a_heartbeat_interval": self.config.a2a_heartbeat_interval,
            },
        }

    def _initialize_core_components(self) -> None:
        """Initialize Steps 1-40 components."""
        # These would be initialized through ResearchAgentBootstrap
        # For now, register them as components

        core_steps = [
            (1, "scanner", "Codebase Scanner"),
            (2, "python_parser", "Python Parser"),
            (3, "typescript_parser", "TypeScript Parser"),
            (4, "symbol_store", "Symbol Store"),
            (5, "call_graph", "Call Graph Builder"),
            (6, "dependency_builder", "Dependency Builder"),
            (7, "doc_extractor", "Documentation Extractor"),
            (8, "readme_parser", "README Parser"),
            (9, "semantic_engine", "Semantic Search Engine"),
            (10, "context_assembler", "Context Assembler"),
            (11, "reference_resolver", "Reference Resolver"),
            (12, "impact_analyzer", "Impact Analyzer"),
            (13, "pattern_detector", "Pattern Detector"),
            (14, "architecture_mapper", "Architecture Mapper"),
            (15, "knowledge_extractor", "Knowledge Extractor"),
            (16, "query_planner", "Query Planner"),
            (17, "cache_manager", "Cache Manager"),
            (18, "orchestrator", "Research Orchestrator"),
            (19, "bootstrap", "Bootstrap Manager"),
            (20, "result_formatter", "Result Formatter"),
            (21, "query_executor", "Query Executor"),
            (22, "result_ranker", "Result Ranker"),
            (23, "answer_synthesizer", "Answer Synthesizer"),
            (24, "citation_generator", "Citation Generator"),
            (25, "confidence_scorer", "Confidence Scorer"),
            (26, "feedback_integrator", "Feedback Integrator"),
            (27, "incremental_updater", "Incremental Updater"),
            (28, "multi_repo_manager", "Multi-Repo Manager"),
            (29, "research_api", "Research API"),
            (30, "research_cli", "Research CLI"),
            (31, "plugin_system", "Plugin System"),
            (32, "caching_layer", "Tiered Caching"),
            (33, "metrics_collector", "Metrics Collector"),
            (34, "structured_logger", "Structured Logger"),
            (35, "error_handler", "Error Handler"),
            (36, "config_manager", "Config Manager"),
            (37, "health_checker", "Health Checker"),
            (38, "rate_limiter", "Rate Limiter"),
            (39, "batch_processor", "Batch Processor"),
            (40, "event_emitter", "Event Emitter"),
        ]

        for step, name, description in core_steps:
            self._register_component(name, step, description)

    def _initialize_final_components(self) -> None:
        """Initialize Steps 41-50 components."""
        # Step 41: Security Module
        if self.config.enable_security:
            from .security_module import SecurityManager, SecurityConfig
            security = SecurityManager(SecurityConfig())
            self._register_component("security_manager", 41, "Security Manager", security)

        # Step 42: Validation
        if self.config.enable_validation:
            from .validation import ValidationManager, ValidationConfig
            validation = ValidationManager(ValidationConfig())
            self._register_component("validation_manager", 42, "Validation Manager", validation)

        # Step 43: Testing Framework
        if self.config.enable_testing:
            from .testing_framework import TestRunner, TestConfig
            testing = TestRunner(TestConfig())
            self._register_component("test_runner", 43, "Test Runner", testing)

        # Step 44: Documentation
        if self.config.enable_documentation:
            from .documentation import DocumentationManager, DocConfig
            docs = DocumentationManager(DocConfig())
            self._register_component("documentation_manager", 44, "Documentation Manager", docs)

        # Step 45: Migration Tools
        if self.config.enable_migration:
            from .migration_tools import MigrationManager, MigrationConfig
            migration = MigrationManager(MigrationConfig())
            self._register_component("migration_manager", 45, "Migration Manager", migration)

        # Step 46: Backup System
        if self.config.enable_backup:
            from .backup_system import BackupManager, BackupConfig
            backup = BackupManager(BackupConfig())
            self._register_component("backup_manager", 46, "Backup Manager", backup)

        # Step 47: Telemetry
        if self.config.enable_telemetry:
            from .telemetry import TelemetryCollector, TelemetryConfig
            telemetry = TelemetryCollector(TelemetryConfig())
            self._register_component("telemetry_collector", 47, "Telemetry Collector", telemetry)

        # Step 48: Versioning
        if self.config.enable_versioning:
            from .versioning import VersionManager, VersionConfig
            versioning = VersionManager(VersionConfig(
                current_version=self.config.agent_version,
            ))
            self._register_component("version_manager", 48, "Version Manager", versioning)

        # Step 49: Deprecation Manager
        if self.config.enable_deprecation:
            from .deprecation_manager import DeprecationManager, DeprecationConfig
            deprecation = DeprecationManager(DeprecationConfig())
            self._register_component("deprecation_manager", 49, "Deprecation Manager", deprecation)

        # Step 50: This orchestrator (self-registration)
        self._register_component("final_orchestrator", 50, "Final Orchestrator", self)

    def _register_component(
        self,
        name: str,
        step: int,
        description: str,
        instance: Any = None,
    ) -> None:
        """Register a component."""
        self._components[name] = ComponentInfo(
            name=name,
            step=step,
            instance=instance,
            initialized=instance is not None,
            status=ComponentStatus.HEALTHY if instance else ComponentStatus.UNKNOWN,
        )

    def _verify_components(self) -> None:
        """Verify all components are healthy."""
        for name, info in self._components.items():
            info.last_check = time.time()

            if info.instance is not None:
                # Try to call get_stats if available
                if hasattr(info.instance, "get_stats"):
                    try:
                        info.instance.get_stats()
                        info.status = ComponentStatus.HEALTHY
                    except Exception as e:
                        info.status = ComponentStatus.DEGRADED
                        info.error = str(e)
                else:
                    info.status = ComponentStatus.HEALTHY
            else:
                info.status = ComponentStatus.UNKNOWN

    def _shutdown_component(self, name: str) -> None:
        """Shutdown a component."""
        info = self._components.get(name)
        if not info or not info.instance:
            return

        # Try shutdown methods
        if hasattr(info.instance, "shutdown"):
            info.instance.shutdown()
        elif hasattr(info.instance, "close"):
            info.instance.close()
        elif hasattr(info.instance, "stop"):
            info.instance.stop()

    def _heartbeat_loop(self) -> None:
        """Background heartbeat loop."""
        while self._running:
            try:
                self._send_heartbeat()
            except Exception:
                pass
            time.sleep(self.config.a2a_heartbeat_interval)

    def _send_heartbeat(self) -> None:
        """Send A2A heartbeat."""
        self._last_heartbeat = time.time()
        self._stats["heartbeats_sent"] += 1

        self._emit_event("a2a.research.heartbeat", {
            "agent": self.config.agent_name,
            "state": self._state.value,
            "uptime": time.time() - self._start_time if self._start_time else 0,
            "queries_processed": self._stats["queries_processed"],
        })

    def _validate_query(self, query: str) -> Dict[str, Any]:
        """Validate a research query."""
        validation = self.get_component("validation_manager")
        if validation:
            result = validation.validate_input("search_query", {"query": query})
            return {"valid": result.valid, "error": result.errors[0].message if result.errors else None}
        return {"valid": True, "error": None}

    def _execute_research_pipeline(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute the research pipeline."""
        # This would integrate with the actual research components
        # For now, return a placeholder
        return {
            "query": query,
            "context": context,
            "results": [],
            "message": "Research pipeline executed",
        }

    def _track_research_telemetry(
        self,
        query: str,
        result: Dict[str, Any],
        start_time: float,
    ) -> None:
        """Track research telemetry."""
        telemetry = self.get_component("telemetry_collector")
        if telemetry:
            latency_ms = (time.time() - start_time) * 1000
            telemetry.track_search(
                query=query,
                results_count=len(result.get("results", [])),
                latency_ms=latency_ms,
            )

    def _track_error_telemetry(self, query: str, error: Exception) -> None:
        """Track error telemetry."""
        telemetry = self.get_component("telemetry_collector")
        if telemetry:
            telemetry.track_error(
                error_type=type(error).__name__,
                error_message=str(error),
            )

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        self.shutdown()

    def _atexit_handler(self) -> None:
        """Handle atexit cleanup."""
        if self._state not in [AgentState.STOPPED, AgentState.SHUTTING_DOWN]:
            self.shutdown()

    def _emit_event(
        self,
        topic: str,
        data: Dict[str, Any],
        level: str = "info",
    ) -> str:
        """Emit event with file locking (DKIN v30 compliant)."""
        bus_path = Path(self.config.bus_path)
        bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "orchestrator",
            "level": level,
            "actor": self.config.agent_name,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": data,
        }

        with open(bus_path, "a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(event) + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return event_id


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Final Orchestrator."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Research Agent Final Orchestrator (Step 50)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the Research Agent")
    start_parser.add_argument("--foreground", "-f", action="store_true", help="Run in foreground")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show agent status")
    status_parser.add_argument("--json", action="store_true")

    # Health command
    health_parser = subparsers.add_parser("health", help="Show health status")
    health_parser.add_argument("--json", action="store_true")

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show statistics")
    stats_parser.add_argument("--json", action="store_true")

    # Components command
    components_parser = subparsers.add_parser("components", help="List components")
    components_parser.add_argument("--json", action="store_true")

    # Research command
    research_parser = subparsers.add_parser("research", help="Execute research query")
    research_parser.add_argument("query", help="Research query")

    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run orchestrator demo")

    args = parser.parse_args()

    orchestrator = ResearchAgentOrchestrator()

    if args.command == "start":
        print("Initializing Research Agent...")
        if orchestrator.initialize():
            print(f"Agent initialized with {len(orchestrator._components)} components")
            orchestrator.start()
            print("Research Agent started")

            if args.foreground:
                try:
                    while orchestrator._running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down...")
                    orchestrator.shutdown()
        else:
            print("Failed to initialize agent")
            return 1

    elif args.command == "status":
        orchestrator.initialize()
        health = orchestrator.get_health()

        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"Agent Status: {health['state']}")
            print(f"Health: {health['status']}")
            print(f"Components: {health['healthy_count']}/{health['total_count']} healthy")
            print(f"Uptime: {health['uptime_seconds']:.1f}s")

    elif args.command == "health":
        orchestrator.initialize()
        health = orchestrator.get_health()

        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"Overall: {health['status'].upper()}")
            print(f"\nComponents ({health['healthy_count']}/{health['total_count']}):")

            for name, info in sorted(health['components'].items(), key=lambda x: x[1]['step']):
                status_icon = "OK" if info['status'] == 'healthy' else "!!"
                print(f"  [{status_icon}] Step {info['step']:02d}: {name}")

    elif args.command == "stats":
        orchestrator.initialize()
        stats = orchestrator.get_stats()

        if args.json:
            print(json.dumps(stats, indent=2))
        else:
            print("Orchestrator Statistics:")
            print(f"  State: {stats['state']}")
            print(f"  Components: {stats['component_count']}")
            print(f"  Queries: {stats['queries_processed']}")
            print(f"  Errors: {stats['errors_handled']}")
            print(f"  Heartbeats: {stats['heartbeats_sent']}")
            print(f"  Uptime: {stats['uptime_seconds']:.1f}s")

    elif args.command == "components":
        orchestrator.initialize()

        if args.json:
            components = {
                name: info.to_dict()
                for name, info in orchestrator._components.items()
            }
            print(json.dumps(components, indent=2))
        else:
            print(f"Research Agent Components ({len(orchestrator._components)}):")
            print("-" * 60)

            # Group by step ranges
            groups = [
                (1, 10, "Core Scanning & Parsing"),
                (11, 20, "Search & Analysis"),
                (21, 30, "Execution & API"),
                (31, 40, "Infrastructure"),
                (41, 50, "Final Components"),
            ]

            for start, end, title in groups:
                print(f"\n{title} (Steps {start}-{end}):")
                for name, info in sorted(
                    orchestrator._components.items(),
                    key=lambda x: x[1].step
                ):
                    if start <= info.step <= end:
                        status = "OK" if info.initialized else "--"
                        print(f"  [{status}] {info.step:02d}. {name}")

    elif args.command == "research":
        orchestrator.initialize()
        orchestrator.start()

        print(f"Executing research query: {args.query}")
        result = orchestrator.research(args.query)

        if result["success"]:
            print(f"Result: {json.dumps(result['data'], indent=2)}")
        else:
            print(f"Error: {result['error']}")
            return 1

        orchestrator.shutdown()

    elif args.command == "demo":
        print("=" * 60)
        print("Research Agent Final Orchestrator Demo (Step 50)")
        print("=" * 60)
        print()

        print("1. Initializing Research Agent...")
        if orchestrator.initialize():
            print(f"   Components: {len(orchestrator._components)}")
            print(f"   State: {orchestrator.state.value}")
        else:
            print("   Failed to initialize!")
            return 1

        print("\n2. Component Summary:")
        health = orchestrator.get_health()
        print(f"   Healthy: {health['healthy_count']}/{health['total_count']}")

        print("\n3. Starting agent...")
        orchestrator.start()
        print(f"   State: {orchestrator.state.value}")

        print("\n4. Executing sample research query...")
        result = orchestrator.research("Find all classes implementing the Observer pattern")
        print(f"   Success: {result['success']}")
        print(f"   Queries processed: {orchestrator._stats['queries_processed']}")

        print("\n5. Health check:")
        health = orchestrator.get_health()
        print(f"   Status: {health['status']}")
        print(f"   Uptime: {health['uptime_seconds']:.2f}s")

        print("\n6. Final Statistics:")
        stats = orchestrator.get_stats()
        for key, value in stats.items():
            if key != "config":
                print(f"   {key}: {value}")

        print("\n7. Shutting down...")
        orchestrator.shutdown()
        print(f"   Final state: {orchestrator.state.value}")

        print("\n" + "=" * 60)
        print("Research Agent (Steps 1-50) Complete!")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
