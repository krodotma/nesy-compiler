#!/usr/bin/env python3
"""
bootstrap.py - Research Agent Bootstrap Module (Step 1)

Implements the Research Agent initialization and lifecycle management.

PBTSO Phase: SKILL, SEQUESTER

Bus Topics:
- a2a.research.bootstrap.start
- a2a.research.bootstrap.complete
- a2a.task.dispatch (subscription)

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import json
import os
import socket
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ResearchAgentConfig:
    """Configuration for the Research Agent."""

    agent_id: str = "research-agent"
    ring_level: int = 2  # Security ring level (0=highest)
    max_context_tokens: int = 100000
    index_batch_size: int = 500
    scan_batch_size: int = 100
    parallel_parse_workers: int = 4

    # Feature flags
    enable_ast_parsing: bool = True
    enable_dependency_graph: bool = True
    enable_call_graph: bool = True
    enable_doc_extraction: bool = True

    # Paths
    cache_dir: Optional[str] = None
    index_dir: Optional[str] = None

    def __post_init__(self):
        if self.cache_dir is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.cache_dir = f"{pluribus_root}/.pluribus/research/cache"
        if self.index_dir is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            self.index_dir = f"{pluribus_root}/.pluribus/research/index"


# ============================================================================
# Simple Agent Bus (standalone for oagent)
# ============================================================================


class AgentBus:
    """
    Lightweight Agent Bus for A2A event emission.

    Writes NDJSON events to the Pluribus bus for cross-agent coordination.
    """

    def __init__(self, bus_path: Optional[Path] = None):
        if bus_path is None:
            pluribus_root = os.environ.get("PLURIBUS_ROOT", "/pluribus")
            bus_path = Path(pluribus_root) / ".pluribus" / "bus" / "events.ndjson"
        self.bus_path = Path(bus_path)
        self.actor = os.environ.get("PLURIBUS_ACTOR", "research-agent")
        self._subscribers: Dict[str, List[Callable]] = {}

    def emit(self, event: Dict[str, Any]) -> str:
        """
        Emit an event to the bus.

        Args:
            event: Event dictionary with topic, kind, actor, data fields

        Returns:
            Event ID
        """
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event_id = str(uuid.uuid4())
        full_event = {
            "id": event_id,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": event.get("topic", "unknown"),
            "kind": event.get("kind", "event"),
            "level": event.get("level", "info"),
            "actor": event.get("actor", self.actor),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "data": event.get("data", {}),
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(full_event) + "\n")

        # Notify local subscribers
        topic = full_event["topic"]
        for subscriber in self._subscribers.get(topic, []):
            try:
                subscriber(full_event)
            except Exception:
                pass

        return event_id

    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to a topic with a handler function."""
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(handler)

    def unsubscribe(self, topic: str, handler: Callable) -> bool:
        """Unsubscribe a handler from a topic."""
        if topic in self._subscribers and handler in self._subscribers[topic]:
            self._subscribers[topic].remove(handler)
            return True
        return False


# ============================================================================
# Research Agent Bootstrap
# ============================================================================


@dataclass
class ResearchAgentState:
    """Current state of the Research Agent."""

    initialized: bool = False
    bootstrapped: bool = False
    scanning: bool = False
    indexing: bool = False
    files_scanned: int = 0
    files_indexed: int = 0
    symbols_indexed: int = 0
    errors: List[str] = field(default_factory=list)
    last_scan_ts: Optional[float] = None
    last_index_ts: Optional[float] = None


class ResearchAgentBootstrap:
    """
    Bootstrap and lifecycle manager for the Research Agent.

    Responsibilities:
    - Initialize agent capabilities
    - Register with A2A dispatcher
    - Coordinate with other subagents
    - Emit lifecycle events to bus

    Example:
        config = ResearchAgentConfig()
        agent = ResearchAgentBootstrap(config)
        await agent.bootstrap()
    """

    LIFECYCLE_TOPICS = {
        "start": "a2a.research.bootstrap.start",
        "complete": "a2a.research.bootstrap.complete",
        "error": "a2a.research.bootstrap.error",
        "shutdown": "a2a.research.shutdown",
    }

    def __init__(self, config: Optional[ResearchAgentConfig] = None):
        """
        Initialize the Research Agent Bootstrap.

        Args:
            config: Agent configuration. Uses defaults if not provided.
        """
        self.config = config or ResearchAgentConfig()
        self.bus = AgentBus()
        self.state = ResearchAgentState()
        self._handlers: Dict[str, Callable] = {}
        self._components: Dict[str, Any] = {}

        # Subscribe to task dispatch
        self.bus.subscribe("a2a.task.dispatch", self._handle_task_dispatch)

    async def bootstrap(self) -> bool:
        """
        Initialize Research Agent capabilities.

        Emits a2a.research.bootstrap.start on init.
        Emits a2a.research.bootstrap.complete on success.

        Returns:
            True if bootstrap successful, False otherwise.
        """
        # Emit start event
        self.bus.emit({
            "topic": self.LIFECYCLE_TOPICS["start"],
            "kind": "lifecycle",
            "actor": self.config.agent_id,
            "data": {
                "ring": self.config.ring_level,
                "max_context_tokens": self.config.max_context_tokens,
                "features": {
                    "ast_parsing": self.config.enable_ast_parsing,
                    "dependency_graph": self.config.enable_dependency_graph,
                    "call_graph": self.config.enable_call_graph,
                    "doc_extraction": self.config.enable_doc_extraction,
                },
            }
        })

        try:
            # Initialize directories
            await self._init_directories()

            # Register capabilities
            await self._register_capabilities()

            # Initialize components
            await self._init_components()

            self.state.initialized = True
            self.state.bootstrapped = True

            # Emit complete event
            self.bus.emit({
                "topic": self.LIFECYCLE_TOPICS["complete"],
                "kind": "lifecycle",
                "actor": self.config.agent_id,
                "data": {
                    "status": "success",
                    "components": list(self._components.keys()),
                }
            })

            return True

        except Exception as e:
            self.state.errors.append(str(e))

            self.bus.emit({
                "topic": self.LIFECYCLE_TOPICS["error"],
                "kind": "lifecycle",
                "level": "error",
                "actor": self.config.agent_id,
                "data": {
                    "error": str(e),
                    "phase": "bootstrap",
                }
            })

            return False

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the Research Agent.

        Returns:
            True if shutdown successful.
        """
        self.bus.emit({
            "topic": self.LIFECYCLE_TOPICS["shutdown"],
            "kind": "lifecycle",
            "actor": self.config.agent_id,
            "data": {
                "files_scanned": self.state.files_scanned,
                "files_indexed": self.state.files_indexed,
                "symbols_indexed": self.state.symbols_indexed,
            }
        })

        self.state.initialized = False
        self.state.bootstrapped = False

        return True

    async def _init_directories(self) -> None:
        """Create required directories."""
        Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.index_dir).mkdir(parents=True, exist_ok=True)

    async def _register_capabilities(self) -> None:
        """Register agent capabilities with the coordination system."""
        capabilities = []

        if self.config.enable_ast_parsing:
            capabilities.extend(["ast.parse.python", "ast.parse.typescript"])

        if self.config.enable_dependency_graph:
            capabilities.append("graph.dependency")

        if self.config.enable_call_graph:
            capabilities.append("graph.callgraph")

        if self.config.enable_doc_extraction:
            capabilities.extend(["docs.extract", "docs.readme"])

        self.bus.emit({
            "topic": "research.capability.register",
            "kind": "capability",
            "actor": self.config.agent_id,
            "data": {
                "capabilities": capabilities,
                "ring": self.config.ring_level,
            }
        })

    async def _init_components(self) -> None:
        """Initialize internal components."""
        # Components will be registered lazily when needed
        self._components["scanner"] = None  # Initialized on first scan
        self._components["parser_registry"] = None  # Initialized on first parse
        self._components["symbol_store"] = None  # Initialized on first index

    def _handle_task_dispatch(self, event: Dict[str, Any]) -> None:
        """Handle incoming task dispatch events."""
        data = event.get("data", {})
        target = data.get("target")

        if target != self.config.agent_id:
            return

        topic = data.get("topic", "")

        # Route to appropriate handler
        if topic.startswith("research."):
            handler = self._handlers.get(topic)
            if handler:
                handler(data)

    def register_handler(self, topic: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """Register a handler for a specific topic."""
        self._handlers[topic] = handler

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state as dictionary."""
        return asdict(self.state)

    def get_component(self, name: str) -> Optional[Any]:
        """Get a registered component by name."""
        return self._components.get(name)

    def register_component(self, name: str, component: Any) -> None:
        """Register a component."""
        self._components[name] = component


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Research Agent Bootstrap."""
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(
        description="Research Agent Bootstrap (Step 1)"
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Bootstrap the Research Agent"
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show agent status"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    # Load config if provided
    config = ResearchAgentConfig()
    if args.config:
        with open(args.config) as f:
            config_data = json.load(f)
            config = ResearchAgentConfig(**config_data)

    agent = ResearchAgentBootstrap(config)

    if args.bootstrap:
        success = asyncio.run(agent.bootstrap())
        if args.json:
            print(json.dumps({"success": success, "state": agent.get_state()}))
        else:
            print(f"Bootstrap {'successful' if success else 'failed'}")
            if not success:
                print(f"Errors: {agent.state.errors}")
        return 0 if success else 1

    if args.status:
        state = agent.get_state()
        if args.json:
            print(json.dumps(state, indent=2))
        else:
            print(f"Research Agent Status:")
            print(f"  Initialized: {state['initialized']}")
            print(f"  Bootstrapped: {state['bootstrapped']}")
            print(f"  Files Scanned: {state['files_scanned']}")
            print(f"  Files Indexed: {state['files_indexed']}")
            print(f"  Symbols Indexed: {state['symbols_indexed']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
