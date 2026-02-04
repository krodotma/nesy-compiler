#!/usr/bin/env python3
"""
AutoGen Bridge for Pluribus
===========================

Bridges Microsoft AutoGen multi-agent conversations to the Pluribus event bus.
Maps AutoGen's ConversableAgent to Pluribus actor model and emits structured
bus events for all inter-agent messages.

Integration points:
- AutoGen GroupChat -> STRp star topology
- AutoGen AssistantAgent -> Pluribus persona
- AutoGen UserProxyAgent -> bus request/response
- Message history -> bus event lineage

Usage:
    from nucleus.tools.autogen_bridge import AutoGenBridge, PluribusAgent

    bridge = AutoGenBridge()

    analyst = bridge.create_agent("analyst", system_message="You analyze data.")
    coder = bridge.create_agent("coder", system_message="You write code.")

    result = bridge.run_conversation(
        agents=[analyst, coder],
        initial_message="Analyze this dataset and generate a report.",
        max_rounds=5
    )

Service Registry Entry:
    id: autogen-bridge
    kind: process
    entry_point: nucleus/tools/autogen_bridge.py
    tags: [autogen, microsoft, orchestration, sota]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypedDict

sys.dont_write_bytecode = True

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from agent_bus import resolve_bus_paths, emit_event, default_actor  # type: ignore


# Type definitions for bus events
class AutoGenMessageEvent(TypedDict, total=False):
    req_id: str
    trace_id: str
    sender: str
    recipient: str
    content: str
    role: str
    round_num: int
    group_id: str
    timestamp: float


class AutoGenConversationEvent(TypedDict, total=False):
    req_id: str
    trace_id: str
    group_id: str
    agents: List[str]
    initial_message: str
    max_rounds: int
    status: str
    total_messages: int
    final_summary: str


# Effects typing alignment with Lens/Collimator
EffectsType = str  # "none" | "file" | "network" | "unknown"


@dataclass
class AgentConfig:
    """Configuration for a Pluribus-wrapped AutoGen agent."""
    name: str
    system_message: str
    model: str = "auto"
    human_input_mode: str = "NEVER"  # NEVER | TERMINATE | ALWAYS
    max_consecutive_auto_reply: int = 10
    code_execution_config: Optional[Dict[str, Any]] = None
    llm_config: Optional[Dict[str, Any]] = None
    effects: EffectsType = "none"
    persona_id: str = ""
    tags: List[str] = field(default_factory=list)


@dataclass
class ConversationResult:
    """Result of a multi-agent conversation."""
    req_id: str
    trace_id: str
    group_id: str
    messages: List[Dict[str, Any]]
    final_message: str
    total_rounds: int
    status: str  # "success" | "terminated" | "max_rounds" | "error"
    cost: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def _default_bus_dir() -> str:
    """Resolve the default bus directory."""
    env = (os.environ.get("PLURIBUS_BUS_DIR") or "").strip()
    if env:
        return env
    repo_bus = Path("/pluribus/.pluribus/bus")
    if repo_bus.exists():
        return str(repo_bus)
    return str(Path.home() / ".local" / "state" / "nucleus" / "bus")


def now_iso_utc() -> str:
    """Return current UTC timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class PluribusAgent:
    """
    Wrapper around AutoGen ConversableAgent that emits Pluribus bus events.

    This class can work in two modes:
    1. With AutoGen installed: Full AutoGen functionality
    2. Without AutoGen (mock mode): Simulated conversations for testing
    """

    def __init__(
        self,
        config: AgentConfig,
        bridge: "AutoGenBridge",
        autogen_agent: Any = None
    ):
        self.config = config
        self.bridge = bridge
        self._autogen_agent = autogen_agent
        self._message_history: List[Dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def autogen_agent(self) -> Any:
        """Return the underlying AutoGen agent if available."""
        return self._autogen_agent

    def send(self, message: str, recipient: "PluribusAgent", request_reply: bool = True) -> Optional[str]:
        """
        Send a message to another agent, emitting bus events.

        Args:
            message: The message content
            recipient: The recipient PluribusAgent
            request_reply: Whether to request a reply

        Returns:
            The reply message if request_reply is True, else None
        """
        msg_event: AutoGenMessageEvent = {
            "req_id": self.bridge.current_req_id,
            "trace_id": self.bridge.current_trace_id,
            "sender": self.name,
            "recipient": recipient.name,
            "content": message,
            "role": "assistant",
            "round_num": len(self._message_history),
            "group_id": self.bridge.current_group_id,
            "timestamp": time.time(),
        }

        # Emit send event
        self.bridge.emit_bus_event(
            topic="autogen.message.send",
            kind="log",
            level="info",
            data=msg_event
        )

        self._message_history.append({
            "sender": self.name,
            "recipient": recipient.name,
            "content": message,
            "timestamp": time.time()
        })

        # If AutoGen is available, use it
        if self._autogen_agent and recipient._autogen_agent:
            try:
                self._autogen_agent.send(message, recipient._autogen_agent, request_reply=request_reply)
                if request_reply:
                    # Get the last message from recipient's chat history
                    chat_hist = self._autogen_agent.chat_messages.get(recipient._autogen_agent, [])
                    if chat_hist:
                        return str(chat_hist[-1].get("content", ""))
            except Exception as e:
                self.bridge.emit_bus_event(
                    topic="autogen.message.error",
                    kind="log",
                    level="error",
                    data={"error": str(e), **msg_event}
                )

        return None

    def receive(self, message: str, sender: "PluribusAgent") -> str:
        """
        Receive and process a message from another agent.

        Args:
            message: The message content
            sender: The sending PluribusAgent

        Returns:
            The generated reply
        """
        msg_event: AutoGenMessageEvent = {
            "req_id": self.bridge.current_req_id,
            "trace_id": self.bridge.current_trace_id,
            "sender": sender.name,
            "recipient": self.name,
            "content": message,
            "role": "user",
            "round_num": len(self._message_history),
            "group_id": self.bridge.current_group_id,
            "timestamp": time.time(),
        }

        # Emit receive event
        self.bridge.emit_bus_event(
            topic="autogen.message.receive",
            kind="log",
            level="info",
            data=msg_event
        )

        self._message_history.append({
            "sender": sender.name,
            "recipient": self.name,
            "content": message,
            "timestamp": time.time()
        })

        # Generate reply (mock implementation if AutoGen not available)
        if self._autogen_agent:
            try:
                reply = self._autogen_agent.generate_reply(
                    messages=[{"role": "user", "content": message}],
                    sender=sender._autogen_agent
                )
                return str(reply) if reply else ""
            except Exception as e:
                self.bridge.emit_bus_event(
                    topic="autogen.reply.error",
                    kind="log",
                    level="error",
                    data={"error": str(e), "agent": self.name}
                )
                return f"[Error generating reply: {e}]"
        else:
            # Mock reply for testing without AutoGen
            return f"[{self.name}] Acknowledged: {message[:100]}..."

    def reset(self) -> None:
        """Reset agent state."""
        self._message_history.clear()
        if self._autogen_agent and hasattr(self._autogen_agent, "reset"):
            self._autogen_agent.reset()


class AutoGenBridge:
    """
    Bridge between AutoGen multi-agent framework and Pluribus event bus.

    Maps AutoGen concepts to Pluribus:
    - ConversableAgent -> PluribusAgent (actor model)
    - GroupChat -> Star topology conversation
    - Message passing -> Bus events
    - Chat history -> Event lineage

    Emits events to topics:
    - autogen.conversation.start
    - autogen.conversation.end
    - autogen.message.send
    - autogen.message.receive
    - autogen.group.formed
    - autogen.agent.created
    """

    def __init__(
        self,
        bus_dir: Optional[str] = None,
        actor: Optional[str] = None,
        use_autogen: bool = True
    ):
        self.bus_dir = bus_dir or _default_bus_dir()
        self.actor = actor or os.environ.get("PLURIBUS_ACTOR") or default_actor()
        self._bus_paths = resolve_bus_paths(self.bus_dir)

        # Conversation state
        self.current_req_id: str = ""
        self.current_trace_id: str = ""
        self.current_group_id: str = ""

        # Agent registry
        self._agents: Dict[str, PluribusAgent] = {}

        # Try to import AutoGen
        self._autogen_available = False
        self._autogen = None

        if use_autogen:
            try:
                import autogen  # type: ignore
                self._autogen = autogen
                self._autogen_available = True
            except ImportError:
                pass

    @property
    def autogen_available(self) -> bool:
        """Check if AutoGen is available."""
        return self._autogen_available

    def emit_bus_event(
        self,
        topic: str,
        kind: str,
        level: str,
        data: Dict[str, Any]
    ) -> str:
        """Emit an event to the Pluribus bus."""
        return emit_event(
            self._bus_paths,
            topic=topic,
            kind=kind,
            level=level,
            actor=self.actor,
            data=data,
            trace_id=self.current_trace_id or None,
            run_id=self.current_req_id or None,
            durable=False
        )

    def create_agent(
        self,
        name: str,
        system_message: str = "",
        model: str = "auto",
        human_input_mode: str = "NEVER",
        code_execution_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        effects: EffectsType = "none",
        persona_id: str = "",
        tags: Optional[List[str]] = None
    ) -> PluribusAgent:
        """
        Create a PluribusAgent wrapping an AutoGen ConversableAgent.

        Args:
            name: Agent name (must be unique)
            system_message: System prompt for the agent
            model: Model to use (default: auto)
            human_input_mode: NEVER, TERMINATE, or ALWAYS
            code_execution_config: Config for code execution
            llm_config: LLM configuration dict
            effects: Effect type for Lens/Collimator alignment
            persona_id: Pluribus persona ID mapping
            tags: Tags for filtering/discovery

        Returns:
            A PluribusAgent instance
        """
        config = AgentConfig(
            name=name,
            system_message=system_message,
            model=model,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            effects=effects,
            persona_id=persona_id,
            tags=tags or []
        )

        autogen_agent = None
        if self._autogen_available and self._autogen:
            try:
                # Build LLM config
                ag_llm_config = llm_config or {}
                if not ag_llm_config and model != "auto":
                    ag_llm_config = {"model": model}

                # Create AutoGen agent
                autogen_agent = self._autogen.ConversableAgent(
                    name=name,
                    system_message=system_message,
                    human_input_mode=human_input_mode,
                    max_consecutive_auto_reply=config.max_consecutive_auto_reply,
                    code_execution_config=code_execution_config or False,
                    llm_config=ag_llm_config if ag_llm_config else None,
                )
            except Exception as e:
                # Fall back to mock mode
                self.emit_bus_event(
                    topic="autogen.agent.error",
                    kind="log",
                    level="warn",
                    data={"error": str(e), "agent": name, "fallback": "mock"}
                )

        agent = PluribusAgent(config, self, autogen_agent)
        self._agents[name] = agent

        # Emit agent created event
        self.emit_bus_event(
            topic="autogen.agent.created",
            kind="metric",
            level="info",
            data={
                "name": name,
                "model": model,
                "effects": effects,
                "persona_id": persona_id,
                "autogen_mode": self._autogen_available and autogen_agent is not None,
                "tags": tags or []
            }
        )

        return agent

    def create_assistant(
        self,
        name: str,
        system_message: str = "You are a helpful AI assistant.",
        model: str = "auto",
        **kwargs: Any
    ) -> PluribusAgent:
        """
        Create an assistant agent (convenience wrapper).

        This maps to AutoGen's AssistantAgent.
        """
        return self.create_agent(
            name=name,
            system_message=system_message,
            model=model,
            human_input_mode="NEVER",
            **kwargs
        )

    def create_user_proxy(
        self,
        name: str = "user_proxy",
        human_input_mode: str = "NEVER",
        code_execution_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> PluribusAgent:
        """
        Create a user proxy agent (convenience wrapper).

        This maps to AutoGen's UserProxyAgent.
        """
        return self.create_agent(
            name=name,
            system_message="",
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            effects="file" if code_execution_config else "none",
            **kwargs
        )

    def run_conversation(
        self,
        agents: List[PluribusAgent],
        initial_message: str,
        max_rounds: int = 10,
        req_id: Optional[str] = None,
        trace_id: Optional[str] = None
    ) -> ConversationResult:
        """
        Run a multi-agent conversation with bus event emission.

        Args:
            agents: List of PluribusAgents to participate
            initial_message: The initial prompt/message
            max_rounds: Maximum conversation rounds
            req_id: Request ID for tracing
            trace_id: Trace ID for lineage

        Returns:
            ConversationResult with messages and metadata
        """
        self.current_req_id = req_id or str(uuid.uuid4())
        self.current_trace_id = trace_id or str(uuid.uuid4())
        self.current_group_id = f"group_{str(uuid.uuid4())[:8]}"

        agent_names = [a.name for a in agents]

        # Emit conversation start
        self.emit_bus_event(
            topic="autogen.conversation.start",
            kind="metric",
            level="info",
            data={
                "req_id": self.current_req_id,
                "trace_id": self.current_trace_id,
                "group_id": self.current_group_id,
                "agents": agent_names,
                "initial_message": initial_message[:500],
                "max_rounds": max_rounds,
            }
        )

        messages: List[Dict[str, Any]] = []
        final_message = ""
        status = "success"

        try:
            if self._autogen_available and all(a.autogen_agent for a in agents):
                # Use AutoGen GroupChat if available
                result = self._run_autogen_conversation(agents, initial_message, max_rounds)
                messages = result.get("messages", [])
                final_message = result.get("final_message", "")
                status = result.get("status", "success")
            else:
                # Mock conversation for testing
                result = self._run_mock_conversation(agents, initial_message, max_rounds)
                messages = result.get("messages", [])
                final_message = result.get("final_message", "")
                status = result.get("status", "success")
        except Exception as e:
            status = "error"
            final_message = f"Error: {e}"
            self.emit_bus_event(
                topic="autogen.conversation.error",
                kind="log",
                level="error",
                data={"error": str(e), "group_id": self.current_group_id}
            )

        # Emit conversation end
        self.emit_bus_event(
            topic="autogen.conversation.end",
            kind="metric",
            level="info",
            data={
                "req_id": self.current_req_id,
                "trace_id": self.current_trace_id,
                "group_id": self.current_group_id,
                "agents": agent_names,
                "status": status,
                "total_messages": len(messages),
                "final_summary": final_message[:500] if final_message else "",
            }
        )

        return ConversationResult(
            req_id=self.current_req_id,
            trace_id=self.current_trace_id,
            group_id=self.current_group_id,
            messages=messages,
            final_message=final_message,
            total_rounds=len(messages),
            status=status,
        )

    def _run_autogen_conversation(
        self,
        agents: List[PluribusAgent],
        initial_message: str,
        max_rounds: int
    ) -> Dict[str, Any]:
        """Run conversation using AutoGen GroupChat."""
        if not self._autogen:
            return {"messages": [], "final_message": "", "status": "error"}

        try:
            autogen_agents = [a.autogen_agent for a in agents if a.autogen_agent]

            groupchat = self._autogen.GroupChat(
                agents=autogen_agents,
                messages=[],
                max_round=max_rounds
            )

            manager = self._autogen.GroupChatManager(
                groupchat=groupchat,
                llm_config={"model": "gpt-4"}  # Default, can be overridden
            )

            # Initiate chat
            initiator = autogen_agents[0] if autogen_agents else None
            if initiator:
                initiator.initiate_chat(manager, message=initial_message)

            # Extract messages
            messages = []
            for msg in groupchat.messages:
                messages.append({
                    "sender": msg.get("name", "unknown"),
                    "content": msg.get("content", ""),
                    "role": msg.get("role", "assistant"),
                    "timestamp": time.time()
                })

                # Emit per-message event
                self.emit_bus_event(
                    topic="autogen.message.exchange",
                    kind="log",
                    level="info",
                    data={
                        "sender": msg.get("name", "unknown"),
                        "content": str(msg.get("content", ""))[:500],
                        "group_id": self.current_group_id,
                    }
                )

            final_msg = messages[-1]["content"] if messages else ""
            return {"messages": messages, "final_message": final_msg, "status": "success"}

        except Exception as e:
            return {"messages": [], "final_message": str(e), "status": "error"}

    def _run_mock_conversation(
        self,
        agents: List[PluribusAgent],
        initial_message: str,
        max_rounds: int
    ) -> Dict[str, Any]:
        """Run a mock conversation for testing without AutoGen."""
        messages: List[Dict[str, Any]] = []

        # Initial message
        messages.append({
            "sender": "user",
            "content": initial_message,
            "role": "user",
            "timestamp": time.time()
        })

        # Simulate rounds
        current_message = initial_message
        for round_num in range(min(max_rounds, len(agents))):
            agent = agents[round_num % len(agents)]

            # Generate mock response
            response = f"[{agent.name}] Processing: {current_message[:50]}... (mock response round {round_num + 1})"

            messages.append({
                "sender": agent.name,
                "content": response,
                "role": "assistant",
                "timestamp": time.time()
            })

            # Emit message event
            self.emit_bus_event(
                topic="autogen.message.exchange",
                kind="log",
                level="info",
                data={
                    "sender": agent.name,
                    "content": response[:500],
                    "group_id": self.current_group_id,
                    "round": round_num + 1,
                    "mock": True
                }
            )

            current_message = response

        final_msg = messages[-1]["content"] if messages else ""
        return {"messages": messages, "final_message": final_msg, "status": "success"}

    def get_agent(self, name: str) -> Optional[PluribusAgent]:
        """Get an agent by name."""
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def reset_all(self) -> None:
        """Reset all agents."""
        for agent in self._agents.values():
            agent.reset()


# Service Registry Entry (for BUILTIN_SERVICES)
AUTOGEN_SERVICE_DEF = {
    "id": "autogen-bridge",
    "name": "AutoGen Bridge",
    "kind": "process",
    "entry_point": "nucleus/tools/autogen_bridge.py",
    "description": "Bridges Microsoft AutoGen multi-agent conversations to Pluribus event bus",
    "tags": ["autogen", "microsoft", "orchestration", "sota", "multi-agent"],
    "lineage": "sota.microsoft",
    "omega_motif": False,
    "gates": {"E": "conversation_completion", "P": "message_provenance"},
}


def cmd_demo(args: argparse.Namespace) -> int:
    """Run a demo conversation."""
    bridge = AutoGenBridge(bus_dir=args.bus_dir)

    print(f"AutoGen available: {bridge.autogen_available}")
    print("Creating agents...")

    analyst = bridge.create_assistant(
        name="analyst",
        system_message="You are a data analyst. Analyze data and provide insights."
    )

    coder = bridge.create_assistant(
        name="coder",
        system_message="You are a Python developer. Write clean, efficient code."
    )

    reviewer = bridge.create_assistant(
        name="reviewer",
        system_message="You review work and provide constructive feedback."
    )

    print(f"Agents created: {bridge.list_agents()}")
    print("Running conversation...")

    result = bridge.run_conversation(
        agents=[analyst, coder, reviewer],
        initial_message=args.message or "Analyze the concept of prime numbers and write Python code to find the first 10 primes.",
        max_rounds=args.rounds
    )

    print(f"\nConversation completed:")
    print(f"  Status: {result.status}")
    print(f"  Messages: {len(result.messages)}")
    print(f"  Group ID: {result.group_id}")
    print(f"  Trace ID: {result.trace_id}")
    print(f"\nFinal message:\n{result.final_message}")

    return 0


def cmd_list_agents(args: argparse.Namespace) -> int:
    """List available agent types."""
    print("Available agent types:")
    print("  assistant - General purpose AI assistant")
    print("  user_proxy - User proxy for code execution")
    print("  custom - Custom agent with full configuration")
    print("\nAutoGen concepts mapped to Pluribus:")
    print("  ConversableAgent -> PluribusAgent (actor model)")
    print("  GroupChat -> Star topology conversation")
    print("  Message passing -> Bus events")
    print("  Chat history -> Event lineage")
    return 0


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="autogen_bridge.py",
        description="AutoGen Bridge for Pluribus - Multi-agent conversation orchestration"
    )
    parser.add_argument("--bus-dir", default=None, help="Bus directory")

    sub = parser.add_subparsers(dest="cmd", required=True)

    demo_p = sub.add_parser("demo", help="Run a demo conversation")
    demo_p.add_argument("--message", default=None, help="Initial message")
    demo_p.add_argument("--rounds", type=int, default=3, help="Max rounds")
    demo_p.set_defaults(func=cmd_demo)

    list_p = sub.add_parser("list-agents", help="List agent types")
    list_p.set_defaults(func=cmd_list_agents)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
