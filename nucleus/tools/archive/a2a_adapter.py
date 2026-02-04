#!/usr/bin/env python3
"""
A2A Adapter - Agent2Agent Protocol Implementation for Pluribus
==============================================================

Implements the Agent2Agent Protocol (a2a-protocol.org) for multi-agent interoperability.
Enables Pluribus agents to discover and communicate with external agents via Agent Cards.

DKIN v25 compliant - maps A2A messages to Pluribus bus events.

Features:
- Agent Card generation and serving at /.well-known/agent.json
- A2A message -> bus event translation
- External agent discovery via Agent Card URLs
- Skill negotiation and capability matching

Usage:
    python3 a2a_adapter.py --serve                     # Serve Agent Card
    python3 a2a_adapter.py --discover URL              # Discover external agent
    python3 a2a_adapter.py --send URL MESSAGE          # Send A2A message
    python3 a2a_adapter.py --generate-card AGENT_ID    # Generate Agent Card
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus_event(topic: str, data: dict, level: str = "info") -> None:
    """Emit event to Pluribus bus."""
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if not bus_dir:
        # parents[2] = /pluribus (from nucleus/tools/a2a_adapter.py)
        bus_dir = str(Path(__file__).resolve().parents[2] / ".pluribus" / "bus")

    bus_path = Path(bus_dir)
    if not bus_path.exists():
        bus_path.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": "event",
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "a2a-adapter"),
        "data": data,
    }

    events_file = bus_path / "events.ndjson"
    with events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class AgentSkill:
    """A2A Agent Skill definition."""
    id: str
    name: str
    description: str
    input_schema: dict[str, Any] | None = None
    output_schema: dict[str, Any] | None = None


@dataclass
class AgentCard:
    """A2A Agent Card per a2a-protocol.org specification."""
    name: str
    description: str
    version: str
    url: str
    skills: list[AgentSkill] = field(default_factory=list)
    authentication: dict[str, Any] | None = None
    capabilities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "url": self.url,
            "skills": [asdict(s) for s in self.skills],
            "authentication": self.authentication,
            "capabilities": self.capabilities,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# Predefined Pluribus Agent Cards
PLURIBUS_AGENTS: dict[str, AgentCard] = {
    "pluribus-coordinator": AgentCard(
        name="Pluribus Coordinator",
        description="Multi-agent orchestration and task dispatch for the Pluribus ecosystem",
        version="1.0.0",
        url="https://kroma.live/.well-known/agent.json",
        skills=[
            AgentSkill(
                id="task-dispatch",
                name="Task Dispatch",
                description="Dispatch tasks to specialized agents based on capability matching",
                input_schema={"type": "object", "properties": {"task": {"type": "string"}, "constraints": {"type": "object"}}},
            ),
            AgentSkill(
                id="agent-discovery",
                name="Agent Discovery",
                description="Discover and register agents in the Pluribus mesh",
            ),
            AgentSkill(
                id="lane-coordination",
                name="Lane Coordination",
                description="Coordinate work across parallel agent lanes (PBLANES)",
            ),
        ],
        capabilities=["multi-agent", "orchestration", "mcp", "bus-events"],
        metadata={"protocol_version": "DKIN v25", "bus_enabled": True},
    ),
    "sota-researcher": AgentCard(
        name="SOTA Researcher",
        description="State-of-the-art research, paper distillation, and technology analysis",
        version="1.0.0",
        url="https://kroma.live/.well-known/agents/sota-researcher.json",
        skills=[
            AgentSkill(
                id="paper-distill",
                name="Paper Distillation",
                description="Extract key insights from academic papers and documentation",
            ),
            AgentSkill(
                id="sota-analysis",
                name="SOTA Analysis",
                description="Analyze state-of-the-art in a technology domain",
            ),
            AgentSkill(
                id="capability-mapping",
                name="Capability Mapping",
                description="Map capabilities across tools, models, and frameworks",
            ),
        ],
        capabilities=["research", "distillation", "web-search"],
    ),
    "strp-worker": AgentCard(
        name="STRp Worker",
        description="Task execution agent for code generation, testing, and file operations",
        version="1.0.0",
        url="https://kroma.live/.well-known/agents/strp-worker.json",
        skills=[
            AgentSkill(
                id="code-generation",
                name="Code Generation",
                description="Generate code in multiple languages with test coverage",
            ),
            AgentSkill(
                id="file-operations",
                name="File Operations",
                description="Read, write, and modify files with bus event emission",
            ),
            AgentSkill(
                id="test-execution",
                name="Test Execution",
                description="Run tests and report results via bus events",
            ),
        ],
        capabilities=["code-gen", "file-ops", "testing", "mcp"],
    ),
    "code-reviewer": AgentCard(
        name="Code Reviewer",
        description="Automated code review with security, style, and correctness analysis",
        version="1.0.0",
        url="https://kroma.live/.well-known/agents/code-reviewer.json",
        skills=[
            AgentSkill(
                id="pr-review",
                name="PR Review",
                description="Review pull requests for issues and improvements",
            ),
            AgentSkill(
                id="security-audit",
                name="Security Audit",
                description="Identify security vulnerabilities in code",
            ),
        ],
        capabilities=["review", "security", "git"],
    ),
}


@dataclass
class A2AMessage:
    """A2A Protocol message."""
    id: str
    type: str  # request, response, notification
    method: str | None = None
    params: dict[str, Any] | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.method:
            d["method"] = self.method
        if self.params:
            d["params"] = self.params
        if self.result is not None:
            d["result"] = self.result
        if self.error:
            d["error"] = self.error
        return d


def _validate_url(url: str) -> bool:
    """Validate URL to prevent SSRF attacks."""
    from urllib.parse import urlparse
    import socket

    try:
        parsed = urlparse(url)

        # Only allow http/https schemes
        if parsed.scheme not in ("http", "https"):
            return False

        # Block localhost and private IP ranges
        hostname = parsed.hostname or ""
        if hostname in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False

        # Try to resolve hostname and check for private IPs
        try:
            ip = socket.gethostbyname(hostname)
            # Block private IP ranges (10.x, 172.16-31.x, 192.168.x)
            octets = [int(o) for o in ip.split(".")]
            if octets[0] == 10:
                return False
            if octets[0] == 172 and 16 <= octets[1] <= 31:
                return False
            if octets[0] == 192 and octets[1] == 168:
                return False
            if octets[0] == 169 and octets[1] == 254:  # Link-local / AWS metadata
                return False
        except socket.gaierror:
            pass  # Allow if DNS fails - will fail on connect anyway

        return True
    except Exception:
        return False


def discover_agent(url: str) -> AgentCard | None:
    """Discover an agent via its Agent Card URL."""
    try:
        # Normalize URL to /.well-known/agent.json
        if not url.endswith("agent.json"):
            url = url.rstrip("/") + "/.well-known/agent.json"

        # Validate URL to prevent SSRF
        if not _validate_url(url):
            print(f"[ERROR] URL validation failed (blocked): {url}", file=sys.stderr)
            return None

        req = Request(url, headers={"Accept": "application/json", "User-Agent": "Pluribus-A2A/1.0"})
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Parse into AgentCard
        skills = [
            AgentSkill(
                id=s.get("id", ""),
                name=s.get("name", ""),
                description=s.get("description", ""),
                input_schema=s.get("input_schema"),
                output_schema=s.get("output_schema"),
            )
            for s in data.get("skills", [])
        ]

        card = AgentCard(
            name=data.get("name", "Unknown"),
            description=data.get("description", ""),
            version=data.get("version", "0.0.0"),
            url=url,
            skills=skills,
            authentication=data.get("authentication"),
            capabilities=data.get("capabilities", []),
            metadata=data.get("metadata", {}),
        )

        # Emit discovery event
        emit_bus_event("a2a.agent.discovered", {
            "name": card.name,
            "url": url,
            "skills": [s.id for s in card.skills],
            "capabilities": card.capabilities,
        })

        return card

    except URLError as e:
        print(f"[ERROR] Cannot reach {url}: {e}", file=sys.stderr)
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON from {url}: {e}", file=sys.stderr)
        return None


def send_a2a_message(url: str, method: str, params: dict[str, Any] | None = None) -> A2AMessage:
    """Send an A2A message to a remote agent."""
    # Validate URL to prevent SSRF
    if not _validate_url(url):
        return A2AMessage(
            id=str(uuid.uuid4()),
            type="response",
            error={"code": -32600, "message": f"URL validation failed (blocked): {url}"},
        )

    msg = A2AMessage(
        id=str(uuid.uuid4()),
        type="request",
        method=method,
        params=params,
    )

    # Emit send event
    emit_bus_event("a2a.message.send", {
        "target": url,
        "method": method,
        "message_id": msg.id,
    })

    try:
        req = Request(
            url,
            data=json.dumps(msg.to_dict()).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Pluribus-A2A/1.0",
            },
            method="POST",
        )

        with urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))

        response = A2AMessage(
            id=response_data.get("id", msg.id),
            type="response",
            result=response_data.get("result"),
            error=response_data.get("error"),
        )

        # Emit receive event
        emit_bus_event("a2a.message.received", {
            "from": url,
            "message_id": response.id,
            "has_error": response.error is not None,
        })

        return response

    except Exception as e:
        return A2AMessage(
            id=msg.id,
            type="response",
            error={"code": -1, "message": str(e)},
        )


def a2a_to_bus_event(msg: A2AMessage, source: str) -> None:
    """Translate A2A message to Pluribus bus event."""
    if msg.type == "request" and msg.method:
        # Map A2A method to bus topic
        topic = f"a2a.request.{msg.method.replace('/', '.')}"
        emit_bus_event(topic, {
            "message_id": msg.id,
            "source": source,
            "params": msg.params,
        })
    elif msg.type == "notification":
        topic = f"a2a.notification.{msg.method.replace('/', '.')}" if msg.method else "a2a.notification"
        emit_bus_event(topic, {
            "message_id": msg.id,
            "source": source,
            "params": msg.params,
        })


class AgentCardHandler(BaseHTTPRequestHandler):
    """HTTP handler for serving Agent Cards."""

    agent_id: str = "pluribus-coordinator"

    def do_GET(self):
        if self.path == "/.well-known/agent.json" or self.path == "/agent.json":
            card = PLURIBUS_AGENTS.get(self.agent_id)
            if card:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(card.to_json().encode("utf-8"))
            else:
                self.send_error(404, "Agent not found")
        elif self.path.startswith("/.well-known/agents/"):
            # Serve specific agent cards
            agent_file = self.path.split("/")[-1].replace(".json", "")
            card = PLURIBUS_AGENTS.get(agent_file)
            if card:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(card.to_json().encode("utf-8"))
            else:
                self.send_error(404, f"Agent {agent_file} not found")
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        # Handle A2A messages
        try:
            content_length_str = self.headers.get("Content-Length", "0")
            content_length = int(content_length_str) if content_length_str else 0
            if content_length <= 0:
                self.send_error(411, "Content-Length required")
                return
            if content_length > 1024 * 1024:  # 1MB limit
                self.send_error(413, "Request too large")
                return
        except ValueError:
            self.send_error(400, "Invalid Content-Length header")
            return

        try:
            body = self.rfile.read(content_length).decode("utf-8")
        except UnicodeDecodeError:
            self.send_error(400, "Invalid UTF-8 encoding")
            return

        try:
            data = json.loads(body)
            msg = A2AMessage(
                id=data.get("id", str(uuid.uuid4())),
                type=data.get("type", "request"),
                method=data.get("method"),
                params=data.get("params"),
            )

            # Translate to bus event
            a2a_to_bus_event(msg, self.client_address[0])

            # Send acknowledgment
            method_str = msg.method if msg.method else "unknown"
            response = A2AMessage(
                id=msg.id,
                type="response",
                result={"status": "received", "bus_topic": f"a2a.request.{method_str}"},
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response.to_dict()).encode("utf-8"))

        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")

    def log_message(self, format, *args):
        # Suppress default logging
        pass


def serve_agent_card(agent_id: str, port: int = 9210) -> None:
    """Start HTTP server for Agent Card."""
    AgentCardHandler.agent_id = agent_id

    server = HTTPServer(("0.0.0.0", port), AgentCardHandler)
    print(f"Serving Agent Card for {agent_id} at http://0.0.0.0:{port}/.well-known/agent.json")

    emit_bus_event("a2a.server.started", {
        "agent_id": agent_id,
        "port": port,
    })

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluribus A2A Protocol Adapter")
    parser.add_argument("--serve", action="store_true", help="Serve Agent Card via HTTP")
    parser.add_argument("--agent", type=str, default="pluribus-coordinator", help="Agent ID to serve")
    parser.add_argument("--port", type=int, default=9210, help="HTTP port for serving")
    parser.add_argument("--discover", type=str, metavar="URL", help="Discover agent at URL")
    parser.add_argument("--send", nargs=2, metavar=("URL", "METHOD"), help="Send A2A message")
    parser.add_argument("--generate-card", type=str, metavar="AGENT_ID", help="Generate Agent Card JSON")
    parser.add_argument("--list-agents", action="store_true", help="List predefined Pluribus agents")

    args = parser.parse_args()

    if args.list_agents:
        print("Predefined Pluribus Agents:\n")
        for agent_id, card in PLURIBUS_AGENTS.items():
            print(f"  {agent_id}")
            print(f"    Name: {card.name}")
            print(f"    Skills: {', '.join(s.id for s in card.skills)}")
            print(f"    Capabilities: {', '.join(card.capabilities)}")
            print()
        return 0

    if args.generate_card:
        card = PLURIBUS_AGENTS.get(args.generate_card)
        if card:
            print(card.to_json())
            return 0
        else:
            print(f"Unknown agent: {args.generate_card}", file=sys.stderr)
            print(f"Available: {', '.join(PLURIBUS_AGENTS.keys())}", file=sys.stderr)
            return 1

    if args.discover:
        card = discover_agent(args.discover)
        if card:
            print(f"Discovered: {card.name}")
            print(f"Description: {card.description}")
            print(f"Skills: {', '.join(s.name for s in card.skills)}")
            print(f"Capabilities: {', '.join(card.capabilities)}")
            return 0
        return 1

    if args.send:
        url, method = args.send
        response = send_a2a_message(url, method)
        print(json.dumps(response.to_dict(), indent=2))
        return 0 if not response.error else 1

    if args.serve:
        serve_agent_card(args.agent, args.port)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
