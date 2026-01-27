#!/usr/bin/env python3
"""
hexis_buffer.py - HEXIS Epistemic Shim / Agent Inbox Buffer

"Hexis epistemic shim, equivariant polyfill..." - entelexis.md

The HEXIS buffer provides per-agent inbox/outbox for notifications
and acknowledgements in the portal inception flow.

Ring: 1 (Infrastructure)
Protocol: DKIN v28 | PAIP v15 | Citizen v1

Usage:
    python3 hexis_buffer.py publish <agent> <message> [--data JSON] [--ttl SECONDS]
    python3 hexis_buffer.py consume <agent> [--ack] [--limit N]
    python3 hexis_buffer.py status [<agent>]
    python3 hexis_buffer.py gc [--dry-run]

Bus Topics:
    hexis.buffer.published
    hexis.buffer.consumed
    hexis.buffer.expired
"""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional


# Configuration
HEXIS_DIR = Path(os.environ.get("PLURIBUS_HEXIS_DIR", "/tmp/hexis"))
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
DEFAULT_TTL_S = 3600  # 1 hour

# Stigmergic buffer path (old convention for /tmp/<agent>.buffer)
# Used by: hexis_status.py, hexis_entelechy_bridge.py, pbnotify_operator.py
STIGMERGIC_BUFFER_DIR = Path(os.environ.get("HEXIS_BUFFER_DIR", "/tmp"))


@dataclass
class HexisMessage:
    """A message in the HEXIS buffer."""
    id: str
    sender: str
    recipient: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    created_ts: float = field(default_factory=time.time)
    expires_ts: float = 0.0
    acked: bool = False
    acked_ts: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HexisMessage":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def is_expired(self) -> bool:
        return self.expires_ts > 0 and time.time() > self.expires_ts


# =============================================================================
# Stigmergic Buffer API (for pbnotify_operator, hexis_entelechy_bridge, etc.)
# =============================================================================

def buffer_path(agent: str) -> Path:
    """
    Return path to agent's stigmergic buffer file.

    Convention: /tmp/<agent>.buffer (ephemeral pheromone trail)
    Used by: pbnotify_operator.py, hexis_status.py, hexis_entelechy_bridge.py
    """
    return STIGMERGIC_BUFFER_DIR / f"{agent}.buffer"


@dataclass
class HexisNotifyMessage:
    """
    Notify-style message for stigmergic buffers (agent_notify_protocol_v1).

    This is the format expected by pbnotify_operator.py and documented
    in pluribus_lexicon.md §6.3.
    """
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ts: float = field(default_factory=time.time)
    iso: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    actor: str = ""
    agent_type: str = "worker"
    req_id: str = ""
    trace_id: str = ""
    topic: str = "agent.notify"
    kind: str = "request"
    effects: str = "none"
    lane: str = "strp"
    topology: str = "single"
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def emit_bus_event(
    topic: str,
    kind: str,
    level: str,
    actor: str,
    data: Dict[str, Any],
) -> None:
    """
    Emit event to Pluribus bus (module-level convenience function).

    Used by pbnotify_operator.py for hexis.buffer.published events.
    """
    events_path = BUS_DIR / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }

    with open(events_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


class HexisBuffer:
    """
    Per-agent inbox/outbox buffer for HEXIS notifications.
    
    Features:
    - File-based persistence per agent
    - TTL expiration
    - Acknowledgement tracking
    - Bus event integration
    """

    def __init__(self, hexis_dir: Path = None, bus_dir: Path = None):
        self.hexis_dir = hexis_dir or HEXIS_DIR
        self.bus_dir = bus_dir or BUS_DIR
        self.hexis_dir.mkdir(parents=True, exist_ok=True)

    def _inbox_path(self, agent: str) -> Path:
        """Get the inbox file path for an agent."""
        return self.hexis_dir / agent / "inbox.ndjson"

    def _outbox_path(self, agent: str) -> Path:
        """Get the outbox file path for an agent."""
        return self.hexis_dir / agent / "outbox.ndjson"

    def _load_inbox(self, agent: str) -> List[HexisMessage]:
        """Load all messages from an agent's inbox."""
        inbox_path = self._inbox_path(agent)
        if not inbox_path.exists():
            return []
        
        messages = []
        for line in inbox_path.read_text().strip().split("\n"):
            if line:
                try:
                    msg = HexisMessage.from_dict(json.loads(line))
                    messages.append(msg)
                except (json.JSONDecodeError, TypeError):
                    continue
        return messages

    def _save_inbox(self, agent: str, messages: List[HexisMessage]):
        """Save messages to an agent's inbox."""
        inbox_path = self._inbox_path(agent)
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(inbox_path, "w") as f:
            for msg in messages:
                f.write(json.dumps(msg.to_dict()) + "\n")

    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info"):
        """Emit event to the Pluribus bus."""
        bus_path = self.bus_dir / "events.ndjson"
        bus_path.parent.mkdir(parents=True, exist_ok=True)
        
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "event",
            "level": level,
            "actor": "hexis_buffer",
            "data": data,
        }
        
        with open(bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def publish(
        self,
        sender: str,
        recipient: str,
        message: str,
        data: Dict[str, Any] = None,
        ttl_s: int = DEFAULT_TTL_S,
        emit_bus: bool = True,
    ) -> HexisMessage:
        """
        Publish a message to an agent's inbox.
        
        Args:
            sender: Sending agent ID
            recipient: Receiving agent ID
            message: Text message
            data: Optional JSON payload
            ttl_s: Time to live in seconds (0 = no expiry)
            emit_bus: Whether to emit bus event
            
        Returns:
            The created HexisMessage
        """
        msg = HexisMessage(
            id=uuid.uuid4().hex[:16],
            sender=sender,
            recipient=recipient,
            message=message,
            data=data or {},
            created_ts=time.time(),
            expires_ts=time.time() + ttl_s if ttl_s > 0 else 0.0,
        )
        
        # Append to recipient's inbox
        inbox_path = self._inbox_path(recipient)
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(inbox_path, "a") as f:
            f.write(json.dumps(msg.to_dict()) + "\n")
        
        # Also record in sender's outbox
        outbox_path = self._outbox_path(sender)
        outbox_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(outbox_path, "a") as f:
            f.write(json.dumps(msg.to_dict()) + "\n")
        
        if emit_bus:
            self._emit_bus_event("hexis.buffer.published", {
                "msg_id": msg.id,
                "sender": sender,
                "recipient": recipient,
                "message": message[:100],
                "ttl_s": ttl_s,
            })
        
        return msg

    def consume(
        self,
        agent: str,
        ack: bool = False,
        limit: int = 10,
        emit_bus: bool = True,
    ) -> List[HexisMessage]:
        """
        Consume messages from an agent's inbox.
        
        Args:
            agent: Agent ID
            ack: Whether to mark messages as acknowledged
            limit: Maximum number of messages to return
            emit_bus: Whether to emit bus event
            
        Returns:
            List of HexisMessage objects
        """
        messages = self._load_inbox(agent)
        
        # Filter out expired and optionally already-acked
        valid = [m for m in messages if not m.is_expired()]
        unacked = [m for m in valid if not m.acked]
        
        # Take up to limit
        result = unacked[:limit]
        
        if ack and result:
            # Mark as acknowledged
            ack_ts = time.time()
            for msg in result:
                msg.acked = True
                msg.acked_ts = ack_ts
            
            # Update inbox
            self._save_inbox(agent, valid)
            
            if emit_bus:
                self._emit_bus_event("hexis.buffer.consumed", {
                    "agent": agent,
                    "msg_ids": [m.id for m in result],
                    "count": len(result),
                })
        
        return result

    def status(self, agent: str = None) -> Dict[str, Any]:
        """
        Get buffer status for one or all agents.
        
        Args:
            agent: Optional agent ID (None for all agents)
            
        Returns:
            Status dictionary
        """
        if agent:
            messages = self._load_inbox(agent)
            valid = [m for m in messages if not m.is_expired()]
            return {
                "agent": agent,
                "total": len(messages),
                "valid": len(valid),
                "unacked": len([m for m in valid if not m.acked]),
                "expired": len(messages) - len(valid),
            }
        
        # All agents
        result = {"agents": {}}
        for agent_dir in self.hexis_dir.iterdir():
            if agent_dir.is_dir():
                agent_name = agent_dir.name
                messages = self._load_inbox(agent_name)
                valid = [m for m in messages if not m.is_expired()]
                result["agents"][agent_name] = {
                    "total": len(messages),
                    "unacked": len([m for m in valid if not m.acked]),
                }
        
        result["agent_count"] = len(result["agents"])
        return result

    def gc(self, dry_run: bool = False) -> Dict[str, int]:
        """
        Garbage collect expired messages.
        
        Args:
            dry_run: If True, don't actually delete
            
        Returns:
            Dict with counts of removed messages per agent
        """
        removed = {}
        
        for agent_dir in self.hexis_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            
            agent = agent_dir.name
            messages = self._load_inbox(agent)
            valid = [m for m in messages if not m.is_expired()]
            expired_count = len(messages) - len(valid)
            
            if expired_count > 0:
                removed[agent] = expired_count
                if not dry_run:
                    self._save_inbox(agent, valid)
                    self._emit_bus_event("hexis.buffer.expired", {
                        "agent": agent,
                        "expired_count": expired_count,
                    })
        
        return removed


def cmd_publish(args):
    """Publish a message to an agent."""
    buffer = HexisBuffer()
    
    data = {}
    if args.data:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON data", file=sys.stderr)
            return 1
    
    msg = buffer.publish(
        sender=os.environ.get("PLURIBUS_ACTOR", "operator"),
        recipient=args.agent,
        message=args.message,
        data=data,
        ttl_s=args.ttl or DEFAULT_TTL_S,
    )
    
    print(f"Published: {msg.id}")
    print(f"To: {args.agent}")
    print(f"Expires: {datetime.fromtimestamp(msg.expires_ts).isoformat() if msg.expires_ts else 'never'}")
    return 0


def cmd_consume(args):
    """Consume messages from an agent's inbox."""
    buffer = HexisBuffer()
    
    messages = buffer.consume(
        agent=args.agent,
        ack=args.ack,
        limit=args.limit or 10,
    )
    
    if not messages:
        print("No unread messages")
        return 0
    
    for msg in messages:
        ts = datetime.fromtimestamp(msg.created_ts).strftime("%H:%M:%S")
        print(f"[{ts}] {msg.sender}: {msg.message}")
        if msg.data:
            print(f"    Data: {json.dumps(msg.data)[:100]}")
    
    print(f"\n{len(messages)} message(s){' (acknowledged)' if args.ack else ''}")
    return 0


def cmd_status(args):
    """Show buffer status."""
    buffer = HexisBuffer()
    status = buffer.status(args.agent)
    
    if args.agent:
        print(f"Agent: {status['agent']}")
        print(f"  Total:   {status['total']}")
        print(f"  Unacked: {status['unacked']}")
        print(f"  Expired: {status['expired']}")
    else:
        print(f"HEXIS Buffer Status")
        print(f"  Agents: {status['agent_count']}")
        for agent, data in status["agents"].items():
            print(f"  {agent}: {data['unacked']} unacked / {data['total']} total")
    
    return 0


def cmd_gc(args):
    """Garbage collect expired messages."""
    buffer = HexisBuffer()
    removed = buffer.gc(dry_run=args.dry_run)
    
    if not removed:
        print("No expired messages found")
        return 0
    
    total = sum(removed.values())
    prefix = "[Dry run] Would remove" if args.dry_run else "Removed"
    print(f"{prefix} {total} expired messages:")
    for agent, count in removed.items():
        print(f"  {agent}: {count}")
    
    return 0


def main():
    parser = argparse.ArgumentParser(description="HEXIS Buffer - Agent Inbox System")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # publish
    p_pub = subparsers.add_parser("publish", help="Publish a message")
    p_pub.add_argument("agent", help="Recipient agent ID")
    p_pub.add_argument("message", help="Message text")
    p_pub.add_argument("--data", help="JSON payload")
    p_pub.add_argument("--ttl", type=int, help="TTL in seconds")
    
    # consume
    p_con = subparsers.add_parser("consume", help="Consume messages")
    p_con.add_argument("agent", help="Agent ID")
    p_con.add_argument("--ack", action="store_true", help="Acknowledge messages")
    p_con.add_argument("--limit", type=int, help="Max messages")
    
    # status
    p_status = subparsers.add_parser("status", help="Show status")
    p_status.add_argument("agent", nargs="?", help="Agent ID (optional)")
    
    # gc
    p_gc = subparsers.add_parser("gc", help="Garbage collect")
    p_gc.add_argument("--dry-run", action="store_true", help="Dry run")
    
    args = parser.parse_args()
    
    if args.command == "publish":
        return cmd_publish(args)
    elif args.command == "consume":
        return cmd_consume(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "gc":
        return cmd_gc(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
