#!/usr/bin/env python3
"""
Cut-Out Intermediary - Intelligence cell structure isolation for agent communication.

Historical Context:
-------------------
This implementation draws from verified intelligence tradecraft:

1. CUT-OUTS IN ESPIONAGE:
   A cut-out is a mutually trusted intermediary facilitating information exchange
   between agents. The cut-out knows only the source and destination, remaining
   unaware of other network members' identities.

   Two forms exist:
   - BLOCK cut-out: Aware of the entire spy network/cell
   - CHAIN cut-out: Knows only the provider and receiver

2. CELL STRUCTURE:
   Soviet operations organized networks as cell systems where each operator knows:
   - People in their own cell
   - Their external case officer
   - An emergency contact method for higher levels

   This compartmentalization limits damage if any member is compromised.
   The 1943 Prosper network collapse (50+ arrests) demonstrated the catastrophic
   consequences of poor cut-out discipline.

3. HANDLER/CASE OFFICER MODEL:
   Case officers manage human agents and networks, spotting potential agents,
   recruiting, and training in tradecraft. They serve as the nexus point
   between isolated cells.

Pluribus Implementation:
-----------------------
- Agents communicate via the bus, not directly (chain cut-out pattern)
- Coordinators serve as block cut-outs with full visibility
- Task IDs are opaque - workers don't know the broader mission
- Context windows are minimal and task-specific
- State is purged after session end (capture resistance)

The goal: An agent compromise reveals only their specific task fragment,
not the overall system architecture or other agents' work.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

sys.dont_write_bytecode = True

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from nucleus.tools import agent_bus
except ImportError:
    agent_bus = None


# =============================================================================
# CUT-OUT TYPES
# =============================================================================

class CutOutType(Enum):
    """
    Types of cut-out intermediaries.

    Historical parallel:
    - CHAIN: Minimal knowledge, just source and destination
    - BLOCK: Full network awareness (coordinators only)
    - DEAD_DROP: Asynchronous exchange, no direct contact
    """
    CHAIN = "chain"      # Knows only source → destination
    BLOCK = "block"      # Knows full network topology
    DEAD_DROP = "drop"   # Asynchronous bus exchange


# =============================================================================
# MESSAGE ENVELOPE (Sanitized for Cut-Out)
# =============================================================================

@dataclass
class CutOutEnvelope:
    """
    Sanitized message envelope for cut-out transmission.

    The envelope contains:
    - Opaque task ID (worker doesn't know what it means)
    - Minimal context (only what's needed for the specific task)
    - No reference to broader mission or other workers
    - Cryptographic binding to prevent tampering

    Historical parallel:
    - Dead drop containers with minimal identifying marks
    - One-time pad encrypted messages
    - Coded signals with no plaintext context
    """
    envelope_id: str
    task_id: str  # Opaque - worker doesn't know the meaning
    source_alias: str  # Pseudonym, not real agent ID
    destination_alias: str  # Pseudonym, not real agent ID
    payload_hash: str  # Integrity check
    context_window: dict  # Minimal, task-specific context
    classification: str  # "RING2" style marking
    ttl_s: int = 3600  # Time to live before purge
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "envelope_id": self.envelope_id,
            "task_id": self.task_id,
            "source_alias": self.source_alias,
            "destination_alias": self.destination_alias,
            "payload_hash": self.payload_hash,
            "context_window": self.context_window,
            "classification": self.classification,
            "ttl_s": self.ttl_s,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CutOutEnvelope":
        return cls(**data)

    def is_expired(self) -> bool:
        return time.time() > self.created_at + self.ttl_s


# =============================================================================
# ALIAS REGISTRY (Pseudonymization)
# =============================================================================

class AliasRegistry:
    """
    Maps real agent IDs to pseudonyms for compartmentalized communication.

    Historical parallel:
    - Code names used in intelligence operations (e.g., "VENONA")
    - Agent pseudonyms that hide true identities
    - Operational security (OPSEC) practices

    The registry is held only by block cut-outs (coordinators).
    Chain cut-outs see only aliases, never real IDs.
    """

    def __init__(self):
        self._real_to_alias: dict[str, str] = {}
        self._alias_to_real: dict[str, str] = {}

    def get_or_create_alias(self, real_id: str) -> str:
        """Get or create a pseudonym for an agent."""
        if real_id not in self._real_to_alias:
            # Generate pseudonym: hash + short UUID
            hash_prefix = hashlib.sha256(real_id.encode()).hexdigest()[:4]
            alias = f"AGENT-{hash_prefix.upper()}-{uuid.uuid4().hex[:4].upper()}"
            self._real_to_alias[real_id] = alias
            self._alias_to_real[alias] = real_id
        return self._real_to_alias[real_id]

    def resolve_alias(self, alias: str) -> Optional[str]:
        """Resolve pseudonym to real ID (block cut-outs only)."""
        return self._alias_to_real.get(alias)

    def clear_aliases(self, real_id: str):
        """Clear all aliases for an agent (on session end)."""
        if real_id in self._real_to_alias:
            alias = self._real_to_alias.pop(real_id)
            self._alias_to_real.pop(alias, None)


# =============================================================================
# CONTEXT SANITIZER
# =============================================================================

class ContextSanitizer:
    """
    Strips sensitive information from context before passing to workers.

    Historical parallel:
    - Manhattan Project workers knowing HOW but not WHY
    - Redacted documents with need-to-know portions removed
    - "Sanitized" intelligence products for lower clearances

    Implementation:
    - Removes references to overall mission
    - Strips agent IDs and replaces with aliases
    - Removes file paths outside worker's scope
    - Limits context to immediate task requirements
    """

    # Keys that must be stripped from context
    SENSITIVE_KEYS = {
        "full_mission",
        "other_agents",
        "coordinator_id",
        "network_topology",
        "ring0_access",
        "pqc_keys",
        "session_history",
        "global_state",
    }

    # Path patterns that must be redacted
    SENSITIVE_PATHS = [
        ".pluribus/secrets/",
        "nucleus/specs/pqc_",
        "GENOME.json",
    ]

    @classmethod
    def sanitize(cls, context: dict, worker_ring: int) -> dict:
        """
        Sanitize context for a worker at the given ring level.

        Higher ring numbers = less access = more sanitization.
        """
        sanitized = {}

        for key, value in context.items():
            # Strip explicitly sensitive keys
            if key in cls.SENSITIVE_KEYS:
                continue

            # Recursively sanitize nested dicts
            if isinstance(value, dict):
                sanitized[key] = cls.sanitize(value, worker_ring)
            elif isinstance(value, str):
                # Redact sensitive paths
                is_sensitive = any(p in value for p in cls.SENSITIVE_PATHS)
                if is_sensitive and worker_ring > 0:
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = value
            elif isinstance(value, list):
                # Filter lists for sensitive items
                sanitized[key] = [
                    cls.sanitize(item, worker_ring) if isinstance(item, dict)
                    else item
                    for item in value
                    if not (isinstance(item, str) and any(p in item for p in cls.SENSITIVE_PATHS))
                ]
            else:
                sanitized[key] = value

        return sanitized

    @classmethod
    def create_minimal_context(
        cls,
        full_context: dict,
        task_description: str,
        worker_ring: int,
    ) -> dict:
        """
        Create minimal context window for a specific task.

        Historical parallel:
        - Oak Ridge worker briefing: "Operate this machine following these steps"
        - No mention of uranium, bombs, or the overall project
        """
        return {
            "task": task_description,
            "scope": cls.sanitize(full_context.get("scope", {}), worker_ring),
            "constraints": full_context.get("constraints", {}),
            "outputs_expected": full_context.get("outputs_expected", []),
            # Explicitly no: mission, other_workers, coordinator, history
        }


# =============================================================================
# CUT-OUT MANAGER
# =============================================================================

class CutOutManager:
    """
    Manages cut-out mediated communication between agents.

    This is the central handler that:
    1. Sanitizes all inter-agent messages
    2. Replaces real IDs with aliases
    3. Routes via bus (dead drop pattern)
    4. Maintains minimal state (purged after TTL)

    Historical parallel:
    - Case officer managing a network of agents
    - Each agent knows only the case officer, not other agents
    - If captured, can only reveal limited information
    """

    def __init__(self, bus_dir: Optional[str] = None):
        self.bus_dir = bus_dir or str(REPO_ROOT / ".pluribus/bus")
        self.alias_registry = AliasRegistry()
        self.pending_envelopes: dict[str, CutOutEnvelope] = {}
        self.sanitizer = ContextSanitizer()

        # Cut-out type for this instance
        self.cut_out_type = CutOutType.CHAIN  # Default to minimal knowledge

    def _emit_bus(self, topic: str, level: str, data: dict):
        """Emit event to bus (dead drop)."""
        if agent_bus is None:
            return
        try:
            paths = agent_bus.resolve_bus_paths(self.bus_dir)
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind="request",
                level=level,
                actor="cut_out",
                data=data,
                trace_id=None,
                run_id=None,
                durable=True,
            )
        except Exception:
            pass

    def create_envelope(
        self,
        source_id: str,
        destination_id: str,
        task_id: str,
        context: dict,
        destination_ring: int,
        classification: str = "RING2",
    ) -> CutOutEnvelope:
        """
        Create a sanitized envelope for cut-out transmission.

        The source and destination are pseudonymized.
        The context is sanitized based on destination ring.
        The task_id remains opaque to the worker.
        """
        # Pseudonymize identities
        source_alias = self.alias_registry.get_or_create_alias(source_id)
        dest_alias = self.alias_registry.get_or_create_alias(destination_id)

        # Sanitize context for destination's ring level
        minimal_context = self.sanitizer.create_minimal_context(
            context,
            context.get("task_description", "Execute assigned task"),
            destination_ring,
        )

        # Create integrity hash
        payload_str = json.dumps(minimal_context, sort_keys=True)
        payload_hash = hashlib.sha256(payload_str.encode()).hexdigest()[:16]

        envelope = CutOutEnvelope(
            envelope_id=str(uuid.uuid4()),
            task_id=task_id,  # Opaque to worker
            source_alias=source_alias,
            destination_alias=dest_alias,
            payload_hash=payload_hash,
            context_window=minimal_context,
            classification=classification,
        )

        # Store for tracking
        self.pending_envelopes[envelope.envelope_id] = envelope

        return envelope

    def dispatch_via_dead_drop(
        self,
        envelope: CutOutEnvelope,
    ) -> str:
        """
        Dispatch envelope via bus (dead drop pattern).

        Historical parallel:
        - Agent leaves package at predetermined location
        - Handler retrieves it later
        - No direct contact between parties
        """
        self._emit_bus("cutout.dispatch.request", "info", {
            "envelope_id": envelope.envelope_id,
            "task_id": envelope.task_id,
            "destination_alias": envelope.destination_alias,
            "classification": envelope.classification,
            "payload_hash": envelope.payload_hash,
            "context": envelope.context_window,
        })

        return envelope.envelope_id

    def receive_result(
        self,
        envelope_id: str,
        result: dict,
        worker_alias: str,
    ) -> dict:
        """
        Receive result from worker via cut-out.

        The result is verified against the original envelope
        and the worker alias is checked for validity.
        """
        envelope = self.pending_envelopes.get(envelope_id)
        if envelope is None:
            return {"error": "Unknown envelope ID"}

        if worker_alias != envelope.destination_alias:
            return {"error": "Worker alias mismatch"}

        if envelope.is_expired():
            self._purge_envelope(envelope_id)
            return {"error": "Envelope expired"}

        # Log result receipt
        self._emit_bus("cutout.result.received", "info", {
            "envelope_id": envelope_id,
            "task_id": envelope.task_id,
            "worker_alias": worker_alias,
            "result_hash": hashlib.sha256(json.dumps(result).encode()).hexdigest()[:16],
        })

        # Purge envelope after processing
        self._purge_envelope(envelope_id)

        return {"status": "received", "envelope_id": envelope_id}

    def _purge_envelope(self, envelope_id: str):
        """
        Purge envelope and associated aliases.

        Historical parallel:
        - Destruction of one-time pads after use
        - Clearing dead drop after retrieval
        - State purge for capture resistance
        """
        envelope = self.pending_envelopes.pop(envelope_id, None)
        if envelope:
            # Clear aliases to prevent tracking
            # (In chain cut-out mode, we don't know real IDs)
            pass

    def purge_expired(self) -> int:
        """Purge all expired envelopes (capture resistance)."""
        expired = [
            eid for eid, env in self.pending_envelopes.items()
            if env.is_expired()
        ]
        for eid in expired:
            self._purge_envelope(eid)
        return len(expired)


# =============================================================================
# CELL STRUCTURE
# =============================================================================

@dataclass
class Cell:
    """
    Intelligence cell structure.

    Historical parallel:
    - Soviet espionage cells with 3-5 members
    - Each member knows only their handler and emergency contact
    - Cell leader is the only one with broader network awareness
    """
    cell_id: str
    leader_id: str  # Block cut-out - knows network
    members: set[str] = field(default_factory=set)  # Chain cut-outs - minimal knowledge
    parent_cell: Optional[str] = None  # For hierarchical networks
    classification: str = "RING2"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "leader_id": self.leader_id,
            "members": list(self.members),
            "parent_cell": self.parent_cell,
            "classification": self.classification,
            "created_at": self.created_at,
        }


class CellNetwork:
    """
    Manages the overall cell structure.

    Historical parallel:
    - British SOE resistance networks
    - Soviet intelligence networks in the West
    - Hierarchical cell structures with limited cross-cell visibility

    Key principle: Compromise of one cell should not expose others.
    """

    def __init__(self):
        self.cells: dict[str, Cell] = {}
        self.agent_to_cell: dict[str, str] = {}

    def create_cell(
        self,
        leader_id: str,
        classification: str = "RING2",
        parent_cell: Optional[str] = None,
    ) -> Cell:
        """Create a new cell with a designated leader."""
        cell_id = f"CELL-{uuid.uuid4().hex[:8].upper()}"
        cell = Cell(
            cell_id=cell_id,
            leader_id=leader_id,
            classification=classification,
            parent_cell=parent_cell,
        )
        self.cells[cell_id] = cell
        self.agent_to_cell[leader_id] = cell_id
        return cell

    def add_member(self, cell_id: str, agent_id: str) -> bool:
        """Add a member to a cell."""
        if cell_id not in self.cells:
            return False
        self.cells[cell_id].members.add(agent_id)
        self.agent_to_cell[agent_id] = cell_id
        return True

    def get_visible_agents(self, agent_id: str) -> set[str]:
        """
        Get agents visible to a given agent.

        Chain cut-outs see: only cell leader
        Cell leaders see: all cell members + parent leader
        """
        cell_id = self.agent_to_cell.get(agent_id)
        if cell_id is None:
            return set()

        cell = self.cells[cell_id]

        if agent_id == cell.leader_id:
            # Leader sees members and parent leader
            visible = cell.members.copy()
            if cell.parent_cell and cell.parent_cell in self.cells:
                visible.add(self.cells[cell.parent_cell].leader_id)
            return visible
        else:
            # Member sees only leader
            return {cell.leader_id}

    def simulate_compromise(self, agent_id: str) -> dict:
        """
        Simulate what information would be exposed if agent is compromised.

        Historical parallel:
        - Damage assessment after agent capture
        - Estimating network exposure from interrogation
        """
        visible = self.get_visible_agents(agent_id)
        cell_id = self.agent_to_cell.get(agent_id)
        cell = self.cells.get(cell_id) if cell_id else None

        return {
            "compromised_agent": agent_id,
            "visible_agents": list(visible),
            "cell_exposed": cell_id if agent_id == cell.leader_id else None,
            "network_exposed": False,  # Only if Ring 0 coordinator
            "damage_level": "CRITICAL" if len(visible) > 5 else "MODERATE" if len(visible) > 2 else "MINIMAL",
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Cut-Out Intermediary - Cell structure isolation")
    subparsers = parser.add_subparsers(dest="command")

    # Create envelope
    env_parser = subparsers.add_parser("envelope", help="Create sanitized envelope")
    env_parser.add_argument("--source", required=True)
    env_parser.add_argument("--dest", required=True)
    env_parser.add_argument("--task-id", required=True)
    env_parser.add_argument("--ring", type=int, default=2)
    env_parser.add_argument("--context", default="{}")

    # Dispatch
    dispatch_parser = subparsers.add_parser("dispatch", help="Dispatch envelope via dead drop")
    dispatch_parser.add_argument("envelope_id")

    # Create cell
    cell_parser = subparsers.add_parser("cell", help="Create intelligence cell")
    cell_parser.add_argument("--leader", required=True)
    cell_parser.add_argument("--classification", default="RING2")

    # Add member
    member_parser = subparsers.add_parser("member", help="Add member to cell")
    member_parser.add_argument("cell_id")
    member_parser.add_argument("agent_id")

    # Simulate compromise
    compromise_parser = subparsers.add_parser("compromise", help="Simulate agent compromise")
    compromise_parser.add_argument("agent_id")

    args = parser.parse_args()

    if args.command == "envelope":
        manager = CutOutManager()
        context = json.loads(args.context)
        envelope = manager.create_envelope(
            args.source,
            args.dest,
            args.task_id,
            context,
            args.ring,
        )
        print(json.dumps(envelope.to_dict(), indent=2))

    elif args.command == "cell":
        network = CellNetwork()
        cell = network.create_cell(args.leader, args.classification)
        print(json.dumps(cell.to_dict(), indent=2))

    elif args.command == "compromise":
        # Demo with sample network
        network = CellNetwork()
        cell = network.create_cell("coordinator", "RING1")
        network.add_member(cell.cell_id, "worker-1")
        network.add_member(cell.cell_id, "worker-2")
        network.add_member(cell.cell_id, "worker-3")

        result = network.simulate_compromise(args.agent_id)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
