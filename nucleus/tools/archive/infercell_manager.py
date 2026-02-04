#!/usr/bin/env python3
"""
InferCell Fork/Merge Manager
=============================

Implements the InferCell trace_id fork/merge lifecycle per comprehensive_implementation_matrix.md:

InferCell := {
  cell_id: uuid,
  trace_id: uuid,          # Primary trace correlation
  parent_trace_id: uuid?,  # Forked from (None for genesis)
  fork_point: {            # Where divergence occurred
    event_id: uuid,
    timestamp: iso8601,
    reason: string
  },
  state: pending | active | paused | complete | merged,
  children: [trace_id...], # Forked children
  merge_strategy: first_wins | consensus | manual
}

Usage:
  python3 infercell_manager.py create --reason "genesis"
  python3 infercell_manager.py fork --parent <trace_id> --reason "user-fork"
  python3 infercell_manager.py merge --sources <trace_id1>,<trace_id2> --strategy consensus
  python3 infercell_manager.py list
  python3 infercell_manager.py get --trace-id <trace_id>
  python3 infercell_manager.py state --trace-id <trace_id> --set active
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal

sys.dont_write_bytecode = True


@dataclass
class ForkPoint:
    """Records where a context fork occurred."""
    event_id: str
    timestamp: str
    reason: str


@dataclass
class InferCell:
    """
    A living context cell with trace correlation and fork/merge semantics.

    Each InferCell represents a branch of reality—a context that can diverge,
    evolve independently, and potentially merge back.
    """
    cell_id: str
    trace_id: str
    parent_trace_id: Optional[str]
    fork_point: ForkPoint
    state: Literal["pending", "active", "paused", "complete", "merged"]
    children: list[str] = field(default_factory=list)
    merge_strategy: Literal["first_wins", "consensus", "manual"] = "first_wins"
    metadata: dict = field(default_factory=dict)
    working_memory: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fork_point"] = asdict(self.fork_point)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InferCell":
        fp = d.pop("fork_point", {})
        fork_point = ForkPoint(
            event_id=fp.get("event_id", ""),
            timestamp=fp.get("timestamp", ""),
            reason=fp.get("reason", "")
        )
        return cls(fork_point=fork_point, **d)


class InferCellManager:
    """
    Manages InferCell lifecycle with fork/merge semantics.

    Storage: .pluribus/infercells/
      - cells.ndjson: All cells (append-only)
      - index.json: trace_id → cell_id mapping
    """

    def __init__(self, pluribus_root: Optional[Path] = None):
        self.root = pluribus_root or Path("/pluribus")
        self.storage_dir = self.root / ".pluribus" / "infercells"
        self.cells_file = self.storage_dir / "cells.ndjson"
        self.index_file = self.storage_dir / "index.json"
        self.bus_dir = os.environ.get("PLURIBUS_BUS_DIR") or str(self.root / ".pluribus" / "bus")

        # In-memory state
        self.cells: dict[str, InferCell] = {}  # cell_id → InferCell
        self.trace_index: dict[str, str] = {}  # trace_id → cell_id

        self._ensure_storage()
        self._load_state()

    def _ensure_storage(self):
        """Ensure storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_state(self):
        """Load cells from storage."""
        if self.cells_file.exists():
            with open(self.cells_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        cell = InferCell.from_dict(d)
                        self.cells[cell.cell_id] = cell
                        self.trace_index[cell.trace_id] = cell.cell_id
                    except Exception:
                        continue

        # Load index if exists (for faster lookups)
        if self.index_file.exists():
            try:
                self.trace_index = json.loads(self.index_file.read_text())
            except Exception:
                pass

    def _save_cell(self, cell: InferCell):
        """Append cell to storage and update index."""
        with open(self.cells_file, "a") as f:
            f.write(json.dumps(cell.to_dict()) + "\n")

        # Update index
        self.trace_index[cell.trace_id] = cell.cell_id
        self.index_file.write_text(json.dumps(self.trace_index, indent=2))

    def _emit_bus_event(self, topic: str, data: dict):
        """Emit event to the bus."""
        if not self.bus_dir:
            return
        tool = self.root / "nucleus" / "tools" / "agent_bus.py"
        if not tool.exists():
            return
        subprocess.run(
            [
                sys.executable,
                str(tool),
                "--bus-dir",
                self.bus_dir,
                "pub",
                "--topic",
                topic,
                "--kind",
                "artifact",
                "--level",
                "info",
                "--actor",
                os.environ.get("PLURIBUS_ACTOR", "infercell_manager"),
                "--data",
                json.dumps(data, ensure_ascii=False),
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _current_event_id(self) -> str:
        """Generate a unique event ID."""
        return str(uuid.uuid4())

    def create_genesis(self, reason: str = "genesis", trace_id: Optional[str] = None) -> InferCell:
        """Create a genesis cell (no parent)."""
        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        cell = InferCell(
            cell_id=str(uuid.uuid4()),
            trace_id=trace_id or str(uuid.uuid4()),
            parent_trace_id=None,
            fork_point=ForkPoint(
                event_id=self._current_event_id(),
                timestamp=now,
                reason=reason
            ),
            state="active",
            children=[],
            merge_strategy="first_wins"
        )

        self.cells[cell.cell_id] = cell
        self.trace_index[cell.trace_id] = cell.cell_id
        self._save_cell(cell)

        self._emit_bus_event("infercell.genesis", {
            "cell_id": cell.cell_id,
            "trace_id": cell.trace_id,
            "reason": reason
        })

        return cell

    def ensure_cell_for_trace(self, trace_id: str, parent_trace_id: Optional[str] = None, reason: str = "auto") -> InferCell:
        """Get existing cell for trace_id, or create a new one (fork/genesis) if missing."""
        cell = self.get_cell(trace_id)
        if cell:
            return cell
        
        if parent_trace_id and self.get_cell(parent_trace_id):
            return self.fork(parent_trace_id, reason=reason, child_trace_id=trace_id)
        
        return self.create_genesis(reason=reason, trace_id=trace_id)

    def fork(self, parent_trace_id: str, reason: str, child_trace_id: Optional[str] = None) -> InferCell:
        """Fork a new context from parent."""
        parent_cell_id = self.trace_index.get(parent_trace_id)
        if not parent_cell_id:
            raise ValueError(f"Parent trace_id not found: {parent_trace_id}")

        parent = self.cells.get(parent_cell_id)
        if not parent:
            raise ValueError(f"Parent cell not found: {parent_cell_id}")

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        child = InferCell(
            cell_id=str(uuid.uuid4()),
            trace_id=child_trace_id or str(uuid.uuid4()),
            parent_trace_id=parent.trace_id,
            fork_point=ForkPoint(
                event_id=self._current_event_id(),
                timestamp=now,
                reason=reason
            ),
            state="active",
            children=[],
            merge_strategy="first_wins",
            working_memory=parent.working_memory.copy()  # Inherit working memory
        )

        # Update parent's children
        parent.children.append(child.trace_id)

        # Register
        self.cells[child.cell_id] = child
        self.trace_index[child.trace_id] = child.cell_id
        self._save_cell(child)

        # Emit fork event
        self._emit_bus_event("infercell.fork", {
            "parent_trace_id": parent.trace_id,
            "child_trace_id": child.trace_id,
            "child_cell_id": child.cell_id,
            "reason": reason
        })

        return child

    def merge(
        self,
        source_trace_ids: list[str],
        strategy: Literal["first_wins", "consensus", "manual"] = "consensus"
    ) -> InferCell:
        """Merge multiple traces into one."""
        if len(source_trace_ids) < 2:
            raise ValueError("Need at least 2 traces to merge")

        sources = []
        for tid in source_trace_ids:
            cell_id = self.trace_index.get(tid)
            if not cell_id:
                raise ValueError(f"Trace not found: {tid}")
            cell = self.cells.get(cell_id)
            if not cell:
                raise ValueError(f"Cell not found: {cell_id}")
            sources.append(cell)

        # Validate merge compatibility (all must be active or complete)
        for source in sources:
            if source.state not in ("active", "complete", "paused"):
                raise ValueError(f"Cannot merge cell in state: {source.state}")

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Create merged cell
        merged = InferCell(
            cell_id=str(uuid.uuid4()),
            trace_id=str(uuid.uuid4()),
            parent_trace_id=None,  # Multiple parents - stored in metadata
            fork_point=ForkPoint(
                event_id=self._current_event_id(),
                timestamp=now,
                reason=f"merge({strategy})"
            ),
            state="active",
            children=[],
            merge_strategy=strategy,
            metadata={"parent_traces": source_trace_ids}
        )

        # Apply merge strategy
        if strategy == "first_wins":
            merged.working_memory = sources[0].working_memory.copy()
        elif strategy == "consensus":
            merged.working_memory = self._consensus_merge(sources)
        elif strategy == "manual":
            merged.state = "pending"
            merged.metadata["pending_merge_resolution"] = True
            # Collect all working memories for manual resolution
            merged.metadata["source_memories"] = {
                s.trace_id: s.working_memory for s in sources
            }

        # Mark sources as merged
        for source in sources:
            source.state = "merged"
            source.metadata["merged_into"] = merged.trace_id

        # Register
        self.cells[merged.cell_id] = merged
        self.trace_index[merged.trace_id] = merged.cell_id
        self._save_cell(merged)

        # Emit merge event
        self._emit_bus_event("infercell.merge", {
            "source_trace_ids": source_trace_ids,
            "merged_trace_id": merged.trace_id,
            "merged_cell_id": merged.cell_id,
            "strategy": strategy
        })

        return merged

    def _consensus_merge(self, sources: list[InferCell]) -> dict:
        """Merge working memories via consensus algorithm."""
        all_facts: dict[str, list[tuple[str, any]]] = {}

        for source in sources:
            for key, value in source.working_memory.items():
                if key not in all_facts:
                    all_facts[key] = []
                all_facts[key].append((source.trace_id, value))

        merged: dict = {}
        for key, values in all_facts.items():
            if len(values) == 1:
                merged[key] = values[0][1]
            elif all(v[1] == values[0][1] for v in values):
                # All agree
                merged[key] = values[0][1]
            else:
                # Conflict: use most recent (last in list)
                merged[key] = values[-1][1]
                merged[f"_conflict_{key}"] = values

        return merged

    def set_state(self, trace_id: str, state: str):
        """Update cell state."""
        cell_id = self.trace_index.get(trace_id)
        if not cell_id:
            raise ValueError(f"Trace not found: {trace_id}")

        cell = self.cells.get(cell_id)
        if not cell:
            raise ValueError(f"Cell not found: {cell_id}")

        old_state = cell.state
        cell.state = state
        self._save_cell(cell)

        self._emit_bus_event("infercell.state.changed", {
            "trace_id": trace_id,
            "cell_id": cell_id,
            "old_state": old_state,
            "new_state": state
        })

    def get_cell(self, trace_id: str) -> Optional[InferCell]:
        """Get cell by trace_id."""
        cell_id = self.trace_index.get(trace_id)
        if not cell_id:
            return None
        return self.cells.get(cell_id)

    def list_cells(self, state_filter: Optional[str] = None) -> list[InferCell]:
        """List all cells, optionally filtered by state."""
        cells = list(self.cells.values())
        if state_filter:
            cells = [c for c in cells if c.state == state_filter]
        return sorted(cells, key=lambda c: c.fork_point.timestamp, reverse=True)

    def get_lineage(self, trace_id: str) -> list[str]:
        """Get lineage path from genesis to current."""
        lineage = []
        current_id = trace_id

        while current_id:
            lineage.append(current_id)
            cell = self.get_cell(current_id)
            if not cell:
                break
            current_id = cell.parent_trace_id

        return list(reversed(lineage))


def main():
    parser = argparse.ArgumentParser(description="InferCell Fork/Merge Manager")
    parser.add_argument("--root", default="/pluribus", help="Pluribus root directory")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # create
    create_p = subparsers.add_parser("create", help="Create genesis cell")
    create_p.add_argument("--reason", default="genesis", help="Creation reason")

    # fork
    fork_p = subparsers.add_parser("fork", help="Fork from parent")
    fork_p.add_argument("--parent", required=True, help="Parent trace_id")
    fork_p.add_argument("--reason", required=True, help="Fork reason")

    # merge
    merge_p = subparsers.add_parser("merge", help="Merge multiple traces")
    merge_p.add_argument("--sources", required=True, help="Comma-separated trace_ids")
    merge_p.add_argument("--strategy", default="consensus", choices=["first_wins", "consensus", "manual"])

    # list
    list_p = subparsers.add_parser("list", help="List all cells")
    list_p.add_argument("--state", help="Filter by state")

    # get
    get_p = subparsers.add_parser("get", help="Get cell by trace_id")
    get_p.add_argument("--trace-id", required=True, help="Trace ID")

    # state
    state_p = subparsers.add_parser("state", help="Change cell state")
    state_p.add_argument("--trace-id", required=True, help="Trace ID")
    state_p.add_argument("--set", required=True, choices=["pending", "active", "paused", "complete", "merged"])

    # lineage
    lineage_p = subparsers.add_parser("lineage", help="Show cell lineage")
    lineage_p.add_argument("--trace-id", required=True, help="Trace ID")

    args = parser.parse_args()
    manager = InferCellManager(Path(args.root))

    if args.command == "create":
        cell = manager.create_genesis(args.reason)
        print(json.dumps(cell.to_dict(), indent=2))

    elif args.command == "fork":
        cell = manager.fork(args.parent, args.reason)
        print(json.dumps(cell.to_dict(), indent=2))

    elif args.command == "merge":
        sources = [s.strip() for s in args.sources.split(",")]
        cell = manager.merge(sources, args.strategy)
        print(json.dumps(cell.to_dict(), indent=2))

    elif args.command == "list":
        cells = manager.list_cells(args.state)
        for cell in cells:
            state_marker = {"active": "*", "pending": "?", "paused": "~", "complete": "+", "merged": "→"}.get(cell.state, " ")
            parent = f" ← {cell.parent_trace_id[:8]}" if cell.parent_trace_id else " (genesis)"
            children = f" → [{len(cell.children)}]" if cell.children else ""
            print(f"[{state_marker}] {cell.trace_id[:8]}  {cell.fork_point.reason}{parent}{children}")

    elif args.command == "get":
        cell = manager.get_cell(args.trace_id)
        if cell:
            print(json.dumps(cell.to_dict(), indent=2))
        else:
            print(f"Cell not found: {args.trace_id}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "state":
        manager.set_state(args.trace_id, args.set)
        print(f"State updated to: {args.set}")

    elif args.command == "lineage":
        lineage = manager.get_lineage(args.trace_id)
        print(" → ".join([t[:8] for t in lineage]))


if __name__ == "__main__":
    main()
