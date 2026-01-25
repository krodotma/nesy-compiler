#!/usr/bin/env python3
"""
pbtso_task_daemon.py - PBTSO task ingress daemon

Listens for task.create events and appends to task_ledger.
Emits pbtso.task.created (and legacy tbtso.task.created) acknowledgements.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nucleus.tools import agent_bus, task_ledger


DEFAULT_ACTOR = os.environ.get("PLURIBUS_ACTOR", "pbtso-task-daemon")

STATUS_MAP = {
    "todo": "planned",
    "planned": "planned",
    "in_progress": "in_progress",
    "doing": "in_progress",
    "blocked": "blocked",
    "done": "completed",
    "completed": "completed",
    "abandoned": "abandoned",
}


@dataclass
class TaskCreateResult:
    entry: dict
    ack_topics: list[str]


class TaskCreateProcessor:
    def __init__(
        self,
        *,
        bus_dir: Optional[str] = None,
        ledger_path: Optional[Path] = None,
        actor: str = DEFAULT_ACTOR,
        emit_bus: bool = True,
        emit_legacy: bool = True,
    ) -> None:
        self.bus_paths = agent_bus.resolve_bus_paths(bus_dir)
        self.ledger_path = ledger_path
        self.actor = actor
        self.emit_bus = emit_bus
        self.emit_legacy = emit_legacy

    def _normalize_status(self, raw: Optional[str]) -> str:
        if not raw:
            return "planned"
        return STATUS_MAP.get(str(raw).strip().lower(), "planned")

    def _build_entry(self, event: Dict[str, Any]) -> tuple[dict, str, str]:
        data = event.get("data") or {}
        correlation_id = data.get("correlation_id") or data.get("correlationId") or event.get("id")
        title = data.get("title") or data.get("name") or "Untitled Task"
        lane = data.get("lane") or "inbox"
        status = self._normalize_status(data.get("status"))

        entry = {
            "req_id": str(correlation_id),
            "actor": data.get("actor") or event.get("actor") or "dialogos-ingress",
            "topic": "task.create",
            "status": status,
            "intent": title,
            "meta": {
                "title": title,
                "lane": lane,
                "source": data.get("source") or "dialogos",
                "correlation_id": correlation_id,
                "bus_event_id": event.get("id"),
            },
        }
        return entry, str(correlation_id), lane

    def _emit_ack(self, entry: dict, correlation_id: str, lane: str) -> list[str]:
        topics = ["pbtso.task.created"]
        if self.emit_legacy:
            topics.append("tbtso.task.created")
        ack = {
            "correlationId": correlation_id,
            "taskId": entry.get("id"),
            "req_id": entry.get("req_id"),
            "status": entry.get("status"),
            "lane": lane,
        }
        for topic in topics:
            agent_bus.emit_event(
                self.bus_paths,
                topic=topic,
                kind="event",
                level="info",
                actor=self.actor,
                data=ack,
            )
        return topics

    def handle_event(self, event: Dict[str, Any]) -> Optional[TaskCreateResult]:
        topic = str(event.get("topic", ""))
        if topic not in ("task.create", "pbtso.task.create"):
            return None

        entry_payload, correlation_id, lane = self._build_entry(event)
        entry = task_ledger.append_entry(
            entry_payload,
            ledger_path=self.ledger_path,
            emit_bus=self.emit_bus,
            bus_dir=self.bus_paths.bus_dir,
        )

        ack_topics: list[str] = []
        if self.emit_bus:
            ack_topics = self._emit_ack(entry, correlation_id, lane)

        return TaskCreateResult(entry=entry, ack_topics=ack_topics)


def _iter_existing_lines(path: Path) -> Iterable[str]:
    if not path.exists():
        return []
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line:
                lines.append(line.rstrip("\n"))
    return lines


def run_daemon(
    *,
    bus_dir: Optional[str],
    ledger_path: Optional[Path],
    actor: str,
    emit_bus: bool,
    emit_legacy: bool,
    from_start: bool,
    once: bool,
) -> int:
    processor = TaskCreateProcessor(
        bus_dir=bus_dir,
        ledger_path=ledger_path,
        actor=actor,
        emit_bus=emit_bus,
        emit_legacy=emit_legacy,
    )

    bus_paths = agent_bus.resolve_bus_paths(bus_dir)
    events_path = Path(bus_paths.events_path)

    if from_start:
        for line in _iter_existing_lines(events_path):
            _handle_line(line, processor)
        if once:
            return 0

    for line in agent_bus.iter_lines_follow(str(events_path)):
        _handle_line(line, processor)
        if once:
            break
    return 0


def _handle_line(line: str, processor: TaskCreateProcessor) -> None:
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return
    processor.handle_event(event)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PBTSO task ingress daemon (task.create -> task_ledger + ack)"
    )
    parser.add_argument("--bus-dir", help="Bus directory (default: PLURIBUS_BUS_DIR)")
    parser.add_argument("--ledger-path", help="Override task_ledger.ndjson path")
    parser.add_argument("--actor", default=DEFAULT_ACTOR, help="Actor name for ack events")
    parser.add_argument("--no-bus", action="store_true", help="Disable bus emits (ledger only)")
    parser.add_argument("--no-legacy", action="store_true", help="Disable legacy tbtso.* ack")
    parser.add_argument("--from-start", action="store_true", help="Process existing events before follow")
    parser.add_argument("--once", action="store_true", help="Process once and exit")
    args = parser.parse_args()

    ledger_path = Path(args.ledger_path) if args.ledger_path else None
    emit_bus = not args.no_bus
    emit_legacy = not args.no_legacy

    return run_daemon(
        bus_dir=args.bus_dir,
        ledger_path=ledger_path,
        actor=args.actor,
        emit_bus=emit_bus,
        emit_legacy=emit_legacy,
        from_start=args.from_start,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
