#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
PBNOTIFY = TOOLS_DIR / "pbnotify_operator.py"


def run_notify(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(PBNOTIFY), *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def test_pbnotify_emits_bus_and_hexis():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bus_dir = tmp_path / "bus"
        buf_dir = tmp_path / "buffers"
        bus_dir.mkdir(parents=True, exist_ok=True)
        buf_dir.mkdir(parents=True, exist_ok=True)

        env = {
            **os.environ,
            "PLURIBUS_BUS_DIR": str(bus_dir),
            "HEXIS_BUFFER_DIR": str(buf_dir),
            "PLURIBUS_ACTOR": "tester",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = run_notify(
            env,
            "--message",
            "Ping agent",
            "--target",
            "gemini",
            "--json",
        )
        assert result.returncode == 0, result.stderr

        events_path = bus_dir / "events.ndjson"
        events = _read_events(events_path)
        notify_events = [e for e in events if e.get("topic") == "agent.notify.request"]
        assert notify_events, "agent.notify.request not emitted"
        payload = notify_events[-1]["data"]
        assert payload["message"] == "Ping agent"
        assert payload["targets"] == ["gemini"]
        assert payload["hexis_delivery"] == "sent"
        assert payload["class_map"] is None

        buffer_path = buf_dir / "gemini.buffer"
        assert buffer_path.exists()
        lines = [line for line in buffer_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines
        msg = json.loads(lines[-1])
        assert msg["payload"]["message"] == "Ping agent"


def test_pbnotify_broadcast_sets_auto_delivery():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bus_dir = tmp_path / "bus"
        buf_dir = tmp_path / "buffers"
        bus_dir.mkdir(parents=True, exist_ok=True)
        buf_dir.mkdir(parents=True, exist_ok=True)

        env = {
            **os.environ,
            "PLURIBUS_BUS_DIR": str(bus_dir),
            "HEXIS_BUFFER_DIR": str(buf_dir),
            "PLURIBUS_ACTOR": "tester",
            "PYTHONDONTWRITEBYTECODE": "1",
        }

        result = run_notify(
            env,
            "--message",
            "Broadcast ping",
            "--broadcast",
            "--json",
        )
        assert result.returncode == 0, result.stderr

        events_path = bus_dir / "events.ndjson"
        events = _read_events(events_path)
        notify_events = [e for e in events if e.get("topic") == "agent.notify.request"]
        assert notify_events, "agent.notify.request not emitted"
        payload = notify_events[-1]["data"]
        assert payload["broadcast"] is True
        assert payload["hexis_delivery"] == "auto"
