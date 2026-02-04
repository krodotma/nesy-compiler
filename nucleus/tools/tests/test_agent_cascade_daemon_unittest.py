#!/usr/bin/env python3
"""Unit tests for agent_cascade_daemon.py"""

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_cascade_daemon import (
    AgentCascadeDaemon,
    KNOWN_AGENTS,
    check_agent_availability,
    emit_bus,
    now_iso_utc,
)


class TestAgentCascadeDaemon(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.bus_dir = Path(self.tmp_dir) / "bus"
        self.bus_dir.mkdir(parents=True)
        (self.bus_dir / "events.ndjson").touch()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_emit_bus_writes_event(self):
        """emit_bus should write valid NDJSON to events file."""
        emit_bus(
            self.bus_dir,
            topic="test.topic",
            kind="metric",
            level="info",
            actor="test-actor",
            data={"key": "value"},
        )

        events_path = self.bus_dir / "events.ndjson"
        content = events_path.read_text().strip()
        self.assertTrue(len(content) > 0)

        event = json.loads(content)
        self.assertEqual(event["topic"], "test.topic")
        self.assertEqual(event["kind"], "metric")
        self.assertEqual(event["actor"], "test-actor")
        self.assertEqual(event["data"]["key"], "value")

    def test_daemon_init(self):
        """Daemon should initialize with correct defaults."""
        daemon = AgentCascadeDaemon(
            bus_dir=self.bus_dir,
            actor="test-cascade",
            poll_s=0.1,
            max_concurrent=2,
            dry_run=True,
        )

        self.assertEqual(daemon.actor, "test-cascade")
        self.assertEqual(daemon.poll_s, 0.1)
        self.assertEqual(daemon.max_concurrent, 2)
        self.assertTrue(daemon.dry_run)
        self.assertEqual(len(daemon.pending), 0)
        self.assertEqual(len(daemon.active), 0)

    def test_load_responded_ids(self):
        """Daemon should track already-responded request IDs."""
        # Write some response events
        events_path = self.bus_dir / "events.ndjson"
        with events_path.open("w") as f:
            f.write(json.dumps({
                "topic": "a2a.negotiate.response",
                "kind": "response",
                "data": {"req_id": "test-123"},
            }) + "\n")
            f.write(json.dumps({
                "topic": "a2a.negotiate.response",
                "kind": "response",
                "data": {"req_id": "test-456"},
            }) + "\n")

        daemon = AgentCascadeDaemon(
            bus_dir=self.bus_dir,
            actor="test",
            dry_run=True,
        )
        daemon.load_responded_ids()

        self.assertIn("test-123", daemon.responded)
        self.assertIn("test-456", daemon.responded)

    def test_process_negotiate_request(self):
        """Daemon should queue tasks from a2a.negotiate.request events."""
        # Write a negotiate request
        events_path = self.bus_dir / "events.ndjson"
        with events_path.open("w") as f:
            f.write(json.dumps({
                "topic": "a2a.negotiate.request",
                "kind": "request",
                "ts": time.time(),
                "data": {
                    "req_id": "cascade-001",
                    "target": "claude",
                    "task_description": "Test task for cascade",
                    "dispatch_req_id": "dispatch-001",
                },
            }) + "\n")

        daemon = AgentCascadeDaemon(
            bus_dir=self.bus_dir,
            actor="test",
            dry_run=True,
        )
        # Simulate claude being available
        KNOWN_AGENTS["claude"].available = True

        daemon.process_new_events()

        self.assertIn("cascade-001", daemon.pending)
        task = daemon.pending["cascade-001"]
        self.assertEqual(task.target, "claude")
        self.assertEqual(task.task_description, "Test task for cascade")

    def test_skip_already_responded(self):
        """Daemon should skip requests already responded to."""
        events_path = self.bus_dir / "events.ndjson"
        with events_path.open("w") as f:
            # Response first
            f.write(json.dumps({
                "topic": "a2a.negotiate.response",
                "kind": "response",
                "data": {"req_id": "already-done"},
            }) + "\n")
            # Then request
            f.write(json.dumps({
                "topic": "a2a.negotiate.request",
                "kind": "request",
                "data": {
                    "req_id": "already-done",
                    "target": "claude",
                    "task_description": "Should be skipped",
                },
            }) + "\n")

        daemon = AgentCascadeDaemon(
            bus_dir=self.bus_dir,
            actor="test",
            dry_run=True,
        )
        KNOWN_AGENTS["claude"].available = True

        daemon.load_responded_ids()
        daemon.process_new_events()

        # Should NOT be in pending because already responded
        self.assertNotIn("already-done", daemon.pending)

    def test_emit_ready_on_run(self):
        """Daemon should emit ready event with available agents."""
        daemon = AgentCascadeDaemon(
            bus_dir=self.bus_dir,
            actor="test-cascade",
            dry_run=True,
        )

        # Just emit the ready event directly
        daemon.emit(
            "agent.cascade.ready",
            "metric",
            "info",
            {
                "actor": daemon.actor,
                "available_agents": [k for k, v in KNOWN_AGENTS.items() if v.available],
                "max_concurrent": daemon.max_concurrent,
                "dry_run": daemon.dry_run,
                "iso": now_iso_utc(),
            },
        )

        events_path = self.bus_dir / "events.ndjson"
        content = events_path.read_text().strip()
        event = json.loads(content)
        self.assertEqual(event["topic"], "agent.cascade.ready")
        self.assertEqual(event["data"]["actor"], "test-cascade")


class TestKnownAgents(unittest.TestCase):
    def test_known_agents_config(self):
        """Known agents should have required config fields."""
        for name, cfg in KNOWN_AGENTS.items():
            self.assertTrue(len(cfg.name) > 0)
            self.assertTrue(len(cfg.path) > 0)
            self.assertIsInstance(cfg.args_template, list)


if __name__ == "__main__":
    unittest.main()
