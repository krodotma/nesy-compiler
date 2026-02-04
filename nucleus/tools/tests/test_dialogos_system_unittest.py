#!/usr/bin/env python3
"""Comprehensive tests for the dialogos ground truth system.

Tests cover:
1. dialogosd.py - daemon for processing dialogos.submit events
2. dialogos_hook.py - Claude Code hook for trace integration
3. Integration tests for the full dialogos pipeline
4. Edge cases for robustness

Reference: DKIN v25 ground truth recovery protocol
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

# Add tools directory to path
TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(TOOLS_DIR / "hooks"))

import dialogosd  # noqa: E402


class TestDialogosDHandleSubmit(unittest.TestCase):
    """Test dialogosd.handle_submit with various inputs."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")

        self.trace_dir = pathlib.Path(self.tmp.name) / "dialogos"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / "trace.ndjson"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_handle_submit_with_mock_provider(self) -> None:
        """Test handle_submit processes mock provider correctly."""
        submit_event = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {
                "req_id": "test-req-001",
                "mode": "llm",
                "providers": ["mock"],
                "prompt": "Hello world",
            },
        }

        counters: dict[str, int] = {}
        result = dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test-actor",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )

        self.assertTrue(result)

        # Verify bus events emitted
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        topics = [json.loads(line).get("topic") for line in lines if line.strip()]

        self.assertIn("dialogos.cell.start", topics)
        self.assertIn("dialogos.cell.output", topics)
        self.assertIn("dialogos.cell.end", topics)

    def test_handle_submit_writes_trace_file(self) -> None:
        """Test handle_submit writes to trace file for ground truth recovery."""
        submit_event = {
            "id": "e2",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {
                "req_id": "trace-test-001",
                "mode": "llm",
                "providers": ["mock"],
                "prompt": "Test prompt for trace",
            },
        }

        counters: dict[str, int] = {}
        dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="trace-actor",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )

        # Verify trace file written
        self.assertTrue(self.trace_path.exists())
        trace_content = self.trace_path.read_text(encoding="utf-8")
        self.assertGreater(len(trace_content), 0)

        trace_record = json.loads(trace_content.strip())
        self.assertEqual(trace_record["req_id"], "trace-test-001")
        self.assertEqual(trace_record["mode"], "llm")
        self.assertEqual(trace_record["providers"], ["mock"])
        self.assertTrue(trace_record["ok"])
        self.assertEqual(trace_record["actor"], "trace-actor")
        self.assertIn("prompt_sha256", trace_record)
        self.assertEqual(trace_record["prompt_len"], len("Test prompt for trace"))

    def test_handle_submit_no_trace_when_none(self) -> None:
        """Test handle_submit works when trace_path is None."""
        submit_event = {
            "id": "e3",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {
                "req_id": "no-trace-001",
                "mode": "llm",
                "providers": ["mock"],
                "prompt": "No trace test",
            },
        }

        counters: dict[str, int] = {}
        result = dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=None,  # No trace
            actor="test-actor",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )

        self.assertTrue(result)
        # Trace file should not exist
        self.assertFalse(self.trace_path.exists())

    def test_handle_submit_invalid_data_returns_false(self) -> None:
        """Test handle_submit returns False for invalid data."""
        # Missing data dict
        submit_event = {"id": "e4", "topic": "dialogos.submit"}
        counters: dict[str, int] = {}
        result = dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )
        self.assertFalse(result)

        # Data is not a dict
        submit_event = {"id": "e5", "data": "not a dict"}
        result = dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )
        self.assertFalse(result)

    def test_handle_submit_missing_req_id(self) -> None:
        """Test handle_submit returns False when req_id is missing."""
        submit_event = {
            "id": "e6",
            "data": {"mode": "llm", "prompt": "test"},
        }
        counters: dict[str, int] = {}
        result = dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )
        self.assertFalse(result)

    def test_handle_submit_unsupported_mode(self) -> None:
        """Test handle_submit handles unsupported modes gracefully."""
        submit_event = {
            "id": "e7",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {
                "req_id": "unsupported-mode-001",
                "mode": "strp",  # Not fully supported yet
                "providers": ["mock"],
                "prompt": "Test",
            },
        }

        counters: dict[str, int] = {}
        result = dialogosd.handle_submit(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            submit_event=submit_event,
            emit_infer_sync=False,
            counters=counters,
        )

        self.assertTrue(result)

        # Check error was recorded
        lines = self.events_path.read_text(encoding="utf-8").splitlines()
        end_events = [
            json.loads(line) for line in lines
            if line.strip() and json.loads(line).get("topic") == "dialogos.cell.end"
        ]
        self.assertEqual(len(end_events), 1)
        self.assertFalse(end_events[0]["data"]["ok"])
        self.assertIn("unsupported_mode:strp", end_events[0]["data"]["errors"])


class TestDialogosDProcessEventsOnce(unittest.TestCase):
    """Test dialogosd.process_events_once function."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.trace_path = pathlib.Path(self.tmp.name) / "dialogos" / "trace.ndjson"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_process_events_once_processes_pending(self) -> None:
        """Test process_events_once processes pending events."""
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "r1", "mode": "llm", "providers": ["mock"], "prompt": "hello"},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 1)

    def test_process_events_once_skips_completed(self) -> None:
        """Test process_events_once skips already completed requests."""
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "r1", "mode": "llm", "providers": ["mock"], "prompt": "hello"},
        }
        end = {
            "id": "e2",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.cell.end",
            "kind": "response",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "r1", "ok": True, "errors": []},
        }

        content = json.dumps(submit) + "\n" + json.dumps(end) + "\n"
        self.events_path.write_text(content, encoding="utf-8")

        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 0)

    def test_process_events_once_multi_provider(self) -> None:
        """Test process_events_once emits multiple outputs for multi-provider."""
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "r2", "mode": "llm", "providers": ["mock", "mock"], "prompt": "hi"},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        lines = [
            json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        outputs = [
            e for e in lines
            if e.get("topic") == "dialogos.cell.output" and e.get("data", {}).get("req_id") == "r2"
        ]
        self.assertGreaterEqual(len(outputs), 2)


class TestDialogosDTraceFileFormat(unittest.TestCase):
    """Test trace file format and rotation."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")
        self.trace_path = pathlib.Path(self.tmp.name) / "dialogos" / "trace.ndjson"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_trace_file_ndjson_format(self) -> None:
        """Test trace file uses valid NDJSON format."""
        for i in range(3):
            submit = {
                "id": f"e{i}",
                "ts": time.time(),
                "iso": dialogosd.now_iso_utc(),
                "topic": "dialogos.submit",
                "kind": "request",
                "level": "info",
                "actor": "test",
                "data": {"req_id": f"r{i}", "mode": "llm", "providers": ["mock"], "prompt": f"test{i}"},
            }
            self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")
            dialogosd.process_events_once(
                bus_dir=self.bus_dir,
                trace_path=self.trace_path,
                actor="test",
                emit_infer_sync=False,
            )
            # Reset events file for next iteration
            self.events_path.write_text("", encoding="utf-8")

        # Verify trace file has 3 valid JSON lines
        lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)

        for line in lines:
            record = json.loads(line)
            self.assertIn("id", record)
            self.assertIn("ts", record)
            self.assertIn("iso", record)
            self.assertIn("req_id", record)

    def test_trace_file_includes_prompt_hash(self) -> None:
        """Test trace file includes SHA256 hash of prompt."""
        prompt = "This is a test prompt for hashing"
        expected_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "r1", "mode": "llm", "providers": ["mock"], "prompt": prompt},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")
        dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        record = json.loads(self.trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(record["prompt_sha256"], expected_hash)


class TestDialogosHook(unittest.TestCase):
    """Test dialogos_hook.py hook handlers."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")

        self.trace_dir = pathlib.Path(self.tmp.name) / "dialogos"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / "trace.ndjson"
        self.trace_path.write_text("", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_user_prompt_submit_handling(self) -> None:
        """Test UserPromptSubmit hook event handling."""
        # Import hook module
        import dialogos_hook

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
            "PLURIBUS_ACTOR": "test-hook-actor",
        }):
            dialogos_hook.handle_user_prompt_submit({
                "prompt": "Test prompt from hook",
                "session_id": "sess-12345678",
                "cwd": "/tmp/test",
            })

        # Check bus event emitted
        lines = self.events_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(lines), 0)

        event = json.loads(lines[-1])
        self.assertEqual(event["topic"], "dialogos.claude_code.prompt")
        self.assertEqual(event["data"]["session_id"], "sess-12345678")
        self.assertIn("prompt_len", event["data"])

        # Check trace event
        trace_lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(trace_lines), 0)
        trace = json.loads(trace_lines[-1])
        self.assertEqual(trace["event_type"], "user_prompt")
        self.assertEqual(trace["session_id"], "sess-12345678")

    def test_stop_handling(self) -> None:
        """Test Stop hook event handling."""
        import dialogos_hook

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
            "PLURIBUS_ACTOR": "test-hook-actor",
        }):
            dialogos_hook.handle_stop({
                "session_id": "sess-87654321",
                "transcript_path": "/tmp/transcript.json",
            })

        # Check bus event emitted
        lines = self.events_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(lines), 0)

        event = json.loads(lines[-1])
        self.assertEqual(event["topic"], "dialogos.claude_code.stop")
        self.assertEqual(event["data"]["session_id"], "sess-87654321")

        # Check trace event
        trace_lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(trace_lines), 0)
        trace = json.loads(trace_lines[-1])
        self.assertEqual(trace["event_type"], "assistant_stop")

    def test_trace_file_output_format(self) -> None:
        """Test trace file output format from hook."""
        import dialogos_hook

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
            "PLURIBUS_ACTOR": "format-test",
        }):
            dialogos_hook.handle_user_prompt_submit({
                "prompt": "Format test",
                "session_id": "format-sess",
                "cwd": "/test",
            })

        trace = json.loads(self.trace_path.read_text(encoding="utf-8").strip())

        # Required fields
        self.assertIn("id", trace)
        self.assertIn("ts", trace)
        self.assertIn("iso", trace)
        self.assertIn("req_id", trace)
        self.assertIn("session_id", trace)
        self.assertIn("event_type", trace)
        self.assertIn("actor", trace)
        self.assertIn("source", trace)

        # Format validation
        self.assertIsInstance(trace["ts"], float)
        self.assertTrue(trace["iso"].endswith("Z"))
        self.assertEqual(trace["source"], "claude-code-hook")


class TestDialogosIntegration(unittest.TestCase):
    """Integration tests for the full dialogos pipeline."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")

        self.trace_dir = pathlib.Path(self.tmp.name) / "dialogos"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.trace_dir / "trace.ndjson"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_emit_submit_to_daemon_to_trace(self) -> None:
        """Test full pipeline: emit dialogos.submit -> dialogosd processes -> trace written."""
        # Emit dialogos.submit event
        req_id = f"integration-{int(time.time() * 1000)}"
        submit = {
            "id": "int-e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "integration-test",
            "data": {
                "req_id": req_id,
                "mode": "llm",
                "providers": ["mock"],
                "prompt": "Integration test prompt",
            },
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        # Process with dialogosd
        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="dialogosd-integration",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 1)

        # Verify bus events
        lines = self.events_path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]

        topics = [e["topic"] for e in events]
        self.assertIn("dialogos.submit", topics)
        self.assertIn("dialogos.cell.start", topics)
        self.assertIn("dialogos.cell.output", topics)
        self.assertIn("dialogos.cell.end", topics)

        # Verify trace written
        self.assertTrue(self.trace_path.exists())
        trace = json.loads(self.trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(trace["req_id"], req_id)
        self.assertTrue(trace["ok"])

    def test_bus_events_emitted_correctly(self) -> None:
        """Test bus events have correct structure."""
        submit = {
            "id": "bus-e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "bus-test",
            "data": {"req_id": "bus-req-1", "mode": "llm", "providers": ["mock"], "prompt": "test"},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="bus-actor",
            emit_infer_sync=False,
        )

        events = [
            json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        for event in events:
            # All events should have required fields
            self.assertIn("id", event)
            self.assertIn("ts", event)
            self.assertIn("iso", event)
            self.assertIn("topic", event)
            self.assertIn("kind", event)
            self.assertIn("level", event)
            self.assertIn("actor", event)
            self.assertIn("data", event)


class TestDialogosEdgeCases(unittest.TestCase):
    """Edge case tests for robustness."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")
        self.trace_path = pathlib.Path(self.tmp.name) / "dialogos" / "trace.ndjson"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_empty_prompt(self) -> None:
        """Test handling of empty prompt."""
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "empty-prompt-1", "mode": "llm", "providers": ["mock"], "prompt": ""},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 1)

        # Should still write trace
        trace = json.loads(self.trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(trace["prompt_len"], 0)

    def test_missing_session_id_in_hook(self) -> None:
        """Test hook handles missing session_id gracefully."""
        import dialogos_hook

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
        }):
            # Should not raise
            dialogos_hook.handle_user_prompt_submit({
                "prompt": "Test without session_id",
                # No session_id
            })

        trace_lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(trace_lines), 0)
        trace = json.loads(trace_lines[-1])
        self.assertEqual(trace["session_id"], "unknown")

    def test_unicode_in_prompt(self) -> None:
        """Test handling of Unicode in prompts."""
        unicode_prompt = "Hello world with unicode: cafe emoji test chinese"
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "unicode-1", "mode": "llm", "providers": ["mock"], "prompt": unicode_prompt},
        }
        self.events_path.write_text(json.dumps(submit, ensure_ascii=False) + "\n", encoding="utf-8")

        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 1)

    def test_concurrent_writes_simulation(self) -> None:
        """Test trace file handles concurrent writes (simulated)."""
        results = []
        errors = []

        def write_trace(idx: int) -> None:
            try:
                submit = {
                    "id": f"concurrent-{idx}",
                    "ts": time.time(),
                    "iso": dialogosd.now_iso_utc(),
                    "topic": "dialogos.submit",
                    "kind": "request",
                    "level": "info",
                    "actor": "concurrent-test",
                    "data": {"req_id": f"concurrent-req-{idx}", "mode": "llm", "providers": ["mock"], "prompt": f"test {idx}"},
                }
                # Write to separate events files to avoid contention
                events_file = self.bus_dir / f"events_{idx}.ndjson"
                events_file.write_text(json.dumps(submit) + "\n", encoding="utf-8")

                dialogosd.handle_submit(
                    bus_dir=self.bus_dir,
                    trace_path=self.trace_path,
                    actor=f"actor-{idx}",
                    submit_event=submit,
                    emit_infer_sync=False,
                    counters={},
                )
                results.append(idx)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=write_trace, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 5)

        # All traces should be written
        trace_lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(trace_lines), 5)

    def test_malformed_events_in_bus(self) -> None:
        """Test process_events_once handles malformed JSON gracefully."""
        content = "not json\n{\"valid\": true}\n{broken\n"
        self.events_path.write_text(content, encoding="utf-8")

        # Should not raise
        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 0)

    def test_very_long_prompt(self) -> None:
        """Test handling of very long prompts."""
        long_prompt = "x" * 100_000  # 100KB prompt
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "long-prompt-1", "mode": "llm", "providers": ["mock"], "prompt": long_prompt},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        processed = dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=False,
        )

        self.assertEqual(processed, 1)

        trace = json.loads(self.trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(trace["prompt_len"], 100_000)

    def test_default_provider_is_auto(self) -> None:
        """Test that missing providers defaults to auto."""
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "no-provider-1", "mode": "llm", "prompt": "test"},
            # No providers field
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        # We mock run_provider to avoid actually calling the router
        with mock.patch.object(dialogosd, "run_provider", return_value=(0, "mocked", "")) as mock_run:
            dialogosd.process_events_once(
                bus_dir=self.bus_dir,
                trace_path=self.trace_path,
                actor="test",
                emit_infer_sync=False,
            )

            # Should have been called with auto provider
            mock_run.assert_called_once_with(provider="auto", prompt="test")


class TestDialogosHookMain(unittest.TestCase):
    """Test dialogos_hook.main() function with stdin input."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")

        self.trace_path = pathlib.Path(self.tmp.name) / "dialogos" / "trace.ndjson"
        pathlib.Path(self.tmp.name, "dialogos").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_main_user_prompt_submit(self) -> None:
        """Test main() handles UserPromptSubmit from stdin."""
        import dialogos_hook

        input_data = {
            "hook_event_name": "UserPromptSubmit",
            "prompt": "Main test prompt",
            "session_id": "main-sess-123",
            "cwd": "/test/cwd",
        }

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
        }):
            with mock.patch("sys.stdin", io.StringIO(json.dumps(input_data))):
                result = dialogos_hook.main()

        self.assertEqual(result, 0)

        # Verify trace was written
        trace_lines = self.trace_path.read_text(encoding="utf-8").strip().splitlines()
        self.assertGreater(len(trace_lines), 0)

    def test_main_stop_event(self) -> None:
        """Test main() handles Stop event from stdin."""
        import dialogos_hook

        input_data = {
            "hook_event_name": "Stop",
            "session_id": "stop-sess-456",
            "transcript_path": "/tmp/trans.json",
        }

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
        }):
            with mock.patch("sys.stdin", io.StringIO(json.dumps(input_data))):
                result = dialogos_hook.main()

        self.assertEqual(result, 0)

    def test_main_session_start(self) -> None:
        """Test main() handles SessionStart event."""
        import dialogos_hook

        input_data = {
            "hook_event_name": "SessionStart",
            "session_id": "start-sess-789",
            "cwd": "/start/cwd",
        }

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
        }):
            with mock.patch("sys.stdin", io.StringIO(json.dumps(input_data))):
                result = dialogos_hook.main()

        self.assertEqual(result, 0)

        trace = json.loads(self.trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(trace["event_type"], "session_start")

    def test_main_session_end(self) -> None:
        """Test main() handles SessionEnd event."""
        import dialogos_hook

        input_data = {
            "hook_event_name": "SessionEnd",
            "session_id": "end-sess-000",
        }

        with mock.patch.dict(os.environ, {
            "PLURIBUS_BUS_DIR": str(self.bus_dir),
            "PLURIBUS_DIALOGOS_TRACE": str(self.trace_path),
        }):
            with mock.patch("sys.stdin", io.StringIO(json.dumps(input_data))):
                result = dialogos_hook.main()

        self.assertEqual(result, 0)

        trace = json.loads(self.trace_path.read_text(encoding="utf-8").strip())
        self.assertEqual(trace["event_type"], "session_end")

    def test_main_invalid_json(self) -> None:
        """Test main() handles invalid JSON gracefully."""
        import dialogos_hook

        with mock.patch("sys.stdin", io.StringIO("not json")):
            result = dialogos_hook.main()

        # Should return 0 (silent fail)
        self.assertEqual(result, 0)


class TestDialogosHelperFunctions(unittest.TestCase):
    """Test helper functions in dialogosd."""

    def test_now_iso_utc_format(self) -> None:
        """Test now_iso_utc returns proper ISO format."""
        result = dialogosd.now_iso_utc()
        self.assertTrue(result.endswith("Z"))
        self.assertEqual(len(result), 20)  # YYYY-MM-DDTHH:MM:SSZ

    def test_default_bus_dir(self) -> None:
        """Test default_bus_dir returns correct path."""
        with mock.patch.dict(os.environ, {"PLURIBUS_BUS_DIR": "/custom/bus"}, clear=False):
            result = dialogosd.default_bus_dir()
            self.assertEqual(result, pathlib.Path("/custom/bus"))

    def test_default_trace_path(self) -> None:
        """Test default_trace_path returns correct path."""
        with mock.patch.dict(os.environ, {"PLURIBUS_DIALOGOS_TRACE": "/custom/trace.ndjson"}, clear=False):
            result = dialogosd.default_trace_path()
            self.assertEqual(result, pathlib.Path("/custom/trace.ndjson"))

    def test_default_actor(self) -> None:
        """Test default_actor uses environment variables."""
        with mock.patch.dict(os.environ, {"PLURIBUS_ACTOR": "custom-actor"}, clear=False):
            result = dialogosd.default_actor()
            self.assertEqual(result, "custom-actor")

    def test_iter_ndjson_nonexistent_file(self) -> None:
        """Test iter_ndjson handles nonexistent file."""
        result = list(dialogosd.iter_ndjson(pathlib.Path("/nonexistent/file.ndjson")))
        self.assertEqual(result, [])

    def test_iter_ndjson_malformed_lines(self) -> None:
        """Test iter_ndjson skips malformed lines."""
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "test.ndjson"
            path.write_text('{"a":1}\nnot json\n{"b":2}\n', encoding="utf-8")

            result = list(dialogosd.iter_ndjson(path))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0]["a"], 1)
            self.assertEqual(result[1]["b"], 2)


class TestDialogosDInferSyncCheckin(unittest.TestCase):
    """Test infer_sync.checkin emissions."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.bus_dir = pathlib.Path(self.tmp.name) / "bus"
        self.bus_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.bus_dir / "events.ndjson"
        self.events_path.write_text("", encoding="utf-8")
        self.trace_path = pathlib.Path(self.tmp.name) / "dialogos" / "trace.ndjson"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_infer_sync_checkin_emitted(self) -> None:
        """Test infer_sync.checkin is emitted when enabled."""
        submit = {
            "id": "e1",
            "ts": time.time(),
            "iso": dialogosd.now_iso_utc(),
            "topic": "dialogos.submit",
            "kind": "request",
            "level": "info",
            "actor": "test",
            "data": {"req_id": "r1", "mode": "llm", "providers": ["mock"], "prompt": "test"},
        }
        self.events_path.write_text(json.dumps(submit) + "\n", encoding="utf-8")

        dialogosd.process_events_once(
            bus_dir=self.bus_dir,
            trace_path=self.trace_path,
            actor="test",
            emit_infer_sync=True,  # Enabled
        )

        events = [
            json.loads(line) for line in self.events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        checkin_events = [e for e in events if e.get("topic") == "infer_sync.checkin"]
        self.assertGreater(len(checkin_events), 0)

        checkin = checkin_events[0]
        self.assertEqual(checkin["data"]["subproject"], "dialogos")
        self.assertIn("done", checkin["data"])
        self.assertIn("errors", checkin["data"])


if __name__ == "__main__":
    unittest.main()
