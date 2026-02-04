#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from contextlib import redirect_stdout
import io
import json
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import agent_bus  # noqa: E402


@contextmanager
def as_euid(euid: int):
    """Temporarily drop privileges (works when tests run as root)."""
    orig = os.geteuid()
    if orig != 0:
        # Not root; cannot switch euid reliably.
        yield
        return
    try:
        os.seteuid(euid)
        yield
    finally:
        os.seteuid(orig)


class TestAgentBusResolveAndMirror(unittest.TestCase):
    def test_resolve_prefers_primary_when_events_file_appendable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            os.chmod(tmp_path, 0o777)
            primary = tmp_path / "primary"
            primary.mkdir(parents=True, exist_ok=True)

            # Directory is not writable, but the events file is world-writable.
            events = primary / "events.ndjson"
            events.write_text("", encoding="utf-8")
            os.chmod(primary, 0o555)
            os.chmod(events, 0o666)

            with as_euid(65534):  # nobody
                paths = agent_bus.resolve_bus_paths(str(primary))

            self.assertEqual(paths.active_dir, str(primary))
            self.assertEqual(paths.primary_dir, str(primary))
            self.assertIsNone(paths.fallback_dir)

    def test_cmd_resolve_reports_active_dir_and_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            os.chmod(tmp_path, 0o777)
            primary = tmp_path / "primary"
            fallback = tmp_path / "fallback"
            primary.mkdir(parents=True, exist_ok=True)
            fallback.mkdir(parents=True, exist_ok=True)
            os.chmod(fallback, 0o777)

            # Make primary inaccessible to non-root so resolver falls back.
            os.chmod(primary, 0o000)

            prev_fallback = os.environ.get("PLURIBUS_FALLBACK_BUS_DIR")
            try:
                os.environ["PLURIBUS_FALLBACK_BUS_DIR"] = str(fallback)

                with as_euid(65534):  # nobody
                    out = io.StringIO()
                    with redirect_stdout(out):
                        rc = agent_bus.main(["--bus-dir", str(primary), "resolve"])
                    self.assertEqual(rc, 0)
                    self.assertEqual(out.getvalue().strip(), str(fallback))

                with as_euid(65534):  # nobody
                    out = io.StringIO()
                    with redirect_stdout(out):
                        rc = agent_bus.main(["--bus-dir", str(primary), "resolve", "--json"])
                    self.assertEqual(rc, 0)
                    payload = json.loads(out.getvalue())
                    self.assertEqual(payload["active_dir"], str(fallback))
                    self.assertEqual(payload["primary_dir"], str(primary))
                    self.assertEqual(payload["fallback_dir"], str(fallback))
                    self.assertTrue(payload["events_path"].endswith("events.ndjson"))
            finally:
                if prev_fallback is None:
                    os.environ.pop("PLURIBUS_FALLBACK_BUS_DIR", None)
                else:
                    os.environ["PLURIBUS_FALLBACK_BUS_DIR"] = prev_fallback

    def test_fallback_is_stable_and_mirrors_when_primary_recovers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            os.chmod(tmp_path, 0o777)
            primary = tmp_path / "primary"
            fallback = tmp_path / "fallback"
            primary.mkdir(parents=True, exist_ok=True)
            fallback.mkdir(parents=True, exist_ok=True)
            os.chmod(fallback, 0o777)

            # Make primary inaccessible to non-root so resolver falls back.
            os.chmod(primary, 0o000)

            prev_fallback = os.environ.get("PLURIBUS_FALLBACK_BUS_DIR")
            try:
                os.environ["PLURIBUS_FALLBACK_BUS_DIR"] = str(fallback)
                with as_euid(65534):  # nobody
                    paths = agent_bus.resolve_bus_paths(str(primary))
            finally:
                if prev_fallback is None:
                    os.environ.pop("PLURIBUS_FALLBACK_BUS_DIR", None)
                else:
                    os.environ["PLURIBUS_FALLBACK_BUS_DIR"] = prev_fallback

            self.assertEqual(paths.active_dir, str(fallback))
            self.assertEqual(paths.primary_dir, str(primary))
            self.assertEqual(paths.fallback_dir, str(fallback))

            # Ensure fallback is writable for the non-root writer.
            os.chmod(fallback, 0o777)

            # Primary "recovers" after resolution (e.g., sandbox lifted or perms fixed).
            os.chmod(primary, 0o777)
            (primary / "events.ndjson").write_text("", encoding="utf-8")
            os.chmod(primary / "events.ndjson", 0o666)

            # Emit should write to fallback and also mirror to primary.
            with as_euid(65534):  # nobody
                agent_bus.emit_event(
                    paths,
                    topic="tests.bus.mirror",
                    kind="log",
                    level="info",
                    actor="unit",
                    data={"ok": True},
                    trace_id=None,
                    run_id=None,
                    durable=False,
                )

            fallback_lines = [l for l in (fallback / "events.ndjson").read_text(encoding="utf-8").splitlines() if l.strip()]
            primary_lines = [l for l in (primary / "events.ndjson").read_text(encoding="utf-8").splitlines() if l.strip()]
            self.assertTrue(any('"topic":"tests.bus.mirror"' in l for l in fallback_lines))
            self.assertTrue(any('"topic":"tests.bus.mirror"' in l for l in primary_lines))


if __name__ == "__main__":
    unittest.main()
