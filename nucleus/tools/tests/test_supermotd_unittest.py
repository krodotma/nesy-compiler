import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import supermotd  # noqa: E402


class TestSuperMotd(unittest.TestCase):
    def test_event_to_line_filters_dialogos_output_noise(self):
        e = {"topic": "dialogos.cell.output", "level": "info", "actor": "x", "iso": "2025-01-01T00:00:00Z", "data": {"type": "text", "content": "hi"}}
        self.assertIsNone(supermotd.event_to_line(e))

        e2 = {"topic": "dialogos.cell.output", "level": "error", "actor": "x", "iso": "2025-01-01T00:00:00Z", "data": {"type": "error", "content": "bad"}}
        l2 = supermotd.event_to_line(e2)
        self.assertIsNotNone(l2)
        self.assertEqual(l2.subsystem, "DIALOGOS")

    def test_event_to_line_formats_lens_decision(self):
        e = {
            "topic": "plurichat.lens.decision",
            "level": "info",
            "actor": "plurichat",
            "iso": "2025-01-01T00:00:00Z",
            "data": {"depth": "deep", "lane": "strp", "topology": "star", "fanout": 3, "selected_provider": "codex-cli", "persona": "ring0.architect"},
        }
        l = supermotd.event_to_line(e)
        self.assertIsNotNone(l)
        self.assertIn("depth=deep", l.message)
        self.assertIn("→ codex-cli", l.message)

    def test_event_to_line_formats_pbflush(self):
        e = {
            "topic": "operator.pbflush.request",
            "level": "warn",
            "actor": "tester",
            "iso": "2025-01-01T00:00:00Z",
            "data": {"req_id": "abcd1234-ffff", "intent": "pbflush", "message": "wrap + await next ckin"},
        }
        l = supermotd.event_to_line(e)
        self.assertIsNotNone(l)
        self.assertEqual(l.subsystem, "PBFLUSH")
        self.assertIn("req=abcd1234", l.message)

    def test_event_to_line_formats_mcp(self):
        e = {
            "topic": "mcp.host.call",
            "level": "info",
            "actor": "mcp-host",
            "iso": "2025-01-01T00:00:00Z",
            "data": {"req_id": "abcd1234-ffff", "server": "pluribus-kg", "tool": "query"},
        }
        l = supermotd.event_to_line(e)
        self.assertIsNotNone(l)
        self.assertEqual(l.subsystem, "MCP")
        self.assertIn("pluribus-kg.query", l.message)

    def test_event_to_line_formats_a2a_negotiate(self):
        e = {
            "topic": "a2a.negotiate.request",
            "level": "info",
            "actor": "initiator",
            "iso": "2025-01-01T00:00:00Z",
            "data": {"req_id": "abcd1234-ffff", "initiator": "initiator", "target": "agent-x", "constraints": {"required_capabilities": ["python"]}},
        }
        l = supermotd.event_to_line(e)
        self.assertIsNotNone(l)
        self.assertEqual(l.subsystem, "A2A")
        self.assertIn("negotiate.request", l.message)
        self.assertIn("req=abcd1234", l.message)

        e2 = {
            "topic": "a2a.negotiate.response",
            "level": "info",
            "actor": "agent-x",
            "iso": "2025-01-01T00:00:01Z",
            "data": {"req_id": "abcd1234-ffff", "decision": "agree"},
        }
        l2 = supermotd.event_to_line(e2)
        self.assertIsNotNone(l2)
        self.assertEqual(l2.subsystem, "A2A")
        self.assertIn("decision=agree", l2.message)

    def test_event_to_line_formats_studio_flow_roundtrip(self):
        e = {
            "topic": "studio.flow.roundtrip",
            "level": "info",
            "actor": "tester",
            "iso": "2025-01-01T00:00:02Z",
            "data": {"req_id": "abcd1234-ffff", "ok": True, "in_path": "/tmp/example_flow.py"},
        }
        l = supermotd.event_to_line(e)
        self.assertIsNotNone(l)
        self.assertEqual(l.subsystem, "STUDIO")
        self.assertIn("ok", l.message)


if __name__ == "__main__":
    unittest.main()
