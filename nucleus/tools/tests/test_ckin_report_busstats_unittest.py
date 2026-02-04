import json
import pathlib
import tempfile
import time
import unittest

import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import ckin_report  # noqa: E402


def _evt(topic: str, *, kind: str, actor: str, data: dict) -> dict:
    return {
        "id": "e",
        "ts": time.time(),
        "iso": "2025-01-01T00:00:00Z",
        "topic": topic,
        "kind": kind,
        "level": "info",
        "actor": actor,
        "data": data,
    }


class TestCkinReportBusStats(unittest.TestCase):
    def test_busstats_includes_a2a_and_studio_flow(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            bus_path = bus_dir / "events.ndjson"
            events = [
                _evt("a2a.negotiate.request", kind="request", actor="init", data={"req_id": "r1"}),
                _evt("a2a.negotiate.response", kind="response", actor="agent", data={"req_id": "r1", "decision": "agree"}),
                _evt("a2a.decline", kind="response", actor="agent", data={"req_id": "r2", "reason": "nope"}),
                _evt("a2a.redirect", kind="response", actor="agent", data={"req_id": "r2", "redirect_to": "agent-rust"}),
                _evt("studio.flow.roundtrip", kind="metric", actor="studio", data={"req_id": "r3", "ok": True}),
                _evt("studio.flow.roundtrip", kind="metric", actor="studio", data={"req_id": "r4", "ok": False}),
            ]
            bus_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

            stats = ckin_report.analyze_bus(bus_path, window_s=900)
            self.assertEqual(stats.a2a_negotiate_requests, 1)
            self.assertEqual(stats.a2a_negotiate_responses, 1)
            self.assertEqual(stats.a2a_declines, 1)
            self.assertEqual(stats.a2a_redirects, 1)
            self.assertEqual(stats.studio_flow_roundtrips, 2)
            self.assertEqual(stats.studio_flow_roundtrip_failures, 1)

    def test_busstats_tracks_pbdeep_events(self):
        with tempfile.TemporaryDirectory(prefix="pluribus_bus_") as td:
            bus_dir = pathlib.Path(td)
            bus_path = bus_dir / "events.ndjson"
            events = [
                _evt("operator.pbdeep.request", kind="request", actor="op", data={"req_id": "r1"}),
                _evt(
                    "operator.pbdeep.report",
                    kind="artifact",
                    actor="op",
                    data={"req_id": "r1", "report_path": "agent_reports/pbdeep_r1.json"},
                ),
                _evt(
                    "operator.pbdeep.index.updated",
                    kind="artifact",
                    actor="op",
                    data={
                        "req_id": "r1",
                        "index_path": "agent_reports/pbdeep_index_r1.json",
                        "summary": {"rag_doc_id": "doc1", "kg": {"nodes": 4}},
                    },
                ),
            ]
            bus_path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")

            stats = ckin_report.analyze_bus(bus_path, window_s=900)
            self.assertEqual(stats.pbdeep_requests, 1)
            self.assertEqual(stats.pbdeep_reports, 1)
            self.assertEqual(stats.pbdeep_index_updates, 1)

            summary = ckin_report.extract_pbdeep_summary(stats)
            self.assertEqual(summary["latest_report"]["report_path"], "agent_reports/pbdeep_r1.json")
            self.assertEqual(summary["latest_index"]["rag_items"], 1)
            self.assertEqual(summary["latest_index"]["kg_nodes"], 4)


if __name__ == "__main__":
    unittest.main()
